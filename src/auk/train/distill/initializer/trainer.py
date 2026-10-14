from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from ema_pytorch import EMA
from torch import nn
from torch.optim import AdamW
from transformers import get_scheduler

from auk.train.distill.common import prepare_batch, reduce_metrics
from auk.train.distill.initializer.consistency import ConsistencyInitSystem, InitConfig


logger = get_logger(__name__)


class InitializerTrainer:
    def __init__(
        self,
        *,
        system: ConsistencyInitSystem,
        vae_model: nn.Module,
        config: InitConfig,
        resolved_config: Mapping[str, Any],
        accelerator: Accelerator,
    ):
        self.accelerator = accelerator
        self.vae_model = vae_model.to(self.accelerator.device)
        self.resolved_config = dict(resolved_config)
        self.max_grad_norm = config.max_grad_norm
        self.updates = 0
        self.epoch = 0

        optimizer = AdamW(system.student.transformer.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
        self.scheduler = get_scheduler(
            config.scheduler_type,
            optimizer,
            num_warmup_steps=config.warmup_updates,
            num_training_steps=config.max_updates,
        )
        self.system, self.optimizer = self.accelerator.prepare(system, optimizer)
        # EMA of the whole student CFMEdit, sharing its frozen Qwen instead of copying it.
        self.student_ema = EMA(
            system.student,
            ema_model=copy.deepcopy(system.student, memo={id(system.student.text_encoder): system.student.text_encoder}),
            beta=config.ema_beta,
            update_after_step=config.ema_update_after_step,
            update_every=config.ema_update_every,
            include_online_model=False,
            ignore_startswith_names={"text_encoder."},
        ).to(self.accelerator.device)

    def step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        system = self.accelerator.unwrap_model(self.system)
        student = system.student.transformer
        prepared = prepare_batch(
            batch,
            vae_model=self.vae_model,
            text_model=system.teacher,
            dtype=next(student.parameters()).dtype,
        )
        step_start = time.perf_counter()
        # The CM time-pair schedule advances by global samples, so all ranks must agree on it.
        global_batch_size = int(
            self.accelerator.reduce(torch.tensor(prepared["clean"].shape[0], device=self.accelerator.device)).item()
        )
        self.optimizer.zero_grad()
        metrics = self.system(prepared, self.updates, global_batch_size=global_batch_size)
        self.accelerator.backward(metrics["total_loss"])
        grad_norm = self.accelerator.clip_grad_norm_(student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.student_ema.update()
        self.updates += 1
        metrics.update(
            {
                "grad_norm": float(grad_norm.detach()),
                "lr": float(self.scheduler.get_last_lr()[0]),
                "local_samples": float(prepared["clean"].shape[0]),
                "target_frames_mean": float(prepared["lens"].float().mean().detach()),
                "gpu_memory_peak_gib": float(torch.cuda.max_memory_allocated(self.accelerator.device) / 2**30),
                "step_time_sec": float(time.perf_counter() - step_start),
            }
        )
        return reduce_metrics(self.accelerator, metrics)

    def save_checkpoint(self, path: Path, *, with_training_state: bool) -> None:
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        checkpoint = {
            "ema_model_state_dict": {k: v for k, v in self.student_ema.state_dict().items() if "text_encoder." not in k},
            "update": self.updates,
            "metadata": {"resolved_config": self.resolved_config},
        }
        if with_training_state:
            student = self.accelerator.unwrap_model(self.system).student
            checkpoint.update(
                {
                    "student_model_state_dict": {k: v for k, v in student.state_dict().items() if "text_encoder." not in k},
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "epoch": self.epoch,
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.accelerator.save(checkpoint, path)
        logger.info(f"saved {path} | update={self.updates}")

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        student = self.accelerator.unwrap_model(self.system).student
        for module, key in ((student, "student_model_state_dict"), (self.student_ema, "ema_model_state_dict")):
            missing, unexpected = module.load_state_dict(checkpoint[key], strict=False)
            missing = [name for name in missing if "text_encoder." not in name]
            if missing or unexpected:
                raise RuntimeError(f"{key}: {len(missing)} missing {missing[:5]}, {len(unexpected)} unexpected {unexpected[:5]}")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.updates = checkpoint["update"]
        self.epoch = checkpoint["epoch"]
