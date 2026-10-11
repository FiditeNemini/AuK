"""Alternating fake-score / student optimizer loop for DMD."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn
from torch.optim import AdamW
from transformers import get_scheduler

from auk.train.distill.common import ema_from_inference_state, ema_to_inference_state, prepare_batch, reduce_metrics
from auk.train.distill.dmd.system import DMDConfig, DMDSystem


class DMDTrainer:
    def __init__(
        self,
        *,
        system: DMDSystem,
        vae_model: nn.Module,
        text_encoder: nn.Module,
        fusion: Mapping[str, torch.Tensor],
        config: DMDConfig,
        resolved_config: Mapping[str, Any],
        accelerator: Accelerator,
    ):
        self.accelerator = accelerator
        self.vae_model = vae_model.to(self.accelerator.device)
        self.text_encoder = text_encoder.to(self.accelerator.device)
        self.fusion = {name: value.float().to(self.accelerator.device) for name, value in fusion.items()}
        self.resolved_config = dict(resolved_config)
        self.student_update_interval = config.fake_score_updates_per_student
        self.max_fake_score_updates = config.max_student_updates * config.fake_score_updates_per_student
        self.max_grad_norm = config.max_grad_norm
        self.fake_score_updates = 0
        self.student_updates = 0
        self.fake_updates_since_student = 0
        self.epoch = 0

        student_optimizer = AdamW(
            system.student.parameters(), lr=config.student_learning_rate, betas=(0.9, 0.95), weight_decay=0.0
        )
        fake_score_optimizer = AdamW(
            system.fake_score.parameters(), lr=config.fake_score_learning_rate, betas=(0.9, 0.95), weight_decay=0.0
        )
        self.student_scheduler = get_scheduler(
            config.scheduler_type,
            student_optimizer,
            num_warmup_steps=config.warmup_updates,
            num_training_steps=config.max_student_updates,
        )
        self.fake_score_scheduler = get_scheduler(
            config.scheduler_type,
            fake_score_optimizer,
            num_warmup_steps=config.warmup_updates,
            num_training_steps=self.max_fake_score_updates,
        )
        self.system, self.student_optimizer, self.fake_score_optimizer = self.accelerator.prepare(
            system, student_optimizer, fake_score_optimizer
        )
        self.student_ema = EMA(
            system.student,
            beta=config.ema_beta,
            update_after_step=config.ema_update_after_step,
            update_every=config.ema_update_every,
            include_online_model=False,
        ).to(self.accelerator.device)

    def step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        unwrapped = self.accelerator.unwrap_model(self.system)
        prepared = prepare_batch(
            batch,
            vae_model=self.vae_model,
            text_encoder=self.text_encoder,
            fusion=self.fusion,
            dtype=next(unwrapped.student.parameters()).dtype,
        )
        step_start = time.perf_counter()
        self.system.train()
        fake_loss, metrics = self.system("fake_score", prepared)
        self.accelerator.backward(fake_loss)
        fake_grad_norm = self.accelerator.clip_grad_norm_(unwrapped.fake_score.parameters(), self.max_grad_norm)
        self.fake_score_optimizer.step()
        self.fake_score_scheduler.step()
        self.fake_score_optimizer.zero_grad()
        self.fake_score_updates += 1
        self.fake_updates_since_student += 1

        student_updated = self.fake_updates_since_student >= self.student_update_interval
        if student_updated:
            student_loss, student_metrics = self.system("student", prepared)
            self.accelerator.backward(student_loss)
            student_grad_norm = self.accelerator.clip_grad_norm_(unwrapped.student.parameters(), self.max_grad_norm)
            self.student_optimizer.step()
            self.student_scheduler.step()
            self.student_ema.update()
            self.student_optimizer.zero_grad()
            self.student_updates += 1
            self.fake_updates_since_student -= self.student_update_interval
            metrics.update(student_metrics)
            metrics["student_grad_norm"] = float(student_grad_norm.detach())

        metrics.update(
            {
                "student_updated": float(student_updated),
                "student_updates": float(self.student_updates),
                "fake_updates_since_student": float(self.fake_updates_since_student),
                "fake_score_grad_norm": float(fake_grad_norm.detach()),
                "fake_score_lr": float(self.fake_score_scheduler.get_last_lr()[0]),
                "student_lr": float(self.student_scheduler.get_last_lr()[0]),
                "local_samples": float(prepared.clean.shape[0]),
                "local_target_frames": float(prepared.lens.sum().detach()),
                "local_reference_frames": float(prepared.ref_lens.sum().detach()),
                "target_frames_max": float(prepared.lens.max().detach()),
                "target_frames_mean": float(prepared.lens.float().mean().detach()),
            }
        )
        torch.cuda.synchronize(self.accelerator.device)
        metrics["gpu_memory_peak_gib"] = float(torch.cuda.max_memory_allocated(self.accelerator.device) / 2**30)
        metrics["step_time_sec"] = float(time.perf_counter() - step_start)
        return reduce_metrics(
            self.accelerator,
            metrics,
            sum_keys={
                "local_samples": "global_samples",
                "local_target_frames": "global_target_frames",
                "local_reference_frames": "global_reference_frames",
            },
            max_keys=("target_frames_max",),
        )

    def save_checkpoint(self, path: Path, *, with_training_state: bool) -> None:
        """Write a file AukInfer loads directly; with_training_state adds what resuming needs."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            system = self.accelerator.unwrap_model(self.system)
            checkpoint = {
                "ema_model_state_dict": ema_to_inference_state(self.student_ema, self.fusion),
                "update": self.student_updates,
                "metadata": {
                    "nfe": system.time_grid.numel() - 1,
                    "time_grid": system.time_grid.tolist(),
                    "resolved_config": self.resolved_config,
                },
            }
            if with_training_state:
                checkpoint.update(
                    {
                        "student_model_state_dict": system.student.state_dict(),
                        "fake_score_model_state_dict": system.fake_score.state_dict(),
                        "student_optimizer_state_dict": self.student_optimizer.state_dict(),
                        "fake_score_optimizer_state_dict": self.fake_score_optimizer.state_dict(),
                        "student_scheduler_state_dict": self.student_scheduler.state_dict(),
                        "fake_score_scheduler_state_dict": self.fake_score_scheduler.state_dict(),
                        "update_state": {
                            "fake_score_updates": self.fake_score_updates,
                            "student_updates": self.student_updates,
                            "fake_updates_since_student": self.fake_updates_since_student,
                        },
                        "epoch": self.epoch,
                    }
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash mid-write never leaves a truncated model_last.pt.
            temporary = path.with_name(f".{path.name}.tmp")
            self.accelerator.save(checkpoint, temporary)
            temporary.replace(path)
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: Path) -> None:
        """Restore a file written by save_checkpoint(with_training_state=True)."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        system = self.accelerator.unwrap_model(self.system)
        system.student.load_state_dict(checkpoint["student_model_state_dict"])
        system.fake_score.load_state_dict(checkpoint["fake_score_model_state_dict"])
        self.student_optimizer.load_state_dict(checkpoint["student_optimizer_state_dict"])
        self.fake_score_optimizer.load_state_dict(checkpoint["fake_score_optimizer_state_dict"])
        self.student_scheduler.load_state_dict(checkpoint["student_scheduler_state_dict"])
        self.fake_score_scheduler.load_state_dict(checkpoint["fake_score_scheduler_state_dict"])
        ema_from_inference_state(self.student_ema, checkpoint["ema_model_state_dict"])
        update_state = checkpoint["update_state"]
        self.fake_score_updates = update_state["fake_score_updates"]
        self.student_updates = update_state["student_updates"]
        self.fake_updates_since_student = update_state["fake_updates_since_student"]
        self.epoch = checkpoint["epoch"]
