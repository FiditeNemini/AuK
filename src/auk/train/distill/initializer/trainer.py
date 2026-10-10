from __future__ import annotations

import time
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from auk.train.distill.common import PreparedBatch, VAEConditioningAdapter, reduce_metrics
from auk.train.distill.initializer.consistency import ConsistencyInitSystem, InitConfig


class InitializerTrainer:
    def __init__(
        self,
        *,
        system: ConsistencyInitSystem,
        adapter: VAEConditioningAdapter,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        config: InitConfig,
        resolved_config: Mapping[str, Any],
        accelerator: Accelerator,
    ):
        self.accelerator = accelerator
        self.adapter = adapter.to(self.accelerator.device)
        self.system, self.optimizer = self.accelerator.prepare(system, optimizer)
        self.scheduler = scheduler
        self.resolved_config = dict(resolved_config)
        self.max_grad_norm = config.max_grad_norm
        self.updates = 0
        self.epoch = 0
        self.student_ema = EMA(
            system.student,
            beta=config.ema_beta,
            update_after_step=config.ema_update_after_step,
            update_every=config.ema_update_every,
            include_online_model=False,
        ).to(self.accelerator.device)

    def step(self, prepared: PreparedBatch) -> dict[str, float]:
        step_start = time.perf_counter()
        self.system.train()
        student = self.accelerator.unwrap_model(self.system).student
        prepared = prepared.to(self.accelerator.device, dtype=next(student.parameters()).dtype)
        # The CM time-pair schedule advances by global samples, so all ranks must agree on it.
        global_batch_size = int(
            self.accelerator.reduce(
                torch.tensor(prepared.clean.shape[0], device=self.accelerator.device, dtype=torch.long),
                reduction="sum",
            ).item()
        )
        # Prime strides keep every (update, rank) pair on distinct noise.
        generator = torch.Generator(device=self.accelerator.device).manual_seed(
            1_000_003 * (self.updates + 1) + 10_000_019 * self.accelerator.process_index
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.system(prepared, self.updates, generator, global_batch_size=global_batch_size)
        self.accelerator.backward(loss)
        grad_norm = self.accelerator.clip_grad_norm_(student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.student_ema.update()
        self.updates += 1
        metrics.update(
            {
                "grad_norm": float(grad_norm.detach()),
                "lr": float(self.scheduler.get_last_lr()[0]),
                "local_samples": float(prepared.clean.shape[0]),
                "target_frames_mean": float(prepared.lens.float().mean().detach()),
                "gpu_memory_peak_gib": float(torch.cuda.max_memory_allocated(self.accelerator.device) / 2**30),
                "step_time_sec": float(time.perf_counter() - step_start),
            }
        )
        return reduce_metrics(self.accelerator, metrics)
