"""Alternating fake-score / student optimizer loop for DMD."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from auk.train.distill.common import PreparedBatch, VAEConditioningAdapter, reduce_metrics
from auk.train.distill.dmd.checkpoint import load_training_checkpoint, save_training_checkpoint
from auk.train.distill.dmd.system import DistillConfig, DMDEditSystem


@dataclass
class DMDUpdateState:
    fake_score_updates: int = 0
    student_updates: int = 0
    fake_updates_since_student: int = 0


class DMDVAETrainer:
    def __init__(
        self,
        *,
        system: DMDEditSystem,
        adapter: VAEConditioningAdapter,
        student_optimizer: Optimizer,
        fake_score_optimizer: Optimizer,
        student_scheduler: LRScheduler,
        fake_score_scheduler: LRScheduler,
        time_grid: torch.Tensor,
        config: DistillConfig,
        resolved_config: Mapping[str, Any],
        accelerator: Accelerator,
    ):
        self.accelerator = accelerator
        self.adapter = adapter.to(self.accelerator.device)
        self.system, self.student_optimizer, self.fake_score_optimizer = self.accelerator.prepare(
            system, student_optimizer, fake_score_optimizer
        )
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.time_grid = time_grid.to(device=self.accelerator.device, dtype=torch.float32)
        self.resolved_config = dict(resolved_config)
        self.student_update_interval = config.fake_score_updates_per_student
        self.max_grad_norm = config.max_grad_norm
        self.update_state = DMDUpdateState()
        self.epoch = 0
        self.batch_cursor = 0
        self.student_ema = EMA(
            system.student,
            beta=config.ema_beta,
            update_after_step=config.ema_update_after_step,
            update_every=config.ema_update_every,
            include_online_model=False,
        ).to(self.accelerator.device)

    def step_generator(self, stream: int) -> torch.Generator:
        # Prime strides keep every (update, rank, loss term) on distinct noise.
        seed = (
            1_000_003 * (self.update_state.fake_score_updates + 1)
            + 97 * self.update_state.student_updates
            + 10_000_019 * self.accelerator.process_index
            + stream
        )
        return torch.Generator(device=self.accelerator.device).manual_seed(seed)

    def step(self, prepared: PreparedBatch) -> dict[str, float]:
        step_start = time.perf_counter()
        self.system.train()
        unwrapped = self.accelerator.unwrap_model(self.system)
        prepared = prepared.to(
            self.accelerator.device,
            dtype=next(unwrapped.student.parameters()).dtype,
        )
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        fake_loss, metrics = self.system("fake_score", prepared, self.time_grid, self.step_generator(stream=1))
        self.accelerator.backward(fake_loss)
        fake_grad_norm = self.accelerator.clip_grad_norm_(unwrapped.fake_score.parameters(), self.max_grad_norm)
        self.fake_score_optimizer.step()
        self.fake_score_scheduler.step()
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        self.update_state.fake_score_updates += 1
        self.update_state.fake_updates_since_student += 1

        student_updated = self.update_state.fake_updates_since_student >= self.student_update_interval
        if student_updated:
            self.student_optimizer.zero_grad(set_to_none=True)
            student_loss, student_metrics = self.system("student", prepared, self.time_grid, self.step_generator(stream=2))
            self.accelerator.backward(student_loss)
            student_grad_norm = self.accelerator.clip_grad_norm_(unwrapped.student.parameters(), self.max_grad_norm)
            self.student_optimizer.step()
            self.student_scheduler.step()
            self.student_ema.update()
            self.student_optimizer.zero_grad(set_to_none=True)
            self.update_state.student_updates += 1
            self.update_state.fake_updates_since_student -= self.student_update_interval
            metrics.update(student_metrics)
            metrics["student_grad_norm"] = float(student_grad_norm.detach())

        self.batch_cursor += 1
        metrics.update(
            {
                "student_updated": float(student_updated),
                "fake_score_updates": float(self.update_state.fake_score_updates),
                "student_updates": float(self.update_state.student_updates),
                "fake_updates_since_student": float(self.update_state.fake_updates_since_student),
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

    def save_checkpoint(self, path: Path) -> None:
        save_training_checkpoint(
            path,
            accelerator=self.accelerator,
            system=self.accelerator.unwrap_model(self.system),
            student_optimizer=self.student_optimizer,
            fake_score_optimizer=self.fake_score_optimizer,
            student_scheduler=self.student_scheduler,
            fake_score_scheduler=self.fake_score_scheduler,
            student_ema_state=self.student_ema.ema_model.state_dict(),
            update_state=asdict(self.update_state),
            epoch=self.epoch,
            batch_cursor=self.batch_cursor,
            resolved_config=self.resolved_config,
        )

    def load_checkpoint(self, path: Path) -> None:
        restored = load_training_checkpoint(
            torch.load(path, map_location="cpu", weights_only=True),
            system=self.accelerator.unwrap_model(self.system),
            student_optimizer=self.student_optimizer,
            fake_score_optimizer=self.fake_score_optimizer,
            student_scheduler=self.student_scheduler,
            fake_score_scheduler=self.fake_score_scheduler,
        )
        self.student_ema.ema_model.load_state_dict(restored["student_ema_state"], strict=True)
        self.update_state = DMDUpdateState(**restored["update_state"])
        self.epoch = restored["epoch"]
        self.batch_cursor = restored["batch_cursor"]
