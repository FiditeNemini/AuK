"""Resume-checkpoint IO for DMD training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from auk.train.distill.dmd.system import DMDEditSystem


def save_training_checkpoint(
    path: Path,
    *,
    accelerator: Accelerator,
    system: DMDEditSystem,
    student_optimizer: Optimizer,
    fake_score_optimizer: Optimizer,
    student_scheduler: LRScheduler,
    fake_score_scheduler: LRScheduler,
    student_ema_state: Mapping[str, torch.Tensor],
    update_state: Mapping[str, int],
    epoch: int,
    batch_cursor: int,
    resolved_config: Mapping[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint = {
            "student_model_state_dict": system.student.state_dict(),
            "fake_score_model_state_dict": system.fake_score.state_dict(),
            "student_ema_model_state_dict": dict(student_ema_state),
            "student_optimizer_state_dict": student_optimizer.state_dict(),
            "fake_score_optimizer_state_dict": fake_score_optimizer.state_dict(),
            "student_scheduler_state_dict": student_scheduler.state_dict(),
            "fake_score_scheduler_state_dict": fake_score_scheduler.state_dict(),
            "update_state": dict(update_state),
            "epoch": epoch,
            "batch_cursor": batch_cursor,
            "resolved_config": dict(resolved_config),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write never leaves a truncated model_last.pt.
        temporary = path.with_name(f".{path.name}.tmp")
        accelerator.save(checkpoint, temporary)
        temporary.replace(path)
    accelerator.wait_for_everyone()


def load_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    system: DMDEditSystem,
    student_optimizer: Optimizer,
    fake_score_optimizer: Optimizer,
    student_scheduler: LRScheduler,
    fake_score_scheduler: LRScheduler,
) -> dict[str, Any]:
    system.student.load_state_dict(checkpoint["student_model_state_dict"], strict=True)
    system.fake_score.load_state_dict(checkpoint["fake_score_model_state_dict"], strict=True)
    student_optimizer.load_state_dict(checkpoint["student_optimizer_state_dict"])
    fake_score_optimizer.load_state_dict(checkpoint["fake_score_optimizer_state_dict"])
    student_scheduler.load_state_dict(checkpoint["student_scheduler_state_dict"])
    fake_score_scheduler.load_state_dict(checkpoint["fake_score_scheduler_state_dict"])
    return {
        "student_ema_state": checkpoint["student_ema_model_state_dict"],
        "update_state": dict(checkpoint["update_state"]),
        "epoch": int(checkpoint["epoch"]),
        "batch_cursor": int(checkpoint["batch_cursor"]),
    }
