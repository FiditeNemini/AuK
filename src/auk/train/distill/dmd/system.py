"""Three-role DMD objective for a velocity-prediction backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn

from auk.train.distill.common import (
    PreparedBatch,
    add_velocity_noise,
    batch_time,
    clean_prediction,
    expand_time,
    guided_velocity,
    normal_like,
    predict_clean,
    task_selection_weight,
)

APG_EPS = 1e-12


@dataclass
class DistillConfig:
    """DMD hyper-parameters."""

    nfe: int = 4
    sway_sampling_coef: float = -1.0
    teacher_cfg_scale: float = 4.0
    fake_score_updates_per_student: int = 5
    # True noises the ground truth (no compounding rollout error); False rolls the student out from noise.
    simulation_free: bool = True
    regression_weight: float = 1.0
    eps: float = 1.0e-5

    task_routing: bool = False
    # samples of these tasks only contribute reg_loss, never DMD or critic
    regression_only_tasks: str = "ss,music,fisher"

    ca_weight: float = 1.0
    dm_weight: float = 1.0
    ca_time_schedule: str = "decoupled-hybrid"  # 'uniform' | 'decoupled-hybrid'
    dm_time_schedule: str = "uniform"
    ca_apg: bool = True
    ca_apg_eta: float = 0.0  # 0 = pure orthogonal projection
    ca_apg_norm_threshold: float = 0.0

    student_learning_rate: float = 1.0e-5
    fake_score_learning_rate: float = 1.0e-5
    scheduler_type: str = "constant"
    warmup_updates: int = 0
    max_student_updates: int = 2500
    max_grad_norm: float = 1.0

    frames_threshold: int = 2700
    max_samples: int = 8
    dataloader_num_workers: int = 4
    seed: int = 666

    ema_beta: float = 0.999
    ema_update_after_step: int = 0
    ema_update_every: int = 1

    save_per_student_updates: int = 500
    last_per_student_updates: int = 100
    logging_steps: int = 1


def routing_weights(
    batch: PreparedBatch,
    regression_only_tasks: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample (dmd, regression) weights; both all-ones when routing is off."""
    ones = torch.ones(batch.clean.shape[0], device=batch.clean.device, dtype=torch.float32)
    if not regression_only_tasks:
        return ones, ones
    regression = task_selection_weight(batch, regression_only_tasks)
    return ones - regression, regression


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    eps: float,
    *,
    mean_per_sample: bool = False,
) -> torch.Tensor:
    """MSE over masked frames, every sample's frames scaled by its routing weight.

    Frames are pooled across the whole batch by default, so longer samples carry
    proportionally more of the loss. With mean_per_sample each sample is reduced
    to its own masked MSE first and the weights then average those, making every
    sample count the same regardless of length.
    """
    mask = loss_mask.expand_as(prediction).to(prediction.dtype)
    squared = (prediction - target).square()
    if mean_per_sample:
        reduce_dims = tuple(range(1, prediction.ndim))
        per_sample = (squared * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(eps)
        weight = sample_weight.to(per_sample.dtype)
        return (per_sample * weight).sum() / weight.sum().clamp_min(eps)
    weight = expand_time(sample_weight, prediction) * mask
    return (squared * weight).sum() / weight.sum().clamp_min(eps)


def sample_decoupled_time(
    batch: PreparedBatch,
    *,
    schedule: str,
    generator: torch.Generator | None,
    lower_bound: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = torch.rand(
        (batch.clean.shape[0],),
        device=batch.clean.device,
        dtype=batch.clean.dtype,
        generator=generator,
    )
    # "decoupled-hybrid" draws from U(lower_bound, 1): CA only aligns states cleaner than the rollout.
    if schedule == "decoupled-hybrid":
        bound = lower_bound.to(raw).clamp(0.0, 1.0 - 1e-6)
        return bound + (1.0 - bound) * raw
    if schedule == "uniform":
        return raw
    raise ValueError(f"unsupported decoupled time schedule: {schedule!r}")


@dataclass
class RolloutResult:
    clean: torch.Tensor
    time: torch.Tensor
    step_index: int


class DMDEditSystem(nn.Module):
    def __init__(self, *, student: nn.Module, teacher: nn.Module, fake_score: nn.Module, config: DistillConfig):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.fake_score = fake_score
        self.config = config
        self.eps = config.eps
        self.regression_only_tasks = (
            tuple(task.strip() for task in config.regression_only_tasks.split(",") if task.strip()) if config.task_routing else ()
        )

    def train(self, mode: bool = True) -> "DMDEditSystem":
        # nn.Module.train recurses into children; the teacher has dropout and must stay in eval.
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        mode: Literal["fake_score", "student"],
        batch: PreparedBatch,
        time_grid: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if mode == "fake_score":
            return self.fake_score_loss(batch, time_grid, generator)
        return self.student_loss(batch, time_grid, generator)

    def student_rollout(
        self,
        batch: PreparedBatch,
        time_grid: torch.Tensor,
        generator: torch.Generator,
        *,
        with_grad: bool,
    ) -> RolloutResult:
        """Produce a student clean prediction at one inference-grid time."""
        step_index = int(
            torch.randint(
                time_grid.numel() - 1,
                (1,),
                device=batch.clean.device,
                generator=generator,
            ).item()
        )
        selected_time = time_grid[step_index : step_index + 1]
        valid_mask = batch.valid_mask.unsqueeze(-1)
        if self.config.simulation_free:
            noise = normal_like(batch.clean, generator=generator)
            current = add_velocity_noise(batch.clean, noise, selected_time) * valid_mask
        else:
            current = normal_like(batch.clean, generator=generator) * valid_mask
            with torch.no_grad():
                for index in range(step_index):
                    step_time = time_grid[index : index + 1]
                    velocity = self.student(
                        x=current,
                        text=batch.text,
                        time=batch_time(step_time, current.shape[0], current),
                        mask=batch.valid_mask,
                        c_mask=batch.c_mask,
                        ref=batch.ref,
                        ref_mask=batch.ref_mask,
                        drop_audio_cond=False,
                        drop_text=False,
                    )
                    current = (current + (time_grid[index + 1] - time_grid[index]) * velocity) * valid_mask

        with torch.set_grad_enabled(with_grad):
            clean = predict_clean(self.student, current, batch, selected_time, conditional=True)
        return RolloutResult(clean=clean, time=selected_time, step_index=step_index)

    def fake_score_loss(
        self,
        batch: PreparedBatch,
        time_grid: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rollout = self.student_rollout(batch, time_grid, generator, with_grad=False)
        time = torch.rand(
            (batch.clean.shape[0],),
            device=batch.clean.device,
            dtype=batch.clean.dtype,
            generator=generator,
        )
        noise = normal_like(rollout.clean, generator=generator)
        noisy = add_velocity_noise(rollout.clean, noise, time)
        prediction = self.fake_score(
            x=noisy,
            text=batch.text,
            time=batch_time(time, noisy.shape[0], noisy),
            mask=batch.valid_mask,
            c_mask=batch.c_mask,
            ref=batch.ref,
            ref_mask=batch.ref_mask,
            drop_audio_cond=False,
            drop_text=False,
        )
        dmd_weight, _ = routing_weights(batch, self.regression_only_tasks)
        loss = weighted_mse(prediction, rollout.clean - noise, batch.loss_mask, dmd_weight, self.eps)
        return loss, {
            "fake_score_loss": float(loss.detach()),
            "fake_score_time": float(time.detach().mean()),
            "student_rollout_time": float(rollout.time.detach().mean()),
            "student_rollout_step": float(rollout.step_index),
        }

    def ca_gradient_term(
        self,
        detached_clean: torch.Tensor,
        batch: PreparedBatch,
        generator: torch.Generator,
        rollout_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classifier-alignment direction, its un-projected baseline, and the time drawn."""
        cfg = self.config
        time = sample_decoupled_time(
            batch,
            schedule=cfg.ca_time_schedule,
            generator=generator,
            lower_bound=rollout_time,
        )
        noise = normal_like(detached_clean, generator=generator)
        noisy = add_velocity_noise(detached_clean, noise, time)
        both = self.teacher(
            x=noisy,
            text=batch.text,
            time=batch_time(time, noisy.shape[0], noisy),
            mask=batch.valid_mask,
            c_mask=batch.c_mask,
            ref=batch.ref,
            ref_mask=batch.ref_mask,
            cfg_infer=True,
        )
        conditional, unconditional = both.chunk(2)
        cond_clean = clean_prediction(noisy, conditional, time, batch)
        uncond_clean = clean_prediction(noisy, unconditional, time, batch)
        cfg_clean = clean_prediction(noisy, guided_velocity(conditional, unconditional, cfg.teacher_cfg_scale), time, batch)
        baseline = cond_clean - cfg_clean
        if not cfg.ca_apg or (cfg.ca_apg_eta == 1.0 and cfg.ca_apg_norm_threshold == 0.0):
            return baseline, baseline, time

        mask = batch.loss_mask.expand_as(cond_clean).to(torch.float32)
        reduce_dims = tuple(range(1, cond_clean.ndim))
        guidance = (cond_clean.float() - uncond_clean.float()) * mask
        if cfg.ca_apg_norm_threshold > 0:
            guidance_norm = guidance.square().sum(dim=reduce_dims).sqrt()
            cap = torch.minimum(
                torch.ones_like(guidance_norm),
                cfg.ca_apg_norm_threshold / guidance_norm.clamp_min(APG_EPS),
            )
            guidance = guidance * cap[:, None, None]

        anchor = cond_clean.float() * mask
        anchor_norm_sq = anchor.square().sum(dim=reduce_dims, keepdim=True)
        parallel = torch.where(
            anchor_norm_sq > APG_EPS,
            (guidance * anchor).sum(dim=reduce_dims, keepdim=True) / anchor_norm_sq.clamp_min(APG_EPS) * anchor,
            torch.zeros_like(guidance),
        )
        applied = (guidance - parallel + cfg.ca_apg_eta * parallel).to(baseline.dtype)
        return -(cfg.teacher_cfg_scale - 1.0) * applied, baseline, time

    def dmd_loss(
        self,
        student_clean: torch.Tensor,
        batch: PreparedBatch,
        generator: torch.Generator,
        rollout_time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Decompose DMD into CA/DM terms with independent re-noising and one norm."""
        detached_clean = student_clean.detach()
        with torch.no_grad():
            dm_time = sample_decoupled_time(batch, schedule=self.config.dm_time_schedule, generator=generator)
            dm_noise = normal_like(detached_clean, generator=generator)
            dm_noisy = add_velocity_noise(detached_clean, dm_noise, dm_time)
            fake_clean = predict_clean(self.fake_score, dm_noisy, batch, dm_time, conditional=True)
            teacher_dm_cond_clean = predict_clean(self.teacher, dm_noisy, batch, dm_time, conditional=True)
            dm_raw = fake_clean - teacher_dm_cond_clean

            ca_raw, baseline_ca_raw, ca_time = self.ca_gradient_term(detached_clean, batch, generator, rollout_time)

            residual = (detached_clean - (teacher_dm_cond_clean - baseline_ca_raw)).abs()
            mask = batch.loss_mask.expand_as(residual).to(residual.dtype)
            reduce_dims = tuple(range(1, residual.ndim))
            denominator = (residual * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(self.eps)
            denominator = torch.nan_to_num(
                denominator,
                nan=self.eps,
                posinf=torch.finfo(denominator.dtype).max,
                neginf=self.eps,
            ).clamp_min(self.eps)

            # Each term is divided before the weighted sum: in bf16 the two orders
            # do not agree, and this is the one the trained runs used.
            ca_gradient = ca_raw / denominator[:, None, None]
            dm_gradient = dm_raw / denominator[:, None, None]
            dmd_gradient = self.config.ca_weight * ca_gradient + self.config.dm_weight * dm_gradient
            dmd_gradient = torch.nan_to_num(dmd_gradient, nan=0.0, posinf=0.0, neginf=0.0) * batch.loss_mask

        surrogate_target = (student_clean - dmd_gradient).detach()
        dmd_weight, _ = routing_weights(batch, self.regression_only_tasks)
        loss = 0.5 * weighted_mse(
            student_clean.float(),
            surrogate_target.float(),
            batch.loss_mask,
            dmd_weight,
            self.eps,
        )
        return loss, {
            "dmd_loss": float(loss.detach()),
            "dmd_denominator": float(denominator.detach().mean()),
            "decoupled_ca_time": float(ca_time.detach().mean()),
            "decoupled_dm_time": float(dm_time.detach().mean()),
        }

    def student_loss(
        self,
        batch: PreparedBatch,
        time_grid: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rollout = self.student_rollout(batch, time_grid, generator, with_grad=True)
        dmd_loss, metrics = self.dmd_loss(rollout.clean, batch, generator, rollout.time)
        _, reg_sample_weight = routing_weights(batch, self.regression_only_tasks)
        reg_loss = weighted_mse(
            rollout.clean.float(),
            batch.clean.float(),
            batch.loss_mask,
            reg_sample_weight,
            self.eps,
            mean_per_sample=True,
        )
        metrics.update(
            {
                "student_rollout_time": float(rollout.time.detach().mean()),
                "student_rollout_step": float(rollout.step_index),
                "reg_loss": float(reg_loss.detach()),
            }
        )
        return dmd_loss + self.config.regression_weight * reg_loss, metrics
