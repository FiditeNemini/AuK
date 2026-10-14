"""Three-role DMD objective for a velocity-prediction backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn

from auk.model import CFMEdit


@dataclass
class DMDConfig:
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

    frames_threshold: int = 10000
    max_samples: int = 24
    dataloader_num_workers: int = 4

    ema_beta: float = 0.999
    ema_update_after_step: int = 0
    ema_update_every: int = 1

    save_per_student_updates: int = 500
    last_per_student_updates: int = 100
    logging_steps: int = 1


APG_EPS = 1e-12


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
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
    mask = valid_mask.unsqueeze(-1).expand_as(prediction).to(prediction.dtype)
    squared = (prediction - target).square()
    if mean_per_sample:
        reduce_dims = tuple(range(1, prediction.ndim))
        per_sample = (squared * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(eps)
        return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(eps)
    weight = sample_weight[:, None, None].to(prediction.dtype) * mask
    return (squared * weight).sum() / weight.sum().clamp_min(eps)


def sample_decoupled_time(
    batch: Mapping[str, Any],
    *,
    schedule: str,
    lower_bound: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = torch.rand((batch["clean"].shape[0],), device=batch["clean"].device, dtype=batch["clean"].dtype)
    # "decoupled-hybrid" draws from U(lower_bound, 1): CA only aligns states cleaner than the rollout.
    if schedule == "decoupled-hybrid":
        bound = lower_bound.clamp(0.0, 1.0 - 1e-6)
        return bound + (1.0 - bound) * raw
    if schedule == "uniform":
        return raw
    raise ValueError(f"unsupported decoupled time schedule: {schedule!r}")


class DMDSystem(nn.Module):
    def __init__(self, *, student: CFMEdit, teacher: CFMEdit, fake_score: CFMEdit, config: DMDConfig):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.fake_score = fake_score
        self.config = config
        self.eps = config.eps
        self.regression_only_tasks = (
            tuple(task.strip() for task in config.regression_only_tasks.split(",") if task.strip()) if config.task_routing else ()
        )
        time_grid = torch.linspace(0, 1, config.nfe + 1)
        self.time_grid = time_grid + config.sway_sampling_coef * (torch.cos(torch.pi / 2 * time_grid) - 1 + time_grid)

    def forward(self, mode: Literal["fake_score", "student"], batch: Mapping[str, Any]) -> dict[str, Any]:
        if mode == "fake_score":
            return self.fake_score_loss(batch)
        return self.student_loss(batch)

    def routing_weights(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample (dmd, regression) weights; both all-ones when routing is off."""
        ones = torch.ones(batch["clean"].shape[0], device=batch["clean"].device, dtype=torch.float32)
        if not self.regression_only_tasks:
            return ones, ones
        if None in batch["tasks"]:
            raise ValueError("task_routing needs a 'task' field on every training record")
        regression = torch.tensor(
            [1.0 if task in self.regression_only_tasks else 0.0 for task in batch["tasks"]],
            device=batch["clean"].device,
            dtype=torch.float32,
        )
        return ones - regression, regression

    def student_rollout(self, batch: Mapping[str, Any], *, with_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (clean, time): a student clean prediction at one inference-grid time."""
        time_grid = self.time_grid.to(batch["clean"].device)
        step_index = int(torch.randint(time_grid.numel() - 1, (1,), device=batch["clean"].device).item())
        selected_time = time_grid[step_index : step_index + 1].to(batch["clean"].dtype)
        if self.config.simulation_free:
            noise = torch.randn_like(batch["clean"])
            current = (1.0 - selected_time[:, None, None]) * noise + selected_time[:, None, None] * batch["clean"]
        else:
            current = torch.randn_like(batch["clean"])
            with torch.no_grad():
                for index in range(step_index):
                    step_time = time_grid[index : index + 1]
                    velocity = self.student.transformer(
                        x=current,
                        text=batch["text"],
                        time=step_time.expand(current.shape[0]).to(current.dtype),
                        mask=batch["valid_mask"],
                        c_mask=batch["c_mask"],
                        ref=batch["ref"],
                        ref_mask=batch["ref_mask"],
                    )
                    current = current + (time_grid[index + 1] - time_grid[index]) * velocity

        with torch.set_grad_enabled(with_grad):
            velocity = self.student.transformer(
                x=current,
                text=batch["text"],
                time=selected_time.expand(current.shape[0]),
                mask=batch["valid_mask"],
                c_mask=batch["c_mask"],
                ref=batch["ref"],
                ref_mask=batch["ref_mask"],
            )
            clean = current + (1.0 - selected_time[:, None, None]) * velocity
        return clean, selected_time

    def fake_score_loss(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        rollout_clean, rollout_time = self.student_rollout(batch, with_grad=False)
        time = torch.rand((batch["clean"].shape[0],), device=batch["clean"].device, dtype=batch["clean"].dtype)
        noise = torch.randn_like(rollout_clean)
        noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * rollout_clean
        prediction = self.fake_score.transformer(
            x=noisy,
            text=batch["text"],
            time=time,
            mask=batch["valid_mask"],
            c_mask=batch["c_mask"],
            ref=batch["ref"],
            ref_mask=batch["ref_mask"],
        )
        dmd_weight, _ = self.routing_weights(batch)
        loss = weighted_mse(prediction, rollout_clean - noise, batch["valid_mask"], dmd_weight, self.eps)
        return {
            "fake_score_loss": loss,
            "fake_score_time": float(time.detach().mean()),
            "student_rollout_time": float(rollout_time.detach().mean()),
        }

    def dm_gradient_term(
        self,
        detached_clean: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Distribution-matching direction, the teacher conditional clean it uses, and the time drawn."""
        time = sample_decoupled_time(batch, schedule=self.config.dm_time_schedule)
        noise = torch.randn_like(detached_clean)
        noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * detached_clean
        fake_velocity = self.fake_score.transformer(
            x=noisy,
            text=batch["text"],
            time=time,
            mask=batch["valid_mask"],
            c_mask=batch["c_mask"],
            ref=batch["ref"],
            ref_mask=batch["ref_mask"],
        )
        teacher_velocity = self.teacher.transformer(
            x=noisy,
            text=batch["text"],
            time=time,
            mask=batch["valid_mask"],
            c_mask=batch["c_mask"],
            ref=batch["ref"],
            ref_mask=batch["ref_mask"],
        )
        fake_clean = noisy + (1.0 - time[:, None, None]) * fake_velocity
        teacher_cond_clean = noisy + (1.0 - time[:, None, None]) * teacher_velocity
        return fake_clean - teacher_cond_clean, teacher_cond_clean, time

    def ca_gradient_term(
        self,
        detached_clean: torch.Tensor,
        batch: Mapping[str, Any],
        rollout_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classifier-alignment direction, its un-projected baseline, and the time drawn."""
        config = self.config
        time = sample_decoupled_time(batch, schedule=config.ca_time_schedule, lower_bound=rollout_time)
        noise = torch.randn_like(detached_clean)
        noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * detached_clean
        conditional, unconditional = self.teacher.transformer(
            x=noisy,
            text=batch["text"],
            time=time,
            mask=batch["valid_mask"],
            c_mask=batch["c_mask"],
            ref=batch["ref"],
            ref_mask=batch["ref_mask"],
            cfg_infer=True,
        ).chunk(2)
        guided = unconditional + config.teacher_cfg_scale * (conditional - unconditional)
        cond_clean = noisy + (1.0 - time[:, None, None]) * conditional
        uncond_clean = noisy + (1.0 - time[:, None, None]) * unconditional
        cfg_clean = noisy + (1.0 - time[:, None, None]) * guided
        baseline = cond_clean - cfg_clean
        if not config.ca_apg or (config.ca_apg_eta == 1.0 and config.ca_apg_norm_threshold == 0.0):
            return baseline, baseline, time

        mask = batch["valid_mask"].unsqueeze(-1).expand_as(cond_clean).to(torch.float32)
        reduce_dims = tuple(range(1, cond_clean.ndim))
        guidance = (cond_clean.float() - uncond_clean.float()) * mask
        if config.ca_apg_norm_threshold > 0:
            guidance_norm = guidance.square().sum(dim=reduce_dims).sqrt()
            cap = torch.minimum(
                torch.ones_like(guidance_norm),
                config.ca_apg_norm_threshold / guidance_norm.clamp_min(APG_EPS),
            )
            guidance = guidance * cap[:, None, None]

        anchor = cond_clean.float() * mask
        anchor_norm_sq = anchor.square().sum(dim=reduce_dims, keepdim=True)
        parallel = torch.where(
            anchor_norm_sq > APG_EPS,
            (guidance * anchor).sum(dim=reduce_dims, keepdim=True) / anchor_norm_sq.clamp_min(APG_EPS) * anchor,
            torch.zeros_like(guidance),
        )
        applied = (guidance - parallel + config.ca_apg_eta * parallel).to(baseline.dtype)
        return -(config.teacher_cfg_scale - 1.0) * applied, baseline, time

    def dmd_loss(
        self,
        student_clean: torch.Tensor,
        batch: Mapping[str, Any],
        rollout_time: torch.Tensor,
    ) -> dict[str, Any]:
        """Decompose DMD into CA/DM terms with independent re-noising and one norm."""
        detached_clean = student_clean.detach()
        with torch.no_grad():
            dm_raw, teacher_dm_cond_clean, dm_time = self.dm_gradient_term(detached_clean, batch)
            ca_raw, baseline_ca_raw, ca_time = self.ca_gradient_term(detached_clean, batch, rollout_time)

            residual = (detached_clean - (teacher_dm_cond_clean - baseline_ca_raw)).abs()
            mask = batch["valid_mask"].unsqueeze(-1).expand_as(residual).to(residual.dtype)
            reduce_dims = tuple(range(1, residual.ndim))
            denominator = (residual * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(self.eps)
            denominator = torch.nan_to_num(denominator, nan=self.eps, neginf=self.eps).clamp_min(self.eps)

            # Each term is divided before the weighted sum: in bf16 the two orders
            # do not agree, and this is the one the trained runs used.
            ca_gradient = ca_raw / denominator[:, None, None]
            dm_gradient = dm_raw / denominator[:, None, None]
            dmd_gradient = self.config.ca_weight * ca_gradient + self.config.dm_weight * dm_gradient
            dmd_gradient = torch.nan_to_num(dmd_gradient, posinf=0.0, neginf=0.0)

        surrogate_target = (student_clean - dmd_gradient).detach()
        dmd_weight, _ = self.routing_weights(batch)
        loss = 0.5 * weighted_mse(
            student_clean.float(),
            surrogate_target.float(),
            batch["valid_mask"],
            dmd_weight,
            self.eps,
        )
        return {
            "dmd_loss": loss,
            "dmd_denominator": float(denominator.detach().mean()),
            "decoupled_ca_time": float(ca_time.detach().mean()),
            "decoupled_dm_time": float(dm_time.detach().mean()),
        }

    def student_loss(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        rollout_clean, rollout_time = self.student_rollout(batch, with_grad=True)
        losses = self.dmd_loss(rollout_clean, batch, rollout_time)
        _, reg_sample_weight = self.routing_weights(batch)
        reg_loss = weighted_mse(
            rollout_clean.float(),
            batch["clean"].float(),
            batch["valid_mask"],
            reg_sample_weight,
            self.eps,
            mean_per_sample=True,
        )
        losses.update(
            {
                "student_rollout_time": float(rollout_time.detach().mean()),
                "reg_loss": reg_loss,
                "student_loss": losses["dmd_loss"] + self.config.regression_weight * reg_loss,
            }
        )
        return losses
