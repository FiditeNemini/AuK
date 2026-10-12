"""Stage-one consistency objective, with t=0 noise / t=1 data and clean-prediction targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from auk.model import CFMEdit
from auk.train.distill.common import (
    PreparedBatch,
    add_noise,
    guided_velocity,
    predict_clean,
)


@dataclass
class InitConfig:
    """Initializer hyper-parameters."""

    eps: float = 1.0e-5
    max_updates: int = 500
    learning_rate: float = 1.0e-5
    scheduler_type: str = "constant"
    warmup_updates: int = 0
    max_grad_norm: float = 1.0

    use_cd: bool = True
    teacher_cfg_scale: float = 3.0
    train_p_mean: float = 0.0
    train_p_std: float = 1.0
    min_r: float = 1.0e-4
    min_time: float = 1.0e-4
    max_time: float = 0.999
    ct_resume_updates: int = 0
    ct_q: float = 2.0
    ct_ratio_limit: float = 0.999
    ct_kimg_per_stage: int = 12500
    huber_const: float = 1.0e-8
    use_squared_l2: bool = False
    weighting: Literal["default", "sqrt", "one"] = "default"

    frames_threshold: int = 10000
    max_samples: int = 24
    dataloader_num_workers: int = 2

    ema_beta: float = 0.999
    ema_update_after_step: int = 0
    ema_update_every: int = 1

    save_per_updates: int = 500
    last_per_updates: int = 100
    logging_steps: int = 1


class ConsistencyInitSystem(nn.Module):
    """Stage-one objective: pull the student towards a few-step solution."""

    def __init__(self, *, student: CFMEdit, teacher: CFMEdit, config: InitConfig):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.config = config
        self.eps = config.eps

    def train(self, mode: bool = True) -> "ConsistencyInitSystem":
        super().train(mode)
        self.teacher.eval()
        return self

    def pair_times(
        self,
        batch: PreparedBatch,
        iteration: int,
        *,
        global_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Sample the (current, endpoint) time pair on the staged shrink schedule.

        The schedule is defined on noise levels (1 = pure noise).
        """
        config = self.config
        normal = torch.randn(batch.clean.shape[0], device=batch.clean.device, dtype=batch.clean.dtype)
        current_noise_level = torch.sigmoid(normal * config.train_p_std + config.train_p_mean)
        current_noise_level = current_noise_level * (config.max_time - config.min_time) + config.min_time
        stage = (iteration + config.ct_resume_updates) * global_batch_size // (config.ct_kimg_per_stage * 1000)
        ratio = min(1.0 - 1.0 / config.ct_q ** (stage + 1), config.ct_ratio_limit)
        # 8.0 is the empirical constant of the ECT sigmoid time gap (https://github.com/locuslab/ect),
        # carried over verbatim from FastGen CM.py t_to_r_sigmoid.
        endpoint_noise_level = current_noise_level - current_noise_level * (1.0 - ratio) * (
            1.0 + 8.0 * torch.sigmoid(-current_noise_level)
        )
        endpoint_noise_level = endpoint_noise_level.clamp_min(config.min_r)
        invalid = endpoint_noise_level >= current_noise_level - self.eps
        current_noise_level = torch.where(
            invalid, (endpoint_noise_level + self.eps).clamp(max=config.max_time), current_noise_level
        )
        endpoint_noise_level = endpoint_noise_level.minimum(current_noise_level - self.eps)
        return 1.0 - current_noise_level, 1.0 - endpoint_noise_level, ratio

    def cm_distance(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: PreparedBatch,
        current: torch.Tensor,
        endpoint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample (weighted, unweighted) consistency distance."""
        config = self.config
        difference = prediction.float() - target.float()
        mask = batch.loss_mask.expand_as(difference).to(difference.dtype)
        distance = ((difference * mask).square().sum(dim=tuple(range(1, difference.ndim)))).sqrt()
        if config.huber_const > 0:
            unweighted = torch.sqrt(distance.square() + config.huber_const**2) - config.huber_const
        elif config.use_squared_l2:
            unweighted = distance.square()
        else:
            unweighted = distance
        delta = (endpoint - current).clamp_min(self.eps)
        if config.weighting == "default":
            return unweighted / delta, unweighted
        if config.weighting == "sqrt":
            return unweighted / delta.sqrt(), unweighted
        if config.weighting == "one":
            return unweighted, unweighted
        raise ValueError(f"unsupported CM weighting: {config.weighting}")

    def forward(
        self,
        batch: PreparedBatch,
        iteration: int,
        *,
        global_batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        config = self.config
        current, endpoint, ratio = self.pair_times(batch, iteration, global_batch_size=global_batch_size)

        noise = torch.randn_like(batch.clean)
        noisy_current = add_noise(batch.clean, noise, current)
        if config.use_cd:
            with torch.no_grad():
                conditional, unconditional = self.teacher.transformer(
                    x=noisy_current,
                    text=batch.text,
                    time=current,
                    mask=batch.valid_mask,
                    c_mask=batch.c_mask,
                    ref=batch.ref,
                    ref_mask=batch.ref_mask,
                    cfg_infer=True,
                ).chunk(2)
                velocity = guided_velocity(conditional, unconditional, config.teacher_cfg_scale)
                noisy_endpoint = noisy_current + (endpoint - current)[:, None, None] * velocity
        else:
            noisy_endpoint = add_noise(batch.clean, noise, endpoint)

        # Both student passes must see the same dropout draw, otherwise the pair
        # differs by dropout noise rather than by the time step alone.
        with torch.random.fork_rng(devices=[noisy_current.device]):
            prediction = predict_clean(self.student.transformer, noisy_current, batch, current)
        with torch.no_grad():
            target = predict_clean(self.student.transformer, noisy_endpoint, batch, endpoint)
        per_sample, unweighted = self.cm_distance(prediction, target, batch, current, endpoint)

        total = per_sample.sum() / per_sample.shape[0]
        return total, {
            "current_time": float(current.mean()),
            "endpoint_time": float(endpoint.mean()),
            "ct_ratio": ratio,
            "consistency_unweighted": float(unweighted.detach().mean()),
            "total_loss": float(total.detach()),
        }
