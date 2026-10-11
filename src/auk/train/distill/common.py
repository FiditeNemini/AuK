"""Flow parameterization, conditioning, and run plumbing shared by both distillation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch
from accelerate import Accelerator
from torch import nn

from auk.model import CFMEdit
from auk.model.utils import lens_to_mask


@dataclass
class DistillScriptArgs:
    """Paths and resources shared by both distillation stages."""

    train_jsonl: str
    config: str = "ckpts/AuK/config.yaml"
    teacher_ckpt: str = "ckpts/AuK/auk_base.safetensors"
    weight_type: Literal["bf16", "fp16", "fp32"] = "bf16"
    attn_backend: str = field(default="torch", metadata={"help": "Overrides model.arch.attn_backend."})


@dataclass
class PreparedBatch:
    """Conditioning shared by teacher, student, and fake-score roles."""

    clean: torch.Tensor
    ref: torch.Tensor
    text: torch.Tensor
    lens: torch.Tensor
    ref_lens: torch.Tensor
    valid_mask: torch.Tensor
    ref_mask: torch.Tensor
    c_mask: torch.Tensor
    tasks: Sequence[str | None] | None = None

    @property
    def loss_mask(self) -> torch.Tensor:
        return self.valid_mask.unsqueeze(-1)


@torch.no_grad()
def prepare_batch(
    batch: Mapping[str, Any],
    *,
    vae_model: nn.Module,
    text_model: CFMEdit,
    dtype: torch.dtype,
) -> PreparedBatch:
    """Encode each physical input once; the text goes through text_model.encode_text."""
    device = next(vae_model.parameters()).device

    clean, lens = vae_model.encoding_and_normalization(
        batch["audio"].to(device), sample_lengths=batch["audio_lengths"].to(device=device, dtype=torch.long)
    )
    ref, ref_lens = vae_model.encoding_and_normalization(
        batch["ref_audio"].to(device), sample_lengths=batch["ref_audio_lengths"].to(device=device, dtype=torch.long)
    )
    text, c_mask = text_model.encode_text(batch["cond_inputs"], device)

    return PreparedBatch(
        clean=clean.to(dtype),
        ref=ref.to(dtype),
        text=text.to(dtype),
        lens=lens,
        ref_lens=ref_lens,
        valid_mask=lens_to_mask(lens, length=clean.shape[1]),
        ref_mask=lens_to_mask(ref_lens, length=ref.shape[1]),
        c_mask=c_mask,
        tasks=batch.get("_tasks"),
    )


def expand_time(time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while time.ndim < target.ndim:
        time = time.unsqueeze(-1)
    return time.to(device=target.device, dtype=target.dtype)


def batch_time(time: torch.Tensor, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    if time.ndim == 0:
        time = time[None]
    if time.numel() == 1:
        time = time.expand(batch_size)
    return time.to(device=reference.device, dtype=reference.dtype)


def add_noise(clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Linear flow interpolation with noise at t=0 and data at t=1."""
    expanded = expand_time(time, clean)
    return (1.0 - expanded) * noise + expanded * clean


def velocity_to_clean(noisy: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor, batch: PreparedBatch) -> torch.Tensor:
    """Clean endpoint implied by a velocity, with padded frames restored to the ground truth."""
    clean = noisy + (1.0 - expand_time(time, noisy)) * velocity
    return torch.where(batch.loss_mask, clean, batch.clean)


def guided_velocity(conditional: torch.Tensor, unconditional: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    """Blend the two CFG branches into one guided velocity.

    Training uses v_g = v_u + s*(v_c-v_u), which equals the inference-side
    v_c + (s-1)*(v_c-v_u); a cfg_scale of s corresponds to w = s-1.
    """
    return unconditional + cfg_scale * (conditional - unconditional)


def predict_clean(backbone: nn.Module, noisy: torch.Tensor, batch: PreparedBatch, time: torch.Tensor) -> torch.Tensor:
    """Fully conditioned clean prediction of one backbone."""
    velocity = backbone(
        x=noisy,
        text=batch.text,
        time=batch_time(time, noisy.shape[0], noisy),
        mask=batch.valid_mask,
        c_mask=batch.c_mask,
        ref=batch.ref,
        ref_mask=batch.ref_mask,
    )
    return velocity_to_clean(noisy, velocity, time, batch)


def reduce_metrics(
    accelerator: Accelerator,
    metrics: Mapping[str, float],
    *,
    sum_keys: Mapping[str, str] | None = None,
    max_keys: Sequence[str] = (),
) -> dict[str, float]:
    """All-reduce every scalar in three grouped collectives instead of one each.

    Keys in sum_keys are summed and renamed to their mapped name, keys in
    max_keys take the maximum, everything else is averaged.
    """
    sum_keys = sum_keys or {}
    grouped: dict[str, list[str]] = {"sum": [], "max": [], "mean": []}
    for name in metrics:
        if name in sum_keys:
            grouped["sum"].append(name)
        elif name in max_keys:
            grouped["max"].append(name)
        else:
            grouped["mean"].append(name)

    reduced: dict[str, float] = {}
    for reduction, names in grouped.items():
        if not names:
            continue
        stacked = torch.tensor(
            [float(metrics[name]) for name in names],
            device=accelerator.device,
            dtype=torch.float64,
        )
        stacked = accelerator.reduce(stacked, reduction=reduction)
        for name, value in zip(names, stacked.tolist()):
            reduced[sum_keys[name] if reduction == "sum" else name] = float(value)
    return reduced
