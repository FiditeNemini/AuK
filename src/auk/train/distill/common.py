"""Conditioning and run plumbing shared by both distillation stages."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from accelerate import Accelerator
from torch import nn

from auk.model import CFMEdit
from auk.model.utils import lens_to_mask


@torch.no_grad()
def prepare_batch(
    batch: Mapping[str, Any],
    *,
    vae_model: nn.Module,
    text_model: CFMEdit,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Encode each physical input once."""
    device = next(vae_model.parameters()).device

    clean, lens = vae_model.encoding_and_normalization(
        batch["audio"].to(device), sample_lengths=batch["audio_lengths"].to(device)
    )
    ref, ref_lens = vae_model.encoding_and_normalization(
        batch["ref_audio"].to(device), sample_lengths=batch["ref_audio_lengths"].to(device)
    )
    text, c_mask = text_model.encode_text(batch["cond_inputs"], device)

    return {
        "clean": clean.to(dtype),
        "ref": ref.to(dtype),
        "text": text.to(dtype),
        "lens": lens,
        "ref_lens": ref_lens,
        "valid_mask": lens_to_mask(lens, length=clean.shape[1]),
        "ref_mask": lens_to_mask(ref_lens, length=ref.shape[1]),
        "c_mask": c_mask,
        "tasks": batch["tasks"],
    }


def reduce_metrics(
    accelerator: Accelerator,
    metrics: Mapping[str, float | torch.Tensor],
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
