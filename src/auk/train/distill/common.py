"""Flow parameterization, conditioning, and run plumbing shared by both distillation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.model.utils import lens_to_mask
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig


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


def load_frozen_encoders(
    model_config: DictConfig,
) -> tuple[Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, nn.Module]:
    """Return the frozen Qwen text encoder, its processor, and the frozen VAE."""
    text_encoder_path = model_config.text_encoder.text_encoder_path
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
    del thinker.visual
    thinker.visual = None
    thinker.eval().requires_grad_(False)
    text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_path)

    vae_config = model_config.vae
    model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
    vae_model = load_vae_model(
        vae_name=vae_config.vae_name,
        vae_cfg=BigVGANFlowVAEConfig.from_dict(model_init_kwargs),
        vae_ckpt=vae_config.vae_model_path,
        map_location="cpu",
    )
    vae_model.eval().requires_grad_(False)
    return thinker, text_processor, vae_model


def load_teacher(path: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return (backbone state, layer fusion) of a released AuK safetensors checkpoint."""
    teacher_file = load_file(path)
    backbone = {key.removeprefix("transformer."): value for key, value in teacher_file.items() if key.startswith("transformer.")}
    fusion = {name: teacher_file[name] for name in ("layer_weights", "layer_scale")}
    return backbone, fusion


@torch.no_grad()
def prepare_batch(
    batch: Mapping[str, Any],
    *,
    vae_model: nn.Module,
    text_encoder: nn.Module,
    fusion: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> PreparedBatch:
    """Encode each physical input once, with the checkpoint's fp32 layer fusion over the Qwen hidden states."""
    device = next(vae_model.parameters()).device

    clean, lens = vae_model.encoding_and_normalization(
        batch["audio"].to(device), sample_lengths=batch["audio_lengths"].to(device=device, dtype=torch.long)
    )
    ref, ref_lens = vae_model.encoding_and_normalization(
        batch["ref_audio"].to(device), sample_lengths=batch["ref_audio_lengths"].to(device=device, dtype=torch.long)
    )

    inputs = batch["cond_inputs"].to(device)
    hidden_states = text_encoder(**inputs, output_hidden_states=True).hidden_states[1:]
    width = hidden_states[0].shape[-1]
    stacked = torch.stack([F.layer_norm(value, [width]) for value in hidden_states])
    weights = F.softmax(fusion["layer_weights"], dim=0).to(stacked.dtype)
    text = (stacked * weights[:, None, None, None]).sum(0) * fusion["layer_scale"].to(stacked.dtype)

    return PreparedBatch(
        clean=clean.to(dtype),
        ref=ref.to(dtype),
        text=text.to(dtype),
        lens=lens,
        ref_lens=ref_lens,
        valid_mask=lens_to_mask(lens, length=clean.shape[1]),
        ref_mask=lens_to_mask(ref_lens, length=ref.shape[1]),
        c_mask=inputs["attention_mask"].bool(),
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


def predict_clean(model: nn.Module, noisy: torch.Tensor, batch: PreparedBatch, time: torch.Tensor) -> torch.Tensor:
    """Fully conditioned clean prediction of one backbone."""
    velocity = model(
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


def ema_to_inference_state(ema: EMA, fusion: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """EMA student in the layout AukInfer loads: CFMEdit keys under ema_model., plus EMA's initted/step."""
    state = {key.replace("ema_model.", "ema_model.transformer.", 1): value for key, value in ema.state_dict().items()}
    state.update({f"ema_model.{name}": value for name, value in fusion.items()})
    return state


def ema_from_inference_state(ema: EMA, state: Mapping[str, torch.Tensor]) -> None:
    """Inverse of ema_to_inference_state; the layer fusion stays frozen, so its keys are dropped."""
    ema.load_state_dict(
        {
            key.replace("ema_model.transformer.", "ema_model.", 1): value
            for key, value in state.items()
            if key.startswith("ema_model.transformer.") or key in ("initted", "step")
        }
    )
