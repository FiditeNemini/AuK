"""Teacher state extraction and student export shared by both distillation stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


# Bumped whenever the layout of a training checkpoint or an exported artifact changes.
CHECKPOINT_SCHEMA = 1


def extract_teacher_state(
    checkpoint: Mapping[str, Any],
    *,
    use_ema: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return (backbone, text_fusion) from full or online artifacts."""
    if use_ema:
        root_key = "ema_model_state_dict"
        prefix = "ema_model.transformer."
        fusion_prefix = "ema_model."
    else:
        root_key = "model_state_dict"
        prefix = "transformer."
        fusion_prefix = ""
    if root_key not in checkpoint:
        raise KeyError(f"checkpoint does not contain {root_key!r}")

    state = checkpoint[root_key]
    backbone = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not backbone:
        raise KeyError(f"no backbone keys with prefix {prefix!r}")
    fusion = {name: state[f"{fusion_prefix}{name}"] for name in ("layer_weights", "layer_scale")}
    return backbone, fusion


def make_student_export(
    *,
    ema_state: Mapping[str, torch.Tensor],
    online_state: Mapping[str, torch.Tensor],
    fusion_state: Mapping[str, torch.Tensor],
    update: int,
    export_type: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an export carrying the student backbone under both state keys."""
    ema = {f"ema_model.transformer.{key}": value for key, value in ema_state.items()}
    online = {f"transformer.{key}": value for key, value in online_state.items()}
    for name in ("layer_weights", "layer_scale"):
        ema[f"ema_model.{name}"] = fusion_state[name]
        online[name] = fusion_state[name]
    ema["initted"] = torch.tensor(True)
    ema["step"] = torch.tensor(update)
    return {
        "ema_model_state_dict": ema,
        "model_state_dict": online,
        "update": update,
        "schema_version": CHECKPOINT_SCHEMA,
        "export_type": export_type,
        "metadata": dict(metadata),
    }


def load_initializer_artifact(path: str | Path, *, use_ema: bool) -> dict[str, torch.Tensor]:
    """Return the stage-one student backbone."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    export_type = str(checkpoint["export_type"])
    schema_version = int(checkpoint["schema_version"])
    if export_type != "dmd_initializer":
        raise ValueError(f"expected dmd_initializer artifact, got {export_type!r}")
    if schema_version != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported initializer schema {schema_version}; expected {CHECKPOINT_SCHEMA}")
    backbone, _ = extract_teacher_state(checkpoint, use_ema=use_ema)
    return backbone
