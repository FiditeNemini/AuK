# Distilling AuK

AuK distillation trains a few-step student from the AuK base model in two
stages: a consistency initialization, then DMD (distribution matching
distillation). The distilled student samples with 4 steps and no CFG, the
AuK-Flash recipe.

## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training Configuration](#training-configuration)
- [Dynamic batching & memory](#dynamic-batching--memory)
- [Checkpoints, resuming & EMA](#checkpoints-resuming--ema)
- [Monitoring](#monitoring)
- [Inference with a distilled checkpoint](#inference-with-a-distilled-checkpoint)

## Overview

Each stage has its own entry point and launch script:

| Stage | Entry point | Launch script |
| --- | --- | --- |
| 1. Consistency initialization | [`src/auk/train/distill/initializer/train.py`](../src/auk/train/distill/initializer/train.py) | [`scripts/train_init.sh`](../scripts/train_init.sh) |
| 2. DMD | [`src/auk/train/distill/dmd/train.py`](../src/auk/train/distill/dmd/train.py) | [`scripts/train_dmd.sh`](../scripts/train_dmd.sh) |

Code shared by the two stages is in
[`src/auk/train/distill/common.py`](../src/auk/train/distill/common.py).

## Quick Start

### 1. Install training dependencies

```bash
pip install -e ".[train]"
```

### 2. Prepare a training manifest

Same as [fine-tuning](FINETUNING.md#2-prepare-a-training-manifest). Add the `task` field only if you enable `--task_routing` in stage 2. Put `task` at the top level of each line, next to `duration` and `messages`. Records whose `task` is in `--regression_only_tasks` (default `ss,music,fisher`) then only contribute the regression loss, and all other records only contribute the DMD loss:

```jsonl
{"task": "tts", "duration": 5.66, "messages": [{"role": "user", "content": [{"type": "text", "text": "Say the following with the same voice: \"你好，这是一次语音合成测试。\""}, {"type": "audio", "audio_url": "/abs/path/ref.wav"}]}, {"role": "assistant", "content": [{"type": "audio", "audio_url": "/abs/path/target.wav"}]}]}
{"task": "ss", "duration": 4.20, "messages": [{"role": "user", "content": [{"type": "text", "text": "Keep only the first speaker to start talking and remove all other speakers."}, {"type": "audio", "audio_url": "/abs/path/mixture.wav"}]}, {"role": "assistant", "content": [{"type": "audio", "audio_url": "/abs/path/speaker1.wav"}]}]}
```

### 3. Launch stage 1

Edit the resource paths at the top of [`scripts/train_init.sh`](../scripts/train_init.sh):

```bash
train_jsonl=/path/to/train.jsonl # train dataset
config=ckpts/AuK/config.yaml # model config path
teacher_ckpt=ckpts/AuK/auk_base.safetensors # teacher ckpt
output_dir=ckpts/auk_cm_init # save dir
num_gpus=8 # GPUs
```
Then launch training:
```bash
bash scripts/train_init.sh
```

Stage 1 saves the weights to `ckpts/auk_cm_init/model_500.pt`.

### 4. Launch stage 2

Edit the resource paths at the top of [`scripts/train_dmd.sh`](../scripts/train_dmd.sh) and point `initialized_student_path` at the stage 1 weights:

```bash
train_jsonl=/path/to/train.jsonl # train dataset
config=ckpts/AuK/config.yaml # model config path
teacher_ckpt=ckpts/AuK/auk_base.safetensors # teacher ckpt
output_dir=ckpts/auk_dmd # save dir
initialized_student_path=ckpts/auk_cm_init/model_500.pt # stage 1 weights
num_gpus=8 # GPUs
```
Then launch training:
```bash
bash scripts/train_dmd.sh
```

Stage 2 saves the weights to `ckpts/auk_dmd/model_2500.pt`.

## Training Configuration

A distillation run is configured by:
1. The model architecture in the YAML file (`--config`, same as [fine-tuning](FINETUNING.md#model-configuration---config))
2. The training hyperparameters passed through command-line flags

### Stage 1 options

Defaults come from the `InitConfig` dataclass in
[`src/auk/train/distill/initializer/consistency.py`](../src/auk/train/distill/initializer/consistency.py).
Paths come from `ScriptArgs` in
[`src/auk/train/distill/initializer/train.py`](../src/auk/train/distill/initializer/train.py).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--train_jsonl` | *(required)* | Training JSONL. |
| `--output_dir` | `ckpts/auk_cm_init` | Where the weights and a copy of `config.yaml` are saved. |
| `--teacher_ckpt` | `ckpts/AuK/auk_base.safetensors` | Teacher weights to initialize from. |
| `--max_updates` | `500` | Total updates. |
| `--learning_rate` | `1e-5` | AdamW learning rate, constant schedule, no warmup. |
| `--teacher_cfg_scale` | `3.0` | CFG scale of the teacher walk (`--use_cd`). |
| `--use_cd` | `true` | `false` re-noises the ground truth instead of walking the teacher forward. |
| `--weighting` | `default` | Loss weighting: `default` divides by the time gap, `sqrt` by its square root, `one` disables. |
| `--frames_threshold` / `--max_samples` | `10000` / `24` | Batch budget, see [Dynamic batching & memory](#dynamic-batching--memory). |
| `--checkpoint_every_n_layers` | config value | Pass `1`, see [Dynamic batching & memory](#dynamic-batching--memory). |
| `--save_per_updates` / `--last_per_updates` | `500` / `100` | Snapshot / resume-point save intervals. |
| `--ema_beta` | `0.999` | EMA decay. |
| `--attn_backend` | `torch` | `flash_attn` needs a matching wheel. |

### Stage 2 options

Defaults come from the `DMDConfig` dataclass in
[`src/auk/train/distill/dmd/system.py`](../src/auk/train/distill/dmd/system.py).
Paths come from `ScriptArgs` in
[`src/auk/train/distill/dmd/train.py`](../src/auk/train/distill/dmd/train.py).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--train_jsonl` | *(required)* | Training JSONL. |
| `--output_dir` | `ckpts/auk_dmd` | Where the weights and a copy of `config.yaml` are saved. Use a new empty directory, see [Checkpoints, resuming & EMA](#checkpoints-resuming--ema). |
| `--initialized_student_path` | `None` | Stage-one checkpoint for the student and the critic, e.g. `ckpts/auk_cm_init/model_500.pt`. |
| `--initialized_student_use_ema` | `true` | `false` takes the online weights instead (only a stage-one `model_last.pt` has them). |
| `--max_student_updates` | `2500` | Student updates. |
| `--fake_score_updates_per_student` | `5` | Critic updates per student update. |
| `--student_learning_rate` / `--fake_score_learning_rate` | `1e-5` | AdamW learning rates, constant schedule. |
| `--teacher_cfg_scale` | `4.0` | CFG scale of the teacher guidance that the student matches. |
| `--nfe` / `--sway_sampling_coef` | `4` / `-1.0` | Sampling grid of the rollout. Keep these, they are the deployed recipe. |
| `--ca_weight` / `--dm_weight` | `1.0` | Weights of the classifier-alignment and distribution-matching terms. |
| `--regression_weight` | `1.0` | Weight of the regression term on the rollout. |
| `--simulation_free` | `true` | `false` rolls the student out from noise instead of noising the ground truth. Slower. |
| `--task_routing` | `false` | Samples of `--regression_only_tasks` (default `ss,music,fisher`) then only contribute the regression loss, and all other samples only the DMD loss. Needs the `task` field on every record. |
| `--frames_threshold` / `--max_samples` | `10000` / `24` | Batch budget, see [Dynamic batching & memory](#dynamic-batching--memory). |
| `--checkpoint_every_n_layers` | config value | Pass `1`, see [Dynamic batching & memory](#dynamic-batching--memory). |
| `--save_per_student_updates` / `--last_per_student_updates` | `500` / `100` | Snapshot / resume-point save intervals. |
| `--logging_steps` | `1` | Metrics line every N updates (the launch script sets 5). |
| `--ema_beta` | `0.999` | EMA decay. |
| `--attn_backend` | `torch` | `flash_attn` needs a matching wheel. |

## Dynamic batching & memory

Batching works the same as in
[fine-tuning](FINETUNING.md#dynamic-batching--memory).

Pass `--checkpoint_every_n_layers 1` to reduce GPU memory.

## Checkpoints, resuming & EMA

The trainer saves all files under `--output_dir`:

| Path | What it is |
| --- | --- |
| `model_last.pt` | Rolling **resume point**, refreshed every `last_per_updates` / `last_per_student_updates`. Contains model + EMA + optimizer + update/epoch state. |
| `model_{update}.pt` | Numbered **snapshots** every `save_per_updates` / `save_per_student_updates`. Same as `model_last.pt`, plus the sampling `time_grid` in `metadata` — this is the file you deploy. |
| `config.yaml` | Copy of the model config used for this run. |
| `<timestamp>_run.sh` | The exact run command for this run. |

Automatic resume, EMA, and the missing `text_encoder.*` keys work the same as in [fine-tuning](FINETUNING.md#checkpoints-resuming--ema).

## Monitoring

Both trainers print one metrics line per `logging_steps` updates:

| Key | Meaning |
| --- | --- |
| `total_loss` (stage 1) | The consistency loss that stage 1 backpropagates. |
| `consistency_unweighted` (stage 1) | Same distance without the time-gap weighting. Comparable across steps. |
| `dmd_loss` (stage 2) | The distribution-matching term alone. |
| `reg_loss` (stage 2) | Regression of the rollout against the ground truth. |
| `student_loss` (stage 2) | The loss the student step backpropagates: `dmd_loss + regression_weight × reg_loss`. |
| `fake_score_loss` (stage 2) | Critic loss on the rollout. |

`dmd_loss` and `student_loss` are not monotonic even in a healthy run — the
student and the critic improve together and chase each other. Read
`fake_score_loss` and `reg_loss` for decline, and judge the model by generated
audio.

## Inference with a distilled checkpoint

Set the sampling step count to the training `--nfe` (4) and turn CFG off:

```bash
auk-infer \
    --ckpt ckpts/auk_dmd/model_2500.pt \
    --nfe 4 \
    --cfg 0 \
    --audio assets/demo-input-audio/zero-shot-tts/ref.wav \
    --instruction "Say the following with the same voice: 'hello, ...'" \
    --output out.wav \
    --gen_seconds 2.0
```

```python
from auk.infer.infer_auk import AukInfer, save_audio

engine = AukInfer(
    "ckpts/AuK/config.yaml",
    "ckpts/auk_dmd/model_2500.pt",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Say the following with the same voice: 'hello, ...'"},
            {"type": "audio", "audio": "assets/demo-input-audio/zero-shot-tts/ref.wav"},
        ],
    }
]
audio, sr = engine.generate(messages, gen_seconds=2.0, nfe=4, cfg_strength=0.0)
save_audio(audio, sr, "out.wav")
```
