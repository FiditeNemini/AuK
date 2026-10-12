"""Entry point for stage-two DMD distillation."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, SequentialSampler
from transformers import HfArgumentParser, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.model import CFMEdit, Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig
from auk.train.distill.common import DistillScriptArgs
from auk.train.distill.dmd.system import DMDConfig, DMDSystem
from auk.train.distill.dmd.trainer import DMDTrainer
from auk.train.train import AukJsonlDataset, DynamicBatchSampler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())
logger = get_logger(__name__)


@dataclass
class ScriptArgs(DistillScriptArgs):
    output_dir: str = "ckpts/auk_dmd"
    checkpoint_every_n_layers: int | None = field(
        default=None,
        metadata={"help": "Overrides model.arch.checkpoint_every_n_layers; DMD holds three backbones, so 1 may be needed."},
    )
    initialized_student_path: str | None = field(
        default=None,
        metadata={"help": "Stage-one checkpoint for student and fake_score; unset starts both from the teacher."},
    )
    initialized_student_use_ema: bool = field(
        default=True,
        metadata={"help": "False takes the online student, which only a stage-one model_last.pt carries."},
    )


class AukJsonlDistillDataset(AukJsonlDataset):
    """AukJsonlDataset plus the optional per-record task label that task routing selects on."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["task"] = self.data[index].get("task")
        return item

    def collate_fn(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        collated = super().collate_fn(batch)
        collated["_tasks"] = [item["task"] for item in batch]
        return collated


def main() -> None:
    parser = HfArgumentParser((ScriptArgs, DMDConfig))
    args, dmd_config = parser.parse_args_into_dataclasses()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    model_config = OmegaConf.load(args.config).model

    saved_config = os.path.join(args.output_dir, "config.yaml")
    if accelerator.is_main_process and not os.path.exists(saved_config):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config, saved_config)

    # load models
    text_encoder_config = model_config.text_encoder
    vae_config = model_config.vae

    logger.info(f"Loading Qwen text encoder from {text_encoder_config.text_encoder_path} ...")
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        text_encoder_config.text_encoder_path, torch_dtype=torch.bfloat16
    )
    del thinker.visual
    thinker.visual = None
    text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_config.text_encoder_path)

    logger.info(f"Loading VAE from {vae_config.vae_model_path} ...")
    model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
    vae_model = load_vae_model(
        vae_name=vae_config.vae_name,
        vae_cfg=BigVGANFlowVAEConfig.from_dict(model_init_kwargs),
        vae_ckpt=vae_config.vae_model_path,
        map_location="cpu",
    )
    vae_model.eval()
    vae_model.requires_grad_(False)

    logger.info("Building teacher / student / fake_score CFMEdit models around one shared Qwen ...")
    model_arc = OmegaConf.to_container(model_config.arch, resolve=True)
    model_arc["attn_backend"] = args.attn_backend
    if args.checkpoint_every_n_layers is not None:
        model_arc["checkpoint_every_n_layers"] = args.checkpoint_every_n_layers
    schedule_config = OmegaConf.to_container(model_config.get("schedule", OmegaConf.create({})), resolve=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.weight_type]
    student, teacher, fake_score = (
        CFMEdit(
            transformer=Flux2Edit(**model_arc, latent_dim=vae_config.latent_dim).to(dtype),
            text_encoder=thinker,
            text_processor=text_processor,
            num_channels=vae_config.latent_dim,
            **schedule_config,
        )
        for _ in range(3)
    )
    for model in (student, teacher, fake_score):
        model.layer_weights.requires_grad_(False)
        model.layer_scale.requires_grad_(False)

    logger.info(f"Loading teacher weights from {args.teacher_ckpt} ...")
    teacher_state = load_file(args.teacher_ckpt)
    if args.initialized_student_path:
        initialized = torch.load(args.initialized_student_path, map_location="cpu", weights_only=True, mmap=True)
        if args.initialized_student_use_ema:
            student_state = {
                key.removeprefix("ema_model."): value
                for key, value in initialized["ema_model_state_dict"].items()
                if key not in ("initted", "step")
            }
        else:
            student_state = initialized["student_model_state_dict"]
        logger.info(f"student and fake_score start from {args.initialized_student_path}")
    else:
        student_state = teacher_state
        logger.warning(
            "no --initialized_student_path given: student and fake_score start from the teacher. "
            "Run auk.train.distill.initializer.train first for faster convergence."
        )
    for role, model, state in (
        ("teacher", teacher, teacher_state),
        ("student", student, student_state),
        ("fake_score", fake_score, student_state),
    ):
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [key for key in missing if "text_encoder." not in key]
        if missing or unexpected:
            raise RuntimeError(f"{role}: {len(missing)} missing {missing[:5]}, {len(unexpected)} unexpected {unexpected[:5]}")

    trainer = DMDTrainer(
        system=DMDSystem(student=student, teacher=teacher, fake_score=fake_score, config=dmd_config),
        vae_model=vae_model,
        config=dmd_config,
        resolved_config={"dmd": asdict(dmd_config), "script_args": asdict(args)},
        accelerator=accelerator,
    )

    train_dataset = AukJsonlDistillDataset(args.train_jsonl, text_processor)
    accelerator.even_batches = False
    batch_sampler = DynamicBatchSampler(
        SequentialSampler(train_dataset),
        dmd_config.frames_threshold,
        max_samples=dmd_config.max_samples,
        random_seed=666,
    )
    dataloader = accelerator.prepare(
        DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            num_workers=dmd_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=dmd_config.dataloader_num_workers > 0,
            batch_sampler=batch_sampler,
        )
    )

    max_fake_updates = trainer.max_fake_score_updates
    logger.info(
        f"train={len(train_dataset)} batches/epoch={len(batch_sampler)} | "
        f"student_updates={dmd_config.max_student_updates} (fake={max_fake_updates}) | "
        f"config: {asdict(dmd_config)}"
    )

    # resume
    output_dir = Path(args.output_dir)
    last_path = output_dir / "model_last.pt"
    if last_path.exists():
        trainer.load_checkpoint(last_path)
        logger.info(
            f"resumed from {last_path} | fake_update={trainer.fake_score_updates} "
            f"student_update={trainer.student_updates} epoch={trainer.epoch}"
        )
    accelerator.wait_for_everyone()

    # train
    saved_last_at = -1
    saved_at = -1
    while trainer.fake_score_updates < max_fake_updates:
        batch_sampler.set_epoch(trainer.epoch)
        for batch in dataloader:
            metrics = trainer.step(batch)

            update = trainer.fake_score_updates
            if update % dmd_config.logging_steps == 0:
                logger.info(
                    f"fake_update={update}/{max_fake_updates} student_update={trainer.student_updates} "
                    + " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
                )

            student_update = trainer.student_updates
            if metrics["student_updated"] and student_update % dmd_config.last_per_student_updates == 0:
                trainer.save_checkpoint(last_path, with_training_state=True)
                saved_last_at = student_update
            if metrics["student_updated"] and student_update % dmd_config.save_per_student_updates == 0:
                trainer.save_checkpoint(output_dir / f"model_{student_update}.pt", with_training_state=False)
                saved_at = student_update
            if update >= max_fake_updates:
                break
        else:
            trainer.epoch += 1

    if saved_last_at != trainer.student_updates:
        trainer.save_checkpoint(last_path, with_training_state=True)
    if saved_at != trainer.student_updates:
        trainer.save_checkpoint(output_dir / f"model_{trainer.student_updates}.pt", with_training_state=False)


if __name__ == "__main__":
    sys.exit(main())
