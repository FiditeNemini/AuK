"""Entry point for stage-one consistency initializer training."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

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
from auk.train.distill.initializer.consistency import ConsistencyInitSystem, InitConfig
from auk.train.distill.initializer.trainer import InitializerTrainer
from auk.train.train import AukJsonlDataset, DynamicBatchSampler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())
logger = get_logger(__name__)


@dataclass
class ScriptArgs(DistillScriptArgs):
    output_dir: str = "ckpts/auk_cm_init"


def main() -> None:
    parser = HfArgumentParser((ScriptArgs, InitConfig))
    args, init_config = parser.parse_args_into_dataclasses()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    model_config = OmegaConf.load(args.config).model

    saved_config = os.path.join(args.output_dir, "config.yaml")
    if accelerator.is_main_process and not os.path.exists(saved_config):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config, saved_config)

    # loading models
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

    logger.info("Building student / teacher CFMEdit models around one shared Qwen ...")
    model_arc = OmegaConf.to_container(model_config.arch, resolve=True)
    model_arc["attn_backend"] = args.attn_backend
    schedule_config = OmegaConf.to_container(model_config.get("schedule", OmegaConf.create({})), resolve=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.weight_type]
    student, teacher = (
        CFMEdit(
            transformer=Flux2Edit(**model_arc, latent_dim=vae_config.latent_dim).to(dtype),
            text_encoder=thinker,
            text_processor=text_processor,
            num_channels=vae_config.latent_dim,
            **schedule_config,
        )
        for _ in range(2)
    )
    for model in (student, teacher):
        model.layer_weights.requires_grad_(False)
        model.layer_scale.requires_grad_(False)

    logger.info(f"Loading teacher weights from {args.teacher_ckpt} ...")
    teacher_state = load_file(args.teacher_ckpt)
    for role, model in (("teacher", teacher), ("student", student)):
        missing, unexpected = model.load_state_dict(teacher_state, strict=False)
        missing = [key for key in missing if "text_encoder." not in key]
        if missing or unexpected:
            raise RuntimeError(f"{role}: {len(missing)} missing {missing[:5]}, {len(unexpected)} unexpected {unexpected[:5]}")

    # trainer
    trainer = InitializerTrainer(
        system=ConsistencyInitSystem(student=student, teacher=teacher, config=init_config),
        vae_model=vae_model,
        config=init_config,
        resolved_config={"initializer": asdict(init_config), "script_args": asdict(args)},
        accelerator=accelerator,
    )

    train_dataset = AukJsonlDataset(args.train_jsonl, text_processor)
    accelerator.even_batches = False
    batch_sampler = DynamicBatchSampler(
        SequentialSampler(train_dataset),
        init_config.frames_threshold,
        max_samples=init_config.max_samples,
        random_seed=666,
    )
    dataloader = accelerator.prepare(
        DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            num_workers=init_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=init_config.dataloader_num_workers > 0,
            batch_sampler=batch_sampler,
        )
    )

    logger.info(
        f"train={len(train_dataset)} batches/epoch={len(batch_sampler)} | "
        f"max_updates={init_config.max_updates} | config: {asdict(init_config)}"
    )

    # resume
    output_dir = Path(args.output_dir)
    last_path = output_dir / "model_last.pt"
    if last_path.exists():
        trainer.load_checkpoint(last_path)
        logger.info(f"resumed from {last_path} | update={trainer.updates} epoch={trainer.epoch}")
    accelerator.wait_for_everyone()

    # train
    saved_last_at = -1
    saved_at = -1
    while trainer.updates < init_config.max_updates:
        batch_sampler.set_epoch(trainer.epoch)
        for batch in dataloader:
            metrics = trainer.step(batch)

            update = trainer.updates
            if update % init_config.logging_steps == 0:
                logger.info(
                    f"update={update}/{init_config.max_updates} "
                    + " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
                )

            if update % init_config.last_per_updates == 0:
                trainer.save_checkpoint(last_path, with_training_state=True)
                saved_last_at = update
            if update % init_config.save_per_updates == 0:
                trainer.save_checkpoint(output_dir / f"model_{update}.pt", with_training_state=False)
                saved_at = update
            if update >= init_config.max_updates:
                break
        else:
            trainer.epoch += 1

    if saved_last_at != trainer.updates:
        trainer.save_checkpoint(last_path, with_training_state=True)
    if saved_at != trainer.updates:
        trainer.save_checkpoint(output_dir / f"model_{trainer.updates}.pt", with_training_state=False)


if __name__ == "__main__":
    sys.exit(main())
