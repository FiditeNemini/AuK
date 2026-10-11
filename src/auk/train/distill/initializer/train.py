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
from torch.utils.data import DataLoader, SequentialSampler
from transformers import HfArgumentParser

from auk.model import Flux2Edit
from auk.train.distill.common import DistillScriptArgs, load_frozen_encoders, load_teacher
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

    torch.manual_seed(init_config.seed)

    model_config = OmegaConf.load(args.config).model

    saved_config = os.path.join(args.output_dir, "config.yaml")
    if accelerator.is_main_process and not os.path.exists(saved_config):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config, saved_config)

    logger.info("Loading Qwen text encoder and VAE ...")
    thinker, text_processor, vae_model = load_frozen_encoders(model_config)

    logger.info("Building student / teacher backbones ...")
    arch = OmegaConf.to_container(model_config.arch, resolve=True)
    arch["attn_backend"] = args.attn_backend
    latent_dim = model_config.vae.latent_dim
    system = ConsistencyInitSystem(
        student=Flux2Edit(**arch, latent_dim=latent_dim),
        teacher=Flux2Edit(**arch, latent_dim=latent_dim),
        config=init_config,
    )

    logger.info(f"Loading teacher weights from {args.teacher_ckpt} ...")
    teacher_state, fusion = load_teacher(args.teacher_ckpt)
    system.teacher.load_state_dict(teacher_state)
    system.student.load_state_dict(teacher_state)
    system = system.to(dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.weight_type])

    trainer = InitializerTrainer(
        system=system,
        vae_model=vae_model,
        text_encoder=thinker,
        fusion=fusion,
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
        random_seed=init_config.seed,
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

    output_dir = Path(args.output_dir)
    last_path = output_dir / "model_last.pt"
    if last_path.exists():
        trainer.load_checkpoint(last_path)
        logger.info(f"resumed from {last_path} | update={trainer.updates} epoch={trainer.epoch}")
    accelerator.wait_for_everyone()

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
