"""Entry point for stage-one consistency initializer training."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    HfArgumentParser,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    get_scheduler,
)

from auk.model import Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig
from auk.train.distill.checkpoint import extract_teacher_state, make_student_export
from auk.train.distill.common import (
    WEIGHT_DTYPES,
    DistillScriptArgs,
    FrozenQwenConditioner,
    VAEConditioningAdapter,
    build_train_dataset,
)
from auk.train.distill.initializer.consistency import ConsistencyInitSystem, InitConfig
from auk.train.distill.initializer.trainer import InitializerTrainer
from auk.train.train import DynamicBatchSampler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())
logger = get_logger(__name__)


@dataclass
class ScriptArgs(DistillScriptArgs):
    output_dir: str = "ckpts/auk_cm_init"


def save_initializer(trainer: InitializerTrainer, fusion: Mapping[str, torch.Tensor], output_dir: Path) -> None:
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        update = trainer.updates
        ema_state = {
            key.replace("ema_model.", ""): value
            for key, value in trainer.student_ema.ema_model.state_dict().items()
            if key not in ("initted", "step")
        }
        export = make_student_export(
            ema_state=ema_state,
            online_state=trainer.accelerator.unwrap_model(trainer.system).student.state_dict(),
            fusion_state=fusion,
            update=update,
            export_type="dmd_initializer",
            metadata={"resolved_config": trainer.resolved_config},
        )
        path = output_dir / f"model_{update}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.accelerator.save(export, path)
        logger.info(f"saved initializer artifact {path}")
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    parser = HfArgumentParser((ScriptArgs, InitConfig))
    args, init_config = parser.parse_args_into_dataclasses()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    torch.manual_seed(init_config.seed)

    model_config = OmegaConf.load(args.config).model
    text_encoder_config = model_config.text_encoder
    vae_config = model_config.vae

    saved_config = os.path.join(args.output_dir, "config.yaml")
    if accelerator.is_main_process and not os.path.exists(saved_config):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config, saved_config)

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
    vae_model.eval().requires_grad_(False)

    logger.info("Building student / teacher backbones ...")
    arch = OmegaConf.to_container(model_config.arch, resolve=True)
    arch["attn_backend"] = args.attn_backend
    system = ConsistencyInitSystem(
        student=Flux2Edit(**arch, latent_dim=vae_config.latent_dim),
        teacher=Flux2Edit(**arch, latent_dim=vae_config.latent_dim),
        config=init_config,
    )

    logger.info(f"Loading teacher weights from {args.teacher_ckpt} ...")
    teacher_state, fusion = extract_teacher_state({"model_state_dict": load_file(args.teacher_ckpt, device="cpu")}, use_ema=False)
    system.teacher.load_state_dict(teacher_state, strict=True)
    system.student.load_state_dict(teacher_state, strict=True)
    system = system.to(dtype=WEIGHT_DTYPES[args.weight_type])

    adapter = VAEConditioningAdapter(
        vae_model=vae_model,
        text_conditioner=FrozenQwenConditioner(
            text_encoder=thinker,
            layer_weights=fusion["layer_weights"],
            layer_scale=fusion["layer_scale"],
        ),
    )

    optimizer = AdamW(system.student.parameters(), lr=init_config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = get_scheduler(
        init_config.scheduler_type,
        optimizer,
        num_warmup_steps=init_config.warmup_updates,
        num_training_steps=init_config.max_updates,
    )

    trainer = InitializerTrainer(
        system=system,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        config=init_config,
        resolved_config={"initializer": asdict(init_config), "script_args": asdict(args)},
        accelerator=accelerator,
    )

    train_dataset = build_train_dataset(args, text_processor)
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
    accelerator.wait_for_everyone()

    saved_at = -1
    while trainer.updates < init_config.max_updates:
        batch_sampler.set_epoch(trainer.epoch)
        reached_limit = False
        for batch in dataloader:
            metrics = trainer.step(trainer.adapter.prepare_batch(batch))

            update = trainer.updates
            if update % init_config.logging_steps == 0:
                logger.info(
                    f"[init] update={update}/{init_config.max_updates} "
                    + " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
                )

            if update % init_config.save_per_updates == 0:
                save_initializer(trainer, fusion, output_dir)
                saved_at = update
            if update >= init_config.max_updates:
                reached_limit = True
                break
        if reached_limit:
            break
        trainer.epoch += 1

    if saved_at != trainer.updates:
        save_initializer(trainer, fusion, output_dir)


if __name__ == "__main__":
    sys.exit(main())
