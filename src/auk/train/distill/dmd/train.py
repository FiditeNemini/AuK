"""Entry point for stage-two DMD distillation."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
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
from auk.train.distill.checkpoint import extract_teacher_state, load_initializer_artifact, make_student_export
from auk.train.distill.common import (
    WEIGHT_DTYPES,
    DistillScriptArgs,
    FrozenQwenConditioner,
    VAEConditioningAdapter,
    build_train_dataset,
)
from auk.train.distill.dmd.system import DistillConfig, DMDEditSystem
from auk.train.distill.dmd.trainer import DMDVAETrainer
from auk.train.train import DynamicBatchSampler


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
        metadata={"help": "Stage-one artifact for student and fake_score; unset starts both from the teacher."},
    )
    initialized_student_use_ema: bool = True


def save_inference_export(trainer: DMDVAETrainer, fusion: Mapping[str, torch.Tensor], path: Path) -> None:
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        student_state = {
            key.replace("ema_model.", ""): value
            for key, value in trainer.student_ema.ema_model.state_dict().items()
            if key not in ("initted", "step")
        }
        export = make_student_export(
            ema_state=student_state,
            online_state=student_state,
            fusion_state=fusion,
            update=trainer.update_state.student_updates,
            export_type="dmd_student_inference",
            metadata={"time_grid": trainer.time_grid.tolist(), "nfe": trainer.time_grid.numel() - 1},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.accelerator.save(export, path)
        logger.info(f"saved inference export {path}")
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    parser = HfArgumentParser((ScriptArgs, DistillConfig))
    args, distill_config = parser.parse_args_into_dataclasses()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    model_config = OmegaConf.load(args.config).model
    text_encoder_config = model_config.text_encoder
    vae_config = model_config.vae

    torch.manual_seed(distill_config.seed)

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

    logger.info("Building teacher / student / fake_score backbones ...")
    arch = OmegaConf.to_container(model_config.arch, resolve=True)
    arch["attn_backend"] = args.attn_backend
    if args.checkpoint_every_n_layers is not None:
        arch["checkpoint_every_n_layers"] = args.checkpoint_every_n_layers
    system = DMDEditSystem(
        student=Flux2Edit(**arch, latent_dim=vae_config.latent_dim),
        teacher=Flux2Edit(**arch, latent_dim=vae_config.latent_dim),
        fake_score=Flux2Edit(**arch, latent_dim=vae_config.latent_dim),
        config=distill_config,
    )

    logger.info(f"Loading teacher weights from {args.teacher_ckpt} ...")
    teacher_state, fusion = extract_teacher_state({"model_state_dict": load_file(args.teacher_ckpt, device="cpu")}, use_ema=False)
    system.teacher.load_state_dict(teacher_state, strict=True)
    if args.initialized_student_path:
        student_state = load_initializer_artifact(args.initialized_student_path, use_ema=args.initialized_student_use_ema)
        logger.info(f"student and fake_score start from {args.initialized_student_path}")
    else:
        student_state = teacher_state
        logger.warning(
            "no --initialized_student_path given: student and fake_score start from the teacher. "
            "Run auk.train.distill.initializer.train first for faster convergence."
        )
    system.student.load_state_dict(student_state, strict=True)
    system.fake_score.load_state_dict(student_state, strict=True)

    # The three roles run in a single dtype (no autocast): the DMD losses compare
    # teacher/student/fake_score outputs directly, and flash_attn needs bf16/fp16.
    system = system.to(dtype=WEIGHT_DTYPES[args.weight_type])

    adapter = VAEConditioningAdapter(
        vae_model=vae_model,
        text_conditioner=FrozenQwenConditioner(
            text_encoder=thinker,
            layer_weights=fusion["layer_weights"],
            layer_scale=fusion["layer_scale"],
        ),
    )

    student_optimizer = AdamW(
        system.student.parameters(), lr=distill_config.student_learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    fake_score_optimizer = AdamW(
        system.fake_score.parameters(), lr=distill_config.fake_score_learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    max_fake_updates = distill_config.max_student_updates * distill_config.fake_score_updates_per_student
    student_scheduler = get_scheduler(
        distill_config.scheduler_type,
        student_optimizer,
        num_warmup_steps=distill_config.warmup_updates,
        num_training_steps=distill_config.max_student_updates,
    )
    fake_score_scheduler = get_scheduler(
        distill_config.scheduler_type,
        fake_score_optimizer,
        num_warmup_steps=distill_config.warmup_updates,
        num_training_steps=max_fake_updates,
    )

    time_grid = torch.linspace(0, 1, distill_config.nfe + 1)
    time_grid = time_grid + distill_config.sway_sampling_coef * (torch.cos(torch.pi / 2 * time_grid) - 1 + time_grid)
    trainer = DMDVAETrainer(
        system=system,
        adapter=adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        time_grid=time_grid,
        config=distill_config,
        resolved_config={"dmd": asdict(distill_config), "script_args": asdict(args)},
        accelerator=accelerator,
    )

    train_dataset = build_train_dataset(args, text_processor)
    accelerator.even_batches = False
    batch_sampler = DynamicBatchSampler(
        SequentialSampler(train_dataset),
        distill_config.frames_threshold,
        max_samples=distill_config.max_samples,
        random_seed=distill_config.seed,
    )
    dataloader = accelerator.prepare(
        DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            num_workers=distill_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=distill_config.dataloader_num_workers > 0,
            batch_sampler=batch_sampler,
        )
    )

    logger.info(
        f"train={len(train_dataset)} batches/epoch={len(batch_sampler)} | "
        f"student_updates={distill_config.max_student_updates} (fake={max_fake_updates}) | "
        f"config: {asdict(distill_config)}"
    )

    output_dir = Path(args.output_dir)
    resume_ckpt = output_dir / "model_last.pt"
    if resume_ckpt.exists():
        trainer.load_checkpoint(resume_ckpt)
        logger.info(
            f"resumed from {resume_ckpt} | fake_update={trainer.update_state.fake_score_updates} "
            f"student_update={trainer.update_state.student_updates} epoch={trainer.epoch}"
        )
    accelerator.wait_for_everyone()

    saved_resume_at = -1
    saved_export_at = -1
    while trainer.update_state.fake_score_updates < max_fake_updates:
        batch_sampler.set_epoch(trainer.epoch)
        reached_update_limit = False
        for batch in dataloader:
            metrics = trainer.step(trainer.adapter.prepare_batch(batch))

            update = trainer.update_state.fake_score_updates
            if update % distill_config.logging_steps == 0:
                logger.info(
                    f"[dmd] fake_update={update}/{max_fake_updates} "
                    f"student_update={trainer.update_state.student_updates} "
                    + " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
                )

            student_update = trainer.update_state.student_updates
            if metrics["student_updated"] and student_update % distill_config.last_per_student_updates == 0:
                trainer.save_checkpoint(resume_ckpt)
                saved_resume_at = student_update
            if metrics["student_updated"] and student_update % distill_config.save_per_student_updates == 0:
                save_inference_export(trainer, fusion, output_dir / f"model_{student_update}.pt")
                saved_export_at = student_update
            if update >= max_fake_updates:
                reached_update_limit = True
                break
        if reached_update_limit:
            break
        trainer.epoch += 1

    final_update = trainer.update_state.student_updates
    if saved_resume_at != final_update:
        trainer.save_checkpoint(resume_ckpt)
    if saved_export_at != final_update:
        save_inference_export(trainer, fusion, output_dir / f"model_{final_update}.pt")


if __name__ == "__main__":
    sys.exit(main())
