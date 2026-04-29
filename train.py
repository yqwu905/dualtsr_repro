from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dualtsr.checkpoint import load_checkpoint, save_checkpoint, set_rng_state
from dualtsr.config import load_config, save_config
from dualtsr.data import build_dataset, make_collate_fn, render_text_panel
from dualtsr.device import autocast_context, cleanup_runtime, dtype_from_precision, make_grad_scaler, setup_runtime
from dualtsr.diffusion import (
    antithetic_timesteps,
    cfm_interpolate,
    corrupt_text,
    guided_velocity_target,
    sample_uniform_t,
    token_cross_entropy,
)
from dualtsr.ema import make_ema, update_ema, unwrap_model
from dualtsr.logging import make_summary_writer
from dualtsr.model import build_model
from dualtsr.tokenizer import CharTokenizer
from dualtsr.vae import build_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DualTSR")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Checkpoint path, 'auto', or empty for fresh training.")
    return parser.parse_args()


def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_model_config(config: dict, vae) -> None:
    config.setdefault("model", {})
    config["model"]["latent_channels"] = int(vae.info.latent_channels)
    config["model"]["latent_size"] = list(vae.info.latent_size)


def make_dataloader(config: dict, split: str, tokenizer: CharTokenizer, distributed: bool) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = build_dataset(config, split)
    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if distributed else None
    loader_cfg = config.get("loader", {})
    batch_size = int(loader_cfg.get(f"{split}_batch_size", loader_cfg.get("batch_size", 1)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        drop_last=bool(loader_cfg.get("drop_last", split == "train")),
        collate_fn=make_collate_fn(tokenizer, int(config["data"].get("max_text_length", 24))),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and int(loader_cfg.get("num_workers", 0)) > 0,
    )
    return dataloader, sampler


def make_scheduler(config: dict, optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    max_steps = int(config["train"]["max_steps"])
    warmup = int(config["train"].get("warmup_steps", 0))
    min_lr_ratio = float(config["train"].get("min_lr_ratio", 0.0))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, float(step + 1) / float(warmup))
        denom = max(1, max_steps - warmup)
        progress = min(1.0, max(0.0, float(step - warmup) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def decode_for_log(vae, latent: torch.Tensor, max_items: int) -> torch.Tensor:
    with torch.no_grad():
        return vae.decode(latent[:max_items]).detach().cpu().float().clamp(0, 1)


def maybe_log(
    *,
    writer,
    config: dict,
    step: int,
    tokenizer: CharTokenizer,
    vae,
    batch: dict,
    lr: torch.Tensor,
    hr_latent: torch.Tensor,
    img_latent_pred: torch.Tensor,
    joint_latent_pred: torch.Tensor,
    logits_txt: torch.Tensor,
    logits_joint: torch.Tensor,
    losses: dict[str, torch.Tensor],
    optimizer,
    grad_norm: float,
) -> None:
    log_cfg = config.get("logging", {})
    scalar_every = int(log_cfg.get("scalar_every", 10))
    image_every = int(log_cfg.get("image_every", 500))
    if step % scalar_every == 0:
        for name, value in losses.items():
            writer.add_scalar(f"loss/{name}", float(value.detach().cpu()), step)
        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), step)
        writer.add_scalar("train/grad_norm", float(grad_norm), step)
    if step == 1 or step % image_every == 0:
        max_images = int(log_cfg.get("max_images", 4))
        writer.add_images("images/lq", lr[:max_images].detach().cpu().clamp(0, 1), step)
        writer.add_images("images/gt", vae.decode(hr_latent[:max_images]).detach().cpu().float().clamp(0, 1), step)
        writer.add_images("images/pred_image_path", decode_for_log(vae, img_latent_pred, max_images), step)
        writer.add_images("images/pred_joint_path", decode_for_log(vae, joint_latent_pred, max_images), step)

        pred_txt = tokenizer.batch_decode(logits_txt[:max_images].argmax(dim=-1))
        pred_joint = tokenizer.batch_decode(logits_joint[:max_images].argmax(dim=-1))
        gt_text = batch["text"][:max_images]
        lines = [f"{i}: gt={gt} | text={pt} | joint={pj}" for i, (gt, pt, pj) in enumerate(zip(gt_text, pred_txt, pred_joint))]
        writer.add_text("text/predictions", "\n".join(lines), step)
        writer.add_image("images/text_predictions", render_text_panel(lines), step)
    writer.flush()


def train_step(
    *,
    config: dict,
    model,
    ema_model,
    vae,
    batch: dict,
    tokenizer: CharTokenizer,
    runtime,
    precision: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train_cfg = config["train"]
    device = runtime.device
    hr = batch["hr"].to(device, non_blocking=True)
    lr = batch["lr"].to(device, non_blocking=True)
    tokens = batch["tokens"].to(device, non_blocking=True)

    with torch.no_grad():
        hr_latent = vae.encode(hr).float()
        lr_latent = vae.encode(lr).float()

    b = hr_latent.shape[0]
    guidance_w = float(train_cfg.get("guidance_scale", 1.0))
    cfg_dropout = float(train_cfg.get("cfg_dropout", 0.1))
    delta = float(train_cfg.get("text_delta", 1e-4))
    text_weight_by_time = bool(train_cfg.get("text_loss_weight_by_time", True))

    with autocast_context(device, precision):
        # 1. L_IMG-MG
        t_img = sample_uniform_t(b, device=device, eps=delta)
        noise_img = torch.randn_like(hr_latent)
        x_img_t, u_img = cfm_interpolate(hr_latent, noise_img, t_img)
        with torch.no_grad():
            u_img_cond = ema_model(x_img_t, t_img, text_tokens=tokens, lr=lr_latent)["velocity"]
            u_img_uncond = ema_model(x_img_t, t_img, text_tokens=None, lr=lr_latent)["velocity"]
            u_img_target = guided_velocity_target(u_img, u_img_cond, u_img_uncond, guidance_w)
        img_text_cond = tokens if random.random() > cfg_dropout else None
        out_img = model(x_img_t, t_img, text_tokens=img_text_cond, lr=lr_latent)
        loss_img = F.mse_loss(out_img["velocity"].float(), u_img_target.float())

        # 2. L_TXT with K antithetic timesteps.
        k = int(train_cfg.get("text_timesteps", 8))
        t_txt = antithetic_timesteps(b, k, device=device, delta=delta)
        tokens_rep = tokens[:, None, :].expand(b, k, tokens.shape[1]).reshape(b * k, tokens.shape[1])
        t_txt_flat = t_txt.reshape(b * k)
        txt_masked = corrupt_text(tokens_rep, t_txt_flat, tokenizer.mask_id, tokenizer.pad_id)
        hr_rep = hr_latent[:, None].expand(b, k, *hr_latent.shape[1:]).reshape(b * k, *hr_latent.shape[1:])
        lr_rep = lr_latent[:, None].expand(b, k, *lr_latent.shape[1:]).reshape(b * k, *lr_latent.shape[1:])
        out_txt = model(hr_rep, t_txt_flat, text_tokens=txt_masked, lr=lr_rep)
        loss_txt = token_cross_entropy(
            out_txt["logits"],
            tokens_rep,
            tokenizer.pad_id,
            timestep=t_txt_flat,
            weight_by_time=text_weight_by_time,
            delta=delta,
        )

        # 3. L_Joint-MG
        t_joint = sample_uniform_t(b, device=device, eps=delta)
        noise_joint = torch.randn_like(hr_latent)
        x_joint_t, u_joint = cfm_interpolate(hr_latent, noise_joint, t_joint)
        txt_joint_t = corrupt_text(tokens, t_joint, tokenizer.mask_id, tokenizer.pad_id)
        with torch.no_grad():
            u_joint_cond = ema_model(x_joint_t, t_joint, text_tokens=txt_joint_t, lr=lr_latent)["velocity"]
            u_joint_uncond = ema_model(x_joint_t, t_joint, text_tokens=None, lr=lr_latent)["velocity"]
            u_joint_target = guided_velocity_target(u_joint, u_joint_cond, u_joint_uncond, guidance_w)
        joint_text_cond = txt_joint_t if random.random() > cfg_dropout else None
        out_joint = model(x_joint_t, t_joint, text_tokens=joint_text_cond, lr=lr_latent)
        loss_joint_img = F.mse_loss(out_joint["velocity"].float(), u_joint_target.float())
        loss_joint_txt = token_cross_entropy(out_joint["logits"], tokens, tokenizer.pad_id)
        loss_joint = loss_joint_img + loss_joint_txt

        loss_total = (
            float(train_cfg.get("loss_img_weight", 1.0)) * loss_img
            + float(train_cfg.get("loss_txt_weight", 1.0)) * loss_txt
            + float(train_cfg.get("loss_joint_weight", 1.0)) * loss_joint
        )

    log_t_img = t_img.view(-1, 1, 1, 1)
    log_t_joint = t_joint.view(-1, 1, 1, 1)
    artifacts = {
        "lr": lr,
        "hr_latent": hr_latent,
        "img_latent_pred": (x_img_t - log_t_img * out_img["velocity"]).detach(),
        "joint_latent_pred": (x_joint_t - log_t_joint * out_joint["velocity"]).detach(),
        "logits_txt": out_txt["logits"].detach().view(b, k, tokens.shape[1], -1)[:, 0],
        "logits_joint": out_joint["logits"].detach(),
    }
    losses = {
        "total": loss_total.detach(),
        "img": loss_img.detach(),
        "txt": loss_txt.detach(),
        "joint": loss_joint.detach(),
        "joint_img": loss_joint_img.detach(),
        "joint_txt": loss_joint_txt.detach(),
    }
    return loss_total, losses, artifacts


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = setup_runtime(config)
    try:
        set_seed(int(config.get("runtime", {}).get("seed", 1234)), runtime.rank)
        precision = str(config.get("runtime", {}).get("precision", "fp32"))
        dtype = dtype_from_precision(precision)
        output_dir = Path(config.get("output_dir", "outputs/dualtsr"))
        ckpt_dir = output_dir / "checkpoints"
        log_dir = output_dir / "tensorboard"
        if runtime.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, output_dir / "config.yaml")

        tokenizer = CharTokenizer.from_config(config)
        vae = build_vae(config, runtime.device, dtype=dtype)
        prepare_model_config(config, vae)
        model = build_model(config, tokenizer.vocab_size, tokenizer.mask_id).to(runtime.device)
        ema_model = make_ema(model).to(runtime.device)

        if runtime.distributed:
            device_ids = [runtime.local_rank] if runtime.device.type in {"cuda", "npu"} else None
            model = DistributedDataParallel(model, device_ids=device_ids)

        train_loader, train_sampler = make_dataloader(config, "train", tokenizer, runtime.distributed)
        optim_cfg = config.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            unwrap_model(model).parameters(),
            lr=float(optim_cfg.get("lr", 0.0001)),
            betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
            weight_decay=float(optim_cfg.get("weight_decay", 0.05)),
        )
        scheduler = make_scheduler(config, optimizer)
        scaler = make_grad_scaler(runtime.device, precision)
        writer = make_summary_writer(log_dir) if runtime.is_main else None

        start_step = 0
        start_epoch = 0
        resume = args.resume or config.get("train", {}).get("resume")
        if resume == "auto":
            resume = ckpt_dir / "latest.pt"
        if resume:
            resume_path = Path(resume)
            if resume_path.exists():
                checkpoint = load_checkpoint(resume_path, map_location=runtime.device)
                unwrap_model(model).load_state_dict(checkpoint["model"])
                if checkpoint.get("ema") is not None:
                    ema_model.load_state_dict(checkpoint["ema"])
                if checkpoint.get("optimizer") is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if checkpoint.get("scheduler") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                if checkpoint.get("scaler") is not None:
                    scaler.load_state_dict(checkpoint["scaler"])
                set_rng_state(checkpoint.get("rng"))
                start_step = int(checkpoint.get("step", 0))
                start_epoch = int(checkpoint.get("epoch", 0))
                if runtime.is_main:
                    print(f"Resumed from {resume_path} at step {start_step}.")
            elif runtime.is_main:
                print(f"Resume checkpoint not found, starting fresh: {resume_path}")

        max_steps = int(config["train"]["max_steps"])
        ema_decay = float(config["train"].get("ema_decay", 0.9999))
        grad_clip = float(config["train"].get("grad_clip_norm", 0.0))
        ckpt_every = int(config["checkpoint"].get("save_every", 5000))
        pbar = tqdm(total=max_steps, initial=start_step, disable=not runtime.is_main, desc="train")

        step = start_step
        epoch = start_epoch
        while step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if step >= max_steps:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, losses, artifacts = train_step(
                    config=config,
                    model=model,
                    ema_model=ema_model,
                    vae=vae,
                    batch=batch,
                    tokenizer=tokenizer,
                    runtime=runtime,
                    precision=precision,
                )
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer) if hasattr(scaler, "unscale_") else None
                    grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), float("inf"))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                update_ema(model, ema_model, ema_decay)
                step += 1

                if runtime.is_main and writer is not None:
                    maybe_log(
                        writer=writer,
                        config=config,
                        step=step,
                        tokenizer=tokenizer,
                        vae=vae,
                        batch=batch,
                        lr=artifacts["lr"],
                        hr_latent=artifacts["hr_latent"],
                        img_latent_pred=artifacts["img_latent_pred"],
                        joint_latent_pred=artifacts["joint_latent_pred"],
                        logits_txt=artifacts["logits_txt"],
                        logits_joint=artifacts["logits_joint"],
                        losses=losses,
                        optimizer=optimizer,
                        grad_norm=float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    )
                    if step % ckpt_every == 0 or step == max_steps:
                        save_checkpoint(
                            ckpt_dir / f"step_{step:08d}.pt",
                            model=model,
                            ema_model=ema_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            step=step,
                            epoch=epoch,
                            config=config,
                            tokenizer=tokenizer,
                        )
                        save_checkpoint(
                            ckpt_dir / "latest.pt",
                            model=model,
                            ema_model=ema_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            step=step,
                            epoch=epoch,
                            config=config,
                            tokenizer=tokenizer,
                        )
                pbar.update(1)
            epoch += 1

        pbar.close()
        if writer is not None:
            writer.close()
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    main()

