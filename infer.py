from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dualtsr.checkpoint import load_checkpoint
from dualtsr.config import load_config
from dualtsr.data import load_rgb, pil_to_tensor
from dualtsr.device import autocast_context, resolve_device
from dualtsr.model import build_model
from dualtsr.tokenizer import BaseTokenizer, build_tokenizer, tokenizer_from_state
from dualtsr.vae import build_vae, update_model_latent_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DualTSR inference")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Override infer.checkpoint.")
    parser.add_argument("--input", default=None, help="Override infer.input_dir or infer.input_manifest.")
    parser.add_argument("--output", default=None, help="Override infer.output_dir.")
    return parser.parse_args()


def list_inputs(config: dict, override: str | None = None) -> list[dict]:
    infer_cfg = config.get("infer", {})
    source = override or infer_cfg.get("input_manifest") or infer_cfg.get("input_dir")
    if not source:
        raise ValueError("infer.input_dir or infer.input_manifest is required.")
    path = Path(source)
    if path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return [{"id": p.stem, "image": str(p)} for p in sorted(path.iterdir()) if p.suffix.lower() in exts]
    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    root = path.parent
    for i, row in enumerate(rows):
        row.setdefault("id", str(i))
        image_path = row.get("lr") or row.get("lq") or row.get("image")
        if image_path is None:
            raise KeyError("Inference manifest rows require lr, lq, or image.")
        p = Path(image_path)
        row["image"] = str(p if p.is_absolute() else root / p)
    return rows


def save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().clamp(0, 1)
    arr = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(path)


@torch.no_grad()
def joint_sample(model, vae, lr: torch.Tensor, tokenizer: BaseTokenizer, config: dict, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    infer_cfg = config.get("infer", {})
    precision = str(config.get("runtime", {}).get("precision", "fp32"))
    steps = int(infer_cfg.get("steps", 4))
    cfg_scale = float(infer_cfg.get("cfg_scale", 1.0))
    seq_len = int(config.get("data", {}).get("max_text_length", 24))
    sampling = str(infer_cfg.get("text_sampling", "sample")).lower()
    allow_special = bool(infer_cfg.get("allow_special_tokens", False))

    lr = lr.to(device)
    lr_latent = vae.encode(lr).float()
    b, c, h, w = lr_latent.shape
    x = torch.randn((b, c, h, w), device=device, dtype=lr_latent.dtype)
    text = torch.full((b, seq_len), tokenizer.mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

    with autocast_context(device, precision):
        for k in range(steps):
            t = timesteps[k].expand(b)
            s = timesteps[k + 1]
            out = model(x, t, text_tokens=text, lr=lr_latent)
            velocity = out["velocity"]
            logits = out["logits"]
            if cfg_scale != 1.0:
                uncond = model(x, t, text_tokens=None, lr=lr_latent)["velocity"]
                velocity = uncond + cfg_scale * (velocity - uncond)
            x = x - (timesteps[k] - s) * velocity

            if not allow_special:
                logits = logits.clone()
                logits[..., tokenizer.pad_id] = -1e9
                logits[..., tokenizer.mask_id] = -1e9
            alpha_t = 1.0 - timesteps[k]
            alpha_s = 1.0 - s
            prob_unmask = ((alpha_s - alpha_t) / (1.0 - alpha_t + 1e-8)).clamp(0.0, 1.0)
            if sampling == "argmax":
                candidates = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits.float(), dim=-1)
                candidates = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(b, seq_len)
            update = text.eq(tokenizer.mask_id) & (torch.rand_like(text.float()) < prob_unmask)
            text[update] = candidates[update]
    decoded = vae.decode(x.float()).float().clamp(0, 1)
    return decoded, tokenizer.batch_decode(text)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config.get("infer", {}).get("checkpoint")
    if not checkpoint_path:
        raise ValueError("infer.checkpoint or --checkpoint is required.")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"]) if checkpoint.get("tokenizer") else build_tokenizer(config)
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    vae = build_vae(config, device)
    update_model_latent_shape(config, vae, device)
    model = build_model(config, tokenizer.vocab_size, tokenizer.mask_id).to(device)
    state = checkpoint.get("ema") if config.get("infer", {}).get("use_ema", True) and checkpoint.get("ema") is not None else checkpoint["model"]
    model.load_state_dict(state)
    model.eval()

    output_dir = Path(args.output or config.get("infer", {}).get("output_dir", "outputs/infer"))
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list_inputs(config, args.input)
    results: list[dict] = []
    hr_size = config.get("data", {}).get("hr_size", [128, 512])
    for row in tqdm(rows, desc="infer"):
        image = pil_to_tensor(load_rgb(row["image"], hr_size)).unsqueeze(0)
        sr, text = joint_sample(model, vae, image, tokenizer, config, device)
        image_path = image_dir / f"{row['id']}.png"
        save_image(sr[0], image_path)
        result = {"id": row["id"], "image": str(image_path), "text": text[0], "input": row["image"]}
        results.append(result)

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image", "text", "input"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()

