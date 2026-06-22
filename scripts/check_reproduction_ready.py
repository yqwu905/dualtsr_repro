from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import lmdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualtsr.config import load_config
from dualtsr.emmdit import EMMDiTBackbone
from train import gradient_accumulation_steps


PLACEHOLDER_MARKERS = ("/path/to/", "<path", "TODO", "CHANGE_ME")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a DualTSR reproduction config is ready to launch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    return parser.parse_args()


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and any(marker.lower() in value.lower() for marker in PLACEHOLDER_MARKERS)


def lmdb_num_samples(path: Path) -> int | None:
    if not path.is_dir() or not (path / "data.mdb").exists():
        return None
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin(write=False) as txn:
            value = txn.get(b"num-samples")
        return int(value.decode("utf-8")) if value else None
    finally:
        env.close()


def add_path_check(
    checks: list[dict[str, Any]],
    blockers: list[str],
    name: str,
    value: str | None,
    *,
    kind: str,
) -> None:
    if not value or is_placeholder(value):
        blockers.append(f"{name} is not configured")
        checks.append({"name": name, "status": "missing", "path": value})
        return
    path = Path(value).expanduser()
    exists = path.is_dir() if kind == "directory" else path.is_file()
    checks.append({"name": name, "status": "ok" if exists else "missing", "path": str(path)})
    if not exists:
        blockers.append(f"{name} does not exist: {path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []

    data_cfg = config.get("data", {})
    for split in ("train", "val"):
        split_cfg = data_cfg.get(split, {})
        if split_cfg.get("type", data_cfg.get("type")) == "ctr_lmdb":
            value = split_cfg.get("lmdb_path", data_cfg.get("lmdb_path"))
            add_path_check(checks, blockers, f"data.{split}.lmdb_path", value, kind="directory")
            if value and not is_placeholder(value):
                count = lmdb_num_samples(Path(value).expanduser())
                checks.append({"name": f"data.{split}.num_samples", "status": "ok" if count else "missing", "value": count})
                if count is None:
                    blockers.append(f"data.{split}.lmdb_path is not a valid CTR LMDB")

    tokenizer_cfg = config.get("tokenizer", {})
    if tokenizer_cfg.get("vocab_path"):
        add_path_check(checks, blockers, "tokenizer.vocab_path", tokenizer_cfg["vocab_path"], kind="file")

    vae_cfg = config.get("vae", {})
    if vae_cfg.get("type") == "rdp_vae_f8c32":
        add_path_check(checks, blockers, "vae.vae_path", vae_cfg.get("vae_path"), kind="file")
        warnings.append("RDP VAE weights are not publicly discoverable; provide a compatible local checkpoint.")
    elif vae_cfg.get("type") == "autoencoder_dc":
        pretrained_path = vae_cfg.get("pretrained_path")
        if not pretrained_path or is_placeholder(pretrained_path):
            blockers.append("vae.pretrained_path is not configured")
            checks.append({"name": "vae.pretrained_path", "status": "missing", "path": pretrained_path})
        elif "/" in pretrained_path and not Path(pretrained_path).expanduser().exists():
            checks.append({"name": "vae.pretrained_path", "status": "remote", "repo_id": pretrained_path})
            warnings.append("AutoencoderDC will be downloaded from Hugging Face on first launch.")
        else:
            add_path_check(checks, blockers, "vae.pretrained_path", pretrained_path, kind="directory")
    elif vae_cfg.get("class_path", "").endswith("AutoencoderKLVAE"):
        pretrained_path = (vae_cfg.get("init_args") or {}).get("pretrained_path")
        add_path_check(checks, blockers, "vae.init_args.pretrained_path", pretrained_path, kind="directory")

    eval_cfg = config.get("evaluation", {})
    if args.stage in {"evaluate", "all"}:
        transocr_checkpoint = eval_cfg.get("transocr_checkpoint")
        add_path_check(
            checks,
            blockers,
            "evaluation.transocr_checkpoint",
            transocr_checkpoint,
            kind="file",
        )
        add_path_check(
            checks,
            blockers,
            "evaluation.ground_truth",
            eval_cfg.get("ground_truth"),
            kind="file",
        )

    if args.stage in {"train", "all"}:
        try:
            accumulation = gradient_accumulation_steps(config, args.world_size)
            checks.append(
                {
                    "name": "global_batch",
                    "status": "ok",
                    "world_size": args.world_size,
                    "accumulation_steps": accumulation,
                }
            )
        except ValueError as exc:
            blockers.append(str(exc))
            checks.append({"name": "global_batch", "status": "invalid", "error": str(exc)})

    model_cfg = config.get("model", {})
    mmdit_cfg = model_cfg.get("mmdit", {})
    if args.stage in {"train", "all"} and mmdit_cfg.get("class_path") == "dualtsr.emmdit:EMMDiTBackbone":
        init_args = dict(mmdit_cfg.get("init_args", {}) or {})
        try:
            backbone = EMMDiTBackbone(
                hidden_dim=int(model_cfg["hidden_dim"]),
                latent_channels=int(vae_cfg.get("latent_channels", model_cfg.get("latent_channels", 32))),
                latent_size=vae_cfg.get("latent_size", model_cfg.get("latent_size", [4, 16])),
                **init_args,
            )
            parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
            checks.append({"name": "emmdit_parameters", "status": "ok", "value": parameter_count})
        except Exception as exc:
            blockers.append(f"E-MMDiT config is invalid: {exc}")
            checks.append({"name": "emmdit_parameters", "status": "invalid", "error": str(exc)})

    result = {
        "ready": not blockers,
        "config": str(Path(args.config)),
        "stage": args.stage,
        "checks": checks,
        "warnings": warnings,
        "blockers": blockers,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
