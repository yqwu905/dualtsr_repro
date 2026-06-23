from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dualtsr.config import deep_update, load_config, save_config


DC_AE_REPO = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DualTSR assets and runtime config for Colab.")
    parser.add_argument("--base-config", default="configs/train/colab_emmdit_synth.yaml")
    parser.add_argument("--out-config", default="configs/train/colab_runtime.yaml")
    parser.add_argument("--drive-root", default=None, help="Mounted Google Drive directory for outputs/checkpoints.")
    parser.add_argument("--run-name", default="colab_emmdit_synth")
    parser.add_argument("--dataset", choices=["synth", "ctr"], default="synth")
    parser.add_argument("--ctr-url", default=None, help="Optional Google Drive/HTTP URL for a CTR archive or folder.")
    parser.add_argument("--ctr-root", default="data/ctr_colab", help="Where CTR data is downloaded/extracted.")
    parser.add_argument("--ctr-train-lmdb", default=None)
    parser.add_argument("--ctr-val-lmdb", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-fonts", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def resolve_precision(requested: str | None) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "fp16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    if torch.cuda.is_available():
        return "fp16"
    return "fp32"


def ensure_fonts() -> None:
    run([sys.executable, "scripts/download_fonts.py"])


def ensure_vae(local_dir: Path) -> None:
    if (local_dir / "config.json").exists():
        print(f"VAE already present: {local_dir}")
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=DC_AE_REPO,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_url(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    if "drive.google.com/drive/folders/" in url:
        gdown.download_folder(url=url, output=str(out_dir), quiet=False, remaining_ok=True)
        return
    output = out_dir / Path(url.split("?")[0]).name
    if not output.suffix:
        output = out_dir / "downloaded_ctr_asset"
    if output.exists():
        print(f"Dataset archive already present: {output}")
    else:
        result = gdown.download(url=url, output=str(output), quiet=False, fuzzy=True)
        if result is None:
            raise RuntimeError(f"Failed to download dataset URL: {url}")
    extract_archive(output, out_dir)


def extract_archive(path: Path, out_dir: Path) -> None:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_dir)
    elif suffixes.endswith((".tar", ".tar.gz", ".tgz")):
        mode = "r:gz" if suffixes.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(path, mode) as tf:
            tf.extractall(out_dir)


def find_lmdb_dirs(root: Path) -> list[Path]:
    roots: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        names = set(filenames) | set(dirnames)
        path = Path(dirpath)
        if {"data.mdb", "lock.mdb"} & names:
            roots.append(path)
        elif any(name.startswith("label-") for name in names) and any(name.startswith("image-") for name in names):
            roots.append(path)
    return sorted(set(roots))


def pick_lmdb(candidates: list[Path], split: str) -> Path | None:
    split_words = {
        "train": ("train", "training"),
        "val": ("test", "val", "valid", "validation", "eval"),
    }[split]
    for path in candidates:
        lowered = str(path).lower()
        if any(word in lowered for word in split_words):
            return path
    return candidates[0] if candidates else None


def prepare_ctr(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    ctr_root = ROOT / args.ctr_root
    if args.ctr_url:
        download_url(args.ctr_url, ctr_root)
    candidates = find_lmdb_dirs(ctr_root)
    train_lmdb = Path(args.ctr_train_lmdb) if args.ctr_train_lmdb else pick_lmdb(candidates, "train")
    val_lmdb = Path(args.ctr_val_lmdb) if args.ctr_val_lmdb else pick_lmdb(candidates, "val")
    if train_lmdb is None or val_lmdb is None:
        raise RuntimeError(
            "Could not locate CTR LMDB directories. Pass --ctr-train-lmdb and --ctr-val-lmdb, "
            "or provide --ctr-url pointing to an extracted CTR dataset."
        )
    cfg = deep_update(
        cfg,
        {
            "prepare": {
                "source_lmdb": str(train_lmdb),
                "output_dir": "data/ctr_tsr",
                "vocab_path": "data/ctr_tsr/vocab.txt",
            },
            "data": {
                "type": "ctr_lmdb",
                "train": {"type": "ctr_lmdb", "lmdb_path": str(train_lmdb), "online_degradation": True},
                "val": {
                    "type": "ctr_lmdb",
                    "lmdb_path": str(val_lmdb),
                    "online_degradation": True,
                    "deterministic_degradation": True,
                    "degradation_seed": 1234,
                },
            },
            "tokenizer": {"vocab_path": "data/ctr_tsr/vocab.txt"},
        },
    )
    runtime_cfg = ROOT / ".colab_ctr_runtime.yaml"
    save_config(cfg, runtime_cfg)
    run([sys.executable, "scripts/prepare_ctr_tsr.py", "--config", str(runtime_cfg)])
    runtime_cfg.unlink(missing_ok=True)
    return cfg


def prepare_synth(cfg: dict[str, Any]) -> dict[str, Any]:
    run([sys.executable, "scripts/synthesize_pretrain_data.py", "--config", cfg["_config_path"], "--vocab-only"])
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.base_config)

    output_root = Path(args.drive_root) if args.drive_root else ROOT
    output_dir = output_root / "outputs" / args.run_name
    vae_dir = ROOT / "weights" / "dc-ae-f32c32-sana-1.0-diffusers"

    overrides: dict[str, Any] = {
        "output_dir": str(output_dir),
        "vae": {"pretrained_path": str(vae_dir)},
    }
    if args.max_steps is not None:
        overrides.setdefault("train", {})["max_steps"] = args.max_steps
    if args.save_every is not None:
        overrides.setdefault("checkpoint", {})["save_every"] = args.save_every
    if args.batch_size is not None:
        overrides.setdefault("loader", {})["batch_size"] = args.batch_size
    if args.global_batch_size is not None:
        overrides.setdefault("train", {})["global_batch_size"] = args.global_batch_size
    if args.num_workers is not None:
        overrides.setdefault("loader", {})["num_workers"] = args.num_workers
        overrides.setdefault("loader", {})["persistent_workers"] = args.num_workers > 0
    selected_precision = resolve_precision(args.precision)
    overrides.setdefault("runtime", {})["precision"] = selected_precision
    if args.device is not None:
        overrides.setdefault("runtime", {})["device"] = args.device
    cfg = deep_update(cfg, overrides)

    if not args.skip_fonts:
        ensure_fonts()
    if not args.skip_vae:
        ensure_vae(vae_dir)

    if args.dataset == "ctr":
        cfg = prepare_ctr(args, cfg)
    else:
        synth_cfg = cfg.get("synth", {})
        synth_cfg["output_dir"] = cfg.get("tokenizer", {}).get("vocab_path", "data/synth_colab/vocab.txt")
        synth_cfg["output_dir"] = str(Path(synth_cfg["output_dir"]).parent)
        cfg["synth"] = synth_cfg
        cfg.setdefault("data", {})["synth"] = synth_cfg
        tmp_config = ROOT / ".colab_synth_runtime.yaml"
        save_config(cfg, tmp_config)
        cfg["_config_path"] = str(tmp_config)
        prepare_synth(cfg)
        tmp_config.unlink(missing_ok=True)

    out_config = ROOT / args.out_config
    save_config(cfg, out_config)
    summary = {
        "config": str(out_config),
        "output_dir": str(output_dir),
        "latest_checkpoint": str(output_dir / "checkpoints" / "latest.pt"),
        "resume_command": f"{sys.executable} train.py --config {out_config} --resume auto",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
