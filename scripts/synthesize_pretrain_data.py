"""Offline synthesis pipeline for DualTSR pretraining data.

从开源字体渲染文本行图像,输出两种格式之一:

- ``lmdb``:与官方 CTR LMDB 相同的键格式(num-samples / image-%09d /
  label-%09d),可直接用 ``data.type: ctr_lmdb`` 训练。
- ``images``:PNG/JPEG 文件夹 + manifest.jsonl,可用 ``data.type: manifest``。

同时生成 vocab.txt(字符集 ∪ 实际采样文本)。LR 不落盘,训练时在线退化。

用法::

    python3 scripts/download_fonts.py
    python3 scripts/synthesize_pretrain_data.py --config configs/train/dualtsr_pretrain_synth.yaml
    python3 scripts/synthesize_pretrain_data.py --out data/synth_demo --num-samples 1000 --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from dualtsr.config import load_config
from dualtsr.tokenizer import CharTokenizer

DEFAULT_SYNTH_CFG: dict[str, Any] = {
    "font_dir": "assets/fonts",
    "output_dir": "data/synth_pretrain",
    "num_samples": 10000,
    "format": "lmdb",  # lmdb | images
    "image_format": "jpeg",  # jpeg | png
    "jpeg_quality": 92,
    "seed": 1234,
    "workers": 0,  # 0 -> os.cpu_count()
    "hr_size": [128, 512],
    "resize_to_hr": True,
    "max_text_length": 24,
    "charset_min_fonts": 6,
    "corpus_path": None,
    "bg_image_dir": None,
    "text": {},
    "render": {},
}

_WORKER: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize font-rendered pretraining data for DualTSR.")
    parser.add_argument("--config", default=None, help="YAML with a top-level `synth` section.")
    parser.add_argument("--font-dir", default=None)
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--format", choices=["lmdb", "images"], default=None)
    parser.add_argument("--image-format", choices=["jpeg", "png"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--corpus", default=None, help="Optional corpus file, one text line per row.")
    parser.add_argument("--bg-dir", default=None, help="Optional directory of background photos.")
    parser.add_argument("--charset-min-fonts", type=int, default=None)
    parser.add_argument("--no-resize", action="store_true", help="Keep natural render size instead of hr_size.")
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only write vocab.txt from the font-coverage charset (for online synth_render training).",
    )
    return parser.parse_args()


def resolve_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_SYNTH_CFG)
    if args.config:
        file_cfg = load_config(args.config)
        cfg.update(file_cfg.get("synth", {}))
        data_cfg = file_cfg.get("data", {})
        cfg.setdefault("hr_size", data_cfg.get("hr_size", cfg["hr_size"]))
        if "max_text_length" in data_cfg:
            cfg["max_text_length"] = data_cfg["max_text_length"]
    overrides = {
        "font_dir": args.font_dir,
        "output_dir": args.out,
        "num_samples": args.num_samples,
        "format": args.format,
        "image_format": args.image_format,
        "seed": args.seed,
        "workers": args.workers,
        "corpus_path": args.corpus,
        "bg_image_dir": args.bg_dir,
        "charset_min_fonts": args.charset_min_fonts,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.no_resize:
        cfg["resize_to_hr"] = False
    return cfg


def _init_worker(cfg: dict[str, Any]) -> None:
    from dualtsr.synth import SynthTextRenderer, TextSampler

    hr_size = tuple(int(v) for v in cfg["hr_size"]) if cfg["resize_to_hr"] else None
    renderer = SynthTextRenderer(
        cfg["font_dir"],
        hr_size=hr_size,
        cfg=cfg.get("render") or {},
        bg_image_dir=cfg.get("bg_image_dir"),
    )
    charset = renderer.fonts.build_charset(int(cfg["charset_min_fonts"]))
    sampler = TextSampler(
        charset,
        max_text_length=int(cfg["max_text_length"]),
        corpus_path=cfg.get("corpus_path"),
        cfg=cfg.get("text") or {},
    )
    _WORKER.update(renderer=renderer, sampler=sampler, cfg=cfg)


def _render_one(idx: int) -> tuple[int, bytes, str]:
    cfg = _WORKER["cfg"]
    rng = random.Random((int(cfg["seed"]) << 32) ^ idx)
    image = None
    text = ""
    for _ in range(8):
        text = _WORKER["sampler"].sample(rng)
        image = _WORKER["renderer"].render(text, rng)
        if image is not None:
            break
    if image is None:
        raise RuntimeError(f"No font covers sampled text: {text!r}")
    buf = io.BytesIO()
    if cfg["image_format"] == "png":
        image.save(buf, format="PNG")
    else:
        image.save(buf, format="JPEG", quality=int(cfg["jpeg_quality"]))
    return idx, buf.getvalue(), text


def _iter_rendered(cfg: dict[str, Any], n: int):
    workers = int(cfg["workers"]) or (mp.cpu_count() or 1)
    if workers <= 1:
        _init_worker(cfg)
        for idx in range(1, n + 1):
            yield _render_one(idx)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        yield from pool.imap(_render_one, range(1, n + 1), chunksize=32)


def write_lmdb(cfg: dict[str, Any], out_dir: Path, n: int) -> list[str]:
    import lmdb

    per_image = 400_000 if cfg["image_format"] == "png" else 120_000
    map_size = max(1 << 30, n * per_image * 2)
    lmdb_dir = out_dir / "lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    env = lmdb.open(str(lmdb_dir), map_size=map_size)
    try:
        txn = env.begin(write=True)
        for count, (idx, payload, text) in enumerate(
            tqdm(_iter_rendered(cfg, n), total=n, desc="synthesize"), start=1
        ):
            txn.put(f"image-{idx:09d}".encode("ascii"), payload)
            txn.put(f"label-{idx:09d}".encode("ascii"), text.encode("utf-8"))
            texts.append(text)
            if count % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.put(b"num-samples", str(n).encode("utf-8"))
        txn.commit()
    finally:
        env.close()
    print(f"lmdb: {lmdb_dir}")
    return texts


def write_images(cfg: dict[str, Any], out_dir: Path, n: int) -> list[str]:
    image_dir = out_dir / "hr"
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = "png" if cfg["image_format"] == "png" else "jpg"
    manifest_path = out_dir / "manifest.jsonl"
    texts: list[str] = []
    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, payload, text in tqdm(_iter_rendered(cfg, n), total=n, desc="synthesize"):
            name = f"{idx:09d}.{suffix}"
            (image_dir / name).write_bytes(payload)
            mf.write(json.dumps({"id": str(idx), "hr": f"hr/{name}", "text": text}, ensure_ascii=False) + "\n")
            texts.append(text)
    print(f"manifest: {manifest_path} images: {image_dir}")
    return texts


def main() -> None:
    args = parse_args()
    cfg = resolve_cfg(args)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg["num_samples"])

    # 字符集仅用于 vocab;渲染进程各自重建(字体句柄不可跨进程)。
    from dualtsr.synth import FontPool

    charset = FontPool(cfg["font_dir"]).build_charset(int(cfg["charset_min_fonts"]))
    if args.vocab_only:
        vocab_path = out_dir / "vocab.txt"
        CharTokenizer.write_vocab(vocab_path, [charset])
        print(f"charset={len(charset)} vocab={vocab_path}")
        return
    print(f"fonts ready, charset={len(charset)} chars; rendering {n} samples...")

    if cfg["format"] == "lmdb":
        texts = write_lmdb(cfg, out_dir, n)
    else:
        texts = write_images(cfg, out_dir, n)

    vocab_path = out_dir / "vocab.txt"
    CharTokenizer.write_vocab(vocab_path, [charset, *texts])
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    meta["charset_size"] = len(charset)
    (out_dir / "synth_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"kept={len(texts)} vocab={vocab_path} meta={out_dir / 'synth_meta.json'}")


if __name__ == "__main__":
    main()
