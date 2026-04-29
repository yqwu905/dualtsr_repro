from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import lmdb
from PIL import Image
from tqdm import tqdm

from dualtsr.config import load_config
from dualtsr.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CTR-TSR manifests/vocabulary from CTR LMDB.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_lmdb(env, key: str) -> bytes | None:
    with env.begin(write=False) as txn:
        return txn.get(key.encode("ascii"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prep_cfg = config.get("prepare", {})
    lmdb_path = prep_cfg.get("source_lmdb") or config.get("data", {}).get("train", {}).get("lmdb_path")
    if not lmdb_path:
        raise ValueError("prepare.source_lmdb or data.train.lmdb_path is required.")
    output_dir = Path(prep_cfg.get("output_dir", "data/ctr_tsr"))
    output_dir.mkdir(parents=True, exist_ok=True)
    export_images = bool(prep_cfg.get("export_images", False))
    image_dir = output_dir / "hr"
    if export_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config.get("data", {})
    hr_size = tuple(int(v) for v in data_cfg.get("hr_size", [128, 512]))
    max_text_length = int(data_cfg.get("max_text_length", 24))
    min_long_side = int(data_cfg.get("min_long_side", 64))
    min_aspect_ratio = float(data_cfg.get("min_aspect_ratio", 2.0))

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    raw_n = read_lmdb(env, "num-samples")
    if raw_n is None:
        raise RuntimeError(f"LMDB missing num-samples: {lmdb_path}")
    n = int(raw_n.decode("utf-8"))
    texts: list[str] = []
    manifest_path = output_dir / "manifest.jsonl"
    index_path = output_dir / "ctr_indices.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf, index_path.open("w", encoding="utf-8") as idxf:
        for idx in tqdm(range(1, n + 1), desc="scan_ctr"):
            raw_label = read_lmdb(env, f"label-{idx:09d}") or read_lmdb(env, f"label-{idx}")
            raw_image = read_lmdb(env, f"image-{idx:09d}") or read_lmdb(env, f"image-{idx}")
            if raw_label is None or raw_image is None:
                continue
            text = raw_label.decode("utf-8", errors="ignore")
            if not text or text == "###" or len(text) > max_text_length:
                continue
            try:
                image = Image.open(io.BytesIO(raw_image)).convert("RGB")
            except Exception:
                continue
            w, h = image.size
            if max(w, h) < min_long_side or h == 0 or (w / h) <= min_aspect_ratio:
                continue
            texts.append(text)
            idxf.write(json.dumps({"lmdb_index": idx, "text": text}, ensure_ascii=False) + "\n")
            if export_images:
                name = f"{idx:09d}.png"
                image.resize((hr_size[1], hr_size[0]), Image.BICUBIC).save(image_dir / name)
                mf.write(json.dumps({"id": str(idx), "hr": f"hr/{name}", "text": text}, ensure_ascii=False) + "\n")

    vocab_path = Path(prep_cfg.get("vocab_path", config.get("tokenizer", {}).get("vocab_path", output_dir / "vocab.txt")))
    CharTokenizer.write_vocab(vocab_path, texts)
    print(f"kept={len(texts)} vocab={vocab_path} index={index_path}")
    if export_images:
        print(f"manifest={manifest_path} images={image_dir}")


if __name__ == "__main__":
    main()

