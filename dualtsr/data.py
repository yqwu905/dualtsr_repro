from __future__ import annotations

import csv
import io
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset

from .tokenizer import BaseTokenizer


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def load_rgb(path: str | Path, size: Iterable[int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None:
        h, w = [int(v) for v in size]
        image = image.resize((w, h), Image.BICUBIC)
    return image


def resize_tensor(image: torch.Tensor, size: Iterable[int]) -> torch.Tensor:
    h, w = [int(v) for v in size]
    image4 = image.unsqueeze(0)
    out = F.interpolate(image4, size=(h, w), mode="bicubic", align_corners=False)
    return out.squeeze(0).clamp(0, 1)


def resolve_degradation_strategy(
    cfg: dict[str, Any],
    draw: float | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Select and merge one weighted blind-degradation strategy."""

    strategies = cfg.get("strategies")
    if not strategies:
        return cfg
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("degradation.strategies must be a non-empty list")
    weights = [float(strategy.get("probability", strategy.get("weight", 1.0))) for strategy in strategies]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("degradation strategy probabilities must be non-negative with a positive sum")
    value = (rng or random).random() if draw is None else float(draw)
    threshold = value * sum(weights)
    cumulative = 0.0
    selected = strategies[-1]
    for strategy, weight in zip(strategies, weights):
        cumulative += weight
        if threshold < cumulative:
            selected = strategy
            break
    base = {key: value for key, value in cfg.items() if key != "strategies"}
    return {**base, **{key: value for key, value in selected.items() if key not in {"probability", "weight", "name"}}}


def degrade_tensor(
    hr: torch.Tensor,
    scale: int,
    cfg: dict[str, Any] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    rng = random.Random(seed) if seed is not None else random
    np_rng = np.random.default_rng(seed) if seed is not None else np.random
    cfg = resolve_degradation_strategy(cfg or {}, rng=rng if isinstance(rng, random.Random) else None)
    h, w = hr.shape[-2:]
    lr_h = max(1, h // int(scale))
    lr_w = max(1, w // int(scale))
    image = tensor_to_pil(hr)

    if cfg.get("random_order", True):
        steps = ["blur", "down_up", "noise", "jpeg"]
        rng.shuffle(steps)
    else:
        steps = ["blur", "down_up", "noise", "jpeg"]

    for step in steps:
        if step == "blur" and rng.random() < float(cfg.get("blur_prob", 0.8)):
            radius = rng.uniform(float(cfg.get("blur_min", 0.1)), float(cfg.get("blur_max", 2.0)))
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif step == "down_up":
            down_modes = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS]
            image = image.resize((lr_w, lr_h), rng.choice(down_modes))
            image = image.resize((w, h), rng.choice(down_modes))
        elif step == "noise" and rng.random() < float(cfg.get("noise_prob", 0.6)):
            arr = np.asarray(image).astype(np.float32)
            sigma = rng.uniform(float(cfg.get("noise_min", 0.0)), float(cfg.get("noise_max", 8.0)))
            arr = np.clip(arr + np_rng.normal(0.0, sigma, arr.shape), 0, 255)
            image = Image.fromarray(arr.astype(np.uint8))
        elif step == "jpeg" and rng.random() < float(cfg.get("jpeg_prob", 0.7)):
            quality = rng.randint(int(cfg.get("jpeg_min", 35)), int(cfg.get("jpeg_max", 95)))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

    return pil_to_tensor(image)


class CTRLMDBDataset(Dataset):
    """Reads the official FudanVI CTR LMDB format used by TransOCR baselines."""

    def __init__(
        self,
        lmdb_path: str | Path,
        split: str,
        hr_size: Iterable[int],
        scale: int,
        degradation_cfg: dict[str, Any] | None = None,
        max_text_length: int = 24,
        min_long_side: int = 64,
        min_aspect_ratio: float = 2.0,
        scan_images: bool = True,
        online_degradation: bool = True,
        deterministic_degradation: bool = False,
        degradation_seed: int = 1234,
    ) -> None:
        self.lmdb_path = Path(lmdb_path)
        self.split = split
        self.hr_size = tuple(int(v) for v in hr_size)
        self.scale = int(scale)
        self.degradation_cfg = degradation_cfg or {}
        self.max_text_length = int(max_text_length)
        self.min_long_side = int(min_long_side)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.scan_images = bool(scan_images)
        self.online_degradation = bool(online_degradation)
        self.deterministic_degradation = bool(deterministic_degradation)
        self.degradation_seed = int(degradation_seed)
        self._env: lmdb.Environment | None = None
        self.indices = self._build_index()

    def _open(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=2048,
            )
        return self._env

    def _read_raw(self, key: str) -> bytes | None:
        with self._open().begin(write=False) as txn:
            return txn.get(key.encode("ascii"))

    def _num_samples(self) -> int:
        raw = self._read_raw("num-samples")
        if raw is None:
            raise RuntimeError(f"LMDB missing num-samples key: {self.lmdb_path}")
        return int(raw.decode("utf-8"))

    def _label(self, idx: int) -> str:
        raw = self._read_raw(f"label-{idx:09d}")
        if raw is None:
            raw = self._read_raw(f"label-{idx}")
        if raw is None:
            return ""
        return raw.decode("utf-8", errors="ignore")

    def _image(self, idx: int) -> Image.Image:
        raw = self._read_raw(f"image-{idx:09d}")
        if raw is None:
            raw = self._read_raw(f"image-{idx}")
        if raw is None:
            raise KeyError(f"LMDB image key not found for sample {idx}")
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _build_index(self) -> list[int]:
        indices: list[int] = []
        for idx in range(1, self._num_samples() + 1):
            text = self._label(idx)
            if not text or text == "###" or len(text) > self.max_text_length:
                continue
            if self.scan_images:
                try:
                    image = self._image(idx)
                except Exception:
                    continue
                w, h = image.size
                if max(w, h) < self.min_long_side:
                    continue
                if h == 0 or (w / h) <= self.min_aspect_ratio:
                    continue
            indices.append(idx)
        if not indices:
            raise RuntimeError(f"No usable samples found in CTR LMDB: {self.lmdb_path}")
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = self.indices[item]
        text = self._label(idx)
        hr_img = self._image(idx).resize((self.hr_size[1], self.hr_size[0]), Image.BICUBIC)
        hr = pil_to_tensor(hr_img)
        seed = self.degradation_seed + idx if self.deterministic_degradation else None
        lr = degrade_tensor(hr, self.scale, self.degradation_cfg, seed=seed) if self.online_degradation else hr.clone()
        return {"hr": hr, "lr": lr, "text": text, "id": str(idx)}


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path,
        hr_size: Iterable[int],
        scale: int,
        degradation_cfg: dict[str, Any] | None = None,
        online_degradation: bool = True,
        deterministic_degradation: bool = False,
        degradation_seed: int = 1234,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root)
        self.hr_size = tuple(int(v) for v in hr_size)
        self.scale = int(scale)
        self.degradation_cfg = degradation_cfg or {}
        self.online_degradation = bool(online_degradation)
        self.deterministic_degradation = bool(deterministic_degradation)
        self.degradation_seed = int(degradation_seed)
        self.rows = self._read_rows()
        if not self.rows:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")

    def _read_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        suffix = self.manifest_path.suffix.lower()
        if suffix == ".jsonl":
            with self.manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
                dialect = csv.excel_tab if suffix in {".tsv", ".txt"} else csv.excel
                rows.extend(csv.DictReader(f, dialect=dialect))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path: str | None) -> Path | None:
        if not path:
            return None
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        hr_path = self._resolve(row.get("hr") or row.get("gt") or row.get("image"))
        if hr_path is None:
            raise KeyError("Manifest row must contain hr, gt, or image path.")
        hr = pil_to_tensor(load_rgb(hr_path, self.hr_size))
        lr_path = self._resolve(row.get("lr") or row.get("lq"))
        if lr_path is not None:
            lr = pil_to_tensor(load_rgb(lr_path, self.hr_size))
        elif self.online_degradation:
            seed = self.degradation_seed + idx if self.deterministic_degradation else None
            lr = degrade_tensor(hr, self.scale, self.degradation_cfg, seed=seed)
        else:
            lr = hr.clone()
        return {"hr": hr, "lr": lr, "text": str(row.get("text", "")), "id": str(row.get("id", idx))}


class SyntheticTextDataset(Dataset):
    def __init__(
        self,
        length: int,
        hr_size: Iterable[int],
        scale: int,
        alphabet: str,
        max_text_length: int,
        degradation_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.length = int(length)
        self.hr_size = tuple(int(v) for v in hr_size)
        self.scale = int(scale)
        self.alphabet = alphabet or "ABCDE12345中文复现"
        self.max_text_length = int(max_text_length)
        self.degradation_cfg = degradation_cfg or {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(idx)
        h, w = self.hr_size
        image = Image.new("RGB", (w, h), (rng.randint(210, 255), rng.randint(210, 255), rng.randint(210, 255)))
        draw = ImageDraw.Draw(image)
        text_len = rng.randint(2, min(self.max_text_length, 8))
        text = "".join(rng.choice(self.alphabet) for _ in range(text_len))
        for _ in range(5):
            x0 = rng.randint(0, max(1, w - 2))
            y0 = rng.randint(0, max(1, h - 2))
            x1 = rng.randint(x0 + 1, w)
            y1 = rng.randint(y0 + 1, h)
            color = tuple(rng.randint(0, 255) for _ in range(3))
            draw.rectangle((x0, y0, x1, y1), outline=color)
        draw.text((max(2, w // 20), max(2, h // 3)), text, fill=(20, 20, 20))
        hr = pil_to_tensor(image)
        lr = degrade_tensor(hr, self.scale, self.degradation_cfg)
        return {"hr": hr, "lr": lr, "text": text, "id": str(idx)}


def build_dataset(config: dict, split: str) -> Dataset:
    data_cfg = config["data"]
    split_cfg = data_cfg.get(split, {})
    dataset_type = split_cfg.get("type", data_cfg.get("type", "manifest"))
    hr_size = data_cfg.get("hr_size", [128, 512])
    scale = int(data_cfg.get("scale", 4))
    degradation_cfg = data_cfg.get("degradation", {})
    if dataset_type == "ctr_lmdb":
        return CTRLMDBDataset(
            lmdb_path=split_cfg.get("lmdb_path", data_cfg.get("lmdb_path")),
            split=split,
            hr_size=hr_size,
            scale=scale,
            degradation_cfg=degradation_cfg,
            max_text_length=int(data_cfg.get("max_text_length", 24)),
            min_long_side=int(data_cfg.get("min_long_side", 64)),
            min_aspect_ratio=float(data_cfg.get("min_aspect_ratio", 2.0)),
            scan_images=bool(split_cfg.get("scan_images", data_cfg.get("scan_images", True))),
            online_degradation=bool(split_cfg.get("online_degradation", split == "train")),
            deterministic_degradation=bool(split_cfg.get("deterministic_degradation", split != "train")),
            degradation_seed=int(split_cfg.get("degradation_seed", data_cfg.get("degradation_seed", 1234))),
        )
    if dataset_type == "manifest":
        return ManifestDataset(
            manifest_path=split_cfg.get("manifest_path", data_cfg.get("manifest_path")),
            root=split_cfg.get("root", data_cfg.get("root", ".")),
            hr_size=hr_size,
            scale=scale,
            degradation_cfg=degradation_cfg,
            online_degradation=bool(split_cfg.get("online_degradation", split == "train")),
            deterministic_degradation=bool(split_cfg.get("deterministic_degradation", split != "train")),
            degradation_seed=int(split_cfg.get("degradation_seed", data_cfg.get("degradation_seed", 1234))),
        )
    if dataset_type == "synth_render":
        from .synth import SynthRenderDataset

        synth_cfg = {**data_cfg.get("synth", {}), **split_cfg.get("synth", {})}
        return SynthRenderDataset(
            length=int(split_cfg.get("length", data_cfg.get("length", 1000))),
            hr_size=hr_size,
            scale=scale,
            font_dir=synth_cfg.get("font_dir", "assets/fonts"),
            max_text_length=int(data_cfg.get("max_text_length", 24)),
            charset_min_fonts=int(synth_cfg.get("charset_min_fonts", 6)),
            corpus_path=synth_cfg.get("corpus_path"),
            text_cfg=synth_cfg.get("text"),
            render_cfg=synth_cfg.get("render"),
            bg_image_dir=synth_cfg.get("bg_image_dir"),
            degradation_cfg=degradation_cfg,
            seed=int(synth_cfg.get("seed", 0)) + (1 if split != "train" else 0),
        )
    if dataset_type == "synthetic":
        alphabet = "".join(config.get("tokenizer", {}).get("alphabet", list("ABCDE12345中文复现")))
        return SyntheticTextDataset(
            length=int(split_cfg.get("length", data_cfg.get("length", 64))),
            hr_size=hr_size,
            scale=scale,
            alphabet=alphabet,
            max_text_length=int(data_cfg.get("max_text_length", 24)),
            degradation_cfg=degradation_cfg,
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def make_collate_fn(tokenizer: BaseTokenizer, max_text_length: int):
    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        hr = torch.stack([sample["hr"] for sample in samples], dim=0)
        lr = torch.stack([sample["lr"] for sample in samples], dim=0)
        texts = [sample["text"] for sample in samples]
        tokens = torch.stack([tokenizer.encode(text, max_text_length) for text in texts], dim=0)
        return {
            "hr": hr,
            "lr": lr,
            "text": texts,
            "tokens": tokens,
            "id": [sample.get("id", str(i)) for i, sample in enumerate(samples)],
        }

    return collate


def build_dataloader(config: dict, split: str, tokenizer: BaseTokenizer, sampler=None) -> DataLoader:
    dataset = build_dataset(config, split)
    data_cfg = config["data"]
    loader_cfg = config.get("loader", {})
    batch_size = int(loader_cfg.get(f"{split}_batch_size", loader_cfg.get("batch_size", 1)))
    max_text_length = int(data_cfg.get("text_sequence_length", data_cfg.get("max_text_length", 24)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == "train"),
        sampler=sampler,
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        drop_last=bool(loader_cfg.get("drop_last", split == "train")),
        collate_fn=make_collate_fn(tokenizer, max_text_length),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and int(loader_cfg.get("num_workers", 0)) > 0,
    )


def make_image_grid(images: list[torch.Tensor]) -> torch.Tensor:
    if not images:
        raise ValueError("images must be non-empty")
    images = [img.detach().cpu().clamp(0, 1) for img in images]
    c, h, w = images[0].shape
    canvas = torch.ones(c, h, w * len(images))
    for i, img in enumerate(images):
        canvas[:, :, i * w : (i + 1) * w] = img
    return canvas


def render_text_panel(lines: list[str], width: int = 512, line_height: int = 22) -> torch.Tensor:
    height = max(line_height, line_height * len(lines))
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((4, i * line_height + 3), line, fill=(0, 0, 0))
    return pil_to_tensor(image)
