"""Synthetic text-image rendering for DualTSR pretraining.

渲染流程仿照 SynthText 风格的文本行合成:从语料/字符集采样文本,随机选
字体、颜色、背景与几何扰动,渲染出 HR 文本行图像。LR 仍由训练时的在线
blind degradation 生成(见 data.degrade_tensor),与 CTR-TSR 保持一致。
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}

# 字符集允许的 Unicode 区间:ASCII 可见字符、CJK 标点、全角符号、CJK 统一汉字。
CHARSET_RANGES: tuple[tuple[int, int], ...] = (
    (0x21, 0x7E),
    (0x3001, 0x3011),
    (0xFF01, 0xFF1F),
    (0x4E00, 0x9FFF),
)

DEFAULT_RENDER_CFG: dict[str, Any] = {
    "font_size_min": 64,
    "font_size_max": 112,
    "margin_ratio_min": 0.10,
    "margin_ratio_max": 0.45,
    "spacing_ratio_min": -0.04,
    "spacing_ratio_max": 0.30,
    "baseline_jitter_ratio": 0.04,
    "rotation_max_deg": 3.0,
    "perspective_jitter_ratio": 0.05,
    "perspective_prob": 0.4,
    "min_aspect_ratio": 2.2,
    "max_aspect_ratio": 5.0,
    "contrast_min": 60,
    "stroke_prob": 0.25,
    "stroke_width_max": 3,
    "shadow_prob": 0.25,
    "background_weights": {"solid": 0.35, "gradient": 0.3, "noise": 0.25, "image": 0.1},
    "noise_sigma_min": 4.0,
    "noise_sigma_max": 28.0,
    "final_blur_prob": 0.15,
    "final_blur_max": 0.6,
}

DEFAULT_TEXT_CFG: dict[str, Any] = {
    "corpus_prob": 0.7,
    "mode_weights": {"chinese": 0.55, "mixed": 0.2, "alnum": 0.15, "digits": 0.1},
    "mean_length": 6.0,
}


def list_font_files(font_dir: str | Path) -> list[Path]:
    font_dir = Path(font_dir)
    return sorted(p for p in font_dir.glob("*") if p.suffix.lower() in FONT_SUFFIXES)


@lru_cache(maxsize=None)
def _font_codepoints(font_path: str) -> frozenset[int]:
    """Best cmap of a font, restricted to CHARSET_RANGES."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:  # pragma: no cover - guarded by requirements
        raise RuntimeError("fontTools is required for synthesis: pip install fonttools") from exc

    font = TTFont(font_path, fontNumber=0, lazy=True)
    try:
        cmap = font.getBestCmap() or {}
    finally:
        font.close()
    allowed = set()
    for lo, hi in CHARSET_RANGES:
        allowed.update(cp for cp in cmap if lo <= cp <= hi)
    return frozenset(allowed)


@lru_cache(maxsize=256)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


class FontPool:
    """Scans a directory of fonts and answers glyph-coverage queries."""

    def __init__(self, font_dir: str | Path) -> None:
        self.font_dir = Path(font_dir)
        self.paths = list_font_files(self.font_dir)
        if not self.paths:
            raise RuntimeError(
                f"No fonts found in {self.font_dir}. Run: python3 scripts/download_fonts.py"
            )
        self.coverage: dict[Path, frozenset[int]] = {
            path: _font_codepoints(str(path)) for path in self.paths
        }

    def __len__(self) -> int:
        return len(self.paths)

    def supports(self, path: Path, text: str) -> bool:
        cover = self.coverage[path]
        return all(ord(ch) in cover for ch in text)

    def candidates(self, text: str) -> list[Path]:
        return [path for path in self.paths if self.supports(path, text)]

    def pick(self, rng: random.Random, text: str) -> Path | None:
        candidates = self.candidates(text)
        return rng.choice(candidates) if candidates else None

    def build_charset(self, min_fonts: int = 6) -> str:
        """Characters covered by at least ``min_fonts`` fonts (clamped to pool size).

        Noto/霞鹜文楷这类全集字体覆盖全部 CJK 区间,手写体只覆盖常用字;
        提高 min_fonts 即可把字符集收紧到常用字附近。
        """
        threshold = max(1, min(int(min_fonts), len(self.paths)))
        counts: dict[int, int] = {}
        for cover in self.coverage.values():
            for cp in cover:
                counts[cp] = counts.get(cp, 0) + 1
        return "".join(sorted(chr(cp) for cp, n in counts.items() if n >= threshold))


class TextSampler:
    """Samples text lines from an optional corpus plus random generation modes."""

    def __init__(
        self,
        charset: str,
        max_text_length: int = 24,
        corpus_path: str | Path | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = {**DEFAULT_TEXT_CFG, **(cfg or {})}
        self.max_text_length = int(max_text_length)
        chars = set(charset)
        self.charset = "".join(sorted(chars))
        self.cjk = "".join(ch for ch in self.charset if ord(ch) >= 0x4E00)
        self.ascii_alnum = "".join(ch for ch in self.charset if ch.isascii() and ch.isalnum())
        self.digits = "".join(ch for ch in self.charset if ch.isdigit())
        self.punct = "".join(
            ch for ch in self.charset if not ch.isalnum() and not 0x4E00 <= ord(ch) <= 0x9FFF
        )
        self.corpus: list[str] = []
        if corpus_path:
            self.corpus = self._load_corpus(Path(corpus_path), chars)

    def _load_corpus(self, path: Path, allowed: set[str]) -> list[str]:
        lines: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                text = "".join(ch for ch in raw.strip() if ch in allowed)
                if len(text) >= 1:
                    lines.append(text)
        return lines

    def _length(self, rng: random.Random, lo: int = 1) -> int:
        mean = float(self.cfg.get("mean_length", 6.0))
        length = lo + int(rng.expovariate(1.0 / max(mean - lo, 1.0)))
        return max(lo, min(length, self.max_text_length))

    def _random_text(self, rng: random.Random) -> str:
        weights = self.cfg.get("mode_weights", DEFAULT_TEXT_CFG["mode_weights"])
        modes, probs = zip(*[(k, float(v)) for k, v in weights.items()])
        mode = rng.choices(modes, weights=probs, k=1)[0]
        if mode == "chinese" and self.cjk:
            body = "".join(rng.choice(self.cjk) for _ in range(self._length(rng)))
        elif mode == "alnum" and self.ascii_alnum:
            body = "".join(rng.choice(self.ascii_alnum) for _ in range(self._length(rng)))
        elif mode == "digits" and self.digits:
            body = "".join(rng.choice(self.digits) for _ in range(self._length(rng, lo=2)))
        else:
            pool = (self.cjk or "") + (self.ascii_alnum or "")
            body = "".join(rng.choice(pool) for _ in range(self._length(rng, lo=2)))
        if self.punct and rng.random() < 0.15 and len(body) < self.max_text_length:
            body += rng.choice(self.punct)
        return body or rng.choice(self.charset)

    def sample(self, rng: random.Random) -> str:
        if self.corpus and rng.random() < float(self.cfg.get("corpus_prob", 0.7)):
            line = rng.choice(self.corpus)
            if len(line) > self.max_text_length:
                start = rng.randint(0, len(line) - self.max_text_length)
                length = rng.randint(2, self.max_text_length)
                line = line[start : start + length]
            if line:
                return line
        return self._random_text(rng)


def _random_color(rng: random.Random) -> tuple[int, int, int]:
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def _luminance(color: Iterable[int]) -> float:
    r, g, b = list(color)[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _contrasting_color(rng: random.Random, reference: float, min_delta: float) -> tuple[int, int, int]:
    for _ in range(24):
        color = _random_color(rng)
        if abs(_luminance(color) - reference) >= min_delta:
            return color
    return (12, 12, 12) if reference > 127 else (243, 243, 243)


class SynthTextRenderer:
    """Renders one text line into an HR image."""

    def __init__(
        self,
        font_dir: str | Path,
        hr_size: Iterable[int] | None = (128, 512),
        cfg: dict[str, Any] | None = None,
        bg_image_dir: str | Path | None = None,
    ) -> None:
        self.fonts = FontPool(font_dir)
        self.hr_size = tuple(int(v) for v in hr_size) if hr_size is not None else None
        self.cfg = {**DEFAULT_RENDER_CFG, **(cfg or {})}
        self.bg_images: list[Path] = []
        if bg_image_dir:
            bg_dir = Path(bg_image_dir)
            self.bg_images = sorted(
                p
                for p in bg_dir.glob("**/*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )

    # --- backgrounds --------------------------------------------------------
    # 背景先采样为"规格"(颜色参数),再物化为任意尺寸图像,这样文字颜色
    # 可以在绘制前根据真实背景亮度选取,保证对比度约束成立。
    def _background_spec(self, rng: random.Random) -> dict[str, Any]:
        weights = dict(self.cfg["background_weights"])
        if not self.bg_images:
            weights.pop("image", None)
        kinds, probs = zip(*[(k, float(v)) for k, v in weights.items()])
        kind = rng.choices(kinds, weights=probs, k=1)[0]
        if kind == "image":
            crop = self._photo_crop(rng)
            if crop is None:
                kind = "solid"
            else:
                return {"kind": "image", "crop": crop}
        spec: dict[str, Any] = {"kind": kind, "c0": _random_color(rng)}
        if kind == "gradient":
            spec["c1"] = _random_color(rng)
            spec["horizontal"] = rng.random() < 0.7
        elif kind == "noise":
            spec["sigma"] = rng.uniform(float(self.cfg["noise_sigma_min"]), float(self.cfg["noise_sigma_max"]))
            spec["noise_seed"] = rng.getrandbits(32)
        return spec

    def _spec_luminance(self, spec: dict[str, Any]) -> float:
        if spec["kind"] == "image":
            thumb = spec["crop"].resize((16, 8), Image.BILINEAR)
            return _luminance(np.asarray(thumb, dtype=np.float32).reshape(-1, 3).mean(axis=0))
        if spec["kind"] == "gradient":
            return (_luminance(spec["c0"]) + _luminance(spec["c1"])) / 2.0
        return _luminance(spec["c0"])

    def _materialize_background(self, spec: dict[str, Any], w: int, h: int) -> Image.Image:
        kind = spec["kind"]
        if kind == "image":
            return spec["crop"].resize((w, h), Image.BICUBIC)
        if kind == "gradient":
            c0 = np.array(spec["c0"], dtype=np.float32)
            c1 = np.array(spec["c1"], dtype=np.float32)
            t = np.linspace(0.0, 1.0, w if spec["horizontal"] else h, dtype=np.float32)
            ramp = c0[None, :] + t[:, None] * (c1 - c0)[None, :]
            if spec["horizontal"]:
                arr = np.broadcast_to(ramp[None, :, :], (h, w, 3))
            else:
                arr = np.broadcast_to(ramp[:, None, :], (h, w, 3))
            return Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)))
        image = Image.new("RGB", (w, h), spec["c0"])
        if kind == "noise":
            noise = np.random.default_rng(spec["noise_seed"]).normal(0.0, spec["sigma"], (h, w, 3))
            arr = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255)
            image = Image.fromarray(arr.astype(np.uint8))
        return image

    def _photo_crop(self, rng: random.Random) -> Image.Image | None:
        path = rng.choice(self.bg_images)
        try:
            photo = Image.open(path).convert("RGB")
        except Exception:
            return None
        crop_w = rng.randint(max(1, photo.width // 4), photo.width)
        crop_h = rng.randint(max(1, photo.height // 4), photo.height)
        x0 = rng.randint(0, photo.width - crop_w)
        y0 = rng.randint(0, photo.height - crop_h)
        return photo.crop((x0, y0, x0 + crop_w, y0 + crop_h))

    # --- text layer ---------------------------------------------------------
    def _text_layer(
        self, rng: random.Random, text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]
    ) -> Image.Image:
        cfg = self.cfg
        size = int(font.size)
        ascent, descent = font.getmetrics()
        stroke_width = 0
        stroke_fill = None
        if rng.random() < float(cfg["stroke_prob"]):
            stroke_width = rng.randint(1, int(cfg["stroke_width_max"]))
            stroke_fill = _contrasting_color(rng, _luminance(fill), 50.0)
        spacing = rng.uniform(float(cfg["spacing_ratio_min"]), float(cfg["spacing_ratio_max"])) * size
        jitter = float(cfg["baseline_jitter_ratio"]) * size
        margin = math.ceil(size * rng.uniform(float(cfg["margin_ratio_min"]), float(cfg["margin_ratio_max"])))
        pad = margin + stroke_width + math.ceil(jitter) + math.ceil(size * 0.08)

        advances = [max(1.0, font.getlength(ch)) for ch in text]
        total_w = sum(advances) + spacing * max(0, len(text) - 1)
        layer_w = math.ceil(total_w) + 2 * pad
        layer_h = ascent + descent + 2 * pad
        layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        shadow = rng.random() < float(cfg["shadow_prob"])
        shadow_offset = (rng.randint(1, max(1, size // 24)), rng.randint(1, max(1, size // 24)))
        x = float(pad)
        baseline = pad + ascent
        for ch, advance in zip(text, advances):
            y = baseline + rng.uniform(-jitter, jitter)
            if shadow:
                draw.text(
                    (x + shadow_offset[0], y + shadow_offset[1]),
                    ch,
                    font=font,
                    fill=(0, 0, 0, 140),
                    anchor="ls",
                    stroke_width=stroke_width,
                    stroke_fill=(0, 0, 0, 140) if stroke_fill else None,
                )
            draw.text(
                (x, y),
                ch,
                font=font,
                fill=(*fill, 255),
                anchor="ls",
                stroke_width=stroke_width,
                stroke_fill=(*stroke_fill, 255) if stroke_fill else None,
            )
            x += advance + spacing
        return layer

    def _warp(self, rng: random.Random, layer: Image.Image) -> Image.Image:
        cfg = self.cfg
        angle = rng.uniform(-float(cfg["rotation_max_deg"]), float(cfg["rotation_max_deg"]))
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
        if rng.random() < float(cfg["perspective_prob"]):
            w, h = layer.size
            jx, jy = w * float(cfg["perspective_jitter_ratio"]), h * float(cfg["perspective_jitter_ratio"])
            quad = (
                rng.uniform(0, jx), rng.uniform(0, jy),
                rng.uniform(0, jx), h - rng.uniform(0, jy),
                w - rng.uniform(0, jx), h - rng.uniform(0, jy),
                w - rng.uniform(0, jx), rng.uniform(0, jy),
            )
            layer = layer.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)
        return layer

    # --- main entry ---------------------------------------------------------
    def render(self, text: str, rng: random.Random) -> Image.Image | None:
        """Render one HR text-line image; None if no font covers ``text``."""
        font_path = self.fonts.pick(rng, text)
        if font_path is None:
            return None
        cfg = self.cfg
        size = rng.randint(int(cfg["font_size_min"]), int(cfg["font_size_max"]))
        font = _load_font(str(font_path), size)

        spec = self._background_spec(rng)
        fill = _contrasting_color(rng, self._spec_luminance(spec), float(cfg["contrast_min"]))

        layer = self._text_layer(rng, text, font, fill)
        layer = self._warp(rng, layer)

        canvas_h = layer.height
        min_w = math.ceil(canvas_h * rng.uniform(float(cfg["min_aspect_ratio"]), float(cfg["max_aspect_ratio"])))
        canvas_w = max(layer.width, min_w)
        background = self._materialize_background(spec, canvas_w, canvas_h)
        x = rng.randint(0, canvas_w - layer.width)
        background.paste(layer, (x, 0), layer)

        if rng.random() < float(cfg["final_blur_prob"]):
            background = background.filter(
                ImageFilter.GaussianBlur(radius=rng.uniform(0.1, float(cfg["final_blur_max"])))
            )
        if self.hr_size is not None:
            background = background.resize((self.hr_size[1], self.hr_size[0]), Image.BICUBIC)
        return background


class SynthRenderDataset(Dataset):
    """Online font-rendered pretraining dataset; LR 由在线退化生成."""

    def __init__(
        self,
        length: int,
        hr_size: Iterable[int],
        scale: int,
        font_dir: str | Path,
        max_text_length: int = 24,
        charset_min_fonts: int = 6,
        corpus_path: str | Path | None = None,
        text_cfg: dict[str, Any] | None = None,
        render_cfg: dict[str, Any] | None = None,
        bg_image_dir: str | Path | None = None,
        degradation_cfg: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        from .data import degrade_tensor, pil_to_tensor  # 延迟导入避免循环依赖

        self._degrade = degrade_tensor
        self._to_tensor = pil_to_tensor
        self.length = int(length)
        self.hr_size = tuple(int(v) for v in hr_size)
        self.scale = int(scale)
        self.max_text_length = int(max_text_length)
        self.degradation_cfg = degradation_cfg or {}
        self.seed = int(seed)
        self.renderer = SynthTextRenderer(
            font_dir, hr_size=self.hr_size, cfg=render_cfg, bg_image_dir=bg_image_dir
        )
        charset = self.renderer.fonts.build_charset(charset_min_fonts)
        self.sampler = TextSampler(
            charset, max_text_length=self.max_text_length, corpus_path=corpus_path, cfg=text_cfg
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random((self.seed << 32) ^ idx)
        image: Image.Image | None = None
        text = ""
        for _ in range(8):
            text = self.sampler.sample(rng)
            image = self.renderer.render(text, rng)
            if image is not None:
                break
        if image is None:  # 字符集来自字体覆盖,正常不会发生
            raise RuntimeError(f"No font covers sampled text: {text!r}")
        hr = self._to_tensor(image)
        lr = self._degrade(hr, self.scale, self.degradation_cfg)
        return {"hr": hr, "lr": lr, "text": text, "id": str(idx)}
