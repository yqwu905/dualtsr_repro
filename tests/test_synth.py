from __future__ import annotations

import random
import unittest
from pathlib import Path

from dualtsr.synth import FontPool, SynthRenderDataset, SynthTextRenderer, TextSampler, list_font_files

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
HAS_FONTS = bool(list_font_files(FONT_DIR)) if FONT_DIR.exists() else False
SKIP_REASON = "fonts missing; run: python3 scripts/download_fonts.py"


@unittest.skipUnless(HAS_FONTS, SKIP_REASON)
class SynthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = FontPool(FONT_DIR)
        cls.charset = cls.pool.build_charset(6)

    def test_charset_has_ascii_and_cjk(self) -> None:
        self.assertIn("A", self.charset)
        self.assertIn("9", self.charset)
        self.assertGreater(sum(1 for c in self.charset if ord(c) >= 0x4E00), 3000)

    def test_charset_threshold_monotonic(self) -> None:
        self.assertGreaterEqual(len(self.pool.build_charset(1)), len(self.pool.build_charset(6)))

    def test_font_pick_covers_text(self) -> None:
        rng = random.Random(0)
        path = self.pool.pick(rng, "中文Mix123")
        assert path is not None
        self.assertTrue(self.pool.supports(path, "中文Mix123"))

    def test_sampler_within_charset_and_length(self) -> None:
        sampler = TextSampler(self.charset, max_text_length=10)
        rng = random.Random(1)
        allowed = set(self.charset)
        for _ in range(50):
            text = sampler.sample(rng)
            self.assertTrue(0 < len(text) <= 10, text)
            self.assertTrue(all(ch in allowed for ch in text), text)

    def test_render_shape_and_determinism(self) -> None:
        renderer = SynthTextRenderer(FONT_DIR, hr_size=(64, 256))
        first = renderer.render("测试Aa1", random.Random(42))
        second = renderer.render("测试Aa1", random.Random(42))
        assert first is not None and second is not None
        self.assertEqual(first.size, (256, 64))
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_render_natural_size_keeps_aspect(self) -> None:
        renderer = SynthTextRenderer(FONT_DIR, hr_size=None)
        image = renderer.render("你好", random.Random(3))
        assert image is not None
        w, h = image.size
        self.assertGreaterEqual(w / h, 2.0)
        self.assertGreaterEqual(max(w, h), 64)

    def test_dataset_getitem(self) -> None:
        dataset = SynthRenderDataset(
            length=4,
            hr_size=(32, 128),
            scale=2,
            font_dir=FONT_DIR,
            max_text_length=8,
            seed=7,
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["hr"].shape), (3, 32, 128))
        self.assertEqual(tuple(sample["lr"].shape), (3, 32, 128))
        self.assertTrue(0 < len(sample["text"]) <= 8)
        again = dataset[0]
        self.assertEqual(sample["text"], again["text"])


if __name__ == "__main__":
    unittest.main()
