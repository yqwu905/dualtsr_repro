from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dualtsr.checkpoint import load_checkpoint, save_checkpoint
from dualtsr.config import load_config
from dualtsr.device import resolve_device
from dualtsr.diffusion import cfm_interpolate, corrupt_text
from dualtsr.ema import make_ema, update_ema
from dualtsr.model import build_model
from dualtsr.tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[1]


class CoreTest(unittest.TestCase):
    def test_tokenizer_roundtrip(self) -> None:
        tok = CharTokenizer(["你", "好", "A"])
        ids = tok.encode("你好A", 6)
        self.assertEqual(tok.decode(ids), "你好A")
        self.assertEqual(ids[-1].item(), tok.pad_id)

    def test_absorbing_mask_schedule(self) -> None:
        tok = CharTokenizer(["A", "B"])
        tokens = torch.tensor([[tok.stoi["A"], tok.stoi["B"], tok.pad_id]])
        no_mask = corrupt_text(tokens, torch.tensor([0.0]), tok.mask_id, tok.pad_id)
        all_mask = corrupt_text(tokens, torch.tensor([1.0]), tok.mask_id, tok.pad_id)
        self.assertTrue(torch.equal(no_mask, tokens))
        self.assertEqual(all_mask[0, 0].item(), tok.mask_id)
        self.assertEqual(all_mask[0, 1].item(), tok.mask_id)
        self.assertEqual(all_mask[0, 2].item(), tok.pad_id)

    def test_cfm_target(self) -> None:
        x0 = torch.zeros(2, 3, 4, 4)
        noise = torch.ones_like(x0)
        xt, target = cfm_interpolate(x0, noise, torch.full((2,), 0.25))
        self.assertTrue(torch.allclose(xt, torch.full_like(xt, 0.25)))
        self.assertTrue(torch.allclose(target, torch.ones_like(target)))

    def test_ema_update(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        tok = CharTokenizer.from_config(cfg)
        model = build_model(cfg, tok.vocab_size, tok.mask_id)
        ema = make_ema(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        before = next(ema.parameters()).clone()
        update_ema(model, ema, decay=0.5)
        after = next(ema.parameters())
        self.assertFalse(torch.equal(before, after))

    def test_model_forward_tiny(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        tok = CharTokenizer.from_config(cfg)
        model = build_model(cfg, tok.vocab_size, tok.mask_id)
        x = torch.randn(2, 3, 32, 64)
        t = torch.rand(2)
        tokens = torch.stack([tok.encode("AB", 8), tok.encode("中文", 8)])
        out = model(x, t, text_tokens=tokens, lr=x)
        self.assertEqual(tuple(out["velocity"].shape), (2, 3, 32, 64))
        self.assertEqual(tuple(out["logits"].shape), (2, 8, tok.vocab_size))

    def test_checkpoint_roundtrip(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        tok = CharTokenizer.from_config(cfg)
        model = build_model(cfg, tok.vocab_size, tok.mask_id)
        ema = make_ema(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(
                path,
                model=model,
                ema_model=ema,
                optimizer=None,
                scheduler=None,
                scaler=None,
                step=3,
                epoch=1,
                config=cfg,
                tokenizer=tok,
            )
            loaded = load_checkpoint(path)
        self.assertEqual(loaded["step"], 3)
        self.assertEqual(loaded["tokenizer"]["mask_token"], "<mask>")
        self.assertIn("model", loaded)

    def test_device_cpu(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_paper_config_defaults(self) -> None:
        cfg = load_config(ROOT / "configs/train/dualtsr_ctr_4x.yaml")
        self.assertEqual(cfg["data"]["hr_size"], [128, 512])
        self.assertEqual(cfg["data"]["scale"], 4)
        self.assertEqual(cfg["data"]["max_text_length"], 24)
        self.assertEqual(cfg["train"]["max_steps"], 700000)
        self.assertEqual(cfg["train"]["global_batch_size"], 32)
        self.assertEqual(cfg["train"]["text_timesteps"], 8)
        self.assertEqual(cfg["train"]["guidance_scale"], 1.0)
        self.assertEqual(cfg["infer"]["steps"], 4)


if __name__ == "__main__":
    unittest.main()

