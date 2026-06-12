from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from dualtsr.checkpoint import load_checkpoint, save_checkpoint
from dualtsr.config import load_config
from dualtsr.device import resolve_device
from dualtsr.diffusion import cfm_interpolate, corrupt_text
from dualtsr.ema import make_ema, update_ema
from dualtsr.model import build_mmdit, build_model
from dualtsr.tokenizer import CharTokenizer, WordTokenizer, build_tokenizer, tokenizer_from_state
from dualtsr.vae import build_vae, update_model_latent_shape


ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def prepare_latent_shape(cfg: dict) -> nn.Module:
    """Build the VAE and infer latent shape into cfg, mirroring train/infer startup."""
    vae = build_vae(cfg, CPU)
    update_model_latent_shape(cfg, vae, CPU)
    return vae


class DummyTextEncoder(nn.Module):
    output_dim = 16

    def forward(
        self,
        text_tokens: torch.Tensor | None,
        batch_size: int,
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if text_tokens is None:
            return torch.zeros(batch_size, max_length, self.output_dim, device=device)
        values = text_tokens[:, :max_length].float().to(device)
        if values.shape[1] < max_length:
            pad = torch.zeros(batch_size, max_length - values.shape[1], device=device)
            values = torch.cat([values, pad], dim=1)
        return values.unsqueeze(-1).expand(-1, -1, self.output_dim) / 10.0


class DummyMMDiT(nn.Module):
    """Minimal MMDiT under the new contract: latent in, (velocity, text) out."""

    def __init__(self, **kwargs) -> None:  # accepts injected hidden_dim/latent_channels/latent_size
        super().__init__()

    def forward(
        self,
        x_img: torch.Tensor,
        timesteps: torch.Tensor,
        text: torch.Tensor,
        lr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del timesteps, lr
        return x_img, text


class DummyVAE(nn.Module):
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :1]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.expand(-1, 3, -1, -1)


class CoreTest(unittest.TestCase):
    def test_tokenizer_roundtrip(self) -> None:
        tok = CharTokenizer(["你", "好", "A"])
        ids = tok.encode("你好A", 6)
        self.assertEqual(tok.decode(ids), "你好A")
        self.assertEqual(ids[-1].item(), tok.pad_id)

    def test_word_tokenizer_roundtrip(self) -> None:
        tok = WordTokenizer(["hello", "world", "foo"])
        ids = tok.encode("hello world foo", 6)
        self.assertEqual(tok.decode(ids), "hello world foo")
        self.assertEqual(ids[-1].item(), tok.pad_id)
        # Out-of-vocabulary words fall back to the unk token.
        self.assertEqual(tok.decode(tok.encode("hello bar", 4)), "hello <unk>")

    def test_build_tokenizer_dispatch(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        self.assertIsInstance(build_tokenizer(cfg), CharTokenizer)
        word_cfg = {"tokenizer": {"class_path": "dualtsr.tokenizer:WordTokenizer", "words": ["a", "b"]}}
        word_tok = build_tokenizer(word_cfg)
        self.assertIsInstance(word_tok, WordTokenizer)
        # state_dict carries class_path so reconstruction picks the right class.
        self.assertIsInstance(tokenizer_from_state(word_tok.state_dict()), WordTokenizer)

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
        tok = build_tokenizer(cfg)
        prepare_latent_shape(cfg)
        model = build_model(cfg, tok.vocab_size, tok.mask_id)
        ema = make_ema(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        before = next(ema.parameters()).clone()
        update_ema(model, ema, decay=0.5)
        after = next(ema.parameters())
        self.assertFalse(torch.equal(before, after))

    def test_vae_dry_run_latent_shape(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        vae = build_vae(cfg, CPU)
        latent_channels, latent_size = update_model_latent_shape(cfg, vae, CPU)
        self.assertEqual(latent_channels, 3)
        self.assertEqual(latent_size, [32, 64])
        self.assertEqual(cfg["model"]["latent_channels"], 3)
        self.assertEqual(cfg["model"]["latent_size"], [32, 64])

    def test_build_mmdit_class_path(self) -> None:
        model_cfg = {
            "mmdit": {
                "class_path": "dualtsr.model:NativeMMDiTBackbone",
                "init_args": {"patch_size": [8, 8], "num_heads": 4, "depth": 1, "mlp_ratio": 2.0},
            }
        }
        mmdit = build_mmdit(model_cfg, hidden_dim=32, latent_channels=3, latent_size=[32, 64])
        x = torch.randn(2, 3, 32, 64)
        t = torch.rand(2)
        text = torch.randn(2, 8, 32)
        velocity, text_out = mmdit(x, t, text, lr=x)
        self.assertEqual(tuple(velocity.shape), (2, 3, 32, 64))
        self.assertEqual(tuple(text_out.shape), (2, 8, 32))

    def test_model_forward_tiny(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        tok = build_tokenizer(cfg)
        prepare_latent_shape(cfg)
        model = build_model(cfg, tok.vocab_size, tok.mask_id)
        x = torch.randn(2, 3, 32, 64)
        t = torch.rand(2)
        tokens = torch.stack([tok.encode("AB", 8), tok.encode("中文", 8)])
        out = model(x, t, text_tokens=tokens, lr=x)
        self.assertEqual(tuple(out["velocity"].shape), (2, 3, 32, 64))
        self.assertEqual(tuple(out["logits"].shape), (2, 8, tok.vocab_size))

    def test_custom_model_components(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        cfg["model"]["text_encoder"] = {
            "type": "custom",
            "class_path": f"{DummyTextEncoder.__module__}:DummyTextEncoder",
            "output_dim": 16,
        }
        cfg["model"]["mmdit"] = {
            "class_path": f"{DummyMMDiT.__module__}:DummyMMDiT",
            "init_args": {},
        }
        tok = build_tokenizer(cfg)
        prepare_latent_shape(cfg)
        model = build_model(cfg, tok.vocab_size, tok.mask_id)
        x = torch.randn(2, 3, 32, 64)
        t = torch.rand(2)
        tokens = torch.stack([tok.encode("AB", 8), tok.encode("中文", 8)])
        out = model(x, t, text_tokens=tokens, lr=x)
        self.assertEqual(tuple(out["velocity"].shape), (2, 3, 32, 64))
        self.assertEqual(tuple(out["logits"].shape), (2, 8, tok.vocab_size))

    def test_custom_vae_component(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        cfg["vae"] = {
            "class_path": f"{DummyVAE.__module__}:DummyVAE",
            "init_args": {},
        }
        vae = build_vae(cfg, CPU)
        image = torch.rand(2, 3, 32, 64)
        latent = vae.encode(image)
        decoded = vae.decode(latent)
        self.assertEqual(tuple(latent.shape), (2, 1, 32, 64))
        self.assertEqual(tuple(decoded.shape), (2, 3, 32, 64))

    def test_checkpoint_roundtrip(self) -> None:
        cfg = load_config(ROOT / "configs/train/smoke.yaml")
        tok = build_tokenizer(cfg)
        prepare_latent_shape(cfg)
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
