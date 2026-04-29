from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class VAEInfo:
    latent_channels: int
    latent_size: tuple[int, int]
    scale_factor: float


class IdentityVAE(nn.Module):
    def __init__(self, image_size: Iterable[int], channels: int = 3) -> None:
        super().__init__()
        self.info = VAEInfo(latent_channels=int(channels), latent_size=tuple(int(v) for v in image_size), scale_factor=1.0)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.clamp(0, 1)


class AutoencoderKLAdapter(nn.Module):
    def __init__(
        self,
        pretrained_path: str | Path,
        latent_size: Iterable[int],
        scaling_factor: float = 0.18215,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise RuntimeError("diffusers is required for vae.type=autoencoder_kl") from exc
        self.vae = AutoencoderKL.from_pretrained(str(pretrained_path))
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(dtype=dtype)
        latent_channels = int(getattr(self.vae.config, "latent_channels", 4))
        self.info = VAEInfo(
            latent_channels=latent_channels,
            latent_size=tuple(int(v) for v in latent_size),
            scale_factor=float(scaling_factor),
        )

    def _param_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(dtype=self._param_dtype()).mul(2.0).sub(1.0)
        dist = self.vae.encode(image).latent_dist
        return dist.sample() * self.info.scale_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(dtype=self._param_dtype()) / self.info.scale_factor
        image = self.vae.decode(latent).sample
        return image.add(1.0).mul(0.5).clamp(0, 1).float()


def build_vae(config: dict, device: torch.device, dtype: torch.dtype) -> nn.Module:
    data_cfg = config.get("data", {})
    vae_cfg = config.get("vae", {})
    vae_type = str(vae_cfg.get("type", "identity")).lower()
    if vae_type == "identity":
        vae = IdentityVAE(data_cfg.get("hr_size", [128, 512]), channels=int(vae_cfg.get("channels", 3)))
    elif vae_type == "autoencoder_kl":
        pretrained_path = vae_cfg.get("pretrained_path")
        if not pretrained_path:
            raise ValueError("vae.pretrained_path is required for vae.type=autoencoder_kl")
        vae = AutoencoderKLAdapter(
            pretrained_path=pretrained_path,
            latent_size=vae_cfg.get("latent_size", config.get("model", {}).get("latent_size", [16, 64])),
            scaling_factor=float(vae_cfg.get("scaling_factor", 0.18215)),
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported vae.type: {vae_type}")
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae

