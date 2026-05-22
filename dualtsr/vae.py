from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from dualtsr.registry import load_class


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


class CustomVAEAdapter(nn.Module):
    def __init__(
        self,
        class_path: str,
        latent_size: Iterable[int],
        latent_channels: int | None = None,
        scaling_factor: float = 1.0,
        kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        cls = load_class(class_path)
        self.vae = cls(**(kwargs or {}))
        if not isinstance(self.vae, nn.Module):
            raise TypeError("Custom VAE must inherit torch.nn.Module.")
        backend_info = getattr(self.vae, "info", None)
        info_channels = getattr(backend_info, "latent_channels", None)
        info_size = getattr(backend_info, "latent_size", None)
        info_scale = getattr(backend_info, "scale_factor", None)
        channels = latent_channels if latent_channels is not None else info_channels
        if channels is None:
            raise ValueError("vae.latent_channels is required when the custom VAE does not expose info.latent_channels.")
        self.info = VAEInfo(
            latent_channels=int(channels),
            latent_size=tuple(int(v) for v in (info_size or latent_size)),
            scale_factor=float(info_scale if info_scale is not None else scaling_factor),
        )

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(image)
        if isinstance(latent, dict):
            latent = latent["latent"] if "latent" in latent else latent.get("latents")
        if not torch.is_tensor(latent):
            raise TypeError("Custom VAE encode() must return a Tensor or a dict containing 'latent'/'latents'.")
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latent)
        if isinstance(image, dict):
            image = image["image"] if "image" in image else image.get("sample")
        if not torch.is_tensor(image):
            raise TypeError("Custom VAE decode() must return a Tensor or a dict containing 'image'/'sample'.")
        return image


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
    elif vae_type == "custom":
        class_path = vae_cfg.get("class_path")
        if not class_path:
            raise ValueError("vae.class_path is required for vae.type=custom")
        vae = CustomVAEAdapter(
            class_path=class_path,
            latent_channels=vae_cfg.get("latent_channels"),
            latent_size=vae_cfg.get(
                "latent_size",
                config.get("model", {}).get("latent_size", data_cfg.get("hr_size", [128, 512])),
            ),
            scaling_factor=float(vae_cfg.get("scaling_factor", 1.0)),
            kwargs=vae_cfg.get("kwargs", {}),
        )
    elif vae_type == "rdp_vae_f8c32":
        from dualtsr.vae.rdp_vae import VAE16X as RdpVAE
        vae_path = vae_cfg.get("vae_path")
        if not vae_path:
            raise ValueError("vae.vae_path is required for vae.type=rdp_vae_f8c32")
        vae = RdpVAEAdapter(
            vae_path=vae_path,
            latent_size=tuple(int(v) for v in vae_cfg.get(
                "latent_size",
                config.get("model", {}).get("latent_size", data_cfg.get("hr_size", [128, 512])),
            )),
            scaling_factor=float(vae_cfg.get("scaling_factor", 0.2517327)),
            shift_factor=float(vae_cfg.get("shift_factor", 0.07050679)),
            latent_channels=int(vae_cfg.get("latent_channels", 32)),
            use_checkpoint=vae_cfg.get("use_checkpoint", False),
        )
    else:
        raise ValueError(f"Unsupported vae.type: {vae_type}")
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae
