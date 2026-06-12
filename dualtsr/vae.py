from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from dualtsr.device import dtype_from_precision
from dualtsr.registry import load_class


class IdentityVAE(nn.Module):
    """Pass-through VAE used for smoke tests; latents are the images themselves."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.channels = int(channels)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.clamp(0, 1)


class AutoencoderKLVAE(nn.Module):
    """diffusers ``AutoencoderKL`` wrapper. Scaling and normalization live here.

    ``encode`` maps an image in ``[0, 1]`` to a scaled latent; ``decode`` reverses
    both the scaling and the normalization, returning an image in ``[0, 1]``.
    """

    def __init__(
        self,
        pretrained_path: str | Path,
        scaling_factor: float = 0.18215,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise RuntimeError("diffusers is required for AutoencoderKLVAE") from exc
        self.scaling_factor = float(scaling_factor)
        self.vae = AutoencoderKL.from_pretrained(str(pretrained_path))
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(dtype=dtype_from_precision(dtype))

    def _param_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(dtype=self._param_dtype()).mul(2.0).sub(1.0)
        dist = self.vae.encode(image).latent_dist
        return dist.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.to(dtype=self._param_dtype()) / self.scaling_factor
        image = self.vae.decode(latent).sample
        return image.add(1.0).mul(0.5).clamp(0, 1).float()


def build_vae(config: dict, device: torch.device) -> nn.Module:
    """Instantiate the VAE from ``vae.class_path`` + ``vae.init_args``.

    The VAE only needs ``encode`` and ``decode``; any scaling factor is its own
    responsibility. The model learns the latent shape via :func:`update_model_latent_shape`.
    """
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


def update_model_latent_shape(config: dict, vae: nn.Module, device: torch.device) -> tuple[int, list[int]]:
    """Infer ``latent_channels`` / ``latent_size`` by dry-running ``vae.encode``.

    Writes the result into ``config['model']`` so the model can build its
    text/output projections without the VAE exposing anything beyond encode/decode.
    """
    data_cfg = config.get("data", {})
    hr_size: Iterable[int] = data_cfg.get("hr_size", [128, 512])
    h, w = (int(v) for v in hr_size)
    channels = int(data_cfg.get("image_channels", 3))
    dummy = torch.zeros(1, channels, h, w, device=device)
    with torch.no_grad():
        latent = vae.encode(dummy)
    latent_channels = int(latent.shape[1])
    latent_size = [int(latent.shape[-2]), int(latent.shape[-1])]
    config.setdefault("model", {})
    config["model"]["latent_channels"] = latent_channels
    config["model"]["latent_size"] = latent_size
    return latent_channels, latent_size
