from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from dualtsr.device import dtype_from_precision
from dualtsr.registry import load_class


def _extract_tensor(output: Any, keys: tuple[str, ...], operation: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in keys:
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    raise TypeError(f"VAE {operation}() must return a Tensor or a dict containing one of {keys}.")


class IdentityVAE(nn.Module):
    """Pass-through VAE used for smoke tests."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.channels = int(channels)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.clamp(0, 1)


class AutoencoderKLVAE(nn.Module):
    """Wrapper around diffusers AutoencoderKL with DualTSR normalization."""

    def __init__(
        self,
        pretrained_path: str | Path,
        scaling_factor: float = 0.18215,
        dtype: str | torch.dtype = "fp32",
    ) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise RuntimeError("diffusers is required for AutoencoderKLVAE") from exc
        self.scaling_factor = float(scaling_factor)
        target_dtype = dtype_from_precision(dtype) if isinstance(dtype, str) else dtype
        self.vae = AutoencoderKL.from_pretrained(str(pretrained_path))
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(dtype=target_dtype)

    def _param_dtype(self) -> torch.dtype:
        parameter = next(self.vae.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(dtype=self._param_dtype()).mul(2.0).sub(1.0)
        return self.vae.encode(image).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latent.to(dtype=self._param_dtype()) / self.scaling_factor).sample
        return image.add(1.0).mul(0.5).clamp(0, 1).float()


class AutoencoderDCVAE(nn.Module):
    """Nitro-E's public 32x spatial-compression, 32-channel visual tokenizer."""

    def __init__(
        self,
        pretrained_path: str = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
        scaling_factor: float | None = None,
        dtype: str | torch.dtype = "fp32",
    ) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderDC
        except ImportError as exc:
            raise RuntimeError("diffusers>=0.32.2 is required for AutoencoderDCVAE") from exc
        target_dtype = dtype_from_precision(dtype) if isinstance(dtype, str) else dtype
        self.vae = AutoencoderDC.from_pretrained(str(pretrained_path), torch_dtype=target_dtype)
        configured_scale = getattr(self.vae.config, "scaling_factor", 1.0)
        self.scaling_factor = float(configured_scale if scaling_factor is None else scaling_factor)
        self.vae.requires_grad_(False)
        self.vae.eval()

    def _param_dtype(self) -> torch.dtype:
        parameter = next(self.vae.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        normalized = image.to(dtype=self._param_dtype()).mul(2.0).sub(1.0)
        latent = _extract_tensor(self.vae.encode(normalized), ("latent", "latents", "sample"), "encode")
        return latent * self.scaling_factor

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        output = self.vae.decode(latent.to(dtype=self._param_dtype()) / self.scaling_factor)
        image = _extract_tensor(output, ("image", "images", "sample"), "decode")
        return image.add(1.0).mul(0.5).clamp(0, 1).float()


class CustomVAEAdapter(nn.Module):
    """Normalizes custom VAE tensor/dict outputs to the local encode/decode contract."""

    def __init__(self, class_path: str, kwargs: dict[str, Any] | None = None) -> None:
        super().__init__()
        cls = load_class(class_path)
        self.vae = cls(**(kwargs or {}))
        if not isinstance(self.vae, nn.Module):
            raise TypeError("Custom VAE must inherit torch.nn.Module.")

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return _extract_tensor(self.vae.encode(image), ("latent", "latents", "sample"), "encode")

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return _extract_tensor(self.vae.decode(latent), ("image", "images", "sample"), "decode")


def _build_legacy_vae(config: dict, vae_cfg: dict) -> nn.Module:
    vae_type = str(vae_cfg.get("type", "identity")).lower()
    if vae_type == "identity":
        return IdentityVAE(channels=int(vae_cfg.get("channels", 3)))
    if vae_type == "autoencoder_kl":
        pretrained_path = vae_cfg.get("pretrained_path")
        if not pretrained_path:
            raise ValueError("vae.pretrained_path is required for vae.type=autoencoder_kl")
        return AutoencoderKLVAE(
            pretrained_path=pretrained_path,
            scaling_factor=float(vae_cfg.get("scaling_factor", 0.18215)),
            dtype=str(config.get("runtime", {}).get("precision", "fp32")),
        )
    if vae_type == "autoencoder_dc":
        return AutoencoderDCVAE(
            pretrained_path=str(
                vae_cfg.get("pretrained_path", "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")
            ),
            scaling_factor=vae_cfg.get("scaling_factor"),
            dtype=str(config.get("runtime", {}).get("precision", "fp32")),
        )
    if vae_type == "custom":
        class_path = vae_cfg.get("class_path")
        if not class_path:
            raise ValueError("vae.class_path is required for vae.type=custom")
        return CustomVAEAdapter(class_path=class_path, kwargs=vae_cfg.get("kwargs", {}))
    if vae_type == "rdp_vae_f8c32":
        from dualtsr.vae.rdp_vae import RdpVAEAdapter

        vae_path = vae_cfg.get("vae_path")
        if not vae_path:
            raise ValueError("vae.vae_path is required for vae.type=rdp_vae_f8c32")
        latent_size = vae_cfg.get("latent_size", config.get("model", {}).get("latent_size"))
        if latent_size is None:
            hr_size = config.get("data", {}).get("hr_size", [128, 512])
            latent_size = [int(hr_size[0]) // 8, int(hr_size[1]) // 8]
        return RdpVAEAdapter(
            vae_path=str(vae_path),
            latent_size=tuple(int(v) for v in latent_size),
            scaling_factor=float(vae_cfg.get("scaling_factor", 0.2517327)),
            shift_factor=float(vae_cfg.get("shift_factor", 0.07050679)),
            latent_channels=int(vae_cfg.get("latent_channels", 32)),
            use_checkpoint=bool(vae_cfg.get("use_checkpoint", False)),
        )
    raise ValueError(f"Unsupported vae.type: {vae_type}")


def build_vae(config: dict, device: torch.device) -> nn.Module:
    """Build a VAE from class_path/init_args, with legacy type configs supported."""

    vae_cfg = config.get("vae", {})
    class_path = vae_cfg.get("class_path")
    if class_path:
        cls = load_class(class_path)
        init_args = dict(vae_cfg.get("init_args", {}) or {})
        if cls in {AutoencoderKLVAE, AutoencoderDCVAE}:
            init_args.setdefault("dtype", str(config.get("runtime", {}).get("precision", "fp32")))
        vae = cls(**init_args)
        if not isinstance(vae, nn.Module):
            raise TypeError("Configured VAE must inherit torch.nn.Module.")
    else:
        vae = _build_legacy_vae(config, vae_cfg)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def update_model_latent_shape(config: dict, vae: nn.Module, device: torch.device) -> tuple[int, list[int]]:
    """Infer the actual latent shape using the configured training resolution."""

    data_cfg = config.get("data", {})
    hr_size: Iterable[int] = data_cfg.get("hr_size", [128, 512])
    h, w = (int(v) for v in hr_size)
    channels = int(data_cfg.get("image_channels", 3))
    dummy = torch.zeros(1, channels, h, w, device=device)
    with torch.no_grad():
        latent = _extract_tensor(vae.encode(dummy), ("latent", "latents", "sample"), "encode")
    if latent.ndim != 4:
        raise ValueError(f"VAE encode() must produce a BCHW tensor, got shape {tuple(latent.shape)}")
    latent_channels = int(latent.shape[1])
    latent_size = [int(latent.shape[-2]), int(latent.shape[-1])]
    config.setdefault("model", {})
    config["model"]["latent_channels"] = latent_channels
    config["model"]["latent_size"] = latent_size
    return latent_channels, latent_size
