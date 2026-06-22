import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union
from safetensors.torch import load_file

from .rdp_vae_ch32_8x import EncoderX_f8c32, DecoderX_f8c32, VAE16X
from .rdp_unet import RdpUnet, NAFBlock, LayerNorm2d, SimpleGate


class RdpVAEAdapter(nn.Module):
    def __init__(
        self,
        vae_path: str,
        latent_size: Tuple[int, int],
        scaling_factor: float = 0.2517327,
        shift_factor: float = 0.07050679,
        latent_channels: int = 32,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EncoderX_f8c32(use_checkpoint=use_checkpoint)
        self.decoder = DecoderX_f8c32(use_checkpoint=use_checkpoint)

        if vae_path.endswith(".safetensors"):
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, weights_only=True, map_location="cpu")
        temp_vae = VAE16X(use_checkpoint=use_checkpoint)
        temp_vae.load_state_dict(state_dict, strict=False)
        self.encoder.load_state_dict(temp_vae.encoder.state_dict(), strict=True)
        self.decoder.load_state_dict(temp_vae.decoder.state_dict(), strict=True)

        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.encoder.eval()
        self.decoder.eval()

        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.info = _VAEInfo(
            latent_channels=latent_channels,
            latent_size=latent_size,
            scale_factor=scaling_factor,
        )

    @torch.no_grad()
    def encode(self, image: Tensor) -> Tensor:
        # RDP VAE is trained on RGB tensors normalized to [-1, 1].
        z = self.encoder(image.mul(2.0).sub(1.0))
        z = (z - self.shift_factor) * self.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        latent = latent / self.scaling_factor + self.shift_factor
        image = self.decoder(latent)
        image = (image + 1.0) / 2.0
        return image.clamp(0.0, 1.0)


class _VAEInfo:
    __slots__ = ("latent_channels", "latent_size", "scale_factor")

    def __init__(self, latent_channels: int, latent_size: Tuple[int, int], scale_factor: float) -> None:
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.scale_factor = scale_factor


__all__ = [
    "RdpVAEAdapter",
    "EncoderX_f8c32",
    "DecoderX_f8c32",
    "VAE16X",
    "RdpUnet",
    "NAFBlock",
    "LayerNorm2d",
    "SimpleGate",
]
