from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from dualtsr.model import timestep_embedding


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(f"Expected two values, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _sincos_1d(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim == 0:
        return torch.empty(length, 0, device=device, dtype=dtype)
    half = dim // 2
    omega = torch.arange(half, device=device, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * omega / max(half, 1))
    positions = torch.arange(length, device=device, dtype=torch.float32)
    values = positions[:, None] * omega[None]
    embedding = torch.cat([values.sin(), values.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding[:, :dim].to(dtype=dtype)


def rectangular_sincos_position(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a deterministic 2D sinusoidal embedding for rectangular text latents."""

    height_dim = dim // 2
    width_dim = dim - height_dim
    emb_h = _sincos_1d(height, height_dim, device, dtype)
    emb_w = _sincos_1d(width, width_dim, device, dtype)
    grid_h = emb_h[:, None, :].expand(height, width, height_dim)
    grid_w = emb_w[None, :, :].expand(height, width, width_dim)
    return torch.cat([grid_h, grid_w], dim=-1).reshape(1, height * width, dim)


def _subregion_pack(tokens: torch.Tensor, ratio: int, sequence_chunk: int) -> tuple[torch.Tensor, int]:
    if ratio == 1:
        return tokens, 0
    unit = ratio * sequence_chunk
    pad = (-tokens.shape[1]) % unit
    if pad:
        tokens = F.pad(tokens, (0, 0, 0, pad))
    batch, length, channels = tokens.shape
    tokens = tokens.reshape(batch, length // unit, ratio, sequence_chunk, channels)
    tokens = tokens.permute(0, 2, 1, 3, 4).reshape(batch * ratio, -1, channels)
    return tokens, pad


def _subregion_unpack(
    tokens: torch.Tensor,
    batch_size: int,
    ratio: int,
    sequence_chunk: int,
    pad: int,
) -> torch.Tensor:
    if ratio == 1:
        return tokens
    channels = tokens.shape[-1]
    groups = tokens.shape[1] // sequence_chunk
    tokens = tokens.reshape(batch_size, ratio, groups, sequence_chunk, channels)
    tokens = tokens.permute(0, 2, 1, 3, 4).reshape(batch_size, groups * ratio * sequence_chunk, channels)
    return tokens[:, : tokens.shape[1] - pad] if pad else tokens


class EMMDiTJointBlock(nn.Module):
    """E-MMDiT joint block with AdaLN-affine and alternating subregion attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        subregion_ratio: int = 1,
        subregion_chunk: int = 1,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.subregion_ratio = int(subregion_ratio)
        self.subregion_chunk = int(subregion_chunk)

        self.image_norm1 = nn.LayerNorm(dim)
        self.text_norm1 = nn.LayerNorm(dim)
        self.image_qkv = nn.Linear(dim, dim * 3)
        self.text_qkv = nn.Linear(dim, dim * 3)
        self.image_attn_out = nn.Linear(dim, dim)
        self.text_attn_out = nn.Linear(dim, dim)
        self.image_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.text_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.image_mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(approximate="tanh"), nn.Linear(mlp_dim, dim))
        self.text_mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(approximate="tanh"), nn.Linear(mlp_dim, dim))
        self.dropout = nn.Dropout(dropout)

        # Official E-MMDiT uses a learned affine transform of the shared timestep
        # embedding instead of an MLP per block.
        self.image_affine_bias = nn.Parameter(torch.randn(6, dim) / math.sqrt(dim))
        self.image_affine_scale = nn.Parameter(torch.randn(6, dim) / math.sqrt(dim))
        self.text_affine_bias = nn.Parameter(torch.randn(6, dim) / math.sqrt(dim))
        self.text_affine_scale = nn.Parameter(torch.randn(6, dim) / math.sqrt(dim))

    def _modulation(self, temb: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = bias[None] + temb.reshape(temb.shape[0], 6, self.dim) * (1.0 + scale[None])
        return tuple(values.chunk(6, dim=1))

    def _qkv(self, tokens: torch.Tensor, projection: nn.Linear) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = tokens.shape
        qkv = projection(tokens).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_mod = self._modulation(temb, self.image_affine_bias, self.image_affine_scale)
        text_mod = self._modulation(temb, self.text_affine_bias, self.text_affine_scale)
        image_shift_attn, image_scale_attn, image_gate_attn, image_shift_mlp, image_scale_mlp, image_gate_mlp = image_mod
        text_shift_attn, text_scale_attn, text_gate_attn, text_shift_mlp, text_scale_mlp, text_gate_mlp = text_mod

        image_norm = self.image_norm1(image) * (1.0 + image_scale_attn) + image_shift_attn
        text_norm = self.text_norm1(text) * (1.0 + text_scale_attn) + text_shift_attn
        batch_size = image.shape[0]
        image_norm, image_pad = _subregion_pack(image_norm, self.subregion_ratio, self.subregion_chunk)
        text_norm, text_pad = _subregion_pack(text_norm, self.subregion_ratio, self.subregion_chunk)

        image_length = image_norm.shape[1]
        q_img, k_img, v_img = self._qkv(image_norm, self.image_qkv)
        q_txt, k_txt, v_txt = self._qkv(text_norm, self.text_qkv)
        q = torch.cat([q_img, q_txt], dim=2)
        k = torch.cat([k_img, k_txt], dim=2)
        v = torch.cat([v_img, v_txt], dim=2)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(q.shape[0], q.shape[2], self.dim)
        image_attention = self.image_attn_out(attention[:, :image_length])
        text_attention = self.text_attn_out(attention[:, image_length:])
        image_attention = _subregion_unpack(
            image_attention, batch_size, self.subregion_ratio, self.subregion_chunk, image_pad
        )
        text_attention = _subregion_unpack(
            text_attention, batch_size, self.subregion_ratio, self.subregion_chunk, text_pad
        )

        image = image + self.dropout(image_gate_attn * image_attention)
        text = text + self.dropout(text_gate_attn * text_attention)
        image_norm = self.image_norm2(image) * (1.0 + image_scale_mlp) + image_shift_mlp
        text_norm = self.text_norm2(text) * (1.0 + text_scale_mlp) + text_shift_mlp
        image = image + self.dropout(image_gate_mlp * self.image_mlp(image_norm))
        text = text + self.dropout(text_gate_mlp * self.text_mlp(text_norm))
        return image, text


class MultiPathTokenCompressor(nn.Module):
    """E-MMDiT 2x/4x visual-token compression."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.path_2x = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(dim * 4, dim, 1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(dim, dim, 1),
        )
        self.path_4x = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(dim * 16, dim, 1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if height % 4 or width % 4:
            raise ValueError("E-MMDiT token compression requires image token dimensions divisible by 4")
        image = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
        tokens_2x = self.path_2x(image).flatten(2).transpose(1, 2)
        tokens_4x = self.path_4x(image).flatten(2).transpose(1, 2)
        return torch.cat([tokens_2x, tokens_4x], dim=1)


class MultiPathTokenReconstructor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 16:
            raise ValueError("E-MMDiT token reconstruction requires hidden_dim divisible by 16")
        self.path_2x = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(dim // 4, dim, 1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(dim, dim, 1),
        )
        self.path_4x = nn.Sequential(
            nn.PixelShuffle(4),
            nn.Conv2d(dim // 16, dim, 1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(dim, dim, 1),
        )
        self.merge = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

    def forward(self, tokens: torch.Tensor, skip: torch.Tensor, height: int, width: int) -> torch.Tensor:
        length_2x = (height // 2) * (width // 2)
        tokens_2x, tokens_4x = torch.split(tokens, [length_2x, tokens.shape[1] - length_2x], dim=1)
        image_2x = tokens_2x.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height // 2, width // 2)
        image_4x = tokens_4x.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height // 4, width // 4)
        restored_2x = self.path_2x(image_2x).flatten(2).transpose(1, 2)
        restored_4x = self.path_4x(image_4x).flatten(2).transpose(1, 2)
        return self.merge(torch.cat([restored_2x, restored_4x, skip], dim=-1))


class EMMDiTBackbone(nn.Module):
    """DualTSR backbone based on AMD's E-MMDiT design.

    It preserves E-MMDiT's efficient image-token path while keeping the text
    stream alive through the final block, because DualTSR predicts both image
    velocity and clean text tokens from the shared transformer.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_channels: int,
        latent_size: Sequence[int],
        patch_size: int | Sequence[int] = 1,
        num_heads: int = 12,
        group_depths: Sequence[int] = (4, 16, 4),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        use_subregion_attention: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if len(group_depths) != 3:
            raise ValueError("E-MMDiT requires three group depths")
        self.hidden_dim = int(hidden_dim)
        self.latent_channels = int(latent_channels)
        self.latent_size = _pair(latent_size)
        self.patch_size = _pair(patch_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.latent_size[0] % self.patch_size[0] or self.latent_size[1] % self.patch_size[1]:
            raise ValueError("latent_size must be divisible by patch_size")
        self.grid_size = (
            self.latent_size[0] // self.patch_size[0],
            self.latent_size[1] // self.patch_size[1],
        )
        if self.grid_size[0] % 4 or self.grid_size[1] % 4:
            raise ValueError("E-MMDiT image-token grid must be divisible by 4")

        self.image_embed = nn.Conv2d(
            self.latent_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.lr_embed = nn.Conv2d(
            self.latent_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 6),
        )
        ratios = (1, 4, 4) if use_subregion_attention else (1, 1, 1)
        chunks = (1, 1, 4) if use_subregion_attention else (1, 1, 1)
        block_index = 0
        groups: list[nn.ModuleList] = []
        for depth in group_depths:
            blocks = []
            for _ in range(int(depth)):
                schedule_index = block_index % 3
                blocks.append(
                    EMMDiTJointBlock(
                        dim=self.hidden_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        subregion_ratio=ratios[schedule_index],
                        subregion_chunk=chunks[schedule_index],
                    )
                )
                block_index += 1
            groups.append(nn.ModuleList(blocks))
        self.block_groups = nn.ModuleList(groups)
        self.compressor = MultiPathTokenCompressor(self.hidden_dim)
        self.reconstructor = MultiPathTokenReconstructor(self.hidden_dim)
        self.image_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        ph, pw = self.patch_size
        self.velocity_head = nn.Linear(self.hidden_dim, self.latent_channels * ph * pw)

    def _position(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return rectangular_sincos_position(*self.grid_size, self.hidden_dim, device, dtype)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        grid_h, grid_w = self.grid_size
        patch_h, patch_w = self.patch_size
        patches = self.velocity_head(self.image_norm(tokens))
        patches = patches.reshape(batch, grid_h, grid_w, self.latent_channels, patch_h, patch_w)
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch,
            self.latent_channels,
            grid_h * patch_h,
            grid_w * patch_w,
        )

    def forward(
        self,
        x_img: torch.Tensor,
        timesteps: torch.Tensor,
        text: torch.Tensor,
        lr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(x_img.shape[-2:]) != self.latent_size:
            raise ValueError(f"Expected latent size {self.latent_size}, got {tuple(x_img.shape[-2:])}")
        image = self.image_embed(x_img).flatten(2).transpose(1, 2)
        if lr is not None:
            if tuple(lr.shape[-2:]) != self.latent_size:
                raise ValueError(f"Expected LR latent size {self.latent_size}, got {tuple(lr.shape[-2:])}")
            image = image + self.lr_embed(lr).flatten(2).transpose(1, 2)
        position = self._position(image.device, image.dtype)
        image = image + position
        temb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim).to(dtype=image.dtype))

        for block in self.block_groups[0]:
            image, text = self._run_block(block, image, text, temb)
        skip = image
        image = self.compressor(image, *self.grid_size)
        for block in self.block_groups[1]:
            image, text = self._run_block(block, image, text, temb)
        image = self.reconstructor(image, skip, *self.grid_size)
        image = image + position  # E-MMDiT position reinforcement.
        for block in self.block_groups[2]:
            image, text = self._run_block(block, image, text, temb)
        return self._unpatchify(image), text

    def _run_block(
        self,
        block: nn.Module,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, image, text, temb, use_reentrant=False)
        return block(image, text, temb)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
