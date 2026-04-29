from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / max(half, 1))
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.img_norm1 = AdaLayerNorm(dim, cond_dim)
        self.txt_norm1 = AdaLayerNorm(dim, cond_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.img_norm2 = AdaLayerNorm(dim, cond_dim)
        self.txt_norm2 = AdaLayerNorm(dim, cond_dim)
        self.img_mlp = MLP(dim, mlp_ratio)
        self.txt_mlp = MLP(dim, mlp_ratio)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_len = img.shape[1]
        joint = torch.cat([self.img_norm1(img, cond), self.txt_norm1(txt, cond)], dim=1)
        qkv = self.qkv(joint).view(joint.shape[0], joint.shape[1], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else self.drop.p)
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            out = attn.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(joint.shape[0], joint.shape[1], self.dim)
        out = self.proj(out)
        img = img + self.drop(out[:, :img_len])
        txt = txt + self.drop(out[:, img_len:])
        img = img + self.drop(self.img_mlp(self.img_norm2(img, cond)))
        txt = txt + self.drop(self.txt_mlp(self.txt_norm2(txt, cond)))
        return img, txt


class DualTSRModel(nn.Module):
    def __init__(self, config: dict, vocab_size: int, mask_id: int) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        self.vocab_size = int(vocab_size)
        self.mask_id = int(mask_id)
        self.max_text_length = int(data_cfg.get("max_text_length", model_cfg.get("max_text_length", 24)))
        self.latent_channels = int(model_cfg.get("latent_channels", 3))
        self.latent_size = tuple(int(v) for v in model_cfg.get("latent_size", data_cfg.get("hr_size", [128, 512])))
        patch = model_cfg.get("patch_size", [8, 8])
        self.patch_size = tuple(int(v) for v in patch)
        if self.latent_size[0] % self.patch_size[0] or self.latent_size[1] % self.patch_size[1]:
            raise ValueError("model.latent_size must be divisible by model.patch_size")
        self.grid_size = (self.latent_size[0] // self.patch_size[0], self.latent_size[1] // self.patch_size[1])
        self.num_image_tokens = self.grid_size[0] * self.grid_size[1]
        dim = int(model_cfg.get("hidden_dim", 768))
        heads = int(model_cfg.get("num_heads", 12))
        depth = int(model_cfg.get("depth", 12))
        mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
        dropout = float(model_cfg.get("dropout", 0.0))
        self.hidden_dim = dim

        ph, pw = self.patch_size
        self.patch_embed = nn.Conv2d(self.latent_channels, dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.lr_embed = nn.Conv2d(self.latent_channels, dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.image_pos = nn.Parameter(torch.zeros(1, self.num_image_tokens, dim))
        self.text_embed = nn.Embedding(self.vocab_size, dim)
        self.text_pos = nn.Parameter(torch.zeros(1, self.max_text_length, dim))
        self.null_text = nn.Parameter(torch.zeros(1, self.max_text_length, dim))

        time_dim = int(model_cfg.get("time_dim", dim * 4))
        self.time_mlp = nn.Sequential(nn.Linear(dim, time_dim), nn.SiLU(), nn.Linear(time_dim, dim))
        self.blocks = nn.ModuleList(
            [JointAttentionBlock(dim, heads, mlp_ratio, cond_dim=dim, dropout=dropout) for _ in range(depth)]
        )
        self.final_img_norm = nn.LayerNorm(dim)
        self.final_txt_norm = nn.LayerNorm(dim)
        self.velocity_head = nn.Linear(dim, self.latent_channels * ph * pw)
        self.text_head = nn.Linear(dim, self.vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.image_pos, std=0.02)
        nn.init.trunc_normal_(self.text_pos, std=0.02)
        nn.init.trunc_normal_(self.null_text, std=0.02)

    def _patchify(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.latent_size:
            raise ValueError(f"Expected latent spatial size {self.latent_size}, got {tuple(x.shape[-2:])}")
        tokens = conv(x)
        return tokens.flatten(2).transpose(1, 2)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        ph, pw = self.patch_size
        gh, gw = self.grid_size
        patches = self.velocity_head(self.final_img_norm(tokens))
        patches = patches.view(b, gh, gw, self.latent_channels, ph, pw)
        image = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return image.view(b, self.latent_channels, gh * ph, gw * pw)

    def forward(
        self,
        x_img: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        lr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = x_img.shape[0]
        if t.ndim == 0:
            t = t.expand(b)
        img_tokens = self._patchify(x_img, self.patch_embed)
        if lr is not None:
            img_tokens = img_tokens + self._patchify(lr, self.lr_embed)
        img_tokens = img_tokens + self.image_pos

        if text_tokens is None:
            txt_tokens = self.null_text.expand(b, -1, -1)
        else:
            txt_tokens = self.text_embed(text_tokens[:, : self.max_text_length])
            if txt_tokens.shape[1] < self.max_text_length:
                pad_len = self.max_text_length - txt_tokens.shape[1]
                txt_tokens = torch.cat([txt_tokens, self.null_text.expand(b, -1, -1)[:, :pad_len]], dim=1)
            txt_tokens = txt_tokens + self.text_pos

        cond = self.time_mlp(timestep_embedding(t, self.hidden_dim))
        for block in self.blocks:
            img_tokens, txt_tokens = block(img_tokens, txt_tokens, cond)
        velocity = self._unpatchify(img_tokens)
        logits = self.text_head(self.final_txt_norm(txt_tokens))
        return {"velocity": velocity, "logits": logits}


def build_model(config: dict, vocab_size: int, mask_id: int) -> DualTSRModel:
    return DualTSRModel(config, vocab_size=vocab_size, mask_id=mask_id)

