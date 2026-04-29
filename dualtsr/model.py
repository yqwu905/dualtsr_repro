from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dualtsr.registry import load_class


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


def _extract_text_embeddings(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("embeddings", "hidden_states", "text_tokens"):
            if key in output:
                return output[key]
    if torch.is_tensor(output):
        return output
    raise TypeError("Text encoder must return a Tensor or a dict with embeddings/hidden_states/text_tokens.")


def _extract_backbone_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, dict):
        for image_key in ("image_tokens", "img_tokens", "image"):
            if image_key in output:
                image_tokens = output[image_key]
                break
        else:
            raise KeyError("MMDiT output dict must contain image_tokens/img_tokens/image.")
        for text_key in ("text_tokens", "txt_tokens", "text"):
            if text_key in output:
                text_tokens = output[text_key]
                break
        else:
            raise KeyError("MMDiT output dict must contain text_tokens/txt_tokens/text.")
        return image_tokens, text_tokens
    if isinstance(output, tuple) and len(output) == 2:
        return output
    raise TypeError("MMDiT backbone must return (image_tokens, text_tokens) or a compatible dict.")


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.requires_grad_(trainable)
    if not trainable:
        module.eval()


class CharTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, max_text_length: int) -> None:
        super().__init__()
        self.output_dim = int(hidden_dim)
        self.max_text_length = int(max_text_length)
        self.embedding = nn.Embedding(int(vocab_size), self.output_dim)
        self.position = nn.Parameter(torch.zeros(1, self.max_text_length, self.output_dim))
        self.null_text = nn.Parameter(torch.zeros(1, self.max_text_length, self.output_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.null_text, std=0.02)

    def forward(self, text_tokens: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if text_tokens is None:
            return self.null_text.expand(batch_size, -1, -1)
        txt_tokens = self.embedding(text_tokens[:, : self.max_text_length])
        if txt_tokens.shape[1] < self.max_text_length:
            pad_len = self.max_text_length - txt_tokens.shape[1]
            txt_tokens = torch.cat([txt_tokens, self.null_text.expand(batch_size, -1, -1)[:, :pad_len]], dim=1)
        return txt_tokens + self.position


class CustomTextEncoderAdapter(nn.Module):
    def __init__(
        self,
        class_path: str,
        output_dim: int | None,
        max_text_length: int,
        kwargs: dict[str, Any] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        cls = load_class(class_path)
        self.encoder = cls(**(kwargs or {}))
        if not isinstance(self.encoder, nn.Module):
            raise TypeError("Custom text encoder must inherit torch.nn.Module.")
        self.trainable = bool(trainable)
        self.max_text_length = int(max_text_length)
        inferred_dim = getattr(self.encoder, "output_dim", None) or getattr(self.encoder, "hidden_size", None)
        if output_dim is None and inferred_dim is None:
            raise ValueError("model.text_encoder.output_dim is required when the custom encoder does not expose output_dim.")
        self.output_dim = int(output_dim if output_dim is not None else inferred_dim)
        _set_trainable(self.encoder, self.trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.encoder.eval()
        return self

    def forward(self, text_tokens: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        output = self.encoder(
            text_tokens=text_tokens,
            batch_size=batch_size,
            max_length=self.max_text_length,
            device=device,
        )
        return _extract_text_embeddings(output)


class NativeMMDiTBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        time_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, int(time_dim)),
            nn.SiLU(),
            nn.Linear(int(time_dim), self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                JointAttentionBlock(
                    self.hidden_dim,
                    int(num_heads),
                    float(mlp_ratio),
                    cond_dim=self.hidden_dim,
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )

    def forward(
        self,
        img_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim))
        for block in self.blocks:
            img_tokens, text_tokens = block(img_tokens, text_tokens, cond)
        return img_tokens, text_tokens


class CustomMMDiTAdapter(nn.Module):
    def __init__(
        self,
        class_path: str,
        kwargs: dict[str, Any] | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        cls = load_class(class_path)
        self.backbone = cls(**(kwargs or {}))
        if not isinstance(self.backbone, nn.Module):
            raise TypeError("Custom MMDiT backbone must inherit torch.nn.Module.")
        self.trainable = bool(trainable)
        _set_trainable(self.backbone, self.trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self

    def forward(
        self,
        img_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.backbone(img_tokens=img_tokens, text_tokens=text_tokens, timesteps=timesteps)
        return _extract_backbone_output(output)


def build_text_encoder(config: dict, vocab_size: int, hidden_dim: int, max_text_length: int) -> nn.Module:
    text_cfg = config.get("text_encoder", {})
    encoder_type = str(text_cfg.get("type", "char")).lower()
    if encoder_type in {"char", "native"}:
        return CharTextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim, max_text_length=max_text_length)
    if encoder_type == "custom":
        class_path = text_cfg.get("class_path")
        if not class_path:
            raise ValueError("model.text_encoder.class_path is required for type=custom")
        return CustomTextEncoderAdapter(
            class_path=class_path,
            output_dim=text_cfg.get("output_dim"),
            max_text_length=max_text_length,
            kwargs=text_cfg.get("kwargs", {}),
            trainable=bool(text_cfg.get("trainable", True)),
        )
    raise ValueError(f"Unsupported model.text_encoder.type: {encoder_type}")


def build_mmdit(config: dict, hidden_dim: int) -> nn.Module:
    mmdit_cfg = config.get("mmdit", {})
    mmdit_type = str(mmdit_cfg.get("type", "native")).lower()
    if mmdit_type in {"native", "mmdit"}:
        return NativeMMDiTBackbone(
            hidden_dim=hidden_dim,
            num_heads=int(mmdit_cfg.get("num_heads", config.get("num_heads", 12))),
            depth=int(mmdit_cfg.get("depth", config.get("depth", 12))),
            mlp_ratio=float(mmdit_cfg.get("mlp_ratio", config.get("mlp_ratio", 4.0))),
            dropout=float(mmdit_cfg.get("dropout", config.get("dropout", 0.0))),
            time_dim=int(mmdit_cfg.get("time_dim", config.get("time_dim", hidden_dim * 4))),
        )
    if mmdit_type == "custom":
        class_path = mmdit_cfg.get("class_path")
        if not class_path:
            raise ValueError("model.mmdit.class_path is required for type=custom")
        return CustomMMDiTAdapter(
            class_path=class_path,
            kwargs=mmdit_cfg.get("kwargs", {}),
            trainable=bool(mmdit_cfg.get("trainable", True)),
        )
    raise ValueError(f"Unsupported model.mmdit.type: {mmdit_type}")


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
        self.hidden_dim = dim

        ph, pw = self.patch_size
        self.patch_embed = nn.Conv2d(self.latent_channels, dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.lr_embed = nn.Conv2d(self.latent_channels, dim, kernel_size=(ph, pw), stride=(ph, pw))
        self.image_pos = nn.Parameter(torch.zeros(1, self.num_image_tokens, dim))
        self.text_encoder = build_text_encoder(
            model_cfg,
            vocab_size=self.vocab_size,
            hidden_dim=dim,
            max_text_length=self.max_text_length,
        )
        text_dim = int(getattr(self.text_encoder, "output_dim", dim))
        self.text_proj = nn.Identity() if text_dim == dim else nn.Linear(text_dim, dim)
        self.mmdit = build_mmdit(model_cfg, hidden_dim=dim)
        self.final_img_norm = nn.LayerNorm(dim)
        self.final_txt_norm = nn.LayerNorm(dim)
        self.velocity_head = nn.Linear(dim, self.latent_channels * ph * pw)
        self.text_head = nn.Linear(dim, self.vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.image_pos, std=0.02)

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

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        migrated: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("text_embed."):
                new_key = key.replace("text_embed.", "text_encoder.embedding.", 1)
            elif key == "text_pos":
                new_key = "text_encoder.position"
            elif key == "null_text":
                new_key = "text_encoder.null_text"
            elif key.startswith("time_mlp."):
                new_key = key.replace("time_mlp.", "mmdit.time_mlp.", 1)
            elif key.startswith("blocks."):
                new_key = key.replace("blocks.", "mmdit.blocks.", 1)
            migrated[new_key] = value
        return super().load_state_dict(migrated, strict=strict, assign=assign)

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

        txt_tokens = self.text_encoder(text_tokens, batch_size=b, device=x_img.device)
        if txt_tokens.shape[1] != self.max_text_length:
            raise ValueError(f"Expected text token length {self.max_text_length}, got {txt_tokens.shape[1]}")
        if txt_tokens.shape[-1] != self.hidden_dim:
            txt_tokens = self.text_proj(txt_tokens)
        img_tokens, txt_tokens = self.mmdit(img_tokens, txt_tokens, t)
        velocity = self._unpatchify(img_tokens)
        logits = self.text_head(self.final_txt_norm(txt_tokens))
        return {"velocity": velocity, "logits": logits}


def build_model(config: dict, vocab_size: int, mask_id: int) -> DualTSRModel:
    return DualTSRModel(config, vocab_size=vocab_size, mask_id=mask_id)
