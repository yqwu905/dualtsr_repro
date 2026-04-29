from __future__ import annotations

import torch
import torch.nn.functional as F


def expand_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    while t.ndim < ndim:
        t = t.view(*t.shape, 1)
    return t


def sample_uniform_t(batch_size: int, device: torch.device, eps: float = 1e-4) -> torch.Tensor:
    return torch.rand(batch_size, device=device).clamp(eps, 1.0)


def cfm_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t_exp = expand_t(t, x0.ndim)
    xt = (1.0 - t_exp) * x0 + t_exp * noise
    target = noise - x0
    return xt, target


def corrupt_text(tokens: torch.Tensor, t: torch.Tensor, mask_id: int, pad_id: int) -> torch.Tensor:
    """Absorbing-state corruption with alpha_t=1-t, so mask probability is t."""

    if t.ndim == 0:
        t = t.expand(tokens.shape[0])
    probs = t.to(tokens.device).view(-1, 1).expand_as(tokens)
    rand = torch.rand(tokens.shape, device=tokens.device)
    can_mask = tokens.ne(pad_id)
    masked = tokens.clone()
    masked[(rand < probs) & can_mask] = mask_id
    return masked


def antithetic_timesteps(batch_size: int, k: int, device: torch.device, delta: float = 1e-4) -> torch.Tensor:
    base = torch.arange(k, device=device, dtype=torch.float32).view(1, k)
    offset = torch.rand(batch_size, 1, device=device)
    t = (base + offset) / float(k)
    return (delta + t * (1.0 - delta)).clamp(delta, 1.0)


def guided_velocity_target(
    base_velocity: torch.Tensor,
    cond_velocity: torch.Tensor,
    uncond_velocity: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    return base_velocity + float(guidance_scale) * (cond_velocity.detach() - uncond_velocity.detach())


def token_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_id: int,
    timestep: torch.Tensor | None = None,
    weight_by_time: bool = False,
    delta: float = 1e-4,
) -> torch.Tensor:
    b, seq_len, vocab = logits.shape
    loss = F.cross_entropy(
        logits.reshape(b * seq_len, vocab).float(),
        target.reshape(b * seq_len),
        ignore_index=pad_id,
        reduction="none",
    ).view(b, seq_len)
    valid = target.ne(pad_id).float()
    denom = valid.sum(dim=1).clamp_min(1.0)
    loss = (loss * valid).sum(dim=1) / denom
    if weight_by_time and timestep is not None:
        loss = loss / timestep.float().clamp_min(delta)
    return loss.mean()

