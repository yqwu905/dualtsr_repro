from __future__ import annotations

import copy

import torch
from torch import nn


def make_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(unwrap_model(model))
    ema.requires_grad_(False)
    ema.eval()
    return ema


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    src = unwrap_model(model).state_dict()
    dst = ema_model.state_dict()
    for key, value in dst.items():
        if key not in src:
            continue
        src_value = src[key].detach()
        if value.dtype.is_floating_point:
            value.mul_(decay).add_(src_value.to(value.dtype), alpha=1.0 - decay)
        else:
            value.copy_(src_value)

