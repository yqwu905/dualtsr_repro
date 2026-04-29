from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .data import load_rgb, pil_to_tensor


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def ned(pred: str, gt: str) -> float:
    denom = max(len(pred), len(gt), 1)
    return 1.0 - levenshtein(pred, gt) / denom


def load_image_tensor(path: str | Path, size=None) -> torch.Tensor:
    return pil_to_tensor(load_rgb(path, size))


def maybe_lpips(preds: list[torch.Tensor], gts: list[torch.Tensor], device: torch.device) -> float | None:
    try:
        import lpips
    except Exception:
        return None
    metric = lpips.LPIPS(net="alex").to(device)
    values = []
    with torch.no_grad():
        for pred, gt in zip(preds, gts):
            p = pred.unsqueeze(0).to(device).mul(2).sub(1)
            g = gt.unsqueeze(0).to(device).mul(2).sub(1)
            values.append(float(metric(p, g).detach().cpu()))
    return float(np.mean(values)) if values else None


def maybe_fid(preds: list[torch.Tensor], gts: list[torch.Tensor], device: torch.device) -> float | None:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        return None
    metric = FrechetInceptionDistance(normalize=True).to(device)
    for pred, gt in zip(preds, gts):
        metric.update(pred.unsqueeze(0).to(device), real=False)
        metric.update(gt.unsqueeze(0).to(device), real=True)
    return float(metric.compute().detach().cpu())

