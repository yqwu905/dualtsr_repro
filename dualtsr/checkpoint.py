from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import config_to_jsonable
from .ema import unwrap_model


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "npu"):
        try:
            state["npu"] = torch.npu.get_rng_state_all()
        except Exception:
            pass
    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "npu" in state and hasattr(torch, "npu"):
        try:
            torch.npu.set_rng_state_all(state["npu"])
        except Exception:
            pass


def save_checkpoint(
    path: str | Path,
    *,
    model,
    ema_model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    config: dict,
    tokenizer,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "rng": rng_state(),
        "config": config_to_jsonable(config),
        "tokenizer": tokenizer.state_dict() if tokenizer is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
