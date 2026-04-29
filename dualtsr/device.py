from __future__ import annotations

import contextlib
import importlib.util
import os
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    requested: str
    backend: str | None
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool


def npu_is_available() -> bool:
    if importlib.util.find_spec("torch_npu") is None:
        return False
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def resolve_device(requested: str = "auto", local_rank: int | None = None) -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif npu_is_available():
            requested = "npu"
        else:
            requested = "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device=cuda was requested, but CUDA is unavailable.")
        idx = local_rank if local_rank is not None and local_rank >= 0 else 0
        return torch.device(f"cuda:{idx}")
    if requested == "npu":
        if not npu_is_available():
            raise RuntimeError("runtime.device=npu was requested, but Ascend NPU/torch_npu is unavailable.")
        idx = local_rank if local_rank is not None and local_rank >= 0 else 0
        return torch.device(f"npu:{idx}")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported runtime.device: {requested}")


def setup_runtime(config: dict) -> RuntimeInfo:
    runtime_cfg = config.get("runtime", {})
    requested = str(runtime_cfg.get("device", "auto")).lower()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    device = resolve_device(requested, local_rank if distributed else None)
    backend: str | None = None
    if distributed:
        if device.type == "cuda":
            torch.cuda.set_device(device)
            backend = "nccl"
        elif device.type == "npu":
            import torch_npu  # noqa: F401

            torch.npu.set_device(device)
            backend = "hccl"
        else:
            backend = "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
    return RuntimeInfo(
        device=device,
        requested=requested,
        backend=backend,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main=rank == 0,
    )


def cleanup_runtime() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def dtype_from_precision(precision: str) -> torch.dtype:
    precision = (precision or "fp32").lower()
    if precision in {"fp32", "float32", "32"}:
        return torch.float32
    if precision in {"fp16", "float16", "16"}:
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


@contextlib.contextmanager
def autocast_context(device: torch.device, precision: str) -> Iterator[None]:
    dtype = dtype_from_precision(precision)
    enabled = dtype != torch.float32 and device.type in {"cuda", "npu"}
    if not enabled:
        yield
        return
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    elif device.type == "npu":
        with torch.autocast(device_type="npu", dtype=dtype):
            yield
    else:
        yield


def make_grad_scaler(device: torch.device, precision: str):
    precision = (precision or "fp32").lower()
    if device.type == "cuda" and precision in {"fp16", "float16", "16"}:
        return torch.cuda.amp.GradScaler(enabled=True)
    return _NullGradScaler()


class _NullGradScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None

