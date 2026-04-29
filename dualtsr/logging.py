from __future__ import annotations

from pathlib import Path


class NullSummaryWriter:
    enabled = False

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "tensorboard_missing.txt").write_text(
            "Install tensorboard to enable real TensorBoard event logging.\n",
            encoding="utf-8",
        )

    def add_scalar(self, *args, **kwargs):
        return None

    def add_image(self, *args, **kwargs):
        return None

    def add_images(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def make_summary_writer(log_dir: str | Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return NullSummaryWriter(log_dir)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.enabled = True
    return writer

