from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from dualtsr.config import load_config
from dualtsr.device import resolve_device
from dualtsr.metrics import load_image_tensor, maybe_fid, maybe_lpips, ned, psnr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DualTSR outputs")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    pred_rows = read_rows(eval_cfg["predictions"])
    gt_rows = {str(row.get("id", i)): row for i, row in enumerate(read_rows(eval_cfg["ground_truth"]))}
    size = eval_cfg.get("hr_size", config.get("data", {}).get("hr_size"))
    psnrs: list[float] = []
    accs: list[float] = []
    neds: list[float] = []
    pred_imgs: list[torch.Tensor] = []
    gt_imgs: list[torch.Tensor] = []
    for row in pred_rows:
        gt = gt_rows.get(str(row.get("id")))
        if gt is None:
            continue
        pred_img = load_image_tensor(row["image"], size)
        gt_path = gt.get("hr") or gt.get("gt") or gt.get("image")
        gt_img = load_image_tensor(gt_path, size)
        pred_imgs.append(pred_img)
        gt_imgs.append(gt_img)
        psnrs.append(psnr(pred_img, gt_img))
        if "text" in row and "text" in gt:
            accs.append(float(row["text"] == gt["text"]))
            neds.append(ned(row["text"], gt["text"]))

    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    result = {
        "psnr": sum(psnrs) / max(len(psnrs), 1),
        "num_images": len(psnrs),
        "lpips": maybe_lpips(pred_imgs, gt_imgs, device),
        "fid": maybe_fid(pred_imgs, gt_imgs, device),
        "acc": (sum(accs) / len(accs)) if accs else None,
        "ned": (sum(neds) / len(neds)) if neds else None,
    }
    if result["lpips"] is None:
        result["lpips_note"] = "Skipped: install lpips and model weights to enable LPIPS."
    if result["fid"] is None:
        result["fid_note"] = "Skipped: install torchmetrics[image] and torch-fidelity to enable FID."
    if result["acc"] is None or result["ned"] is None:
        result["text_metric_note"] = "Skipped: provide predictions with text and ground-truth text, or configure external TransOCR inference."
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

