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
    ocr_rows = None
    if eval_cfg.get("ocr_predictions"):
        ocr_rows = {
            str(row.get("id", i)): row
            for i, row in enumerate(read_rows(eval_cfg["ocr_predictions"]))
        }
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
        text_row = ocr_rows.get(str(row.get("id"))) if ocr_rows is not None else row
        if text_row is not None and "text" in text_row and "text" in gt:
            accs.append(float(text_row["text"] == gt["text"]))
            neds.append(ned(text_row["text"], gt["text"]))

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
        result["text_metric_note"] = (
            "Skipped: provide evaluation.ocr_predictions from TransOCR and ground-truth text. "
            "Falling back to DualTSR's internal text output is supported for diagnostics but is not the paper protocol."
        )
    elif ocr_rows is not None:
        result["text_metric_source"] = str(eval_cfg["ocr_predictions"])
    else:
        result["text_metric_source"] = "DualTSR internal text output (diagnostic; paper uses TransOCR)"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
