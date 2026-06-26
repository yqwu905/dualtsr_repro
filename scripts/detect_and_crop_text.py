#!/usr/bin/env python3
"""从大图中检测文字区域，裁剪并填充到固定 128x512 横图。

使用 PaddleOCR PP-OCRv6 文字检测器。对每个检测到的文字框，按原始宽高比
缩放，短边居中填充到 128x512 (H=128, W=512)，不产生拉伸变形，贴合 DualTSR
训练数据的 HR 尺寸与宽高比分布。

依赖安装（不在项目 requirements.txt 中，需单独安装）：
    pip install paddleocr onnxruntime

默认使用 onnxruntime 推理引擎，绕过 PaddlePaddle 静态图引擎在 arm64 (鲲鹏/Apple
Silicon) 上已知的段错误 (PaddlePaddle issue #78744: SaveOrLoadPirParameters 中
std::filesystem::path 析构崩溃)。如需改用 paddle 引擎，通过 --engine 指定，但
arm64 上可能触发上述崩溃。

首次运行会自动下载 PP-OCRv6 检测模型权重；如网络访问不便可设置环境变量
`PADDLE_PDX_MODEL_SOURCE=BOS` 切换到百度 BOS 源。

用法:
    # 单张图片
    python3 scripts/detect_and_crop_text.py --input big.jpg --output crops/

    # 整个目录
    python3 scripts/detect_and_crop_text.py --input /path/to/images --output crops/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from paddleocr import TextDetection


TARGET_H = 128
TARGET_W = 512
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect text regions and crop to 128x512.")
    p.add_argument("--input", required=True, help="输入图片路径或目录。")
    p.add_argument("--output", required=True, help="输出目录。")
    p.add_argument("--model-name", default="PP-OCRv6_medium_det",
                   help="PP-OCRv6 检测模型名 (默认 PP-OCRv6_medium_det；可选 PP-OCRv6_small_det / PP-OCRv6_tiny_det)。")
    p.add_argument("--engine", default="onnxruntime",
                   help="推理引擎 (默认 onnxruntime，arm64 推荐；可选 paddle / paddle_static / paddle_dynamic)。")
    p.add_argument("--device", default=None, help="推理设备: cpu / gpu / gpu:0。默认自动选择。")
    p.add_argument("--box-thresh", type=float, default=0.6, help="检测框阈值 (默认 0.6)。")
    p.add_argument("--thresh", type=float, default=0.3, help="像素得分阈值 (默认 0.3)。")
    p.add_argument("--unclip-ratio", type=float, default=2.0, help="框扩张系数 (默认 2.0)。")
    p.add_argument("--margin-ratio", type=float, default=0.1,
                   help="裁剪时相对 bbox 尺寸的外扩比例 (默认 0.1)。")
    p.add_argument("--min-box-side", type=int, default=8,
                   help="过滤短边小于该像素的框 (默认 8)。")
    p.add_argument("--pad-color", default="255,255,255",
                   help="填充色 R,G,B (默认 255,255,255 白色)。")
    p.add_argument("--save-vis", action="store_true",
                   help="额外保存带检测框的可视化图 (<stem>_detvis.png)。")
    return p.parse_args()


def list_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [path]


def extract_polys(res) -> list:
    """从 TextDetection 结果中安全提取检测框多边形。

    PaddleOCR 3.x 不同引擎/版本下结果结构不一致，兼容以下访问路径：
      - res.json["res"]["dt_polys"]  (onnxruntime 常见，嵌套在 "res" 下)
      - res.json["dt_polys"]         (paddle_static 部分版本)
      - res["dt_polys"]              (直接 dict 访问)
    """
    candidates = []
    json_attr = getattr(res, "json", None)
    if isinstance(json_attr, dict):
        inner = json_attr.get("res")
        if isinstance(inner, dict):
            candidates.append(inner.get("dt_polys"))
        candidates.append(json_attr.get("dt_polys"))
    if isinstance(res, dict):
        candidates.append(res.get("dt_polys"))
    for poly in candidates:
        if poly is None:
            continue
        polys = np.asarray(poly)
        if polys.size == 0:
            continue
        return [np.asarray(p) for p in polys]
    return []


def polys_to_boxes(polys) -> list[tuple[int, int, int, int]]:
    """把 4 点多边形转成轴对齐矩形 (x1, y1, x2, y2)。"""
    boxes = []
    for poly in polys:
        poly = np.asarray(poly)
        boxes.append((int(poly[:, 0].min()), int(poly[:, 1].min()),
                      int(poly[:, 0].max()), int(poly[:, 1].max())))
    return boxes


def expand_box(box, margin_ratio, img_w, img_h):
    """按比例外扩 bbox 并裁剪到原图边界内。"""
    x1, y1, x2, y2 = box
    mx = int((x2 - x1) * margin_ratio)
    my = int((y2 - y1) * margin_ratio)
    return (max(0, x1 - mx), max(0, y1 - my),
            min(img_w, x2 + mx), min(img_h, y2 + my))


def crop_to_target(image: Image.Image, box, target_h: int, target_w: int, pad_color):
    """裁剪 box 区域，保持宽高比缩放，短边居中填充到 target_h x target_w。"""
    crop = image.crop(box)
    cw, ch = crop.size
    if cw <= 0 or ch <= 0:
        return None
    scale = min(target_w / cw, target_h / ch)
    new_w = max(1, round(cw * scale))
    new_h = max(1, round(ch * scale))
    resized = crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def draw_boxes(image: Image.Image, boxes) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        draw.rectangle(box, outline=(255, 0, 0), width=3)
    return canvas


def main() -> None:
    args = parse_args()
    pad_color = tuple(int(v) for v in args.pad_color.split(","))
    if len(pad_color) != 3:
        raise ValueError("--pad-color 格式应为 R,G,B")

    kwargs = dict(
        model_name=args.model_name,
        engine=args.engine,
        thresh=args.thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )
    if args.device:
        kwargs["device"] = args.device
    detector = TextDetection(**kwargs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for img_path in list_inputs(Path(args.input)):
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        boxes: list[tuple[int, int, int, int]] = []
        for res in detector.predict(str(img_path)):
            boxes.extend(polys_to_boxes(extract_polys(res)))

        kept = [b for b in boxes
                if (b[2] - b[0]) >= args.min_box_side and (b[3] - b[1]) >= args.min_box_side]

        for idx, box in enumerate(kept):
            ebox = expand_box(box, args.margin_ratio, img_w, img_h)
            cropped = crop_to_target(image, ebox, TARGET_H, TARGET_W, pad_color)
            if cropped is None:
                continue
            cropped.save(output_dir / f"{img_path.stem}_{idx:03d}.png")
            total += 1

        if args.save_vis and kept:
            draw_boxes(image, kept).save(output_dir / f"{img_path.stem}_detvis.png")

        print(f"{img_path.name}: 检测到 {len(kept)} 个文字区域")

    print(f"完成，共裁剪 {total} 个文字区域，输出到 {output_dir}")


if __name__ == "__main__":
    main()
