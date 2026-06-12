from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT_DIR = REPO_ROOT / "assets" / "fonts"

# 全部为可自由再分发的开源字体(SIL OFL 1.1)。
# 中文覆盖:思源黑体/宋体(全量 CJK)、霞鹜文楷(楷体)、站酷系列(美术体)、
# 马善政/智芒星/刘建毛草/龙藏(手写体,约覆盖常用 3500-7000 字)。
# Lato 提供独立的西文风格;CJK 字体本身也覆盖 ASCII。
FONT_MANIFEST: list[dict[str, str]] = [
    {
        "name": "NotoSansCJKsc-Regular.otf",
        "family": "Noto Sans CJK SC (思源黑体)",
        "style": "sans",
        "license": "OFL-1.1",
        "url": "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    },
    {
        "name": "NotoSansCJKsc-Bold.otf",
        "family": "Noto Sans CJK SC (思源黑体)",
        "style": "sans-bold",
        "license": "OFL-1.1",
        "url": "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
    },
    {
        "name": "NotoSerifCJKsc-Regular.otf",
        "family": "Noto Serif CJK SC (思源宋体)",
        "style": "serif",
        "license": "OFL-1.1",
        "url": "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf",
    },
    {
        "name": "LXGWWenKai-Regular.ttf",
        "family": "霞鹜文楷 (LXGW WenKai)",
        "style": "kai",
        "license": "OFL-1.1",
        "url": "https://github.com/lxgw/LxgwWenKai/releases/download/v1.520/LXGWWenKai-Regular.ttf",
    },
    {
        "name": "ZCOOLKuaiLe-Regular.ttf",
        "family": "站酷快乐体 (ZCOOL KuaiLe)",
        "style": "display",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/zcoolkuaile/ZCOOLKuaiLe-Regular.ttf",
    },
    {
        "name": "ZCOOLXiaoWei-Regular.ttf",
        "family": "站酷小薇体 (ZCOOL XiaoWei)",
        "style": "display",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf",
    },
    {
        "name": "ZCOOLQingKeHuangYou-Regular.ttf",
        "family": "站酷庆科黄油体 (ZCOOL QingKe HuangYou)",
        "style": "display",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/zcoolqingkehuangyou/ZCOOLQingKeHuangYou-Regular.ttf",
    },
    {
        "name": "MaShanZheng-Regular.ttf",
        "family": "马善政毛笔楷书 (Ma Shan Zheng)",
        "style": "handwriting",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/mashanzheng/MaShanZheng-Regular.ttf",
    },
    {
        "name": "ZhiMangXing-Regular.ttf",
        "family": "智芒星行书 (Zhi Mang Xing)",
        "style": "handwriting",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/zhimangxing/ZhiMangXing-Regular.ttf",
    },
    {
        "name": "LiuJianMaoCao-Regular.ttf",
        "family": "刘建毛草 (Liu Jian Mao Cao)",
        "style": "handwriting",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/liujianmaocao/LiuJianMaoCao-Regular.ttf",
    },
    {
        "name": "LongCang-Regular.ttf",
        "family": "龙藏体 (Long Cang)",
        "style": "handwriting",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/longcang/LongCang-Regular.ttf",
    },
    {
        "name": "Lato-Regular.ttf",
        "family": "Lato",
        "style": "latin",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf",
    },
    {
        "name": "Lato-Bold.ttf",
        "family": "Lato",
        "style": "latin-bold",
        "license": "OFL-1.1",
        "url": "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf",
    },
]

# TTF/OTF/TTC 文件头魔数,用于下载后做最小有效性检查。
_FONT_MAGIC = {b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download open-licensed fonts for DualTSR pretraining synthesis.")
    parser.add_argument("--out", default=str(DEFAULT_FONT_DIR), help="Font output directory.")
    parser.add_argument("--only", default=None, help="Only download fonts whose name contains this substring.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists.")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def download(url: str, dest: Path, retries: int = 3) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "DualTSR-Repro/0.1"})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                payload = resp.read()
            if payload[:4] not in _FONT_MAGIC:
                raise RuntimeError(f"Not a TTF/OTF payload (first bytes {payload[:4]!r})")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - retry on any network/IO error
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def verify_renderable(path: Path) -> bool:
    """Try loading the font with PIL and rendering one glyph."""
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(str(path), 32)
        return font.getbbox("永A1") is not None
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        entry
        for entry in FONT_MANIFEST
        if args.only is None or args.only.lower() in entry["name"].lower()
    ]
    if not selected:
        print(f"No manifest entry matches --only={args.only}", file=sys.stderr)
        return 2

    results: list[dict[str, str]] = []
    failures: list[str] = []
    for entry in selected:
        dest = out_dir / entry["name"]
        status = "cached"
        if args.force or not dest.exists():
            print(f"downloading {entry['name']} <- {entry['url']}")
            try:
                download(entry["url"], dest, retries=args.retries)
                status = "downloaded"
            except RuntimeError as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                failures.append(entry["name"])
                continue
        if not verify_renderable(dest):
            print(f"  ERROR: {dest} is not loadable by PIL", file=sys.stderr)
            failures.append(entry["name"])
            continue
        size_mb = dest.stat().st_size / 1e6
        print(f"  ok: {entry['name']} ({size_mb:.1f} MB, {status})")
        results.append({**entry, "bytes": str(dest.stat().st_size)})

    manifest_path = out_dir / "fonts.json"
    manifest_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{len(results)}/{len(selected)} fonts ready in {out_dir} (manifest: {manifest_path})")
    if failures:
        print(f"failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
