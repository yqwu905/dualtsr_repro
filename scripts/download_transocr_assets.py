from __future__ import annotations

import argparse
from pathlib import Path


FUDAN_BASELINE_FOLDER = "https://drive.google.com/drive/folders/14v3hHhq4AOVEYY1hQfA1d1vI2HtZQZey"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TransOCR weights from the official FudanVI folder.")
    parser.add_argument("--output", default="weights/transocr")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def is_transocr_asset(path: str) -> bool:
    return "transocr" in path.lower()


def main() -> None:
    args = parse_args()
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt (gdown is required for this script).") from exc

    files = gdown.download_folder(
        url=FUDAN_BASELINE_FOLDER,
        skip_download=True,
        remaining_ok=True,
        quiet=False,
    )
    matches = [file for file in files or [] if is_transocr_asset(file.path)]
    if not matches:
        raise RuntimeError("No TransOCR files were found in the official baseline folder.")
    for file in matches:
        print(f"{file.id}\t{file.path}")
    if args.list_only:
        return

    output_dir = Path(args.output)
    for file in matches:
        destination = output_dir / file.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(id=file.id, output=str(destination), quiet=False)
        if result is None:
            raise RuntimeError(f"Failed to download {file.path}")


if __name__ == "__main__":
    main()
