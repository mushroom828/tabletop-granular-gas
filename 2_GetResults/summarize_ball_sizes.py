"""Summarize per-ball YOLO box sizes exported by measure_ball_sizes.py.

mean_width_px is mean box width (x span) and mean_height_px is mean box height
(y span) in the rotated YOLO input frame.
The final summary pools all valid ball detections, as in the original script.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "processed_data" / "ball_sizes_means.csv"
FILE_PATTERN = re.compile(r"^(\d+)r_(\d+)_ball_sizes\.xlsx$", re.IGNORECASE)


def _find_column(columns, keywords):
    for column in columns:
        name = str(column).strip().lower()
        if any(keyword in name for keyword in keywords):
            return column
    return None


def _read_valid_sizes(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path)
    height_column = _find_column(data.columns, ("height", "length", "长"))
    width_column = _find_column(data.columns, ("width", "宽"))
    if height_column is None or width_column is None:
        raise ValueError(
            f"{path.name}: expected Height/Length and Width columns; "
            f"found {list(data.columns)}"
        )

    # Older notebook files contain a video-name row. It is not a measurement.
    sizes = pd.DataFrame({
        "height": pd.to_numeric(data[height_column], errors="coerce"),
        "width": pd.to_numeric(data[width_column], errors="coerce"),
    })
    sizes = sizes.replace([np.inf, -np.inf], np.nan).dropna()
    sizes = sizes[(sizes["height"] > 0) & (sizes["width"] > 0)]
    if sizes.empty:
        raise ValueError(f"{path.name}: no valid ball box sizes")
    return sizes


def create_summary(input_dir: Path, output_csv: Path, overwrite: bool = False) -> int:
    input_dir = input_dir.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Ball-size Excel directory does not exist: {input_dir}")
    files = sorted(
        (path for path in input_dir.glob("*_ball_sizes.xlsx")
         if FILE_PATTERN.fullmatch(path.name)),
        key=lambda path: tuple(map(int, FILE_PATTERN.fullmatch(path.name).groups())),
    )
    if not files:
        raise FileNotFoundError(f"No named ball-size Excel files found in {input_dir}")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"{output_csv} already exists; use --overwrite to replace it")

    rows = []
    total_height = total_width = 0.0
    total_detections = 0
    for path in files:
        sizes = _read_valid_sizes(path)
        count = len(sizes)
        height_mean = float(sizes["height"].mean())
        width_mean = float(sizes["width"].mean())
        rows.append({
            "filename": path.name,
            "mean_width_px": round(width_mean, 2),
            "mean_height_px": round(height_mean, 2),
        })
        total_height += float(sizes["height"].sum())
        total_width += float(sizes["width"].sum())
        total_detections += count
        print(f"{path.name}: {count} balls, height={height_mean:.2f} px, "
              f"width={width_mean:.2f} px")

    rows.append({
        "filename": "Overall Summary",
        "mean_width_px": round(total_width / total_detections, 2),
        "mean_height_px": round(total_height / total_detections, 2),
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_name(output_csv.name + ".tmp")
    try:
        pd.DataFrame(rows, columns=["filename", "mean_width_px", "mean_height_px"]).to_csv(
            temporary_csv, index=False, encoding="utf-8-sig"
        )
        temporary_csv.replace(output_csv)
    finally:
        temporary_csv.unlink(missing_ok=True)
    print(f"Saved {len(files)} conditions and {total_detections} detections to {output_csv}")
    analysis_csv = DEFAULT_OUTPUT_CSV.resolve()
    if output_csv != analysis_csv:
        print(f"Note: analysis figures still read {analysis_csv}; publish the complete "
              "calibration table there when it is ready.")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ball_sizes_means.csv from per-video ball-size Excel files"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=SCRIPT_DIR / "ball_size_measurements",
        help="Directory containing <rpm>r_<balls>_ball_sizes.xlsx",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_CSV,
        help="Output CSV (default: 2_GetResults/processed_data/ball_sizes_means.csv)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacement of an existing output CSV",
    )
    args = parser.parse_args()
    try:
        create_summary(args.input_dir, args.output, args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
