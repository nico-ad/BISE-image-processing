"""Keep only the columns produced by the initial image-detection stage.

Usage:
    python strip_detection_columns.py input.csv output.csv

The input file is read in chunks so the complete CSV is never held in memory.
The original file is not modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DETECTION_COLUMNS = [
    "frame",
    "main_path",
    "name",
    "time",
    "bound_box",
    "coords_pixels",
    "x",
    "y",
    "x_subpixel",
    "y_subpixel",
    "diameter",
    "perimeter",
    "area",
    "circularity",
    "intensity_max",
    "intensity_min",
    "intensity_mean",
    "intensity_std",
    "equivalent_diameter_area",
    "axis_major_length",
    "axis_minor_length",
    "image_width",
    "image_height",
]


def count_csv_rows(input_path: Path) -> int:
    """Count data rows without loading the CSV into memory."""

    newline_count = 0
    with input_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            newline_count += chunk.count(b"\n")

    return max(0, newline_count - 1)


def strip_detection_columns(
    input_path: Path,
    output_path: Path,
    chunksize: int = 10_000,
) -> None:
    """Copy only initial image-detection columns from one CSV to another."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if chunksize < 1:
        raise ValueError("chunksize must be greater than zero")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. Choose another path or remove it first."
        )

    total_rows = count_csv_rows(input_path)
    print(f"Total data rows: {total_rows:,}")
    print("Starting CSV reduction...")

    reader = pd.read_csv(
        input_path,
        usecols=DETECTION_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )

    rows_written = 0
    for chunk_number, chunk in enumerate(reader):
        chunk.to_csv(
            output_path,
            mode="w" if chunk_number == 0 else "a",
            header=chunk_number == 0,
            index=False,
        )
        rows_written += len(chunk)
        print(f"Processed {rows_written:,} rows", flush=True)

    print(f"Saved {rows_written:,} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove post-detection columns from a particle-analysis CSV."
    )
    parser.add_argument("input_csv", type=Path, nargs="?")
    parser.add_argument("output_csv", type=Path, nargs="?")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="Rows per batch; lower this if memory usage is too high.",
    )
    args = parser.parse_args()

    if args.input_csv is None:
        input_text = input("CSV to strip: ").strip().strip('"')
        if not input_text:
            raise SystemExit("No input CSV was provided.")
        args.input_csv = Path(input_text)

    if args.output_csv is None:
        default_output = args.input_csv.with_name(
            f"{args.input_csv.stem}_detection_only.csv"
        )
        output_text = input(f"Output CSV [{default_output}]: ").strip().strip('"')
        args.output_csv = Path(output_text) if output_text else default_output

    print(f"Input:  {args.input_csv}")
    print(f"Output: {args.output_csv}")
    confirmation = input("Create this duplicate? [y/N]: ").strip().lower()
    if confirmation not in {"y", "yes"}:
        raise SystemExit("Cancelled.")

    strip_detection_columns(args.input_csv, args.output_csv, args.chunksize)


if __name__ == "__main__":
    main()
