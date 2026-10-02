import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the main window
CSV_PATH = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
)
parquet_output_name = input("Enter output Parquet file name (default: velocity_data.parquet): ")
ACQUISITION_FREQUENCY = 8000
TIME_INTERVAL = 1 / ACQUISITION_FREQUENCY

# The GUI recomputes velocity after sorting each track by frame. Do the same
# before packing so notebook users do not depend on the misaligned CSV column.
cols_needed = ["frame", "label", "x", "y", "dt", "cluster_id"]
chunks = pd.read_csv(CSV_PATH, usecols=cols_needed, chunksize=500_000)
df = pd.concat(chunks, ignore_index=True)
df = df.sort_values(["label", "frame"]).reset_index(drop=True)

df["dt"] = TIME_INTERVAL
df["dx"] = df.groupby("label")["x"].diff().fillna(0.0)
df["dy"] = df.groupby("label")["y"].diff().fillna(0.0)
df["vx"] = df["dx"] / df["dt"]
df["vy"] = df["dy"] / df["dt"]
df["velocity"] = np.hypot(df["dx"], df["dy"]) / df["dt"]
df["ax"] = df.groupby("label")["vx"].diff().fillna(0.0) / df["dt"]
df["ay"] = df.groupby("label")["vy"].diff().fillna(0.0) / df["dt"]
df["acceleration"] = (
    df.groupby("label")["velocity"].diff().fillna(0.0) / df["dt"]
)
df["time"] = df["frame"] * TIME_INTERVAL
df["cluster_id"] = df["cluster_id"].fillna(0.0)

packed_columns = [
    "frame", "time", "label", "x", "y", "dt", "dx", "dy",
    "vx", "vy", "velocity", "ax", "ay", "acceleration",
    "cluster_id"
]
df[packed_columns].to_parquet(
    parquet_output_name or "velocity_data.parquet",
    compression="zstd",
    index=False,
)

print(f"Packed {len(df):,} rows and {df['label'].nunique():,} labels")