import numpy as np
import pandas as pd

CSV_PATH = r"C:\BISE data\Juin\Run Bise 02.06\3\3_Particle_analysis_Essai_7_4x10mm3_run_1_freqacq_8000Hz_512_640.csv"
ACQUISITION_FREQUENCY = 8000
TIME_INTERVAL = 1 / ACQUISITION_FREQUENCY

# The GUI recomputes velocity after sorting each track by frame. Do the same
# before packing so notebook users do not depend on the misaligned CSV column.
cols_needed = ["frame", "label", "x", "y", "dt"]
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

packed_columns = [
    "frame", "time", "label", "x", "y", "dt", "dx", "dy",
    "vx", "vy", "velocity", "ax", "ay", "acceleration",
]
df[packed_columns].to_parquet(
    "velocity_data.parquet",
    compression="zstd",
    index=False,
)

print(f"Packed {len(df):,} rows and {df['label'].nunique():,} labels")