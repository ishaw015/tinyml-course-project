
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"

DATASETS = {
    "Normal": DATASET_ROOT / "ceiling-fan-deploy" / "fan_0_low-deploy",
    "Anomaly": DATASET_ROOT / "ceiling-fan" / "fan_0_low_1_weight",
}


def extract_mad(sample):
    return np.median(
        np.abs(sample - np.median(sample, axis=0)),
        axis=0
    )


features = {}

for label, folder in DATASETS.items():
    values = []

    for file_path in folder.glob("*.csv"):
        sample = pd.read_csv(file_path, header=None).to_numpy(dtype=float)

        if sample.shape == (200, 3):
            values.append(extract_mad(sample))

    features[label] = np.asarray(values)


normal = features["Normal"]
anomaly = features["Anomaly"]

for i, axis in enumerate(["X", "Y", "Z"]):
    n = normal[:, i]
    a = anomaly[:, i]

    # Midpoint between class medians, used only as an exploratory threshold
    threshold = (np.median(n) + np.median(a)) / 2

    normal_above = np.mean(n > threshold) * 100
    anomaly_below = np.mean(a < threshold) * 100

    print(f"\n{axis}-axis")
    print(f"Exploratory threshold: {threshold:.6f}")
    print(f"Normal samples above threshold: {normal_above:.2f}%")
    print(f"Anomaly samples below threshold: {anomaly_below:.2f}%")
