
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


features_by_class = {}

for label, folder in DATASETS.items():
    features = []

    for file_path in sorted(folder.glob("*.csv")):
        sample = pd.read_csv(file_path, header=None).to_numpy(dtype=float)

        if sample.shape == (200, 3):
            features.append(extract_mad(sample))

    features = np.asarray(features)
    features_by_class[label] = features

    print(f"\n{label}")
    print(f"Valid samples: {len(features)}")

    for axis, index in zip(["X", "Y", "Z"], range(3)):
        values = features[:, index]
        print(
            f"{axis}: mean={np.mean(values):.6f}, "
            f"median={np.median(values):.6f}, "
            f"std={np.std(values):.6f}, "
            f"min={np.min(values):.6f}, "
            f"max={np.max(values):.6f}"
        )


# Compare feature distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for index, axis in enumerate(["X", "Y", "Z"]):
    axes[index].boxplot(
        [
            features_by_class["Normal"][:, index],
            features_by_class["Anomaly"][:, index]
        ],
        tick_labels=["Normal", "Anomaly"]
    )
    axes[index].set_title(f"{axis}-axis MAD")
    axes[index].set_ylabel("MAD")

plt.suptitle("Normal vs Anomaly Feature Distribution")
plt.tight_layout()
plt.show()
