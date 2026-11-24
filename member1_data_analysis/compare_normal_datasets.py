
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"

CANDIDATES = {
    "deployment_normal": DATASET_ROOT / "ceiling-fan-deploy" / "fan_0_low-deploy",
    "moving_normal": DATASET_ROOT / "ceiling-fan" / "fan_0_low_0_weight-moving",
}


def extract_mad(sample):
    """Calculate one MAD feature for each of the three axes."""
    return np.median(
        np.abs(sample - np.median(sample, axis=0)),
        axis=0
    )


for name, folder in CANDIDATES.items():
    features = []

    for file_path in sorted(folder.glob("*.csv")):
        sample = pd.read_csv(file_path, header=None).to_numpy(dtype=float)

        if sample.shape != (200, 3):
            continue

        features.append(extract_mad(sample))

    features = np.asarray(features)

    print(f"\n{name}")
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
