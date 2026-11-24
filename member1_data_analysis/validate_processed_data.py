
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_ROOT
    / "member1_data_analysis"
    / "processed_data"
    / "fan_dataset.npz"
)

data = np.load(DATA_FILE)

print("========== DATASET VALIDATION ==========")

array_names = [
    "X_train",
    "X_val",
    "X_test_normal",
    "X_test_anomaly",
]

for name in array_names:
    arr = data[name]

    print(f"\n{name}:")
    print("Shape:", arr.shape)
    print("Data type:", arr.dtype)
    print("All finite:", np.isfinite(arr).all())
    print("Minimum:", arr.min())
    print("Maximum:", arr.max())

    assert arr.ndim == 2
    assert arr.shape[1] == 3
    assert np.isfinite(arr).all()

# Verify that normal splits have no filename overlap.
train = set(data["train_files"].tolist())
val = set(data["val_files"].tolist())
test = set(data["test_normal_files"].tolist())

assert train.isdisjoint(val)
assert train.isdisjoint(test)
assert val.isdisjoint(test)

print("\nFilename overlap check: PASSED")

# Verify anomaly files are separate from normal files.
anomaly = set(data["test_anomaly_files"].tolist())
assert anomaly.isdisjoint(train | val | test)

print("Anomaly separation check: PASSED")
print("\nALL VALIDATION CHECKS PASSED")
