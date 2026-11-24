
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "member1_data_analysis" / "processed_data"

NORMAL_DIR = DATASET_ROOT / "ceiling-fan-deploy" / "fan_0_low-deploy"
ANOMALY_DIR = DATASET_ROOT / "ceiling-fan" / "fan_0_low_1_weight"

RANDOM_SEED = 42
EXPECTED_SHAPE = (200, 3)


# --------------------------------------------------
# 2. MAD feature extraction
# --------------------------------------------------

def extract_mad(sample):
    """
    Calculate one Median Absolute Deviation feature
    for each accelerometer axis.
    """
    median = np.median(sample, axis=0)

    mad = np.median(
        np.abs(sample - median),
        axis=0
    )

    return mad


# --------------------------------------------------
# 3. Load dataset and extract features
# --------------------------------------------------

def load_features(folder, label):
    features = []
    filenames = []

    csv_files = sorted(folder.glob("*.csv"))

    for file_path in csv_files:
        sample = pd.read_csv(
            file_path,
            header=None
        ).to_numpy(dtype=np.float64)

        if sample.shape != EXPECTED_SHAPE:
            print(f"Skipping invalid sample: {file_path.name}")
            continue

        if not np.isfinite(sample).all():
            print(f"Skipping non-finite sample: {file_path.name}")
            continue

        features.append(extract_mad(sample))
        filenames.append(str(file_path.relative_to(DATASET_ROOT)))

    features = np.asarray(features, dtype=np.float32)

    print(f"\n{label} dataset")
    print(f"Valid samples: {len(features)}")
    print(f"Feature shape: {features.shape}")

    return features, np.asarray(filenames)


# --------------------------------------------------
# 4. Split normal data: 60% / 20% / 20%
# --------------------------------------------------

def split_normal(features, filenames):
    rng = np.random.default_rng(RANDOM_SEED)

    indices = rng.permutation(len(features))

    train_end = int(0.60 * len(indices))
    val_end = int(0.80 * len(indices))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "X_train": features[train_idx],
        "X_val": features[val_idx],
        "X_test_normal": features[test_idx],
        "train_files": filenames[train_idx],
        "val_files": filenames[val_idx],
        "test_normal_files": filenames[test_idx],
    }


# --------------------------------------------------
# 5. Main preprocessing pipeline
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normal_features, normal_files = load_features(
        NORMAL_DIR, "Normal"
    )

    anomaly_features, anomaly_files = load_features(
        ANOMALY_DIR, "Anomaly"
    )

    splits = split_normal(normal_features, normal_files)

    # Save processed arrays and their source filenames.
    output_path = OUTPUT_DIR / "fan_dataset.npz"

    np.savez_compressed(
        output_path,
        X_train=splits["X_train"],
        X_val=splits["X_val"],
        X_test_normal=splits["X_test_normal"],
        X_test_anomaly=anomaly_features,
        train_files=splits["train_files"],
        val_files=splits["val_files"],
        test_normal_files=splits["test_normal_files"],
        test_anomaly_files=anomaly_files,
    )

    # --------------------------------------------------
    # 6. Print final summary
    # --------------------------------------------------

    print("\n========== PREPROCESSING SUMMARY ==========")

    for name in [
        "X_train",
        "X_val",
        "X_test_normal",
        "X_test_anomaly",
    ]:
        print(f"{name}: {splits[name].shape if name in splits else anomaly_features.shape}")

    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Saved to: {output_path}")
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
