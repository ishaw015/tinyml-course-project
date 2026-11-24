import os
import csv
import math
import hashlib
import statistics

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATHS = {
    "main": PROJECT_ROOT / "datasets" / "ceiling-fan",
    "deployment": PROJECT_ROOT / "datasets" / "ceiling-fan-deploy"
}

def inspect_dataset(name, root):
    print(f"\n{'=' * 60}")
    print(f"DATASET: {name.upper()}")
    print(f"{'=' * 60}")
    print(f"Dataset path: {root}")
    print(f"Path exists: {root.exists()}")

    total_files = 0
    invalid_rows = 0
    non_numeric = 0
    non_finite = 0
    missing_values = 0

    hashes = {}
    all_values = [[], [], []]

    for folder, _, files in os.walk(str(root)):
        for filename in files:
            if not filename.endswith(".csv"):
                continue

            filepath = os.path.join(folder, filename)
            total_files += 1

            sample_hash = hashlib.sha256()

            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            sample_values = [[], [], []]

            for row in rows:
                if len(row) != 3:
                    invalid_rows += 1
                    continue

                for axis in range(3):
                    value = row[axis].strip()

                    if value == "":
                        missing_values += 1
                        continue

                    try:
                        number = float(value)
                    except ValueError:
                        non_numeric += 1
                        continue

                    if not math.isfinite(number):
                        non_finite += 1
                        continue

                    sample_values[axis].append(number)
                    all_values[axis].append(number)

            # Identify exact duplicate samples
            sample_hash.update(
                open(filepath, "rb").read()
            )
            digest = sample_hash.hexdigest()

            hashes[digest] = hashes.get(digest, 0) + 1

    duplicate_files = sum(
        count - 1 for count in hashes.values() if count > 1
    )

    print(f"Total CSV files: {total_files}")
    print(f"Invalid rows: {invalid_rows}")
    print(f"Missing values: {missing_values}")
    print(f"Non-numeric values: {non_numeric}")
    print(f"NaN / Infinity values: {non_finite}")
    print(f"Duplicate files: {duplicate_files}")

    print("\nAXIS STATISTICS")

    for i, axis in enumerate(["X", "Y", "Z"]):
        values = all_values[i]

        if values:
            print(f"\n{axis}-axis:")
            print(f"  Total readings: {len(values)}")
            print(f"  Minimum: {min(values):.6f}")
            print(f"  Maximum: {max(values):.6f}")
            print(f"  Mean: {statistics.mean(values):.6f}")
            print(f"  Std Dev: {statistics.pstdev(values):.6f}")

if __name__ == "__main__":
    for name, path in DATASET_PATHS.items():
        inspect_dataset(name, path)
