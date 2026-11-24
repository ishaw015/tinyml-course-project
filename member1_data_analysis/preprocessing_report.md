
# Dataset Preprocessing Report
## TinyML Fan Anomaly Detection

### 1. Objective

The objective of this preprocessing stage is to convert raw accelerometer readings into numerical features suitable for training and evaluating an Autoencoder-based anomaly detection model.

Each original CSV contains 200 readings across three accelerometer axes (X, Y, Z).

### 2. Dataset Selection

Two datasets were selected:

| Category | Dataset | Samples |
|---|---|---:|
| Normal | ceiling-fan-deploy/fan_0_low-deploy | 286 |
| Anomaly | ceiling-fan/fan_0_low_1_weight | 580 |

The deployment dataset was selected as the normal dataset because its MAD features showed more consistent distributions than the alternative moving-normal dataset.

The weighted low-speed fan dataset was selected as the primary anomaly dataset.

### 3. Feature Extraction

The Median Absolute Deviation (MAD) was calculated independently for each accelerometer axis.

For each axis:

1. Calculate the median of the readings.
2. Calculate the absolute deviation of each reading from the median.
3. Calculate the median of those absolute deviations.

Formula:

MAD(X) = median(|X - median(X)|)

Each 200 × 3 input sample is converted into three features:

[MAD_X, MAD_Y, MAD_Z]

The resulting feature vector has shape (3,).

### 4. Dataset Splitting

Only normal samples were divided into training, validation and testing sets.

| Split | Percentage | Samples |
|---|---:|---:|
| Training | 60% | 171 |
| Validation | 20% | 57 |
| Normal testing | 20% | 58 |
| Anomaly testing | Separate | 580 |

Random seed: 42

Anomaly samples were not included in Autoencoder training.

### 5. Processed Dataset Structure

The processed dataset is saved as:

member1_data_analysis/processed_data/fan_dataset.npz

It contains:

| Array | Shape | Purpose |
|---|---|---|
| X_train | (171, 3) | Normal training features |
| X_val | (57, 3) | Normal validation features |
| X_test_normal | (58, 3) | Normal test features |
| X_test_anomaly | (580, 3) | Anomaly test features |

Source filenames are also stored for traceability.

### 6. Validation Results

The following checks were completed:

- All selected CSV files had the expected shape (200, 3).
- All processed feature arrays had three columns.
- All feature values were finite.
- No overlap was found between normal training, validation and test files.
- No overlap was found between normal and anomaly source files.
- The processed dataset was successfully saved and reloaded.

All validation checks passed.

### 7. Handover Notes for Model Development

The processed dataset is ready for the model development stage.

Important implementation details:

- Model input dimension: 3
- Input features: MAD_X, MAD_Y, MAD_Z
- Data type: float32
- Training data: normal samples only
- Validation data: normal samples only
- Evaluation data: normal and weighted-fan anomaly samples

The model should learn to reconstruct normal feature vectors. Reconstruction error can then be used to identify anomalous samples.

### 8. Important Deployment Requirement

The ESP32 implementation must reproduce the same preprocessing calculation used in Python.

For every input window:

1. Collect 200 readings from each of the three axes.
2. Calculate the median for each axis.
3. Calculate the MAD independently for each axis.
4. Pass the three MAD values to the model.

The model must receive features in the same order:

[MAD_X, MAD_Y, MAD_Z]

Any difference in window length, axis ordering or MAD calculation may affect inference results.

### 9. Current Scope and Limitations

This report covers dataset inspection, feature analysis and preprocessing.

The following are outside this stage:

- Autoencoder training and threshold selection.
- Isolation Forest baseline evaluation.
- Quantization-aware or post-training quantization.
- Pruning and model conversion.
- ESP32 integration and hardware testing.

These tasks belong to the subsequent model development and deployment stages.
