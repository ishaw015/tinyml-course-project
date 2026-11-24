# TinyML Fan Anomaly Detection – Dataset Preprocessing Plan

## 1. Objective

Prepare consistent accelerometer data for training a normal-only Autoencoder and deploying it on ESP32.

## 2. Dataset Selection

### Normal samples
`datasets/ceiling-fan-deploy/fan_0_low-deploy`

### Primary anomaly samples
`datasets/ceiling-fan/fan_0_low_1_weight`

### Additional anomaly conditions for evaluation
- `fan_0_med_0_weight`
- `fan_0_high_0_weight`
- `fan_0_med_1_weight`
- `fan_0_high_1_weight`

Different fan speeds will be evaluated separately from the primary low-speed anomaly condition.

## 3. Input Format

- 200 accelerometer readings per sample
- Three axes: X, Y and Z
- Original sample shape: 200 × 3

## 4. Feature Extraction

Calculate Median Absolute Deviation (MAD) independently for each axis.

Each sample produces three features:

- X-axis MAD
- Y-axis MAD
- Z-axis MAD

Final feature vector shape: 3

## 5. Model Training Strategy

- Train the Autoencoder using normal samples only.
- Keep separate training, validation and test samples.
- Evaluate primary anomalies separately.
- Evaluate additional operating speeds as separate conditions.
- Use reconstruction error for anomaly detection.

## 6. Deployment Consistency

The Python preprocessing pipeline and ESP32 implementation must use exactly the same:
- Input window length
- Axis ordering
- MAD calculation
- Numerical representation
- Model input dimensions

The planned demonstration transfers 200 readings per axis sample from Python to ESP32 over USB serial.

## 7. Status

This is the proposed initial configuration. Final dataset selection and preprocessing must be confirmed with the model-development and hardware team members before implementation.