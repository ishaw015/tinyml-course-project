# Feature Analysis Report – TinyML Fan Anomaly Detection

## 1. Objective

The objective of this analysis was to investigate the vibration characteristics of normal and anomalous ceiling-fan operating conditions using accelerometer data.

The analysis explored statistical and frequency-domain features to identify patterns that could help distinguish normal fan behaviour from abnormal behaviour.

## 2. Dataset Configuration

The analysis used the following dataset folders:

### Normal data
- `ceiling-fan-deploy/fan_0_low-deploy`

### Anomaly data
- `ceiling-fan/fan_0_med_0_weight`
- `ceiling-fan/fan_0_high_0_weight`
- `ceiling-fan/fan_0_low_1_weight`
- `ceiling-fan/fan_0_med_1_weight`
- `ceiling-fan/fan_0_high_1_weight`

Each CSV sample contains:
- 200 accelerometer readings
- 3 axes: X, Y and Z
- 600 numerical values in total

For the feature visualisations, 200 normal and 200 anomaly samples were selected.

## 3. Statistical Feature Analysis

### 3.1 Mean after DC Removal

The mean of each axis was calculated after subtracting that sample's mean from its readings.

**Observation:**
Both normal and anomaly samples clustered very close to zero.

**Conclusion:**
The mean after per-sample DC removal does not provide useful separation between the two classes. This is expected because centring each sample forces its axis-wise mean to approximately zero.

### 3.2 Variance

Variance was calculated independently for the X, Y and Z axes.

**Observation:**
Normal samples formed a relatively compact cluster, while anomaly samples showed a wider spread and several distinct clusters.

**Conclusion:**
Variance appears promising for distinguishing normal and anomalous vibration patterns because the weighted fan conditions exhibit larger vibration fluctuations in many samples.

### 3.3 Kurtosis

Kurtosis was calculated independently for each axis using SciPy's `stats.kurtosis()` function.

**Observation:**
Some anomaly samples showed substantially higher positive kurtosis values than normal samples. However, the two classes also showed overlap.

**Conclusion:**
Kurtosis may help identify samples with unusually pronounced extreme values, but its usefulness requires further evaluation.

### 3.4 Skewness

Skewness was calculated independently for each axis using SciPy's `stats.skew()` function.

**Observation:**
The normal and anomaly samples showed considerable overlap, although some anomaly samples exhibited more extreme skewness values.

**Conclusion:**
Skewness showed limited visual separation in this analysis.

### 3.5 Median Absolute Deviation (MAD)

MAD was calculated independently for each axis using the median absolute deviation formula:

`MAD = median(abs(x - median(x)))`

**Observation:**
Normal samples formed a relatively compact cluster, whereas anomaly samples showed higher and more widely distributed MAD values.

**Conclusion:**
MAD appears promising for distinguishing normal and anomalous vibration patterns. It is also less sensitive to extreme individual readings than standard deviation.

### 3.6 Correlation Between Accelerometer Axes

Correlation matrices were calculated for each sample after DC removal.

The three unique axis relationships were examined:
- X–Y
- X–Z
- Y–Z

**Observation:**
The histograms showed differences between normal and anomaly correlation distributions, particularly for X–Y and Y–Z. Some overlap remained.

**Conclusion:**
Relationships between accelerometer axes may provide additional information for identifying abnormal fan behaviour.

## 4. Frequency-Domain Analysis Using FFT

### 4.1 FFT Feature Extraction

A Fast Fourier Transform (FFT) was calculated independently for each accelerometer axis.

The extraction process:
1. Applied a Hann window to the signal.
2. Calculated the real-valued FFT using `np.fft.rfft()`.
3. Extracted the magnitude of the frequency components.
4. Removed the DC component.

Each input sample had a shape of `(200, 3)`, and the resulting FFT feature array had a shape of `(100, 3)`.

### 4.2 Average FFT Comparison

FFT features were calculated for 200 normal and 200 anomaly samples. The average FFT magnitude was then calculated separately for each class.

**Observation:**
The average FFT plots showed noticeably larger anomaly magnitudes at selected frequency bins, particularly around bins 78 and 91 across the three axes.

**Conclusion:**
The weighted fan conditions exhibit different frequency-domain patterns. FFT features appear promising for capturing frequency-specific vibration changes.

Note: The plotted horizontal axis represents frequency-bin positions rather than explicitly labelled frequencies in Hz.

## 5. Overall Findings

| Feature | Visual observation | Initial assessment |
|---|---|---|
| Mean after DC removal | Both classes approximately zero | Not useful |
| Variance | Clear cluster separation | Promising |
| Kurtosis | Some separation, with overlap | Potentially useful |
| Skewness | Considerable overlap | Limited |
| MAD | Clear cluster separation | Promising |
| Axis correlation | Differences in correlation distributions | Potentially useful |
| FFT | Strong differences at selected bins | Promising |

## 6. Conclusion

The feature analysis indicates that normal and weighted-fan operating conditions exhibit distinguishable vibration characteristics.

Variance and MAD showed promising separation in the visualisations. FFT analysis also revealed differences at selected frequency bins. Correlation and kurtosis may provide additional useful information.

These observations provide a foundation for subsequent model development and evaluation.

However, the plots are exploratory results based on selected samples. They do not establish final feature importance or classification performance. Further evaluation on appropriate training, validation and test datasets is required.

## 7. Handover to Model Development

The following findings can be considered during model development:

- Investigate vibration magnitude differences using variance and MAD.
- Explore frequency-domain patterns using FFT.
- Consider relationships between X, Y and Z axes.
- Do not rely on the mean after per-sample DC removal as a discriminative feature.
- Validate all candidate features using held-out data before deciding which are useful.

The preprocessing and feature-extraction implementation used during training must also be documented so that the same operations can be reproduced during ESP32 deployment.