// model_config.h -- single source of truth for the deployed model.
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H
#define NUM_AXES        3      // X, Y, Z
#define WINDOW_SIZE     200    // readings per axis per sample (Member 1: 200 x 3 CSV)
// MAD = median(|x - median(x)|), RAW. No 1.4826 scale and not 128 readings
// (the upstream Shawn Hymel sketch differs; Member 1/2 trained on raw MAD of 200).
#define THRESHOLD_FP32  4.8191197e-07f   // Member 2 threshold.txt (95th pct of validation MSE)
// error = mean((mad - reconstruction)^2) over 3 features; error > threshold => ANOMALY
#define THRESHOLD_INT8  4.825775e-07f
#ifndef USE_INT8
#define USE_INT8 1
#endif
#if USE_INT8
#define ANOMALY_THRESHOLD THRESHOLD_INT8
#else
#define ANOMALY_THRESHOLD THRESHOLD_FP32
#endif
#endif
