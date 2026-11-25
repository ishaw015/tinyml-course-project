// fan_int8.h -- integer-only forward pass matching the TFLite INT8 model
#ifndef FAN_INT8_H
#define FAN_INT8_H
#include <math.h>
#include <stdint.h>
#include "model_int8.h"

static int32_t fi_clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// round half to even (same as numpy.round)
static int32_t fi_rne(float x) {
  float f = floorf(x), d = x - f;
  int32_t i = (int32_t)f;
  if (d > 0.5f) return i + 1;
  if (d < 0.5f) return i;
  return (i & 1) ? i + 1 : i;
}

static float fc_mse_int8(const float mad[3]) {
  int32_t xq[3], h[2];
  for (int i = 0; i < 3; i++) xq[i] = fi_clamp(fi_rne(mad[i] / Q_S_IN) + Q_ZP_IN, -128, 127);
  for (int j = 0; j < 2; j++) {
    int64_t a = 0;
    for (int i = 0; i < 3; i++) a += (int64_t)(xq[i] - Q_ZP_IN) * Q_W1[j * 3 + i];
    h[j] = fi_clamp(fi_rne((float)a * Q_M1[j]), 0, 127);
  }
  float mse = 0;
  for (int k = 0; k < 3; k++) {
    int64_t a = Q_B2[k];
    for (int j = 0; j < 2; j++) a += (int64_t)h[j] * Q_W2[k * 2 + j];
    int32_t yq = fi_clamp(fi_rne((float)a * Q_M2[k]) + Q_ZP_OUT, -128, 127);
    float d = mad[k] - (float)(yq - Q_ZP_OUT) * Q_S_OUT;
    mse += d * d;
  }
  return mse / 3.0f;
}
#endif
