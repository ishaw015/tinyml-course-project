// fan_core.h -- hardware-independent logic (plain C). Shared by the ESP32 sketch and host_test.
#ifndef FAN_CORE_H
#define FAN_CORE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model_config.h"
#include "model_weights.h"
#include "fan_int8.h"

static int fc_cmp(const void *p, const void *q) {
  float a = *(const float *)p, b = *(const float *)q;
  return (a < b) ? -1 : (a > b);
}
static float fc_median(float *arr, int n) {      // sorts arr in place
  qsort(arr, n, sizeof(float), fc_cmp);
  return (n % 2) ? arr[n / 2] : (arr[n / 2 - 1] + arr[n / 2]) / 2.0f;
}
static float fc_mad(float *arr, int n) {         // destroys arr
  float med = fc_median(arr, n);
  for (int i = 0; i < n; i++) arr[i] = fabsf(arr[i] - med);
  return fc_median(arr, n);
}

// Autoencoder forward pass: returns MSE between mad[3] and its reconstruction.
static float fc_infer(const float mad[3]) {
#if USE_INT8
  return fc_mse_int8(mad);
#endif
  float h[MODEL_HIDDEN], r[3];
  for (int j = 0; j < MODEL_HIDDEN; j++) {
    float s = MODEL_B1[j];
    for (int i = 0; i < 3; i++) s += mad[i] * MODEL_W1[i][j];
    h[j] = s > 0 ? s : 0;                        // ReLU
  }
  float mse = 0;
  for (int k = 0; k < 3; k++) {
    float s = MODEL_B2[k];
    for (int j = 0; j < MODEL_HIDDEN; j++) s += h[j] * MODEL_W2[j][k];
    r[k] = s;                                    // linear output
    float d = mad[k] - r[k];
    mse += d * d;
  }
  return mse / 3.0f;
}

// ---- Serial protocol --------------------------------------------------------
// PC  : S,<id>            start sample
//       <x>,<y>,<z>       x WINDOW_SIZE lines
//       E,<count>         end (count must equal WINDOW_SIZE)
// ESP : R,<id>,<label>,<mse>,<madx>,<mady>,<madz>   label 0=NORMAL 1=ANOMALY
//       ERR,<id>,<reason>
typedef struct {
  int active, id, n;
  float buf[NUM_AXES][WINDOW_SIZE];
  // last result (for display)
  int has_result, label;
  float mse, mad[3];
} fc_state;

static void fc_init(fc_state *s) { memset(s, 0, sizeof(*s)); }

// Feed one text line (no newline). Writes a reply into out (may be empty). Returns 1 if reply is non-empty.
static int fc_line(fc_state *s, char *line, char *out, int outsz) {
  out[0] = 0;
  if (line[0] == 'S' && line[1] == ',') {
    s->active = 1; s->id = atoi(line + 2); s->n = 0; return 0;
  }
  if (line[0] == 'E' && line[1] == ',') {
    int cnt = atoi(line + 2), ok = s->active;
    s->active = 0;
    if (!ok || cnt != WINDOW_SIZE || s->n != WINDOW_SIZE) {
      snprintf(out, outsz, "ERR,%d,count_%d_of_%d", s->id, s->n, WINDOW_SIZE); return 1;
    }
    float tmp[WINDOW_SIZE];
    for (int a = 0; a < NUM_AXES; a++) {
      memcpy(tmp, s->buf[a], sizeof(tmp));
      s->mad[a] = fc_mad(tmp, WINDOW_SIZE);
    }
    s->mse = fc_infer(s->mad);
    s->label = s->mse > ANOMALY_THRESHOLD;
    s->has_result = 1;
    snprintf(out, outsz, "R,%d,%d,%.6e,%.6e,%.6e,%.6e", s->id, s->label, s->mse, s->mad[0], s->mad[1], s->mad[2]);
    return 1;
  }
  if (s->active) {
    if (s->n < WINDOW_SIZE) {
      float x, y, z;
      if (sscanf(line, "%f,%f,%f", &x, &y, &z) == 3) {
        s->buf[0][s->n] = x; s->buf[1][s->n] = y; s->buf[2][s->n] = z; s->n++;
      }
    } else s->n++;                               // overflow -> caught at E
  }
  return 0;
}
#endif
