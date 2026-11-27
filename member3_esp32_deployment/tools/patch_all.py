from pathlib import Path
F = Path("esp32_firmware/fan_anomaly")

INT8_H = r'''// fan_int8.h -- integer-only forward pass matching the TFLite INT8 model
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
'''

def patch(path, pairs):
    t = Path(path).read_text()
    if "USE_INT8" in t or "fan_int8" in t or "--model" in t:
        print("SKIP (already patched):", path); return
    for old, new in pairs:
        if old not in t:
            print("FAIL in", path, "-> not found:", old[:50]); return
        t = t.replace(old, new, 1)
    Path(path).write_text(t); print("patched", path)

(F/"fan_int8.h").write_text(INT8_H); print("wrote fan_int8.h")

patch(F/"model_config.h", [(
  "#define ANOMALY_THRESHOLD THRESHOLD_FP32\n",
  "#define THRESHOLD_INT8  4.825775e-07f\n#ifndef USE_INT8\n#define USE_INT8 1\n#endif\n"
  "#if USE_INT8\n#define ANOMALY_THRESHOLD THRESHOLD_INT8\n#else\n#define ANOMALY_THRESHOLD THRESHOLD_FP32\n#endif\n")])

patch(F/"fan_core.h", [
  ('#include "model_weights.h"\n', '#include "model_weights.h"\n#include "fan_int8.h"\n'),
  ("static float fc_infer(const float mad[3]) {\n",
   "static float fc_infer(const float mad[3]) {\n#if USE_INT8\n  return fc_mse_int8(mad);\n#endif\n")])

patch(F/"fan_anomaly.ino", [
  ("  Serial.begin(115200);\n", "  Serial.setRxBufferSize(4096);\n  Serial.begin(115200);\n"),
  ('Serial.println("READY");', 'Serial.println(USE_INT8 ? "READY int8" : "READY fp32");')])

patch("python_serial_sender.py", [
  ("def load_ref():\n    w = np.load(HERE / \"tools\" / \"weights.npz\")\n",
   "def load_ref(model=\"fp32\"):\n    if model == \"int8\":\n        return dict(np.load(HERE / \"tools\" / \"int8_params.npz\")), float((HERE / \"tools\" / \"threshold_int8.txt\").read_text().strip())\n    w = np.load(HERE / \"tools\" / \"weights.npz\")\n"),
  ('    h = np.maximum(mad @ w["W1"] + w["B1"], 0)\n    r = h @ w["W2"] + w["B2"]\n',
   '    if "M1" in w:\n'
   '        xq = np.clip(np.round(mad / np.float32(w["s_in"])) + int(w["zp_in"]), -128, 127).astype(np.int64)\n'
   '        a1 = (xq - int(w["zp_in"])) @ w["W1"].astype(np.int64).T\n'
   '        hq = np.clip(np.round(a1.astype(np.float32) * w["M1"]), 0, 127).astype(np.int64)\n'
   '        a2 = hq @ w["W2"].astype(np.int64).T + w["B2"].astype(np.int64)\n'
   '        yq = np.clip(np.round(a2.astype(np.float32) * w["M2"]) + int(w["zp_o"]), -128, 127)\n'
   '        r = (yq - int(w["zp_o"])) * np.float32(w["s_o"])\n'
   '    else:\n'
   '        h = np.maximum(mad @ w["W1"] + w["B1"], 0)\n'
   '        r = h @ w["W2"] + w["B2"]\n'),
  ('p.add_argument("--exec"); p.add_argument("--repo")',
   'p.add_argument("--exec"); p.add_argument("--repo"); p.add_argument("--model", choices=["int8", "fp32"], default="int8")'),
  ("w, thr = load_ref(); link = Link(a)", "w, thr = load_ref(a.model); link = Link(a)")])