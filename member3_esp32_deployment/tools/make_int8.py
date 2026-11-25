import numpy as np, tensorflow as tf, sys
from pathlib import Path
T = Path(__file__).resolve().parent
repo = Path(sys.argv[1])
d = np.load(repo/"member1_data_analysis/processed_data/fan_dataset.npz", allow_pickle=True)
Xtr, Xv, Xn, Xa = [d[k].astype(np.float32) for k in ("X_train", "X_val", "X_test_normal", "X_test_anomaly")]
m = tf.keras.models.load_model(str(T/"fan_autoencoder_linear.h5"), compile=False)

def rep():
    for s in Xtr[:100]:
        yield [s.reshape(1, 3)]

c = tf.lite.TFLiteConverter.from_keras_model(m)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.int8
c.inference_output_type = tf.int8
tfl = c.convert()
(T/"fan_autoencoder_int8.tflite").write_bytes(tfl)
print("tflite bytes", len(tfl))

it = tf.lite.Interpreter(model_content=tfl)
it.allocate_tensors()
i = it.get_input_details()[0]
o = it.get_output_details()[0]
si, zi = i["quantization"]
so, zo = o["quantization"]

def run(X):
    out = []
    for x in X:
        q = np.clip(np.round(x/si)+zi, -128, 127).astype(np.int8).reshape(1, 3)
        it.set_tensor(i["index"], q)
        it.invoke()
        out.append((it.get_tensor(o["index"])[0].astype(np.float32)-zo)*so)
    return np.array(out)

mse = lambda X: np.mean((X-run(X))**2, axis=1)
thr = float(np.percentile(mse(Xv), 95))
en, ea = mse(Xn), mse(Xa)
cm = [[int((en <= thr).sum()), int((en > thr).sum())], [int((ea <= thr).sum()), int((ea > thr).sum())]]
print("thr", thr, "cm", cm, "acc", (cm[0][0]+cm[1][1])/(len(Xn)+len(Xa)))
(T/"threshold_int8.txt").write_text(repr(thr))
print("in q", si, zi, "out q", so, zo)
for t in it.get_tensor_details():
    print(t["index"], t["name"], t["shape"], t["dtype"].__name__, t["quantization"])