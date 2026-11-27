#!/usr/bin/env python3
"""
Member 3 sender: streams recorded 200x3 fan samples to the ESP32 (or the host simulator),
compares replies with a Python reference, and prints a confusion matrix.

  python python_serial_sender.py --exec ./host_test/fan_host_sim --set both --limit 0
  python python_serial_sender.py --port COM5 --set both --limit 3
  python python_serial_sender.py --port rfc2217://localhost:4000 --limit 3     (Wokwi)

--limit 0 = all test samples. Test sets are exactly Member 1's held-out lists (fan_dataset.npz).
"""
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np

W = 200
HERE = Path(__file__).resolve().parent

def find_repo(arg):
    for c in [arg, HERE, HERE.parent]:
        if c and (Path(c) / "datasets").is_dir():
            return Path(c)
    sys.exit("Cannot find the repo root (folder containing datasets/). Use --repo <path>.")

def load_ref(model="fp32"):
    if model == "int8":
        return dict(np.load(HERE / "tools" / "int8_params.npz")), float((HERE / "tools" / "threshold_int8.txt").read_text().strip())
    w = np.load(HERE / "tools" / "weights.npz")
    thr = float((HERE / "tools" / "threshold.txt").read_text().strip())
    return w, thr

def py_predict(x, w, thr):
    mad = np.median(np.abs(x - np.median(x, axis=0)), axis=0).astype(np.float32)
    if "M1" in w:
        xq = np.clip(np.round(mad / np.float32(w["s_in"])) + int(w["zp_in"]), -128, 127).astype(np.int64)
        a1 = (xq - int(w["zp_in"])) @ w["W1"].astype(np.int64).T
        hq = np.clip(np.round(a1.astype(np.float32) * w["M1"]), 0, 127).astype(np.int64)
        a2 = hq @ w["W2"].astype(np.int64).T + w["B2"].astype(np.int64)
        yq = np.clip(np.round(a2.astype(np.float32) * w["M2"]) + int(w["zp_o"]), -128, 127)
        r = (yq - int(w["zp_o"])) * np.float32(w["s_o"])
    else:
        h = np.maximum(mad @ w["W1"] + w["B1"], 0)
        r = h @ w["W2"] + w["B2"]
    mse = float(np.mean((mad - r) ** 2))
    return int(mse > thr), mse

class Link:
    def __init__(self, a):
        self.proc = self.ser = None
        if a.exec:
            self.proc = subprocess.Popen([a.exec], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        else:
            import serial  # pip install pyserial
            self.ser = serial.serial_for_url(a.port, a.baud, timeout=5)
            time.sleep(2); self.ser.reset_input_buffer()
    def send(self, lines, delay):
        data = "".join(l + "\n" for l in lines)
        if self.proc:
            self.proc.stdin.write(data); self.proc.stdin.flush()
        else:
            for l in lines:
                self.ser.write((l + "\n").encode())
                if delay: time.sleep(delay)
    def read(self):
        if self.proc: return self.proc.stdout.readline().strip()
        while True:
            l = self.ser.readline().decode(errors="ignore").strip()
            if not l or l.startswith(("R,", "ERR,")): return l

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port"); p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--exec"); p.add_argument("--repo"); p.add_argument("--model", choices=["int8", "fp32"], default="int8")
    p.add_argument("--set", choices=["normal", "anomaly", "both"], default="both")
    p.add_argument("--limit", type=int, default=3, help="samples per set (0 = all)")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between lines (serial only)")
    a = p.parse_args()
    if not a.port and not a.exec: sys.exit("give --port or --exec")
    repo = find_repo(a.repo); ds = repo / "datasets"
    npz = repo / "member1_data_analysis" / "processed_data" / "fan_dataset.npz"
    d = np.load(npz, allow_pickle=True)
    items = []
    if a.set in ("normal", "both"):  items += [(f, 0) for f in d["test_normal_files"][: a.limit or None]]
    if a.set in ("anomaly", "both"): items += [(f, 1) for f in d["test_anomaly_files"][: a.limit or None]]
    w, thr = load_ref(a.model); link = Link(a)
    tn = fp = fn = tp = errs = agree = 0
    print(f"threshold={thr:.4e}  samples={len(items)}\n  id  true  py   esp   mse(esp)   mse(py)   file")
    for i, (f, y) in enumerate(items):
        x = np.loadtxt(ds / str(f).replace("\\", "/"), delimiter=",", dtype=np.float32)
        assert x.shape == (W, 3), (f, x.shape)
        link.send([f"S,{i}"] + [f"{r[0]:.9g},{r[1]:.9g},{r[2]:.9g}" for r in x] + [f"E,{W}"], a.delay)
        rep = link.read().split(",")
        if rep[0] != "R": errs += 1; print(f"{i:4d}  ERROR {rep}"); continue
        lab, mse = int(rep[2]), float(rep[3]); pl, pm = py_predict(x, w, thr)
        agree += lab == pl
        tn += (y == 0 and lab == 0); fp += (y == 0 and lab == 1); fn += (y == 1 and lab == 0); tp += (y == 1 and lab == 1)
        if a.limit or lab != pl:
            print(f"{i:4d}  {'ANOM' if y else 'NORM'}  {'A' if pl else 'N'}    {'A' if lab else 'N'}   {mse:.3e}  {pm:.3e}  {f}")
    n = len(items) - errs
    print(f"\nsent={len(items)} errors={errs}\nESP/host == Python label: {agree}/{n}")
    print(f"accuracy vs ground truth: {tn+tp}/{n} = {(tn+tp)/max(n,1):.4f}\nconfusion [[TN FP],[FN TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")

if __name__ == "__main__":
    main()
