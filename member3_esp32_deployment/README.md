# Member 3 - ESP32 deployment of Member 2's fan autoencoder

Pipeline: PC (python_serial_sender.py) --USB serial--> ESP32: raw MAD (200 readings/axis) -> 3-2-3 autoencoder -> MSE vs threshold -> LCD.

## Layout
- esp32_firmware/fan_anomaly/  sketch (fan_anomaly.ino), fan_core.h (MAD, model, protocol), model_config.h, model_weights.h (generated)
- host_test/fan_host_sim.c     same fan_core.h compiled on a PC
- tools/make_model_header.py   regenerates model_weights.h from fan_autoencoder_linear.h5
- wokwi_simulation/            diagram.json, wokwi.toml, platformio.ini (ESP32 + I2C LCD 16x2 at 0x27)
- python_serial_sender.py      streams Member 1's held-out test samples, checks replies

## Run (put this folder inside the cloned repo, next to datasets/, or pass --repo)
    gcc -O2 -I esp32_firmware/fan_anomaly host_test/fan_host_sim.c -o host_test/fan_host_sim -lm
    python python_serial_sender.py --exec host_test/fan_host_sim --set both --limit 0
Expected: confusion [[53, 5], [4, 576]], accuracy 629/638 = 0.9859.
Real board: python python_serial_sender.py --port COM5 --limit 3   (pip install pyserial). Wokwi: --port rfc2217://localhost:4000.

## Must-match settings
Raw MAD (no 1.4826 scale), 200 readings, float32 forward pass, threshold 4.8191197e-07.

## Caveats to report
- Hidden ReLU layer is dead: it outputs 0 for all 638 test samples, so the model returns a constant (the bias vector, ~training mean). Detection is really "distance from the mean MAD".
- Only the FP32 model is deployed here. Member 2's INT8 .tflite was not attached, so TFLM is not wired in.
- Only the "low speed, anomaly = 1 weight" setting was trained/tested. Other speeds/weights are all flagged ANOMALY (see below).
