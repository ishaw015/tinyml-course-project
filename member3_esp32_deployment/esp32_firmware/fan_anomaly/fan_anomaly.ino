// Member 3 -- ESP32 fan anomaly detector (Member 2 autoencoder, float forward pass).
// Receives 200x3 samples over USB serial, computes raw MAD, runs the model, shows result on LCD.
// Libraries: LiquidCrystal_I2C (Frank de Brabander). Set USE_LCD 0 to run serial-only.
#define USE_LCD 0
#include "fan_core.h"
#if USE_LCD
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);
#endif

static fc_state st;
static char line[64], out[160];
static int len = 0;

static void show() {
#if USE_LCD
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(st.label ? "ANOMALY" : "NORMAL");
  lcd.setCursor(0, 1); lcd.print("mse "); lcd.print(st.mse * 1e6f, 3); lcd.print("e-6");
#endif
}

void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(115200);
#if USE_LCD
  lcd.init(); lcd.backlight(); lcd.print("Fan detector"); lcd.setCursor(0, 1); lcd.print("waiting...");
#endif
  fc_init(&st);
  Serial.println(USE_INT8 ? "READY int8" : "READY fp32");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[len] = 0; len = 0;
      if (fc_line(&st, line, out, sizeof(out))) {
        Serial.println(out);
        if (st.has_result && out[0] == 'R') show();
      }
    } else if (len < (int)sizeof(line) - 1) line[len++] = c;
  }
}
