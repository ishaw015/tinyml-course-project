// Runs the exact firmware logic (fan_core.h) on a PC: reads protocol lines on stdin, replies on stdout.
#include "fan_core.h"
int main(void) {
  static fc_state st; char line[256], out[256];
  fc_init(&st); setvbuf(stdout, NULL, _IONBF, 0);
  while (fgets(line, sizeof line, stdin)) {
    line[strcspn(line, "\r\n")] = 0;
    if (fc_line(&st, line, out, sizeof out)) puts(out);
  }
  return 0;
}
