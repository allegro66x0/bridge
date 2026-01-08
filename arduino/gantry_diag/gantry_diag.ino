#include <AccelStepper.h>

/**
 * Gantry Diagnostic Tool
 *
 * Use this to verify:
 * 1. Limit Switch Wiring (Are they reading correctly?)
 * 2. Motor Direction (Does + move towards or away from switch?)
 */

const int x_swPin = 5;
const int y_swPin = 2;
const int z_swPin = 9;

const int x_stepPin = 6;
const int x_dirPin = 7;
const int y_stepPin = 3;
const int y_dirPin = 4;
const int z_stepPin = 11;
const int z_dirPin = 12;

AccelStepper xAxis(AccelStepper::DRIVER, x_stepPin, x_dirPin);
AccelStepper yAxis(AccelStepper::DRIVER, y_stepPin, y_dirPin);
AccelStepper zAxis(AccelStepper::DRIVER, z_stepPin, z_dirPin);

void setup() {
  Serial.begin(9600);
  pinMode(x_swPin, INPUT_PULLUP);
  pinMode(y_swPin, INPUT_PULLUP);
  pinMode(z_swPin, INPUT_PULLUP);

  xAxis.setMaxSpeed(500);
  xAxis.setAcceleration(500);
  yAxis.setMaxSpeed(500);
  yAxis.setAcceleration(500);
  zAxis.setMaxSpeed(500);
  zAxis.setAcceleration(500);

  Serial.println("--- Gantry DIAGNOSTIC ---");
  Serial.println("Keys:");
  Serial.println("  x/X: Jog X (-100 / +100)");
  Serial.println("  y/Y: Jog Y (-100 / +100)");
  Serial.println("  z/Z: Jog Z (-100 / +100)");
  Serial.println("  r:   Report Switch States");
}

void loop() {
  // Always run motors if target set
  xAxis.run();
  yAxis.run();
  zAxis.run();

  if (Serial.available() > 0) {
    char c = Serial.read();

    if (c == 'x') {
      xAxis.move(-100);
      Serial.println("X -100");
    }
    if (c == 'X') {
      xAxis.move(100);
      Serial.println("X +100");
    }

    if (c == 'y') {
      yAxis.move(-100);
      Serial.println("Y -100");
    }
    if (c == 'Y') {
      yAxis.move(100);
      Serial.println("Y +100");
    }

    if (c == 'z') {
      zAxis.move(-100);
      Serial.println("Z -100");
    }
    if (c == 'Z') {
      zAxis.move(100);
      Serial.println("Z +100");
    }

    if (c == 'r') {
      reportSwitches();
    }
  }

  // Periodic Report (every 500ms)
  static unsigned long lastReport = 0;
  if (millis() - lastReport > 500) {
    // Only report if changed? No, keep alive is better for debugging
    // connections reportSwitches(); Commented out to avoid spam, user can press
    // 'r'
    lastReport = millis();
  }
}

void reportSwitches() {
  bool xS = digitalRead(x_swPin) == LOW; // Assuming Active LOW
  bool yS = digitalRead(y_swPin) == LOW;
  bool zS = digitalRead(z_swPin) == LOW;

  Serial.print("SWITCHES [0=Open, 1=Pressed(LOW)]: ");
  Serial.print("X(Pin5):");
  Serial.print(xS);
  Serial.print(" | Y(Pin2):");
  Serial.print(yS);
  Serial.print(" | Z(Pin9):");
  Serial.println(zS);
}
