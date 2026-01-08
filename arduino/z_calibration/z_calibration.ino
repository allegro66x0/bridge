#include <AccelStepper.h>

/**
 * Z-Axis & Electromagnet Calibration Tool
 *
 * Commands:
 * 'w': Up 10 steps
 * 's': Down 10 steps
 * 'W': Up 100 steps
 * 'S': Down 100 steps
 * 'm': Toggle Electromagnet (Pin 13)
 * 'z': Set Current Position as 0 (Home)
 * 'r': Report Current Position
 */

// --- Pin Definitions (From gantry_control.ino + User Request) ---
const int z_dirPin = 12;
const int z_stepPin = 11;
const int magnetPin = 13; // User requested Pin 13
const int z_swPin = 9;    // Limit Switch (from gantry_control.ino)

AccelStepper zAxis(AccelStepper::DRIVER, z_stepPin, z_dirPin);

bool magnetState = false;

void setup() {
  Serial.begin(9600);

  pinMode(magnetPin, OUTPUT);
  digitalWrite(magnetPin, LOW);

  pinMode(z_swPin, INPUT_PULLUP);

  zAxis.setMaxSpeed(500.0);
  zAxis.setAcceleration(200.0);

  Serial.println("--- Z-Axis Calibration ---");
  Serial.println("w/s: Jog Small (10)");
  Serial.println("W/S: Jog Large (100)");
  Serial.println("m:   Toggle Magnet");
  Serial.println("h:   HOME Z-Axis (Up to Switch)");
  Serial.println("z:   Zero Position (Manual)");
  Serial.println("r:   Report Current Position");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();

    // Move commands
    if (c == 'w') {
      zAxis.move(10);
      Serial.println("Jog Up 10");
    } else if (c == 's') {
      zAxis.move(-10);
      Serial.println("Jog Down 10");
    } else if (c == 'W') {
      zAxis.move(100);
      Serial.println("Jog Up 100");
    } else if (c == 'S') {
      zAxis.move(-100);
      Serial.println("Jog Down 100");
    }
    // Magnet
    else if (c == 'm') {
      magnetState = !magnetState;
      digitalWrite(magnetPin, magnetState ? HIGH : LOW);
      Serial.print("Magnet: ");
      Serial.println(magnetState ? "ON" : "OFF");
    }
    // Homing (NEW)
    else if (c == 'h') {
      Serial.println("Homing Z (Moving UP)...");
      zAxis.setMaxSpeed(200); // Slow homing speed
      zAxis.setSpeed(200);    // Positive = UP

      while (digitalRead(z_swPin) == HIGH) { // Wait for Switch (LOW)
        zAxis.runSpeed();
      }

      zAxis.setSpeed(0);
      zAxis.setCurrentPosition(0);
      zAxis.setMaxSpeed(500.0); // Restore max speed
      Serial.println("Homed! Position set to 0.");
    }
    // Zeroing
    else if (c == 'z') {
      zAxis.setCurrentPosition(0);
      Serial.println("Position Zeroed!");
    }
    // Report
    else if (c == 'r') {
      Serial.print("Current Pos: ");
      Serial.println(zAxis.currentPosition());
      Serial.print("Switch: ");
      Serial.println(digitalRead(z_swPin) == LOW ? "TRIGGERED" : "OPEN");
    }
  }

  zAxis.run();
}

void printInstructions() {
  // Optional: periodic status
}
