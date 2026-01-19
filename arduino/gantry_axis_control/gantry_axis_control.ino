#include <AccelStepper.h>

/**
 * Gantry Axis Control (Motion Only)
 * 
 * [Spec]
 * - Controls X, Y, Z stepper motors.
 * - NO Magnet control (Moved to separate microcontroller).
 * - Receives Grid Coordinates (GXGY) and performs Pick & Place MOTION.
 */

// --- 1. Pin Definitions ---
const int x_swPin = 5;
const int x_dirPin = 7;
const int x_stepPin = 6;

const int y_swPin = 2;
const int y_dirPin = 4;
const int y_stepPin = 3;

const int z_swPin = 9; // Z Limit
const int z_dirPin = 12;
const int z_stepPin = 11;

// --- 2. Motor Instances (AccelStepper) ---
AccelStepper xAxis(AccelStepper::DRIVER, x_stepPin, x_dirPin);
AccelStepper yAxis(AccelStepper::DRIVER, y_stepPin, y_dirPin);
AccelStepper zAxis(AccelStepper::DRIVER, z_stepPin, z_dirPin);

// --- 3. Speed & Acceleration ---
const float MAX_SPEED = 1000.0;
const float ACCEL = 400.0;
const float HOMING_SPEED_FAST = 400.0;
const float HOMING_SPEED_SLOW = 50.0;

// --- 4. Coordinates & Calibration ---
const long Z_PICK_DEPTH = -540; // Calibrated Value
const long SUPPLY_X = 275;
const long SUPPLY_Y = 40;

// String buffer for parsing
String inputString = "";
boolean stringComplete = false;

// --- Forward Declarations ---
void homeAxisInitial(AccelStepper *axis, int swPin, int dir);
void smartHomeAll();
void moveToGrid(int gx, int gy);
void moveZ(long targetStep);

void setup() {
  Serial.begin(9600);
  inputString.reserve(200);

  // Pins
  pinMode(x_swPin, INPUT_PULLUP);
  pinMode(y_swPin, INPUT_PULLUP);
  pinMode(z_swPin, INPUT_PULLUP);

  // Motor Config
  xAxis.setMaxSpeed(MAX_SPEED);
  xAxis.setAcceleration(ACCEL);

  yAxis.setMaxSpeed(MAX_SPEED);
  yAxis.setAcceleration(ACCEL);
  yAxis.setPinsInverted(true, false, false);

  zAxis.setMaxSpeed(MAX_SPEED);
  zAxis.setAcceleration(ACCEL);

  // --- Homing Sequence ---
  Serial.println("BUSY: Homing Z (Up)...");
  homeAxisInitial(&zAxis, z_swPin, 1);

  Serial.println("BUSY: Homing X/Y...");
  homeAxisInitial(&xAxis, x_swPin, -1);
  homeAxisInitial(&yAxis, y_swPin, -1);

  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  if (stringComplete) {
    inputString.trim(); // Remove whitespace/newlines
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}

void processCommand(String cmd) {
  // --- Motion Commands ---
  // Format: "M:XXYY" (Grid), "M:SPLY" (Supply), "M:ZERO" (Home)
  if (cmd.startsWith("M:")) {
    String param = cmd.substring(2);
    if (param == "SPLY") {
      // Move to Supply
      Serial.println("BUSY: Moving to Supply");
      xAxis.moveTo(SUPPLY_X);
      yAxis.moveTo(SUPPLY_Y);
      runToTarget();
      Serial.println("READY");
    } 
    else if (param == "ZERO") {
      // Move to Zero
      Serial.println("BUSY: Moving to Zero");
      xAxis.moveTo(0);
      yAxis.moveTo(0);
      runToTarget();
      Serial.println("READY");
    }
    else if (param.length() == 4 && isDigit(param[0])) {
      // Grid XXYY
      int gx = param.substring(0, 2).toInt();
      int gy = param.substring(2, 4).toInt();
      Serial.print("BUSY: Moving to Grid "); Serial.print(gx); Serial.print(","); Serial.println(gy);
      moveToGrid(gx, gy);
      Serial.println("READY");
    }
    else {
      Serial.println("ERROR: Invalid M Command");
    }
  }
  // --- Z-Axis Commands ---
  // Format: "Z:PICK" (Down), "Z:HOME" (Up)
  else if (cmd.startsWith("Z:")) {
    String param = cmd.substring(2);
    if (param == "PICK") {
      Serial.println("BUSY: Z Down");
      moveZ(Z_PICK_DEPTH);
      Serial.println("READY");
    }
    else if (param == "HOME") {
      Serial.println("BUSY: Z Up");
      moveZ(0);
      Serial.println("READY");
    }
    else {
      Serial.println("ERROR: Invalid Z Command");
    }
  }
  // --- Homing Command ---
  else if (cmd == "H:ALL") {
    Serial.println("BUSY: Re-Homing...");
    smartHomeAll();
    Serial.println("READY");
  }
  else {
    Serial.println("ERROR: Unknown Command");
  }
}

void runToTarget() {
  while (xAxis.distanceToGo() != 0 || yAxis.distanceToGo() != 0 || zAxis.distanceToGo() != 0) {
    xAxis.run();
    yAxis.run();
    zAxis.run();
  }
}

void moveToGrid(int gx, int gy) {
  long targetX = getXSteps(gx);
  long targetY = getYSteps(gy);
  xAxis.moveTo(targetX);
  yAxis.moveTo(targetY);
  runToTarget();
}

void moveZ(long targetStep) {
  zAxis.moveTo(targetStep);
  runToTarget();
}

// --- Homing Functions (Same as before) ---
void homeAxisInitial(AccelStepper *axis, int swPin, int dir) {
  axis->setMaxSpeed(HOMING_SPEED_FAST);
  axis->setSpeed(HOMING_SPEED_FAST * dir);
  while (digitalRead(swPin) == HIGH) { axis->runSpeed(); }
  
  axis->setSpeed(0);
  axis->setCurrentPosition(0);

  long backoffDist = 150 * (-dir);
  axis->setMaxSpeed(MAX_SPEED);
  axis->runToNewPosition(backoffDist);

  axis->setMaxSpeed(HOMING_SPEED_SLOW);
  axis->setSpeed(HOMING_SPEED_SLOW * dir);
  while (digitalRead(swPin) == HIGH) { axis->runSpeed(); }
  
  axis->setSpeed(0);
  axis->setCurrentPosition(0);
  axis->setMaxSpeed(MAX_SPEED);
}

void smartHomeAxis(AccelStepper *axis, int swPin, int dir) {
  long safePos = 50 * (-dir);
  bool needsMove = (dir == 1) ? (axis->currentPosition() < safePos)
                              : (axis->currentPosition() > safePos);

  if (needsMove) {
    axis->moveTo(safePos);
    while (axis->distanceToGo() != 0) {
      if (digitalRead(swPin) == LOW) {
        axis->setCurrentPosition(0);
        axis->moveTo(0);
        return;
      }
      axis->run();
    }
  }

  axis->setMaxSpeed(HOMING_SPEED_SLOW);
  axis->moveTo(50 * dir);

  while (digitalRead(swPin) == HIGH) {
    if (axis->distanceToGo() == 0) break;
    axis->run();
  }

  axis->setCurrentPosition(0);
  axis->setSpeed(0);
  axis->setMaxSpeed(MAX_SPEED);
}

void smartHomeAll() {
  smartHomeAxis(&zAxis, z_swPin, 1);
  smartHomeAxis(&xAxis, x_swPin, -1);
  smartHomeAxis(&yAxis, y_swPin, -1);
}
