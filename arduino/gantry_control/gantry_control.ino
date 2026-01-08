#include <AccelStepper.h>

/**
 * Gantry Control Firmware (Reviewed & Updated)
 *
 * [User Feedback & Spec]
 * 1. Z-Axis Calibrated: -540 steps (Down from Home).
 * 2. Electromagnet: Pin 13.
 * 3. Z-Limit Switch: Pin 9 (Active LOW).
 * 4. Z-Homing: Must move UP (Positive) to find switch.
 * 5. Sequence: Supply(X275, Y40) -> Pick(Z-540) -> Target -> Place(Z-540) ->
 * Home.
 * 6. X/Y Homing: Move Negative to find switch.
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

const int magnetPin = 13; // Updated to 13 (was 10)

// --- 2. Motor Instances (AccelStepper) ---
AccelStepper xAxis(AccelStepper::DRIVER, x_stepPin, x_dirPin);
AccelStepper yAxis(AccelStepper::DRIVER, y_stepPin, y_dirPin);
AccelStepper zAxis(AccelStepper::DRIVER, z_stepPin, z_dirPin);

// --- 3. Speed & Acceleration ---
// Sliding screws might need slower speeds if stalling occurs.
const float MAX_SPEED = 1000.0;
const float ACCEL = 400.0;
const float HOMING_SPEED_FAST = 400.0;
const float HOMING_SPEED_SLOW = 50.0;

// --- 4. Coordinates & Calibration ---
const long Z_PICK_DEPTH = -540; // Calibrated Value
const long SUPPLY_X = 275;
const long SUPPLY_Y = 40;

// Grid conversion (Pre-calibrated constants)
long getXSteps(int grid) { return (long)(551.58 - (41.58 * grid)); }
long getYSteps(int grid) { return (long)(1075.31 - (69.24 * grid)); }

// Serial Buffer
char buffer[10];
int bufIndex = 0;

// --- Forward Declarations ---
void homeAxisInitial(AccelStepper *axis, int swPin, int dir);
void smartHomeAll();
void executeSequence(int gx, int gy);

// ==========================================================
// Setup
// ==========================================================
void setup() {
  Serial.begin(9600);

  // Pins
  pinMode(x_swPin, INPUT_PULLUP);
  pinMode(y_swPin, INPUT_PULLUP);
  pinMode(z_swPin, INPUT_PULLUP);
  pinMode(magnetPin, OUTPUT);
  digitalWrite(magnetPin, LOW);

  // Motor Config
  xAxis.setMaxSpeed(MAX_SPEED);
  xAxis.setAcceleration(ACCEL);

  yAxis.setMaxSpeed(MAX_SPEED);
  yAxis.setAcceleration(ACCEL);
  yAxis.setPinsInverted(true, false,
                        false); // Invert Y Direction based on prior settings

  zAxis.setMaxSpeed(MAX_SPEED);
  zAxis.setAcceleration(ACCEL);

  // --- Homing Sequence (Z first for safety) ---
  Serial.println("BUSY: Homing Z (Up)...");
  // Z homes in POSITIVE direction (1)
  homeAxisInitial(&zAxis, z_swPin, 1);

  Serial.println("BUSY: Homing X/Y...");
  // X/Y home in NEGATIVE direction (-1)
  homeAxisInitial(&xAxis, x_swPin, -1);
  homeAxisInitial(&yAxis, y_swPin, -1);

  Serial.println("READY");
}

// ==========================================================
// Main Loop
// ==========================================================
void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (isDigit(c) && bufIndex < 4) {
      buffer[bufIndex++] = c;
    }

    if (c == '\n') {
      if (bufIndex == 4) {
        buffer[4] = '\0';
        int targetX_grid = atoi(buffer) / 100;
        int targetY_grid = atoi(buffer) % 100;

        if (targetX_grid == 0 && targetY_grid == 0) {
          Serial.println("BUSY: SmartHoming...");
          smartHomeAll();
          Serial.println("READY");
        } else if (targetX_grid >= 1 && targetX_grid <= 13) {
          Serial.println("BUSY: Picking & Placing...");
          executeSequence(targetX_grid, targetY_grid);
          Serial.println("READY");
        } else {
          Serial.println("ERROR: Invalid Grid");
          Serial.println("READY");
        }
      }
      bufIndex = 0;
    }
  }
}

// ==========================================================
// Homing & Motion Functions
// ==========================================================

/**
 * Initial Homing (Power-on)
 * dir: 1 (Up/Positive) or -1 (Down/Negative)
 */
void homeAxisInitial(AccelStepper *axis, int swPin, int dir) {
  // 1. Fast Approach
  axis->setMaxSpeed(HOMING_SPEED_FAST);
  axis->setSpeed(HOMING_SPEED_FAST * dir);

  while (digitalRead(swPin) == HIGH) {
    axis->runSpeed();
  }

  // Stop on Switch
  axis->setSpeed(0);
  axis->setCurrentPosition(0);

  // 2. Backoff (Move away from switch)
  long backoffDist = 150 * (-dir); // Move opposite to approach
  axis->setMaxSpeed(MAX_SPEED);
  axis->runToNewPosition(backoffDist);

  // 3. Slow Approach (Precision)
  axis->setMaxSpeed(HOMING_SPEED_SLOW);
  axis->setSpeed(HOMING_SPEED_SLOW * dir);

  while (digitalRead(swPin) == HIGH) {
    axis->runSpeed();
  }

  // 4. Final Stop & Zero
  axis->setSpeed(0);
  axis->setCurrentPosition(0);

  // Restore Speed
  axis->setMaxSpeed(MAX_SPEED);
}

/**
 * Smart Re-Homing (During Run)
 * Checks zero position securely.
 */
void smartHomeAxis(AccelStepper *axis, int swPin, int dir) {
  // 1. Move to "Safe Approach Zone" (e.g. +/- 50 steps from home)
  // If we are far away, move closer quickly.
  // If Z (dir=1), Safe is -50.
  // If X (dir=-1), Safe is 50.
  long safePos = 50 * (-dir);

  // Logic: If Z is at -500, move to -50.
  bool needsMove = (dir == 1) ? (axis->currentPosition() < safePos)
                              : (axis->currentPosition() > safePos);

  if (needsMove) {
    axis->moveTo(safePos);
    while (axis->distanceToGo() != 0) {
      // Safety: If switch hit early, stop.
      if (digitalRead(swPin) == LOW) {
        axis->setCurrentPosition(0);
        axis->moveTo(0);
        return;
      }
      axis->run();
    }
  }

  // 2. Slow Approach to 0
  axis->setMaxSpeed(HOMING_SPEED_SLOW);
  axis->moveTo(50 * dir); // Overshoot target to ensure contact

  while (digitalRead(swPin) == HIGH) {
    if (axis->distanceToGo() == 0)
      break; // Reached limit without switch?
    axis->run();
  }

  axis->setCurrentPosition(0);
  axis->setSpeed(0);
  axis->setMaxSpeed(MAX_SPEED);
}

void smartHomeAll() {
  // Always home Z first to clear obstacles
  smartHomeAxis(&zAxis, z_swPin, 1);
  smartHomeAxis(&xAxis, x_swPin, -1);
  smartHomeAxis(&yAxis, y_swPin, -1);
}

/**
 * Main Sequence: Pick -> Place -> Home
 */
void executeSequence(int gx, int gy) {
  // 1. Convert Grid to Steps
  long targetX = getXSteps(gx);
  long targetY = getYSteps(gy);

  // --- Step 1: Go to Supply Point ---
  // Moving X/Y while Z is 0 (Up)
  xAxis.moveTo(SUPPLY_X);
  yAxis.moveTo(SUPPLY_Y);
  while (xAxis.distanceToGo() != 0 || yAxis.distanceToGo() != 0) {
    xAxis.run();
    yAxis.run();
  }

  // --- Step 2: Pick (Magnet ON) ---
  // Down
  zAxis.moveTo(Z_PICK_DEPTH);
  zAxis.runToPosition();

  // Magnet ON
  digitalWrite(magnetPin, HIGH);
  delay(600); // Wait for magnetization

  // Up
  zAxis.moveTo(0);
  zAxis.runToPosition();

  // --- Step 3: Go to Target Grid ---
  xAxis.moveTo(targetX);
  yAxis.moveTo(targetY);
  while (xAxis.distanceToGo() != 0 || yAxis.distanceToGo() != 0) {
    xAxis.run();
    yAxis.run();
  }

  // --- Step 4: Place (Magnet OFF) ---
  // Down
  zAxis.moveTo(Z_PICK_DEPTH);
  zAxis.runToPosition();

  // Magnet OFF
  digitalWrite(magnetPin, LOW);
  delay(600); // Wait for demagnetization/drop

  // Up
  zAxis.moveTo(0); // Return to Z=0
  zAxis.runToPosition();

  // --- Step 5: Return to Home (0,0) ---
  xAxis.moveTo(0);
  yAxis.moveTo(0);
  while (xAxis.distanceToGo() != 0 || yAxis.distanceToGo() != 0) {
    xAxis.run();
    yAxis.run();
  }

  // --- Step 6: Verify Zero Points ---
  smartHomeAll();
}
