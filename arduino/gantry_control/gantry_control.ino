/**
 * Gantry Control Firmware (Pick & Place)
 *
 * Pins:
 * X-Axis: Step D6, Dir D7
 * Y-Axis: Step D9, Dir D8
 * Z-Axis: Step D11, Dir D12
 * Magnet: D10
 * LimitSW: D2
 *
 * Protocol:
 * "XXYY\n" (e.g., "0405") -> Pick from Supply, Place at Grid(4, 5).
 */

// --- CONFIGURATION (CALIBRATE THESE) ---
const long STEPS_PER_GRID_X = 200; // 1マスのステップ数 (X)
const long STEPS_PER_GRID_Y = 200; // 1マスのステップ数 (Y)

// 供給地点（コマを取りに行く場所）の座標（ホームからのステップ数）
const long SUPPLY_POS_X = 0;
const long SUPPLY_POS_Y = 0;

// 原点オフセット（盤面の(0,0)地点までのステップ数）
const long OFFSET_X = 100;
const long OFFSET_Y = 100;

// Z軸の昇降ステップ数
const long Z_DOWN_STEPS = 500;

// 速度設定 (delay microseconds)
const int STEP_DELAY = 800;

// --- PINS ---
const int PIN_LIM_SW = 2;

const int PIN_M1_STEP = 6; // X
const int PIN_M1_DIR = 7;
const int PIN_M2_DIR = 8; // Y (Note: User specified D8=DIR, D9=STEP)
const int PIN_M2_STEP = 9;
const int PIN_MAG = 10;
const int PIN_M3_STEP = 11; // Z
const int PIN_M3_DIR = 12;

long currentX = 0;
long currentY = 0;
long currentZ = 0;

void setup() {
  Serial.begin(9600);

  pinMode(PIN_M1_STEP, OUTPUT);
  pinMode(PIN_M1_DIR, OUTPUT);
  pinMode(PIN_M2_STEP, OUTPUT);
  pinMode(PIN_M2_DIR, OUTPUT);
  pinMode(PIN_M3_STEP, OUTPUT);
  pinMode(PIN_M3_DIR, OUTPUT);
  pinMode(PIN_MAG, OUTPUT);
  pinMode(PIN_LIM_SW, INPUT_PULLUP);

  digitalWrite(PIN_MAG, LOW); // Magnet OFF

  // Homing Sequence (Simplified)
  // homeX();
  // homeY();
  // homeZ();

  Serial.println("READY");
}

String inputBuffer = "";

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  if (cmd.length() == 4) {
    // Parse "XXYY"
    int gx = cmd.substring(0, 2).toInt();
    int gy = cmd.substring(2, 4).toInt();

    // Calculate targets
    long targetX = OFFSET_X + (gx * STEPS_PER_GRID_X);
    long targetY = OFFSET_Y + (gy * STEPS_PER_GRID_Y);

    Serial.println("BUSY: Picking Stone");

    // 1. Move to Supply
    moveTo(SUPPLY_POS_X, SUPPLY_POS_Y);

    // 2. Pick
    moveZ(Z_DOWN_STEPS);         // Down
    digitalWrite(PIN_MAG, HIGH); // ON
    delay(500);
    moveZ(-Z_DOWN_STEPS); // Up

    Serial.println("BUSY: Placing at " + String(gx) + "," + String(gy));

    // 3. Move to Target
    moveTo(targetX, targetY);

    // 4. Place
    moveZ(Z_DOWN_STEPS);        // Down
    digitalWrite(PIN_MAG, LOW); // OFF
    delay(200);
    moveZ(-Z_DOWN_STEPS); // Up

    // 5. Return to Safety? (Optional)
    // moveTo(SUPPLY_POS_X, SUPPLY_POS_Y);

    Serial.println("READY");
  }
}

// --- MOTION CONTROL ---

void stepMotor(int stepPin, int dirPin, int dir, int steps) {
  digitalWrite(dirPin, dir > 0 ? HIGH : LOW);
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(STEP_DELAY);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(STEP_DELAY);
  }
}

void moveTo(long tx, long ty) {
  long dx = tx - currentX;
  long dy = ty - currentY;

  // NOTE: This assumes independent movement. Ideally Bresenham's algorithm for
  // simultaneous. X Drive
  if (dx != 0) {
    stepMotor(PIN_M1_STEP, PIN_M1_DIR, dx > 0 ? 1 : 0, abs(dx));
    currentX = tx;
  }

  // Y Drive
  if (dy != 0) {
    stepMotor(PIN_M2_STEP, PIN_M2_DIR, dy > 0 ? 1 : 0, abs(dy));
    currentY = ty;
  }
}

void moveZ(long steps) {
  // Positive = Down (assuming Setup)
  int dir = steps > 0 ? 1 : 0;
  stepMotor(PIN_M3_STEP, PIN_M3_DIR, dir, abs(steps));
}
