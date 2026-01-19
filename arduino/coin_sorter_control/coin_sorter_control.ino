/**
 * コイン選別機 制御ファームウェア (最終版)
 *
 * M1 (Feeder): DC Motor (Pin 5)
 * M2 (Conv A): DC Motor (Pin 3)
 * M3 (Conv B): Stepper Motor (Pins 10, 11, 12, 13)
 *
 * 【重要】逆回転の修正について
 * - M3 (ステッピング): プログラム内で逆回転させます。
 * - M1, M2 (DC): ピン1本制御のため、ソフトでは変えられません。
 *              **モーターの赤黒の配線を物理的に入れ替えてください！**
 */

#include <Stepper.h>

// --- ピン定義 ---
const int PIN_M1_FEEDER = 5; // DC
const int PIN_M2_CONV_A = 3; // DC

// ステッピングモーター設定 (28BYJ-48等)
const int STEPS_PER_REV = 2048;
// 逆回転させるため、ピン順序を逆にしてみます (13, 11, 12, 10 or similar)
// 通常: 10, 12, 11, 13
// 逆転: 13, 11, 12, 10 (試行)
Stepper myStepper(STEPS_PER_REV, 13, 11, 12, 10);

// --- 速度設定 ---
// --- 速度設定 (デフォルト値) ---
int SPEED_FEEDER = 5;       // DC (0-255)
int SPEED_CONV_A = 60;      // DC (0-255)
int SPEED_STEPPER_RPM = 20; // Stepper Speed

bool stepperActive = false;
bool stopDelayActive = false;
unsigned long stopTimer = 0;

String inputBuffer = "";

void setup() {
  Serial.begin(9600);

  pinMode(PIN_M1_FEEDER, OUTPUT);
  pinMode(PIN_M2_CONV_A, OUTPUT);

  myStepper.setSpeed(SPEED_STEPPER_RPM);

  stopAll();
  Serial.println("READY: Coin Sorter (Variable Speed)");
}

void loop() {
  // シリアルコマンド受信
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }

  // 遅延停止のチェック
  if (stopDelayActive) {
    if (millis() - stopTimer >= 2000) { // 2秒経過したら
      stepperActive = false;
      stopDelayActive = false;
      Serial.println("INFO: M3 STOPPED (Delayed)");
    }
  }

  // ステッピングモーター駆動
  if (stepperActive) {
    // 少量ずつ回してノンブロッキング風にする
    myStepper.step(50);
  }
}

void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.startsWith("M1:")) {
    // Feeder Speed
    int val = cmd.substring(3).toInt();
    SPEED_FEEDER = constrain(val, 0, 255);
    if (stepperActive) analogWrite(PIN_M1_FEEDER, SPEED_FEEDER);
    Serial.print("OK: M1 Speed="); Serial.println(SPEED_FEEDER);
    
  } else if (cmd.startsWith("M2:")) {
    // Conv A Speed
    int val = cmd.substring(3).toInt();
    SPEED_CONV_A = constrain(val, 0, 255);
    if (stepperActive) analogWrite(PIN_M2_CONV_A, SPEED_CONV_A);
    Serial.print("OK: M2 Speed="); Serial.println(SPEED_CONV_A);

  } else if (cmd.startsWith("M3:")) {
    // Stepper Speed (0-255 input -> Scale to 0-30 RPM for safety?)
    // User wants direct control, but >30 RPM usually stalls.
    // Let's map 0-255 to 1-60 RPM.
    int val = cmd.substring(3).toInt();
    // SPEED_STEPPER_RPM = map(val, 0, 255, 1, 60); 
    // Direct mapping can be dangerous if val is high, but let's trust user or clamp.
    // Let's keep it raw but clamp sane max.
    SPEED_STEPPER_RPM = constrain(val, 1, 60); 
    myStepper.setSpeed(SPEED_STEPPER_RPM);
    Serial.print("OK: M3 RPM="); Serial.println(SPEED_STEPPER_RPM);
    
  } else if (cmd == "STOP" || cmd == "0") {
    stopDCMotors();
    if (stepperActive) {
      stopDelayActive = true;
      stopTimer = millis();
      Serial.println("OK: STOPPING (M3 in 1s)");
    } else {
      Serial.println("OK: STOPPED (Already)");
    }

  } else if (cmd == "START" || cmd == "1") {
    stopDelayActive = false;
    startAll();
    Serial.println("OK: STARTED");
  }
}

void startAll() {
  // 全始動
  stepperActive = true;
  analogWrite(PIN_M1_FEEDER, SPEED_FEEDER);
  analogWrite(PIN_M2_CONV_A, SPEED_CONV_A);
}

void stopAll() {
  stopDCMotors();
  stepperActive = false;
  stopDelayActive = false;
}

void stopDCMotors() {
  analogWrite(PIN_M1_FEEDER, 0);
  analogWrite(PIN_M2_CONV_A, 0);
}
