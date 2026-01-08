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
const int SPEED_FEEDER = 5;       // DC (0-255)
const int SPEED_CONV_A = 60;      // DC (0-255)
const int SPEED_STEPPER_RPM = 20; // Stepper Speed (変更: 10->20)

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
  Serial.println("READY: Coin Sorter (Stepper Mixed)");
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
  if (cmd == "STOP" || cmd == "0") {
    // DCモーターは即停止
    stopDCMotors();

    // ステッピングモーターは1秒後に停止 (フラグを立てる)
    if (stepperActive) {
      stopDelayActive = true;
      stopTimer = millis();
      Serial.println("OK: STOPPING (M3 in 1s)");
    } else {
      Serial.println("OK: STOPPED (Already)");
    }

  } else if (cmd == "START" || cmd == "1") {
    stopDelayActive = false; // 停止キャンセル
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
  // 強制全停止 (初期化用)
  stopDCMotors();
  stepperActive = false;
  stopDelayActive = false;
}

void stopDCMotors() {
  analogWrite(PIN_M1_FEEDER, 0);
  analogWrite(PIN_M2_CONV_A, 0);
}
