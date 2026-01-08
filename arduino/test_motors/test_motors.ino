/**
 * モーター動作確認用テストプログラム (Stepper対応版)
 *
 * M1: フィーダー (Pin 5) - DC
 * M2: コンベアA (Pin 3) - DC
 * M3: コンベアB (Pin 10,11,12,13) - Stepper (仮定)
 */

#include <Stepper.h>

// ピン定義
const int PIN_M1_FEEDER = 5;
const int PIN_M2_CONV_A = 3;

// ステッピングモーター設定
const int STEPS_PER_REV = 2048; // 一般的な28BYJ-48の場合
// ピン順序は 10, 12, 11, 13 (IN1, IN3, IN2, IN4) が一般的ですが、
// 動かない場合は 10, 11, 12, 13 を試してください。
Stepper myStepper(STEPS_PER_REV, 10, 12, 11, 13);

const int TEST_SPEED_DC = 100;

void setup() {
  Serial.begin(9600);
  pinMode(PIN_M1_FEEDER, OUTPUT);
  pinMode(PIN_M2_CONV_A, OUTPUT);

  myStepper.setSpeed(10); // 10 RPM

  Serial.println("--- Motor Test Start (Stepper Added) ---");
}

void loop() {
  // 1. フィーダー
  Serial.println("Testing Feeder (Pin 5)...");
  analogWrite(PIN_M1_FEEDER, TEST_SPEED_DC);
  delay(1000);
  analogWrite(PIN_M1_FEEDER, 0);
  delay(500);

  // 2. コンベアA
  Serial.println("Testing Conveyor A (Pin 3)...");
  analogWrite(PIN_M2_CONV_A, TEST_SPEED_DC);
  delay(1000);
  analogWrite(PIN_M2_CONV_A, 0);
  delay(500);

  // 3. コンベアB (Stepper)
  Serial.println("Testing Conveyor B (Stepper 10-13)...");
  // 1回転分回す (ブロックします)
  myStepper.step(STEPS_PER_REV / 4); // 1/4回転
  delay(500);

  Serial.println("--- Loop End ---");
  delay(1000);
}
