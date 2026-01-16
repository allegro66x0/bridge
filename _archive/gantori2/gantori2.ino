#include <AccelStepper.h>

// --- ピン設定 ---
// Y軸
const int y_swPin = 2;
const int y_dirPin = 4;
const int y_stepPin = 3;
AccelStepper yAxis(AccelStepper::DRIVER, y_stepPin, y_dirPin);

// X軸
const int x_swPin = 9;
const int x_dirPin = 7;
const int x_stepPin = 6;
AccelStepper xAxis(AccelStepper::DRIVER, x_stepPin, x_dirPin);

// Z軸
const int z_swPin = 5;
const int z_dirPin = 12;
const int z_stepPin = 11;
// AccelStepper zAxis(AccelStepper::DRIVER, z_stepPin, z_dirPin);

// --- 制御パラメータ ---
const float MAX_SPEED = 1000.0;
const float ACCELERATION = 500.0;
const float HOMING_SPEED = -500.0; // 原点(スイッチ)へ向かう速度

// --- シリアル通信用 ---
char buffer[10]; 
int bufIndex = 0;

void setup() {
  Serial.begin(9600);
  
  // スイッチピン設定 (INPUT_PULLUP)
  pinMode(y_swPin, INPUT_PULLUP);
  pinMode(x_swPin, INPUT_PULLUP);
  
  // モーター設定
  yAxis.setMaxSpeed(MAX_SPEED);
  yAxis.setAcceleration(ACCELERATION);
  // Y軸反転設定 (環境に合わせて変更してください)
  yAxis.setPinsInverted(true, false, false); 

  xAxis.setMaxSpeed(MAX_SPEED);
  xAxis.setAcceleration(ACCELERATION);

  // --- 初期化動作: 原点復帰 ---
  // 起動時も物理的にスイッチに当てる
  homeAxis(&xAxis, x_swPin);
  homeAxis(&yAxis, y_swPin);

  // 準備完了の合図
  Serial.println("READY"); 
}

void loop() {
  // シリアル通信受信
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    // 数字ならバッファへ
    if (isDigit(c) && bufIndex < 4) {
      buffer[bufIndex++] = c;
    }
    
    // 改行コード(\n)が来たらコマンド確定
    if (c == '\n') {
      if (bufIndex == 4) {
        buffer[4] = '\0'; // 終端
        
        // 解析 "0000" -> x=0, y=0
        char xStr[3] = {buffer[0], buffer[1], '\0'};
        char yStr[3] = {buffer[2], buffer[3], '\0'};
        int targetX = atoi(xStr);
        int targetY = atoi(yStr);

        // 移動実行
        executeMoveSequence(targetX, targetY);
        
        // 動作完了後、Pythonへ合図
        Serial.println("READY");
      }
      // バッファリセット
      bufIndex = 0;
    }
  }
}

// --- 移動シーケンス ---
void executeMoveSequence(int x, int y) {
  long xSteps = calculateXSteps(x);
  long ySteps = calculateYSteps(y);

  // 1. 指定位置へ移動
  xAxis.moveTo(xSteps);
  yAxis.moveTo(ySteps);

  while (xAxis.distanceToGo() != 0 || yAxis.distanceToGo() != 0) {
    xAxis.run();
    yAxis.run();
  }

  // 2. 待機（コマを置く動作）
  delay(1000); 

  // 3. ★変更点★ 毎ターン物理原点復帰を行う
  // 単に0に戻るのではなく、スイッチに当てて位置ズレを直す
  homeAxis(&xAxis, x_swPin);
  homeAxis(&yAxis, y_swPin);
}

// --- 物理原点復帰関数 ---
void homeAxis(AccelStepper* axis, int swPin) {
  // すでに押されている場合は少し離れる動作を入れても良いが、
  // 今回はシンプルに「押されていなければスイッチに向かって進む」
  if (digitalRead(swPin) == HIGH) {
    // 非常に遠い位置を目標にして、スイッチに当たるまで回し続ける
    axis->moveTo(100000 * (HOMING_SPEED > 0 ? 1 : -1));
    axis->setSpeed(HOMING_SPEED); // 定速移動モード
    
    while (digitalRead(swPin) == HIGH) {
      axis->runSpeed();
    }
  }
  
  // スイッチ検出後、即停止
  axis->stop();
  axis->setCurrentPosition(0); // ここを「0」として位置情報をリセット
  axis->setSpeed(0);
  
  // 通常移動のための目標位置をリセット
  axis->moveTo(0); 
}

// --- 座標計算 (前回の修正版: 0=左, 12=右) ---

// X軸: 0(左) -> 512steps, 12(右) -> 42steps
long calculateXSteps(int x) {
  // 式: Steps = 512 - (39.17 * x)
  float steps = 512.0 - (39.17 * x);
  return (long)steps;
}

// Y軸: 0(奥) -> 1020steps, 12(手前) -> 176steps
long calculateYSteps(int y) {
  // 式: Steps = 1020 - (70.33 * y)
  float steps = 1020.0 - (70.33 * y);
  return (long)steps;
}