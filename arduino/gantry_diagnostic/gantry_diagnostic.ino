// --- Gantry Diagnostic Firmware ---
// 目的: ハードウェア接続確認のみを行う単純なコード
// 機能:
// 1. 起動時に全軸を少しだけ強制的に動かす (スイッチ無視) -> モーター結線確認
// 2. ループ内でリミットスイッチの状態を表示し続ける -> スイッチ結線確認

const int x_swPin = 5;
const int x_dirPin = 7;
const int x_stepPin = 6;

const int y_swPin = 2;
const int y_dirPin = 4;
const int y_stepPin = 3;

const int z_swPin = 9;
const int z_dirPin = 12;
const int z_stepPin = 11;

const int magnetPin = 10;

void setup() {
  Serial.begin(9600);
  Serial.println("--- DIAGNOSTIC MODE ---");

  pinMode(x_swPin, INPUT_PULLUP);
  pinMode(x_dirPin, OUTPUT);
  pinMode(x_stepPin, OUTPUT);

  pinMode(y_swPin, INPUT_PULLUP);
  pinMode(y_dirPin, OUTPUT);
  pinMode(y_stepPin, OUTPUT);

  pinMode(z_swPin, INPUT_PULLUP);
  pinMode(z_dirPin, OUTPUT);
  pinMode(z_stepPin, OUTPUT);

  pinMode(magnetPin, OUTPUT);

  // --- 強制動作テスト (Force Move) ---
  Serial.println("Testing Motors (Force Move 200 steps)...");

  // Z Force
  Serial.print("Moving Z... ");
  moveForce(z_stepPin, z_dirPin, LOW, 200);
  Serial.println("Done.");

  // X Force
  Serial.print("Moving X... ");
  moveForce(x_stepPin, x_dirPin, LOW, 200);
  Serial.println("Done.");

  // Y Force
  Serial.print("Moving Y... ");
  moveForce(y_stepPin, y_dirPin, HIGH, 200);
  Serial.println("Done.");

  Serial.println("--- SENSOR TEST STARTING ---");
  Serial.println("Press switches to see change (0=Pressed/LOW, 1=Open/HIGH)");
}

void loop() {
  int xVal = digitalRead(x_swPin);
  int yVal = digitalRead(y_swPin);
  int zVal = digitalRead(z_swPin);

  Serial.print("X_SW:");
  Serial.print(xVal);
  Serial.print("  Y_SW:");
  Serial.print(yVal);
  Serial.print("  Z_SW:");
  Serial.println(zVal);

  delay(500); // 0.5秒おきに表示
}

void moveForce(int stepPin, int dirPin, int dir, int steps) {
  digitalWrite(dirPin, dir);
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(2000);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(2000);
  }
}
