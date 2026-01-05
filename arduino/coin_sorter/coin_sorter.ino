/**
 * Coin Sorter Control Firmware
 * Controls 3 DC Motors via PWM (MOSFET/Motor Driver)
 * 
 * Protocol:
 * - "START": Turn on all motors at default speed.
 * - "STOP": Turn off all motors.
 * - "M1:xxx": Set Motor 1 (Feeder) speed (0-255).
 * - "M2:xxx": Set Motor 2 (Conveyor A) speed (0-255).
 * - "M3:xxx": Set Motor 3 (Conveyor B) speed (0-255).
 */

// --- PIN DEFINITIONS (PWM capable pins on Arduino Uno: 3, 5, 6, 9, 10, 11) ---
const int PIN_M1_FEEDER = 3;   // 回転式コインフェーダー
const int PIN_M2_CONV_A = 5;   // ベルトコンベア 1
const int PIN_M3_CONV_B = 6;   // ベルトコンベア 2

// --- DEFAULT SPEEDS (Can be tuned) ---
const int DEFAULT_SPEED_FEEDER = 200;
const int DEFAULT_SPEED_CONV   = 180;

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  
  pinMode(PIN_M1_FEEDER, OUTPUT);
  pinMode(PIN_M2_CONV_A, OUTPUT);
  pinMode(PIN_M3_CONV_B, OUTPUT);
  
  // Initialize stopped
  stopAll();
  
  Serial.println("READY: Coin Sorter");
}

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
  cmd.trim(); // Remove whitespace
  
  if (cmd == "STOP") {
    stopAll();
    Serial.println("OK: STOPPED");
  }
  else if (cmd == "START") {
    startSequence();
    Serial.println("OK: STARTED");
  }
  else if (cmd.startsWith("M1:")) {
    int speed = cmd.substring(3).toInt();
    setMotor(PIN_M1_FEEDER, speed);
    Serial.println("OK: M1=" + String(speed));
  }
  else if (cmd.startsWith("M2:")) {
    int speed = cmd.substring(3).toInt();
    setMotor(PIN_M2_CONV_A, speed);
    Serial.println("OK: M2=" + String(speed));
  }
  else if (cmd.startsWith("M3:")) {
    int speed = cmd.substring(3).toInt();
    setMotor(PIN_M3_CONV_B, speed);
    Serial.println("OK: M3=" + String(speed));
  }
}

void setMotor(int pin, int speed) {
  // Constrain speed to 0-255
  if (speed < 0) speed = 0;
  if (speed > 255) speed = 255;
  analogWrite(pin, speed);
}

void stopAll() {
  analogWrite(PIN_M1_FEEDER, 0);
  analogWrite(PIN_M2_CONV_A, 0);
  analogWrite(PIN_M3_CONV_B, 0);
}

void startSequence() {
  // Start conveyors first, then feeder (optional logic)
  analogWrite(PIN_M2_CONV_A, DEFAULT_SPEED_CONV);
  analogWrite(PIN_M3_CONV_B, DEFAULT_SPEED_CONV);
  delay(100); 
  analogWrite(PIN_M1_FEEDER, DEFAULT_SPEED_FEEDER);
}
