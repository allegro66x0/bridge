/**
 * Electromagnet Control (Magnet Only)
 * 
 * [Spec]
 * - Controls Electromagnet on Pin 13.７ピンに変更
 * - Commands: '1' (ON), '0' (OFF).
 */

const int magnetPin = 7;

void setup() {
  Serial.begin(9600);
  pinMode(magnetPin, OUTPUT);
  digitalWrite(magnetPin, LOW); // Start OFF
  Serial.println("READY: Magnet Control");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '1') {
      digitalWrite(magnetPin, HIGH);
      Serial.println("OK: Magnet ON");
    } else if (c == '0') {
      digitalWrite(magnetPin, LOW);
      Serial.println("OK: Magnet OFF");
    }
  }
}
