// Electromagnet Test (Pin 13)
// 1 = ON (HIGH), 0 = OFF (LOW)

const int magnetPin = 13;                                                                                                                                       

void setup() {
  Serial.begin(9600);
  pinMode(magnetPin, OUTPUT);
  digitalWrite(magnetPin, LOW); // Start OFF

  Serial.println("--- Electromagnet Test ---");
  Serial.println("Commands:");
  Serial.println(" 1: Turn ON");
  Serial.println(" 0: Turn OFF");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '1') {
      digitalWrite(magnetPin, HIGH);
      Serial.println("Magnet: ON");
    } else if (c == '0') {
      digitalWrite(magnetPin, LOW);
      Serial.println("Magnet: OFF");
    }
  }
}
