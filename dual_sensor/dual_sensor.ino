/*
 * Dual Sensor Data Acquisition: PPG (A0) + GSR (A1)
 * BME Lab 3 - DIY: Music & Autonomic Nervous System
 *
 * Outputs CSV over serial: timestamp_ms,ppg_raw,gsr_raw
 * Sample rate: 100 Hz (10 ms interval)
 * Baud rate: 115200
 *
 * WIRING:
 *   PulseSensor:  purple -> A0, red -> 5V, black -> GND
 *   Grove GSR:    yellow -> A1, red -> 5V, black -> GND (second GND pin)
 */

const int PPG_PIN = A0;
const int GSR_PIN = A1;
const unsigned long SAMPLE_INTERVAL_MS = 10;  // 100 Hz

unsigned long lastSampleTime = 0;
bool headerSent = false;

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port to connect (needed for some boards)
    }
    delay(500);
    Serial.println("time_ms,ppg,gsr");
    headerSent = true;
}

void loop() {
    unsigned long now = millis();
    if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = now;

        int ppgVal = analogRead(PPG_PIN);
        int gsrVal = analogRead(GSR_PIN);

        Serial.print(now);
        Serial.print(",");
        Serial.print(ppgVal);
        Serial.print(",");
        Serial.println(gsrVal);
    }
}
