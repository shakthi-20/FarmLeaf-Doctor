#include <DHT.h>

#define DHTPIN 4        // GPIO4 for DHT11
#define DHTTYPE DHT11   // DHT 11 sensor type

DHT dht(DHTPIN, DHTTYPE);

int soilMoisturePin = 32; // GPIO34 for Soil Moisture sensor

void setup() {
  Serial.begin(115200);  // Start serial communication
  dht.begin();           // Initialize the DHT sensor
}

void loop() {
  float temp = dht.readTemperature();
  float humidity = dht.readHumidity();
  int soilMoistureRaw = analogRead(soilMoisturePin);  // Read raw ADC value

  if (isnan(temp) || isnan(humidity)) {
    Serial.println("<start>Error reading from DHT11<end>");
  } else {
    Serial.print("<start>temperature=");
    Serial.print(temp);
    Serial.print(",humidity=");
    Serial.print(humidity);
    Serial.print(",soil_moisture_raw=");
    Serial.print(soilMoistureRaw);
    Serial.println("<end>");
  }

  delay(2000);  // Wait 2 seconds before next reading
}