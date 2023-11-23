import random
import time
import yagmail

# Mock sensor data generator
def generate_sensor_data():
    return {
        'humidity': random.uniform(30, 70),
        'gyroscopic': random.uniform(0, 5),
        'brightness': random.uniform(0, 1000)
    }

# Mock function to detect uncharacteristic spikes
def detect_anomalies(data):
    return data['humidity'] > 90 or data['gyroscopic'] > 10 or data['brightness'] > 800

# Function to send an email alert
def send_email_alert():
    # Replace with your email credentials and recipient email
    sender_email = 'sillfingmongoose@gmail.com'
    recipient_email = 'cgorgen@uiowa.edu'
    password = ''

    subject = 'Emergency Alert'
    body = 'Uncharacteristic spike detected in sensor data!'

    yag = yagmail.SMTP(sender_email, password)
    yag.send(
        to=recipient_email,
        subject=subject,
        contents=body
    )

# Main loop for data monitoring
while True:
    sensor_data = generate_sensor_data()
    if detect_anomalies(sensor_data):
        print("Anomaly detected. Sending email alert...")
        send_email_alert()
    
    # Adjust the sleep duration to control data flow rate
    time.sleep(5)  # Sleep for 5 seconds
