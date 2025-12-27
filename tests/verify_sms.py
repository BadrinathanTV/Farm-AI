
import requests
import time
import sys

# Wait for server to start
print("Waiting for server to start...")
time.sleep(3)

url = "http://localhost:8000/sms"
data = {
    "Body": "Hello. What crops can I grow in clay soil?",
    "From": "+15551234567",
    "To": "+15557654321"
}

print(f"Sending POST request to {url}...")
try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")
    
    if response.status_code == 200 and "<Response>" in response.text:
        print("SUCCESS: Received valid TwiML response.")
    else:
        print("FAILURE: Invalid response.")
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
