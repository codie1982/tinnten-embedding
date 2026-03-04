import requests

url = "http://localhost:5001/api/v10/categories/search"
payload = {
    "text": "telefon",
    "k": 1
}
try:
    response = requests.post(url, json=payload)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
