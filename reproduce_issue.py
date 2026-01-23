
import sys
import os
import json
from dotenv import load_dotenv

sys.path.append("/home/codie/Documents/developer/tinnten/tinnten-embedding")
os.environ["REQUIRE_KEYCLOAK_AUTH"] = "false"
load_dotenv()

from app import app

# Create a test client
with app.test_client() as client:
    print("Sending POST request to /api/v10/websearch/search")
    payload = {
        "text": "test query",
        "k": 5
    }
    try:
        response = client.post("/api/v10/websearch/search", json=payload)
        print(f"Status Code: {response.status_code}")
        data = response.get_json()
        print(f"Response Body: {json.dumps(data, indent=2)}")
        
        if data and "results" in data:
            results = data["results"]
            print(f"Result Count: {len(results)}")
            if len(results) == 0:
                print("WARNING: Empty results returned.")
        else:
            print("WARNING: 'results' key missing in response.")
            
    except Exception as e:
        print(f"Request failed: {e}")
