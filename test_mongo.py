
import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("MONGO_URI")

print(f"Testing URI: {uri}")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)

try:
    print("Checking server status...")
    info = client.server_info()
    print("Server info obtained successfully.")
    
    print("\nTopology Description:")
    print(client.topology_description)
    
    print("\nChecking Replica Set Status...")
    rs_status = client.admin.command("replSetGetStatus")
    print(f"Replica Set Name: {rs_status.get('set')}")
    for member in rs_status.get('members', []):
        print(f"  - {member.get('name')}: {member.get('stateStr')}")
        
except ServerSelectionTimeoutError as e:
    print(f"\nServerSelectionTimeoutError: {e}")
    print("\nDetailed Topology:")
    print(client.topology_description)
except ConnectionFailure as e:
    print(f"\nConnectionFailure: {e}")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    client.close()
