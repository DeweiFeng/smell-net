import requests
import json

API_TOKEN = "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"

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://huggingface.co/datasets/DeweiFeng/smell-net"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()



# Pretty-print the Croissant metadata
print(json.dumps(data, indent=2))