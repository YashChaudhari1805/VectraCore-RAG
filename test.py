import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

if token:
    print(f"Token Found: {token[:4]}...{token[-4:]}")
    print(f"Token Length: {len(token)}")
else:
    print("Error: HF_TOKEN not found in environment.")