"""
Independent Backend API Testing Script
-------------------------------------
Use this script to test all backend endpoints from the terminal:
1. GET  /api/health
2. GET  /api/status
3. GET  /api/models
4. POST /api/chat

Run with:
    python test_api.py
"""

import sys
import json
import httpx

BASE_URL = "http://127.0.0.1:8000"


def print_step(title):
    print("\n" + "=" * 60)
    print(f"👉 {title}")
    print("=" * 60)


def main():
    print(f"Testing OpenRouter Chatbot Backend at {BASE_URL}...")

    with httpx.Client(timeout=30.0) as client:
        # 1. Health Check
        print_step("1. Testing GET /api/health")
        try:
            res = client.get(f"{BASE_URL}/api/health")
            print(f"Status Code: {res.status_code}")
            print(f"Response: {json.dumps(res.json(), indent=2)}")
        except Exception as e:
            print(f"❌ Failed to reach backend: {e}")
            print("Make sure you started the backend server with: uvicorn main:app --reload")
            sys.exit(1)

        # 2. Connection Status (OpenRouter API Key Verification)
        print_step("2. Testing GET /api/status (Key & Connectivity Test)")
        try:
            res = client.get(f"{BASE_URL}/api/status")
            print(f"Status Code: {res.status_code}")
            status_data = res.json()
            print(f"Response: {json.dumps(status_data, indent=2)}")

            if status_data.get("connected"):
                print("✓ OpenRouter API key is VALID and active!")
            else:
                print("⚠️ OpenRouter connection is NOT connected yet.")
                print(f"Details: {status_data.get('message')}")
        except Exception as e:
            print(f"❌ Error during status check: {e}")

        # 3. Model Discovery
        print_step("3. Testing GET /api/models (Model Discovery)")
        selected_model = None
        try:
            res = client.get(f"{BASE_URL}/api/models")
            print(f"Status Code: {res.status_code}")
            data = res.json()
            models = data.get("models", [])
            selected_model = data.get("selected_model")

            print(f"Total Discovered Chat Models: {data.get('total_count', len(models))}")
            print(f"Auto-selected Default Model: {selected_model}")
            print("\nTop 5 Discovered Models:")
            for idx, m in enumerate(models[:5], 1):
                badge = "[FREE]" if m.get("is_free") else "[PAID]"
                print(f"  {idx}. {badge} {m.get('id')} ({m.get('name')})")
        except Exception as e:
            print(f"❌ Error during model discovery: {e}")

        # 4. Chat Completion Test
        print_step("4. Testing POST /api/chat (Completion Proxy Test)")
        if not selected_model:
            selected_model = "meta-llama/llama-3.3-70b-instruct:free"

        test_payload = {
            "model": selected_model,
            "messages": [
                {"role": "user", "content": "Hello! Reply with exactly: 'Backend test successful.'"}
            ],
            "temperature": 0.5
        }

        try:
            print(f"Sending prompt to model: {selected_model}...")
            res = client.post(f"{BASE_URL}/api/chat", json=test_payload)
            print(f"Status Code: {res.status_code}")
            if res.status_code == 200:
                chat_data = res.json()
                print("✓ Chat response received successfully!")
                print(f"Model: {chat_data.get('model')}")
                print(f"Message Content:\n{chat_data.get('message')}")
            else:
                print(f"⚠️ Chat request returned error: {res.text}")
        except Exception as e:
            print(f"❌ Error during chat completion test: {e}")

    print("\n" + "=" * 60)
    print("Test run completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
