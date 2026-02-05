#!/usr/bin/env python3
"""
test_local.py - Local testing
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

def test_local():
    print("=" * 60)
    print("🧪 LOCAL TESTING")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    api_key = os.getenv('API_KEY')
    
    if not api_key:
        print("❌ API_KEY not found in .env")
        return
    
    # Test health
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   ✅ Health: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health failed: {e}")
        return
    
    # Test scam detection
    print("\n2. Testing scam detection...")
    
    test_cases = [
        ("Bank phishing", "URGENT: Your bank account will be blocked. Verify at http://bank-verify.com"),
        ("UPI fraud", "Send ₹99 to scammer@paytm to activate account"),
        ("Normal message", "Hi, how are you doing?")
    ]
    
    for name, message in test_cases:
        print(f"\n   📝 {name}:")
        
        payload = {
            "sessionId": f"test-{int(time.time())}",
            "message": {
                "sender": "scammer",
                "text": message,
                "timestamp": "2026-01-21T10:15:30Z"
            },
            "conversationHistory": [],
            "metadata": {
                "channel": "SMS",
                "language": "English",
                "locale": "IN"
            }
        }
        
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/honeypot",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Status: {result.get('status')}")
                print(f"      🎯 Scam: {result.get('scamDetected')}")
                print(f"      💬 Reply: {result.get('reply')[:50]}...")
            else:
                print(f"      ❌ Failed: {response.status_code}")
                print(f"      Error: {response.text}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ LOCAL TESTING COMPLETE")

if __name__ == "__main__":
    test_local()