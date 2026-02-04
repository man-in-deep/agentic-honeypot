#!/usr/bin/env python3
"""
debug_guvi_tester.py
Debug what GUVI tester might be sending
"""

import requests
import json

def test_guvi_format():
    """Test what format GUVI might be using"""
    
    YOUR_URL = "https://agentic-honeypot-0p4k.onrender.com/api/honeypot"
    YOUR_API_KEY = "a3da6ac1825ebe8a1dce3520e36657d2"
    
    print("Testing different request formats...")
    print()
    
    headers = {
        'x-api-key': YOUR_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Test different possible formats GUVI might use
    test_formats = [
        {
            "name": "Format 1: Full format",
            "payload": {
                "sessionId": "guvi-test-123",
                "message": {
                    "sender": "scammer",
                    "text": "Your bank account will be blocked. Verify immediately.",
                    "timestamp": "2026-01-21T10:15:30Z"
                },
                "conversationHistory": [],
                "metadata": {
                    "channel": "SMS",
                    "language": "English",
                    "locale": "IN"
                }
            }
        },
        {
            "name": "Format 2: Simple format",
            "payload": {
                "text": "Your account needs verification. Click link now.",
                "sessionId": "test-simple"
            }
        },
        {
            "name": "Format 3: Minimal format",
            "payload": {
                "message": "URGENT: Account suspension pending"
            }
        },
        {
            "name": "Format 4: Empty object",
            "payload": {}
        }
    ]
    
    for test in test_formats:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"   Payload: {json.dumps(test['payload'])}")
        
        try:
            response = requests.post(
                YOUR_URL,
                headers=headers,
                json=test['payload'],
                timeout=20
            )
            
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:150]}...")
            
            if response.status_code == 200:
                print(f"   ✅ Success!")
            else:
                print(f"   ❌ Failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_guvi_format()