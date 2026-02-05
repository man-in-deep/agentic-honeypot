#!/usr/bin/env python3
"""
test_guvi_format.py - Tests ALL GUVI formats
"""

import requests
import json
import time

def test_all_formats():
    print("=" * 70)
    print("🧪 TESTING ALL GUVI FORMATS")
    print("=" * 70)
    
    # Your endpoint
    ENDPOINT = "https://agentic-honeypot.vercel.app/api/honeypot"
    API_KEY = "your-vercel-api-key"
    
    headers = {
        'x-api-key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    # ALL possible formats GUVI might send
    test_formats = [
        {
            "name": "Full GUVI format (timestamp as string)",
            "payload": {
                "sessionId": "guvi-test-123",
                "message": {
                    "sender": "scammer",
                    "text": "Your bank account will be blocked today. Verify immediately.",
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
            "name": "GUVI format (timestamp as number)",
            "payload": {
                "sessionId": "guvi-test-456",
                "message": {
                    "sender": "scammer",
                    "text": "Your account will be suspended.",
                    "timestamp": 1769776085000
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
            "name": "Minimal format",
            "payload": {
                "sessionId": "test-789",
                "message": {
                    "text": "Verify your account now"
                }
            }
        },
        {
            "name": "Simple format",
            "payload": {
                "text": "Your account needs verification"
            }
        }
    ]
    
    for test in test_formats:
        print(f"\n🔍 Testing: {test['name']}")
        
        try:
            response = requests.post(
                ENDPOINT,
                headers=headers,
                json=test['payload'],
                timeout=20
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success!")
                print(f"   Status field: {result.get('status')}")
                print(f"   Reply: {result.get('reply', '')[:50]}...")
                
                if result.get('status') == 'success':
                    print(f"   🎉 Will pass GUVI tester!")
                else:
                    print(f"   ⚠️  Status not 'success'")
                    
            else:
                print(f"   ❌ Failed: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    test_all_formats()