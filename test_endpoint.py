#!/usr/bin/env python3
"""
test_endpoint.py - Tests YOUR Vercel endpoint
"""

import requests
import json
import time

def test_vercel_endpoint():
    print("=" * 70)
    print("🧪 TESTING YOUR VERCEL ENDPOINT")
    print("=" * 70)
    
    # ⚠️ UPDATE THESE WITH YOUR VERCEL VALUES ⚠️
    VERCEL_URL = "https://agentic-honeypot.vercel.app/api/honeypot"
    API_KEY = "your-vercel-api-key"  # Will be set in Vercel
    
    print(f"🌐 URL: {VERCEL_URL}")
    print(f"🔑 API Key: {API_KEY[:15]}...")
    print()
    
    headers = {
        'x-api-key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Test format that GUVI uses
    payload = {
        "sessionId": "guvi-test-123",
        "message": {
            "sender": "scammer",
            "text": "Your bank account will be blocked today. Verify immediately.",
            "timestamp": 1769776085000
        },
        "conversationHistory": [],
        "metadata": {
            "channel": "SMS",
            "language": "English",
            "locale": "IN"
        }
    }
    
    print("📤 Sending GUVI format request...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        start = time.time()
        response = requests.post(
            VERCEL_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        elapsed = time.time() - start
        
        print(f"\n📊 RESPONSE:")
        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS! Response:")
            print(json.dumps(result, indent=2))
            
            # Check required fields
            if result.get('status') == 'success' and 'reply' in result:
                print(f"\n🎉 PERFECT! This will pass GUVI tester!")
            else:
                print(f"\n⚠️  Missing required fields")
                
        else:
            print(f"\n❌ FAILED: {response.text}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_vercel_endpoint()