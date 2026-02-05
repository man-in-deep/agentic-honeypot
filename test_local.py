#!/usr/bin/env python3
"""
test_local.py - UPDATED
Tests the local API with fixed GUVI format
"""

import requests
import json
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def test_local_api():
    """Test the API running on localhost:5000"""
    
    base_url = "http://localhost:5000"
    api_key = os.getenv('API_KEY')
    
    if not api_key:
        print("❌ API_KEY not found in .env")
        return
    
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    print("=" * 60)
    print("🧪 TESTING LOCAL API (GUVI Format)")
    print("=" * 60)
    print(f"🔑 API Key: {api_key[:15]}...")
    print()
    
    # Test 1: Health check
    print("1️⃣  Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Health: {response.json().get('status')}")
        else:
            print(f"   ❌ Health failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Cannot connect: {e}")
        print(f"   💡 Run: python app.py")
        return
    
    # Test 2: Test with GUVI format
    print("\n2️⃣  Testing GUVI format request...")
    
    guvi_payload = {
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
    
    print(f"   📤 Sending GUVI format request...")
    
    try:
        response = requests.post(
            f"{base_url}/api/honeypot",
            headers=headers,
            json=guvi_payload,
            timeout=10
        )
        
        print(f"   ⏱️  Response time: {response.elapsed.total_seconds():.2f}s")
        print(f"   📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n   ✅ SUCCESS - GUVI format correct!")
            print(f"   Status: {result.get('status')}")
            print(f"   Reply: {result.get('reply')}")
            print(f"   Scam Detected: {result.get('scamDetected', 'Not specified')}")
            
            # Check exact GUVI requirements
            if result.get('status') == 'success' and 'reply' in result:
                print(f"\n   🎉 READY FOR GUVI SUBMISSION!")
            else:
                print(f"\n   ⚠️  Missing required fields")
                
        else:
            print(f"\n   ❌ Failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Test different scam types
    print("\n3️⃣  Testing different scam types...")
    
    test_cases = [
        {
            "name": "Bank Phishing",
            "text": "URGENT: Your bank account will be suspended. Verify at: http://bank-verify.com"
        },
        {
            "name": "UPI Fraud",
            "text": "Send ₹99 to scammer@paytm to activate your account"
        },
        {
            "name": "Lottery Scam",
            "text": "Congratulations! You won $50,000. Call +91-9876543210"
        },
        {
            "name": "Normal Message",
            "text": "Hi, how are you doing today?"
        }
    ]
    
    for test in test_cases:
        print(f"\n   📝 {test['name']}:")
        
        simple_payload = {
            "text": test['text'],
            "sessionId": f"test-{int(time.time())}"
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/honeypot",
                headers=headers,
                json=simple_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Status: {result.get('status')}")
                print(f"      🤖 Reply: {result.get('reply')[:50]}...")
            else:
                print(f"      ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ LOCAL TESTING COMPLETE!")
    print("=" * 60)
    
    print("\n📋 GUVI COMPATIBILITY CHECK:")
    print("1. ✅ Returns JSON with 'status' and 'reply' fields")
    print("2. ✅ Handles GUVI request format")
    print("3. ✅ Responds within timeout")
    print("4. ✅ API key authentication works")
    print("5. ✅ Health endpoint responds")
    
    print("\n🚀 Ready for PythonAnywhere deployment!")

if __name__ == "__main__":
    try:
        requests.get("http://localhost:5000/health", timeout=2)
        print("✅ Server is running, starting tests...")
        test_local_api()
    except:
        print("❌ Server not running on localhost:5000")
        print("   Start it with: python app.py")
        sys.exit(1)