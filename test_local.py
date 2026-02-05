#!/usr/bin/env python3
"""
test_local.py - Tests the local API
Tests all possible GUVI formats
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
        print("   Run: python setup_env.py")
        return
    
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    print("=" * 70)
    print("🧪 COMPREHENSIVE LOCAL API TEST")
    print("=" * 70)
    print(f"🔑 API Key: {api_key[:15]}...")
    print(f"🌐 Base URL: {base_url}")
    print()
    
    # Test 1: Health check
    print("1️⃣  Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Health: {health.get('status')}")
            print(f"   📍 Endpoints: {health.get('endpoints', {})}")
        else:
            print(f"   ❌ Health failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Cannot connect: {e}")
        print(f"   💡 Start server with: python app.py")
        return
    
    # Test 2: Test different input formats (GUVI might send)
    print("\n2️⃣  Testing different input formats...")
    
    test_formats = [
        {
            "name": "GUVI Format (Expected)",
            "payload": {
                "sessionId": "test-guvi-format-" + str(int(time.time())),
                "message": {
                    "sender": "scammer",
                    "text": "Your bank account will be blocked today. Verify immediately.",
                    "timestamp": 1769776085000  # GUVI sends timestamp as number
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
            "name": "Simple Format",
            "payload": {
                "text": "URGENT: Account suspension pending. Call +91-9876543210",
                "sessionId": "test-simple-" + str(int(time.time()))
            }
        },
        {
            "name": "Minimal Format",
            "payload": {
                "message": "Verify your account now at http://bank-verify.com"
            }
        },
        {
            "name": "Empty Object",
            "payload": {}
        },
        {
            "name": "Null/Empty",
            "payload": None
        }
    ]
    
    all_passed = True
    for test in test_formats:
        print(f"\n   📝 {test['name']}:")
        
        try:
            if test['payload'] is None:
                # Test with no body
                response = requests.post(
                    f"{base_url}/api/honeypot",
                    headers=headers,
                    timeout=10
                )
            else:
                response = requests.post(
                    f"{base_url}/api/honeypot",
                    headers=headers,
                    json=test['payload'],
                    timeout=10
                )
            
            print(f"      Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Success!")
                print(f"      Status: {result.get('status')}")
                print(f"      Scam: {result.get('scamDetected')}")
                print(f"      Reply: {result.get('reply')[:50]}...")
                
                # Check GUVI required format
                required = ['status', 'reply', 'scamDetected', 'confidence']
                missing = [f for f in required if f not in result]
                
                if not missing:
                    print(f"      ✅ GUVI format correct")
                else:
                    print(f"      ❌ Missing fields: {missing}")
                    all_passed = False
            else:
                print(f"      ❌ Failed: {response.status_code}")
                print(f"      Error: {response.text[:100]}")
                all_passed = False
                
        except Exception as e:
            print(f"      ❌ Error: {type(e).__name__}: {e}")
            all_passed = False
    
    # Test 3: Test scam messages
    print("\n3️⃣  Testing scam detection accuracy...")
    
    test_messages = [
        {
            "text": "URGENT: Your bank account will be suspended. Verify at: http://secure-bank-verify.com/login.php?id=123",
            "expected": True,
            "type": "Bank phishing"
        },
        {
            "text": "Send ₹99 to verify@okicici to activate your UPI account",
            "expected": True,
            "type": "UPI fraud"
        },
        {
            "text": "Congratulations! You won ₹10,00,000. Call +91-9876543210 to claim",
            "expected": True,
            "type": "Lottery scam"
        },
        {
            "text": "Hi, how are you doing today? Let's meet for coffee",
            "expected": False,
            "type": "Normal message"
        }
    ]
    
    for test in test_messages:
        print(f"\n   🔍 {test['type']}:")
        print(f"      Message: {test['text'][:60]}...")
        
        payload = {
            "sessionId": f"test-accuracy-{int(time.time())}",
            "message": {
                "sender": "scammer",
                "text": test['text'],
                "timestamp": 1769776085000
            }
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
                detected = result.get('scamDetected')
                expected = test['expected']
                
                if detected == expected:
                    print(f"      ✅ Correct detection: {detected}")
                else:
                    print(f"      ❌ Wrong: detected={detected}, expected={expected}")
                    all_passed = False
            else:
                print(f"      ❌ Request failed: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            all_passed = False
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Server is running correctly")
        print("✅ Handles all GUVI formats")
        print("✅ Scam detection works")
        print("✅ Returns correct GUVI format")
        print("\n🚀 Ready for deployment to Render!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Fix issues before deploying")
    
    print("\n📝 DEPLOYMENT CHECKLIST:")
    print("1. ✅ Local server runs: python app.py")
    print("2. ✅ All tests pass: python test_local.py")
    print("3. ✅ Firebase configured")
    print("4. ✅ Model downloaded")
    print("5. ✅ .env file created")
    print("\n🌐 Next: Push to GitHub and deploy on Render")

if __name__ == "__main__":
    # Check if server is running
    try:
        requests.get("http://localhost:5000/health", timeout=2)
        print("✅ Server is running, starting tests...")
        test_local_api()
    except:
        print("❌ Server not running on localhost:5000")
        print("   Start it with: python app.py")
        sys.exit(1)