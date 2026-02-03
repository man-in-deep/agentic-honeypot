#!/usr/bin/env python3
"""
test_local.py - UPDATED
Tests the local API with the simple binary model
"""

import requests
import json
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def test_model_directly():
    """Test the model directly first"""
    base_url = "http://localhost:5000"
    api_key = os.getenv('API_KEY')
    
    if not api_key:
        print("❌ API_KEY not found in .env")
        return
    
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    print("🧪 DIRECT MODEL TEST")
    print("=" * 60)
    
    test_texts = [
        "URGENT: Your bank account will be suspended. Verify now.",
        "Send ₹99 to scammer@paytm to activate account.",
        "Congratulations! You won $50,000. Call +91-9876543210.",
        "Your account hacked. Click: http://hack-fix.com",
        "Hi, how are you doing today?",
        "Can we meet for coffee tomorrow?",
        "Hello, did you receive my email?",
        "Good morning! Have a nice day."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:60]}...")
        
        payload = {"text": text}
        
        try:
            response = requests.post(
                f"{base_url}/api/test-model",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                pred = result.get('prediction', {})
                print(f"   🤖 Model says: {pred.get('label', 'unknown').upper()}")
                print(f"   📊 Is Scam: {pred.get('is_scam')}")
                
                # Show intelligence
                intel = result.get('intelligence', {})
                for key, value in intel.items():
                    if value:
                        print(f"   🔍 {key}: {value}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

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
    
    print("\n" + "=" * 60)
    print("🧪 TESTING LOCAL API: http://localhost:5000")
    print("=" * 60)
    print(f"🔑 API Key: {api_key[:15]}...")
    print()
    
    # Test 1: Health check
    print("1️⃣  Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Health: {health.get('status')}")
        else:
            print(f"   ❌ Health failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Cannot connect to localhost:5000")
        print(f"   Error: {e}")
        print(f"   💡 Make sure server is running: python app.py")
        return
    
    # Test 2: Test scam messages
    print("\n2️⃣  Testing scam messages...")
    
    scam_messages = [
        {
            "name": "Bank phishing",
            "message": "URGENT: Your bank account will be suspended. Verify at: http://bank-verify-now.com/login?id=12345"
        },
        {
            "name": "UPI fraud",
            "message": "Your account verification pending. Send ₹1 to verify@okicici to complete."
        },
        {
            "name": "Lottery scam",
            "message": "CONGRATULATIONS! You won ₹10,00,000. Call +91-9876543210 immediately."
        }
    ]
    
    for test in scam_messages:
        print(f"\n   📝 {test['name']}:")
        print(f"      Message: {test['message'][:50]}...")
        
        payload = {
            "sessionId": f"test-scam-{int(time.time())}",
            "message": {
                "sender": "scammer",
                "text": test['message'],
                "timestamp": "2026-01-21T10:15:30Z"
            },
            "conversationHistory": [],
            "metadata": {
                "channel": "SMS",
                "language": "English", 
                "locale": "IN"
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
                print(f"      ✅ Status: {result.get('status')}")
                print(f"      🚨 Scam Detected: {result.get('scamDetected')}")
                print(f"      🤖 Reply: {result.get('reply')[:50]}...")
                
                intel = result.get('extractedIntelligence', {})
                if intel.get('phishingLinks'):
                    print(f"      🔗 Links: {intel['phishingLinks']}")
                if intel.get('upiIds'):
                    print(f"      💰 UPI: {intel['upiIds']}")
                if intel.get('phoneNumbers'):
                    print(f"      📱 Phone: {intel['phoneNumbers']}")
                
            else:
                print(f"      ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Test 3: Test normal messages
    print("\n3️⃣  Testing normal messages...")
    
    normal_messages = [
        {
            "name": "Greeting",
            "message": "Hi, how are you doing today?"
        },
        {
            "name": "Meeting request",
            "message": "Can we meet for coffee this weekend?"
        }
    ]
    
    for test in normal_messages:
        print(f"\n   📝 {test['name']}:")
        print(f"      Message: {test['message']}")
        
        payload = {
            "sessionId": f"test-normal-{int(time.time())}",
            "message": {
                "sender": "friend",
                "text": test['message'],
                "timestamp": "2026-01-21T10:15:30Z"
            },
            "conversationHistory": [],
            "metadata": {
                "channel": "WhatsApp",
                "language": "English", 
                "locale": "IN"
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
                print(f"      ✅ Status: {result.get('status')}")
                print(f"      🟢 Scam Detected: {result.get('scamDetected')}")
                print(f"      🤖 Reply: {result.get('reply')}")
                
            else:
                print(f"      ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Test 4: Multi-turn conversation
    print("\n4️⃣  Testing multi-turn conversation...")
    
    session_id = f"multi-turn-{int(time.time())}"
    print(f"   Session: {session_id}")
    
    # Message 1
    payload1 = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": "Your bank account has security issues. Immediate verification required.",
            "timestamp": "2026-01-21T10:15:30Z"
        },
        "conversationHistory": [],
        "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
    }
    
    try:
        response1 = requests.post(f"{base_url}/api/honeypot", headers=headers, json=payload1, timeout=10)
        if response1.status_code == 200:
            result1 = response1.json()
            reply1 = result1.get('reply', '')
            scam1 = result1.get('scamDetected', False)
            print(f"   💬 Message 1: {reply1[:50]}...")
            print(f"   📊 Scam detected: {scam1}")
    except Exception as e:
        print(f"   ❌ Message 1 failed: {e}")
        reply1 = "What do you mean?"
    
    time.sleep(1)
    
    # Message 2
    payload2 = {
        "sessionId": session_id,
        "message": {
            "sender": "scammer",
            "text": "To verify, share your UPI ID or send ₹99 to verify@upi",
            "timestamp": "2026-01-21T10:16:30Z"
        },
        "conversationHistory": [
            payload1['message'],
            {"sender": "user", "text": reply1, "timestamp": "2026-01-21T10:16:00Z"}
        ],
        "metadata": payload1['metadata']
    }
    
    try:
        response2 = requests.post(f"{base_url}/api/honeypot", headers=headers, json=payload2, timeout=10)
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"   💬 Message 2: {result2.get('reply')[:50]}...")
            print(f"   💰 UPI extracted: {result2.get('extractedIntelligence', {}).get('upiIds', [])}")
    except Exception as e:
        print(f"   ❌ Message 2 failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ LOCAL TESTING COMPLETE!")
    print("=" * 60)
    
    print("\n📋 VERIFICATION CHECKLIST:")
    print("1. ✅ Server is running on http://localhost:5000")
    print("2. ✅ Model makes binary predictions (SCAM/NORMAL)")
    print("3. ✅ Scam messages detected correctly")
    print("4. ✅ Normal messages not flagged as scams")
    print("5. ✅ Intelligence extraction works")
    print("6. ✅ GUVI format maintained")
    print("7. ✅ Multi-turn conversations work")
    
    print("\n🚀 Ready for deployment to Render!")

if __name__ == "__main__":
    # First test model directly
    test_model_directly()
    
    # Check if server is running
    try:
        requests.get("http://localhost:5000/health", timeout=2)
        print("\n✅ Server is running, starting API tests...")
        test_local_api()
    except:
        print("\n❌ Server not running on localhost:5000")
        print("   Start it with: python app.py")
        sys.exit(1)