#!/usr/bin/env python3
"""
test_endpoint.py - Tests YOUR specific Render deployment
Tests: https://agentic-honeypot-0p4k.onrender.com/api/honeypot
"""

import requests
import json
import time

def test_your_endpoint():
    """Test YOUR specific Render API endpoint"""
    
    # ⚠️ YOUR SPECIFIC VALUES ⚠️
    YOUR_ENDPOINT = "https://agentic-honeypot-0p4k.onrender.com/api/honeypot"
    YOUR_API_KEY = "a3da6ac1825ebe8a1dce3520e36657d2"
    
    print("=" * 70)
    print("🧪 TESTING YOUR RENDER ENDPOINT")
    print("=" * 70)
    print(f"🌐 Endpoint: {YOUR_ENDPOINT}")
    print(f"🔑 API Key: {YOUR_API_KEY}")
    print()
    
    headers = {
        'x-api-key': YOUR_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Test 1: Simple scam message
    print("1️⃣  Testing basic scam detection...")
    
    test_payload = {
        "sessionId": f"test-{int(time.time())}",
        "message": {
            "sender": "scammer",
            "text": "URGENT: Your bank account will be blocked. Verify at http://fake-bank-verify.com. Call +91-9876543210",
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
        response = requests.post(YOUR_ENDPOINT, headers=headers, json=test_payload, timeout=25)
        
        print(f"   ⏱️  Response time: {response.elapsed.total_seconds():.2f}s")
        print(f"   📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📊 RESPONSE ANALYSIS:")
            print(f"   Status: {result.get('status')}")
            print(f"   Scam Detected: {result.get('scamDetected')}")
            print(f"   Confidence: {result.get('confidence')}")
            print(f"   Agent Active: {result.get('agentActive', 'Not specified')}")
            print(f"   Reply: {result.get('reply', '')[:80]}...")
            
            # Check extracted intelligence
            intel = result.get('extractedIntelligence', {})
            if intel:
                print(f"\n🎯 INTELLIGENCE EXTRACTED:")
                for key, value in intel.items():
                    if value:
                        print(f"   • {key}: {value}")
            
            # Check GUVI required format
            print(f"\n✅ GUVI FORMAT CHECK:")
            required_fields = ['status', 'reply', 'scamDetected', 'confidence', 'extractedIntelligence']
            missing = [f for f in required_fields if f not in result]
            
            if not missing:
                print(f"   ✅ All required fields present")
                
                # Check extractedIntelligence structure
                intel_fields = ['bankAccounts', 'upiIds', 'phishingLinks', 'phoneNumbers', 'suspiciousKeywords']
                intel_missing = [f for f in intel_fields if f not in intel]
                
                if not intel_missing:
                    print(f"   ✅ Intelligence structure correct")
                else:
                    print(f"   ❌ Missing intelligence fields: {intel_missing}")
            else:
                print(f"   ❌ Missing fields: {missing}")
            
            return True
            
        else:
            print(f"\n❌ REQUEST FAILED:")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n⏱️  TIMEOUT: Render free tier is slow")
        print(f"   Try again - first request after inactivity takes 30-60 seconds")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        return False
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print("Testing YOUR Render endpoint...")
    print()
    
    if test_your_endpoint():
        print("\n🎉 YOUR ENDPOINT IS WORKING CORRECTLY!")
        print("   Ready for GUVI endpoint tester and submission.")
    else:
        print("\n⚠️  ISSUES DETECTED")
        print("   Fix before submitting to GUVI.")