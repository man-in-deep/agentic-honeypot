#!/usr/bin/env python3
"""
test_guvi_format.py - Verifies EXACT GUVI format compliance
"""

import requests
import json

def verify_guvi_format():
    """Verify response matches EXACT GUVI format"""
    
    YOUR_ENDPOINT = "https://agentic-honeypot-0p4k.onrender.com/api/honeypot"
    YOUR_API_KEY = "a3da6ac1825ebe8a1dce3520e36657d2"
    
    print("=" * 70)
    print("✅ GUVI FORMAT VERIFICATION")
    print("=" * 70)
    
    headers = {
        'x-api-key': YOUR_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Test with a message that should extract multiple intelligence types
    payload = {
        "sessionId": "guvi-format-test",
        "message": {
            "sender": "scammer",
            "text": "URGENT: Your account will be suspended. To verify, send ₹99 to scam@upi or call +91-9876543210. Click: http://verify-account-now.com",
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
        response = requests.post(YOUR_ENDPOINT, headers=headers, json=payload, timeout=20)
        
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {response.text[:100]}")
            return False
        
        result = response.json()
        
        print("📊 VERIFYING EACH FIELD:")
        print("-" * 50)
        
        # 1. status must be exactly "success"
        status_ok = result.get('status') == 'success'
        print(f"   {'✅' if status_ok else '❌'} status: {result.get('status')} {'(must be "success")' if not status_ok else ''}")
        
        # 2. reply must be non-empty string
        reply = result.get('reply', '')
        reply_ok = isinstance(reply, str) and len(reply) > 0
        print(f"   {'✅' if reply_ok else '❌'} reply: {len(reply)} chars")
        
        # 3. scamDetected must be boolean
        scam = result.get('scamDetected')
        scam_ok = isinstance(scam, bool)
        print(f"   {'✅' if scam_ok else '❌'} scamDetected: {scam} {'(must be boolean)' if not scam_ok else ''}")
        
        # 4. confidence must be number between 0-1
        confidence = result.get('confidence')
        confidence_ok = isinstance(confidence, (int, float)) and 0 <= confidence <= 1
        print(f"   {'✅' if confidence_ok else '❌'} confidence: {confidence} {'(must be 0-1)' if not confidence_ok else ''}")
        
        # 5. extractedIntelligence must exist with all subfields
        intel = result.get('extractedIntelligence', {})
        intel_exists = 'extractedIntelligence' in result
        print(f"   {'✅' if intel_exists else '❌'} extractedIntelligence: {'Present' if intel_exists else 'Missing'}")
        
        if intel_exists:
            required_intel_fields = ['bankAccounts', 'upiIds', 'phishingLinks', 'phoneNumbers', 'suspiciousKeywords']
            intel_fields_ok = all(field in intel for field in required_intel_fields)
            print(f"   {'✅' if intel_fields_ok else '❌'} All intelligence fields present")
            
            # Check each is a list
            lists_ok = all(isinstance(intel.get(field, None), list) for field in required_intel_fields)
            print(f"   {'✅' if lists_ok else '❌'} All fields are lists")
        
        # 6. agentActive (optional but good to have)
        agent = result.get('agentActive')
        agent_ok = agent is None or isinstance(agent, bool)
        print(f"   {'✅' if agent_ok else '⚠️'} agentActive: {agent} {'(should be boolean or missing)' if not agent_ok else ''}")
        
        # Final assessment
        all_required_ok = status_ok and reply_ok and scam_ok and confidence_ok and intel_exists
        
        print("\n" + "=" * 70)
        print("🎯 FINAL ASSESSMENT")
        print("=" * 70)
        
        if all_required_ok:
            print("✅ PERFECT! Your API returns EXACT GUVI format!")
            print("   Ready for submission.")
            
            # Show sample of what GUVI will see
            print("\n📄 SAMPLE OUTPUT (GUVI will see this):")
            sample = {
                "status": result.get('status'),
                "scamDetected": result.get('scamDetected'),
                "confidence": round(result.get('confidence'), 2),
                "reply": result.get('reply', '')[:60] + "..." if len(result.get('reply', '')) > 60 else result.get('reply', ''),
                "extractedIntelligence_summary": {
                    k: f"{len(v)} items" if v else "empty"
                    for k, v in intel.items()
                }
            }
            print(json.dumps(sample, indent=2))
            
        else:
            print("⚠️  FORMAT ISSUES DETECTED")
            print("   Fix before submitting to GUVI.")
        
        return all_required_ok
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Verifying GUVI format compliance...")
    print()
    
    verify_guvi_format()