"""
guvi_callback.py
Sends final result to GUVI evaluation endpoint (MANDATORY)
"""

import os
import requests
import json
import time
from typing import Dict

class GUVICallback:
    """Handles GUVI callback - MANDATORY FOR SCORING"""
    
    def __init__(self):
        self.callback_url = os.getenv('GUVI_CALLBACK_URL', 
                                     'https://hackathon.guvi.in/api/updateHoneyPotFinalResult')
        self.timeout = int(os.getenv('GUVI_TIMEOUT', 10))
    
    def send_final_result(self, session_id: str, session_data: Dict) -> bool:
        """
        Send final intelligence to GUVI
        THIS IS MANDATORY FOR HACKATHON EVALUATION
        """
        print(f"📤 SENDING GUVI CALLBACK for session {session_id}")
        
        try:
            payload = {
                "sessionId": session_id,
                "scamDetected": session_data.get('scamDetected', False),
                "totalMessagesExchanged": session_data.get('messageCount', 0),
                "extractedIntelligence": session_data.get('intelligence', {
                    "bankAccounts": [],
                    "upiIds": [],
                    "phishingLinks": [],
                    "phoneNumbers": [],
                    "suspiciousKeywords": []
                }),
                "agentNotes": self._generate_agent_notes(session_data)
            }
            
            print(f"   📦 Payload prepared")
            
            response = requests.post(
                self.callback_url,
                json=payload,
                timeout=self.timeout,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 200:
                print(f"   ✅ GUVI callback successful")
                return True
            else:
                print(f"   ⚠️  GUVI callback failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ GUVI callback error: {e}")
            return False
    
    def _generate_agent_notes(self, session_data: Dict) -> str:
        """Generate agent notes for GUVI"""
        intel = session_data.get('intelligence', {})
        notes = []
        
        if intel.get('upiIds'):
            notes.append(f"Extracted {len(intel['upiIds'])} UPI ID(s)")
        if intel.get('bankAccounts'):
            notes.append(f"Extracted {len(intel['bankAccounts'])} bank account(s)")
        if intel.get('phishingLinks'):
            notes.append(f"Found {len(intel['phishingLinks'])} phishing link(s)")
        if intel.get('phoneNumbers'):
            notes.append(f"Collected {len(intel['phoneNumbers'])} phone number(s)")
        
        if session_data.get('scamDetected'):
            scam_type = session_data.get('scamType', 'unknown')
            notes.append(f"Scam type: {scam_type}")
        
        notes.append(f"Total messages: {session_data.get('messageCount', 0)}")
        
        return ". ".join(notes) if notes else "Engaged with scammer"

# Global instance
guvi_callback = GUVICallback()