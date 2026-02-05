"""
guvi_callback.py - Same as before
"""

import os
import requests
import json
from typing import Dict

class GUVICallback:
    """Handles GUVI callback"""
    
    def __init__(self):
        self.callback_url = os.getenv('GUVI_CALLBACK_URL', 
                                     'https://hackathon.guvi.in/api/updateHoneyPotFinalResult')
    
    def send_final_result(self, session_id: str, session_data: Dict) -> bool:
        try:
            payload = {
                "sessionId": session_id,
                "scamDetected": session_data.get('scamDetected', False),
                "totalMessagesExchanged": session_data.get('messageCount', 0),
                "extractedIntelligence": session_data.get('intelligence', {}),
                "agentNotes": "Automated scam detection completed"
            }
            
            response = requests.post(
                self.callback_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        except:
            return False

# Global instance
guvi_callback = GUVICallback()