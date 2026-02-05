"""
firebase_manager.py - Working Firebase integration
"""

import os
import json
import time
from typing import Dict, Optional
import requests

class FirebaseManager:
    """Manages Firebase operations"""
    
    def __init__(self):
        self.initialized = False
        self.database_url = os.getenv('FIREBASE_DATABASE_URL')
        
        if self.database_url:
            print(f"✅ Firebase configured: {self.database_url[:50]}...")
            self.initialized = True
        else:
            print("⚠️ Firebase not configured, using memory storage")
    
    def save_session(self, session_id: str, data: Dict) -> bool:
        if not self.initialized:
            return False
        
        try:
            url = f"{self.database_url}/sessions/{session_id}.json"
            data['_updated'] = time.time()
            response = requests.put(url, json=data, timeout=5)
            return response.status_code in [200, 204]
        except:
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        if not self.initialized:
            return None
        
        try:
            url = f"{self.database_url}/sessions/{session_id}.json"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

# Global instance
firebase_manager = FirebaseManager()

# Memory fallback
memory_sessions = {}

class SessionManager:
    """Session manager with Firebase + memory fallback"""
    
    @staticmethod
    def save(session_id: str, data: Dict) -> bool:
        # Try Firebase
        if firebase_manager.initialized:
            if firebase_manager.save_session(session_id, data):
                return True
        
        # Fallback to memory
        memory_sessions[session_id] = data
        return True
    
    @staticmethod
    def load(session_id: str) -> Optional[Dict]:
        # Try Firebase
        if firebase_manager.initialized:
            session = firebase_manager.load_session(session_id)
            if session:
                return session
        
        # Fallback to memory
        return memory_sessions.get(session_id)