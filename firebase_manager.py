"""
firebase_manager.py - SAME AS BEFORE
Manages session storage using Firebase Realtime Database
"""

import os
import json
import time
from typing import Dict, Optional
import requests

class FirebaseManager:
    """Manages Firebase operations for session storage"""
    
    def __init__(self):
        self.project_id = None
        self.database_url = None
        self.initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            creds_file = os.getenv('FIREBASE_CREDENTIALS_FILE', 'firebase-credentials.json')
            
            if os.path.exists(creds_file):
                with open(creds_file, 'r') as f:
                    credentials = json.load(f)
                
                self.project_id = credentials.get('project_id')
                self.database_url = os.getenv('FIREBASE_DATABASE_URL', 
                                            f'https://{self.project_id}.firebaseio.com')
                self.initialized = True
                print(f"✅ Firebase initialized: {self.project_id}")
            else:
                print("⚠️  Firebase credentials not found. Using memory storage.")
                self.initialized = False
                
        except Exception as e:
            print(f"❌ Firebase initialization failed: {e}")
            self.initialized = False
    
    def save_session(self, session_id: str, data: Dict) -> bool:
        """Save session data to Firebase"""
        if not self.initialized:
            return False
        
        try:
            url = f"{self.database_url}/sessions/{session_id}.json"
            data['_updated'] = time.time()
            
            response = requests.put(url, json=data, timeout=5)
            return response.status_code in [200, 204]
                
        except Exception as e:
            print(f"❌ Firebase save error: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """Load session data from Firebase"""
        if not self.initialized:
            return None
        
        try:
            url = f"{self.database_url}/sessions/{session_id}.json"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data
            return None
            
        except Exception as e:
            print(f"❌ Firebase load error: {e}")
            return None

# Global instance
firebase_manager = FirebaseManager()

# Fallback in-memory storage
memory_sessions = {}

class SessionManager:
    """Unified session manager (Firebase + Memory fallback)"""
    
    @staticmethod
    def save(session_id: str, data: Dict) -> bool:
        """Save session with Firebase fallback to memory"""
        if firebase_manager.initialized:
            success = firebase_manager.save_session(session_id, data)
            if success:
                return True
        
        memory_sessions[session_id] = data
        return True
    
    @staticmethod
    def load(session_id: str) -> Optional[Dict]:
        """Load session from Firebase or memory"""
        if firebase_manager.initialized:
            session = firebase_manager.load_session(session_id)
            if session:
                return session
        
        return memory_sessions.get(session_id)