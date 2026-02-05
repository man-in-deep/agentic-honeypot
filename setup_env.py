#!/usr/bin/env python3
"""
setup_env.py
Auto-generates .env file with PythonAnywhere compatible keys
"""

import os
import secrets
import json
from datetime import datetime

def create_env_file():
    """Create .env file for PythonAnywhere"""
    
    print("=" * 60)
    print("⚙️  SETUP ENVIRONMENT FOR PYTHONANYWHERE")
    print("=" * 60)
    
    # Generate API Key
    api_key = secrets.token_urlsafe(32)
    
    # Check model
    model_path = "models/downloaded_model"
    if not os.path.exists(model_path):
        print("⚠️  Model not downloaded. Run: python model_downloader.py")
        return False
    
    # Check Firebase
    firebase_config = {}
    if os.path.exists('firebase-credentials.json'):
        try:
            with open('firebase-credentials.json', 'r') as f:
                firebase_config = json.load(f)
            print("✅ Firebase credentials found")
        except:
            print("⚠️  Could not read firebase-credentials.json")
    else:
        print("📝 Firebase credentials not found")
        print("   Get from: Firebase Console → Service Accounts")
        print("   Save as: firebase-credentials.json")
    
    # Create .env
    env_content = f"""# AGENTIC HONEY-POT - PythonAnywhere Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# API Configuration
API_KEY={api_key}
PORT=5000
DEBUG=false
ENVIRONMENT=production

# Model Configuration
MODEL_PATH=models/downloaded_model
SCAM_THRESHOLD=0.5

# Firebase Configuration
FIREBASE_DATABASE_URL=https://YOUR_PROJECT.firebaseio.com/
FIREBASE_CREDENTIALS_FILE=firebase-credentials.json

# GUVI Configuration
GUVI_CALLBACK_URL=https://hackathon.guvi.in/api/updateHoneyPotFinalResult
GUVI_TIMEOUT=10

# Conversation Settings
MAX_CONVERSATION_TURNS=10
MIN_ENGAGEMENT_TURNS=2

# Logging
LOG_LEVEL=INFO
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ .env file created!")
    print(f"🔑 API Key: {api_key}")
    print(f"📁 Location: {os.path.abspath('.env')}")
    
    print("\n🔧 MANUAL UPDATES:")
    print("1. Update FIREBASE_DATABASE_URL with your Firebase URL")
    print("2. Ensure firebase-credentials.json exists")
    
    print("\n🚀 LOCAL TEST:")
    print("   pip install -r requirements.txt")
    print("   python app.py")
    print("   python test_local.py")
    
    print("\n🌐 PYTHONANYWHERE DEPLOYMENT:")
    print("1. Push to GitHub: git push origin main")
    print("2. Create PythonAnywhere account")
    print("3. Clone repo, install requirements, configure .env")
    print("4. Set up Web App")
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    create_env_file()