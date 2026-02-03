#!/usr/bin/env python3
"""
setup_env.py
Auto-generates .env file with all required keys
"""

import os
import secrets
import json
from datetime import datetime

def create_environment_file():
    """Create .env file with all configuration"""
    
    print("=" * 60)
    print("⚙️  SETTING UP ENVIRONMENT")
    print("=" * 60)
    
    # Step 1: Generate API Key
    api_key = secrets.token_urlsafe(32)
    print(f"🔑 Generated API Key: {api_key[:15]}...")
    
    # Step 2: Check model
    model_path = "models/downloaded_model"
    if not os.path.exists(model_path):
        print("⚠️  Model not downloaded. Run: python model_downloader.py")
        return False
    
    # Step 3: Firebase setup
    firebase_config = {}
    if os.path.exists('firebase-credentials.json'):
        try:
            with open('firebase-credentials.json', 'r') as f:
                firebase_config = json.load(f)
            print(f"🔥 Firebase project: {firebase_config.get('project_id', 'NOT FOUND')}")
        except:
            print("⚠️  Could not read firebase-credentials.json")
    else:
        print("📝 Firebase credentials not found")
        print("   Get from: Firebase Console → Project Settings → Service Accounts")
        print("   Save as: firebase-credentials.json in project root")
    
    # Step 4: Create .env content
    env_content = f"""# AGENTIC HONEY-POT - Environment Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# DO NOT COMMIT THIS FILE TO GIT

# API Configuration
API_KEY={api_key}
PORT=5000
DEBUG=false
ENVIRONMENT=production

# Model Configuration
MODEL_PATH=models/downloaded_model
SCAM_THRESHOLD=0.65
MAX_SEQUENCE_LENGTH=128

# Firebase Configuration
FIREBASE_DATABASE_URL=https://YOUR_PROJECT.firebaseio.com/
FIREBASE_CREDENTIALS_FILE=firebase-credentials.json

# GUVI Configuration (MANDATORY FOR HACKATHON)
GUVI_CALLBACK_URL=https://hackathon.guvi.in/api/updateHoneyPotFinalResult
GUVI_TIMEOUT=10

# Session Settings
MAX_CONVERSATION_TURNS=15
MIN_ENGAGEMENT_TURNS=3
SESSION_TIMEOUT_SECONDS=3600

# Application Settings
LOG_LEVEL=INFO
REQUEST_TIMEOUT=30
ALLOW_ORIGINS=*
"""
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ .env file created successfully!")
    print(f"📁 Location: {os.path.abspath('.env')}")
    
    # Display important info
    print("\n📋 IMPORTANT INFORMATION:")
    print(f"   API Key: {api_key}")
    print(f"   Local URL: http://localhost:5000")
    print(f"   Endpoint: /api/honeypot")
    print(f"   Health Check: /health")
    
    print("\n🔧 MANUAL UPDATES NEEDED:")
    print("1. Update FIREBASE_DATABASE_URL in .env")
    print("2. Add firebase-credentials.json file")
    
    print("\n🚀 START APPLICATION:")
    print("   pip install -r requirements.txt")
    print("   python app.py")
    
    print("\n🧪 TEST LOCAL:")
    print("   python test_local.py")
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    create_environment_file()