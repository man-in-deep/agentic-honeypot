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
    print(f"🔑 Generated API Key: {api_key}")
    print(f"   (First 15 chars): {api_key[:15]}...")
    
    # Step 2: Check model directory exists
    model_path = "models/downloaded_model"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        print(f"📁 Created model directory: {model_path}")
    
    # Step 3: Firebase setup
    firebase_config = {}
    if os.path.exists('firebase-credentials.json'):
        try:
            with open('firebase-credentials.json', 'r') as f:
                firebase_config = json.load(f)
            project_id = firebase_config.get('project_id', 'NOT_FOUND')
            print(f"🔥 Firebase project found: {project_id}")
        except:
            print("⚠️  Could not read firebase-credentials.json")
    else:
        print("📝 Firebase credentials not found")
        print("   Get from: Firebase Console → Project Settings → Service Accounts")
        print("   Save as: firebase-credentials.json in project root")
    
    # Get Firebase URL if available
    firebase_url = ""
    if firebase_config.get('project_id'):
        project_id = firebase_config['project_id']
        firebase_url = f"https://{project_id}.firebaseio.com/"
    
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
MODEL_PATH={model_path}
SCAM_THRESHOLD=0.5  # Binary threshold
MAX_SEQUENCE_LENGTH=128

# Firebase Configuration
FIREBASE_DATABASE_URL={firebase_url if firebase_url else 'https://YOUR_PROJECT.firebaseio.com/'}
FIREBASE_CREDENTIALS_FILE=firebase-credentials.json

# GUVI Configuration (MANDATORY FOR HACKATHON)
GUVI_CALLBACK_URL=https://hackathon.guvi.in/api/updateHoneyPotFinalResult
GUVI_TIMEOUT=10

# Session Settings
MAX_CONVERSATION_TURNS=20
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
    print(f"   Health Check: http://localhost:5000/health")
    print(f"   Main Endpoint: POST http://localhost:5000/api/honeypot")
    
    print("\n🔧 MANUAL UPDATES NEEDED:")
    print("1. Update FIREBASE_DATABASE_URL in .env if not auto-filled")
    print("2. Add firebase-credentials.json file (from Firebase Console)")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Run: python model_downloader.py")
    print("3. Run: python app.py")
    print("4. Run: python test_local.py")
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    create_environment_file()