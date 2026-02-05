#!/usr/bin/env python3
"""
setup_env.py - Local environment setup
"""

import os
import secrets
from datetime import datetime

def setup_environment():
    print("=" * 60)
    print("⚙️  LOCAL ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Generate API keys
    local_api_key = secrets.token_urlsafe(32)
    
    # Create .env file
    env_content = f"""# LOCAL DEVELOPMENT - DO NOT PUSH TO GIT
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# API Configuration
API_KEY={local_api_key}
PORT=5000
DEBUG=true

# Firebase Configuration (Update with your values)
FIREBASE_DATABASE_URL=https://YOUR_PROJECT.firebaseio.com/
FIREBASE_CREDENTIALS_FILE=firebase-credentials.json

# GUVI Configuration
GUVI_CALLBACK_URL=https://hackathon.guvi.in/api/updateHoneyPotFinalResult
GUVI_TIMEOUT=10

# Model Configuration
MODEL_PATH=models/downloaded_model
SCAM_THRESHOLD=0.5

# Application Settings
MAX_CONVERSATION_TURNS=15
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"✅ .env file created")
    print(f"🔑 Local API Key: {local_api_key}")
    print(f"📁 Firebase: Update FIREBASE_DATABASE_URL in .env")
    print(f"📁 Add firebase-credentials.json to project root")
    print("\n🚀 Next: Run 'python model_downloader.py' then 'python app.py'")

if __name__ == "__main__":
    setup_environment()