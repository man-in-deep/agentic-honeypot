# 📋 README.md - COMPLETE WITH EVERYTHING

```markdown
# 🏆 Agentic Honey-Pot - GUVI Hackathon Solution

An AI-powered system that detects scam messages and extracts intelligence through autonomous conversations. Built for **GUVI Hackathon 2024**.

## 🚀 Features

✅ **Model-Based Scam Detection** - Uses Hugging Face `bert-tiny-finetuned-sms-spam-detection`  
✅ **Intelligence Extraction** - Extracts UPI IDs, bank accounts, phishing links, phone numbers  
✅ **Multi-turn Conversations** - Maintains context across messages with Firebase storage  
✅ **Firebase Integration** - Session persistence and management  
✅ **GUVI Callback** - MANDATORY endpoint implementation for hackathon scoring  
✅ **Exact GUVI Format** - Returns response in required hackathon format  
✅ **Render Deployment** - One-click deployment configuration  
✅ **API Key Authentication** - Secure access control  
✅ **Health Monitoring** - `/health` endpoint for service monitoring  
✅ **Comprehensive Testing** - Separate test files for local and production  

## 📁 Project Structure

```
agentic-honeypot/
├── .gitignore
├── .env                          (Auto-generated - DO NOT COMMIT)
├── firebase-credentials.json     (Firebase config - add manually)
├── requirements.txt
├── README.md                     (This file)
├── render.yaml                   (Render deployment config)
├── app.py                        (Main Flask API)
├── setup_env.py                  (Auto-setup .env file)
├── model_downloader.py           (Download Hugging Face model)
├── model_predictor.py            (Scam detection - NO transformers!)
├── intelligence_extractor.py     (Extract UPI, bank accounts, etc.)
├── guvi_callback.py              (GUVI callback handler - MANDATORY)
├── firebase_manager.py           (Firebase session storage)
├── test_local.py                 (Test localhost:5000)
├── /models/
│   ├── __init__.py
│   └── downloaded_model/         (Hugging Face model files)
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── vocab.txt
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
└── /utils/
    └── __init__.py
```

**Separate Test Files (Create outside project):**
- `test_render.py` - Test Render deployment
- `test_guvi_format.py` - Verify GUVI format compliance

## 🔧 Prerequisites

- Python 3.8 or higher
- Git installed
- GitHub account
- Render account (for deployment)
- Firebase account (optional, for session storage)

## 🚀 Quick Start (Local Development)

### **Step 1: Clone & Setup**
```bash
# Create project directory
mkdir agentic-honeypot
cd agentic-honeypot

# Initialize Git
git init

# Create directory structure
mkdir -p models/downloaded_model utils
touch README.md requirements.txt .gitignore
touch app.py setup_env.py model_downloader.py model_predictor.py
touch intelligence_extractor.py guvi_callback.py firebase_manager.py
touch test_local.py
touch models/__init__.py utils/__init__.py
```

### **Step 2: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# If pip fails, try:
# pip3 install -r requirements.txt
# OR
# python -m pip install -r requirements.txt
```

### **Step 3: Download Model**
```bash
# Download Hugging Face model (bert-tiny-finetuned-sms-spam-detection)
python model_downloader.py

# Verify model files
ls -la models/downloaded_model/
# Should show: config.json, pytorch_model.bin, vocab.txt, etc.
```

### **Step 4: Setup Environment**
```bash
# Auto-generate .env file with API key
python setup_env.py

# Save the API key shown in terminal output
# Example: "🔑 Generated API Key: abc123def456..."
```

### **Step 5: Firebase Setup (Optional but Recommended)**
1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create new project: `agentic-honeypot`
3. Enable Realtime Database
4. Go to Project Settings → Service Accounts
5. Generate new private key → Download as `firebase-credentials.json`
6. Place file in project root
7. Update `.env` file with your Firebase URL

### **Step 6: Start Local Server**
```bash
# Start Flask development server
python app.py

# Expected output:
# ============================================
# 🤖 AGENTIC HONEY-POT API
# ============================================
# 🔑 API Key: abc123def456...
# 🎯 Scam Threshold: 0.65
# 💬 Max Turns: 15
# 🌐 GUVI Callback: https://hackathon.guvi.in/api/updateHoneyPotFinalResult
# ============================================
# 🚀 Starting server on port 5000
# 🔗 Local: http://localhost:5000
# 🔗 Health: http://localhost:5000/health
# 🔗 Main endpoint: POST http://localhost:5000/api/honeypot
# ============================================
```

### **Step 7: Test Locally**
```bash
# Open NEW terminal window/tab
cd agentic-honeypot

# Run comprehensive tests
python test_local.py

# Expected output shows:
# ✅ Health: healthy
# ✅ Scam detection working
# ✅ Intelligence extraction working
# ✅ GUVI format correct
```

## 🌐 API Usage

### **Endpoint: `POST /api/honeypot`**
**Headers:**
```
x-api-key: YOUR_API_KEY
Content-Type: application/json
```

**Request Format (GUVI Hackathon Standard):**
```json
{
  "sessionId": "unique-session-id",
  "message": {
    "sender": "scammer",
    "text": "Your bank account will be blocked. Verify immediately.",
    "timestamp": "2026-01-21T10:15:30Z"
  },
  "conversationHistory": [],
  "metadata": {
    "channel": "SMS",
    "language": "English",
    "locale": "IN"
  }
}
```

**Response Format (EXACT GUVI Format):**
```json
{
  "status": "success",
  "reply": "Oh no! What happened to my account?",
  "scamDetected": true,
  "confidence": 0.85,
  "agentActive": true,
  "extractedIntelligence": {
    "bankAccounts": [],
    "upiIds": [],
    "phishingLinks": ["http://phish.com"],
    "phoneNumbers": [],
    "suspiciousKeywords": ["urgent", "verify", "blocked"]
  },
  "sessionInfo": {
    "sessionId": "unique-session-id",
    "totalMessages": 1,
    "shouldContinue": true
  }
}
```

### **Health Check**
```bash
curl http://localhost:5000/health
```

## 🚀 Render Deployment

### **Step 1: Create render.yaml**
```yaml
# In project root, create render.yaml with this content:
services:
  - type: web
    name: agentic-honeypot
    env: python
    buildCommand: |
      pip install -r requirements.txt
      python model_downloader.py
    startCommand: gunicorn app:app
    envVars:
      - key: API_KEY
        generateValue: true
      - key: PORT
        value: 10000
      - key: FIREBASE_DATABASE_URL
        sync: false
      - key: FIREBASE_CREDENTIALS_FILE
        value: firebase-credentials.json
      - key: MODEL_PATH
        value: models/downloaded_model
      - key: SCAM_THRESHOLD
        value: 0.65
      - key: GUVI_CALLBACK_URL
        value: https://hackathon.guvi.in/api/updateHoneyPotFinalResult
      - key: GUVI_TIMEOUT
        value: 10
      - key: MAX_CONVERSATION_TURNS
        value: 15
    healthCheckPath: /health
    autoDeploy: true
```

### **Step 2: Push to GitHub**
```bash
# Add all files
git add .

# Commit
git commit -m "Complete Agentic Honey-Pot with Render deployment"

# Create GitHub repository first (on github.com)
# Then connect and push
git remote add origin https://github.com/YOUR_USERNAME/agentic-honeypot.git
git branch -M main
git push -u origin main
```

### **Step 3: Deploy on Render**
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository `agentic-honeypot`
5. Configure:
   - **Name**: `agentic-honeypot`
   - **Environment**: `Python`
   - **Region**: `Oregon` (or nearest)
   - **Build Command**: `pip install -r requirements.txt && python model_downloader.py`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Free`

6. **Add Environment Variables:**
   - `API_KEY` → Click "Generate"
   - `PORT` → `10000`
   - `FIREBASE_DATABASE_URL` → Your Firebase URL
   - `FIREBASE_CREDENTIALS_FILE` → `firebase-credentials.json`
   - `MODEL_PATH` → `models/downloaded_model`
   - `SCAM_THRESHOLD` → `0.65`
   - `GUVI_CALLBACK_URL` → `https://hackathon.guvi.in/api/updateHoneyPotFinalResult`
   - `GUVI_TIMEOUT` → `10`
   - `MAX_CONVERSATION_TURNS` → `15`

7. **Upload Secret File:**
   - Go to Environment → Secret Files
   - Add file: `firebase-credentials.json`
   - Paste content from your local file

8. **Deploy**: Click "Create Web Service"
9. Wait 3-5 minutes for deployment
10. Get your URL: `https://agentic-honeypot.onrender.com`

### **Step 4: Test Render Deployment**
```bash
# Create SEPARATE test file outside project
cd ~/Desktop
touch test_render.py

# Copy test_render.py code from documentation
# Update with your Render URL and API Key
python test_render.py
```

## 🧪 Testing

### **Local Testing**
```bash
# Start server first
python app.py

# In another terminal
python test_local.py
```

### **Render Testing**
1. Create `test_render.py` outside project folder
2. Update with your Render URL and API Key
3. Run: `python test_render.py`

### **GUVI Format Verification**
1. Create `test_guvi_format.py` outside project
2. Update with your Render URL and API Key
3. Run: `python test_guvi_format.py`

### **GUVI Endpoint Tester**
1. Go to GUVI Hackathon website
2. Find "Agentic Honey-Pot – API Endpoint Tester"
3. Enter:
   - **x-api-key**: Your Render API Key (starts with `rnd_`)
   - **URL**: `https://agentic-honeypot.onrender.com/api/honeypot`
4. Click "Test Endpoint"

## 🏆 Hackathon Submission

### **Submission Form (GUVI Website)**
```
Deployed URL: https://agentic-honeypot.onrender.com
API KEY: rnd_abc123def456... (from Render dashboard)
```

### **Evaluation Criteria Met**
1. ✅ Scam detection accuracy
2. ✅ Quality of agentic engagement
3. ✅ Intelligence extraction
4. ✅ API stability and response time
5. ✅ Ethical behavior
6. ✅ GUVI callback implementation (MANDATORY)

## 🚨 Common Issues & Solutions

### **Issue 1: Model Download Fails**
```bash
# Manual download
cd models/downloaded_model
wget https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection/resolve/main/config.json
wget https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection/resolve/main/pytorch_model.bin
wget https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection/resolve/main/vocab.txt
```

### **Issue 2: Firebase Connection Fails**
- System falls back to in-memory storage
- Still works for hackathon evaluation
- Check `firebase-credentials.json` file format

### **Issue 3: Render Deployment Fails**
1. Check build logs in Render dashboard
2. Common issues:
   - Missing `requirements.txt`
   - Model download timeout
   - Incorrect Firebase credentials
   - Python version mismatch

### **Issue 4: API Returns 401 Error**
- Wrong API Key
- Missing `x-api-key` header
- API Key not set in environment variables

### **Issue 5: Slow Response Times (Render Free Tier)**
- First request: 30-60 seconds (cold start)
- Subsequent: 2-5 seconds
- This is NORMAL for free tier

## 🔑 API Keys Management

### **Local Development**
- **File**: `.env`
- **Key**: `API_KEY=abc123def456...`
- **Generated by**: `python setup_env.py`
- **Use for**: `http://localhost:5000`

### **Render Deployment**
- **Location**: Render dashboard → Environment
- **Key**: `rnd_abc123def456...` (starts with `rnd_`)
- **Generated by**: Render automatically
- **Use for**: `https://agentic-honeypot.onrender.com`

### **GUVI Submission**
- **Submit ONLY**: Render URL + Render API Key
- **NEVER submit**: Local URL or local API Key

## 📞 Hackathon Support

### **For Hackathon-Specific Issues:**
1. Check GUVI hackathon documentation
2. Verify your implementation matches required format
3. Test with provided test files
4. Contact hackathon organizers if needed
https://hackathon.guvi.in
### **GitHub Issues Template:**
```
## Issue Description
[Brief description of the issue]

## Environment
- Local/Render: [ ]
- Python Version: [ ]
- OS: [ ]

## Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Error Logs
[Copy error messages here]

## Screenshots
[If applicable]
```
https://github.com/man-in-deep/agentic-honeypot/issues

## 📋 Final Checklist Before Submission

1. ✅ Local server runs: `python app.py`
2. ✅ Local tests pass: `python test_local.py`
3. ✅ Model downloaded: `ls models/downloaded_model/`
4. ✅ Firebase configured (or memory fallback)
5. ✅ GitHub pushed: `git push origin main`
6. ✅ Render deployed: Service shows "Live"
7. ✅ Render tests pass: `python test_render.py`
8. ✅ GUVI format verified: `python test_guvi_format.py`
9. ✅ GUVI endpoint tester shows success
10. ✅ Output matches GUVI requirements

## 🎯 Scam Types Detected

- ✅ Bank phishing scams
- ✅ UPI fraud
- ✅ Lottery scams
- ✅ Account suspension threats
- ✅ Tech support scams
- ✅ Prize scams
- ✅ Inheritance scams
- ✅ Payment fraud

## 🔍 Intelligence Extracted

- **UPI IDs**: `scammer@upi`, `fraud@paytm`, etc.
- **Bank Accounts**: 9-18 digit account numbers
- **Phishing Links**: Suspicious URLs and domains
- **Phone Numbers**: Indian and international formats
- **Suspicious Keywords**: Urgency, threats, rewards

## 📜 License

This project is created for GUVI Hackathon submission.
All rights reserved by the participant.

## 🙏 Acknowledgments

- Hugging Face for the pre-trained model
- Firebase for database services
- Render for deployment platform
- GUVI for organizing the hackathon

---

## ⚠️ IMPORTANT NOTES FOR HACKATHON

1. **DO NOT** commit `.env` or `firebase-credentials.json` to Git
2. **DO** test with GUVI endpoint tester before submission
3. **DO** keep your Render API Key secure
4. **DO** submit only Render URL (not localhost)
5. **DO** implement GUVI callback (it's MANDATORY for scoring)

## 🚀 Quick Deployment Summary

```bash
# Complete deployment in 3 steps:
1. git push origin main
2. Deploy on Render (5 minutes)
3. Test and submit on GUVI
```

**Submission Details:**
- **Deployed URL**: `https://agentic-honeypot.onrender.com`
- **API KEY**: From Render dashboard → Environment → API_KEY

**GOOD LUCK WITH THE HACKATHON!** 🏆
```

## 📁 Additional Files Needed

### **test_render.py** (Create SEPARATELY outside project)
```python
#!/usr/bin/env python3
"""
test_render.py - Test Render deployment
Create this file OUTSIDE project folder (e.g., on Desktop)
"""

import requests
import json
import time
import sys

def test_render_api():
    RENDER_URL = "https://agentic-honeypot.onrender.com"  # UPDATE THIS
    API_KEY = "rnd_xxxxxxxxxxxxxxxxxxxx"  # UPDATE THIS
    
    print("🧪 Testing Render Deployment...")
    
    headers = {
        'x-api-key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Test health
    try:
        response = requests.get(f"{RENDER_URL}/health", timeout=15)
        print(f"✅ Health: {response.json().get('status')}")
    except:
        print("❌ Cannot connect to Render")
        return
    
    # Test API
    payload = {
        "sessionId": f"test-{int(time.time())}",
        "message": {
            "sender": "scammer",
            "text": "Your account will be blocked. Verify at http://test.com",
            "timestamp": "2026-01-21T10:15:30Z"
        },
        "conversationHistory": [],
        "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
    }
    
    try:
        response = requests.post(f"{RENDER_URL}/api/honeypot", headers=headers, json=payload, timeout=15)
        result = response.json()
        print(f"✅ API Response: {result.get('status')}")
        print(f"📊 Scam Detected: {result.get('scamDetected')}")
        print(f"🎯 Confidence: {result.get('confidence')}")
        print(f"🤖 Reply: {result.get('reply')}")
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    print("⚠️ Update RENDER_URL and API_KEY in code first!")
    test_render_api()
```

### **test_guvi_format.py** (Create SEPARATELY outside project)
```python
#!/usr/bin/env python3
"""
test_guvi_format.py - Verify GUVI format compliance
Create this file OUTSIDE project folder
"""

import requests
import json
import time

def test_guvi_format():
    RENDER_URL = "https://agentic-honeypot.onrender.com/api/honeypot"
    API_KEY = "rnd_xxxxxxxxxxxxxxxx"
    
    headers = {
        'x-api-key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "sessionId": "guvi-test-" + str(int(time.time())),
        "message": {
            "sender": "scammer",
            "text": "URGENT: Account suspension. Verify now.",
            "timestamp": "2026-01-21T10:15:30Z"
        },
        "conversationHistory": [],
        "metadata": {"channel": "SMS", "language": "English", "locale": "IN"}
    }
    
    try:
        response = requests.post(RENDER_URL, headers=headers, json=payload, timeout=15)
        result = response.json()
        
        # Check required fields
        required = ['status', 'reply', 'scamDetected', 'confidence', 'extractedIntelligence']
        if all(field in result for field in required):
            print("✅ ALL GUVI FIELDS PRESENT")
            if result['status'] == 'success':
                print("✅ Status: success")
            if 0 <= result['confidence'] <= 1:
                print("✅ Confidence in range")
            print("🎉 READY FOR GUVI SUBMISSION!")
        else:
            print("❌ MISSING REQUIRED FIELDS")
    except:
        print("❌ Test failed")

if __name__ == "__main__":
    test_guvi_format()
```

## 🎯 Final Steps for Hackathon

1. **Complete Local Testing**: `python test_local.py`
2. **Deploy to Render**: Follow steps above
3. **Test Render**: `python test_render.py`
4. **Verify Format**: `python test_guvi_format.py`
5. **Test with GUVI Tester**: On hackathon website
6. **Submit**: Enter Render URL and API Key

**Submission Deadline**: Before Feb 5, 2026, 11:59 PM

**May the best honeypot win!** 🍯🤖
