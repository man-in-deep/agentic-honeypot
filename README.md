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
✅ **Python Anywhere Deployment** - One-click deployment configuration  
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
├── Python Anywhere.                 (Python Anywhere deployment config)
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
- `test_Python Anywhere.py` - Test Python Anywhere deployment
- `test_guvi_format.py` - Verify GUVI format compliance

## 🔧 Prerequisites

- Python 3.8 or higher
- Git installed
- GitHub account
- Python Anywhere account (for deployment)
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

### **Step 1: Python anywhere
🌐 Step 1: Create PythonAnywhere Account
Go to pythonanywhere.com

Click "Pricing & signup" → "Create a Beginner account" (Free)

Sign up with email/password

Verify email

💻 Step 2: Setup PythonAnywhere Environment
In PythonAnywhere Dashboard:
Go to Consoles → Bash

Clone your GitHub repo:

bash
git clone https://github.com/man-in-deep/agentic-honeypot.git
cd agentic-honeypot
Install dependencies:

bash
pip3.10 install -r requirements.txt
# OR
python3.10 -m pip install -r requirements.txt
Create .env file:

bash
nano .env
Copy your local .env content and update:

API_KEY: Generate new one: openssl rand -hex 16

FIREBASE_DATABASE_URL: Your Firebase URL

DEBUG: false

PORT: Remove this line (not needed for PythonAnywhere)

Upload firebase-credentials.json:

bash
# Go to Files tab
# Upload firebase-credentials.json to /home/YOUR_USERNAME/agentic-honeypot/
🌐 Step 3: Create Web App
In PythonAnywhere Dashboard:
Go to Web tab

Click Add a new web app

Choose Manual configuration

Python version: Python 3.10

Click Next

Configure WSGI File:
Click on WSGI configuration file link

Delete everything and paste:

python
import sys
import os

# Add your project directory
path = '/home/YOUR_USERNAME/agentic-honeypot'
if path not in sys.path:
    sys.path.append(path)

# Import Flask app
from app import app as application

# Set environment variables
os.environ['FIREBASE_CREDENTIALS_FILE'] = '/home/YOUR_USERNAME/agentic-honeypot/firebase-credentials.json'
Save

Configure Web App:
Go back to Web tab

Source code: /home/YOUR_USERNAME/agentic-honeypot

Working directory: /home/YOUR_USERNAME/agentic-honeypot

Add Static Files (Optional):
URL: /static/

Directory: /home/YOUR_USERNAME/agentic-honeypot/static

🚀 Step 4: Deploy
Go to Web tab

Click Reload button

Wait for green banner: "Reloading..."

When done: "Your web app is now live at https://YOUR_USERNAME.pythonanywhere.com"**

### **Step 2: Push to GitHub**
```bash
# Add all files
git add .

# Commit
git commit -m "Complete Agentic Honey-Pot with Python Anywhere deployment"

# Create GitHub repository first (on github.com)
# Then connect and push
git remote add origin https://github.com/YOUR_USERNAME/agentic-honeypot.git
git branch -M main
git push -u origin main
```

### **Step 3: Deploy on Python Anywhere**
1. Go to [Python Anywhere.com](https://Python Anywhere.com)
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
10. Get your URL: `https://agentic-honeypot.onPython Anywhere.com`

### **Step 4: Test Python Anywhere Deployment**
```bash
# Create SEPARATE test file outside project
cd ~/Desktop
touch test_Python Anywhere.py

# Copy test_Python Anywhere.py code from documentation
# Update with your Python Anywhere URL and API Key
python test_Python Anywhere.py
```

## 🧪 Testing

### **Local Testing**
```bash
# Start server first
python app.py

# In another terminal
python test_local.py
```

### **Python Anywhere Testing**
1. Create `test_Python Anywhere.py` outside project folder
2. Update with your Python Anywhere URL and API Key
3. Run: `python test_Python Anywhere.py`

### **GUVI Format Verification**
1. Create `test_guvi_format.py` outside project
2. Update with your Python Anywhere URL and API Key
3. Run: `python test_guvi_format.py`

### **GUVI Endpoint Tester**
1. Go to GUVI Hackathon website
2. Find "Agentic Honey-Pot – API Endpoint Tester"
3. Enter:
   - **x-api-key**: Your Python Anywhere API Key (starts with `rnd_`)
   - **URL**: `https://agentic-honeypot.onPython Anywhere.com/api/honeypot`
4. Click "Test Endpoint"

## 🏆 Hackathon Submission

### **Submission Form (GUVI Website)**
```
Deployed URL: https://agentic-honeypot.onPython Anywhere.com
API KEY: rnd_abc123def456... (from Python Anywhere dashboard)
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

### **Issue 3: Python Anywhere Deployment Fails**
1. Check build logs in Python Anywhere dashboard
2. Common issues:
   - Missing `requirements.txt`
   - Model download timeout
   - Incorrect Firebase credentials
   - Python version mismatch

### **Issue 4: API Returns 401 Error**
- Wrong API Key
- Missing `x-api-key` header
- API Key not set in environment variables

### **Issue 5: Slow Response Times (Python Anywhere Free Tier)**
- First request: 30-60 seconds (cold start)
- Subsequent: 2-5 seconds
- This is NORMAL for free tier

## 🔑 API Keys Management

### **Local Development**
- **File**: `.env`
- **Key**: `API_KEY=abc123def456...`
- **Generated by**: `python setup_env.py`
- **Use for**: `http://localhost:5000`

### **Python Anywhere Deployment**
- **Location**: Python Anywhere dashboard → Environment
- **Key**: `rnd_abc123def456...` (starts with `rnd_`)
- **Generated by**: Python Anywhere automatically
- **Use for**: `https://agentic-honeypot.onPython Anywhere.com`

### **GUVI Submission**
- **Submit ONLY**: Python Anywhere URL + Python Anywhere API Key
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
- Local/Python Anywhere: [ ]
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
6. ✅ Python Anywhere deployed: Service shows "Live"
7. ✅ Python Anywhere tests pass: `python test_Python Anywhere.py`
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
- Python Anywhere for deployment platform
- GUVI for organizing the hackathon

---

## ⚠️ IMPORTANT NOTES FOR HACKATHON

1. **DO NOT** commit `.env` or `firebase-credentials.json` to Git
2. **DO** test with GUVI endpoint tester before submission
3. **DO** keep your Python Anywhere API Key secure
4. **DO** submit only Python Anywhere URL (not localhost)
5. **DO** implement GUVI callback (it's MANDATORY for scoring)

## 🚀 Quick Deployment Summary

```bash
# Complete deployment in 3 steps:
1. git push origin main
2. Deploy on Python Anywhere (5 minutes)
3. Test and submit on GUVI
```

**Submission Details:**
- **Deployed URL**: `https://agentic-honeypot.onPython Anywhere.com`
- **API KEY**: From Python Anywhere dashboard → Environment → API_KEY

**GOOD LUCK WITH THE HACKATHON!** 🏆
