# agentic-honeypot
# 🏆 Agentic Honey-Pot - GUVI Hackathon Solution

An AI-powered system that detects scam messages and extracts intelligence through autonomous conversations.

## 🚀 Features
- Scam detection using Hugging Face model
- Intelligence extraction (UPI IDs, bank accounts, phishing links)
- Multi-turn conversation handling
- Firebase session storage
- GUVI callback implementation
- Render deployment ready

## 📁 Project Structure
agentic-honeypot/
├── app.py # Main Flask API
├── model_downloader.py # Downloads Hugging Face model
├── model_predictor.py # Scam detection
├── intelligence_extractor.py # Extracts intelligence
├── firebase_manager.py # Firebase session storage
├── guvi_callback.py # GUVI callback (MANDATORY)
├── setup_env.py # Auto-generates .env
├── test_local.py # Test localhost
├── requirements.txt # Dependencies
└── render.yaml # Render deployment

## 🔧 Quick Start
1. `pip install -r requirements.txt`
2. `python model_downloader.py`
3. `python setup_env.py`
4. `python app.py`
5. `python test_local.py`

## 📞 Support
[For hackathon submission issues, contact the hackathon organizers.](https://hackathon.guvi.in/)
for code related issues - (it has issues, as i did it in one day haven't perfectly estblished the model yet but submitted.-if any queries or issues- https://github.com/man-in-deep/agentic-honeypot/issues
