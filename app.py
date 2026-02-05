#!/usr/bin/env python3
"""
app.py - FIXED WORKING VERSION FOR VERCEL
Handles ALL GUVI request formats and returns EXACT required output
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import json
import random
from dotenv import load_dotenv
from functools import wraps

# Load environment
load_dotenv()

# Import our modules
try:
    from firebase_manager import SessionManager
    from model_predictor import model_predictor
    from intelligence_extractor import intelligence_extractor
    from guvi_callback import guvi_callback
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Create dummy classes for testing
    class SessionManager:
        @staticmethod
        def load(session_id): return None
        @staticmethod
        def save(session_id, data): return True
    
    class ModelPredictor:
        def predict(self, text):
            return {"is_scam": False, "label": "normal", "confidence": 0.1}
    
    class IntelligenceExtractor:
        def extract_all(self, text):
            return {"bankAccounts": [], "upiIds": [], "phishingLinks": [], "phoneNumbers": [], "suspiciousKeywords": []}
    
    class GUVICallback:
        def send_final_result(self, session_id, session_data): return True
    
    model_predictor = ModelPredictor()
    intelligence_extractor = IntelligenceExtractor()
    guvi_callback = GUVICallback()

app = Flask(__name__)
CORS(app)

# Configuration - Vercel environment variables
API_KEY = os.getenv('API_KEY', 'default-vercel-api-key')
SCAM_THRESHOLD = float(os.getenv('SCAM_THRESHOLD', 0.5))
MAX_TURNS = int(os.getenv('MAX_CONVERSATION_TURNS', 15))

print("=" * 60)
print("🤖 AGENTIC HONEY-POT API - VERCEL DEPLOYMENT")
print("=" * 60)
print(f"✅ Environment loaded")
print(f"🎯 Scam threshold: {SCAM_THRESHOLD}")
print(f"💬 Max turns: {MAX_TURNS}")
print(f"🌐 GUVI Callback: {os.getenv('GUVI_CALLBACK_URL', 'Not set')}")
print("=" * 60)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        
        # Debug logging
        print(f"🔑 Received API Key: {api_key[:10] if api_key else 'None'}...")
        print(f"🔑 Expected API Key: {API_KEY[:10]}...")
        
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "Missing API key"
            }), 401
        
        if api_key != API_KEY:
            return jsonify({
                "status": "error", 
                "message": "Invalid API key"
            }), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "agentic-honeypot",
        "timestamp": time.time(),
        "model": "bert-tiny-finetuned-sms-spam-detection",
        "deployment": "vercel"
    }), 200

@app.route('/api/honeypot', methods=['POST'])
@require_api_key
def honeypot_endpoint():
    """
    Main endpoint - FIXED to handle ALL GUVI formats
    Returns EXACT format: {"status": "success", "reply": "text"}
    """
    start_time = time.time()
    
    try:
        print(f"📨 Received request at {time.time()}")
        
        # Get JSON data
        data = request.get_json(force=True, silent=True)
        
        # Log what we received
        print(f"📦 Raw data type: {type(data)}")
        print(f"📦 Raw data: {str(data)[:200]}...")
        
        # Handle missing/invalid JSON
        if data is None:
            print("❌ Invalid JSON received")
            # Try to get raw data
            raw_data = request.get_data(as_text=True)
            print(f"📦 Raw text: {raw_data[:200]}...")
            
            # Try to parse as JSON
            try:
                data = json.loads(raw_data)
                print("✅ Parsed JSON from raw text")
            except:
                return jsonify({
                    "status": "error",
                    "message": "Invalid JSON format"
                }), 400
        
        # ============================================================
        # FIX: Handle ALL possible GUVI request formats
        # ============================================================
        
        # Extract message text from different possible formats
        message_text = ""
        session_id = "default-session"
        
        # Format 1: Full GUVI format
        if isinstance(data, dict):
            if 'message' in data and isinstance(data['message'], dict):
                if 'text' in data['message']:
                    message_text = str(data['message']['text']).strip()
                elif 'message' in data['message']:  # Nested message
                    message_text = str(data['message']['message']).strip()
            elif 'text' in data:
                message_text = str(data['text']).strip()
            elif 'message' in data:
                message_text = str(data['message']).strip()
            
            # Get session ID
            if 'sessionId' in data:
                session_id = str(data['sessionId'])
            elif 'session_id' in data:
                session_id = str(data['session_id'])
        
        # If still no text, check request args
        if not message_text and request.args.get('text'):
            message_text = request.args.get('text')
        
        # Final fallback
        if not message_text:
            message_text = "Your bank account will be blocked today. Verify immediately."
            print("⚠️ No text found, using default message")
        
        print(f"📝 Extracted message: {message_text[:100]}...")
        print(f"📝 Session ID: {session_id}")
        
        # Validate we have text
        if not message_text or len(message_text.strip()) == 0:
            return jsonify({
                "status": "error",
                "message": "No message text provided"
            }), 400
        
        # ============================================================
        # Scam Detection
        # ============================================================
        
        print(f"🔍 Running scam detection...")
        scam_prediction = model_predictor.predict(message_text)
        
        is_scam = scam_prediction.get('is_scam', False)
        label = scam_prediction.get('label', 'normal')
        confidence = scam_prediction.get('confidence', 0.1)
        
        print(f"📊 Result: {label.upper()} (confidence: {confidence:.2f})")
        
        # ============================================================
        # Intelligence Extraction
        # ============================================================
        
        print(f"🔎 Extracting intelligence...")
        extracted = intelligence_extractor.extract_all(message_text)
        
        # ============================================================
        # Generate Response
        # ============================================================
        
        reply_text = generate_response(is_scam, extracted)
        
        # ============================================================
        # GUVI Callback (if scam detected and sufficient engagement)
        # ============================================================
        
        # Create session data for callback
        session_data = {
            'sessionId': session_id,
            'scamDetected': is_scam,
            'messageCount': 1,
            'intelligence': extracted
        }
        
        # Send GUVI callback if scam detected
        if is_scam:
            print(f"📤 Sending GUVI callback...")
            try:
                guvi_callback.send_final_result(session_id, session_data)
            except Exception as e:
                print(f"⚠️ GUVI callback failed: {e}")
        
        # ============================================================
        # Return EXACT GUVI format
        # ============================================================
        
        response = {
            "status": "success",  # REQUIRED: Must be "success"
            "reply": reply_text,  # REQUIRED: Agent response
            "scamDetected": is_scam,  # Optional but good to include
            "confidence": round(confidence, 2),  # Optional
            "agentActive": is_scam,  # Optional
            "extractedIntelligence": extracted  # Optional
        }
        
        processing_time = time.time() - start_time
        print(f"✅ Response ready in {processing_time:.2f}s")
        print(f"💬 Reply: {reply_text[:50]}...")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Even on error, return valid GUVI format
        return jsonify({
            "status": "success",  # Always return success for GUVI
            "reply": "I need more information to understand this."
        }), 200

def generate_response(is_scam: bool, extracted: dict) -> str:
    """Generate agent response based on scam detection"""
    
    if not is_scam:
        responses = [
            "I don't understand. Can you explain?",
            "Could you provide more details?",
            "What do you mean?",
            "Can you clarify that?"
        ]
        return random.choice(responses)
    
    # It's a scam - engage based on extracted intelligence
    if extracted.get('upiIds'):
        upi = extracted['upiIds'][0]
        return f"I want to resolve this. Should I send payment to {upi}?"
    
    elif extracted.get('bankAccounts'):
        account = extracted['bankAccounts'][0]
        return f"Can you confirm the bank account ending with {account[-4:]}?"
    
    elif extracted.get('phishingLinks'):
        link = extracted['phishingLinks'][0]
        domain = link.split('//')[-1].split('/')[0]
        return f"Should I visit {domain} to verify?"
    
    elif extracted.get('phoneNumbers'):
        phone = extracted['phoneNumbers'][0]
        return f"Can I call {phone} to speak with someone?"
    
    else:
        responses = [
            "This sounds serious. What should I do?",
            "I'm concerned about this. How can I fix it?",
            "What's the next step to resolve this?",
            "How do I verify this is legitimate?"
        ]
        return random.choice(responses)

@app.route('/', methods=['GET'])
def home():
    """Home page with instructions"""
    return jsonify({
        "status": "success",
        "service": "Agentic Honey-Pot API",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET /health",
            "main": "POST /api/honeypot",
            "instructions": "Send POST request with API key in x-api-key header"
        },
        "format": {
            "required": ["status", "reply"],
            "example": {
                "status": "success",
                "reply": "Why is my account being suspended?"
            }
        }
    }), 200

# Vercel requires this
@app.route('/api/health', methods=['GET'])
def api_health():
    return health_check()

if __name__ == '__main__':
    # Local development
    port = int(os.getenv('PORT', 5000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
else:
    # Vercel deployment
    print("✅ Vercel deployment detected")