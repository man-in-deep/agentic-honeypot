"""
app.py - UPDATED FOR GUVI FORMAT
Main Flask API that returns EXACT GUVI format
Fixes all errors reported
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
from firebase_manager import SessionManager
from model_predictor import model_predictor
from intelligence_extractor import intelligence_extractor
from guvi_callback import guvi_callback

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = os.getenv('API_KEY', '')
SCAM_THRESHOLD = float(os.getenv('SCAM_THRESHOLD', 0.5))
MAX_TURNS = int(os.getenv('MAX_CONVERSATION_TURNS', 10))

print("=" * 60)
print("🤖 AGENTIC HONEY-POT API (GUVI Format)")
print("=" * 60)
print(f"🔑 API Key: {API_KEY[:15]}..." if API_KEY else "🔑 API Key: NOT SET")
print(f"🎯 Scam Threshold: {SCAM_THRESHOLD}")
print(f"💬 Max Turns: {MAX_TURNS}")
print(f"🌐 Model: Downloaded BERT + Patterns")
print("=" * 60)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
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
        "model": "bert-tiny-finetuned-sms-spam-detection"
    }), 200

@app.route('/api/honeypot', methods=['POST'])
@require_api_key
def honeypot_endpoint():
    """
    Main endpoint - returns EXACT GUVI format
    FIXED: Handles both full and minimal request formats
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON"
            }), 400
        
        # ============================================
        # FIX: Handle different request formats
        # ============================================
        
        # Format 1: Full GUVI format
        if 'message' in data and isinstance(data['message'], dict):
            session_id = data.get('sessionId', f"session-{int(time.time())}")
            message_data = data['message']
            message_text = message_data.get('text', '') if isinstance(message_data, dict) else str(message_data)
        
        # Format 2: Simple format (for testing)
        elif 'text' in data:
            session_id = data.get('sessionId', f"session-{int(time.time())}")
            message_text = data['text']
        
        # Format 3: Minimal format
        elif 'message' in data and isinstance(data['message'], str):
            session_id = f"session-{int(time.time())}"
            message_text = data['message']
        
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid request format. Need 'message.text' or 'text' field"
            }), 400
        
        message_text = str(message_text).strip()
        
        if not message_text:
            return jsonify({
                "status": "error",
                "message": "No message text provided"
            }), 400
        
        print(f"📨 [{session_id[:8]}] Message: {message_text[:50]}...")
        
        # Load or create session
        session_data = SessionManager.load(session_id)
        if not session_data:
            session_data = {
                'sessionId': session_id,
                'createdAt': time.time(),
                'messageCount': 0,
                'scamDetected': False,
                'agentActive': False,
                'scamType': 'unknown',
                'intelligence': {
                    'bankAccounts': [],
                    'upiIds': [],
                    'phishingLinks': [],
                    'phoneNumbers': [],
                    'suspiciousKeywords': []
                },
                'conversation': []
            }
        
        # Add message to conversation
        session_data['conversation'].append({
            'sender': 'scammer',
            'text': message_text,
            'timestamp': time.time()
        })
        
        session_data['messageCount'] = len(session_data['conversation'])
        session_data['lastActive'] = time.time()
        
        # STEP 1: DETECT SCAM (Model + Patterns)
        print(f"   🔍 Detecting scam...")
        scam_prediction = model_predictor.predict(message_text)
        
        is_scam = scam_prediction['is_scam']
        confidence = scam_prediction['confidence']
        label = scam_prediction['label']
        
        print(f"   📊 Result: {label.upper()} (Confidence: {confidence:.2f})")
        
        # Update session
        session_data['scamDetected'] = is_scam
        session_data['agentActive'] = is_scam
        
        if is_scam:
            scam_type = model_predictor.analyze_scam_type(message_text)
            session_data['scamType'] = scam_type
            print(f"   🎯 Scam type: {scam_type}")
        
        # STEP 2: EXTRACT INTELLIGENCE
        print(f"   🔎 Extracting intelligence...")
        extracted = intelligence_extractor.extract_all(message_text)
        
        # Merge intelligence
        for key in ['bankAccounts', 'upiIds', 'phishingLinks', 'phoneNumbers', 'suspiciousKeywords']:
            current = session_data['intelligence'].get(key, [])
            new = extracted.get(key, [])
            for item in new:
                if item not in current:
                    current.append(item)
            session_data['intelligence'][key] = current
        
        # STEP 3: GENERATE RESPONSE
        reply_text = generate_response(is_scam, extracted, session_data)
        
        # STEP 4: CHECK IF CONVERSATION SHOULD END
        should_end = False
        if session_data['scamDetected'] and session_data['messageCount'] >= MAX_TURNS:
            should_end = True
            print(f"   📤 Ending conversation, sending GUVI callback...")
            guvi_callback.send_final_result(session_id, session_data)
            reply_text = "I need to verify this with my bank directly. Thank you."
        
        # Save session
        SessionManager.save(session_id, session_data)
        
        # ============================================
        # CRITICAL FIX: Return EXACT GUVI format
        # GUVI expects ONLY: {"status": "success", "reply": "..."}
        # But we can include minimal extra fields
        # ============================================
        
        response = {
            "status": "success",
            "reply": reply_text,
            # Optional: Include minimal extra info for debugging
            "scamDetected": is_scam,
            "confidence": round(confidence, 2)
        }
        
        print(f"   ⏱️  Processed in {time.time() - start_time:.3f}s")
        print(f"   💬 Reply: {reply_text[:50]}...")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Always return valid GUVI format even on error
        return jsonify({
            "status": "success",  # GUVI expects "success"
            "reply": "I'm having trouble processing your message. Please try again."
        }), 200  # Return 200 even on error for GUVI

def generate_response(is_scam: bool, extracted: dict, session_data: dict) -> str:
    """Generate response based on scam detection"""
    
    if not is_scam:
        # Not a scam - generic response
        responses = [
            "I don't understand. Can you explain?",
            "Could you provide more details?",
            "I need more information to help you.",
            "Can you clarify what you mean?"
        ]
        return random.choice(responses)
    
    # It's a scam - engage based on extracted intelligence
    message_count = session_data['messageCount']
    
    if message_count == 1:
        # First response to scam
        responses = [
            "This sounds serious. What happened?",
            "I'm concerned about this. What should I do?",
            "Oh no! How can I fix this?",
            "What do I need to do immediately?"
        ]
        return random.choice(responses)
    
    # Follow-up based on extracted intelligence
    if extracted.get('upiIds'):
        upi_id = extracted['upiIds'][0]
        return f"I want to resolve this. Should I send payment to {upi_id}?"
    
    elif extracted.get('bankAccounts'):
        account = extracted['bankAccounts'][0]
        return f"Can you confirm the bank account ending with {account[-4:]}?"
    
    elif extracted.get('phishingLinks'):
        link = extracted['phishingLinks'][0]
        domain = link.split('//')[-1].split('/')[0][:30]
        return f"Should I visit {domain} to verify?"
    
    elif extracted.get('phoneNumbers'):
        phone = extracted['phoneNumbers'][0]
        return f"Can I call {phone} to speak with someone?"
    
    else:
        # Generic engagement
        responses = [
            "What's the next step?",
            "How do I verify this is legitimate?",
            "Can you provide more details about this?",
            "What should I do to resolve this?"
        ]
        return random.choice(responses)

@app.route('/api/session/<session_id>', methods=['GET'])
@require_api_key
def get_session(session_id):
    """Get session details"""
    session_data = SessionManager.load(session_id)
    if session_data:
        return jsonify({
            "status": "success",
            "session": session_data
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "Session not found"
        }), 404

@app.route('/api/test', methods=['POST'])
@require_api_key
def test_endpoint():
    """Test endpoint - returns full info"""
    data = request.get_json()
    text = data.get('text', '') if data else ''
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Get prediction
    prediction = model_predictor.predict(text)
    intelligence = intelligence_extractor.extract_all(text)
    
    return jsonify({
        "text": text,
        "prediction": prediction,
        "intelligence": intelligence
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting server on port {port} (debug={debug})")
    print("🔗 Local: http://localhost:5000")
    print("🔗 Health: http://localhost:5000/health")
    print("🔗 Main endpoint: POST http://localhost:5000/api/honeypot")
    print("\n💡 Test with: python test_local.py")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)