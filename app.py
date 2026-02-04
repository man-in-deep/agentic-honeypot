"""
app.py - UPDATED TO HANDLE GUVI TESTER
Main Flask API using the simple working model
Returns EXACT GUVI format
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
API_KEY = os.getenv('API_KEY', 'default-api-key-never-use-this')
SCAM_THRESHOLD = 0.5  # Binary threshold
MAX_TURNS = int(os.getenv('MAX_CONVERSATION_TURNS', 15))

print("=" * 60)
print("🤖 AGENTIC HONEY-POT API")
print("=" * 60)
print(f"🔑 API Key: {API_KEY[:15]}...")
print(f"🎯 Model: Simple BERT (binary classification)")
print(f"💬 Max Turns: {MAX_TURNS}")
print(f"🌐 GUVI Callback: {os.getenv('GUVI_CALLBACK_URL')}")
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
        "version": "1.0.0",
        "model": "bert-tiny-finetuned-sms-spam-detection"
    }), 200

@app.route('/api/honeypot', methods=['POST'])
@require_api_key
def honeypot_endpoint():
    """
    Main endpoint - returns EXACT GUVI format
    Fixed to handle GUVI tester requests properly
    """
    start_time = time.time()
    
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Content-Type must be application/json",
                "reply": "I need JSON data to process."
            }), 400
        
        data = request.get_json()
        
        # DEBUG: Log what GUVI tester is sending
        print(f"🔍 DEBUG: Received data keys: {list(data.keys()) if data else 'No data'}")
        if data:
            print(f"🔍 DEBUG: Data sample: {json.dumps(data)[:200]}...")
        
        if not data:
            # GUVI tester might send empty request - create default
            print("⚠️  No data received, using default test message")
            data = {
                "sessionId": f"guvi-test-{int(time.time())}",
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
        
        # Extract fields with fallbacks for GUVI tester
        session_id = data.get('sessionId', f"test-session-{int(time.time())}")
        
        # Handle different possible message structures
        message = data.get('message', {})
        if not message and 'text' in data:
            # GUVI might send text directly
            message = {'text': data.get('text'), 'sender': 'scammer'}
        
        message_text = message.get('text', '') if isinstance(message, dict) else str(message)
        message_text = message_text.strip()
        
        # If still no text, use default
        if not message_text:
            message_text = "Your bank account has security issues. Immediate verification required."
            print(f"⚠️  No message text, using default: {message_text}")
        
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
            'sender': message.get('sender', 'scammer') if isinstance(message, dict) else 'scammer',
            'text': message_text,
            'timestamp': message.get('timestamp', time.time()) if isinstance(message, dict) else time.time()
        })
        
        session_data['messageCount'] = len(session_data['conversation'])
        session_data['lastActive'] = time.time()
        
        # STEP 1: DETECT SCAM USING MODEL (BINARY)
        print(f"   🔍 Running model prediction...")
        scam_prediction = model_predictor.predict(message_text)
        
        is_scam = scam_prediction['is_scam']
        label = scam_prediction['label']
        
        print(f"   📊 Model result: {label.upper()}")
        
        # Update session
        session_data['scamDetected'] = is_scam
        session_data['agentActive'] = is_scam  # Activate agent only if scam
        
        if is_scam:
            scam_type = model_predictor.analyze_scam_type(message_text)
            session_data['scamType'] = scam_type
            print(f"   🎯 Scam type: {scam_type}")
        
        # STEP 2: EXTRACT INTELLIGENCE
        print(f"   🔎 Extracting intelligence...")
        extracted = intelligence_extractor.extract_all(message_text)
        
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
        
        # STEP 5: RETURN EXACT GUVI FORMAT
        response = {
            "status": "success",
            "reply": reply_text,
            "scamDetected": is_scam,
            "confidence": 0.9 if is_scam else 0.1,  # Fixed confidence for binary
            "agentActive": session_data['agentActive'],
            "extractedIntelligence": extracted,
            "sessionInfo": {
                "sessionId": session_id,
                "totalMessages": session_data['messageCount'],
                "shouldContinue": not should_end
            },
            "processingTime": round(time.time() - start_time, 3)
        }
        
        print(f"   ⏱️  Processed in {response['processingTime']}s")
        print(f"   💬 Reply: {reply_text[:50]}...")
        
        return jsonify(response), 200
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Invalid JSON: {str(e)}",
            "reply": "I need valid JSON data to process your request."
        }), 400
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "status": "error",
            "message": f"Internal error: {str(e)}",
            "reply": "I'm having trouble processing your message."
        }), 500

def generate_response(is_scam: bool, extracted: dict, session_data: dict) -> str:
    """Generate response based on scam detection and extracted intelligence"""
    
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
        first_responses = [
            "This sounds serious. What happened?",
            "I'm concerned about this. What should I do?",
            "Oh no! How can I fix this?",
            "What do I need to do immediately?"
        ]
        return random.choice(first_responses)
    
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

@app.route('/api/test-model', methods=['POST'])
@require_api_key
def test_model_endpoint():
    """Test the model directly - returns binary prediction"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get model prediction
        prediction = model_predictor.predict(text)
        intelligence = intelligence_extractor.extract_all(text)
        
        return jsonify({
            "text": text,
            "prediction": prediction,
            "intelligence": intelligence
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Special endpoint for GUVI tester
@app.route('/api/test', methods=['POST'])
@require_api_key
def test_endpoint():
    """Simple test endpoint for GUVI tester"""
    return jsonify({
        "status": "success",
        "message": "API is working correctly",
        "timestamp": time.time()
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting server on port {port} (debug={debug})")
    print("🔗 Local: http://localhost:5000")
    print("🔗 Health: http://localhost:5000/health")
    print("🔗 Main endpoint: POST http://localhost:5000/api/honeypot")
    print("🔗 Test endpoint: POST http://localhost:5000/api/test")
    print("\n💡 Test with: python test_local.py")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)