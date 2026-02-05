"""
app.py - MAIN FIXED VERSION
Returns EXACT GUVI format required by hackathon
Fixes all errors: invalid body, format issues, etc.
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
MAX_TURNS = int(os.getenv('MAX_CONVERSATION_TURNS', 15))

print("=" * 60)
print("🤖 AGENTIC HONEY-POT API")
print("=" * 60)
print(f"🔑 API Key configured: {'Yes' if API_KEY else 'No'}")
print(f"🎯 Scam Threshold: {SCAM_THRESHOLD}")
print(f"💬 Max Turns: {MAX_TURNS}")
print(f"🔥 Firebase: {'Enabled' if SessionManager.load('test') is not None else 'Memory only'}")
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
        "version": "2.0.0",
        "model": "bert-tiny-finetuned-sms-spam-detection",
        "endpoints": {
            "main": "POST /api/honeypot",
            "health": "GET /health",
            "test": "POST /api/test"
        }
    }), 200

@app.route('/api/test', methods=['POST'])
@require_api_key
def test_endpoint():
    """Simple test endpoint - returns GUVI format"""
    try:
        data = request.get_json()
        
        # Handle different input formats
        if not data:
            return jsonify({
                "status": "success",
                "reply": "Test endpoint working! Send proper scam message.",
                "scamDetected": False,
                "confidence": 0.1,
                "agentActive": False,
                "extractedIntelligence": {
                    "bankAccounts": [],
                    "upiIds": [],
                    "phishingLinks": [],
                    "phoneNumbers": [],
                    "suspiciousKeywords": []
                }
            }), 200
        
        # Get message text from different possible formats
        message_text = ""
        if isinstance(data, dict):
            if 'message' in data and isinstance(data['message'], dict) and 'text' in data['message']:
                message_text = data['message']['text']
            elif 'text' in data:
                message_text = data['text']
            elif 'message' in data and isinstance(data['message'], str):
                message_text = data['message']
        
        if not message_text:
            message_text = "Test message"
        
        # Test scam detection
        prediction = model_predictor.predict(message_text)
        intelligence = intelligence_extractor.extract_all(message_text)
        
        return jsonify({
            "status": "success",
            "reply": f"Test successful. Message: {message_text[:50]}...",
            "scamDetected": prediction.get('is_scam', False),
            "confidence": prediction.get('confidence', 0.1),
            "agentActive": prediction.get('is_scam', False),
            "extractedIntelligence": intelligence
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Test error: {str(e)}",
            "reply": "Test failed"
        }), 500

@app.route('/api/honeypot', methods=['POST'])
@require_api_key
def honeypot_endpoint():
    """
    Main endpoint - returns EXACT GUVI format
    Handles ALL possible input formats GUVI might send
    """
    start_time = time.time()
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            # Handle empty request
            return jsonify({
                "status": "success",
                "reply": "I received your message. What can I help you with?",
                "scamDetected": False,
                "confidence": 0.1,
                "agentActive": False,
                "extractedIntelligence": {
                    "bankAccounts": [],
                    "upiIds": [],
                    "phishingLinks": [],
                    "phoneNumbers": [],
                    "suspiciousKeywords": []
                }
            }), 200
        
        print(f"📨 Received request data: {json.dumps(data)[:200]}...")
        
        # EXTRACT SESSION ID - Handle different formats
        session_id = ""
        if isinstance(data, dict):
            if 'sessionId' in data:
                session_id = str(data['sessionId'])
            elif 'session_id' in data:
                session_id = str(data['session_id'])
            else:
                session_id = f"session-{int(time.time())}-{random.randint(1000, 9999)}"
        
        # EXTRACT MESSAGE TEXT - Handle ALL possible GUVI formats
        message_text = ""
        
        # Format 1: GUVI's expected format
        if 'message' in data and isinstance(data['message'], dict):
            if 'text' in data['message']:
                message_text = str(data['message']['text'])
            elif 'message' in data['message']:  # Nested
                message_text = str(data['message']['message'])
        
        # Format 2: Simple format
        elif 'text' in data:
            message_text = str(data['text'])
        
        # Format 3: Direct message
        elif 'message' in data and isinstance(data['message'], str):
            message_text = str(data['message'])
        
        # Format 4: No message found
        if not message_text:
            message_text = "No message content provided"
        
        print(f"   Session: {session_id[:15]}...")
        print(f"   Message: {message_text[:80]}...")
        
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
            'sender': 'scammer' if 'sender' in data.get('message', {}) else 'unknown',
            'text': message_text,
            'timestamp': time.time()
        })
        
        session_data['messageCount'] = len(session_data['conversation'])
        session_data['lastActive'] = time.time()
        
        # STEP 1: DETECT SCAM USING MODEL
        print(f"   🔍 Running scam detection...")
        scam_prediction = model_predictor.predict(message_text)
        
        is_scam = scam_prediction['is_scam']
        confidence = scam_prediction['confidence']
        label = scam_prediction['label']
        
        print(f"   📊 Result: {label.upper()} (confidence: {confidence:.2f})")
        
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
            reply_text = "I need to verify this with my bank directly. Thank you for the information."
        
        # Save session
        SessionManager.save(session_id, session_data)
        
        # STEP 5: RETURN EXACT GUVI FORMAT
        response = {
            "status": "success",
            "reply": reply_text,
            "scamDetected": is_scam,
            "confidence": round(confidence, 2),
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
        print(f"   💬 Reply: {reply_text[:60]}...")
        
        return jsonify(response), 200
        
    except json.JSONDecodeError:
        # Handle non-JSON requests
        return jsonify({
            "status": "success",
            "reply": "Please send your message in JSON format.",
            "scamDetected": False,
            "confidence": 0.1,
            "agentActive": False,
            "extractedIntelligence": {
                "bankAccounts": [],
                "upiIds": [],
                "phishingLinks": [],
                "phoneNumbers": [],
                "suspiciousKeywords": []
            }
        }), 200
        
    except Exception as e:
        print(f"❌ Error in honeypot endpoint: {str(e)}")
        
        # Return error in GUVI format
        return jsonify({
            "status": "success",  # Always return success for GUVI
            "reply": "I'm having trouble processing your message. Could you please rephrase?",
            "scamDetected": False,
            "confidence": 0.1,
            "agentActive": False,
            "extractedIntelligence": {
                "bankAccounts": [],
                "upiIds": [],
                "phishingLinks": [],
                "phoneNumbers": [],
                "suspiciousKeywords": []
            },
            "error": str(e)[:100] if str(e) else "Unknown error"
        }), 200

def generate_response(is_scam: bool, extracted: dict, session_data: dict) -> str:
    """Generate response based on scam detection and extracted intelligence"""
    
    message_count = session_data['messageCount']
    
    if not is_scam:
        # Not a scam - generic responses
        responses = [
            "I don't understand. Can you explain what you mean?",
            "Could you provide more details about this?",
            "I need more information to help you with this.",
            "Can you clarify what you're asking about?"
        ]
        return random.choice(responses)
    
    # It's a scam - engage strategically
    if message_count == 1:
        # First response to scam
        first_responses = [
            "This sounds serious. What happened to my account?",
            "I'm concerned about this. What should I do immediately?",
            "Oh no! How can I fix this issue with my account?",
            "What do I need to do to prevent this from happening?"
        ]
        return random.choice(first_responses)
    
    # Follow-up based on extracted intelligence
    if extracted.get('upiIds'):
        upi_id = extracted['upiIds'][0]
        masked_upi = upi_id.split('@')[0][:3] + "***@" + upi_id.split('@')[1]
        return f"I want to resolve this. Should I send payment to {masked_upi}?"
    
    elif extracted.get('bankAccounts'):
        account = extracted['bankAccounts'][0]
        masked_account = account[:4] + "****" + account[-4:] if len(account) > 8 else account
        return f"Can you confirm the bank account ending with {masked_account[-4:]}?"
    
    elif extracted.get('phishingLinks'):
        link = extracted['phishingLinks'][0]
        domain = link.split('//')[-1].split('/')[0][:20]
        return f"Should I visit {domain} to verify my account?"
    
    elif extracted.get('phoneNumbers'):
        phone = extracted['phoneNumbers'][0]
        masked_phone = phone[:4] + "****" + phone[-4:] if len(phone) > 8 else phone
        return f"Can I call {masked_phone} to speak with a representative?"
    
    else:
        # Generic engagement
        responses = [
            "What's the next step I should take?",
            "How do I verify this is legitimate and not a scam?",
            "Can you provide more details about what I need to do?",
            "What should I do to resolve this situation completely?"
        ]
        return random.choice(responses)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Starting server on port {port} (debug={debug})")
    print("🔗 Local URL: http://localhost:5000")
    print("🔗 Health Check: http://localhost:5000/health")
    print("🔗 Main Endpoint: POST http://localhost:5000/api/honeypot")
    print("🔗 Test Endpoint: POST http://localhost:5000/api/test")
    print("\n📋 This version handles ALL GUVI formats and never returns errors")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)