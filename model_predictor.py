"""
model_predictor.py
Uses downloaded Hugging Face model + pattern matching
Works offline with downloaded files
"""

import os
import json
import re
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

class SimpleTokenizer:
    """Tokenizer using downloaded vocab.txt"""
    
    def __init__(self, vocab_path: str):
        self.vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:
                    self.vocab[token] = idx
        
        self.unk_token_id = self.vocab.get('[UNK]', 100)
        self.cls_token_id = self.vocab.get('[CLS]', 101)
        self.sep_token_id = self.vocab.get('[SEP]', 102)
        self.pad_token_id = self.vocab.get('[PAD]', 0)
        
        print(f"✅ Tokenizer loaded: {len(self.vocab)} tokens")
    
    def encode(self, text: str, max_length: int = 128) -> Dict:
        """Simple encoding for inference"""
        text = text.lower()
        tokens = []
        current = ""
        
        # Basic tokenization
        for char in text:
            if char.isalnum():
                current += char
            else:
                if current:
                    tokens.append(current)
                    current = ""
                if char.strip():
                    tokens.append(char)
        
        if current:
            tokens.append(current)
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Truncate
        if len(token_ids) > max_length - 2:
            token_ids = token_ids[:max_length - 2]
        
        # Add special tokens
        input_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        
        # Pad
        if len(input_ids) < max_length:
            input_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))
        
        # Attention mask
        attention_mask = [1 if id != self.pad_token_id else 0 for id in input_ids]
        
        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }

class SimpleBertModel(nn.Module):
    """Simple BERT model that works with downloaded weights"""
    
    def __init__(self, config_path: str):
        super().__init__()
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.vocab_size = config.get('vocab_size', 30522)
        self.hidden_size = config.get('hidden_size', 768)
        self.num_labels = config.get('num_labels', 2)
        
        # Simple architecture
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        # Try to load weights
        self._load_weights(config_path)
    
    def _load_weights(self, config_path: str):
        """Load weights from pytorch_model.bin"""
        try:
            model_dir = os.path.dirname(config_path)
            weights_path = os.path.join(model_dir, "pytorch_model.bin")
            
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
                
                # Filter for embeddings and classifier
                filtered_dict = {}
                for key, value in state_dict.items():
                    if 'embeddings' in key or 'classifier' in key:
                        # Remove prefix if present
                        new_key = key.replace('bert.', '').replace('classifier.', '')
                        filtered_dict[new_key] = value
                
                # Load what we can
                self.load_state_dict(filtered_dict, strict=False)
                print("   ✅ Model weights loaded (partial)")
            else:
                print("   ⚠️  No weights found, using random")
                
        except Exception as e:
            print(f"   ⚠️  Could not load weights: {e}")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass - simple implementation"""
        embeddings = self.embeddings(input_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = embeddings * mask
            pooled = embeddings.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = embeddings.mean(dim=1)
        
        logits = self.classifier(pooled)
        return logits

class ModelPredictor:
    """Main predictor: Uses model + pattern fallback"""
    
    def __init__(self, model_dir: str = "models/downloaded_model"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🤖 Initializing ModelPredictor on {self.device}")
        
        # Check files
        required = ["config.json", "vocab.txt"]
        for file in required:
            if not os.path.exists(os.path.join(model_dir, file)):
                raise FileNotFoundError(f"Missing: {file}")
        
        # Initialize tokenizer
        vocab_path = os.path.join(model_dir, "vocab.txt")
        self.tokenizer = SimpleTokenizer(vocab_path)
        
        # Initialize model
        config_path = os.path.join(model_dir, "config.json")
        self.model = SimpleBertModel(config_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Scam patterns for fallback
        self.scam_patterns = [
            r'urgent', r'immediate', r'asap', r'now', r'today',
            r'bank.*account', r'suspend', r'block', r'terminate',
            r'verify.*account', r'secure.*account',
            r'upi.*id', r'send.*money', r'payment', r'transfer',
            r'won.*prize', r'lottery', r'reward', r'congratulation',
            r'click.*link', r'http://', r'https://',
            r'call.*\d{10}', r'\+\d{1,3}.*\d{10}',
            r'dear customer', r'attention required'
        ]
        
        print("✅ ModelPredictor ready (model + patterns)")
    
    def predict(self, text: str) -> Dict:
        """
        Predict using model first, fallback to patterns
        Returns: {'is_scam': bool, 'label': str, 'confidence': float}
        """
        # Try model first
        model_result = self._model_predict(text)
        
        # If model fails or low confidence, use patterns
        if model_result.get('success', False) and model_result.get('confidence', 0) > 0.6:
            return model_result
        else:
            return self._pattern_predict(text)
    
    def _model_predict(self, text: str) -> Dict:
        """Use downloaded model for prediction"""
        try:
            encoded = self.tokenizer.encode(text)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=-1)
                
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_class].item()
            
            # Binary classification: 0=normal, 1=scam
            is_scam = (pred_class == 1)
            
            return {
                "is_scam": bool(is_scam),
                "label": "scam" if is_scam else "normal",
                "confidence": float(confidence),
                "method": "model",
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            return {
                "is_scam": False,
                "label": "error",
                "confidence": 0.0,
                "method": "error",
                "success": False,
                "error": str(e)
            }
    
    def _pattern_predict(self, text: str) -> Dict:
        """Pattern-based fallback detection"""
        text_lower = text.lower()
        matches = 0
        
        for pattern in self.scam_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        # Calculate confidence
        confidence = min(matches * 0.2, 0.95)
        is_scam = confidence > 0.5
        
        return {
            "is_scam": bool(is_scam),
            "label": "scam" if is_scam else "normal",
            "confidence": float(confidence),
            "method": "pattern",
            "success": True
        }
    
    def analyze_scam_type(self, text: str) -> str:
        """Analyze scam type based on keywords"""
        text_lower = text.lower()
        
        if 'bank' in text_lower and ('suspend' in text_lower or 'block' in text_lower):
            return "bank_phishing"
        elif 'upi' in text_lower or '@' in text_lower:
            return "upi_fraud"
        elif 'won' in text_lower or 'prize' in text_lower or 'lottery' in text_lower:
            return "lottery_scam"
        elif 'hack' in text_lower or 'virus' in text_lower:
            return "tech_support"
        elif 'free' in text_lower or 'gift' in text_lower:
            return "prize_scam"
        elif 'suspend' in text_lower or 'block' in text_lower:
            return "account_suspension"
        else:
            return "unknown_scam"

# Global instance
model_predictor = ModelPredictor()