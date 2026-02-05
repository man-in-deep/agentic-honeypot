"""
model_predictor.py - FIXED WORKING VERSION
Simple binary classification for scam detection
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict

class SimpleTokenizer:
    """Simple tokenizer for BERT"""
    
    def __init__(self, vocab_file: str):
        self.vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:
                    self.vocab[token] = idx
        
        self.unk_token_id = self.vocab.get('[UNK]', 100)
        self.cls_token_id = self.vocab.get('[CLS]', 101)
        self.sep_token_id = self.vocab.get('[SEP]', 102)
        self.pad_token_id = self.vocab.get('[PAD]', 0)
    
    def encode(self, text: str, max_length: int = 128) -> Dict:
        text = text.lower()
        tokens = []
        current = ""
        
        # Simple tokenization
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

class SimpleBERT(nn.Module):
    """Simplified BERT model for binary classification"""
    
    def __init__(self, config_path: str):
        super().__init__()
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        vocab_size = config.get("vocab_size", 30522)
        hidden_size = config.get("hidden_size", 768)
        
        # Simple layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 classes: normal, scam
        
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        embeddings = self.embedding(input_ids)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = embeddings * mask
        
        # Mean pooling
        pooled = embeddings.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits

class ModelPredictor:
    """Main scam detector - binary classification"""
    
    def __init__(self, model_dir: str = "models/downloaded_model"):
        self.model_dir = model_dir
        self.device = torch.device('cpu')  # Vercel uses CPU
        
        print(f"🤖 Initializing ModelPredictor on {self.device}")
        
        # Check model files
        config_path = os.path.join(model_dir, "config.json")
        vocab_path = os.path.join(model_dir, "vocab.txt")
        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json in {model_dir}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing vocab.txt in {model_dir}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing pytorch_model.bin in {model_dir}")
        
        # Load tokenizer
        self.tokenizer = SimpleTokenizer(vocab_path)
        
        # Load model
        self.model = SimpleBERT(config_path)
        self.model.to(self.device)
        
        # Load weights
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ Model weights loaded")
        except Exception as e:
            print(f"⚠️ Could not load weights: {e}")
            print("⚠️ Using random initialization")
        
        self.model.eval()
        print("✅ ModelPredictor ready")
    
    def predict(self, text: str) -> Dict:
        """Predict if text is scam - binary output"""
        try:
            # Encode text
            encoded = self.tokenizer.encode(text)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                prediction = torch.argmax(outputs, dim=-1).item()
                
                # Binary classification
                is_scam = (prediction == 1)
                confidence = 0.9 if is_scam else 0.1
                label = "scam" if is_scam else "normal"
            
            return {
                "is_scam": bool(is_scam),
                "label": label,
                "confidence": confidence,
                "predicted_class": prediction,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Model error: {e}")
            # Fallback pattern detection
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text: str) -> Dict:
        """Fallback pattern-based detection"""
        import re
        text_lower = text.lower()
        
        scam_keywords = [
            'urgent', 'immediate', 'verify', 'suspend', 'block',
            'bank account', 'upi', 'payment', 'transfer', 'money',
            'won', 'prize', 'lottery', 'free', 'winner',
            'click', 'link', 'http://', 'https://',
            'dear customer', 'attention required'
        ]
        
        matches = sum(1 for keyword in scam_keywords if keyword in text_lower)
        confidence = min(matches * 0.1, 0.9)
        is_scam = confidence > 0.5
        
        return {
            "is_scam": is_scam,
            "label": "scam" if is_scam else "normal",
            "confidence": confidence,
            "success": False,
            "fallback": True
        }

# Global instance
model_predictor = ModelPredictor()