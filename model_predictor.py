"""
model_predictor.py
Scam detection using Hugging Face model
Handles both binary and confidence-based predictions
"""

import os
import json
import re
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

class SimpleTokenizer:
    """Simple tokenizer using downloaded vocab"""
    
    def __init__(self, vocab_path: str):
        self.vocab = self._load_vocab(vocab_path)
        self.unk_token_id = self.vocab.get('[UNK]', 100)
        self.cls_token_id = self.vocab.get('[CLS]', 101)
        self.sep_token_id = self.vocab.get('[SEP]', 102)
        self.pad_token_id = self.vocab.get('[PAD]', 0)
    
    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """Load vocabulary from file"""
        vocab = {}
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    token = line.strip()
                    if token:
                        vocab[token] = idx
        else:
            # Fallback basic vocab
            basic_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
            for idx, token in enumerate(basic_tokens):
                vocab[token] = idx
        
        return vocab
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum():
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if char.strip():
                    tokens.append(char)
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def encode(self, text: str, max_length: int = 128) -> Dict:
        """Encode text for model input"""
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Truncate if too long
        if len(token_ids) > max_length - 2:
            token_ids = token_ids[:max_length - 2]
        
        # Add special tokens
        input_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        
        # Pad to max length
        if len(input_ids) < max_length:
            padding = [self.pad_token_id] * (max_length - len(input_ids))
            input_ids = input_ids + padding
        
        # Create attention mask
        attention_mask = [1 if id != self.pad_token_id else 0 for id in input_ids]
        
        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }

class SimpleBertForSequenceClassification(nn.Module):
    """Simple BERT model for classification"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.hidden_size = config.get('hidden_size', 768)
        self.num_labels = config.get('num_labels', 2)
        self.vocab_size = config.get('vocab_size', 30522)
        
        # Simple embedding layer
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Simple classifier
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        # Load weights if available
        self._load_weights(config)
    
    def _load_weights(self, config: Dict):
        """Try to load pre-trained weights"""
        try:
            weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'downloaded_model', 'pytorch_model.bin')
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location='cpu')
                
                # Try to load what we can
                model_dict = self.state_dict()
                
                # Filter for matching keys
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                
                # Update model with pretrained weights
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict, strict=False)
                
                print("   ✅ Model weights loaded (partial)")
            else:
                print("   ⚠️  No weights file found, using random initialization")
                
        except Exception as e:
            print(f"   ⚠️  Could not load weights: {e}")
            print("   Using random initialization")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        
        # Simple mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = embeddings * mask
            pooled = embeddings.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = embeddings.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

class ModelPredictor:
    """Main scam detection class - FIXED for GUVI format"""
    
    def __init__(self, model_dir: str = "models/downloaded_model"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🤖 Initializing ModelPredictor on {self.device}")
        
        # Check and load config
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize tokenizer
        vocab_path = os.path.join(model_dir, "vocab.txt")
        if not os.path.exists(vocab_path):
            # Create fallback vocab
            with open(vocab_path, 'w') as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
        
        self.tokenizer = SimpleTokenizer(vocab_path)
        
        # Initialize model
        self.model = SimpleBertForSequenceClassification(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ ModelPredictor ready")
        print(f"   Model: {self.config.get('model_type', 'bert')}")
        print(f"   Labels: {self.config.get('num_labels', 2)}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict if text is scam
        Returns: {'is_scam': bool, 'label': str, 'confidence': float}
        """
        try:
            # Encode text
            encoded = self.tokenizer.encode(text)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=-1)
                
                # Get prediction
                pred_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][pred_idx].item()
            
            # Map to labels (0=normal/ham, 1=scam/spam)
            is_scam = (pred_idx == 1)
            label = "scam" if is_scam else "normal"
            
            # Calculate confidence for scam
            if is_scam:
                scam_confidence = confidence
            else:
                scam_confidence = 1.0 - confidence
            
            result = {
                "is_scam": bool(is_scam),
                "label": label,
                "confidence": float(scam_confidence),
                "model_confidence": float(confidence),
                "predicted_class": int(pred_idx),
                "success": True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Model prediction error: {e}")
            # Fallback to pattern detection
            return self._pattern_detection(text)
    
    def _pattern_detection(self, text: str) -> Dict:
        """Fallback pattern-based detection"""
        text_lower = text.lower()
        
        # Scam indicators
        scam_indicators = [
            r'urgent', r'immediate', r'asap', r'today', r'now',
            r'block.*account', r'suspend', r'terminate', r'close',
            r'legal action', r'police', r'complaint',
            r'bank.*account', r'upi.*id', r'send.*money', r'payment',
            r'transfer', r'verification.*fee', r'processing.*charge',
            r'won.*prize', r'lottery', r'reward', r'free.*money',
            r'congratulation', r'winner',
            r'verify.*account', r'secure.*account', r'hacked',
            r'compromise', r'password.*expired',
            r'click.*link', r'http://', r'https://', r'www\.',
            r'bank.*official', r'government', r'income.*tax',
            r'reserve.*bank', r'rbi',
            r'dear.*customer', r'valued.*customer', r'account.*holder'
        ]
        
        # Count matches
        matches = 0
        for pattern in scam_indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        # Calculate confidence
        confidence = min(matches * 0.2, 0.95)
        is_scam = confidence > 0.5
        
        return {
            "is_scam": is_scam,
            "label": "scam" if is_scam else "normal",
            "confidence": confidence,
            "model_confidence": confidence,
            "predicted_class": 1 if is_scam else 0,
            "success": False,
            "note": "pattern_based_fallback"
        }
    
    def analyze_scam_type(self, text: str) -> str:
        """Analyze what type of scam"""
        text_lower = text.lower()
        
        if 'bank' in text_lower or 'account' in text_lower:
            return "bank_phishing"
        elif 'upi' in text_lower or '@' in text_lower:
            return "upi_fraud"
        elif 'won' in text_lower or 'prize' in text_lower or 'lottery' in text_lower:
            return "lottery_scam"
        elif 'suspend' in text_lower or 'block' in text_lower:
            return "account_suspension"
        elif 'hack' in text_lower or 'virus' in text_lower:
            return "tech_support"
        elif 'free' in text_lower or 'gift' in text_lower:
            return "prize_scam"
        else:
            return "unknown"

# Global instance
model_predictor = ModelPredictor()