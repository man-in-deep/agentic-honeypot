"""
model_predictor.py
UPDATED - Uses the simple working model from direct_model_test.py
Binary output: SCAM or NORMAL only
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List

class SimpleTokenizer:
    """Simple tokenizer - same as direct_model_test.py"""
    
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
        
        # Truncate and add special tokens
        if len(token_ids) > max_length - 2:
            token_ids = token_ids[:max_length - 2]
        
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
    """Simple BERT model - same as direct_model_test.py"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.classifier = nn.Linear(config["hidden_size"], 2)  # 2 classes: normal, scam
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        
        # Mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = embeddings * mask
            pooled = embeddings.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = embeddings.mean(dim=1)
        
        return self.classifier(pooled)

class ModelPredictor:
    """Main scam detection - binary output only"""
    
    def __init__(self, model_dir: str = "models/downloaded_model"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🤖 Initializing ModelPredictor on {self.device}")
        
        # Check files
        required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                raise FileNotFoundError(f"Missing model file: {file}")
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize tokenizer
        vocab_path = os.path.join(model_dir, "vocab.txt")
        self.tokenizer = SimpleTokenizer(vocab_path)
        
        # Initialize model
        self.model = SimpleBertModel(self.config)
        self.model.to(self.device)
        
        # Load weights
        self._load_weights()
        
        self.model.eval()
        print("✅ ModelPredictor ready")
    
    def _load_weights(self):
        """Load pre-trained weights"""
        try:
            weights_path = os.path.join(self.model_dir, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Filter for our simple architecture
            simple_state_dict = {}
            for key, value in state_dict.items():
                if 'embeddings' in key or 'classifier' in key:
                    # Remove 'bert.' prefix if present
                    new_key = key.replace('bert.', '')
                    simple_state_dict[new_key] = value
            
            self.model.load_state_dict(simple_state_dict, strict=False)
            print("   ✅ Model weights loaded")
            
        except Exception as e:
            print(f"   ⚠️ Could not load weights: {e}")
            print("   Using random weights (model will not work properly)")
    
    def predict(self, text: str) -> Dict:
        """
        Predict if text is scam
        Returns: {'is_scam': bool, 'label': 'scam' or 'normal'}
        """
        try:
            # Encode text
            encoded = self.tokenizer.encode(text)
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                
                # Binary prediction only
                prediction = torch.argmax(outputs, dim=-1).item()
                
                # Map to labels
                is_scam = (prediction == 1)
                label = "scam" if is_scam else "normal"
            
            return {
                "is_scam": bool(is_scam),
                "label": label,
                "predicted_class": prediction,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Model prediction error: {e}")
            return {
                "is_scam": False,
                "label": "error",
                "predicted_class": -1,
                "success": False,
                "error": str(e)
            }
    
    def analyze_scam_type(self, text: str) -> str:
        """Simple scam type analysis based on keywords"""
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