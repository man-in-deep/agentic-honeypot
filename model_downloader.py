#!/usr/bin/env python3
"""
model_downloader.py
Downloads Hugging Face model for scam detection
Model: bert-tiny-finetuned-sms-spam-detection (17MB)
"""

import os
import requests
import json
import sys

def download_huggingface_model():
    """Download BERT model for spam detection"""
    print("=" * 60)
    print("📥 DOWNLOADING HUGGING FACE MODEL")
    print("=" * 60)
    
    # Model: bert-tiny-finetuned-sms-spam-detection (17MB)
    model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    base_url = f"https://huggingface.co/{model_name}/resolve/main"
    
    # Files we need
    files = [
        "config.json",
        "pytorch_model.bin", 
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    # Create directory
    os.makedirs("models/downloaded_model", exist_ok=True)
    
    print(f"💾 Model: {model_name}")
    print(f"📁 Saving to: models/downloaded_model/")
    print()
    
    downloaded = 0
    for file in files:
        file_path = f"models/downloaded_model/{file}"
        file_url = f"{base_url}/{file}"
        
        print(f"⬇️  Downloading {file}...")
        
        try:
            # Download with progress
            response = requests.get(file_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ Downloaded: {file_size:.2f} MB")
            downloaded += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print(f"   ⚠️  Trying alternative download...")
            
            # Try alternative URL
            try:
                alt_url = f"https://huggingface.co/{model_name}/raw/main/{file}"
                response = requests.get(alt_url, timeout=30)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"   ✅ Downloaded from alternative")
                downloaded += 1
            except:
                print(f"   ❌ Alternative also failed")
    
    print("\n" + "=" * 60)
    if downloaded >= 3:  # Need at least config, model, vocab
        print(f"✅ SUCCESS: Downloaded {downloaded}/5 files")
        print("📊 Model ready for use!")
    else:
        print(f"❌ FAILED: Only {downloaded}/5 files downloaded")
        print("   Creating fallback model...")
        create_fallback_model()
    
    print("=" * 60)
    
    # Test model files
    test_model_files()

def create_fallback_model():
    """Create fallback model if download fails"""
    print("\n🔄 Creating fallback model...")
    
    # Simple config
    config = {
        "model_type": "bert",
        "hidden_size": 128,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "num_labels": 2
    }
    
    # Create vocab
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab.extend([f"word{i}" for i in range(1000)])  # Add some dummy words
    
    os.makedirs("models/downloaded_model", exist_ok=True)
    
    # Save config
    with open("models/downloaded_model/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save vocab
    with open("models/downloaded_model/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")
    
    # Create dummy model file
    import torch
    dummy_model = {"dummy": "weights"}
    torch.save(dummy_model, "models/downloaded_model/pytorch_model.bin")
    
    print("✅ Fallback model created (will use pattern-based detection)")

def test_model_files():
    """Test if model files are valid"""
    print("\n🔍 TESTING MODEL FILES...")
    
    # Check essential files
    essential = ["config.json", "pytorch_model.bin", "vocab.txt"]
    
    for file in essential:
        path = f"models/downloaded_model/{file}"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"   ✅ {file}: {size:.1f} KB")
        else:
            print(f"   ❌ {file}: MISSING")
    
    # Check config
    try:
        with open("models/downloaded_model/config.json", "r") as f:
            config = json.load(f)
        print(f"   ✅ Config valid: {config.get('model_type', 'unknown')} model")
    except:
        print("   ❌ Config invalid")

if __name__ == "__main__":
    print("🤖 AGENTIC HONEY-POT - MODEL DOWNLOADER")
    print()
    
    download_huggingface_model()
    
    print("\n📋 NEXT:")
    print("1. Run: python setup_env.py")
    print("2. Add firebase-credentials.json (from Firebase Console)")
    print("3. Run: python app.py")