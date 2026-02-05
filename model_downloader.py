#!/usr/bin/env python3
"""
model_downloader.py
Downloads Hugging Face model and saves to models/downloaded_model/
Run this ONCE locally, then commit to Git
"""

import os
import requests
import json
import sys

def download_model():
    """Download BERT model for spam detection"""
    print("=" * 60)
    print("📥 DOWNLOADING HUGGING FACE MODEL FOR GIT")
    print("=" * 60)
    
    # Model: bert-tiny-finetuned-sms-spam-detection (17MB)
    model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    base_url = f"https://huggingface.co/{model_name}/resolve/main"
    
    files = [
        "config.json",
        "pytorch_model.bin", 
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    os.makedirs("models/downloaded_model", exist_ok=True)
    
    print(f"💾 Model: {model_name}")
    print(f"📁 Saving to: models/downloaded_model/ (for Git)")
    print()
    
    for file in files:
        file_path = f"models/downloaded_model/{file}"
        file_url = f"{base_url}/{file}"
        
        print(f"⬇️  {file}...", end=" ")
        
        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            size = os.path.getsize(file_path) / 1024
            print(f"✅ {size:.1f} KB")
            
        except Exception as e:
            print(f"❌ {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ MODEL DOWNLOADED FOR GIT")
    print("📋 Next: git add models/ && git commit -m 'Add model files'")
    print("=" * 60)
    return True

if __name__ == "__main__":
    download_model()