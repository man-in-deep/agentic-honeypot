#!/usr/bin/env python3
"""
model_downloader.py
Downloads Hugging Face model for scam detection
RUN THIS ONCE LOCALLY, then push model folder to Git
"""

import os
import requests
import json
import time
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
            response = requests.get(file_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ Downloaded: {file_size:.2f} MB")
            downloaded += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            # Try alternative URL
            try:
                alt_url = f"https://huggingface.co/{model_name}/raw/main/{file}"
                response = requests.get(alt_url, timeout=30)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"   ✅ Downloaded from alternative URL")
                downloaded += 1
            except:
                print(f"   ❌ Alternative also failed")
    
    print("\n" + "=" * 60)
    if downloaded >= 3:
        print(f"✅ SUCCESS: Downloaded {downloaded}/5 files")
        print("📊 Model ready for use!")
    else:
        print(f"❌ FAILED: Only {downloaded}/5 files downloaded")
        print("   Manual download needed:")
        print("   cd models/downloaded_model")
        print("   wget https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection/resolve/main/pytorch_model.bin")
        print("   wget https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection/resolve/main/config.json")
        print("   wget https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection/resolve/main/vocab.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    print("🤖 AGENTIC HONEY-POT - MODEL DOWNLOADER")
    print()
    print("⚠️  IMPORTANT: Run this once locally, then push model folder to Git")
    print("   Vercel will use the model from Git, not download it")
    print()
    
    download_huggingface_model()
    
    print("\n📋 NEXT STEPS:")
    print("1. Verify model files: ls -la models/downloaded_model/")
    print("2. Commit to Git: git add models/downloaded_model/")
    print("3. Push to GitHub")
    print("4. Deploy on Vercel")