#!/usr/bin/env python3
"""
model_downloader.py - Local use only to download model
Download model and push to Git for Vercel
"""

import os
import requests
import json
import sys

def download_model():
    """Download Hugging Face model"""
    print("=" * 60)
    print("📥 DOWNLOADING MODEL (Local only - for Git)")
    print("=" * 60)
    
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
    
    for file in files:
        print(f"Downloading {file}...")
        try:
            response = requests.get(f"{base_url}/{file}")
            with open(f"models/downloaded_model/{file}", "wb") as f:
                f.write(response.content)
            print(f"  ✅ {file}")
        except Exception as e:
            print(f"  ❌ {file}: {e}")
    
    print("\n✅ Model downloaded to models/downloaded_model/")
    print("\n📋 IMPORTANT: Commit these files to Git for Vercel deployment")
    print("   git add models/downloaded_model/")
    print("   git commit -m 'Add model files'")

if __name__ == "__main__":
    download_model()