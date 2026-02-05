#!/usr/bin/env python3
"""
keepalive.py
Keeps Render app alive by pinging every 5 minutes
Prevents 15-minute sleep on free tier
"""

import os
import requests
import time
import schedule
import threading

def ping_app():
    """Ping the web service to keep it alive"""
    url = os.getenv('RENDER_URL', '')
    if not url:
        print("⚠️  RENDER_URL not set")
        return
    
    try:
        # Try health endpoint first
        health_url = f"{url}/health"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Ping successful: {response.json().get('status')}")
        else:
            print(f"⚠️  Ping failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Ping error: {type(e).__name__}")

def run_scheduler():
    """Run scheduler in background"""
    # Schedule ping every 5 minutes
    schedule.every(5).minutes.do(ping_app)
    
    print("🔄 Keep-alive scheduler started")
    print(f"📡 Will ping every 5 minutes")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 KEEP-ALIVE WORKER STARTING")
    print("=" * 60)
    print("This worker prevents Render from sleeping")
    print("Free tier apps sleep after 15 minutes of inactivity")
    print("We ping every 5 minutes to keep it alive")
    print("=" * 60)
    
    # Initial ping
    ping_app()
    
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n👋 Shutting down keep-alive worker")