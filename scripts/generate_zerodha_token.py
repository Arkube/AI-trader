#!/usr/bin/env python3
"""
Generate Zerodha Kite Connect Access Token
──────────────────────────────────────────
Run this script once per day after 7:30 AM to generate a fresh access token.
The token is valid for the trading day.

Usage:
  python scripts/generate_zerodha_token.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from kiteconnect import KiteConnect

def main():
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ KITE_API_KEY and KITE_API_SECRET must be set in .env")
        return 1
    
    kite = KiteConnect(api_key=api_key)
    
    print("=" * 60)
    print("  ZERODHA KITE CONNECT - ACCESS TOKEN GENERATOR")
    print("=" * 60)
    print(f"\nAPI Key: {api_key}")
    print(f"\n📋 Step 1: Open this URL in your browser:")
    print(f"\n   {kite.login_url()}")
    print(f"\n📋 Step 2: Login with your Zerodha credentials")
    print(f"\n📋 Step 3: After login, you'll be redirected to a URL like:")
    print(f"   https://127.0.0.1/?request_token=XXXXXX&action=login&status=success")
    print(f"\n📋 Step 4: Copy the 'request_token' value (the XXXXXX part)")
    
    request_token = input("\n🔑 Paste request_token here: ").strip()
    
    if not request_token:
        print("❌ No request token provided")
        return 1
    
    try:
        print("\n🔄 Generating access token...")
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        
        print(f"\n✅ Access Token Generated!")
        print(f"\n   Access Token: {access_token}")
        print(f"   Valid until: End of trading day")
        
        # Update .env file
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            content = env_path.read_text()
            # Replace or add KITE_ACCESS_TOKEN
            lines = content.splitlines()
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("KITE_ACCESS_TOKEN="):
                    lines[i] = f"KITE_ACCESS_TOKEN={access_token}"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"KITE_ACCESS_TOKEN={access_token}")
            
            env_path.write_text("\n".join(lines) + "\n")
            print(f"\n✅ Updated .env file: {env_path}")
        
        print(f"\n🎉 Done! You can now use Zerodha for live data.")
        print(f"   Set TRADE_MODE=zerodha in .env to enable")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating access token: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())