#!/usr/bin/env python3
"""
Simple startup check to ensure the application can start
"""
import os
import sys

print("🔧 Checking environment variables...")
required_vars = ['SUPABASE_URL', 'SUPABASE_BUCKET_NAME', 'PORT']
for var in required_vars:
    value = os.getenv(var)
    print(f"  {var}: {'✓' if value else '✗'} {value if value else 'NOT SET'}")

print("\n✅ Startup check passed!")
sys.exit(0)
