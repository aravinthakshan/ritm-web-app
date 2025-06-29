#!/usr/bin/env python3
"""
Integrated startup script for the Interactive Segmentation Demo.
This script starts both the Flask backend and Next.js frontend.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def start_flask_backend():
    """Start the Flask backend server."""
    print("🚀 Starting Flask backend...")
    os.chdir("web_demo")
    
    # Install flask-cors if not already installed
    try:
        import flask_cors
    except ImportError:
        print("📦 Installing flask-cors...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask-cors"], check=True)
    
    # Start Flask app
    subprocess.run([sys.executable, "app.py"])

def start_nextjs_frontend():
    """Start the Next.js frontend."""
    print("🎨 Starting Next.js frontend...")
    os.chdir("interactive-segmentation-demo")
    
    # Install dependencies if node_modules doesn't exist
    if not os.path.exists("node_modules"):
        print("📦 Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], check=True)
    
    # Start Next.js development server
    subprocess.run(["npm", "run", "dev"])

def main():
    print("=" * 60)
    print("🎯 Interactive Segmentation Demo - Integrated Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("web_demo") or not os.path.exists("web_demo/interactive-segmentation-demo"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected structure:")
        print("   ├── web_demo/")
        print("   │   ├── app.py")
        print("   │   └── interactive-segmentation-demo/")
        sys.exit(1)
    
    # Store original directory
    original_dir = os.getcwd()
    
    # Start Flask backend in a separate thread
    flask_thread = threading.Thread(target=start_flask_backend, daemon=True)
    flask_thread.start()
    
    # Wait a moment for Flask to start
    time.sleep(3)
    
    # Start Next.js frontend
    try:
        start_nextjs_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
    finally:
        # Return to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main() 