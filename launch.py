#!/usr/bin/env python3
import subprocess
import sys
import webbrowser
import time
import threading

def open_browser():
    time.sleep(2)  # Wait for Streamlit to start
    webbrowser.open('http://localhost:8501')

def main():
    print("🚦 Starting Traffic Monitoring System...")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "traffic_monitor.py"])

if __name__ == "__main__":
    main()
