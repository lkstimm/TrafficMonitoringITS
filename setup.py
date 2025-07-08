#!/usr/bin/env python3
"""
Setup script for Traffic Monitoring System
Automates installation and initial configuration
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print("✅ Python version is compatible")
    return True

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def create_directories():
    """Create necessary directories"""
    directories = [
        "videos",
        "models",
        "roi_configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def download_yolo_model():
    """Download YOLO model if not present"""
    print("🤖 Checking YOLO model...")
    
    # The model will be downloaded automatically by ultralytics on first run
    # We just need to test if ultralytics is working
    try:
        import ultralytics
        print("✅ Ultralytics package is ready")
        print("ℹ️  YOLO model will be downloaded automatically on first run")
        return True
    except ImportError:
        print("❌ Ultralytics package not found. Please install requirements first.")
        return False

def create_launch_script():
    """Create a convenient launch script"""
    launch_script_content = f"""#!/usr/bin/env python3
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
"""
    
    with open("launch.py", "w") as f:
        f.write(launch_script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod("launch.py", 0o755)
    
    print("✅ Created launch script: launch.py")
    return True

def setup_complete_message():
    """Display completion message"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Run the application:")
    print("   python launch.py")
    print("   or")
    print("   streamlit run traffic_monitor.py")
    print("\n2. Open your browser to: http://localhost:8501")
    print("\n3. Load a video (YouTube URL or local file)")
    print("4. Define ROIs by drawing polygons")
    print("5. Monitor traffic in real-time!")
    print("\n🔗 For detailed usage instructions, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print("🚦 Traffic Monitoring System Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Check YOLO model
    if not download_yolo_model():
        sys.exit(1)
    
    # Create launch script
    if not create_launch_script():
        sys.exit(1)
    
    # Setup complete
    setup_complete_message()

if __name__ == "__main__":
    main() 