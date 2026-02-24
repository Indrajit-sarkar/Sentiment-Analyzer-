"""
Setup script for Facial Sentiment Analysis System
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def create_directories():
    """Create necessary directories"""
    directories = ['exports', 'models', 'uploads']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")

def install_requirements():
    """Install required packages"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Error installing dependencies")
        sys.exit(1)

def verify_installation():
    """Verify critical packages are installed"""
    print("\nVerifying installation...")
    try:
        import torch
        import cv2
        import flask
        print("✓ PyTorch installed")
        print("✓ OpenCV installed")
        print("✓ Flask installed")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available (CPU mode will be used)")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("Facial Sentiment Analysis System - Setup")
    print("=" * 60)
    
    check_python_version()
    create_directories()
    install_requirements()
    
    if verify_installation():
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("=" * 60)
        print("\nTo start the application, run:")
        print("  python app.py")
        print("\nThen open your browser to:")
        print("  http://localhost:5000")
    else:
        print("\n✗ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
