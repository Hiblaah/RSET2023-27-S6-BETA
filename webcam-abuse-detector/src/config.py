"""
Configuration file for Webcam Abuse Detection System
Module 1: Webcam Monitor Configuration
"""

import os

# Get the base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Webcam Monitoring Settings
WEBCAM_CHECK_INTERVAL = 2  # Check every 2 seconds
WEBCAM_DEVICE_INDEX = 0    # Default webcam (0 = first camera)

# Logging Settings
LOG_FILE = os.path.join(LOGS_DIR, 'webcam_monitor.log')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

print(f"✓ Configuration loaded")
print(f"✓ Logs will be saved to: {LOG_FILE}")

