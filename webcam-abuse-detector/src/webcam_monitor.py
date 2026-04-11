"""
Module 1: Webcam Access Monitor
Monitors webcam ON/OFF events in real-time

Author: Eva Elizabeth
Date: February 2026
"""

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import time
import logging
from datetime import datetime
import threading
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_identifier import run
current_session_processes = None

# Import configuration
try:
    from config import (
        WEBCAM_CHECK_INTERVAL, 
        WEBCAM_DEVICE_INDEX,
        LOG_FILE,
        LOG_FORMAT,
        LOG_LEVEL
    )
except ImportError:
    # Fallback if config not found
    WEBCAM_CHECK_INTERVAL = 2
    WEBCAM_DEVICE_INDEX = 0
    LOG_FILE = 'logs/webcam_monitor.log'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'


class WebcamMonitor:
    """
    Monitors webcam access and detects ON/OFF events
    
    This class continuously checks if the webcam is being used
    and notifies registered observers when the state changes.
    """
    
    def __init__(self, check_interval=WEBCAM_CHECK_INTERVAL, auto_stop_on_off=True):
        """
        Initialize the webcam monitor
        
        Args:
            check_interval (int): How often to check webcam status (seconds)
            auto_stop_on_off (bool): If True, monitor stops when webcam turns OFF after being ON
        """
        self.check_interval = check_interval
        self.is_running = False
        self.webcam_status = False  # Current status: True = ON, False = OFF
        self.previous_status = False
        self.observers = []  # List of callback functions
        self.logger = self._setup_logger()
        self.auto_stop_on_off = auto_stop_on_off
        self.webcam_was_on = False  # Track if webcam was ever ON
        self.webcam_start_time = None
        print(f"✓ WebcamMonitor initialized (checking every {check_interval}s)")
        if auto_stop_on_off:
            print(f"✓ Auto-stop enabled: Will stop when webcam turns OFF")
    
    def _setup_logger(self):
        """
        Set up logging configuration
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger('WebcamMonitor')
        logger.setLevel(LOG_LEVEL)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create file handler
        try:
            file_handler = logging.FileHandler(LOG_FILE)
            file_handler.setLevel(LOG_LEVEL)
            
            # Create formatter
            formatter = logging.Formatter(LOG_FORMAT)
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
            print(f"✓ Logging configured: {LOG_FILE}")
        except Exception as e:
            print(f"⚠️  Warning: Could not set up file logging: {e}")
            # Add console handler as fallback
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(console_handler)
        
        return logger
    
    def check_webcam_status(self):
        """
        Check if the webcam is currently in use
        
        This method tries to access the webcam. If it can successfully
        open and read from it, the webcam is available (not in use).
        If it fails, another application is using it.
        
        Returns:
            bool: True if webcam is in use, False otherwise
        """
        try:
            # Try to open the webcam
            cap = cv2.VideoCapture(WEBCAM_DEVICE_INDEX)
            
            # Check if webcam opened successfully
            if not cap.isOpened():
                cap.release()
                return True  # Webcam is in use by another app
            
            # Try to read a frame
            ret, frame = cap.read()
            
            # Release the webcam immediately
            cap.release()
            
            # If we couldn't read, webcam might be in use
            if not ret:
                return True
            
            # Successfully accessed webcam - it's available (not in use)
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking webcam status: {e}")
            return False
    
    def register_observer(self, callback):
        """
        Register a callback function to be notified of webcam events
        
        Args:
            callback (function): Function to call when webcam state changes
                                Should accept event_data dictionary as parameter
        """
        if callback not in self.observers:
            self.observers.append(callback)
            self.logger.info(f"Observer registered: {callback.__name__}")
            print(f"✓ Observer registered: {callback.__name__}")
    
    def unregister_observer(self, callback):
        """
        Remove a callback function from the observers list
        
        Args:
            callback (function): The callback function to remove
        """
        if callback in self.observers:
            self.observers.remove(callback)
            self.logger.info(f"Observer unregistered: {callback.__name__}")

    def notify_observers(self, event_type, timestamp, duration_minutes=0):
        """
        Notify all registered observers of a webcam event
        
        Args:
            event_type (str): Type of event ("WEBCAM_ON" or "WEBCAM_OFF")
            timestamp (datetime): When the event occurred
        """
        event_data = {
          'event_type': event_type,
          'timestamp': timestamp.isoformat(),
          'status': self.webcam_status,
          'duration_minutes': duration_minutes
         }
        
        # Call each observer with the event data
        for observer in self.observers:
            try:
                observer(event_data)
            except Exception as e:
                self.logger.error(f"Error notifying observer {observer.__name__}: {e}")
    
    def start_monitoring(self):
        """
        Start continuous webcam monitoring
        
        This runs in a loop, checking webcam status at regular intervals.
        When status changes, it notifies all registered observers.
        If auto_stop_on_off is True, monitoring stops when webcam turns OFF after being ON.
        """
        self.is_running = True
        self.logger.info("=" * 60)
        self.logger.info("Webcam monitoring STARTED")
        self.logger.info("=" * 60)
        
        print("\n" + "=" * 60)
        print("🚀 WEBCAM MONITORING STARTED")
        print("=" * 60)
        print(f"📹 Checking webcam every {self.check_interval} seconds")
        print(f"📝 Logs: {LOG_FILE}")
        if self.auto_stop_on_off:
            print("🔄 Auto-stop: Monitor will stop when webcam turns OFF")
        print("⌨️  Press Ctrl+C to stop manually")
        print("=" * 60 + "\n")
        
        try:
            while self.is_running:
                # Check current webcam status
                current_status = self.check_webcam_status()
                timestamp = datetime.now()
                
                # Detect state change
                if current_status != self.previous_status:
                    self.webcam_status = current_status
                    
                    if current_status:
                        event_type = "WEBCAM_ON"
                        message = "🔴 WEBCAM TURNED ON"
                        log_level = logging.WARNING
                        self.webcam_was_on = True  # Mark that webcam was turned on
                        self.webcam_start_time = timestamp
                        duration_minutes = 0
                    else:
                        event_type = "WEBCAM_OFF"
                        message = "⚪ WEBCAM TURNED OFF"
                        log_level = logging.INFO
                        if self.webcam_start_time:
                           duration_minutes = (timestamp - self.webcam_start_time).total_seconds() / 60
                        else:
                           duration_minutes = 0

                        print(f"⏱ Webcam usage duration: {round(duration_minutes,2)} minutes")
                    # Log the event
                    self.logger.log(log_level, f"{event_type} detected at {timestamp}")
                    
                    # Print to console
                    print(f"\n{'='*60}")
                    print(f"{message}")
                    print(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*60}\n")
                    
                    # Notify all observers
                    self.notify_observers(event_type, timestamp, duration_minutes)
                    
                    # Update previous status
                    self.previous_status = current_status
                    
                    # Auto-stop if enabled and webcam turned OFF after being ON
                    if self.auto_stop_on_off and not current_status and self.webcam_was_on:
                        print("\n🛑 Webcam turned OFF - Auto-stopping monitor...")
                        self.logger.info("Auto-stopping: Webcam turned OFF after being ON")
                        self.stop_monitoring()
                        break
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n⚠️  Received stop signal...")
            self.stop_monitoring()
        except Exception as e:
            self.logger.error(f"Fatal error in monitoring loop: {e}")
            print(f"\n❌ Error: {e}")
            self.stop_monitoring()
    
    def start_monitoring_background(self):
        """
        Start monitoring in a background thread
        
        Returns:
            threading.Thread: The monitoring thread
        """
        monitor_thread = threading.Thread(
            target=self.start_monitoring,
            daemon=True,
            name="WebcamMonitorThread"
        )
        monitor_thread.start()
        return monitor_thread
    
    def stop_monitoring(self):
        """
        Stop the webcam monitoring
        """
        self.is_running = False
        self.logger.info("Webcam monitoring STOPPED")
        print("\n✓ Webcam monitoring stopped")
    
    def get_status(self):
        """
        Get current webcam status
        
        Returns:
            dict: Current status information
        """
        return {
            'is_running': self.is_running,
            'webcam_active': self.webcam_status,
            'check_interval': self.check_interval,
            'observers_count': len(self.observers),
            'auto_stop_on_off': self.auto_stop_on_off
        }


# Simple test callback function
def test_callback(event_data):
    """Example callback function for testing"""
    print(f"📢 Callback received: {event_data['event_type']}")
    print(f"   Timestamp: {event_data['timestamp']}")

def process_identification_callback(event_data):

    global current_session_processes

    print(f"\n📢 Event detected: {event_data['event_type']}")

    # When webcam turns ON → detect processes
    if event_data["event_type"] == "WEBCAM_ON":

        print("🔍 Detecting webcam processes...\n")

        # store detected processes
        current_session_processes = run("json", 0)

    # When webcam turns OFF → run ML detection with duration
    if event_data["event_type"] == "WEBCAM_OFF":

        duration = event_data["duration_minutes"]

        print(f"⏱ Webcam session duration: {duration:.2f} minutes")

        if current_session_processes:
            run("json", duration)

# Main execution (for testing this module standalone)
if __name__ == "__main__":

    monitor = WebcamMonitor(check_interval=2, auto_stop_on_off=True)

    monitor.register_observer(process_identification_callback)

    try:
        monitor.start_monitoring()

    except KeyboardInterrupt:
        monitor.stop_monitoring()