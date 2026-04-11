"""
Test script for Process Identifier module
"""

import sys
import time

sys.path.append('..')
from src.process_identifier import ProcessIdentifier


def main():
    print("=" * 60)
    print("Process Identifier Test Script")
    print("=" * 60)
    print()
    
    # Initialize
    identifier = ProcessIdentifier()
    print("✓ ProcessIdentifier initialized")
    print()
    
    # Test 1: Get all processes
    print("Test 1: Getting all running processes...")
    all_procs = identifier.get_all_processes()
    print(f"✓ Found {len(all_procs)} running processes")
    print()
    
    # Test 2: Find camera processes
    print("Test 2: Looking for processes using camera...")
    print("(Open Zoom, Teams, or any camera app now)")
    print("Scanning in 5 seconds...")
    time.sleep(5)
    
    camera_procs = identifier.get_processes_using_camera()
    
    if camera_procs:
        print(f"✓ Found {len(camera_procs)} process(es) using camera:")
        print()
        
        for proc in camera_procs:
            print(f"Process Name: {proc['name']}")
            print(f"PID: {proc['pid']}")
            print(f"Path: {proc['exe_path']}")
            print(f"User: {proc['username']}")
            print(f"CPU Usage: {proc['cpu_percent']}%")
            print(f"Memory: {proc['memory_mb']} MB")
            print(f"Trusted: {identifier.is_trusted_application(proc['name'])}")
            
            # Test 3: Get network activity
            print("\nNetwork Connections:")
            network = identifier.get_network_activity(proc['pid'])
            if network:
                for conn in network[:5]:  # Show first 5 connections
                    print(f"  {conn['local_address']} -> {conn['remote_address']} [{conn['status']}]")
            else:
                print("  No active network connections")
            
            print("=" * 60)
            print()
    else:
        print("✗ No camera processes detected")
        print("Make sure a camera app (Zoom, Teams, etc.) is running")
        print()
    
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()