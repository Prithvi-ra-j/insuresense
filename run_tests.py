#!/usr/bin/env python3
"""
Test Runner for InsureSense 360
Runs all tests in the correct order
"""

import subprocess
import sys
import os
from pathlib import Path

# Set environment variables
os.environ["USER_AGENT"] = "InsureSense360-TestRunner/1.0"
os.environ["PYTHONIOENCODING"] = "utf-8"

def run_test(test_file):
    """Run a single test file"""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    try:
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["USER_AGENT"] = "InsureSense360-TestRunner/1.0"
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run([
            sys.executable, 
            f"tests/{test_file}"
        ], capture_output=True, text=True, cwd=Path.cwd(), env=env)
        
        if result.returncode == 0:
            print(f"PASSED: {test_file}")
            print(result.stdout)
            return True
        else:
            print(f"FAILED: {test_file}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR: {test_file} - {e}")
        return False

def main():
    """Run all tests in order"""
    print("InsureSense 360 - Test Suite Runner")
    print("=" * 60)
    print("Running all tests in order...")
    
    # Test files in order
    test_files = [
        "test_system.py",
        "test_llm.py", 
        "test_direct_llm.py",
        "test_together_llm.py",
        "test_api.py",
        "test_api_integration.py"
    ]
    
    results = []
    
    for test_file in test_files:
        success = run_test(test_file)
        results.append((test_file, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_file, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{status}: {test_file}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready for deployment.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
