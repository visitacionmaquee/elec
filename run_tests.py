# run_tests.py
#!/usr/bin/env python3
"""
Test runner for the image processing project
"""
import subprocess
import sys

def run_pytest():
    """Run pytest with coverage"""
    print("Running tests with coverage...")
    result = subprocess.run([
        "pytest", 
        "test/", 
        "-v", 
        "--cov=src", 
        "--cov-report=html", 
        "--cov-report=term-missing"
    ])
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_pytest())