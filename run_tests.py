#!/usr/bin/env python
"""
Convenience script to run tests for the Multimodal RAG system.
"""
import argparse
import os
import subprocess
import sys

def main():
    """Run the specified tests."""
    parser = argparse.ArgumentParser(description="Run tests for the Multimodal RAG system.")
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--crawler', action='store_true', help='Run crawler tests only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up pytest command
    pytest_cmd = ["pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    # Add markers based on arguments
    if args.unit:
        pytest_cmd.append("-m unit")
    elif args.integration:
        pytest_cmd.append("-m integration")
    elif args.crawler:
        pytest_cmd.append("-m crawler")
    elif not args.all:
        # Default to unit tests if no flag is specified
        pytest_cmd.append("-m unit")
    
    # Run the tests
    print(f"Running command: {' '.join(pytest_cmd)}")
    result = subprocess.run(pytest_cmd, shell=True if os.name == 'nt' else False)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())