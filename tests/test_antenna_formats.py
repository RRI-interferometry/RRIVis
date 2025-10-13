#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced antenna.py functionality
with different antenna file formats.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.antenna import read_antenna_positions

def test_format(format_name, file_path, expected_count=None):
    """Test a specific antenna format."""
    print(f"\n{'='*60}")
    print(f"Testing {format_name.upper()} Format")
    print(f"File: {file_path}")
    print('='*60)

    try:
        antennas = read_antenna_positions(file_path, format_name)
        print(f"PASS: Successfully loaded {len(antennas)} antennas")

        if expected_count and len(antennas) != expected_count:
            print(f"WARNING: Expected {expected_count} antennas, got {len(antennas)}")

        # Print first antenna details
        if antennas:
            first_ant = next(iter(antennas.values()))
            print(f"First antenna: {first_ant}")

        return True

    except Exception as e:
        print(f"FAIL: Error loading {format_name} format: {e}")
        return False

def main():
    """Test all supported antenna formats."""
    print("RRIvis Enhanced Antenna Format Support Test")
    print("=" * 60)

    examples_dir = Path(__file__).parent / "examples"

    # Test cases: (format_name, file_path, expected_count)
    test_cases = [
        ("rrivis", examples_dir / "antenna_rrivis.txt", 5),
        ("casa", examples_dir / "antenna_casa.cfg", 5),
        ("mwa", examples_dir / "antenna_mwa.txt", 5),
        ("pyuvdata", examples_dir / "antenna_pyuvdata.txt", 5),
    ]

    results = []
    for format_name, file_path, expected_count in test_cases:
        if file_path.exists():
            success = test_format(format_name, str(file_path), expected_count)
            results.append((format_name, success))
        else:
            print(f"\nWARNING: Example file not found: {file_path}")
            results.append((format_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for format_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{format_name:12} : {status}")

    print(f"\nOverall: {passed}/{total} formats working correctly")

    if passed == total:
        print("SUCCESS: All antenna formats are working!")
        return 0
    else:
        print("WARNING: Some formats failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())