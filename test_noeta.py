#!/usr/bin/env python
"""
Quick test script to verify Noeta is working
"""
import sys
import os

# Add noeta to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from noeta_runner import execute_noeta

# Simple test code
test_code = """
# Quick functionality test
load "data/sales_data.csv" as sales
select sales {category, price} as test_data
filter test_data [price > 50] as filtered
describe filtered
"""

print("Testing Noeta DSL Implementation...")
print("=" * 60)

try:
    execute_noeta(test_code, verbose=True)
    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("Noeta DSL is ready for demonstration.")
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    print("Please check the error message above.")
