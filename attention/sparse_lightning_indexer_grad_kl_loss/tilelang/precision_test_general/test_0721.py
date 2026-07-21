#!/usr/bin/env python3
"""Run the shared Case 0 harness against 0721.py."""

import importlib
import os
import runpy
import sys


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TILELANG_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, TILELANG_DIR)

kernel_0721 = importlib.import_module("0721")
sys.modules["0720"] = kernel_0721

print("Case 0 override: testing 0721.py")
runpy.run_path(os.path.join(TEST_DIR, "test_0720.py"), run_name="__main__")
