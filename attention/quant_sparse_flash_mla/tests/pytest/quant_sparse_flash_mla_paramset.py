#!/usr/bin/python
"""QSMLA parameters generated from the shared FD matrix."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.fd_performance_cases import build_test_params


TEST_PARAMS = build_test_params("qsmla")
ENABLED_PARAMS = TEST_PARAMS.values()
