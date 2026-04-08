#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared CLI formatting helpers and test-case loader.

print_header(): the section banner used by every analysis mode.
params_from_testcase(): loads named test parameters from
attention/mla_prolog_v3/tests/pytest/testcases.py.
"""

import os
import sys

from ..model.operator_params import OperatorParams

# Make sibling pytest dir importable for testcases.py
_PYTEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "pytest",
)
if _PYTEST_DIR not in sys.path:
    sys.path.insert(0, _PYTEST_DIR)


def params_from_testcase(name: str) -> OperatorParams:
    """Load OperatorParams from a named test case in testcases.py."""
    try:
        from testcases import TEST_PARAMS
    except ImportError:
        print(f"Error: cannot import testcases.py from ../pytest/", file=sys.stderr)
        sys.exit(1)

    if name not in TEST_PARAMS:
        print(f"Error: unknown test case '{name}'. Available: {list(TEST_PARAMS.keys())}", file=sys.stderr)
        sys.exit(1)

    tc = TEST_PARAMS[name]
    return OperatorParams(
        batch_size=tc["batch_size"][0],
        seq_len=tc["seq_len"][0],
        head_num=tc["head_num"][0],
        He=tc["He"][0],
        weight_quant_mode=tc["weight_quant_mode"][0],
        kv_cache_quant_mode=tc["kv_cache_quant_mode"][0],
        query_quant_mode=tc["query_quant_mode"][0],
    )


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")
