"""Record the installed or explicitly configured QSLI package sources."""

import hashlib
import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def report_indexer_sources():
    expected_ops = os.environ.get("CANNBOTDSL_TEST_OPS_DIR")
    if not expected_ops:
        return
    operator = Path(__file__).resolve().parent.parent.name.removesuffix("_dsl")
    interface = importlib.import_module(
        "cann_ops_transformer.ops.attention." + operator + "_dsl." + operator
    )
    print("TORCH_INTERFACE", interface.__file__, flush=True)
    for name in (operator + "_dsl", operator + "_metadata_dsl"):
        module = importlib.import_module("ops." + name)
        source = Path(module.__file__).resolve()
        assert source.parent == Path(expected_ops).resolve(), str(source)
        print("DSL_SOURCE", str(source), hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
