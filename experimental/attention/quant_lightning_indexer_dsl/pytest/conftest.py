"""Checkout loading, source verification and case cleanup."""

import gc
from pathlib import Path
import sys

import pytest
import torch

TEST_ROOT = Path(__file__).resolve().parent
for directory in (TEST_ROOT, TEST_ROOT.parents[1]):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from indexer_test_runtime import configure, report_sources

configure()


@pytest.fixture(scope="session", autouse=True)
def report_indexer_sources():
    report_sources("qli" if TEST_ROOT.parent.name == "quant_lightning_indexer_dsl" else "qsli")


@pytest.fixture(autouse=True)
def release_case_memory():
    yield
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_initialized():
        torch.npu.empty_cache()
