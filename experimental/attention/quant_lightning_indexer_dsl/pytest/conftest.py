"""Shared pytest path and lifecycle for QLI precision cases."""

import gc
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_ROOT = REPO_ROOT / "samples"
for path in (REPO_ROOT, SAMPLES_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


@pytest.fixture(autouse=True)
def release_case_memory():
    yield
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
