# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for PoolKeyIndexer tests.
"""
import pytest
import torch
import torch_npu


@pytest.fixture(scope="session", autouse=True)
def npu_init():
    """Initialize NPU device once per test session."""
    torch_npu.npu.set_device(0)
    yield
