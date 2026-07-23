#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Pytest hooks for SMLA NPU tests."""

import os
import sys

import pytest
import torch_npu


@pytest.fixture(scope="session", autouse=True)
def preload_sparse_flash_mla_extension():
    """JIT-compile sparse_flash_mla once before any case runs."""
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../../.."))
    os.environ.setdefault("SMLA_OPS_TRANSFORMER_ROOT", repo_root)
    torch_ext = os.path.join(repo_root, "torch_extension")
    if torch_ext not in sys.path:
        sys.path.insert(0, torch_ext)
    # ASCEND_RT_VISIBLE_DEVICES remaps physical cards; always use logical device 0.
    torch_npu.npu.set_device(0)
    import importlib.util
    import types
    if "cann_ops_transformer.ops.sparse_flash_mla" not in sys.modules:
        pkg = types.ModuleType("cann_ops_transformer")
        pkg.__path__ = [os.path.join(torch_ext, "cann_ops_transformer")]
        sys.modules.setdefault("cann_ops_transformer", pkg)
        op_builder_pkg = types.ModuleType("cann_ops_transformer.op_builder")
        op_builder_pkg.__path__ = [os.path.join(torch_ext, "cann_ops_transformer", "op_builder")]
        sys.modules.setdefault("cann_ops_transformer.op_builder", op_builder_pkg)
        mod_path = os.path.join(torch_ext, "cann_ops_transformer", "ops", "sparse_flash_mla.py")
        spec = importlib.util.spec_from_file_location(
            "cann_ops_transformer.ops.sparse_flash_mla", mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules["cann_ops_transformer.ops.sparse_flash_mla"]
    mod.sparse_flash_mla_op_builder.load()
