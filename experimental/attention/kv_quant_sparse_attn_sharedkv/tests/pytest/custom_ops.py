#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Shim to make custom_ops importable by patching torchair version incompatibility."""

import sys
import os

# Ensure NPU libraries on LD_LIBRARY_PATH
for _p in [
    '/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib',
    '/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib',
    '/workspace/vendors/custom_transformer/op_api/lib',
]:
    if _p not in os.environ.get('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = _p + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# Patch torchair before any converter module loads
import torch_npu  # needed to make torchair importable
import torchair.ge._ge_graph as _gg
from torchair.ge._ge_graph import DataType

_torch_dtype_to_ge = {
    0: DataType.DT_FLOAT,
    1: DataType.DT_FLOAT16,
    2: DataType.DT_INT8,
    3: DataType.DT_INT32,
    4: DataType.DT_INT64,
    27: DataType.DT_BF16,
}

def _torch_dtype_value_to_ge_type(v):
    return _torch_dtype_to_ge.get(v, DataType.DT_FLOAT16)

def _torch_dtype_value_to_ge_proto_type(v):
    return _torch_dtype_to_ge.get(v, DataType.DT_FLOAT16)

_gg.torch_dtype_value_to_ge_type = _torch_dtype_value_to_ge_type
_gg.torch_dtype_value_to_ge_proto_type = _torch_dtype_value_to_ge_proto_type

# Load the real custom_ops
_real_path = '/workspace/cann-recipes-infer/.claude/worktrees/fix-ops-common/ops/ascendc/torch_ops_extension/build/lib.linux-x86_64-cpython-311'
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Remove this dir from path, add real path, remove self from modules, import real
_old_path = list(sys.path)
sys.path = [p for p in sys.path if os.path.abspath(p) != _this_dir]
sys.path.insert(0, _real_path)

if 'custom_ops' in sys.modules:
    del sys.modules['custom_ops']

import custom_ops as _real_custom_ops

# Restore path
sys.path = _old_path

# Copy everything from the real module into this module's namespace
# and register this module as custom_ops in sys.modules
_this_mod = sys.modules[__name__]
for _attr in dir(_real_custom_ops):
    if not _attr.startswith('__') or _attr in ('__all__', '__doc__', '__file__', '__path__', '__spec__'):
        setattr(_this_mod, _attr, getattr(_real_custom_ops, _attr))
