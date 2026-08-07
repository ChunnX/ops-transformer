from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch_npu


def load_k_pool_public_api():
    """Load only the KPool wheel modules, without importing unrelated operators."""
    loaded = sys.modules.get("cann_ops_transformer.ops.k_pool")
    if loaded is not None:
        return loaded

    package_spec = importlib.util.find_spec("cann_ops_transformer")
    if package_spec is None or package_spec.origin is None:
        raise ImportError("cann_ops_transformer is not installed")

    package_root = Path(package_spec.origin).resolve().parent
    package = types.ModuleType("cann_ops_transformer")
    package.__file__ = str(package_root / "__init__.py")
    package.__package__ = "cann_ops_transformer"
    package.__path__ = [str(package_root)]
    sys.modules["cann_ops_transformer"] = package

    ops = types.ModuleType("cann_ops_transformer.ops")
    ops.__file__ = str(package_root / "ops" / "__init__.py")
    ops.__package__ = "cann_ops_transformer.ops"
    ops.__path__ = [str(package_root / "ops")]
    sys.modules["cann_ops_transformer.ops"] = ops
    package.ops = ops

    module = importlib.import_module("cann_ops_transformer.ops.k_pool")
    ops.k_pool = module.k_pool
    ops.k_pool_npu = module.k_pool_npu
    torch_npu.k_pool_npu = module.k_pool_npu
    return module
