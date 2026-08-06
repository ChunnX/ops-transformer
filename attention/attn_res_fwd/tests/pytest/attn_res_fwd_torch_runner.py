# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""AttnResFwd Torch 运行器：隔离加载单算子 OpBuilder，避免 import 全量 cann_ops_transformer。"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Callable

TORCH_EXT_ROOT = Path(__file__).resolve().parents[4] / "torch_extension"
PKG_ROOT = TORCH_EXT_ROOT / "cann_ops_transformer"


def _ensure_pkg_stubs() -> None:
    """注册最小包桩，使 attn_res_fwd.py / builder.py 可加载而不触发 ops/__init__.py。"""
    if str(TORCH_EXT_ROOT) not in sys.path:
        sys.path.insert(0, str(TORCH_EXT_ROOT))

    if "cann_ops_transformer" not in sys.modules:
        pkg = types.ModuleType("cann_ops_transformer")
        pkg.__path__ = [str(PKG_ROOT)]
        pkg.__file__ = str(PKG_ROOT / "__init__.py")
        sys.modules["cann_ops_transformer"] = pkg

    if "cann_ops_transformer.op_builder" not in sys.modules:
        sub = types.ModuleType("cann_ops_transformer.op_builder")
        sub.__path__ = [str(PKG_ROOT / "op_builder")]
        sys.modules["cann_ops_transformer.op_builder"] = sub

    if "cann_ops_transformer.ops" not in sys.modules:
        sub = types.ModuleType("cann_ops_transformer.ops")
        sub.__path__ = [str(PKG_ROOT / "ops")]
        sys.modules["cann_ops_transformer.ops"] = sub


def _load_attn_res_fwd_module():
    _ensure_pkg_stubs()
    mod_path = PKG_ROOT / "ops" / "attn_res_fwd.py"
    spec = importlib.util.spec_from_file_location("cann_ops_transformer.ops.attn_res_fwd", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load attn_res_fwd module from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cann_ops_transformer.ops.attn_res_fwd"] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_attn_res_fwd_fn() -> Callable:
    """
    返回 attn_res_fwd(...)，走仓内 OpBuilder.load() JIT 编译路径。
    需已 source cann_env.sh（ASCEND_HOME_PATH + custom OPP）。
    """
    if not os.environ.get("ASCEND_HOME_PATH"):
        raise RuntimeError("ASCEND_HOME_PATH not set; source cann_env.sh first")

    module = _load_attn_res_fwd_module()
    return module.attn_res_fwd
