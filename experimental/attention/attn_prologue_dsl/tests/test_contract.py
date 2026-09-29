# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU 上验证真实 dispatcher/schema；只替换 NPU 环境探测，不模拟计算。"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch


HERE = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def extension():
    # 使用仓库的真实 OpBuilder；隔离包加载，避免注册所有无关算子。
    with pytest.MonkeyPatch.context() as patch:
        npu = types.ModuleType("torch_npu")
        npu.__file__ = str(HERE / "torch_npu_stub.py")
        patch.setitem(sys.modules, "torch_npu", npu)
        builder_path = (
            HERE.parents[3]
            / "torch_extension/cann_ops_transformer/op_builder/builder.py"
        )
        builder = _load_module("_attn_prologue_test_builder", builder_path)
        patch.setitem(sys.modules, "cann_ops_transformer.op_builder", builder)
        for name in ("_attn_test", "_attn_test.ops", "_attn_test.ops.attention"):
            package = types.ModuleType(name)
            package.__path__ = []
            patch.setitem(sys.modules, name, package)
        # 模拟已由其他 ds41 算子创建的共享命名空间。
        shared = types.ModuleType("_attn_test.ops.ds41")
        shared.existing_operator = object()
        patch.setitem(sys.modules, shared.__name__, shared)
        name = "_attn_test.ops.attention.attn_prologue_dsl"
        module = _load_module(name, HERE.parent / "torch_extension/__init__.py")
        yield module
        for key in list(sys.modules):
            if key.startswith(name) or key == "_attn_prologue_test_builder":
                del sys.modules[key]


def meta_inputs(tokens=72):
    shapes = {
        "x": (tokens, 5120),
        "wqa": (1280, 5120),
        "wqb": (32768, 1280),
        "wkv": (512, 5120),
        "descale_x": (tokens, 80, 2),
        "descale_wqa": (1280, 80, 2),
        "descale_wqb": (32768, 20, 2),
        "descale_wkv": (512, 80, 2),
        "norm_weight_qr": (1280,),
        "norm_weight_kv": (512,),
        "rope_sin": (tokens, 64),
        "rope_cos": (tokens, 64),
        "cache_index": (tokens,),
        "kv_cache": (max(2, (tokens + 127) // 128), 128, 1, 512),
    }
    values = {}
    for name, shape in shapes.items():
        dtype = torch.float32
        if name in ("x", "wqa", "wqb", "wkv"):
            dtype = torch.float8_e4m3fn
        elif name.startswith("descale_"):
            dtype = torch.float8_e8m0fnu
        elif name == "cache_index":
            dtype = torch.int64
        elif name == "kv_cache":
            dtype = torch.uint8
        values[name] = torch.empty(shape, dtype=dtype, device="meta")
    return values


@pytest.mark.parametrize("tokens", [1, 71, 72, 129, 257, 4097, 262144])
def test_meta_outputs(extension, tokens):
    outputs = extension.attn_prologue(**meta_inputs(tokens))
    assert [tuple(t.shape) for t in outputs] == [
        (tokens, 64, 512),
        (tokens, 1280),
        (tokens, 20, 2),
    ]
    assert [t.dtype for t in outputs] == [
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.int8,
    ]
    assert all(t.device.type == "meta" and t.is_contiguous() for t in outputs)


def test_fake_tensor(extension):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        outputs = extension.attn_prologue(**meta_inputs(73))
        assert outputs[0].shape == (73, 64, 512)


def test_schema_marks_only_cache_mutable(extension):
    schema = extension.attn_prologue._schema
    mutable = [
        arg.name
        for arg in schema.arguments
        if arg.alias_info and arg.alias_info.is_write
    ]
    assert mutable == ["kv_cache"]
    assert all(result.alias_info is None for result in schema.returns)
    eps = schema.arguments[-1]
    assert eps.name == "norm_eps" and eps.kwarg_only and eps.default_value == 1e-6
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        str(schema.name) + "." + schema.overload_name, "PrivateUse1"
    )


def test_namespace_and_lazy_kernel_import(extension):
    assert extension.attn_prologue is extension.attn_prologue_dsl
    assert extension.ds41 is sys.modules["_attn_test.ops.ds41"]
    assert hasattr(extension.ds41, "existing_operator")
    assert extension.ds41.attn_prologue is extension.attn_prologue
    assert sys.modules["_attn_test.ops"].attn_prologue is extension.attn_prologue
    assert extension.__name__ + "._kernel" not in sys.modules


def test_npu_adapter_forwards_all_arguments(extension, monkeypatch):
    kernel = types.ModuleType(extension.__name__ + "._kernel")
    expected = (object(), object(), object())
    kernel.attn_prologue = Mock(return_value=expected)
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    values = meta_inputs()
    result = extension._attn_prologue_impl(**values, norm_eps=1e-5)
    assert result is expected
    kernel.attn_prologue.assert_called_once_with(*values.values(), norm_eps=1e-5)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_eps(extension, eps):
    with pytest.raises(ValueError, match="finite and positive"):
        extension.attn_prologue(**meta_inputs(), norm_eps=eps)


@pytest.mark.parametrize(
    "name,shape", [("wqa", (1280, 64)), ("wqb", (32767, 1280)), ("wkv", (0, 5120))]
)
def test_invalid_projection_geometry(extension, name, shape):
    values = meta_inputs()
    values[name] = torch.empty(shape, dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(RuntimeError):
        extension.attn_prologue(**values)


def test_payload_dtype_is_strict(extension):
    values = meta_inputs()
    values["x"] = values["x"].to(torch.uint8)
    with pytest.raises(RuntimeError, match="E4M3FN"):
        extension.attn_prologue(**values)
