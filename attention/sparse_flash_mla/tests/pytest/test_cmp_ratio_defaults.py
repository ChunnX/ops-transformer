"""Check the A2/A3 SWA-facing Torch defaults without an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest


@pytest.fixture
def torch_dispatchers():
    source = Path(__file__).resolve().parents[2] / "torch_extension/sparse_flash_mla.py"
    tree = ast.parse(source.read_text())
    names = {"sparse_flash_mla", "sparse_flash_mla_metadata"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    op_module = Mock()
    namespace = {
        "Optional": Optional,
        "torch": SimpleNamespace(Tensor=object),
        "sparse_flash_mla_op_builder": SimpleNamespace(load=lambda: op_module),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, op_module


@pytest.mark.parametrize("has_cmp_kv", [False, True])
def test_omitted_ratio_keeps_torch_default_one(torch_dispatchers, has_cmp_kv):
    namespace, op_module = torch_dispatchers
    namespace["sparse_flash_mla"](object(), cmp_kv=object() if has_cmp_kv else None)
    assert op_module.sparse_flash_mla.call_args.args[19] == 1

    namespace["sparse_flash_mla_metadata"](1, 1, 512, has_cmp_kv=has_cmp_kv)
    assert op_module.sparse_flash_mla_metadata.call_args.args[18] == 1


def test_a2a3_swa_ratio_zero_must_be_explicit(torch_dispatchers):
    namespace, op_module = torch_dispatchers
    namespace["sparse_flash_mla"](object(), cmp_ratio=0)
    assert op_module.sparse_flash_mla.call_args.args[19] == 0

    namespace["sparse_flash_mla_metadata"](1, 1, 512, has_cmp_kv=False, cmp_ratio=0)
    assert op_module.sparse_flash_mla_metadata.call_args.args[18] == 0
