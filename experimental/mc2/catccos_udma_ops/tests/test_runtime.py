# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import importlib.util
import sys
import types
from unittest import mock
from pathlib import Path

import pytest


RUNTIME_FILE = Path(__file__).parents[1] / "ops_transformer_catccos_udma" / "runtime.py"
COMMON_FILE = Path(__file__).parents[1] / "validation" / "common.py"
SPEC = importlib.util.spec_from_file_location("catccos_udma_runtime", RUNTIME_FILE)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
# These tests inspect bootstrap and workspace metadata, and never execute a
# tensor operation. Keep them runnable on the CPU-only development host.
if importlib.util.find_spec("torch") is None:
    with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(Tensor=object, bfloat16=object())}):
        SPEC.loader.exec_module(runtime)
else:
    SPEC.loader.exec_module(runtime)
COMMON_SPEC = importlib.util.spec_from_file_location("catccos_udma_validation_common", COMMON_FILE)
common = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(common)


def test_udma_bootstrap_address_is_numeric_ipv4(monkeypatch):
    assert runtime._resolve_master_addr("127.0.0.1") == "127.0.0.1"
    monkeypatch.setattr(runtime.socket, "gethostbyname", lambda host: "10.20.30.40")
    assert runtime._resolve_master_addr("worker.example") == "10.20.30.40"


def test_udma_bootstrap_address_reports_resolution_failure(monkeypatch):
    def fail(_host):
        raise OSError("not found")

    monkeypatch.setattr(runtime.socket, "gethostbyname", fail)
    with pytest.raises(RuntimeError, match="UDMA_MASTER_ADDR"):
        runtime._resolve_master_addr("invalid.example")


def test_worker_bootstrap_method_includes_tcp_scheme(monkeypatch):
    monkeypatch.setenv("UDMA_MASTER_ADDR", "bootstrap.example")
    monkeypatch.setenv("UDMA_MASTER_PORT", "29610")
    monkeypatch.setattr(common.socket, "gethostbyname", lambda host: "10.20.30.40")
    assert common.udma_init_method_from_env() == "tcp://10.20.30.40:29610"


def test_validation_heap_uses_fixed_remote_workspace_size(monkeypatch):
    monkeypatch.delenv("UDMA_HEAP_SIZE", raising=False)
    assert common.validation_udma_heap_size() == 1 << 30


def test_explicit_heap_rejects_insufficient_capacity(monkeypatch):
    monkeypatch.setenv("UDMA_HEAP_SIZE", str(400 << 20))
    with pytest.raises(RuntimeError, match="too small"):
        common.validation_udma_heap_size()


def test_validation_heap_accepts_no_reuse_workspace_with_headroom(monkeypatch):
    monkeypatch.setenv("UDMA_HEAP_SIZE", str(768 << 20))
    assert common.validation_udma_heap_size() == 768 << 20


def test_runtime_reserves_no_reuse_workspace():
    runtime_source = (
        Path(__file__).parents[1]
        / "csrc"
        / "all_to_allv_grouped_mat_mul_udma"
        / "ascend950"
        / "udma_runtime.h"
    ).read_text(encoding="utf-8")
    layout_source = (
        Path(__file__).parents[1]
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "udma_workspace_layout.hpp"
    ).read_text(encoding="utf-8")

    assert '"catccos/dgemm/kernel/udma_workspace_layout.hpp"' in runtime_source
    assert "UdmaWorkspaceLayout::kDataBytes" in runtime_source
    assert "UdmaWorkspaceLayout::kTotalBytes" in runtime_source
    assert "kDataBytes = 640UL * 1024 * 1024" in layout_source
    assert "kControlBase = 700UL * 1024 * 1024" in layout_source
    assert "kControlBytes = 4096" in layout_source
