# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import importlib.util
import sys
from pathlib import Path

import pytest


RUNTIME_FILE = Path(__file__).parents[1] / "ops_transformer_catccos_udma" / "runtime.py"
COMMON_FILE = Path(__file__).parents[1] / "validation" / "common.py"
SPEC = importlib.util.spec_from_file_location("catccos_udma_runtime", RUNTIME_FILE)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
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
