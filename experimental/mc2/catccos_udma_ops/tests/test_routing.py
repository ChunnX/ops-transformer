# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import importlib.util
from pathlib import Path


ROUTING_FILE = Path(__file__).parents[1] / "ops_transformer_catccos_udma" / "routing.py"
SPEC = importlib.util.spec_from_file_location("catccos_udma_routing", ROUTING_FILE)
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)


def test_balanced_routes_preserve_each_source_and_destination():
    for m in (1, 7, 64, 257):
        for ep_size, expert_num in ((1, 4), (2, 8), (4, 16), (8, 64)):
            matrix = routing.build_global_route_matrix(m, ep_size, expert_num)
            assert [sum(row) for row in matrix] == [m] * ep_size
            local_experts = expert_num // ep_size
            for dst in range(ep_size):
                received = sum(
                    matrix[src][dst * local_experts + expert]
                    for src in range(ep_size)
                    for expert in range(local_experts)
                )
                assert received == m


def test_rank_tables_match_global_matrix_projection():
    matrix = routing.build_global_route_matrix(31, 4, 16)
    for rank in range(4):
        local, global_for_local = routing.build_rank_route_tables(31, rank, 4, 16)
        assert local == matrix[rank]
        assert global_for_local == [
            matrix[src][rank * 4 + expert] for src in range(4) for expert in range(4)
        ]


def test_rank_route_tensors_use_int64(monkeypatch):
    calls = []

    class FakeTensor:
        pass

    class FakeTorch:
        int64 = "int64"

        @staticmethod
        def tensor(values, *, dtype, device):
            calls.append((values, dtype, device))
            return FakeTensor()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    local, global_for_local = routing.build_rank_route_tensors(16, 0, 2, 4, "npu:0")
    assert len(calls) == 2
    assert all(dtype == "int64" for _, dtype, _ in calls)
    assert isinstance(local, FakeTensor)
    assert isinstance(global_for_local, FakeTensor)
    assert len(calls[0][0]) == 4
    assert len(calls[1][0]) == 4


def test_rank_route_returns_host_known_received_rows(monkeypatch):
    calls = []

    class FakeTorch:
        int64 = "int64"

        @staticmethod
        def tensor(values, *, dtype, device):
            calls.append((values, dtype, device))
            return object()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    _, _, received_rows = routing.build_rank_route(31, 2, 4, 16, "npu:2")
    assert received_rows == 31
    assert len(calls) == 2
