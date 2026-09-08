# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import sys
from types import SimpleNamespace

import pytest

from test_all_to_allv_export import _load_ops
from validation.tiling_space import TILING_FIELDS, default_tiling


@pytest.mark.parametrize("operator,export,reuse,schema", [
    ("all_to_allv_grouped_mat_mul", False, False, "all_to_allv_grouped_mat_mul_udma"),
    ("all_to_allv_grouped_mat_mul", False, True, "all_to_allv_grouped_mat_mul_udma_out"),
    ("all_to_allv_grouped_mat_mul", True, False, "all_to_allv_grouped_mat_mul_udma_export"),
    ("all_to_allv_grouped_mat_mul", True, True, "all_to_allv_grouped_mat_mul_udma_out_export"),
    ("grouped_mat_mul_all_to_allv", False, False, "grouped_mat_mul_all_to_allv_udma"),
    ("grouped_mat_mul_all_to_allv", False, True, "grouped_mat_mul_all_to_allv_udma_out"),
])
def test_explicit_tiling_reaches_every_native_entry_in_stable_wire_order(monkeypatch, operator, export, reuse, schema):
    calls = []
    invoke, tensor = _load_ops(monkeypatch, calls, operator_name=operator)
    sentinel = object()

    class Recorder:
        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls.append((name, args, kwargs))
                return sentinel
            return record

    sys.modules["torch"].ops.ops_transformer_catccos_udma = Recorder()
    device = SimpleNamespace(type="npu")
    x = tensor(shape=(4, 256), device=device)
    weight = tensor(shape=(2, 256, 512), device=device)
    route = tensor([1, 1, 1, 1], device=device)
    workload = dict(operator=operator, output_all_to_allv=export, world_size=2, expert_num=4,
                    shape={"m": 4, "k": 256, "n": 512})
    profile = default_tiling(workload)
    options = dict(ep_size=2, expert_num=4, tiling=dict(reversed(list(profile.items()))))
    if reuse:
        options["out"] = tensor(shape=(4, 512), device=device)
    if operator == "all_to_allv_grouped_mat_mul":
        options.update(received_rows=4, output_all_to_allv=export)
        if export:
            options["all_to_allv_out"] = tensor(shape=(4, 256), device=device)
    assert invoke(x, weight, route, route, **options) is sentinel
    assert len(calls) == 1
    name, positional, keywords = calls[0]
    assert name == schema
    assert keywords == {"tiling": [profile[key] for key in TILING_FIELDS[operator]]}
    assert positional[0] is x and positional[1] is weight
    if reuse:
        assert any(argument is options["out"] for argument in positional)


def test_malformed_tiling_is_rejected_before_any_native_launch(monkeypatch):
    calls = []
    invoke, tensor = _load_ops(monkeypatch, calls)
    device = SimpleNamespace(type="npu")
    with pytest.raises(ValueError, match="tiling fields"):
        invoke(tensor(shape=(4, 256), device=device), tensor(shape=(2, 256, 512), device=device),
               tensor([1, 1, 1, 1], device=device), tensor([1, 1, 1, 1], device=device),
               ep_size=2, expert_num=4, tiling={"comm_interval": 8})
    assert calls == []
