# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
import importlib
import inspect
import json
import logging

import pytest

from cases import TESTS, tensor_shapes, ttk_attributes
from device_support import assert_outputs, invoke, make_inputs

pytestmark = pytest.mark.npu


def _run_golden(runtime, request, case, case_dir, graph=False):
    _, _, golden = runtime
    case = dict(case, device_id=request.config.getoption("--device-id"))
    case = golden._normalize_case(case)
    handler = logging.FileHandler(case_dir / "run.log", encoding="utf-8")
    old_level = golden.logger.level
    golden.logger.setLevel(logging.INFO)
    golden.logger.addHandler(handler)
    try:
        result = golden.run_one_case(case["name"], case, {"gen", "cpu", "npu", "compare"},
                                     case["name"], str(case_dir), graph=graph, use_quant_matmul=False)
        (case_dir / "result.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        assert result["status"] == "Pass", result
    finally:
        golden.logger.removeHandler(handler)
        golden.logger.setLevel(old_level)
        handler.close()


@pytest.mark.accuracy
def test_accuracy(runtime, request, stc_case, case_dir, monkeypatch):
    monkeypatch.delenv("QBSA_CHECK_PADDED_LAYOUTS", raising=False)
    _run_golden(runtime, request, stc_case, case_dir)


@pytest.mark.equivalence
def test_same_input_three_layouts(runtime, request, stc_case, case_dir, monkeypatch):
    golden = runtime[2]
    monkeypatch.setenv("QBSA_CHECK_PADDED_LAYOUTS", "1")
    calls = []
    original = golden._check_padded_layouts

    def check(*args, **kwargs):
        prefix = args[3].cpu().tolist()
        lengths = [b - a for a, b in zip(prefix, prefix[1:])]
        assert lengths and min(lengths) > 0 and len(set(lengths)) == 1
        result = original(*args, **kwargs)
        calls.append(True)
        return result

    monkeypatch.setattr(golden, "_check_padded_layouts", check)
    _run_golden(runtime, request, stc_case, case_dir)
    assert calls == [True], "Same-input layout check was not executed"


@pytest.mark.graph
def test_graph_accuracy(runtime, request, stc_case, case_dir, monkeypatch):
    monkeypatch.delenv("QBSA_CHECK_PADDED_LAYOUTS", raising=False)
    _run_golden(runtime, request, stc_case, case_dir, graph=True)


@pytest.mark.metadata
@pytest.mark.parametrize("method", [
    "test_npu_metadata_optional_inputs", "test_metadata_meta_contract",
    "test_main_meta_shapes_and_scale_contract", "test_npu_metadata_and_excess_sparse_rows",
])
def test_existing_contract_regressions(runtime, method):
    cls = importlib.import_module("test_layout_metadata").LayoutMetadataTest
    getattr(cls(method), method)()  # Real production Meta/NPU calls, no mock dispatcher.


@pytest.mark.graph
@pytest.mark.metadata
def test_metadata_ge(runtime):
    cls = importlib.import_module("test_layout_metadata").LayoutMetadataTest
    cls("test_ge_metadata_matches_eager").test_ge_metadata_matches_eager()


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("block", [64, 128])
@pytest.mark.parametrize("with_lse", [False, True])
@pytest.mark.parametrize("pattern,extra,extra_count", [
    ("all", 0, 0), ("first", 0, 0), ("middle", 0, 0), ("last", 0, 0),
    ("mixed", 0, 0), ("mixed", 5, 0), ("mixed", 5, 2),
])
def test_sparse_rows(runtime, layout, block, with_lse, pattern, extra, extra_count):
    torch = runtime[0]
    baseline, _ = make_inputs(torch, "TND", block, pattern=pattern)
    padded, counts = make_inputs(torch, layout, block, pattern=pattern, extra=extra, extra_count=extra_count)
    baseline["return_softmax_lse"] = padded["return_softmax_lse"] = with_lse
    ref, ref_lse = assert_outputs(torch, baseline, invoke(torch, baseline))
    # Repetition catches uninitialized empty-row output and stale allocator content.
    for _ in range(2):
        out, lse = assert_outputs(torch, padded, invoke(torch, padded))
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        if with_lse:
            torch.testing.assert_close(lse, ref_lse, rtol=0, atol=0)
        for batch in range(2):
            for head in range(4):
                for row in range(counts.shape[2]):
                    if int(counts[batch, head, row]) == 0:
                        start, end = row * block, min((row + 1) * block, 257)
                        assert bool((out[batch, start:end, head] == 0).all())
                        if with_lse:
                            assert bool(torch.isneginf(lse[batch, head, start:end]).all())


@pytest.mark.graph
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_graph_replay_a_b_a(runtime, layout):
    torch, npu, _ = runtime
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    a, _ = make_inputs(torch, layout, pattern="dense", seed=12177)
    b, _ = make_inputs(torch, layout, pattern="all", seed=12178)

    class Model(torch.nn.Module):
        def forward(self, **inputs):
            return torch.ops.custom.npu_quant_block_sparse_attn(**inputs)

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
    backend = torchair.get_npu_backend(compiler_config=config)
    compilations = []

    def counting_backend(graph, example_inputs, **kwargs):
        compilations.append(True)
        return backend(graph, example_inputs, **kwargs)

    compiled = torch.compile(Model(), backend=counting_backend, fullgraph=True, dynamic=False)
    # Compile ONCE, alternate tensors/metadata with identical shape contracts.
    for inputs in (a, b, a):
        expected = invoke(torch, inputs)
        actual = compiled(**inputs)
        npu.npu.synchronize()
        actual = tuple(t.cpu().clone() for t in actual)
        assert_outputs(torch, inputs, actual)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        if inputs is b:
            assert bool((actual[0] == 0).all())
            assert bool(torch.isneginf(actual[1]).all())
    assert len(compilations) == 1, f"Expected one graph compilation, saw {len(compilations)}"


@pytest.mark.ttk
def test_ttk_adapter(runtime, stc_case, monkeypatch):
    """Invoke the REAL spec input/golden/API/compare hooks, before the TTK CLI run."""
    torch, npu, _ = runtime
    monkeypatch.syspath_prepend(str(TESTS / "assets"))
    ops = importlib.import_module("bsa_ttk_ops")
    # Use a distinct module name: a global 'spec' import may belong to another operator.
    spec = importlib.util.spec_from_file_location("qbsa_layout_stc_spec", TESTS / "assets" / "spec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hooks = module.QuantBlockSparseAttnSpec
    attrs = ttk_attributes(stc_case)
    ptype = torch.float32 if stc_case["p_scale_type"] == "float32" else torch.float8_e8m0fnu
    dtypes = [torch.float8_e4m3fn] * 3 + [torch.float8_e8m0fnu] * 3 + [torch.int32] * 2 + [ptype] + [torch.int32] * 5 + [torch.uint8, torch.int32]
    tensors = [torch.zeros(shape, dtype=torch.uint8).view(dtype) if "float8" in str(dtype)
               else torch.zeros(shape, dtype=dtype) for shape, dtype in zip(tensor_shapes(stc_case), dtypes)]
    extra = dict(testcase_name=stc_case["name"], input_ranges=[stc_case[k] for k in ("data_range_q", "data_range_k", "data_range_v")])
    hooks.customize_inputs(*tensors, **attrs, **extra)
    golden = hooks.golden(*tensors, **attrs, **extra)
    api_attrs = {k: v for k, v in attrs.items() if k in inspect.signature(ops.quant_block_sparse_attn).parameters}
    actual = ops.quant_block_sparse_attn(*(t.npu() for t in tensors), **api_attrs)
    npu.npu.synchronize()
    actual = tuple(t.cpu() for t in actual)
    assert tuple(actual[0].shape) == tensor_shapes(stc_case)[0]
    assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == torch.float32
    if stc_case["layout_q"] != "TND":
        expected_lse = (stc_case["B"], stc_case["N1"], stc_case["S1_max"]) if stc_case["return_softmax_lse"] else (0,)
        assert tuple(actual[1].shape) == expected_lse
    results = hooks.compare(*actual, *golden, return_softmax_lse=stc_case["return_softmax_lse"])
    results = results if isinstance(results, list) else [results]
    assert results and all(r["pass"] for r in results), results
