# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
"""Asset/coverage checks only; these tests never claim operator accuracy."""
import ast
from collections import Counter, defaultdict
import importlib.util

import pytest

from cases import LAYOUTS, TESTS, TestCases, ttk_attributes
from export_cases import artifacts, ttk_row

pytestmark = pytest.mark.offline


def test_matrix_has_matched_triplets_and_reference_coverage():
    assert len(TestCases) == 177
    assert len({c["name"] for c in TestCases}) == 177
    assert Counter(c["layout_q"] for c in TestCases) == dict.fromkeys(LAYOUTS, 59)
    assert Counter(c["test_tier"] for c in TestCases) == {"P0": 18, "P1": 159}
    grouped = defaultdict(list)
    for case in TestCases:
        grouped[case["layout_scenario"]].append(case)
    for group in grouped.values():
        assert {c["layout_q"] for c in group} == set(LAYOUTS)
        ignored = {"name", "layout_q", "layout_out", "cu_seqlens_q"}
        logical = [{k: v for k, v in c.items() if k not in ignored} for c in group]
        assert logical[0] == logical[1] == logical[2]
    assert {c["layout_scenario"] for c in TestCases if c["layout_scenario"].startswith("REF_")} == {
        "REF_EQ_AA", "REF_EQ_AU", "REF_EQ_UU", "REF_LT_AA", "REF_LT_AU", "REF_LT_UA", "REF_LT_UU",
        "REF_GT_AA", "REF_GT_AU", "REF_GT_UA", "REF_GT_UU",
    }


@pytest.mark.parametrize("case", TestCases, ids=[c["name"] for c in TestCases])
def test_ttk_serialization_matches_operator_contract(case):
    row = ttk_row(case)
    shapes = ast.literal_eval(row["tensor_view_shapes"])
    dtypes = ast.literal_eval(row["tensor_dtypes"])
    attrs = ast.literal_eval(row["attributes"])
    ranges = ast.literal_eval(row["input_data_ranges"])
    assert len(shapes) == len(dtypes) == len(ranges) == 16
    assert attrs == ttk_attributes(case)
    assert shapes[3] == (*shapes[0][:-1], 2, 2)
    assert shapes[6][:-1] == shapes[7]
    assert shapes[6][2] * attrs["sparse_block_size_q"] >= case["S1_max"]
    assert shapes[13][0] == case["B"] and shapes[12] == (case["B"],)
    assert shapes[15] == (0,), "Metadata must be generated per-layout by TTK adapter"
    assert case["layout_out"] == case["layout_q"]
    assert case["N1"] % case["N2"] == 0
    if case["layout_q"] == "TND":
        assert attrs["cu_seqlens_q_value"] == [i * case["S1_max"] for i in range(case["B"] + 1)]
    else:
        assert shapes[9] == shapes[11] == (0,)
        assert attrs["cu_seqlens_q_value"] == attrs["seqused_q_value"] == []
    if case["p_scale_value"] is None:
        assert shapes[8] == (0,)
    elif case["p_scale_type"] == "float32":
        assert dtypes[8] == "float32"
    else:
        assert dtypes[8] == "float8_e8m0"


def test_checked_in_artifacts_are_current():
    for path, content in artifacts().items():
        assert path.read_text(encoding="utf-8") == content, f"Regenerate with export_cases.py: {path}"


def test_adapter_signature_and_spec_registration():
    source = TESTS / "assets" / "bsa_ttk_ops.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quant_block_sparse_attn")
    assert [arg.arg for arg in function.args.args] == [
        "query", "key", "value", "q_descale", "k_descale", "v_descale", "sparse_indices", "sparse_seq_len",
        "p_scale", "cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv", "block_table", "atten_mask", "metadata",
    ]
    spec_tree = ast.parse((TESTS / "assets" / "spec.py").read_text(encoding="utf-8"))
    registry = next(n.value for n in spec_tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__spec__" for t in n.targets))
    assert ast.literal_eval(registry)["bsa_ttk_ops.quant_block_sparse_attn"] == "QuantBlockSparseAttnSpec"


@pytest.fixture(scope="module")
def comparator():
    pytest.importorskip("numpy", reason="TTK comparator needs NumPy; matrix checks remain available")
    spec = importlib.util.spec_from_file_location("qbsa_stc_compare", TESTS / "assets" / "impl" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare


@pytest.mark.parametrize("fault", ["none", "wrong_output", "lse_axes", "missing_lse", "inf_sign", "nan_instead_inf"])
def test_real_ttk_comparator_detects_layout_failures(comparator, fault):
    import numpy as np
    out = np.arange(2 * 4 * 17 * 8, dtype=np.float32).reshape(2, 4, 17, 8) / 100
    lse = np.arange(2 * 4 * 17, dtype=np.float32).reshape(2, 4, 17)
    lse[0, 0] = -np.inf
    actual = [out.copy(), lse.copy()]
    if fault == "wrong_output":
        actual[0] = out + 100
    elif fault == "lse_axes":
        actual[1] = lse.transpose(0, 2, 1)
    elif fault == "missing_lse":
        actual[1] = None
    elif fault in ("inf_sign", "nan_instead_inf"):
        actual[1][0, 0] = np.inf if fault == "inf_sign" else np.nan
    result = comparator(*actual, out, lse, return_softmax_lse=True)
    result = result if isinstance(result, list) else [result]
    assert all(r["pass"] for r in result) == (fault == "none")


@pytest.mark.parametrize("outcome", ["pass", "skipped", "failure", "error", "missing"])
def test_junit_gate_rejects_non_execution(tmp_path, outcome):
    from verify_results import verify_junit
    report = tmp_path / "junit.xml"
    child = "" if outcome == "pass" else f"<{outcome}/>"
    case = "" if outcome == "missing" else f'<testcase name="npu">{child}</testcase>'
    report.write_text(f"<testsuites><testsuite>{case}</testsuite></testsuites>", encoding="utf-8")
    if outcome == "pass":
        assert verify_junit(report, 1) == 1
    else:
        with pytest.raises(ValueError):
            verify_junit(report, 1)


@pytest.mark.parametrize("outcome", ["PASS", "FAIL", "SKIP", "", "duplicate", "missing"])
def test_ttk_gate_rejects_incomplete_reports(tmp_path, outcome):
    from verify_results import verify_ttk
    source = tmp_path / "input.csv"
    report = tmp_path / "result.csv"
    source.write_text("testcase_name\na\n", encoding="utf-8")
    row = "a,PASS\na,PASS\n" if outcome == "duplicate" else ("" if outcome == "missing" else f"a,{outcome}\n")
    report.write_text("testcase_name,precision_status\n" + row, encoding="utf-8")
    if outcome == "PASS":
        assert verify_ttk(report, source) == 1
    else:
        with pytest.raises(ValueError):
            verify_ttk(report, source)
