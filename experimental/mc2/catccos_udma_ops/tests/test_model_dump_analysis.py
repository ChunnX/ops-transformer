# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import dataclasses
from pathlib import Path

import numpy as np

from validation.model_dump_analysis import (
    Artifact,
    LoadedTensor,
    _build_artifact,
    _canonical_key,
    _infer_tensor_name,
    build_timeline,
    compare_tensors,
    derive_diagnosis,
    pair_artifacts,
    parse_args,
    prepare_dump_files,
)


def _artifact(side: str, key: str, io_kind: str, tensor_name: str) -> Artifact:
    return Artifact(
        side=side,
        key=key,
        relative_path=key + ".pt",
        path=Path(key + ".pt"),
        rank=0,
        sequence=7,
        op="all_to_allv_grouped_mat_mul",
        io_kind=io_kind,
        tensor_name=tensor_name,
        invocation="rank=0|sequence=7|op=all_to_allv_grouped_mat_mul",
        order_confidence="explicit",
    )


def _tensor(values, component="root", dtype="bfloat16") -> LoadedTensor:
    array = np.asarray(values, dtype=np.float32)
    return LoadedTensor(
        path=Path(component + ".pt"),
        component=component,
        array=array,
        serialized_dtype=dtype,
        storage_numel=array.size,
        storage_offset=0,
        stride=tuple(
            int(value // array.itemsize) for value in array.strides
        ),
    )


def test_auto_pairing_ignores_udma_marker_and_operator_spelling():
    strip = ("udma",)
    fused_key = _canonical_key(
        "rank_0/call_7/alltoallvgroupedmatmul_udma_output.pt", strip
    )
    reference_key = _canonical_key(
        "rank_0/call_7/all_to_allv_grouped_mat_mul_output.pt", strip
    )
    assert fused_key == reference_key

    fused = dataclasses.replace(
        _artifact("fused", "placeholder", "output", "output"), key=fused_key
    )
    reference = dataclasses.replace(
        _artifact("reference", "placeholder", "output", "output"), key=reference_key
    )
    pairs, fused_only, reference_only = pair_artifacts([fused], [reference])
    assert len(pairs) == 1
    assert not fused_only
    assert not reference_only


def test_final_gmm_output_is_not_misclassified_as_alltoallv_export():
    assert (
        _infer_tensor_name("all_to_allv_grouped_mat_mul_output.pt", "output")
        == "output"
    )
    assert (
        _infer_tensor_name("all_to_allv_grouped_mat_mul_export_output.pt", "output")
        == "all_to_allv_output"
    )


def test_nested_mc2_layout_maps_four_hidden_state_boundaries():
    strip = ("udma",)
    cases = (
        (
            "hidden_states_before_a2a_gemm.pt",
            "all_to_allv_grouped_mat_mul",
            "input",
        ),
        (
            "hidden_states_after_a2a_gemm.pt",
            "all_to_allv_grouped_mat_mul",
            "output",
        ),
        (
            "hidden_states_before_gemm_a2a.pt",
            "grouped_mat_mul_all_to_allv",
            "input",
        ),
        (
            "hidden_states_after_gemm_a2a.pt",
            "grouped_mat_mul_all_to_allv",
            "output",
        ),
    )
    for filename, expected_op, expected_io in cases:
        fused_relative = "rank_00003/mc2_udma/first_call_00007/{}".format(
            filename
        )
        reference_relative = "rank_00003/mc2/first_call_00007/{}".format(
            filename
        )
        fused = _build_artifact(
            "fused", fused_relative, Path(fused_relative), strip
        )
        reference = _build_artifact(
            "reference", reference_relative, Path(reference_relative), strip
        )
        assert fused.key == reference.key
        assert fused.rank == 3
        assert fused.sequence == 7
        assert fused.op == expected_op
        assert fused.io_kind == expected_io
        assert fused.invocation == reference.invocation


def test_dump_dir_is_a_standalone_cli_layout():
    args = parse_args(["--dump-dir", "/tmp/model_dump"])
    assert args.dump_dir == Path("/tmp/model_dump")
    assert args.fused_root is None
    assert args.reference_root is None


def test_comparison_distinguishes_new_nan_from_inherited_nan():
    reference = _tensor([[1.0, 2.0], [float("nan"), 4.0]], "reference")
    fused = _tensor([[1.0, float("nan")], [float("nan"), 4.0]], "fused")
    report = compare_tensors(
        fused,
        reference,
        chunk_elements=2,
        top_k=10,
        nonfinite_samples=10,
        atol=None,
        rtol=None,
    )

    assert report["introduced_nonfinite"] == 1
    assert report["actual_nonfinite"] == 2
    assert report["reference_nonfinite"] == 1
    assert report["counters"]["introduced_nan"] == 1
    assert report["first_issue"]["row"] == 0
    assert report["first_issue"]["column_flat"] == 1
    assert not report["precision_passed"]


def test_timeline_localizes_output_nan_when_paired_inputs_are_exact():
    input_record = {
        **_artifact("fused", "input", "input", "x").__dict__,
        "exact_equal": True,
        "precision_passed": True,
        "introduced_nonfinite": 0,
        "reference_nonfinite": 0,
    }
    output_record = {
        **_artifact("fused", "output", "output", "output").__dict__,
        "exact_equal": False,
        "precision_passed": False,
        "introduced_nonfinite": 3,
        "reference_nonfinite": 0,
    }
    timeline = build_timeline([input_record, output_record])
    first, hints = derive_diagnosis(timeline, [input_record, output_record])

    assert len(timeline) == 1
    assert timeline[0]["classification"] == "operator_introduced_nonfinite"
    assert first is not None
    assert first["sequence"] == 7
    assert any("localizes creation" in hint for hint in hints)


def test_byte_split_torch_file_is_reassembled_in_numeric_order(tmp_path):
    dump_root = tmp_path / "dump"
    output_root = tmp_path / "analysis"
    dump_root.mkdir()
    (dump_root / "rank0_output.pt.part.0").write_bytes(b"abc")
    (dump_root / "rank0_output.pt.part.1").write_bytes(b"def")

    files = prepare_dump_files(
        dump_root,
        output_root / "reconstructed",
        (".pt",),
        output_root,
    )

    assert set(files) == {"rank0_output.pt"}
    assert files["rank0_output.pt"].read_bytes() == b"abcdef"
