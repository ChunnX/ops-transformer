# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import csv

import pytest

from validation.msprof_parser import parse_msprof_case


def _case():
    return {
        "id": "parser", "world_size": 2,
        "operator": "all_to_allv_grouped_mat_mul",
        "performance": {
            "warmup": 1, "align_iterations": 1, "iterations": 2, "repeats": 1,
            "max_rank_skew_ratio": 0.10,
        },
    }


def test_parser_drops_warmup_and_alignment_and_uses_slowest_rank(tmp_path):
    path = tmp_path / "op_summary_test.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Op Name", "Device_id", "Task Start Time(us)", "Task Duration(us)"])
        writer.writeheader()
        for device, durations in (("0", [99, 88, 10, 20]), ("1", [98, 87, 11, 22])):
            for start, duration in enumerate(durations):
                writer.writerow({
                    "Op Name": "Ascend950AllToAllVGroupedMatmul", "Device_id": device,
                    "Task Start Time(us)": start, "Task Duration(us)": duration,
                })
    report = parse_msprof_case(tmp_path, _case())
    assert report["valid"]
    assert report["latency_us"]["p50"] == 16.5
    assert report["per_device"]["0"]["samples"] == 2


def test_parser_rejects_missing_occurrences(tmp_path):
    path = tmp_path / "op_summary_test.csv"
    path.write_text("Op Name,Device_id,Task Duration(us)\nAscend950AllToAllVGroupedMatmul,0,1\n")
    report = parse_msprof_case(tmp_path, _case())
    assert not report["valid"]
    assert report["errors"]


@pytest.mark.parametrize("name,op_type", [
    ("aclnnGroupedMatmul_GroupedMatmul_0", "GroupedMatmul"),
    ("grouped_matmul_123", "N/A"),
    ("kernel_123", "GroupedMatmul"),
])
def test_gmm_only_parser_selects_gmm_and_excludes_fused_rows(tmp_path, name, op_type):
    case = _case()
    case["operator"] = "grouped_mat_mul_all_to_allv"
    case["performance"]["gmm_only"] = True
    with (tmp_path / "op_summary_gmm.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "Op Name", "OP Type", "Device_id", "Task Start Time(us)", "Task Duration(us)",
        ])
        writer.writeheader()
        for device, durations in (("0", [99, 88, 10, 20]), ("1", [98, 87, 11, 22])):
            for start, duration in enumerate(durations):
                for op_name, kind in (
                    (name, op_type), ("GroupedMatmulAllToAllVTla", "MIX_AIC"),
                    ("Ascend950AllToAllVGroupedMatmul", "MIX_AIC"),
                    ("Cast", "Cast"),
                ):
                    writer.writerow({
                        "Op Name": op_name, "OP Type": kind, "Device_id": device,
                        "Task Start Time(us)": start, "Task Duration(us)": duration,
                    })
    report = parse_msprof_case(tmp_path, case)
    assert report["valid"]
    assert report["performance_target"] == "grouped_matmul"
    assert report["latency_us"]["p50"] == 16.5
    assert report["iterations"][0]["rank_durations_us"] == {"0": 10.0, "1": 11.0}


def test_gmm_only_parser_does_not_accept_fused_only_capture(tmp_path):
    (tmp_path / "op_summary_wrong.csv").write_text(
        "Op Name,Device_id,Task Duration(us)\nGroupedMatmulAllToAllVTla,0,1\n"
    )
    case = _case()
    case["operator"] = "grouped_mat_mul_all_to_allv"
    case["performance"]["gmm_only"] = True
    report = parse_msprof_case(tmp_path, case)
    assert not report["valid"]
    assert report["found_target_launches_per_device"] == {}
