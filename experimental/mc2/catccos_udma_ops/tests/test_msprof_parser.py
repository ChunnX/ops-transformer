# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import csv

from validation.msprof_parser import parse_msprof_case


def _case():
    return {
        "id": "parser", "world_size": 2,
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
