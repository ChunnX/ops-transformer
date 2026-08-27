# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_accuracy_export_report_has_route_segments():
    source = (PROJECT_ROOT / "validation" / "accuracy_worker.py").read_text(
        encoding="utf-8"
    )
    assert "global_table=global_table_2d.cpu().tolist()" in source
    assert 'export_enabled = case["output_all_to_allv"]' in source
    assert "all_to_allv_out=reusable_all_to_allv_out" in source
    assert 'path = "auto_repeat_{}".format(repeat)' in source
    assert '"all_to_allv_export_out_repeat_{}".format(repeat)' in source
    assert 'path = "synchronized_control"' in source
    assert '"launch_diagnostics.json"' in source
    assert '"input_diagnostics.json"' in source


def test_performance_uses_export_kernel_by_default():
    source = (PROJECT_ROOT / "validation" / "performance_worker.py").read_text(
        encoding="utf-8"
    )
    assert 'output_all_to_allv = case["output_all_to_allv"]' in source
    assert "output_all_to_allv=output_all_to_allv" in source
    assert "all_to_allv_out=all_to_allv_out" in source


def test_accuracy_launcher_persists_combined_torchrun_console_log():
    source = (PROJECT_ROOT / "validation" / "run_accuracy.py").read_text(
        encoding="utf-8"
    )
    assert '"torchrun_ep{}.log".format(world_size)' in source
    assert "stderr=subprocess.STDOUT" in source
    assert '"launch_ep{}.json".format(world_size)' in source
