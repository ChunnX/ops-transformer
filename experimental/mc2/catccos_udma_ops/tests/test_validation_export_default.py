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


def test_performance_uses_export_kernel_by_default():
    source = (PROJECT_ROOT / "validation" / "performance_worker.py").read_text(
        encoding="utf-8"
    )
    assert 'output_all_to_allv = case["output_all_to_allv"]' in source
    assert "output_all_to_allv=output_all_to_allv" in source
    assert "all_to_allv_out=all_to_allv_out" in source
