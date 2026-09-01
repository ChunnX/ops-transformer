from batch_consistency.pytest_support import format_consistency_summary


def test_format_consistency_summary_lists_all_outcomes():
    report = {
        "baseline_precision": {
            "pass": False,
            "output": {"fulfill_percent": 74.32098388671875},
        },
        "relations": [
            {
                "case": "mode1_reorder_B2",
                "pass": False,
                "batch_consistency": {"pass": True},
                "precision": {
                    "pass": False,
                    "output": {"fulfill_percent": 74.32098388671875},
                },
            },
            {
                "case": "mode2_split_g0_B1",
                "pass": True,
                "batch_consistency": {"pass": True},
                "precision": {
                    "pass": True,
                    "output": {"fulfill_percent": 100.0},
                },
            },
        ],
        "skipped_modes": {
            "token-split": "NOT_APPLICABLE: mask window changed",
        },
        "pass": False,
    }

    summary = format_consistency_summary(report)

    assert "baseline" in summary and "FAILED (74.32%)" in summary
    assert "mode1_reorder_B2" in summary and "PASS" in summary
    assert "mode2_split_g0_B1" in summary and "PASS (100.00%)" in summary
    assert "token-split" in summary and "SKIPPED" in summary
    assert "NOT_APPLICABLE" not in summary
    assert "OVERALL: FAILED" in summary
