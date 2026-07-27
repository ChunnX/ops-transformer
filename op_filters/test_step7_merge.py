# -*- coding: utf-8 -*-
"""
Unit tests for prefix merge logic in op_step7_output.

Covers _merge_prefix_ops (moe_distribute, moe_ep, both, empty, no-match) and
write_soc_json merge behavior for ascend950 (merged) and ascend910b
(passthrough).
"""

import logging
import sys
from pathlib import Path

# Ensure the project root (parent of op_filters/) is on sys.path so that
# op_filters is importable as a package. This mirrors the pattern used in
# op_step5_extractor.py and avoids the relative-import failure that occurs
# when importing op_step7_output directly from the op_filters/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from op_filters.op_step7_output import (  # noqa: E402
    ASCEND950_MERGE_PREFIXES,
    _merge_prefix_ops,
    write_soc_json,
)


def _logger():
    """Return a shared throwaway logger for merge calls."""
    return logging.getLogger("test")


# ---------------------------------------------------------------------------
# _merge_prefix_ops tests
# ---------------------------------------------------------------------------


def test_merge_prefix_ops_moe_distribute_only():
    """All moe_distribute_* ops collapse into a single moe_distribute entry."""
    op_names = [
        "moe_distribute_dispatch",
        "moe_distribute_combine",
        "all_gather_matmul",
    ]
    expected = ["all_gather_matmul", "moe_distribute"]
    result = _merge_prefix_ops(op_names, ASCEND950_MERGE_PREFIXES, _logger())
    assert result == expected


def test_merge_prefix_ops_moe_ep_only():
    """All moe_ep_* ops collapse into a single moe_ep entry."""
    op_names = [
        "moe_ep_combine",
        "moe_ep_dispatch",
        "moe_ep_dispatch_epilogue",
        "all_gather_matmul",
    ]
    expected = ["all_gather_matmul", "moe_ep"]
    result = _merge_prefix_ops(op_names, ASCEND950_MERGE_PREFIXES, _logger())
    assert result == expected


def test_merge_prefix_ops_both_prefixes():
    """Both moe_distribute and moe_ep prefixes merge in the same call."""
    op_names = [
        "moe_distribute_dispatch",
        "moe_ep_combine",
        "all_gather_matmul",
    ]
    expected = ["all_gather_matmul", "moe_distribute", "moe_ep"]
    result = _merge_prefix_ops(op_names, ASCEND950_MERGE_PREFIXES, _logger())
    assert result == expected


def test_merge_prefix_ops_empty_input():
    """An empty input list yields an empty output list."""
    result = _merge_prefix_ops([], ASCEND950_MERGE_PREFIXES, _logger())
    assert result == []


def test_merge_prefix_ops_no_match():
    """Ops that match no configured prefix are returned sorted and unchanged."""
    op_names = ["all_gather_matmul", "matmul_all_reduce"]
    expected = ["all_gather_matmul", "matmul_all_reduce"]
    result = _merge_prefix_ops(op_names, ASCEND950_MERGE_PREFIXES, _logger())
    assert result == expected


# ---------------------------------------------------------------------------
# write_soc_json tests
# ---------------------------------------------------------------------------


def _moe_ep_mock_data(supported_product):
    """Build mock op info data for write_soc_json tests."""
    return [
        {"op_name": "moe_ep_combine", "supported_products": [supported_product]},
        {"op_name": "moe_ep_dispatch", "supported_products": [supported_product]},
        {"op_name": "moe_ep_dispatch_epilogue", "supported_products": [supported_product]},
        {"op_name": "all_gather_matmul", "supported_products": [supported_product]},
    ]


def test_write_soc_json_ascend950_merges_moe_ep(tmp_path):
    """For ascend950, moe_ep_* ops are merged into a single moe_ep line."""
    data = _moe_ep_mock_data("ascend950")
    output_path = write_soc_json(data, "ascend950", str(tmp_path), _logger())
    lines = Path(output_path).read_text(encoding="utf-8").splitlines()

    # Exact expected output: moe_ep_* collapsed into a single moe_ep line.
    assert lines == ["all_gather_matmul", "moe_ep"]
    # Single merged moe_ep line present.
    assert "moe_ep" in lines
    # No unmerged moe_ep_* lines remain.
    assert not any(line.startswith("moe_ep_") for line in lines)


def test_write_soc_json_ascend910b_keeps_separate_lines(tmp_path):
    """For ascend910b, moe_ep_* ops are written as separate lines (no merge)."""
    data = _moe_ep_mock_data("ascend910b")
    output_path = write_soc_json(data, "ascend910b", str(tmp_path), _logger())
    lines = Path(output_path).read_text(encoding="utf-8").splitlines()

    # Exact expected output: separate moe_ep_* lines, no merged moe_ep line.
    assert lines == [
        "all_gather_matmul",
        "moe_ep_combine",
        "moe_ep_dispatch",
        "moe_ep_dispatch_epilogue",
    ]
    # The three separate moe_ep_* lines are all present.
    assert "moe_ep_combine" in lines
    assert "moe_ep_dispatch" in lines
    assert "moe_ep_dispatch_epilogue" in lines
    # No merged moe_ep line is produced.
    assert "moe_ep" not in lines
