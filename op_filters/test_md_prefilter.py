# -*- coding: utf-8 -*-
"""
Unit tests for the .md pre-filter added to select_op_tests_main.

Background: a non-mc2 markdown file (e.g. README.md, docs/foo.md) used to be
caught by the mc2-external check and trigger a full test run. A pre-filter
step now strips .md files right after the whitelist step (Step 1) and before
the mc2-external check, so .md-only changes no longer trigger full tests.

These tests cover:
- is_markdown_file edge cases (the helper reused by the pre-filter)
- The pre-filter logic in isolation: given a mixed file list, .md entries are
  removed and non-.md entries are preserved, so the downstream mc2-external
  check sees no .md files.
"""

import logging
import sys
from pathlib import Path

# Ensure the project root (parent of op_filters/) is on sys.path so that
# op_filters is importable as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from op_filters.op_step2_ignored import is_markdown_file  # noqa: E402


# ---------------------------------------------------------------------------
# is_markdown_file edge cases
# ---------------------------------------------------------------------------


def test_is_markdown_file_plain_md():
    """A simple README.md is recognized as markdown."""
    assert is_markdown_file("README.md") is True


def test_is_markdown_file_uppercase_extension():
    """Uppercase .MD extension is still recognized (case-insensitive)."""
    assert is_markdown_file("DOCS/NOTES.MD") is True


def test_is_markdown_file_nested_path():
    """A .md file nested under a non-mc2 path is still recognized."""
    assert is_markdown_file("docs/zh/context/some_concept.md") is True


def test_is_markdown_file_mc2_nested_md():
    """A .md file inside mc2/ is also recognized (pre-filter is path-agnostic)."""
    assert is_markdown_file("mc2/all_gather_matmul/README.md") is True


def test_is_markdown_file_non_md_source():
    """A .cpp source file is not markdown."""
    assert is_markdown_file("mc2/all_gather_matmul/op_host/foo.cpp") is False


def test_is_markdown_file_non_md_root_config():
    """A root config file without .md extension is not markdown."""
    assert is_markdown_file("CMakeLists.txt") is False


def test_is_markdown_file_md_like_but_not_md():
    """A file whose name contains 'md' but does not end with .md is not markdown."""
    assert is_markdown_file("mc2/metadata.json") is False
    assert is_markdown_file("README.md.bak") is False


def test_is_markdown_file_empty_and_edge():
    """Empty string and edge inputs do not crash and return False."""
    assert is_markdown_file("") is False
    assert is_markdown_file(".md") is True  # a file literally named ".md"


# ---------------------------------------------------------------------------
# Pre-filter logic in isolation
# ---------------------------------------------------------------------------
#
# The pre-filter in select_op_tests_main.main() is:
#   md_files = [f for f in filtered_files if is_markdown_file(f)]
#   if md_files:
#       filtered_files = [f for f in filtered_files if not is_markdown_file(f)]
#
# We replicate that exact logic here against mock inputs to assert the
# downstream mc2-external check (which runs on the post-filter list) will
# never see a .md file. This is a unit test of the pre-filter behavior
# without invoking the full main() pipeline (which needs a real mc2 dir).


def _apply_md_prefilter(file_paths, logger):
    """Replica of the pre-filter block from select_op_tests_main.main()."""
    md_files = [f for f in file_paths if is_markdown_file(f)]
    if md_files:
        file_paths = [f for f in file_paths if not is_markdown_file(f)]
        logger.info("ignored %d md files", len(md_files))
    return file_paths, md_files


def _logger():
    return logging.getLogger("test")


def test_prefilter_strips_non_mc2_md_so_no_full_test_trigger():
    """A non-mc2 README.md alone is removed; remaining list is empty.

    Before the fix this file would hit the mc2-external check and trigger a
    full test run. After the fix the remaining list is empty, so the pipeline
    reaches the 'no remaining files -> no tests' branch instead.
    """
    files = ["README.md"]
    remaining, md = _apply_md_prefilter(files, _logger())
    assert md == ["README.md"]
    assert remaining == []


def test_prefilter_strips_mc2_md_alongside_source_files():
    """A .md inside mc2/ is stripped; sibling source files are preserved.

    This confirms the pre-filter is purely extension-based and does not
    accidentally keep mc2/*.md files that Step 2 would otherwise drop later.
    """
    files = [
        "mc2/all_gather_matmul/README.md",
        "mc2/all_gather_matmul/op_host/all_gather_matmul_def.cpp",
    ]
    remaining, md = _apply_md_prefilter(files, _logger())
    assert md == ["mc2/all_gather_matmul/README.md"]
    assert remaining == ["mc2/all_gather_matmul/op_host/all_gather_matmul_def.cpp"]


def test_prefilter_mixed_md_and_non_md_non_mc2():
    """A mix of non-mc2 .md and non-mc2 source: only .md is removed.

    Crucially, the non-mc2 source file (common/foo.cpp) survives the
    pre-filter, so the mc2-external check will still correctly trigger a
    full test for it. The pre-filter must NOT swallow legitimate non-mc2
    source changes.
    """
    files = ["docs/notes.md", "common/foo.cpp", "README.md"]
    remaining, md = _apply_md_prefilter(files, _logger())
    assert sorted(md) == ["README.md", "docs/notes.md"]
    assert remaining == ["common/foo.cpp"]


def test_prefilter_no_md_leaves_list_unchanged():
    """When there are no .md files, the list is returned unchanged."""
    files = [
        "mc2/all_gather_matmul/op_host/a.cpp",
        "common/foo.cpp",
    ]
    remaining, md = _apply_md_prefilter(files, _logger())
    assert md == []
    assert remaining == files


def test_prefilter_only_md_yields_empty_remaining():
    """An all-.md input (mc2 and non-mc2) yields an empty remaining list."""
    files = [
        "README.md",
        "mc2/all_gather_matmul/README.md",
        "docs/zh/intro.md",
    ]
    remaining, md = _apply_md_prefilter(files, _logger())
    assert sorted(md) == sorted(files)
    assert remaining == []
