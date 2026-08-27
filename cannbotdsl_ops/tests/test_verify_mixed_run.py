from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_VERIFIER_PATH = Path(__file__).with_name("verify_mixed_run.py")
_SPEC = importlib.util.spec_from_file_location("mixed_run_verifier", _VERIFIER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load verifier: {_VERIFIER_PATH}")
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)


def test_required_symbols_cover_l0_standalone_and_custom_l2_delivery():
    required = set(verifier.REQUIRED_SYMBOLS)
    assert {
        "l0op::FlashKda(",
        "aclnnFlashKda",
        "aclnnFlashKdaGetWorkspaceSize",
        "aclnnCustomFlashKda",
        "aclnnCustomFlashKdaGetWorkspaceSize",
    } <= required
    assert "aclnn_custom_flash_kda.h" in verifier.REQUIRED_PUBLIC_HEADERS


def test_symbol_verifier_rejects_missing_custom_l2_export():
    symbols = "\n".join(
        symbol
        for symbol in verifier.REQUIRED_SYMBOLS
        if symbol != "aclnnCustomFlashKdaGetWorkspaceSize"
    )
    with pytest.raises(AssertionError, match="aclnnCustomFlashKdaGetWorkspaceSize"):
        verifier._verify_required_symbols(symbols)


def test_get_workspace_prefix_does_not_mask_missing_execute_export():
    symbols = "\n".join(
        symbol
        for symbol in verifier.REQUIRED_SYMBOLS
        if symbol != "aclnnCustomFlashKda"
    )
    with pytest.raises(AssertionError, match="'aclnnCustomFlashKda'"):
        verifier._verify_required_symbols(symbols)


def test_symbol_verifier_accepts_complete_delivery_surface():
    verifier._verify_required_symbols("\n".join(verifier.REQUIRED_SYMBOLS))
