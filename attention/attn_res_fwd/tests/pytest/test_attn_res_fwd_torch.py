# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""AttnResFwd Torch 接口精度测试（cann_ops_transformer.attn_res_fwd）。"""

from __future__ import annotations

import os

import pytest
import torch

import attn_res_fwd_golden as golden
import attn_res_fwd_torch_runner as torch_runner
import npu_device
from test_attn_res_fwd_paramset import (
    CI_NPU_CASES,
    FORWARD_H_SWEEP_CASES,
    GRID_TN_7168_CASES,
    PREVIOUS_FAIL_CASES,
    PRODUCTION_CASE,
    RESIDENT_CASES,
)

BF16_MERE_THRESHOLD = golden.BF16_MERE_THRESHOLD


def _require_torch_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU not available")


def _require_custom_opp():
    custom = os.environ.get("ATTN_RES_FWD_CUSTOM_SETENV")
    if custom is None and not os.path.exists(
        "/data/kdy/cann/custompkgbase/vendors/custom_transformer/bin/set_env.bash"
    ):
        pytest.skip("custom opp package not installed; source cann_env.sh first")


def _run_torch_case(case: dict) -> dict:
    _require_torch_npu()
    _require_custom_opp()
    attn_res_fwd = torch_runner.get_attn_res_fwd_fn()

    device_id = npu_device.resolve_device_id()
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)

    T, N, H = case["T"], case["N"], case["H"]
    seed = case.get("seed", 0)
    # CPU 生成再 .to(npu)，避免 custom OPP 下 NPU 直接 randn(bf16) 失败
    prefix_sum, block_residual, proj_weight, norm_weight, norm_eps = golden.prepare_tensors(
        T, N, H, seed=seed, device="cpu"
    )
    prefix_sum = prefix_sum.to(device)
    block_residual = block_residual.to(device)
    proj_weight = proj_weight.to(device)
    norm_weight = norm_weight.to(device)

    golden_out = golden.apply_attn_res_cust(
        prefix_sum.cpu(), block_residual.cpu(), proj_weight.cpu(), norm_weight.cpu(), norm_eps
    )

    actual = attn_res_fwd(prefix_sum, block_residual, proj_weight, norm_weight, norm_eps)
    torch.npu.synchronize()

    stats = golden.compare_with_data_compare(actual.cpu(), golden_out)
    stats["case"] = case
    stats["device_id"] = device_id
    return stats


def _assert_torch_pass(stats: dict, case_id: str) -> None:
    assert stats["result"] == "pass", (
        f"[{case_id}] Torch precision fail MERE={stats.get('mere')} MARE={stats.get('mare')} "
        f"device={stats.get('device_id')}"
    )
    assert stats["mere"] < BF16_MERE_THRESHOLD, (
        f"[{case_id}] MERE {stats['mere']} >= {BF16_MERE_THRESHOLD}"
    )


@pytest.fixture(scope="session", autouse=True)
def _select_npu_device():
    """Session 级自动选卡（尊重已设置的 DEVICE_ID）。"""
    device_id = npu_device.resolve_device_id()
    print(npu_device.format_status_summary())
    print(f"Selected NPU device: {device_id}")
    yield device_id


@pytest.mark.ci
def test_attn_res_fwd_torch_production():
    case = PRODUCTION_CASE
    stats = _run_torch_case(case)
    print(f"Torch production: MERE={stats['mere']} device={stats['device_id']}")
    _assert_torch_pass(stats, case["id"])


@pytest.mark.ci
@pytest.mark.parametrize("case", CI_NPU_CASES, ids=[c["id"] for c in CI_NPU_CASES])
def test_attn_res_fwd_torch_ci(case):
    stats = _run_torch_case(case)
    _assert_torch_pass(stats, case["id"])


@pytest.mark.extended
@pytest.mark.parametrize("case", PREVIOUS_FAIL_CASES, ids=[c["id"] for c in PREVIOUS_FAIL_CASES])
def test_attn_res_fwd_torch_previous_fails(case):
    stats = _run_torch_case(case)
    print(f"Torch previous-fail {case['id']}: MERE={stats.get('mere')}")
    _assert_torch_pass(stats, case["id"])


@pytest.mark.resident
@pytest.mark.parametrize("case", RESIDENT_CASES, ids=[c["id"] for c in RESIDENT_CASES])
def test_attn_res_fwd_torch_resident(case):
    stats = _run_torch_case(case)
    print(f"Torch resident {case['id']}: MERE={stats.get('mere')}")
    _assert_torch_pass(stats, case["id"])


@pytest.mark.sweep
@pytest.mark.parametrize("case", FORWARD_H_SWEEP_CASES, ids=[c["id"] for c in FORWARD_H_SWEEP_CASES])
def test_attn_res_fwd_torch_h_sweep(case):
    stats = _run_torch_case(case)
    print(f"Torch H-sweep {case['id']}: T={case['T']} H={case['H']} MERE={stats.get('mere')}")
    _assert_torch_pass(stats, case["id"])


@pytest.mark.grid_tn
@pytest.mark.parametrize("case", GRID_TN_7168_CASES, ids=[c["id"] for c in GRID_TN_7168_CASES])
def test_attn_res_fwd_torch_grid_tn(case):
    """泛化精度：以 data_compare success 为门控（不强制 MERE < 2^-7）。"""
    stats = _run_torch_case(case)
    print(
        f"Torch grid_tn {case['id']}: T={case['T']} N={case['N']} "
        f"dc={stats.get('data_compare')} MERE={stats.get('mere')}"
    )
    assert stats["data_compare"] == "success", (
        f"[{case['id']}] data_compare={stats.get('data_compare')} "
        f"MERE={stats.get('mere')} MARE={stats.get('mare')}"
    )
