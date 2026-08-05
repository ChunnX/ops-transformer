"""
Test script for lightning_indexer_kl_loss.

Compares the NPU operator output with a pure-PyTorch CPU reference
across multiple shapes, dtypes, and edge cases.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torch_npu
from xrun_test import data_compare
import time
from torch.profiler import profile, record_function, ProfilerActivity

# Ensure the torch_extension package is importable
TORCH_EXTENSION_PATH = Path(__file__).resolve().parent
if str(TORCH_EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(TORCH_EXTENSION_PATH))

from cann_ops_transformer.ops import lightning_indexer_kl_loss  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import

# torch.manual_seed(1314)

# ---------------------------------------------------------------------------
# Reference implementation (纯 PyTorch, float 精度基准)
# ---------------------------------------------------------------------------
def lightning_indexer_kl_loss_ref(target_score, index_probs, eps=1e-9, weight_type='logits'):
    y = target_score
    Y = index_probs
    reduce_target = torch.sum(y, dim=-1, keepdim=True)
    P = y / (reduce_target + eps)
    log_P = torch.log(torch.clamp_min(P, eps))             # clamp_min 防止 log(0) 产生 NaN
    log_Y = torch.log(Y + eps)                       # student: +eps 保住 Y→0 处的梯度
    log_ratio = log_P - log_Y
    if weight_type == 'logits':
        weight = y                             # weighted by original y
    elif weight_type == 'probs':
        weight = P                             # weighted by normalized probability p
    else:
        raise ValueError("weight_type must be 'logits' or 'probs'")
    result = weight * log_ratio
    # torch.set_printoptions(profile="full")
    # # print("target_score:")
    # # print(target_score)
    # print("reducesum:")
    # print((reduce_target + eps)[0])
    # print("clamp_min:")
    # print(torch.clamp_min(P, eps)[0])
    # print(result[:10])
    # torch.set_printoptions(profile="default")
    # print(f"logP:{torch.sum(log_P)}")
    # print(f"logY:{torch.sum(log_Y)}")
    # print(f"log_ratio:{torch.sum(log_P - log_Y)}")
    return torch.sum(log_ratio)


# ---------------------------------------------------------------------------
# Unified test interface
# ---------------------------------------------------------------------------
def _gen_data(shape, dtype=torch.float32):
    """Generate random target_score and softmax-normalized index_probs."""
    # y = torch.randn(shape, dtype=dtype)
    # Y = torch.randn(shape, dtype=dtype)
    # return y, Y
    if len(shape) == 2:
        b, d = shape
        # y = torch.randn(b, d, dtype=dtype)
        y = torch.full((b, d), 1, dtype=dtype)
        # y = torch.linspace(1, b * d, b * d, dtype=dtype).reshape(shape)
        Y = torch.softmax(torch.randn(b, d, dtype=dtype), dim=-1)
    else:
        b, h, d = shape
        y = torch.randn(b, h, d, dtype=dtype)
        # y = torch.arange(1, (b * h * d + 1), 1).reshape(shape)
        # y = y.to(dtype)
        Y = torch.softmax(torch.randn(b, h, d, dtype=dtype), dim=-1)
    return y, Y

def get_file(path, file_name):
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if not os.path.isdir(file_path) and "op_statistic" in file:
                file_name.append(file_path)
            else:
                get_file(file_path, file_name)
    except:
        pass

def run_test(name, shape, dtype=torch.float32, eps=1e-9, diff_thd=0.0001,
             max_diff_hd=0.1, zero_target=False, zero_index_probs=False, deterministic=False,
             weight_type='logits'):
    """Run a single test case.

    Args:
        name: test name (printed as header).
        shape: tuple (B, D) for 2D or (B, H, D) for 3D.
        dtype: tensor dtype (half types cast to float32 for ref).
        eps: eps passed to both ref and NPU operator.
        diff_thd: relative error threshold.
        max_diff_hd: absolute error threshold.
        check_fn: optional callable(npu_out, y, Y) -> bool for custom assertion.
        zero_target: if True, target_score is all zeros.
        zero_index_probs: if True, index_probs is all zeros.
        weight_type: 'logits' or 'probs'.

    Returns:
        True if all checks pass.
    """
    print(f"\n=== {name} (weight_type={weight_type}) ===")

    y, Y = _gen_data(shape, dtype=dtype)
    if zero_target:
        y = torch.zeros_like(y)
    if zero_index_probs:
        Y = torch.zeros_like(Y)
    if (Y < 0).any():
        raise ValueError("index_probs 不是有效的概率分布")

    # if dtype in (torch.float16, torch.bfloat16):
    #     ref = lightning_indexer_kl_loss_ref(y.float(), Y.float(), eps=eps, weight_type=weight_type)
    # else:
    #     ref = lightning_indexer_kl_loss_ref(y, Y, eps=eps, weight_type=weight_type)
    if deterministic:
        torch.use_deterministic_algorithms(True)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            ref = lightning_indexer_kl_loss_ref(y.float(), Y.float(), eps=eps, weight_type=weight_type)
    torch.npu.synchronize()
    profile_path = "./fag_profiling"
    os.system(f"rm -rf {profile_path}")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False
            )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config
    ) as prof:
        for step in range(1):
            npu_out = lightning_indexer_kl_loss(y.npu(), Y.npu(), eps=eps, weight_type=weight_type)
            prof.step()
    file_name = []
    get_file(profile_path, file_name)
    result = pd.read_csv(file_name[0])
    print(result)
    print("Total Time(us):",result["Total Time(us)"].sum())
    return data_compare(npu_out.cpu().to(torch.float32).detach().numpy(), ref.to(torch.float32).detach().numpy(), diff_thd=diff_thd, max_diff_hd=max_diff_hd)


# ---------------------------------------------------------------------------
# Custom check functions for edge cases
# ---------------------------------------------------------------------------
def _check_zero_target(npu_out, y_cpu, Y_cpu):
    npu_val = float(npu_out.cpu())
    passed = torch.isfinite(npu_out).item()
    print(f"  zero_target_loss={npu_val:.10f}  expected≈0.0  {'PASS' if passed else 'FAIL'}")
    return passed


def _check_zero_index_probs(npu_out, y_cpu, Y_cpu):
    npu_val = float(npu_out.cpu())
    passed = torch.isfinite(npu_out).item()
    print(f"  zero_index_probs_loss={npu_val:.10f}  isfinite={'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
TESTS = [
    ("big basic_2d: (B, D) float32",         (800000, 2336),           {"deterministic": True}),
    ("basic_2d: (B, D) float32",         (2, 2332),           None),
    ("basic_3d: (B, H, D) float32",      (4, 10, 333),       {"eps": 0.5}),
    ("fp16: (B, H, D) float16",          (4, 2, 4444),       {"dtype": torch.float16, "diff_thd": 0.001}),
    ("bf16: (B, H, D) bfloat16",         (40, 264, 10),       {"eps": 0.5,"dtype": torch.bfloat16, "diff_thd": 0.005}),
    ("custom_eps: eps=1e-6",             (4000, 1),         {"eps": 1e-6}),
    ("large_d: (B, H, 512) float32",     (2125, 45, 8192),        None),
    ("edge: zero target_score",          (2, 4, 128),        {"zero_target": True}),
    ("edge: zero index_probs",           (2, 4, 128),          {"zero_index_probs": True}),
    # --- weight_type='probs' tests ---
    ("probs basic_2d: (B, D) float32",        (8888, 2332),     {"eps": 0.6, "weight_type": "probs", "deterministic": True}),
    ("probs basic_3d: (B, H, D) float32",     (4, 10, 333),     {"eps": 0.5, "weight_type": "probs"}),
    ("probs fp16: (B, H, D) float16",         (4, 2, 4444),     {"dtype": torch.float16, "diff_thd": 0.001, "weight_type": "probs"}),
    ("probs bf16: (B, H, D) bfloat16",        (40, 264, 10),    {"eps": 0.5, "dtype": torch.bfloat16, "diff_thd": 0.005, "weight_type": "probs"}),
    ("probs custom_eps: eps=1e-6",           (4000, 1),         {"eps": 1e-6, "weight_type": "probs"}),
    ("probs large_d: (B, H, 512) float32",   (2125, 45, 8192),  {"weight_type": "probs"}),
    ("probs edge: zero target_score",        (2, 4, 128),       {"zero_target": True, "weight_type": "probs"}),
    ("probs edge: zero index_probs",         (2, 4, 128),       {"zero_index_probs": True, "weight_type": "probs"}),
    # # --- weight_type='probs' with deterministic ---
    ("probs deterministic_2d: (B, D) float32", (4000, 100),     {"weight_type": "probs", "deterministic": True}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch_npu.npu.set_device(2)
    passed = 0
    failed = 0
    failedList = []
    while failed == 0:
        for name, shape, kwargs in TESTS:
            ok, _, _= run_test(name, shape, **(kwargs or {}))
            if ok == "Pass":
                passed += 1
            else:
                failed += 1
                failedList.append(name)

        print(f"\n{'='*50}")
        print(f"Results: {passed}/{len(TESTS)} passed, {failed} failed")
        print(failedList)
        if failed:
            sys.exit(1)
        break


