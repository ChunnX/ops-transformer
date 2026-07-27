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


# ---------------------------------------------------------------------------
# Reference implementation (纯 PyTorch, float 精度基准)
# ---------------------------------------------------------------------------
def lightning_indexer_kl_loss_ref(target_score, index_probs, eps=1e-9, weight_type='logits'):
    y = target_score
    Y = index_probs
    P = y / torch.sum(y, dim=-1, keepdim=True)      # teacher: normalize target_score
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
    return torch.sum(result)


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
        y = torch.randn(b, d, dtype=dtype)
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
    ("basic_2d: (B, D) float32",         (8888, 2332),           None),
    ("basic_3d: (B, H, D) float32",      (4, 10, 333),       {"eps": 0.5}),
    ("fp16: (B, H, D) float16",          (4, 2, 4444),       {"dtype": torch.float16, "diff_thd": 0.001}),
    ("bf16: (B, H, D) bfloat16",         (40, 264, 10),       {"eps": 0.5,"dtype": torch.bfloat16, "diff_thd": 0.005}),
    ("custom_eps: eps=1e-6",             (4000, 1),         {"eps": 1e-6}),
    ("large_d: (B, H, 512) float32",     (2125, 45, 8192),        None),
    ("edge: zero target_score",          (2, 4, 128),        {"zero_target": True}),
    ("edge: zero index_probs",           (2, 4, 128),          {"zero_index_probs": True}),
    # --- weight_type='probs' tests ---
    ("probs basic_2d: (B, D) float32",        (8888, 2332),     {"weight_type": "probs"}),
    ("probs basic_3d: (B, H, D) float32",     (4, 10, 333),     {"eps": 0.5, "weight_type": "probs"}),
    ("probs fp16: (B, H, D) float16",         (4, 2, 4444),     {"dtype": torch.float16, "diff_thd": 0.001, "weight_type": "probs"}),
    ("probs bf16: (B, H, D) bfloat16",        (40, 264, 10),    {"eps": 0.5, "dtype": torch.bfloat16, "diff_thd": 0.005, "weight_type": "probs"}),
    ("probs custom_eps: eps=1e-6",           (4000, 1),         {"eps": 1e-6, "weight_type": "probs"}),
    ("probs large_d: (B, H, 512) float32",   (2125, 45, 8192),  {"weight_type": "probs"}),
    ("probs edge: zero target_score",        (2, 4, 128),       {"zero_target": True, "weight_type": "probs"}),
    ("probs edge: zero index_probs",         (2, 4, 128),       {"zero_index_probs": True, "weight_type": "probs"}),
    # --- weight_type='probs' with deterministic ---
    ("probs deterministic_2d: (B, D) float32", (4000, 100),     {"weight_type": "probs", "deterministic": True}),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch_npu.npu.set_device(7)
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


import os
import torch
import torch_npu
import numpy as np
import math
from einops import rearrange
import random
import torch.nn.functional as F
import csv
from precision_compare import data_compare
from collections import defaultdict
import os
import glob
import shutil
import sys
import traceback
torch.use_deterministic_algorithms(True)

gtype = torch.float32
np.random.seed(111)
torch.manual_seed(111)


def double_check(a, b, thershold = 2**-6):
    # 确保a和b形状一致
    if a.shape != b.shape:
        raise ValueError("a和b的形状必须一致")
    # 筛选a中大于1的元素，计算与b对应位置的差值
    mask = a > 1  # 生成布尔掩码，标记a中大于1的位置
    diffs = (abs(a[mask] - b[mask]) / a[mask])   # 只计算满足条件位置的差值
    if torch.any(diffs > thershold):
        print("double check failed. ", torch.max(diffs))
        return False

    abs_mask = a <= 1
    abs_diffs = (abs(a[abs_mask] - b[abs_mask]))
    if torch.any(abs_diffs > thershold):
        print("abs double check failed. ", torch.max(abs_diffs))
        return False
    return True

def is_similar(result, expected, rtol=0.01, atol=0.01):
    return result.shape == expected.shape and torch.allclose(result, expected, rtol, atol)


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype = torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu[1:]


def truncated_normal(mean, std, min, max, size):
    x = torch.normal(mean, std, size)
    x = torch.where((x < min) | (x > max), torch.tensor(0.0), x)
    return x

def broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    batch_num = kv_shape[0]
    S = kv_shape[2]
    head_dim = kv_shape[3]
    kv_res = torch.zeros([batch_num, numHeads, S, head_dim]).to(dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res

def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum

def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res

def simple_softmax(src, max, sum):
    dst = np.exp(src - max)/sum
    return dst

def tforward(q, k, v, drop_mask, atten_mask, softmax_scale, keep_prob):
    qk = (torch.matmul(q, k.permute(0, 1, 3, 2))).mul(softmax_scale)

    if atten_mask != None:
        qk = qk + atten_mask.bool() * (-4000000000.0)
    softmax_res, x_max, x_sum = tsoftmax(qk)

    if drop_mask is not None:
        drop_res = softmax_res.mul(drop_mask)
    else:
        drop_res = softmax_res
    y = torch.matmul(drop_res, v)
    y = y.mul(1.0 / (keep_prob))
    return y, softmax_res , x_max, x_sum

 

def debug_print(data):
    print("=============python debug print==============")
    print(data.flatten()[:32])

def tbackward(dx, q, k, v, softmax_res, atten_mask, drop_mask, softmax_scale, keep_prob):
    if drop_mask is not None:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / (keep_prob)).permute(0, 1, 3, 2)
    else:
        drop_res = softmax_res.permute(0, 1, 3, 2)

    dv = torch.matmul(drop_res, dx)
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask is not None:
        dp_drop = dp.mul(drop_mask).mul(1.0 / (keep_prob))
    else:
        dp_drop = dp

    softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res))
    dq = torch.matmul(softmax_grad_res, k) * softmax_scale
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q) * softmax_scale

    return dq, dk, dv

def tbackward_new(dx, q, k, v, x_max, x_sum, atten_in, drop_mask, atten_mask, softmax_scale, keep_prob):

    # cube1
    qk = (torch.matmul(q, k.permute(0, 1, 3, 2))).mul(softmax_scale) # BNS1D * BNDS2=BNS1S2
    dyv = torch.matmul(dx, v.permute(0, 1, 3, 2)) # BNS1D * BNDS2 = BNS1S2

    # vec
    # SubGraphA
    if atten_mask is not None:
        qk = qk + atten_mask.bool() * (-4000000000.0) # BNS1S2
    p = simple_softmax(qk, x_max, x_sum) # BNS1S2
    if drop_mask is not None:
        p_drop = p.mul(drop_mask).mul(1.0 / (keep_prob)).permute(0, 1, 3, 2)# BNS2S1
    else:
        p_drop = p.permute(0, 1, 3, 2)# BNS2S1
    # SubGraphB
    softmax_grad = (dx * atten_in).sum(dim=-1, keepdims=True) # BNS1D
    if drop_mask is not None:
        dp = dyv.mul(drop_mask).mul(1.0 / (keep_prob)) # BNS1S2
    else:
        dp = dyv # BNS1S2
    ds = p * (dp - softmax_grad) # BNS1S2

    # cube2
    dv = torch.matmul(p_drop, dx) # BNS2S1 * BNS1D = BNS2D
    dq = torch.matmul(ds, k).mul(softmax_scale) # BNS1S2 * BNS2D = BNS1D
    dk = torch.matmul(ds.permute(0, 1, 3, 2), q).mul(softmax_scale) # BNS2S1 * BNS1D=BNS2D 
    return dq, dk, dv

def get_full_numels(query_head_num, ac_seq_qlen, ac_seq_kvlen):
    numels = query_head_num
    s1 = ac_seq_qlen[0]
    s2 = ac_seq_kvlen[0]
    s2_aligned = (s2 + 7) // 8 * 8
    accum = s1 * s2_aligned
    for i in range(1, len(ac_seq_qlen)):
        s1 = ac_seq_qlen[i] - ac_seq_qlen[i - 1]
        s2 = ac_seq_kvlen[i] - ac_seq_kvlen[i - 1]
        s2_aligned = (s2 + 7) // 8 * 8
        accum += s1 * s2_aligned
    numels *= accum
    return numels

def get_drop_mask(N, ac_seq_qlen, ac_seq_kvlen, seed=2, gen_p=0.1, input_dtype=torch.bfloat16):
    numels = get_full_numels(N, ac_seq_qlen, ac_seq_kvlen)
    align_128_length = (numels + 128 - 1) // 128 * 128
    align_8_length = (align_128_length + 7) // 8 * 8
    drop_masks_uint8 = torch_npu.npu_dropout_gen_mask([align_8_length,], p=gen_p, dtype=input_dtype, device="npu")
    drop_masks_bit_np = np.unpackbits(drop_masks_uint8.cpu().numpy(), count=align_8_length, bitorder='little')
    drop_masks_bit = torch.from_numpy(drop_masks_bit_np)
    drop_masks_bit = torch.tensor(drop_masks_bit, dtype=torch.uint8)
    return drop_masks_uint8.cpu(), drop_masks_bit.cpu()

def get_sink_mask(_atten_mask, _shape, sink_num=4):
    cur_col_idx = 0
    for i in range(sink_num):
        _atten_mask[cur_col_idx, :cur_col_idx] = False
        cur_col_idx += 1
    _atten_mask[cur_col_idx:, :sink_num] = False

    return _atten_mask

def test_flash_attention_training_v2(
    input_dtype,
    q_head_num,
    kv_head_num,
    head_dim,
    keep_prob,
    actual_seq_qlen,
    actual_seq_kvlen,
    sparse_mode,
    pre_tokens,
    next_tokens,
    case_name,
    sink_enable = False,
    enable_prof = False,
    is_check_deterministic = False,
):
    softmax_scale = 1 / (head_dim ** 0.5)
    batch_num = len(actual_seq_qlen)
    max_seqlen_q = max(actual_seq_qlen)
    max_seqlen_k = max(actual_seq_kvlen)
    cu_seqlens_q = get_cu_seqlens(actual_seq_qlen)
    cu_seqlens_k = get_cu_seqlens(actual_seq_kvlen)
    q_total_token_num = sum(actual_seq_qlen)
    kv_total_token_num = sum(actual_seq_kvlen)

    print(f"q_head_num: {q_head_num}")
    print(f"kv_head_num: {kv_head_num}")
    print(f"actual_seq_qlen: {actual_seq_qlen}")
    print(f"actual_seq_kvlen: {actual_seq_kvlen}")
    print("max seqlen q is ", max_seqlen_q)
    print("max seqlen k is ", max_seqlen_k)
    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    print(f"q_total_token_num: {q_total_token_num}")
    print(f"kv_total_token_num: {kv_total_token_num}")

    # generate TND atten_mask
    shape = [max_seqlen_q, max_seqlen_k]
    if sparse_mode == 0:
        opname = 'nomask_flash_attention'
        atten_mask = None
        atten_mask_npu = None
        pre_tokens = 65536
        next_tokens = 65536
    elif sparse_mode == 1:
        opname = 'fullmask_flash_attention'
        # atten_mask = torch.randint(0, 2, (q_total_token_num, kv_total_token_num)).to(torch.bool)
        atten_mask = (torch.triu(torch.ones(shape), diagonal=kv_total_token_num - q_total_token_num + 1)).to(torch.bool)
        atten_mask_npu = atten_mask.unsqueeze(0).unsqueeze(0).npu()
        pre_tokens = -1
        next_tokens = -1
    elif sparse_mode == 3:
        opname = 'sparse_flash_attention'
        atten_mask = (torch.triu(torch.ones(shape), diagonal=kv_total_token_num - q_total_token_num + 1)).to(torch.bool)
        atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()
        pre_tokens = -1
        next_tokens = -1
    elif sparse_mode == 4:
        opname = 'sparse_band_flash_attention'
        atten_mask_u = torch.triu(torch.ones(shape), diagonal=next_tokens + 1 + kv_total_token_num - q_total_token_num)
        atten_mask_l = torch.tril(torch.ones(shape), diagonal=-pre_tokens - 1 + kv_total_token_num - q_total_token_num)
        atten_mask = (atten_mask_u + atten_mask_l).to(torch.bool)
        atten_mask_npu = (torch.triu(torch.ones([2048, 2048]), diagonal=1)).to(torch.bool).npu()
        if sink_enable:
            atten_mask = get_sink_mask(atten_mask, shape, 4)
    else:
        print(f"invaild sparse_mode={sparse_mode}")
        sys.exit()
    print(f"sparse_mode={sparse_mode}")

    # drop
    drop_mask = None

    # q = truncated_normal(0.0, 1.2, -10, 10, (q_total_token_num, q_head_num, head_dim)).to(input_dtype)
    # k = truncated_normal(0.0, 1.5, -12, 12, (kv_total_token_num, kv_head_num, head_dim)).to(input_dtype)
    # v = truncated_normal(0.0, 0.5, -4, 4, (kv_total_token_num, kv_head_num, head_dim)).to(input_dtype)
    # dx = 2 * (torch.rand([q_total_token_num, q_head_num, head_dim]) - 0.5).to(input_dtype)
    q = 2 * (torch.rand([q_total_token_num, q_head_num, head_dim]) - 0.5).to(input_dtype)
    k = 2 * (torch.rand([kv_total_token_num, kv_head_num, head_dim]) - 0.5).to(input_dtype)
    v = 2 * (torch.rand([kv_total_token_num, kv_head_num, head_dim]) - 0.5).to(input_dtype)
    dx = 2 * (torch.rand([q_total_token_num, q_head_num, head_dim]) - 0.5).to(input_dtype)
    out_cpu_golden = torch.zeros_like(q)
    dq_cpu_golden = torch.zeros_like(q)
    dk_cpu_golden = torch.zeros_like(k)
    dv_cpu_golden = torch.zeros_like(v)
    x_max = torch.empty(0)
    x_sum = torch.empty(0)

    for i in range(batch_num):
        cur_q_seq = actual_seq_qlen[i]
        cur_kv_seq = actual_seq_kvlen[i]
        last_q_seq = cu_seqlens_q[i - 1] if i != 0 else 0
        last_kv_seq = cu_seqlens_k[i - 1] if i != 0 else 0

        if cur_q_seq == 0 or cur_kv_seq == 0:
            continue
        
        qi = q[last_q_seq : last_q_seq + cur_q_seq]
        ki = k[last_kv_seq : last_kv_seq + cur_kv_seq]
        vi = v[last_kv_seq : last_kv_seq + cur_kv_seq]

        qi = rearrange(qi, 's n d -> 1 n s d')
        ki = rearrange(ki, 's n d -> 1 n s d')
        vi = rearrange(vi, 's n d -> 1 n s d')

        if not (q_head_num == kv_head_num):
            ki = broadcastKV_sigle(q_head_num, kv_head_num, ki, ki.dtype)
            vi = broadcastKV_sigle(q_head_num, kv_head_num, vi, vi.dtype)


        if sparse_mode == 0:
            atten_maski = None
        else:
            atten_maski = atten_mask[:cur_q_seq, :cur_kv_seq]

        drop_maski = None
        outi_golden, softmax_resi, x_maxi, x_sumi = tforward(qi.to(gtype), ki.to(gtype), vi.to(gtype), 
                                                            drop_maski, atten_maski, softmax_scale, keep_prob)
        out_cpu_golden[last_q_seq : last_q_seq + cur_q_seq] = rearrange(outi_golden, '1 n s d -> s n d')

        x_maxi_tmp = x_maxi.broadcast_to(1, q_head_num, cur_q_seq, 8)
        x_maxi_tmp = x_maxi_tmp.contiguous().view(-1)
        x_sumi_tmp = x_sumi.broadcast_to(1, q_head_num, cur_q_seq, 8)
        x_sumi_tmp = x_sumi_tmp.contiguous().view(-1)
        x_max = torch.cat([x_max, x_maxi_tmp], dim=0)
        x_sum = torch.cat([x_sum, x_sumi_tmp], dim=0)

        dxi = dx[last_q_seq : last_q_seq + cur_q_seq]
        dxi = rearrange(dxi, 's n d -> 1 n s d')

        # drop_maski is the ith batch mask
        dqi_golden, dki_golden, dvi_golden = tbackward(dxi.to(gtype), qi.to(gtype), ki.to(gtype),
                                                    vi.to(gtype), softmax_resi.to(gtype),atten_maski,
                                                    drop_maski, softmax_scale, keep_prob)
        # dqi_golden, dki_golden, dvi_golden = tbackward_new(dxi.to(gtype), qi.to(gtype), ki.to(gtype), vi.to(gtype), 
        #                                                    x_maxi, x_sumi, outi_golden, 
        #                                                    drop_maski, atten_maski, softmax_scale, keep_prob)
        if not (q_head_num == kv_head_num):
            G = int(q_head_num / kv_head_num)
            s2i = cur_kv_seq
            dki_golden = torch.sum(dki_golden.reshape(1, kv_head_num, G, s2i, head_dim), dim=2, keepdim=True).reshape(1, kv_head_num, s2i, head_dim)
            dvi_golden = torch.sum(dvi_golden.reshape(1, kv_head_num, G, s2i, head_dim), dim=2, keepdim=True).reshape(1, kv_head_num, s2i, head_dim)

        # transpose golden from nsd -> snd and concat in batch seqence
        dq_cpu_golden[last_q_seq: last_q_seq + cur_q_seq] = rearrange(dqi_golden, '1 n s d -> s n d')
        dk_cpu_golden[last_kv_seq: last_kv_seq + cur_kv_seq] = rearrange(dki_golden, '1 n s d -> s n d')
        dv_cpu_golden[last_kv_seq: last_kv_seq + cur_kv_seq] = rearrange(dvi_golden, '1 n s d -> s n d')

    print("-"*8 + "finish cpu golden generate" + "-"*8)

    q = q.npu()
    k = k.npu()
    v = v.npu()
    dx = dx.npu()
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    torch.npu.synchronize()

    npu_rst = torch_npu.npu_fusion_attention(
            q, k, v, q_head_num,
            pse=None,
            padding_mask=None,
            atten_mask=atten_mask_npu,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q),
            actual_seq_kvlen=tuple(cu_seqlens_k),
            pre_tockens=pre_tokens,
            next_tockens=next_tokens,
            inner_precise=0,
            sparse_mode=sparse_mode,
            prefix=None)
    out_npu, sft_max_npu, sft_sum_npu = npu_rst[0], npu_rst[1], npu_rst[2]
    out_res = is_similar(out_npu.cpu().float(), out_cpu_golden.cpu().float(), 0.01, 0.01)
    data_compare(out_npu.detach().cpu().float().numpy(), out_cpu_golden.cpu().float().numpy(), 2**-6, 2**-6)
    data_compare(sft_max_npu.detach().cpu().float().numpy(), x_max.cpu().float().numpy(), 2**-6, 2**-6)
    data_compare(sft_sum_npu.detach().cpu().float().numpy(), x_sum.cpu().float().numpy(), 2**-6, 2**-6)
    print(out_res)
    # out_npu.backward(dx)
    # dq = q.grad
    # dk = k.grad
    # dv = v.grad

    x_max_npu = x_max.reshape(q_total_token_num, q_head_num, 8).npu()
    x_sum_npu = x_sum.reshape(q_total_token_num, q_head_num, 8).npu()
    attention_in_npu =out_cpu_golden.to(input_dtype).npu()
    torch.npu.synchronize()
    bwd_out = torch_npu.npu_fusion_attention_grad(
        q,
        k, 
        v, 
        dx, 
        q_head_num,
        input_layout = "TND",
        pse = None,
        padding_mask = None,
        atten_mask = atten_mask_npu,
        softmax_max = x_max_npu,
        softmax_sum = x_sum_npu,
        softmax_in = None,
        attention_in = attention_in_npu,
        scale_value = softmax_scale,
        keep_prob = keep_prob,
        pre_tockens = pre_tokens,
        next_tockens = next_tokens,
        inner_precise=0,
        prefix=None,
        actual_seq_qlen=tuple(cu_seqlens_q),
        actual_seq_kvlen=tuple(cu_seqlens_k),
        sparse_mode=sparse_mode
    )
    dq, dk, dv = bwd_out[0], bwd_out[1], bwd_out[2]
    torch.npu.synchronize()

    imit_ratio_forward = 0.01
    limit_ratio_backward = 0.01
    dq_res = is_similar(dq.cpu().float(), dq_cpu_golden.cpu().float(), imit_ratio_forward, limit_ratio_backward)
    dk_res = is_similar(dk.cpu().float(), dk_cpu_golden.cpu().float(), imit_ratio_forward, limit_ratio_backward)
    dv_res = is_similar(dv.cpu().float(), dv_cpu_golden.cpu().float(), 2**-6, 2**-6)
    # assert dq_res and dk_res and dv_res
    data_compare(dq.cpu().float().numpy(), dq_cpu_golden.cpu().float().numpy(), 2**-6, 2**-6)
    data_compare(dk.cpu().float().numpy(), dk_cpu_golden.cpu().float().numpy(), 2**-6, 2**-6)
    data_compare(dv.cpu().float().numpy(), dv_cpu_golden.cpu().float().numpy(), 2**-6, 2**-6)
    print(dq_res, dk_res, dv_res)

    if enable_prof:
        print("-"*8 + "start to prof" + "-"*8)
        prof_path = f"./prof/"
        os.system(f"rm -rf {prof_path}")
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=True
        )
        with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=5),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                experimental_config=experimental_config,
                with_modules=True,
        ) as prof:
            for i in range(10):
                npu_rst = torch_npu.npu_fusion_attention(
                        q, k, v, q_head_num,
                        pse=None,
                        padding_mask=None,
                        atten_mask=atten_mask_npu,
                        scale=softmax_scale,
                        keep_prob=keep_prob,
                        input_layout="TND",
                        actual_seq_qlen=tuple(cu_seqlens_q),
                        actual_seq_kvlen=tuple(cu_seqlens_k),
                        pre_tockens=pre_tokens,
                        next_tockens=next_tokens,
                        inner_precise=0,
                        sparse_mode=sparse_mode,
                        prefix=None)
                out_npu = npu_rst[0]
                out_npu.backward(dx)
                dq = q.grad
                dk = k.grad
                dv = v.grad
                prof.step()


    if is_check_deterministic:
        print("-"*8 + "start to check deterministic" + "-"*8)
        for i in range(100):
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()
            npu_rst = torch_npu.npu_fusion_attention(
                    q, k, v, q_head_num,
                    pse=None,
                    padding_mask=None,
                    atten_mask=atten_mask_npu,
                    scale=softmax_scale,
                    keep_prob=keep_prob,
                    input_layout="TND",
                    actual_seq_qlen=tuple(cu_seqlens_q),
                    actual_seq_kvlen=tuple(cu_seqlens_k),
                    pre_tockens=pre_tokens,
                    next_tockens=next_tokens,
                    inner_precise=0,
                    sparse_mode=sparse_mode,
                    prefix=None)
            out_npu = npu_rst[0]
            out_npu.backward(dx)

            dq_new = q.grad
            dk_new  = k.grad
            dv_new  = v.grad
            if not (np.array_equal(dq.cpu().float().numpy(), dq_new.cpu().float().numpy()) and
                    np.array_equal(dk.cpu().float().numpy(), dk_new.cpu().float().numpy()) and
                    np.array_equal(dv.cpu().float().numpy(), dv_new.cpu().float().numpy())):
                compare_out("cpu dq", dq_new.cpu().float(), dq_cpu_golden.cpu().float())
                compare_out("cpu dk", dk_new.cpu().float(), dk_cpu_golden.cpu().float())
                compare_out("cpu dv", dv_new.cpu().float(), dv_cpu_golden.cpu().float())
                print(f"loop={i}: th determinstic check = False")
                print(np.array_equal(dq.cpu().float().numpy(), dq_new.cpu().float().numpy()),
                      np.array_equal(dk.cpu().float().numpy(), dk_new.cpu().float().numpy()),
                      np.array_equal(dv.cpu().float().numpy(), dv_new.cpu().float().numpy()))
                sys.exit()
                break
        print(" th determinstic check pass = True")


def generate_random_increasing_sequence():
    # 1. 随机生成序列长度 1 ~ 32
    length = random.randint(1, 16)

    # 2. 第一个数可以任意选，这里选 0 ~ 2048 之间的随机数作为起点
    seq = [random.randint(0, 2048)]

    # 3. 依次生成后续元素
    for _ in range(length - 1):
        delta = random.randint(0, 1024)   # 随机步长 0 ~ 2048
        seq.append(delta)

    return seq


def get_custom_data(tmp_list):
    accumlate_list = [torch.tensor(tmp_list).tolist()]
    data_list = []
    for accumlate_array in accumlate_list:
        single_list = []
        for i in range(len(accumlate_array)):
            if i == 0:
                single_list.append(accumlate_array[i])
            else:
                single_list.append(accumlate_array[i] - accumlate_array[i - 1])
        data_list.append(single_list)
    return data_list


if __name__ == '__main__':
    torch.npu.set_device(2)

    actual_seq_qlen_data=[[1024, 512]]
    actual_seq_kvlen_data=[[1024, 512]]

    for i, x in enumerate(actual_seq_kvlen_data):
        test_flash_attention_training_v2(
            input_dtype = torch.float16,
            q_head_num = 1,
            kv_head_num = 1,
            head_dim = 128,
            keep_prob = 1.0,
            actual_seq_qlen = actual_seq_qlen_data[i],
            actual_seq_kvlen = actual_seq_kvlen_data[i],
            sparse_mode = 0,
            pre_tokens = 65536,
            next_tokens = 65536,
            case_name =  "ZJ_" + str(i),
            sink_enable = False,
            enable_prof = True,
            is_check_deterministic = True
        )
