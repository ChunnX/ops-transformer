#!/usr/bin/env python3
# ============================================================
# perf_aclnn_path.py — aclnn 路径性能测试（最小可跑 / 方案 B）
#
# 背景：
#   sageattn_fp8/aclnn_pv/op_host/sageattn_pv.cpp 是骨架
#   （host 校验 + tiling 打印 + kernel 占位，未接真 kernel）。
#   本脚本定义 Python 端 aclnn 入口 aclnn_sageattn_pv_py()，
#   参数映射对齐 C++ SageAttnPvParams，内部用 Plan B host 链
#   （INT8 QK^T -> softmax -> FP8 PV，fp32 matmul 近似）作
#   语义对齐兜底计算，不依赖 sageattention2 Python 包。
#
# 基线：aclnnFusedInferAttentionScoreV5（aclnn 路径）
#   若 torch_npu 未直接暴露 V5，回退 npu_fused_infer_attention_score
#   并在结果中标注（同接口族，950PR 性能基线）。
#
# 待测：aclnn_sageattn_pv_py（本算子 aclnn 入口 Python 兜底）
#   数字 = PyTorch 路径（Plan B host 链），诚实标注。
#
# 门槛：Speedup = T_V5 / T_aclnn >= 1.6×
# 用法：python3 perf_aclnn_path.py 2>&1 | tee /workspace/perf_aclnn.txt
# ============================================================
import torch
import torch_npu
import time
import sys

# P-01~P-07 shape 定义 (B, H, S, D) —— 与 perf_benchmark.py 对齐
SHAPES = [
    ("P-01", 1, 32, 4096,  128, False, torch.float16),
    ("P-02", 1, 32, 8192,  128, False, torch.float16),
    ("P-03", 1, 32, 8192,  128, True,  torch.float16),
    ("P-04", 2, 32, 8192,  128, False, torch.float16),
    ("P-05", 1, 32, 4096,  128, False, torch.float16),  # GQA Hq=32/Hk=8
    ("P-06", 1, 32, 4096,  128, False, torch.bfloat16),
    ("P-07", 1, 32, 16384, 128, False, torch.float16),
]

WARMUP = 5
REPEAT = 20


def aclnn_sageattn_pv_py(q, k, v, is_causal=False):
    """本算子 aclnn 入口 Python 兜底（对齐 C++ SageAttnPvParams 语义）。

    C++ aclnn_sageattn_pv 当前为骨架（host 校验 + tiling，kernel 占位），
    未接真 kernel。本函数以 Plan B host 链作语义对齐兜底，使 aclnn
    路径有数可报。数字等同 PyTorch 路径（同一 host 链）。
    Plan B: INT8 QK^T -> softmax -> FP8 PV（fp32 matmul 近似量化路径）
    """
    b, h, s, d = q.shape
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    # GQA：kv 头数少于 q 时按 Hq/Hk 广播（对齐 sageattn 适配层语义）
    hkv = k_f.shape[1]
    if hkv != h:
        rep = h // hkv
        k_f = torch.repeat_interleave(k_f, rep, dim=1)
        v_f = torch.repeat_interleave(v_f, rep, dim=1)
    # MM1: INT8 QK^T (fp32 matmul 近似)
    s_mat = torch.matmul(q_f, k_f.transpose(-2, -1))
    if is_causal:
        mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        s_mat = s_mat.masked_fill(mask, float('-inf'))
    # softmax
    p = torch.softmax(s_mat, dim=-1)
    # MM2: FP8 PV (fp32 matmul 近似)
    o = torch.matmul(p, v_f)
    return o.to(q.dtype)


def _resolve_v5():
    """探测 torch_npu 是否直接暴露 aclnnFusedInferAttentionScoreV5。"""
    try:
        from torch_npu import aclnnFusedInferAttentionScoreV5  # noqa
        return ("aclnnFusedInferAttentionScoreV5_top", aclnnFusedInferAttentionScoreV5)
    except Exception:
        pass
    try:
        from torch_npu._npu import aclnnFusedInferAttentionScoreV5  # noqa
        return ("aclnnFusedInferAttentionScoreV5_npu", aclnnFusedInferAttentionScoreV5)
    except Exception:
        pass
    return (None, None)


def bench_v5_fia(b, h, s, d, causal, dtype, is_gqa=False):
    """V5 基线：优先 aclnnFusedInferAttentionScoreV5；不可用回退 npu_fused_infer_attention_score。"""
    hk = h // 4 if is_gqa else h
    q = torch.randn(b, h, s, d, dtype=dtype).npu()
    k = torch.randn(b, hk, s, d, dtype=dtype).npu()
    v = torch.randn(b, hk, s, d, dtype=dtype).npu()

    v5_name, v5_fn = _resolve_v5()
    if v5_fn is not None:
        for _ in range(WARMUP):
            v5_fn(q, k, v)
        torch.npu.synchronize()
        times = []
        for _ in range(REPEAT):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            v5_fn(q, k, v)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        return min(times), sum(times) / len(times), v5_name

    kwargs = dict(input_layout="BNSD", num_heads=h, num_key_value_heads=hk, scale=1.0)
    if causal:
        kwargs["pre_tokens"] = s
        kwargs["next_tokens"] = 0
    for _ in range(WARMUP):
        torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)
    torch.npu.synchronize()
    times = []
    for _ in range(REPEAT):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return min(times), sum(times) / len(times), "npu_fused_infer_attention_score(fallback)"


def bench_aclnn_sage(b, h, s, d, causal, dtype, is_gqa=False):
    """本算子 aclnn 路径：aclnn_sageattn_pv_py（Python 兜底，Plan B host 链）。"""
    hk = h // 4 if is_gqa else h
    q = torch.randn(b, h, s, d, dtype=dtype).npu()
    k = torch.randn(b, hk, s, d, dtype=dtype).npu()
    v = torch.randn(b, hk, s, d, dtype=dtype).npu()

    for _ in range(WARMUP):
        aclnn_sageattn_pv_py(q, k, v, is_causal=causal)
    torch.npu.synchronize()
    times = []
    for _ in range(REPEAT):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        aclnn_sageattn_pv_py(q, k, v, is_causal=causal)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return min(times), sum(times) / len(times)


def main():
    print("=" * 80)
    print("aclnn 路径性能测试 (CANN 9.2.0-beta.1 / Atlas 950PR)")
    print("基线: aclnnFusedInferAttentionScoreV5 (V5)")
    print("待测: aclnn_sageattn_pv_py (本算子 aclnn 入口 Python 兜底)")
    print("门槛: Speedup >= 1.6x")
    print(f"WARMUP={WARMUP} REPEAT={REPEAT}")
    print("=" * 80)

    v5_name, _ = _resolve_v5()
    print(f"[probe] V5 direct import: {v5_name or 'FAIL -> fallback npu_fused_infer_attention_score'}")

    results = []
    all_pass = True

    for item in SHAPES:
        name, b, h, s, d, causal, dtype = item
        is_gqa = (name == "P-05")
        dt_name = "BF16" if dtype == torch.bfloat16 else "FP16"
        cfg = f"B={b} H={h} S={s} D={d} {'causal' if causal else ''} {dt_name}{' GQA' if is_gqa else ''}"
        print(f"\n--- {name}: {cfg} ---")

        try:
            v5_min, v5_avg, v5_src = bench_v5_fia(b, h, s, d, causal, dtype, is_gqa)
            print(f"  V5:   min={v5_min:.2f}ms avg={v5_avg:.2f}ms [{v5_src}]")
        except Exception as e:
            print(f"  V5:   FAIL ({e})")
            results.append((name, cfg, None, None, None, False, str(e)))
            all_pass = False
            continue

        try:
            sage_min, sage_avg = bench_aclnn_sage(b, h, s, d, causal, dtype, is_gqa)
            print(f"  ACLNN:min={sage_min:.2f}ms avg={sage_avg:.2f}ms [aclnn_sageattn_pv_py]")
        except Exception as e:
            print(f"  ACLNN:FAIL ({e})")
            results.append((name, cfg, v5_avg, None, None, False, str(e)))
            all_pass = False
            continue

        speedup = v5_avg / sage_avg if sage_avg > 0 else 0
        passed = speedup >= 1.6
        status = "PASS" if passed else "FAIL"
        print(f"  Speedup = {speedup:.2f}x  [{status}] (门槛 1.6x)")
        if not passed:
            all_pass = False
        results.append((name, cfg, v5_avg, sage_avg, speedup, passed, ""))

    # 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"{'编号':<6} {'配置':<45} {'V5(ms)':<10} {'ACLNN(ms)':<10} {'Speedup':<10} {'判定':<6}")
    print("-" * 80)
    for name, cfg, v5, sage, sp, ok, err in results:
        if v5 is not None and sage is not None:
            print(f"{name:<6} {cfg:<45} {v5:<10.2f} {sage:<10.2f} {sp:<10.2f} {'OK' if ok else 'X':<6}")
        else:
            print(f"{name:<6} {cfg:<45} {'ERR':<10} {'ERR':<10} {'N/A':<10} {'X':<6} {err[:20]}")

    passed_count = sum(1 for r in results if r[5])
    print(f"\n通过: {passed_count}/7")
    print(f"总体: {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    print("\n注意:")
    print("  - V5 基线: 优先 aclnnFusedInferAttentionScoreV5; 不可用回退 npu_fused_infer_attention_score")
    print("  - aclnn 路径: aclnn_sageattn_pv_py 为 Python 兜底入口 (C++ aclnn kernel 未接真 kernel)")
    print("    数字等同 PyTorch 路径 (Plan B host 链), 诚实标注为兜底")


if __name__ == "__main__":
    main()
