#!/usr/bin/env python3
# ============================================================
# perf_benchmark.py — P-01~P-07 性能门禁测试
# 基准：torch_npu.npu_fused_infer_attention_score (FIA)
# 待测：Plan B host 侧分步链 (mm1 INT8 QK^T → softmax → mm2 FP8 PV)
# 门槛：Speedup = T_FIA / T_SAGE >= 1.6×
# 用法：python3 perf_benchmark.py 2>&1 | tee /workspace/perf_benchmark.txt
# ============================================================
import torch
import torch_npu
import time
import sys

# P-01~P-07 shape 定义 (B, H, S, D)
SHAPES = [
    ("P-01", 1, 32, 4096, 128, False, torch.float16),   # fp8_asc, fp32+fp16
    ("P-02", 1, 32, 8192, 128, False, torch.float16),   # fp8_asc, fp32+fp16
    ("P-03", 1, 32, 8192, 128, True,  torch.float16),   # causal
    ("P-04", 2, 32, 8192, 128, False, torch.float16),   # batch=2
    ("P-05", 1, 32, 4096, 128, False, torch.float16),   # GQA (Hq=32,Hk=8 单独处理)
    ("P-06", 1, 32, 4096, 128, False, torch.bfloat16), # BF16
    ("P-07", 1, 32, 16384, 128, False, torch.float16),  # long seq
]

WARMUP = 5
REPEAT = 20

def bench_fia(b, h, s, d, causal, dtype, is_gqa=False):
    """FIA 基准：torch_npu.npu_fused_infer_attention_score
    schema: input_layout="BNSD", causal 用 pre_tokens/next_tokens。
    不用 out= 参数（类型不匹配），直接用返回值。
    """
    hk = h // 4 if is_gqa else h
    # BNSD layout
    q = torch.randn(b, h, s, d, dtype=dtype).npu()
    k = torch.randn(b, hk, s, d, dtype=dtype).npu()
    v = torch.randn(b, hk, s, d, dtype=dtype).npu()

    kwargs = dict(input_layout="BNSD", num_heads=h, num_key_value_heads=hk, scale=1.0)
    if causal:
        kwargs["pre_tokens"] = s
        kwargs["next_tokens"] = 0

    # warmup
    for _ in range(WARMUP):
        torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)
    torch.npu.synchronize()

    # timed
    times = []
    for _ in range(REPEAT):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return min(times), sum(times) / len(times)


def bench_sage_planb(b, h, s, d, causal, dtype, is_gqa=False):
    """Plan B: host 侧分步链（简化版——用 torch_npu matmul 模拟 MM1+MM2 + host softmax）
    注意：真正的 CATLASS kernel 计时需通过 sageattn_chain 二进制（aclrtStream 计时）。
    这里用 PyTorch 等价路径做粗估：INT8 QK^T → softmax → FP8 PV
    """
    hk = h // 4 if is_gqa else h
    # 模拟量化：Q/K 量化到 int8，V 量化到 fp8
    q_f = torch.randn(b, h, s, d, dtype=torch.float32).npu()
    k_f = torch.randn(b, hk, s, d, dtype=torch.float32).npu()
    v_f = torch.randn(b, hk, s, d, dtype=torch.float32).npu()

    # warmup
    for _ in range(WARMUP):
        # MM1: INT8 QK^T (用 fp32 matmul 近似 int8 量化路径)
        s_mat = torch.matmul(q_f, k_f.transpose(-2, -1))  # [b,h,s,s]
        # scale
        if causal:
            mask = torch.triu(torch.ones(s, s, device="npu", dtype=torch.bool), diagonal=1)
            s_mat = s_mat.masked_fill(mask, float('-inf'))
        # softmax
        p = torch.softmax(s_mat, dim=-1)
        # MM2: FP8 PV
        o = torch.matmul(p, v_f)  # [b,h,s,d]

    torch.npu.synchronize()

    # timed
    times = []
    for _ in range(REPEAT):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        s_mat = torch.matmul(q_f, k_f.transpose(-2, -1))
        if causal:
            mask = torch.triu(torch.ones(s, s, device="npu", dtype=torch.bool), diagonal=1)
            s_mat = s_mat.masked_fill(mask, float('-inf'))
        p = torch.softmax(s_mat, dim=-1)
        o = torch.matmul(p, v_f)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return min(times), sum(times) / len(times)


def main():
    print("=" * 80)
    print("P-01~P-07 性能门禁测试 (CANN 9.2.0-beta.1 / Atlas 950PR)")
    print(f"基准: torch_npu.npu_fused_infer_attention_score (FIA)")
    print(f"待测: Plan B host 链 (mm1 INT8 QK^T -> softmax -> mm2 FP8 PV)")
    print(f"门槛: Speedup >= 1.6x")
    print(f"WARMUP={WARMUP} REPEAT={REPEAT}")
    print("=" * 80)

    results = []
    all_pass = True

    for item in SHAPES:
        name, b, h, s, d, causal, dtype = item
        is_gqa = (name == "P-05")
        dt_name = "BF16" if dtype == torch.bfloat16 else "FP16"
        cfg = f"B={b} H={h} S={s} D={d} {'causal' if causal else ''} {dt_name}{' GQA' if is_gqa else ''}"
        print(f"\n--- {name}: {cfg} ---")

        try:
            fia_min, fia_avg = bench_fia(b, h, s, d, causal, dtype, is_gqa)
            print(f"  FIA:  min={fia_min:.2f}ms avg={fia_avg:.2f}ms")
        except Exception as e:
            print(f"  FIA:  FAIL ({e})")
            results.append((name, cfg, None, None, None, False, str(e)))
            all_pass = False
            continue

        try:
            sage_min, sage_avg = bench_sage_planb(b, h, s, d, causal, dtype, is_gqa)
            print(f"  SAGE: min={sage_min:.2f}ms avg={sage_avg:.2f}ms")
        except Exception as e:
            print(f"  SAGE: FAIL ({e})")
            results.append((name, cfg, fia_avg, None, None, False, str(e)))
            all_pass = False
            continue

        speedup = fia_avg / sage_avg if sage_avg > 0 else 0
        passed = speedup >= 1.6
        status = "PASS" if passed else "FAIL"
        print(f"  Speedup = {speedup:.2f}x  [{status}] (门槛 1.6x)")
        if not passed:
            all_pass = False
        results.append((name, cfg, fia_avg, sage_avg, speedup, passed, ""))

    # 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"{'编号':<6} {'配置':<45} {'FIA(ms)':<10} {'SAGE(ms)':<10} {'Speedup':<10} {'判定':<6}")
    print("-" * 80)
    for name, cfg, fia, sage, sp, ok, err in results:
        if fia is not None and sage is not None:
            print(f"{name:<6} {cfg:<45} {fia:<10.2f} {sage:<10.2f} {sp:<10.2f} {'✅' if ok else '❌':<6}")
        else:
            print(f"{name:<6} {cfg:<45} {'ERR':<10} {'ERR':<10} {'N/A':<10} {'❌':<6} {err[:20]}")

    passed_count = sum(1 for r in results if r[5])
    print(f"\n通过: {passed_count}/7")
    print(f"总体: {'✅ ALL PASS' if all_pass else '❌ NOT ALL PASS'}")
    print(f"\n注意: SAGE 路径为 PyTorch 等价近似（fp32 matmul+softmax），非真正 CATLASS kernel。")
    print(f"真正 kernel 计时需通过 sageattn_chain 二进制（aclrtStream 计时），")
    print(f"但该路径受 AIC 断链限制（device-side 串联不可用）。")
    print(f"此结果仅作 Plan B 路径性能水位参考。")


if __name__ == "__main__":
    main()
