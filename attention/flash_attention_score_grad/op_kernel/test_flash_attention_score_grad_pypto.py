"""Test script for FlashAttentionScoreGrad PyPTO Pro kernel.

Computes golden reference using PyTorch and compares with kernel output.
"""

import os
import math
import sys
import threading
import time

import torch
import torch_npu

# 仅看护 NPU 上 kernel 的端到端执行耗时，不含编译与输入生成
KERNEL_TIMEOUT_S = float(os.environ.get("KERNEL_TIMEOUT_S", "120"))
KERNEL_TIMEOUT_EXIT = 124  # 与 timeout(1) 的约定一致

# Add op_kernel directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_pypto import (  # noqa: E501
    decide_swizzle,
    flash_attention_score_grad,
    FagTiling,
    TILE_M, TILE_N, TILE_D,
)


def golden_flash_attention_grad(q, k, v, dy, scale):
    """Compute dQ, dK, dV using PyTorch reference.

    Input layout: BSND [B, S, N, D]. Internally rearranged to BNSD for matmul.
    Uses FP32 for all computation.
    Per-tile dpsum matching kernel behavior (dpsum per 128-wide s2 tile).
    """
    B, S1, N, D = q.shape
    S2 = k.shape[1]
    q_f32 = q.float().transpose(1, 2)
    k_f32 = k.float().transpose(1, 2)
    v_f32 = v.float().transpose(1, 2)
    dy_f32 = dy.float().transpose(1, 2)

    s = torch.matmul(q_f32, k_f32.transpose(-1, -2)) * scale

    # LSE 约定(对标 Dao-AILab/flash-attention)：前向输出单个
    #   lse = max + log(sum(exp(s - max)))        (自然对数)
    # 反向据此一步算出归一化的 P = exp(s - lse)，行和恒为 1。
    s_max = s.max(dim=-1, keepdim=True).values
    s_exp = torch.exp(s - s_max)
    s_sum = s_exp.sum(dim=-1, keepdim=True)
    softmax_lse_t = s_max + torch.log(s_sum)
    p = torch.exp(s - softmax_lse_t)

    # Forward output y = P @ V — the kernel reads this to build softmaxGradFront
    y_f32 = torch.matmul(p, v_f32)

    dv = torch.matmul(p.transpose(-1, -2), dy_f32)
    dp = torch.matmul(dy_f32, v_f32.transpose(-1, -2))

    # Global softmaxGradFront: sfmg = rowsum(dy * y), matching ProcessVec1
    sfmg = (dy_f32 * y_f32).sum(dim=-1, keepdim=True)
    ds = p * (dp - sfmg)
    dq_pt = torch.matmul(ds, k_f32) * scale
    dk_pt = torch.matmul(ds.transpose(-1, -2), q_f32) * scale

    return (dq_pt.half().transpose(1, 2), dk_pt.half().transpose(1, 2),
            dv.half().transpose(1, 2), y_f32.half().transpose(1, 2),
            softmax_lse_t.squeeze(-1), sfmg.squeeze(-1))


def run_kernel_with_timeout(launch, timeout_s=KERNEL_TIMEOUT_S):
    """执行 kernel 并对「设备端执行」设置看护超时。

    编译发生在首次 launch 内部，无法单独触发；而 kernel 卡死时阻塞在
    torch.npu.synchronize() 这个 C 调用里，Python 信号无法打断它。
    因此这里在 launch 返回后（编译已完成、任务已下发）才起计时线程，
    只看护 synchronize 的耗时，即端到端的设备执行时间。
    超时后直接 os._exit：此时设备已处于错误态，正常退出路径会再次卡住。
    """
    launch()                      # 含编译 + 下发，不计入看护窗口
    print(f"[watchdog] kernel dispatched, waiting up to {timeout_s}s "
          f"for device execution", flush=True)

    def on_timeout():
        print(f"\n[watchdog] TIMEOUT: kernel exceeded {timeout_s}s on device "
              f"-- likely a hang (vector/cube core timeout or a missing "
              f"cross-core wait). Killing process.", flush=True)
        os._exit(KERNEL_TIMEOUT_EXIT)

    timer = threading.Timer(timeout_s, on_timeout)
    timer.daemon = True
    timer.start()
    t0 = time.time()
    try:
        torch.npu.synchronize()
    finally:
        timer.cancel()
    print(f"[watchdog] device execution finished in {time.time() - t0:.3f}s",
          flush=True)


def run_case(B, N, S1, S2, D, num_cores):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(42)
    q = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    dy = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)

    # Compute golden reference
    (dq_gold, dk_gold, dv_gold, y_gold, lse_gold,
     _sfmg_ref) = golden_flash_attention_grad(
        q.cpu(), k.cpu(), v.cpu(), dy.cpu(), scale)

    y_in = y_gold.to(device).contiguous()
    lse_gold = lse_gold.to(device).contiguous()

    # workspace 由 kernel 的 pre 阶段清零(对齐 ASC 口径)，host 不再清。
    # 故意填 NaN：若 pre 未生效，结果必为 NaN，不会被误判为通过。
    dq_ws = torch.full((B, S1, N, D), float("nan"), device=device,
                       dtype=torch.float32)
    dk_ws = torch.full((B, S2, N, D), float("nan"), device=device,
                       dtype=torch.float32)
    dv_ws = torch.full((B, S2, N, D), float("nan"), device=device,
                       dtype=torch.float32)
    # fp16 最终输出：由 kernel 的 post 阶段写(scale + cast)
    dq_out_t = torch.zeros(B, S1, N, D, device=device, dtype=torch.float16)
    dk_out_t = torch.zeros(B, S2, N, D, device=device, dtype=torch.float16)
    dv_out_t = torch.zeros(B, S2, N, D, device=device, dtype=torch.float16)
    tiling = FagTiling(
        b=B, n=N, s1=S1, s2=S2, d=D,
        scale_value=scale,
        core_num=num_cores,
        enable_swizzle=decide_swizzle(B, N, S1, S2, D, num_cores),
    )

    # Launch kernel
    print(f"Launching kernel: B={B}, N={N}, S1={S1}, S2={S2}, D={D}, "
          f"scale={scale:.6f}, cores={num_cores}")

    # 先触发编译（不计入执行看护窗口）：编译发生在首次调用内部，
    # 故这里用 launch + 同步作为一次完整执行，并只对「执行」计时看护。
    run_kernel_with_timeout(
        lambda: flash_attention_score_grad[None, num_cores](
            q, k, v, dy, y_in,
            lse_gold,
            dq_ws, dk_ws, dv_ws,
            dq_out_t, dk_out_t, dv_out_t,
            tiling,
        ),
        timeout_s=KERNEL_TIMEOUT_S,
    )

    # scale + cast 已由 kernel 的 post 阶段完成，直接取 fp16 输出
    dq_out = dq_out_t
    dk_out = dk_out_t
    dv_out = dv_out_t

    # Compare with golden
    def check(name, out, gold, rtol, atol):
        out_cpu = out.cpu()
        gold_cpu = gold.cpu()
        max_diff = (out_cpu.float() - gold_cpu.float()).abs().max().item()
        mean_diff = (out_cpu.float() - gold_cpu.float()).abs().mean().item()
        try:
            torch.testing.assert_close(out_cpu, gold_cpu, rtol=rtol, atol=atol)
            print(f"  {name}: PASS  (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
            return True
        except AssertionError as e:
            print(f"  {name}: FAIL  (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
            print(f"    {e}")
            return False


    print("\nResults:")
    # FP16 gradients typically need rtol=1e-2, atol=1e-2
    # For gradients, we may need looser tolerance
    rtol, atol = 5e-2, 5e-2
    ok_dq = check("dQ", dq_out, dq_gold, rtol, atol)
    ok_dk = check("dK", dk_out, dk_gold, rtol, atol)
    ok_dv = check("dV", dv_out, dv_gold, rtol, atol)

    return ok_dq and ok_dk and ok_dv


# 形状用例：覆盖对齐、S1 尾块、S2 尾块、双尾块以及多核
CASES = [
    # (B, N, S1, S2, D, cores)
    (1, 1, 256, 256, 128, 1),   # 全对齐基线
    (1, 1, 200, 256, 128, 1),   # S1 尾块 (200 = 128 + 72)
    (1, 1, 256, 200, 128, 1),   # S2 尾块
    (1, 1, 200, 168, 128, 1),   # 双尾块
    (2, 2, 256, 256, 128, 1),   # 多 batch / 多 head
    # --- S1 尾块二分定位 ---
    (1, 1, 384, 256, 128, 1),   # 5: S1 三整块，无尾块
    (1, 1, 192, 256, 128, 1),   # 6: S1 尾块=64(正好半核对齐)
    (1, 1, 160, 256, 128, 1),   # 7: S1 尾块=32
    (1, 1, 224, 256, 128, 1),   # 8: 尾块=96, halfS1=48
    (1, 1, 136, 256, 128, 1),   # 9: 尾块=8,  halfS1=4
    (1, 1, 168, 256, 128, 1),   # 10: 尾块=40, halfS1=20
    # --- 大 seqlength + 多核 ---
    (1, 1, 8192, 8192, 128, 36),   # 11: 36 个 cube 核
    (1, 1, 8192, 8192, 128, 32),   # 12: 32 个 cube 核
]


# ==================================================================
#  泛化用例(100 个)：B/N/S1/S2 覆盖奇数、偶数、尾块边界，核数同时
#  覆盖「不满核」(1/8/16/24/32) 与「满核」(36，Ascend950DT 36 cube)。
#  用 CASE_FILTER=G 单独运行；unset CASE_FILTER 跑全部用例。
# ==================================================================
def _gen_generalized_cases():
    # 24 个基形 × [1, 8, 24, 36] 核 = 96 个
    base = [
        (1, 1, 128, 128),           # 全对齐基线
        (1, 1, 129, 128),           # S1 尾块=1
        (1, 1, 128, 129),           # S2 尾块=1
        (1, 1, 257, 257),           # 双奇数尾块
        (3, 1, 129, 131),           # 奇数 B + 双尾块
        (1, 3, 131, 129),           # 奇数 N + 双尾块
        (3, 3, 257, 257),           # B/N 奇数 + 双尾块
        (5, 1, 513, 513),           # 大奇数尾块
        (1, 5, 385, 385),           # N 奇数 + 尾块=1
        (5, 5, 640, 640),           # 全奇数全对齐
        (3, 5, 192, 192),           # 尾块=64(半核对齐界)
        (5, 3, 400, 400),           # 尾块=16
        (7, 7, 257, 257),           # B/N=7 + 双奇数尾块
        (1, 7, 512, 512),           # N=7 对齐
        (7, 1, 768, 768),           # B=7 对齐
        (2, 2, 256, 256),           # 偶数 B/N
        (4, 4, 384, 384),           # 尾块=128
        (2, 5, 768, 768),           # 多块整张
        (5, 2, 896, 896),           # 尾块=128
        (6, 3, 1024, 1024),         # 双整块
        (1, 9, 512, 512),           # N=9 奇数对齐
        (9, 1, 768, 768),           # B=9 奇数对齐
        (2, 8, 512, 257),           # S2 奇数尾块 + 多 N
        (8, 2, 257, 512),           # S1 奇数尾块 + 多 B
    ]
    cases = [(*s, 128, c) for s in base for c in (1, 8, 24, 36)]
    # 4 个大 shape 专项：满核 / 不满核
    extra = [
        (1, 1, 8192, 8192, 128, 36),   # 满核大 shape
        (1, 1, 8192, 8192, 128, 24),   # 不满核大 shape
        (1, 1, 8192, 8192, 128, 8),    # 弱不满核大 shape
        (1, 1, 4096, 4096, 128, 16),   # 中等大 shape 不满核
    ]
    return cases + extra

GENERALIZED_CASES = _gen_generalized_cases()


def run_test():
    failed = []
    # CASE_FILTER=i[,j...] 只跑指定下标的用例（设备级错误会终止进程，
    # 逐个跑才能定位是哪个形状出问题）；CASE_FILTER=G 跑 100 个泛化用例。
    flt = os.environ.get("CASE_FILTER", "")
    if flt == "G":
        cases = GENERALIZED_CASES
    elif flt:
        cases = [CASES[int(i)] for i in flt.split(",")]
    else:
        cases = CASES
    for (B, N, S1, S2, D, cores) in cases:
        tag = f"B{B} N{N} S1={S1} S2={S2} D{D} cores={cores}"
        print(f"\n{'='*62}\n[case] {tag}\n{'='*62}")
        if not run_case(B, N, S1, S2, D, cores):
            failed.append(tag)

    print(f"\n{'='*62}")
    if failed:
        print(f"=== {len(failed)}/{len(cases)} case(s) FAILED ===")
        for t in failed:
            print(f"    FAIL: {t}")
        return 1
    print(f"=== All {len(cases)} cases PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(run_test())
