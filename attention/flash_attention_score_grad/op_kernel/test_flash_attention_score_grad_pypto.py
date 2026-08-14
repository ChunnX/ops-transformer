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
from flash_attention_score_grad_pypto import (
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

    s_max = s.max(dim=-1, keepdim=True).values
    s_exp = torch.exp(s - s_max)
    s_sum = s_exp.sum(dim=-1, keepdim=True)
    p = s_exp / s_sum

    # Forward output y = P @ V — the kernel reads this to build softmaxGradFront
    y_f32 = torch.matmul(p, v_f32)

    dv = torch.matmul(p.transpose(-1, -2), dy_f32)
    dp = torch.matmul(dy_f32, v_f32.transpose(-1, -2))

    # Global softmaxGradFront: sfmg = rowsum(dy * y), matching ProcessVec1
    sfmg = (dy_f32 * y_f32).sum(dim=-1, keepdim=True)
    ds = p * (dp - sfmg)
    dq_pt = torch.matmul(ds, k_f32) * scale
    dk_pt = torch.matmul(ds.transpose(-1, -2), q_f32) * scale

    softmax_max = s_max.squeeze(-1)
    softmax_sum = s_sum.squeeze(-1)

    return (dq_pt.half().transpose(1, 2), dk_pt.half().transpose(1, 2),
            dv.half().transpose(1, 2), y_f32.half().transpose(1, 2),
            softmax_max, softmax_sum, sfmg.squeeze(-1))


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


def run_test():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    # Test parameters — keep S1, S2, D as multiples of 128 for simplicity
    B, N, S1, S2, D = 1, 1, 256, 256, 128
    scale = 1.0 / math.sqrt(D)
    num_cores = 1

    torch.manual_seed(42)
    q = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    dy = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)

    # Compute golden reference
    (dq_gold, dk_gold, dv_gold, y_gold, smax_gold, ssum_gold,
     sfmg_ref) = golden_flash_attention_grad(
        q.cpu(), k.cpu(), v.cpu(), dy.cpu(), scale)

    y_in = y_gold.to(device).contiguous()
    smax_gold = smax_gold.to(device).contiguous()
    ssum_gold = ssum_gold.to(device).contiguous()

    # Allocate workspaces (FP32, initialized to zero for atomic add)
    dq_ws = torch.zeros(B, S1, N, D, device=device, dtype=torch.float32)
    dk_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)
    dv_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)
    # kernel 内排布为 [S1, B*N]（store 的 order 必须升序，不能转置写）
    dbg_sfmg = torch.zeros(S1, B * N, device=device, dtype=torch.float32)

    tiling = FagTiling(
        b=B, n=N, s1=S1, s2=S2, d=D,
        scale_value=scale,
        core_num=num_cores,
    )

    # Launch kernel
    print(f"Launching kernel: B={B}, N={N}, S1={S1}, S2={S2}, D={D}, "
          f"scale={scale:.6f}, cores={num_cores}")

    # 先触发编译（不计入执行看护窗口）：编译发生在首次调用内部，
    # 故这里用 launch + 同步作为一次完整执行，并只对「执行」计时看护。
    run_kernel_with_timeout(
        lambda: flash_attention_score_grad[None, num_cores](
            q, k, v, dy, y_in,
            smax_gold, ssum_gold,
            dq_ws, dk_ws, dv_ws, dbg_sfmg,
            tiling,
        ),
        timeout_s=KERNEL_TIMEOUT_S,
    )

    # Cast workspace results to FP16
    dq_out = dq_ws.half()
    dk_out = dk_ws.half()
    dv_out = dv_ws.half()

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

    # sfmg 对比：golden sfmg = rowsum(dy*y)，形状 [B,N,S1] -> [B*N,S1]
    sfmg_gold = sfmg_ref.reshape(B * N, S1).t().contiguous()
    sfmg_got = dbg_sfmg.cpu()
    sd = (sfmg_gold - sfmg_got).abs()
    print(f"\n[debug] sfmg max_diff={sd.max().item():.6e} "
          f"mean={sd.mean().item():.6e}")
    print(f"[debug] sfmg gold[:4,0]={sfmg_gold[:4,0].tolist()}")
    print(f"[debug] sfmg got [:4,0]={sfmg_got[:4,0].tolist()}")

    # ---- dQ 误差结构诊断 ----------------------------------------------
    # dQ 是唯一失败项，且 dK/dV 已证明 dS 正确。用几个候选公式与分块视图
    # 定位错误形态：整体错 / 分 s2 块错 / 分 vector 子核错 / 转置或缩放错。
    def diag_dq():
        qc, kc, vc, dyc = (t.cpu().float() for t in (q, k, v, dy))
        # [B,N,S,D]
        qb = qc.transpose(1, 2)
        kb = kc.transpose(1, 2)
        vb = vc.transpose(1, 2)
        dyb = dyc.transpose(1, 2)
        s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
        sm = s.max(dim=-1, keepdim=True).values
        se = torch.exp(s - sm)
        p = se / se.sum(dim=-1, keepdim=True)
        yb = torch.matmul(p, vb)
        dp = torch.matmul(dyb, vb.transpose(-1, -2))
        sfmg = (dyb * yb).sum(dim=-1, keepdim=True)
        ds = p * (dp - sfmg)                       # [B,N,S1,S2]
        got = dq_ws.cpu().float().transpose(1, 2)  # -> [B,N,S1,D]

        def d(name, ref):
            e = (got - ref).abs()
            print(f"  [diag] {name:32s} max={e.max().item():.4e} "
                  f"mean={e.mean().item():.4e}")

        d("ds@k*scale (golden)", torch.matmul(ds, kb) * scale)
        d("ds@k (no scale)", torch.matmul(ds, kb))
        d("ds@k*scale^2", torch.matmul(ds, kb) * scale * scale)
        d("p@k*scale (P instead of dS)", torch.matmul(p, kb) * scale)
        d("(dp-sfmg)@k*scale (no *P)",
          torch.matmul(dp - sfmg, kb) * scale)
        d("ds@v*scale (V instead of K)", torch.matmul(ds, vb) * scale)
        d("ds^T@k*scale (left transposed)",
          torch.matmul(ds.transpose(-1, -2), kb) * scale)

        # 幅度与相关性：相关系数接近 1 说明只是缩放/系数问题，
        # 接近 0 说明读到的根本不是 dS@K 的结果。
        # L0C 被后续 matmul 冲掉的假设：dq_ws 里可能是 dK/dV 的数据。
        # 三者形状都是 [B,N,S,D]，可直接比。
        dk_ref = torch.matmul(ds.transpose(-1, -2), qb) * scale
        dv_ref = torch.matmul(p.transpose(-1, -2), dyb)
        for nm, cand in (("dK values", dk_ref), ("dV values", dv_ref)):
            e = (got - cand).abs()
            a = got.reshape(-1)
            b2 = cand.reshape(-1)
            c = ((a * b2).sum() / (a.norm() * b2.norm())).item()
            print(f"  [diag] got vs {nm:22s} max={e.max().item():.4e} "
                  f"corr={c:.6f} |cand|={b2.norm().item():.4e}")

        ref = torch.matmul(ds, kb) * scale
        gf = got.reshape(-1)
        rf = ref.reshape(-1)
        num = (gf * rf).sum()
        corr = (num / (gf.norm() * rf.norm())).item()
        print(f"  [diag] |got|={gf.norm().item():.4e}  "
              f"|ref|={rf.norm().item():.4e}  "
              f"corr={corr:.6f}  best_fit_k={(num / rf.pow(2).sum()).item():.6f}")
        # 逐 128 行(cube 块) / 逐 64 行(vector 子核) 的相关性
        g4 = got.reshape(-1, D)
        r4 = ref.reshape(-1, D)
        for i in range(0, min(S1, 256), 64):
            a = g4[i:i + 64].reshape(-1)
            b2 = r4[i:i + 64].reshape(-1)
            c = ((a * b2).sum() / (a.norm() * b2.norm())).item()
            print(f"  [diag] corr rows[{i}:{i+64}] = {c:.6f}")

        # 只累加了某一个 s2 块？逐块检查
        nt = S2 // TILE_N
        full = torch.matmul(ds, kb) * scale
        for i in range(nt):
            sl = slice(i * TILE_N, (i + 1) * TILE_N)
            part = torch.matmul(ds[..., sl], kb[:, :, sl, :]) * scale
            e = (got - part).abs()
            print(f"  [diag] only s2 tile {i:<20d} max={e.max().item():.4e} "
                  f"mean={e.mean().item():.4e}")

        # 误差在 S1 方向的分布：按 128 行(cube 块)和 64 行(vector 子核)看
        err = (got - full).abs()
        rows = err.amax(dim=-1).reshape(-1)        # [B*N*S1]
        for i in range(0, min(S1, 256), 64):
            print(f"  [diag] err rows[{i}:{i+64}] max="
                  f"{rows[i:i+64].max().item():.4e}")
        cols = err.amax(dim=-2).reshape(-1)[:D]
        print(f"  [diag] err cols[0:64] max={cols[:64].max().item():.4e} "
              f"cols[64:128] max={cols[64:].max().item():.4e}")

    diag_dq()

    print("\nResults:")
    # FP16 gradients typically need rtol=1e-2, atol=1e-2
    # For gradients, we may need looser tolerance
    rtol, atol = 5e-2, 5e-2
    ok_dq = check("dQ", dq_out, dq_gold, rtol, atol)
    ok_dk = check("dK", dk_out, dk_gold, rtol, atol)
    ok_dv = check("dV", dv_out, dv_gold, rtol, atol)

    if ok_dq and ok_dk and ok_dv:
        print("\n=== All tests PASSED ===")
        return 0
    else:
        print("\n=== Some tests FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(run_test())
