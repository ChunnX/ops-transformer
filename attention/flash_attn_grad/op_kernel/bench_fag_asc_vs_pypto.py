"""FlashAttentionScoreGrad 性能对比：AscendC(npu_fusion_attention_grad) vs pypto。

只测 S=8192、D=128、BSND、无可选输入的场景，与 pypto kernel 当前支持的
范围一致。两侧喂完全相同的输入张量。

计时口径：warmup 后跑 ITERS 次，只在最后同步一次，取平均。这样测到的是
kernel 的稳态吞吐，不含首次编译与输入构造。
"""

import math
import os
import sys
import time

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_pypto import (  # noqa: E501
    decide_swizzle,
    flash_attention_score_grad,
    FagTiling,
)

WARMUP = int(os.environ.get("WARMUP", "3"))
ITERS = int(os.environ.get("ITERS", "20"))
# 单次 launch 的看护上限；8192 单次约几 ms，给足余量
KERNEL_TIMEOUT_S = float(os.environ.get("KERNEL_TIMEOUT_S", "120"))

INT32_MAX = 2147483647


def build_inputs(B, N, S1, S2, D, device, dtype=torch.float16):
    """构造两侧共用的输入。

    softmax_max/sum 用前向的真实值（对 exp 做过 max 归约），保证两侧
    数值路径一致；这里在 fp32 上算，避免 8192 长度下 exp 溢出。
    """
    torch.manual_seed(42)
    q = torch.randn(B, S1, N, D, device=device, dtype=dtype)
    k = torch.randn(B, S2, N, D, device=device, dtype=dtype)
    v = torch.randn(B, S2, N, D, device=device, dtype=dtype)
    dy = torch.randn(B, S1, N, D, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    # BSND -> BNSD 做参考计算
    qb = q.float().transpose(1, 2)
    kb = k.float().transpose(1, 2)
    vb = v.float().transpose(1, 2)
    s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
    smax = s.max(dim=-1, keepdim=True).values            # [B,N,S1,1]
    sexp = torch.exp(s - smax)
    ssum = sexp.sum(dim=-1, keepdim=True)                # [B,N,S1,1]
    p = sexp / ssum
    y = torch.matmul(p, vb)                              # [B,N,S1,D]
    # 前向输出回到 BSND
    y_bsnd = y.transpose(1, 2).contiguous().to(dtype)
    del s, sexp, p, y
    # ASC 仍是 max/sum 口径；pypto 改用 LSE，故这里两者都返回，
    # lse = max + log(sum(exp(s-max)))，形状 [B,N,S1]
    lse = (smax + torch.log(ssum)).squeeze(-1)           # [B,N,S1]
    return q, k, v, dy, y_bsnd, smax, ssum, lse, scale


def bench(fn, warmup=WARMUP, iters=ITERS):
    """返回单次平均耗时(ms)。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def bench_sync(fn, warmup=WARMUP, iters=ITERS):
    """每次迭代都同步，测单次真实端到端延迟。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    total = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        total += time.perf_counter() - t0
    return total * 1000.0 / iters


def check_close(name, got, ref, atol=2e-2):
    g = got.cpu().float()
    r = ref.cpu().float()
    md = (g - r).abs().max().item()
    ok = md <= atol
    print(f"    {name}: {'OK ' if ok else 'MISMATCH'} max_diff={md:.4e}")
    return ok


def run(B, N, S1, S2, D, cores):
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    q, k, v, dy, y, smax, ssum, lse, scale = build_inputs(
        B, N, S1, S2, D, device)

    print(f"\n{'='*66}")
    print(f"[bench] B={B} N={N} S1={S1} S2={S2} D={D} layout=BSND fp16 "
          f"cores={cores}")
    print(f"{'='*66}")

    results = {}

    # ---------------- AscendC ----------------
    # ASC 的 softmax_max/sum 约定为 [B,N,S,8]（末维广播 8 份）
    smax_asc = smax.expand(-1, -1, -1, 8).contiguous().float()
    ssum_asc = ssum.expand(-1, -1, -1, 8).contiguous().float()

    def asc_once():
        return torch_npu.npu_fusion_attention_grad(
            q, k, v, dy, N,
            "BSND",
            pse=None,
            padding_mask=None,
            atten_mask=None,
            softmax_max=smax_asc,
            softmax_sum=ssum_asc,
            softmax_in=None,
            attention_in=y,
            scale_value=scale,
            keep_prob=1.0,
            pre_tockens=INT32_MAX,
            next_tockens=INT32_MAX,
            inner_precise=0,
            sparse_mode=0,
        )

    asc_out = None
    try:
        asc_out = asc_once()
        torch.npu.synchronize()
        results["AscendC"] = bench(asc_once)
        # 交叉核对：每次迭代都同步的口径，排除「异步未真正执行」的假象
        results["AscendC_sync"] = bench_sync(asc_once)
    except Exception as e:  # noqa: BLE001
        print(f"  AscendC: 调用失败 -> {type(e).__name__}: {str(e)[:300]}")

    # ---------------- pypto ----------------
    # pypto 改用 LSE 口径：[B,N,S1] fp32
    lse_p = lse.contiguous().float()                     # [B,N,S1]
    dq_ws = torch.zeros(B, S1, N, D, device=device, dtype=torch.float32)
    dk_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)
    dv_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)
    # fp16 最终输出：由 kernel 的 post 阶段写(scale + cast)，
    # 口径与 ASC 一致 —— pre(清零)/post(scale+cast) 都在 kernel 内。
    dq_o = torch.zeros(B, S1, N, D, device=device, dtype=torch.float16)
    dk_o = torch.zeros(B, S2, N, D, device=device, dtype=torch.float16)
    dv_o = torch.zeros(B, S2, N, D, device=device, dtype=torch.float16)
    tiling = FagTiling(
        b=B, n=N, s1=S1, s2=S2, d=D, scale_value=scale, core_num=cores,
        enable_swizzle=decide_swizzle(B, N, S1, S2, D, cores),
    )

    def pypto_once():
        flash_attention_score_grad[None, cores](
            q, k, v, dy, y, lse_p,
            dq_ws, dk_ws, dv_ws,
            dq_o, dk_o, dv_o, tiling,
        )

    # 首次调用含编译，单独跑并计时，不计入均值
    t0 = time.perf_counter()
    pypto_once()
    torch.npu.synchronize()
    print(f"  (pypto 首次调用含编译: {(time.perf_counter()-t0)*1000:.1f} ms)")
    results["pypto"] = bench(pypto_once)
    results["pypto_sync"] = bench_sync(pypto_once)

    # ---------------- 正确性交叉验证 ----------------
    # 确认两边算的是同一个东西，否则性能数字没有可比性
    if asc_out is not None:
        print("\n  [check] AscendC vs pypto 输出比对:")
        dq_a, dk_a, dv_a = asc_out[0], asc_out[1], asc_out[2]
        # workspace 填 NaN 而非 0：清零由 kernel 的 pre 阶段负责，
        # 若 pre 失效结果必为 NaN，不会被误判为通过。
        for t in (dq_ws, dk_ws, dv_ws):
            t.fill_(float("nan"))
        pypto_once()
        torch.npu.synchronize()
        # 比对 kernel 的 fp16 输出(post 阶段已做 scale+cast)，
        # 与 ASC 完全同口径
        check_close("dQ", dq_o, dq_a)
        check_close("dK", dk_o, dk_a)
        check_close("dV", dv_o, dv_a)

    # ---------------- 汇总 ----------------
    print(f"\n  {'实现':<22}{'单次耗时(ms)':>14}")
    print(f"  {'-'*36}")
    for name in ("AscendC", "AscendC_sync", "pypto", "pypto_sync"):
        if name in results:
            print(f"  {name:<22}{results[name]:>14.3f}")
    if "AscendC" in results and "pypto" in results:
        a, pv = results["AscendC"], results["pypto"]
        print(f"\n  pypto / AscendC = {pv/a:.2f}x "
              f"({'pypto 更快' if pv < a else 'AscendC 更快'})")
    # FAG 主体是 5 个 [S1,S2]x[.,D] 量级的 matmul
    gflop = 5 * 2 * B * N * S1 * S2 * D / 1e9
    print(f"\n  计算量 ~{gflop:.1f} GFLOP -> 等效算力:")
    for name in ("AscendC", "pypto"):
        if name in results:
            print(f"    {name:<20}{gflop/results[name]:>8.1f} TFLOPS")
    # 910D(DAV_3510) fp16 峰值接近 500 TFLOPS
    print(f"    (参考: 910D fp16 峰值约 500 TFLOPS)")

    print("\n  口径说明：两侧均在 kernel 内完成 pre(workspace 清零)"
          "与 post(dQ/dK scale + cast fp16)。")
    print("  注意 wall-clock 含 pypto 约 3ms 固定 launch 开销，"
          "性能结论一律以 msprof device kernel time 为准。")
    return results


def main():
    for cores in (int(x) for x in os.environ.get("CORES", "36,32").split(",")):
        run(1, 1, 8192, 8192, 128, cores)
    return 0


if __name__ == "__main__":
    sys.exit(main())
