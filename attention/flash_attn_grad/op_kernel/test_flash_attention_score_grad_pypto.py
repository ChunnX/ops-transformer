"""Test script for FlashAttentionScoreGrad PyPTO Pro kernel.

Computes golden reference using PyTorch and compares with kernel output.
"""

import os
import math
import sys
import threading
import time

import pypto_pro.language as pl
import torch
import torch_npu

# 仅看护 NPU 上 kernel 的端到端执行耗时，不含编译与输入生成
KERNEL_TIMEOUT_S = float(os.environ.get("KERNEL_TIMEOUT_S", "120"))
KERNEL_TIMEOUT_EXIT = 124  # 与 timeout(1) 的约定一致

# Add op_kernel directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attn_grad import (  # noqa: E501
    decide_swizzle,
    flash_attn_grad,
    FlashAttnGradTilingData,
    TILE_M, TILE_N, TILE_D,
)


def golden_flash_attention_grad(q, k, v, dy, scale):
    """Compute dQ, dK, dV using PyTorch reference.

    Input layout: BSND [B, S, N, D]. Internally rearranged to BNSD for matmul.
    Uses FP32 for all computation.
    Per-tile dpsum matching kernel behavior (dpsum per 128-wide s2 tile).

    GQA: q/dy 有 N1 个 head，k/v 有 N2 个，N1 = N2*G。计算时把 k/v 沿 head
    维展开 G 份(每个 KV head 服务 G 个 Q head)，最后把 dk/dv 按 G 求和折回
    N2 —— 这正对应 kernel 侧 G 个 Q head 对同一 KV head 做 atomicAdd。
    返回的 dq/lse 是 N1 个 head，dk/dv 是 N2 个。
    """
    # v/dy 的最后一维是 Dv，可以短于 q/k 的 D —— 下面的矩阵乘天然分离这两个
    # 宽度(P@V 与 dO@V^T 只涉及 Dv，dS@K 与 dS^T@Q 只涉及 D)，故无需特殊处理。
    B, S1, N1, D = q.shape
    S2, N2 = k.shape[1], k.shape[2]
    G = N1 // N2
    q_f32 = q.float().transpose(1, 2)
    dy_f32 = dy.float().transpose(1, 2)
    # [B,N2,S2,D] -> [B,N1,S2,D]，沿 head 维每份重复 G 次(与 n2*G+g 的
    # 编号一致：head n1 对应 KV head n1//G)
    k_f32 = k.float().transpose(1, 2).repeat_interleave(G, dim=1)
    v_f32 = v.float().transpose(1, 2).repeat_interleave(G, dim=1)

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

    # dk/dv 目前是 N1 个 head；GQA 下把服务同一 KV head 的 G 份求和折回 N2。
    # G=1 时 reshape+sum 是恒等操作，故无需分支。
    dk_pt = dk_pt.reshape(B, N2, G, S2, D).sum(dim=2)
    dv = dv.reshape(B, N2, G, S2, dv.shape[-1]).sum(dim=2)

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


def run_case(B, N, S1, S2, D, num_cores, G=1, Dv=None):
    """N 是 KV 的 head 数(N2)；Q 侧为 N1 = N*G。G=1 即普通 MHA。

    Dv 是 V/dO/attn_out/dv 的 head dim，缺省等于 D。Dv < D 即不等长 headdim
    (如 D=192/Dv=128)：Q/K/dq/dk 用 D，V/dO/y/dv 用 Dv。
    """
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    if Dv is None:
        Dv = D
    if Dv > D:
        raise SystemExit(f"Dv({Dv}) 不能大于 D({D})")
    # scale 按 Q/K 的 D 算(与 host 的缺省一致)
    scale = 1.0 / math.sqrt(D)
    N2 = N
    N1 = N2 * G

    torch.manual_seed(42)
    q = torch.randn(B, S1, N1, D, device=device, dtype=torch.float16)
    k = torch.randn(B, S2, N2, D, device=device, dtype=torch.float16)
    v = torch.randn(B, S2, N2, Dv, device=device, dtype=torch.float16)
    dy = torch.randn(B, S1, N1, Dv, device=device, dtype=torch.float16)

    # Compute golden reference
    (dq_gold, dk_gold, dv_gold, y_gold, lse_gold,
     _sfmg_ref) = golden_flash_attention_grad(
        q.cpu(), k.cpu(), v.cpu(), dy.cpu(), scale)

    y_in = y_gold.to(device).contiguous()
    lse_gold = lse_gold.to(device).contiguous()

    # ---- 布局 ----
    # golden 始终按 BSND 计算；BNSD 只是把喂给 kernel 的张量 permute 成
    # [B,N,S,D]，再把 kernel 的输出 permute 回 BSND 做比对，故比对逻辑与
    # 布局无关。lse 恒为 [B,N,S1]，不参与 permute。
    layout = os.environ.get("FAG_LAYOUT", "BSND").upper()
    if layout not in ("BSND", "BNSD"):
        raise SystemExit(f"FAG_LAYOUT 只支持 BSND / BNSD，收到 {layout}")
    is_bnsd = layout == "BNSD"

    def to_dev_layout(x):
        """BSND [B,S,N,D] -> 目标布局的连续张量"""
        return x.permute(0, 2, 1, 3).contiguous() if is_bnsd else x.contiguous()

    def from_dev_layout(x):
        """目标布局 -> BSND，用于与 golden 比对"""
        return x.permute(0, 2, 1, 3).contiguous() if is_bnsd else x

    q = to_dev_layout(q)
    k = to_dev_layout(k)
    v = to_dev_layout(v)
    dy = to_dev_layout(dy)
    y_in = to_dev_layout(y_in)

    # 单块 workspace，kernel 内切成 dq/dk/dv 三段 fp32（见 make_ptr 处）。
    # 由 kernel 的 pre 阶段清零(对齐 ASC 口径)，host 不再清。
    # 故意填 NaN：若 pre 未生效，结果必为 NaN，不会被误判为通过。
    # dq/dk 段是 D 宽，dv 段是 Dv 宽 —— 与 host 的 dqElems/dkElems/dvElems 一致
    ws_elems = (B * S1 * N1 * D) + (B * S2 * N2 * D) + (B * S2 * N2 * Dv)
    workspace = torch.full((ws_elems,), float("nan"), device=device,
                           dtype=torch.float32)
    # fp16 最终输出：由 kernel 的 post 阶段写(scale + cast)
    # 输出张量按目标布局分配（dq/dk/dv 的布局跟随 q/k/v）
    # dq/dk 的 head dim 是 D，dv 的是 Dv
    _oq = (B, N1, S1, D) if is_bnsd else (B, S1, N1, D)
    _okv = (B, N2, S2, D) if is_bnsd else (B, S2, N2, D)
    _okvv = (B, N2, S2, Dv) if is_bnsd else (B, S2, N2, Dv)
    dq_out_t = torch.zeros(*_oq, device=device, dtype=torch.float16)
    dk_out_t = torch.zeros(*_okv, device=device, dtype=torch.float16)
    dv_out_t = torch.zeros(*_okvv, device=device, dtype=torch.float16)
    # 视图参数与 host 的 TilingFunc 一致：两种布局统一成 [viewD0,S,viewD2,D]
    tiling = FlashAttnGradTilingData(
        b=B, s1=S1, s2=S2, n1=N1, n2=N2, d=D, dv=Dv, scaleValue=scale,
        gSize=G,
        viewD0Q=(B * N1) if is_bnsd else B,
        viewD2Q=1 if is_bnsd else N1,
        coefB0Q=N1 if is_bnsd else 1,
        viewD0KV=(B * N2) if is_bnsd else B,
        viewD2KV=1 if is_bnsd else N2,
        coefB0KV=N2 if is_bnsd else 1,
        coefN0=1 if is_bnsd else 0,
        coefN2=0 if is_bnsd else 1,
    )
    # swizzle 由 tilingkey 区分(每种分核各编一份二进制)，不再走 tiling 字段。
    # 判据与 host 的 TilingFunc 保持一致。
    swizzle = decide_swizzle(B, N1, S1, S2, D, num_cores)
    # FAG_FORCE_SWIZZLE 用于覆盖判据，单独验证另一条编译产物的正确性 ——
    # 两种分核对同一输入必须给出相同结果(仅分块顺序不同)。
    _force = os.environ.get("FAG_FORCE_SWIZZLE")
    if _force is not None:
        swizzle = int(_force)

    # D/Dv 的分配宽度分档，与 host 的 dAlign/dvAlign 判定一致
    d_align = 128 if D <= 128 else 192
    dv_align = 128 if Dv <= 128 else 192

    # Launch kernel
    print(f"Launching kernel: B={B}, N1={N1}, N2={N2}, G={G}, S1={S1}, "
          f"S2={S2}, D={D}, Dv={Dv}, d_align={d_align}, "
          f"dv_align={dv_align}, "
          f"scale={scale:.6f}, cores={num_cores}, "
          f"layout={layout}, swizzle={swizzle}")

    # 先触发编译（不计入执行看护窗口）：编译发生在首次调用内部，
    # 故这里用 launch + 同步作为一次完整执行，并只对「执行」计时看护。
    run_kernel_with_timeout(
        lambda: flash_attn_grad[
            None, num_cores,
            # tilingkey：swizzle 与 d_align 走这里(各编一份二进制)。
            # d_align 是 D 的分配宽度分档，取值即实际列数。
            {"template": 0, "layout": 0,
             "has_atten_mask": 0, "swizzle": swizzle,
             "d_align": d_align, "dv_align": dv_align},
            # datatype：对应 @pl.jit 的 datatype={"q": "input_dtype"}
            {"q": pl.DT_FP16},
        ](
            q, k, v, dy, y_in, lse_gold,
            # 6 个未使用的可选输入 + metadata：位置必须占齐
            None, None, None, None, None, None, None,
            dq_out_t, dk_out_t, dv_out_t,
            workspace,
            tiling,
        ),
        timeout_s=KERNEL_TIMEOUT_S,
    )

    # scale + cast 已由 kernel 的 post 阶段完成，直接取 fp16 输出。
    # BNSD 时 permute 回 BSND，使下面的比对与布局无关。
    dq_out = from_dev_layout(dq_out_t)
    dk_out = from_dev_layout(dk_out_t)
    dv_out = from_dev_layout(dv_out_t)

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
        finally:
            # FAG_DIFF_COLS=1 打印按 headdim 列聚合的误差分布。D 分块的
            # bug 会在块边界(96/128)上留下明显台阶，比单个 max_diff 好定位。
            if os.environ.get("FAG_DIFF_COLS"):
                d = (out_cpu.float() - gold_cpu.float()).abs()
                per_col = d.reshape(-1, d.shape[-1]).max(dim=0).values
                print(f"    [{name}] per-column max_diff:")
                for c0 in range(0, per_col.numel(), 32):
                    seg = per_col[c0:c0 + 32]
                    print(f"      col {c0:3d}-{c0 + seg.numel() - 1:3d}: "
                          f"max={seg.max().item():.4e} mean={seg.mean().item():.4e}")


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

# GQA 用例：N 列是 KV 的 head 数(N2)，末列为 G，Q 侧 head 数 N1 = N2*G。
# 覆盖 G 为 2 的幂与奇数、N2=1(所有 Q head 共享单个 KV head)、
# 尾块与 G 的组合、以及多核大 shape。
# (B, N2, S1, S2, D, cores, G)
GQA_CASES = [
    (1, 1, 256, 256, 128, 1, 2),    # 0: 最小 GQA，N1=2 N2=1
    (1, 1, 256, 256, 128, 1, 8),    # 1: G=8，单 KV head
    (1, 2, 256, 256, 128, 1, 2),    # 2: N1=4 N2=2
    (1, 2, 256, 256, 128, 1, 3),    # 3: G 为奇数，N1=6 N2=2
    (2, 3, 256, 256, 128, 1, 2),    # 4: 多 batch + N2 奇数
    (1, 1, 200, 168, 128, 1, 4),    # 5: 双尾块 + G=4
    (1, 2, 200, 256, 128, 1, 3),    # 6: S1 尾块 + G 奇数
    (1, 2, 256, 200, 128, 1, 2),    # 7: S2 尾块 + G=2
    (2, 2, 384, 256, 128, 4),       # 8: 多核(G 默认 1，作对照)
    (1, 2, 1024, 1024, 128, 8, 4),  # 9: 多核 + G=4
    (1, 1, 8192, 8192, 128, 36, 4), # 10: 大 shape 36 核 + G=4
]


# 不等长 headdim 用例：末两列是 (G, Dv)。Q/K/dq/dk 用 D，V/dO/y/dv 用 Dv。
# 实际只有两种组合需要覆盖：不等长时恒为 D=192/Dv=128（需求场景），
# 等长时为 D=Dv=192。其余 Dv 取值（96/160/64）不是真实配置，不再泛化。
# (B, N2, S1, S2, D, cores, G, Dv)
DV_CASES = [
    # 单块(S1=S2=128)：最少变量的 192 档基线，用于把「分块算错」与
    # 「跨 task 状态错」两类问题分开定位。
    (1, 1, 128, 128, 192, 1, 1, 192),   # 0: 单块 + 等长 192
    (1, 1, 128, 128, 192, 1, 1, 128),   # 1: 单块 + D=192/Dv=128
    (1, 1, 256, 256, 192, 1, 1, 192),   # 2: 多块 + 等长 192
    (1, 1, 256, 256, 192, 1, 1, 128),   # 3: 多块 + 主场景
    (1, 1, 200, 256, 192, 1, 1, 128),   # 4: S1 尾块
    (1, 1, 256, 200, 192, 1, 1, 128),   # 5: S2 尾块
    (1, 1, 200, 168, 192, 1, 1, 128),   # 6: 双尾块
    (2, 2, 256, 256, 192, 1, 1, 128),   # 7: 多 batch / 多 head
    (1, 1, 256, 256, 192, 1, 4, 128),   # 8: GQA G=4
    (1, 2, 200, 168, 192, 1, 3, 128),   # 9: 双尾块 + G 奇数
    (1, 2, 1024, 1024, 192, 8, 1, 128), # 10: 多核
    (1, 1, 4096, 4096, 192, 36, 1, 128),# 11: 大 shape 满核
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
    # 也支持 "NAME:i,j" 只跑某个命名用例集里的指定下标，
    # 例如 CASE_FILTER=DV:0,3（设备级错误会终止进程，逐个跑才能定位形状）。
    flt = os.environ.get("CASE_FILTER", "")
    _named = {"G": GENERALIZED_CASES, "GQA": GQA_CASES, "DV": DV_CASES}
    if ":" in flt:
        name, _, idxs = flt.partition(":")
        pool = _named.get(name, CASES)
        cases = [pool[int(i)] for i in idxs.split(",")]
    elif flt in _named:
        cases = _named[flt]
    elif flt:
        cases = [CASES[int(i)] for i in flt.split(",")]
    else:
        cases = CASES
    for case in cases:
        # 用例元组可带可选的第 7 项 G(缺省 1) 与第 8 项 Dv(缺省 = D)
        B, N, S1, S2, D, cores = case[:6]
        G = case[6] if len(case) > 6 else 1
        Dv = case[7] if len(case) > 7 else D
        tag = (f"B{B} N2={N} G={G} N1={N * G} S1={S1} S2={S2} D{D}"
               + (f" Dv{Dv}" if Dv != D else "")
               + f" cores={cores}")
        print(f"\n{'='*62}\n[case] {tag}\n{'='*62}")
        if not run_case(B, N, S1, S2, D, cores, G, Dv):
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
