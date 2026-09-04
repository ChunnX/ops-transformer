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
    decide_bn2_multiblk,
    decide_d_align,
    decide_dv_align,
    decide_swizzle,
    decide_template,
    flash_attn_grad,
    FlashAttnGradTilingData,
    META_FAG_AIV_LIST_NUM,
    META_FAG_BLOCK_ENDS_OFFSET,
    META_FAG_BLOCK_STARTS_OFFSET,
    META_FAG_DKV_ROW_ENDS,
    META_FAG_DKV_ROW_STARTS,
    META_FAG_DKV_ROWS,
    META_FAG_DQ_ROW_ENDS,
    META_FAG_DQ_ROW_STARTS,
    META_FAG_DQ_ROWS,
    META_FAG_PRE_AIV_NUM,
    META_HEAD_FAG_START_INDEX,
    META_VIEW_LEN,
    TILE_M, TILE_N, TILE_D,
)

META_FAG_CORE_LIST_NUM = 36


def _split_rows(total_rows, n_aiv, aiv_num=META_FAG_AIV_LIST_NUM):
    """与 AICPU FagSplitRows 相同：余数给前 rem 个核，空槽 [0,0)。"""
    starts = [0] * aiv_num
    ends = [0] * aiv_num
    if n_aiv <= 0 or total_rows <= 0:
        return starts, ends
    live = min(int(n_aiv), aiv_num)
    base = int(total_rows) // live
    rem = int(total_rows) % live
    cur = 0
    for i in range(live):
        n = base + (1 if i < rem else 0)
        starts[i] = cur
        ends[i] = cur + n
        cur += n
    return starts, ends


def build_fake_metadata(b, n2, s1_outer, s2_outer, cores, tmpl, device, g_size=1,
                        s1=None, s2=None, n1=None):
    """JIT 没有 AICPU，按同样布局造 metadata；GQA 下块网格含 G。

    tmpl==1 对齐 DoFagBn2DenseSplit：先按 B*N2*G 分核，再把 factor 乘上
    s1Outer*s2Outer。单块 BN2 时 s1Outer=s2Outer=1，退化为 total=B*N2。
    pre/post 行区间按 nAiv=usedCube*2 连续均分 dqRows=B*S1*N1、dkvRows=B*S2*N2。
    """
    heads = b * n2 * g_size
    s1s2 = s1_outer * s2_outer
    if tmpl == 1:
        fused_bn = max(heads, 1)
        factor_bn = (fused_bn + cores - 1) // cores
        total = fused_bn * s1s2
        factor = factor_bn * s1s2
    else:
        total = heads * s1s2
        factor = (total + cores - 1) // cores if cores else total
    meta = torch.zeros(META_VIEW_LEN, dtype=torch.int32)
    fag = int(os.environ.get("FAG_META_OFF", "1744"))
    meta[META_HEAD_FAG_START_INDEX] = fag
    if factor <= 0:
        factor = 1
    used_cube = (total + factor - 1) // factor if factor else 0
    meta[fag + 0] = tmpl
    meta[fag + 1] = used_cube
    meta[fag + 2] = factor
    meta[fag + 3] = s1_outer
    meta[fag + 4] = s2_outer
    for c in range(min(cores, META_FAG_CORE_LIST_NUM)):
        start = min(c * factor, total)
        end = min(start + factor, total)
        meta[fag + META_FAG_BLOCK_STARTS_OFFSET + c] = start
        meta[fag + META_FAG_BLOCK_ENDS_OFFSET + c] = end
    if s1 is None:
        s1 = s1_outer * 128
    if s2 is None:
        s2 = s2_outer * 128
    if n1 is None:
        n1 = n2 * g_size
    n_aiv = min(int(used_cube) * 2, META_FAG_AIV_LIST_NUM) if used_cube else 0
    dq_rows = int(b) * int(s1) * int(n1)
    dkv_rows = int(b) * int(s2) * int(n2)
    meta[fag + META_FAG_PRE_AIV_NUM] = n_aiv
    meta[fag + META_FAG_DQ_ROWS] = dq_rows
    meta[fag + META_FAG_DKV_ROWS] = dkv_rows
    dq_starts, dq_ends = _split_rows(dq_rows, n_aiv)
    dkv_starts, dkv_ends = _split_rows(dkv_rows, n_aiv)
    for i in range(META_FAG_AIV_LIST_NUM):
        meta[fag + META_FAG_DQ_ROW_STARTS + i] = dq_starts[i]
        meta[fag + META_FAG_DQ_ROW_ENDS + i] = dq_ends[i]
        meta[fag + META_FAG_DKV_ROW_STARTS + i] = dkv_starts[i]
        meta[fag + META_FAG_DKV_ROW_ENDS + i] = dkv_ends[i]
    return meta.to(device)


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
    # G 必须传入：GQA 下 KV 侧字节按 G 摊薄，否则与 host 判据不一致，
    # 会验证到与线上派发不同的那份二进制(实测 19 个 shape 分歧)。
    swizzle = decide_swizzle(B, N1, S1, S2, D, num_cores, g_size=G)
    # FAG_FORCE_SWIZZLE 用于覆盖判据，单独验证另一条编译产物的正确性 ——
    # 两种分核对同一输入必须给出相同结果(仅分块顺序不同)。
    _force = os.environ.get("FAG_FORCE_SWIZZLE")
    if _force is not None:
        swizzle = int(_force)

    d_align = decide_d_align(D)
    dv_align = decide_dv_align(Dv, d_align)
    tmpl = decide_template(N1, N2, S1, S2, D, is_tnd=False, b=B, dv=Dv)
    is_mb = decide_bn2_multiblk(B, N1, N2, S1, S2, D, dv=Dv)
    _force_tmpl = os.environ.get("FAG_FORCE_TEMPLATE")
    if _force_tmpl is not None:
        tmpl = int(_force_tmpl)
    if tmpl == 1:
        swizzle = 0
    else:
        is_mb = 0
    s1_outer = (S1 + 127) // 128
    s2_outer = (S2 + 127) // 128
    metadata = build_fake_metadata(
        B, N2, s1_outer, s2_outer, num_cores, tmpl, device, g_size=G,
        s1=S1, s2=S2, n1=N1)

    # Launch kernel
    print(f"Launching kernel: B={B}, N1={N1}, N2={N2}, G={G}, S1={S1}, "
          f"S2={S2}, D={D}, Dv={Dv}, d_align={d_align}, "
          f"dv_align={dv_align}, "
          f"scale={scale:.6f}, cores={num_cores}, "
          f"layout={layout}, template={tmpl}, swizzle={swizzle}, "
          f"is_bn2_multiblk={is_mb}")

    # 先触发编译（不计入执行看护窗口）：编译发生在首次调用内部，
    # 故这里用 launch + 同步作为一次完整执行，并只对「执行」计时看护。
    run_kernel_with_timeout(
        lambda: flash_attn_grad[
            None, num_cores,
            # tilingkey：swizzle 与 d_align 走这里(各编一份二进制)。
            # d_align 是 D 的分配宽度分档，取值即实际列数。
            {"template": tmpl, "layout": 0,
             "has_atten_mask": 0, "swizzle": swizzle,
             "d_align": d_align, "dv_align": dv_align,
             "is_bn2_multiblk": is_mb},
            # datatype：对应 @pl.jit 的 datatype={"q": "input_dtype"}
            {"q": pl.DT_FP16},
        ](
            q, k, v, dy, y_in, lse_gold,
            # 6 个未使用的可选输入 + metadata：位置必须占齐
            None, None, None, None, None, None, metadata,
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
    # --- BN2 经典路径：S1/S2<=128, N1==N2 ---
    (1, 1, 128, 128, 128, 1),   # BN2 全对齐
    (1, 1, 64, 64, 64, 1),      # BN2 D=64 档
    (1, 1, 80, 96, 40, 1),      # BN2 D 尾块 + 非对齐 S
    (2, 4, 128, 128, 128, 8),   # BN2 多 head 多核
    (1, 32, 128, 128, 128, 36), # BN2 满核
    # --- BN2GS1S2 ---
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
    # --- D=64 ---
    (1, 1, 256, 256, 64, 1),    # 13: D=64 全对齐基线
    (1, 1, 200, 256, 64, 1),    # 14: D=64 S1 尾块
    (1, 1, 256, 200, 64, 1),    # 15: D=64 S2 尾块
    (1, 1, 200, 168, 64, 1),    # 16: D=64 双尾块
    (2, 2, 256, 256, 64, 1),    # 17: D=64 多 batch / 多 head
    (1, 1, 192, 256, 64, 1),    # 18: D=64 S1 尾块=64
    # --- D=64 的 B/N 泛化 ---
    (3, 1, 129, 131, 64, 1),    # 19: 奇数 B + 双尾块
    (1, 3, 131, 129, 64, 1),    # 20: 奇数 N + 双尾块
    (3, 3, 257, 257, 64, 8),    # 21: B/N 奇数 + 双奇数尾块 + 多核
    (7, 7, 257, 257, 64, 24),   # 22: B/N=7 + 不满核
    (2, 8, 512, 257, 64, 8),    # 23: 多 N + S2 奇数尾块
    (8, 2, 257, 512, 64, 24),   # 24: 多 B + S1 奇数尾块
    (5, 5, 640, 640, 64, 36),   # 25: B/N 奇数全对齐 + 满核
    (1, 1, 8192, 8192, 64, 36), # 26: D=64 满核大 shape
    # --- D 尾块：D 不再等于 d_template 档位 ---
    # 27~31 走 64 档(d_template=64)，32~36 走 128 档。
    # 16 的倍数与非 16 倍数分开取：后者连 dAlign 都对不齐，
    # 是 NZ 分形补零那条路径的真正压力点。
    (1, 1, 256, 256, 16, 1),    # 27: D=16，恰好一个分形
    (1, 1, 256, 256, 32, 1),    # 28: D=32，16 的倍数
    (1, 1, 200, 168, 48, 1),    # 29: D=48 + S1/S2 双尾块
    (1, 1, 256, 256, 40, 1),    # 30: D=40，非 16 倍数(dAlign=48)
    (2, 3, 257, 257, 8, 8),     # 31: D=8，小于一个分形 + 多核双尾块
    (1, 1, 256, 256, 80, 1),    # 32: D=80，16 的倍数
    (1, 1, 256, 256, 96, 1),    # 33: D=96
    (1, 1, 200, 168, 112, 1),   # 34: D=112 + 双尾块
    (1, 1, 256, 256, 72, 1),    # 35: D=72，非 16 倍数(dAlign=80)
    (2, 3, 257, 257, 100, 8),   # 36: D=100，非 16 倍数 + 多核双尾块
    (1, 1, 4096, 4096, 72, 36), # 37: D 尾块 + 满核大 shape
    # --- BN2 MultiBlk：S in (128,640]，BN 门 ---
    (1, 128, 256, 256, 128, 36), # 38: BN=128 且 S 对齐
    (1, 256, 129, 129, 64, 36),  # 39: BN>=256 双尾块 D64
]


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


# S1 != S2 泛化用例：S1 与 S2 相互独立，覆盖两个方向的极端比例、
# 不足一个 cube 块(< 128)、双非对齐、以及与 D=192/GQA/多核的组合。
# 分核判据(decide_swizzle)对 S1/S2 的**方向**敏感 —— S2 更长时 swizzle
# 从不获益、S1 明显更长时拐点更早 —— 故这里两个方向都要覆盖。
# (B, N2, S1, S2, D, cores, G, Dv)
ASYM_CASES = [
    (1, 1,  128,  512, 128, 1, 1, 128),   # 0: S2 = 4x S1
    (1, 1,  512,  128, 128, 1, 1, 128),   # 1: S1 = 4x S2
    (1, 1,  128, 1024, 128, 1, 1, 128),   # 2: S2 = 8x S1
    (1, 1, 1024,  128, 128, 1, 1, 128),   # 3: S1 = 8x S2
    (1, 1,   64,  256, 128, 1, 1, 128),   # 4: S1 < 128(不足一个 cube 块)
    (1, 1,  256,   64, 128, 1, 1, 128),   # 5: S2 < 128
    (1, 1,   64,   64, 128, 1, 1, 128),   # 6: 两者都 < 128
    (1, 1,    1,  128, 128, 1, 1, 128),   # 7: S1 = 1(退化)
    (1, 1,  128,    1, 128, 1, 1, 128),   # 8: S2 = 1(退化)
    (1, 1,  100,  300, 128, 1, 1, 128),   # 9: 双非对齐 + 不等
    (1, 1,  300,  100, 128, 1, 1, 128),   # 10: 反向
    (1, 1,  129,  257, 128, 1, 1, 128),   # 11: 双奇数尾块 + 不等
    (1, 1,  257,  129, 128, 1, 1, 128),   # 12: 反向
    (1, 1,  128, 8192, 128, 1, 1, 128),   # 13: 极端比 1:64
    (1, 1, 8192,  128, 128, 1, 1, 128),   # 14: 极端比 64:1
    (1, 1,  512, 2048, 128, 8, 1, 128),   # 15: 多核 + 不对称
    (1, 1, 2048,  512, 128, 8, 1, 128),   # 16: 反向
    (1, 1,  200,  456, 192, 1, 1, 128),   # 17: D=192/Dv=128 + 不等
    (1, 1,  456,  200, 192, 1, 1, 128),   # 18: 反向
    (1, 1,  128,  384, 192, 1, 4, 128),   # 19: + GQA G=4
    (1, 1,  384,  128, 192, 1, 4, 128),   # 20: 反向
    (1, 16, 8192, 1024, 128, 36, 1, 128), # 21: 判据翻转点(走 swizzle)
    (1, 16, 1024, 8192, 128, 36, 1, 128), # 22: 反向(应走线性)
]


# ==================================================================
#  泛化用例(100 个)：B/N/S1/S2 覆盖奇数、偶数、尾块边界，核数同时
#  覆盖「不满核」(1/8/16/24/32) 与「满核」(36，Ascend950DT 36 cube)。
#  用 CASE_FILTER=G 单独运行；unset CASE_FILTER 跑全部用例。
# ==================================================================
def _gen_generalized_cases(dims=(128, 64)):
    # 24 个基形 × [1, 8, 24, 36] 核 = 96 个，每个 D 模板各来一遍
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
    # 4 个大 shape 专项：满核 / 不满核
    big = [
        (1, 1, 8192, 8192, 36),   # 满核大 shape
        (1, 1, 8192, 8192, 24),   # 不满核大 shape
        (1, 1, 8192, 8192, 8),    # 弱不满核大 shape
        (1, 1, 4096, 4096, 16),   # 中等大 shape 不满核
    ]
    cases = []
    for d in dims:
        cases += [(*s, d, c) for s in base for c in (1, 8, 24, 36)]
        cases += [(b, n, s1, s2, d, c) for (b, n, s1, s2, c) in big]
    return cases

GENERALIZED_CASES = _gen_generalized_cases()
GENERALIZED_CASES_D64 = _gen_generalized_cases((64,))
GENERALIZED_CASES_D128 = _gen_generalized_cases((128,))


# ==================================================================
#  D 尾块泛化用例：D 扫过 (0,128] 里两个档位的各种取值，
#  与 B/N/S1/S2 尾块交叉。CASE_FILTER=GD 跑这一组。
# ==================================================================
def _gen_dtail_cases():
    # 两个档位各挑几个代表：16 的倍数(分形对齐)、非 16 倍数(dAlign 也补)、
    # 档位边界(64/128 本身)、以及小于一个分形的 D。
    dims = [1, 8, 16, 24, 32, 40, 47, 48, 56, 64,        # d_template=64 档
            65, 69, 72, 80, 88, 96, 104, 112, 120, 128]  # d_template=128 档
    # 形状挑 6 个：全对齐、单尾块、双尾块、多 B/N、多核。
    # D 已经是被扫的轴，形状不必再铺满，否则用例数爆炸。
    shapes = [
        (1, 1, 128, 128, 1),      # 全对齐基线
        (1, 1, 257, 257, 1),      # 双奇数尾块
        (2, 3, 200, 168, 8),      # 多 B/N + 双尾块 + 多核
        (3, 2, 129, 512, 24),     # S1 尾块=1 + 不满核
        (1, 5, 512, 129, 36),     # S2 尾块=1 + 满核
        (5, 5, 640, 640, 36),     # B/N 奇数全对齐 + 满核
        (10, 7, 3378, 4499, 36),     # B/N 奇数全对齐 + 满核
    ]
    return [(b, n, s1, s2, d, c) for d in dims
            for (b, n, s1, s2, c) in shapes]


GENERALIZED_CASES_DTAIL = _gen_dtail_cases()


def run_test():
    failed = []
    # CASE_FILTER=i[,j...] 只跑指定下标的用例（设备级错误会终止进程，
    # 逐个跑才能定位是哪个形状出问题）；CASE_FILTER=G 跑 100 个泛化用例。
    # 也支持 "NAME:i,j" 只跑某个命名用例集里的指定下标，
    # 例如 CASE_FILTER=DV:0,3（设备级错误会终止进程，逐个跑才能定位形状）。
    flt = os.environ.get("CASE_FILTER", "")
    # CASE_FILTER=GATE 只跑 host 判据一致性守护（纯算术，不下发设备）
    if flt == "GATE":
        return 0 if test_gate_matches_host() else 1
    _named = {"G": GENERALIZED_CASES, "G64": GENERALIZED_CASES_D64,
              "G128": GENERALIZED_CASES_D128, "GD": GENERALIZED_CASES_DTAIL,
              "GQA": GQA_CASES, "DV": DV_CASES, "ASYM": ASYM_CASES,
              "MB": [(1, 128, 256, 256, 128, 36),
                     (1, 256, 129, 129, 64, 36)]}
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


def test_gate_matches_host():
    """守护 decide_swizzle() 与 host tiling.cpp 的判据逐位一致。

    生产路径的判据由 host 的 FlashAttnGradTilingFunc() 承担，
    `decide_swizzle()` 只是它的 Python 镜像，供 JIT 自己拼 tilingkey 用。
    两边算得不一样**不会报任何错**，只会让 JIT 验证的二进制与 host 实际
    派发的那份不是同一个 —— GQA 下曾因 KV 字节未按 G 摊薄而静默不一致
    (19 个 shape)。故这里把 tiling.cpp 的整数运算照抄一份做交叉校验：
    改判据时若只改了一边，这个用例会立刻失败。

    CASE_FILTER=GATE 单独运行（纯 host 侧算术，不下发设备）。
    """
    CM = CN = 128
    L2 = 128 * 1024 * 1024

    def host_gate(b, n2, g, s1, s2, d, cores):
        """逐行照抄 op_host/flash_attn_grad_tiling.cpp 的 swizzle 段。
        注意用整除以复现 C++ int64 语义（含 5*L2/4）。"""
        s1o = (s1 + CM - 1) // CM
        s2o = (s2 + CN - 1) // CN
        per_head_blocks = s1o * s2o
        fused_heads = b * n2 * g
        if per_head_blocks <= 0 or cores <= 0:
            return 0
        per_core = (fused_heads * per_head_blocks + cores - 1) // cores
        live = len({(c * per_core) // per_head_blocks for c in range(cores)})
        per_head = (3 * s1 * d * 2 + s1 * d * 4
                    + (2 * s2 * d * 2 + 2 * s2 * d * 4) // g)
        threshold = (5 * L2) // 4 if s1o >= 2 * s2o else 2 * L2
        if s2o > s1o:
            return 0
        if live * per_head <= threshold:
            return 0
        if fused_heads * s2o < 2 * cores:
            return 0
        return 1

    Ss = [1, 32, 64, 100, 128, 129, 200, 256, 257, 300, 456,
          512, 1024, 2048, 4096, 8192]
    bad = []
    total = 0
    for g in (1, 2, 3, 4, 8):
        for n2 in (1, 2, 3, 16):
            for b in (1, 2):
                for s1 in Ss:
                    for s2 in Ss:
                        for cores in (1, 8, 36):
                            for d in (128, 192):
                                total += 1
                                mine = decide_swizzle(b, n2 * g, s1, s2, d,
                                                      cores, g_size=g)
                                theirs = host_gate(b, n2, g, s1, s2, d, cores)
                                if mine != theirs:
                                    bad.append((b, n2, g, s1, s2, d, cores,
                                                mine, theirs))
    print(f"[gate] checked {total} combos against the tiling.cpp formula")
    if bad:
        print(f"[gate] MISMATCH in {len(bad)} combos "
              f"(kernel mirror vs host authority):")
        for x in bad[:10]:
            print(f"    b{x[0]} n2={x[1]} G={x[2]} S1={x[3]} S2={x[4]} "
                  f"D{x[5]} cores={x[6]}: mirror={x[7]} host={x[8]}")
        print("    -> fix op_host/flash_attn_grad_tiling.cpp and "
              "decide_swizzle() together (tiling.cpp is authoritative)")
        return False
    print("[gate] PASS: mirror matches host on every combo")
    return True


if __name__ == "__main__":
    sys.exit(run_test())
