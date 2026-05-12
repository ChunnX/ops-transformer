#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch_npu.npu_fused_infer_attention_score 批量一致性验证脚本

聚焦 SplitFuse FD (FlashDecode) 路径的 batch variance FAIL 用例。

FD 路径根因 (flash_attention_infer_tiling.h:619-630):
  totalTaskNum = sum(numHeads * qSeqlen_i * kvSeqlen_i)  ← 跨所有 batch 求和
  perCoreTaskNum = ceil(totalTaskNum / blockNum)           ← 决定 core 切分边界
  → batch 增删改 → totalTaskNum 变 → perCoreTaskNum 变
  → 同一 batch 的 KV 切分边界漂移 → CombineScale 浮点归约无法 bit-exact

平台 FD 触发条件:
  arch35 (Ascend 9xx): qSeqlen=1, maxKV≥256, loopTimes < 0.4*aicNum, 不强制 paged
  arch32 (Ascend 910B): pagedCacheFlag==true (block_table 参数) + isShortSeq/isLongSeq

测试策略 — 绕过 TND layout 的复杂约束，使用 BSH layout:
  - 非 paged: BSH layout, q=1, long KV → arch35 触发 FD
  - Paged:   BSH layout + block_table, KV cache (blocks, block_size, H) → 双平台触发 FD
"""

import torch
import torch_npu
import math
import sys
import numpy as np
from typing import Optional


# ==============================================================================
# 工具函数
# ==============================================================================

def _make_scale(head_dim: int, scale: Optional[float] = None) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(float(head_dim))
    return torch.tensor(scale, dtype=torch.float, device="npu")


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """逐元素比较，返回 bit-exact 状态和误差统计。"""
    eq = torch.equal(a, b)
    diff = (a.float() - b.float()).abs()
    return {
        "bitwise": eq,
        "max_diff": float(diff.max().cpu()),
        "mean_diff": float(diff.mean().cpu()),
        "max_rel": float((diff / (b.float().abs() + 1e-8)).max().cpu()),
    }


def _status(r: dict) -> str:
    tag = "PASS" if r["bitwise"] else "FAIL"
    return (f"[{tag}] bitwise={r['bitwise']}, max_diff={r['max_diff']:.3e}, "
            f"mean_diff={r['mean_diff']:.3e}")


def _detect_arch():
    name = torch_npu.npu.get_device_name(0)
    is_32 = "910" in name
    label = "arch32" if is_32 else "arch35"
    return is_32, f"{name} ({label})"


# ==============================================================================
# API 封装
# ==============================================================================

def run_fia_bsh(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_heads: int,
    n_kv_heads: int = 0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """BSH layout, 无 paged attention, 最小参数集。"""
    d = query.shape[-1] // n_heads
    s = _make_scale(d, scale)
    out = torch_npu.npu_fused_infer_attention_score(
        query=query, key=key, value=value,
        num_heads=n_heads, num_key_value_heads=n_kv_heads,
        input_layout="BSH", scale=s,
        pre_tokens=2147483647, next_tokens=2147483647,
        sparse_mode=0, inner_precise=0,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


def run_fia_bsh_paged(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    n_heads: int,
    n_kv_heads: int = 0,
    actual_seq_lengths: Optional[torch.Tensor] = None,
    actual_seq_lengths_kv: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """BSH layout + paged attention, KV cache 为 (total_blocks, block_size, H)。

    KV cache 格式 (blocknum, blocksize, H) — BSH 唯一支持的 paged 格式。

    page attention 强制要求 actual_seq_lengths 和 actual_seq_lengths_kv
    (actual_seq_len_checker.cpp:336), 即使 Q_S=1 也不例外。
    """
    d = query.shape[-1] // n_heads
    s = _make_scale(d, scale)
    B = query.shape[0]
    if actual_seq_lengths is None:
        actual_seq_lengths = torch.full((B,), 1, dtype=torch.int32, device=query.device)
    if actual_seq_lengths_kv is None:
        raise ValueError("actual_seq_lengths_kv is required for page attention")
    out = torch_npu.npu_fused_infer_attention_score(
        query=query, key=k_cache, value=v_cache,
        num_heads=n_heads, num_key_value_heads=n_kv_heads,
        input_layout="BSH", scale=s,
        pre_tokens=2147483647, next_tokens=2147483647,
        sparse_mode=0, inner_precise=0,
        block_table=block_table, block_size=block_size,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
    )
    return out[0] if isinstance(out, (tuple, list)) else out


# ==============================================================================
# 测试用例
# ==============================================================================

# ---------------------------------------------------------------------------
# 基线: 必须 PASS
# ---------------------------------------------------------------------------

def test_determinism():
    """单样本确定性 — 任何路径都必须 bit-exact。"""
    N, NKV, D = 8, 8, 128
    q = torch.randn(1, 1, N * D, dtype=torch.float16, device="npu")
    k = torch.randn(1, 4096, NKV * D, dtype=torch.float16, device="npu")
    v = torch.randn(1, 4096, NKV * D, dtype=torch.float16, device="npu")
    a = run_fia_bsh(q, k, v, N, NKV)
    b = run_fia_bsh(q, k, v, N, NKV)
    return _compare(a, b)


def test_prompt_baseline():
    """Prompt 场景: q>1, 走 PFA/SplitFuse non-FD, batch 独立调度 → PASS。"""
    N, NKV, D = 8, 1, 128
    q1 = torch.randn(1, 16, N * D, dtype=torch.float16, device="npu")
    k1 = torch.randn(1, 2048, NKV * D, dtype=torch.float16, device="npu")
    v1 = torch.randn(1, 2048, NKV * D, dtype=torch.float16, device="npu")
    ref = run_fia_bsh(q1, k1, v1, N, NKV)

    q4 = torch.randn(4, 16, N * D, dtype=torch.float16, device="npu")
    k4 = torch.randn(4, 2048, NKV * D, dtype=torch.float16, device="npu")
    v4 = torch.randn(4, 2048, NKV * D, dtype=torch.float16, device="npu")
    q4[2:3] = q1; k4[2:3] = k1; v4[2:3] = v1
    out4 = run_fia_bsh(q4, k4, v4, N, NKV)
    return _compare(ref, out4[2:3])


# ---------------------------------------------------------------------------
# arch35 FD 触发 (非 paged, BSH layout)
#
# arch35 条件: q=1, maxKV≥256, loopTimes = batch * kvHeads < 0.4*aicNum
# MHA (gSize=1): 仅需 loopTimes < 0.4*aicNum
# GQA: 还需 maxKV ≥ 2048
#
# 这些用例在 arch32 上预期 PASS (arch32 FD 强制要求 pagedCacheFlag)
# ---------------------------------------------------------------------------

def test_fd_mha_batch_size():
    """MHA 下 batch size 变化触发 FD 开关。

    arch35: B=1 → loopTimes=kvHeads < 0.4*aicNum → FD ON
            B=3 → loopTimes=3*kvHeads, 可能 ≥ 0.4*aicNum → FD OFF
            同一输入走不同 kernel 路径 → FAIL

    使用较大输入 (kv=8192, N=NKV=8, D=128) 以稳定触发 FD。
    """
    N, NKV, D = 8, 8, 128
    KV = 8192

    q3 = torch.randn(3, 1, N * D, dtype=torch.float16, device="npu")
    k3 = torch.randn(3, KV, NKV * D, dtype=torch.float16, device="npu")
    v3 = torch.randn(3, KV, NKV * D, dtype=torch.float16, device="npu")

    # 对照组: batch[1] 单独算 (B=1 → FD ON)
    ref = run_fia_bsh(q3[1:2], k3[1:2], v3[1:2], N, NKV)
    # 实验组: 3-batch 一起算 (B=3 → FD 可能 OFF, 或至少 perCoreTaskNum 不同)
    out3 = run_fia_bsh(q3, k3, v3, N, NKV)
    return _compare(ref, out3[1:2])


def test_fd_gqa_batch_size():
    """GQA 下 batch size 变化。GQA 增大 gSize → loopTimes 更小 → FD 更易触发。

    N=16, NKV=1, D=128: gSize=16, loopTimes = batch * 1
    kv=16384 → totalTaskNum 很大，切分边界漂移效应明显。

    arch35: B=1(FD ON) vs B=4(FD 状态不同或切分边界不同) → FAIL
    """
    N, NKV, D = 16, 1, 128
    KV = 16384

    q1 = torch.randn(1, 1, N * D, dtype=torch.float16, device="npu")
    k1 = torch.randn(1, KV, NKV * D, dtype=torch.float16, device="npu")
    v1 = torch.randn(1, KV, NKV * D, dtype=torch.float16, device="npu")
    ref = run_fia_bsh(q1, k1, v1, N, NKV)

    q4 = torch.randn(4, 1, N * D, dtype=torch.float16, device="npu")
    k4 = torch.randn(4, KV, NKV * D, dtype=torch.float16, device="npu")
    v4 = torch.randn(4, KV, NKV * D, dtype=torch.float16, device="npu")
    q4[3:4] = q1; k4[3:4] = k1; v4[3:4] = v1
    out4 = run_fia_bsh(q4, k4, v4, N, NKV)
    return _compare(ref, out4[3:4])


def test_fd_reorder():
    """batch 重排序: 同一 batch 集合，仅调换顺序。

    totalTaskNum 不变 → perCoreTaskNum 不变，但 batch 在四维任务空间
    (B, N1, S1, S2) 中的起始偏移改变，导致 KV 切分边界相对该 batch 漂移。

    arch35: FD → FAIL
    """
    B, N, NKV, D = 3, 8, 8, 128
    KV = 8192

    q = torch.randn(B, 1, N * D, dtype=torch.float16, device="npu")
    k = torch.randn(B, KV, NKV * D, dtype=torch.float16, device="npu")
    v = torch.randn(B, KV, NKV * D, dtype=torch.float16, device="npu")

    out_orig = run_fia_bsh(q, k, v, N, NKV)

    # 循环右移 1 位: [0,1,2] → [2,0,1], 逆排列 [1,2,0]
    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    inv = torch.tensor([1, 2, 0], dtype=torch.long)
    out_perm = run_fia_bsh(q[perm], k[perm], v[perm], N, NKV)

    return _compare(out_orig, out_perm[inv])


def test_fd_batch_endpoints():
    """batch 内首尾位置与单样本对比。

    B=4 中:
      - batch[0] 单独 vs batch[0] 在 4-batch 中
      - batch[3] 单独 vs batch[3] 在 4-batch 中

    首尾两个位置在四维任务空间中偏移差距最大，
    如果 FD 切分边界漂移，至少一个位置应检测到不一致。
    """
    N, NKV, D = 8, 8, 128
    KV = 8192

    q4 = torch.randn(4, 1, N * D, dtype=torch.float16, device="npu")
    k4 = torch.randn(4, KV, NKV * D, dtype=torch.float16, device="npu")
    v4 = torch.randn(4, KV, NKV * D, dtype=torch.float16, device="npu")
    out4 = run_fia_bsh(q4, k4, v4, N, NKV)

    # batch[0] 单独
    ref0 = run_fia_bsh(q4[0:1], k4[0:1], v4[0:1], N, NKV)
    r0 = _compare(ref0, out4[0:1])

    # batch[3] 单独
    ref3 = run_fia_bsh(q4[3:4], k4[3:4], v4[3:4], N, NKV)
    r3 = _compare(ref3, out4[3:4])

    # 任一处不一致即报告 FAIL
    return r0 if not r0["bitwise"] else r3


# ---------------------------------------------------------------------------
# Paged FD 触发 (BSH layout + block_table, 双平台均应 FAIL)
#
# KV cache: (total_blocks, block_size, n_kv_heads * D) — BSH 唯一支持的格式
# block_table: (B, blocks_per_seq), int32
#
# arch32: pagedCacheFlag=true + isShortSeq(batch*kvHeads≤0.4*aicNum, KV≥1024)
#         或 isLongSeq(batch*kvHeads≤0.8*aicNum, KV≥aicNum*512)
# arch35: q=1 + longKV → FD
#
# 关键: 目标样本在两种配置中看到的 Q/K/V 数据完全相同，
# 仅 batch 组合不同 → totalTaskNum 不同 → 切分边界不同。
# ---------------------------------------------------------------------------

def _make_paged_kv(batch_size: int, kv_len: int, block_size: int,
                   n_kv_heads: int, d: int, q_len: int = 1):
    """创建 BSH-paged KV cache、block_table 和 actual_seq_lengths。

    KV cache: (batch_size * blocks_per_seq, block_size, n_kv_heads * D)

    返回: (k_cache, v_cache, block_table, blocks_per_seq,
           actual_seq_lengths, actual_seq_lengths_kv)
    """
    blocks_per_seq = (kv_len + block_size - 1) // block_size
    total_blocks = batch_size * blocks_per_seq
    H = n_kv_heads * d

    k_cache = torch.randn(total_blocks, block_size, H, dtype=torch.float16, device="npu")
    v_cache = torch.randn(total_blocks, block_size, H, dtype=torch.float16, device="npu")
    block_table = torch.arange(total_blocks, dtype=torch.int32, device="npu").reshape(
        batch_size, blocks_per_seq)

    # page attention 强制要求 actual_seq_lengths / actual_seq_lengths_kv
    # BSH layout 下使用 per-batch 值 (非累积), int32
    act_q = torch.full((batch_size,), q_len, dtype=torch.int32, device="npu")
    act_kv = torch.full((batch_size,), kv_len, dtype=torch.int32, device="npu")

    return k_cache, v_cache, block_table, blocks_per_seq, act_q, act_kv


def test_paged_fd_batch_size():
    """BSH-paged: batch size 变化。

    N=NKV=8, KV=16384, block=512 → 32 blocks/seq, H=1024.
    使用较大的 block_size=512 以获得更好的对齐，较大的 KV=16384 以放大差异。

    B=1 (FD ON) vs B=4 → totalTaskNum 从 131072 变为 524288
    → perCoreTaskNum 显著不同 → 切分边界完全不同 → FAIL (双平台)
    """
    N, NKV, D = 8, 8, 128
    KV = 16384
    BLK = 512

    # --- 对照组: B=1 ---
    k1, v1, bt1, nblk, aq1, akv1 = _make_paged_kv(1, KV, BLK, NKV, D)
    q1 = torch.randn(1, 1, N * D, dtype=torch.float16, device="npu")
    ref = run_fia_bsh_paged(q1, k1, v1, bt1, BLK, N, NKV,
                            actual_seq_lengths=aq1,
                            actual_seq_lengths_kv=akv1)

    # --- 实验组: B=4, 目标样本在 batch[1] ---
    k4, v4, bt4, _, aq4, akv4 = _make_paged_kv(4, KV, BLK, NKV, D)
    # 将 batch[1] 的 KV blocks 复制为与对照组相同的数据
    k4[bt4[1]] = k1[bt1[0]]
    v4[bt4[1]] = v1[bt1[0]]

    q4 = torch.randn(4, 1, N * D, dtype=torch.float16, device="npu")
    q4[1:2] = q1
    out4 = run_fia_bsh_paged(q4, k4, v4, bt4, BLK, N, NKV,
                             actual_seq_lengths=aq4,
                             actual_seq_lengths_kv=akv4)

    return _compare(ref, out4[1:2])


def test_paged_fd_reorder():
    """BSH-paged: batch 循环右移重排。

    同一组 3-batch，循环右移一位 [0,1,2] → [2,0,1]。
    totalTaskNum 不变但所有 batch 在四维任务空间 (B,N1,S1,S2) 中的
    起始偏移全部改变 → KV 切分边界相对于每个 batch 均漂移 → FAIL。

    使用较大的 kv=16384, block=512 以匹配已验证触发 FD 的参数。
    """
    B, N, NKV, D = 3, 8, 8, 128
    KV = 16384
    BLK = 512

    k_cache, v_cache, bt, nblk, aq, akv = _make_paged_kv(B, KV, BLK, NKV, D)
    q = torch.randn(B, 1, N * D, dtype=torch.float16, device="npu")

    out_orig = run_fia_bsh_paged(q, k_cache, v_cache, bt, BLK, N, NKV,
                                 actual_seq_lengths=aq,
                                 actual_seq_lengths_kv=akv)

    # 循环右移: [0,1,2] → [2,0,1]; 逆排列 [1,2,0] 用于还原比较
    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    inv = torch.tensor([1, 2, 0], dtype=torch.long)
    out_perm = run_fia_bsh_paged(
        q[perm], k_cache, v_cache, bt[perm], BLK, N, NKV,
        actual_seq_lengths=aq[perm],
        actual_seq_lengths_kv=akv[perm])

    return _compare(out_orig, out_perm[inv])


def test_paged_fd_add_dummy():
    """BSH-paged: 新增 dummy batch 改变 perCoreTaskNum。

    B=2 → B=3 (新增 batch 的 kvLen 不同, taskCount 不同):
    对照组 [b0, b1] vs 实验组 [b0, b1, b_dummy]
    b1 数据完全相同，但 totalTaskNum 因 b_dummy 的 taskCount 而改变。

    b_dummy 的 kvLen=100 (远小于 b0/b1 的 16384) → taskCount 差异明确
    → totalTaskNum 变化 → perCoreTaskNum 变化 → b1 的切分边界可能改变 → FAIL
    """
    N, NKV, D = 8, 8, 128
    KV_MAIN = 16384
    KV_DUMMY = 100   # 远小于主 KV，制造 taskCount 不对称
    BLK = 512

    # 对照组: B=2 (两个相同 KV 长度的 batch)
    nblk_main = (KV_MAIN + BLK - 1) // BLK  # 32
    nblk_dummy = (KV_DUMMY + BLK - 1) // BLK  # 1

    total_blocks_ctl = 2 * nblk_main
    k_ctl = torch.randn(total_blocks_ctl, BLK, NKV * D, dtype=torch.float16, device="npu")
    v_ctl = torch.randn(total_blocks_ctl, BLK, NKV * D, dtype=torch.float16, device="npu")
    bt_ctl = torch.arange(total_blocks_ctl, dtype=torch.int32, device="npu").reshape(2, nblk_main)
    aq_ctl = torch.tensor([1, 1], dtype=torch.int32, device="npu")
    akv_ctl = torch.tensor([KV_MAIN, KV_MAIN], dtype=torch.int32, device="npu")

    q_ctl = torch.randn(2, 1, N * D, dtype=torch.float16, device="npu")
    out_ctl = run_fia_bsh_paged(q_ctl, k_ctl, v_ctl, bt_ctl, BLK, N, NKV,
                                actual_seq_lengths=aq_ctl,
                                actual_seq_lengths_kv=akv_ctl)

    # b1 的输入 (用于单独计算对照)
    q_b1 = q_ctl[1:2].clone()
    k_b1 = k_ctl[bt_ctl[1]].clone()
    v_b1 = v_ctl[bt_ctl[1]].clone()
    bt_b1 = torch.arange(nblk_main, dtype=torch.int32, device="npu").reshape(1, nblk_main)
    aq_b1 = torch.tensor([1], dtype=torch.int32, device="npu")
    akv_b1 = torch.tensor([KV_MAIN], dtype=torch.int32, device="npu")
    ref_b1 = run_fia_bsh_paged(q_b1, k_b1, v_b1, bt_b1, BLK, N, NKV,
                               actual_seq_lengths=aq_b1,
                               actual_seq_lengths_kv=akv_b1)

    # 验证 B=2 中 batch[1] 的输出与单独算一致 (基线)
    r_ctl = _compare(ref_b1, out_ctl[1:2])

    # 实验组: B=3 [b0, b1, b_dummy], b1 数据不变
    total_blocks_exp = 2 * nblk_main + nblk_dummy
    k_exp = torch.randn(total_blocks_exp, BLK, NKV * D, dtype=torch.float16, device="npu")
    v_exp = torch.randn(total_blocks_exp, BLK, NKV * D, dtype=torch.float16, device="npu")
    # bt_exp: 3 行分别对应 3 个 batch, 每行长度 = max(nblk_main, nblk_dummy)
    max_blk = max(nblk_main, nblk_dummy)
    bt_exp = torch.zeros(3, max_blk, dtype=torch.int32, device="npu")
    bt_exp[0, :nblk_main] = torch.arange(nblk_main, dtype=torch.int32, device="npu")
    bt_exp[1, :nblk_main] = torch.arange(nblk_main, 2 * nblk_main, dtype=torch.int32, device="npu")
    bt_exp[2, :nblk_dummy] = torch.arange(2 * nblk_main, 2 * nblk_main + nblk_dummy, dtype=torch.int32, device="npu")

    # 复制 b1 的 KV 数据到实验组对应位置
    k_exp[bt_exp[1, :nblk_main]] = k_b1[bt_b1[0]]
    v_exp[bt_exp[1, :nblk_main]] = v_b1[bt_b1[0]]

    q_exp = torch.randn(3, 1, N * D, dtype=torch.float16, device="npu")
    q_exp[1:2] = q_b1
    aq_exp = torch.tensor([1, 1, 1], dtype=torch.int32, device="npu")
    akv_exp = torch.tensor([KV_MAIN, KV_MAIN, KV_DUMMY], dtype=torch.int32, device="npu")
    out_exp = run_fia_bsh_paged(q_exp, k_exp, v_exp, bt_exp, BLK, N, NKV,
                                actual_seq_lengths=aq_exp,
                                actual_seq_lengths_kv=akv_exp)

    r_exp = _compare(ref_b1, out_exp[1:2])

    # 返回实验组结果: b1 在 B=2 vs B=3 中是否 bit-exact
    return r_exp


# ==============================================================================
# 测试注册
# ==============================================================================

TestCase = tuple  # (name, desc, fn, expect_arch35, expect_arch32)

ALL_TESTS: list = [
    # (name, desc, fn, expect_arch35, expect_arch32)

    # ---- 基线 (任何平台必须 PASS) ----
    ("determinism",
     "同一 (B=1) 输入连续计算两次",
     test_determinism,
     "PASS", "PASS"),

    ("prompt-baseline",
     "q=16, kv=2048, GQA 8:1 — Prompt 路径, batch 独立调度",
     test_prompt_baseline,
     "PASS", "PASS"),

    # ---- arch35 FD 触发 (非 paged, arch35 FAIL, arch32 PASS) ----
    ("fd-mha-bs",
     "MHA 8:8, kv=8192, B=1(FD ON) vs B=3, 非 paged",
     test_fd_mha_batch_size,
     "FAIL", "PASS"),

    ("fd-gqa-bs",
     "GQA 16:1, kv=16384, B=1 vs B=4, 非 paged, 大 KV 放大差异 — 910B3 实测 FAIL",
     test_fd_gqa_batch_size,
     "FAIL", "FAIL"),

    ("fd-reorder",
     "MHA 8:8, kv=8192, B=3 循环右移重排, 非 paged",
     test_fd_reorder,
     "FAIL", "PASS"),

    ("fd-endpoints",
     "MHA 8:8, kv=8192, B=4 首尾 vs 单样本, 非 paged",
     test_fd_batch_endpoints,
     "FAIL", "PASS"),

    # ---- Paged FD 触发 (BSH+block_table, 双平台均应 FAIL) ----
    ("paged-fd-bs",
     "BSH-paged MHA 8:8, kv=16384, block=512, B=1 vs B=4",
     test_paged_fd_batch_size,
     "FAIL", "FAIL"),

    ("paged-fd-reorder",
     "BSH-paged MHA 8:8, kv=16384, block=512, B=3 循环右移重排",
     test_paged_fd_reorder,
     "FAIL", "FAIL"),

    ("paged-fd-dummy",
     "BSH-paged MHA 8:8, B=2 vs B=3(新增 kv=100 dummy batch), 切分边界漂移",
     test_paged_fd_add_dummy,
     "FAIL", "FAIL"),
]


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    print("=" * 72)
    print("FusedInferAttentionScore — Batch Invariance Verification")
    print("Focus: SplitFuse FD path FAIL cases")
    print("=" * 72)

    if not torch_npu.npu.is_available():
        print("\nERROR: NPU device not available. Requires Ascend hardware.")
        sys.exit(1)

    is_arch32, device_label = _detect_arch()
    print(f"\nDevice: {device_label}")
    if is_arch32:
        print("arch32 (Ascend 910B): FD requires block_table (pagedCacheFlag).")
        print("  Non-paged tests expected PASS on this platform.")
    else:
        print("arch35 (Ascend 9xx): FD does not require paged. Non-paged q=1+longKV triggers FD.")
    print()

    results = []
    for name, desc, fn, exp35, exp32 in ALL_TESTS:
        expected = exp32 if is_arch32 else exp35
        print("-" * 72)
        print(f"Test: {name}")
        print(f"  {desc}")
        print(f"  Expected ({'arch32' if is_arch32 else 'arch35'}): {expected}")

        try:
            r = fn()
            actual = "PASS" if r["bitwise"] else "FAIL"
            match = "OK" if actual == expected else "MISMATCH"
            marker = "✓" if actual == expected else "✗"
            print(f"  Result:  {_status(r)}")
            print(f"  Verdict: {actual} (expected {expected}) {marker} {match}")
            results.append((name, expected, actual, r))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()
            results.append((name, expected, "ERROR", None))

    # ---- 汇总 ----
    print("\n" + "=" * 72)
    print(f"Summary ({device_label})")
    print("=" * 72)

    n_total = len(results)
    n_pass = sum(1 for _, _, a, _ in results if a == "PASS")
    n_fail = sum(1 for _, _, a, _ in results if a == "FAIL")
    n_error = sum(1 for _, _, a, _ in results if a == "ERROR")
    n_exp_fail = sum(1 for _, e, _, _ in results if e == "FAIL")
    n_exp_pass = sum(1 for _, e, _, _ in results if e == "PASS")
    n_detected = sum(1 for _, e, a, _ in results if e == "FAIL" and a == "FAIL")
    n_missed = sum(1 for _, e, a, _ in results if e == "FAIL" and a == "PASS")
    n_broken = sum(1 for _, e, a, _ in results if e == "PASS" and a == "FAIL")

    print(f"  Total:                {n_total}")
    print(f"  PASS:                 {n_pass}")
    print(f"  FAIL:                 {n_fail}")
    print(f"  ERROR:                {n_error}")
    print()
    print(f"  Expected FAIL:  {n_exp_fail}")
    print(f"    → detected:   {n_detected}")
    print(f"    → missed:     {n_missed}")
    print(f"  Expected PASS:  {n_exp_pass}")
    print(f"    → broken:     {n_broken}")
    print()

    if n_detected > 0:
        print("CONCLUSION: Batch invariance violation CONFIRMED.")
        print("  SplitFuse FD path perCoreTaskNum depends on totalTaskNum across ALL batches,")
        print("  causing KV split boundaries to shift. CombineScale float reduction")
        print("  cannot guarantee bit-exact consistency.")
        print()
        print("  Reference: fused_infer_attention_score_batch_consistency_analysis.md")
    elif n_exp_fail > 0 and n_missed == n_exp_fail:
        print("NOTE: All expected-FAIL tests PASSED.")
        print("  FD conditions may not be met on this device configuration.")
        print("  Try adjusting kv_len, num_heads, or batch sizes.")
    else:
        print("All tests behaved as expected.")


if __name__ == "__main__":
    main()
