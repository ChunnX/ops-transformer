#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
非量化 PA/非PA 非连续 kvcache 测试

规格：
  BNBD: dim0 / dim0+dim1 非连续
  BBND: 仅 dim0 非连续
  NZ:   必须连续
  非PA: 必须连续
"""

import logging
import math
import random

import pytest
import torch
import torch_npu

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

DEVICE_ID = 0
B = 2
N_q = 4
N_kv = 2
D = 128
Q_SEQ = 4
KV_SEQ = 512
BLOCK_SIZE = 512
DTYPE = torch.float16


# ==============================================================================
# 正例
# ==============================================================================
NORMAL_CASES = [
    {"name": "BNBD_BNSD_dim0",  "pa": True,  "kv_l": "BNBD", "q_l": "BNSD", "nc": "dim0"},
    {"name": "BNBD_BNSD_dim01", "pa": True,  "kv_l": "BNBD", "q_l": "BNSD", "nc": "dim01"},
    {"name": "BNBD_BSND_dim0",  "pa": True,  "kv_l": "BNBD", "q_l": "BSND", "nc": "dim0"},
    {"name": "BNBD_BSND_dim01", "pa": True,  "kv_l": "BNBD", "q_l": "BSND", "nc": "dim01"},
    {"name": "BNBD_TND_dim0",   "pa": True,  "kv_l": "BNBD", "q_l": "TND",  "nc": "dim0"},
    {"name": "BNBD_TND_dim01",  "pa": True,  "kv_l": "BNBD", "q_l": "TND",  "nc": "dim01"},
    {"name": "BBND_BNSD_dim0",  "pa": True,  "kv_l": "BBND", "q_l": "BNSD", "nc": "dim0"},
    {"name": "BBND_BSND_dim0",  "pa": True,  "kv_l": "BBND", "q_l": "BSND", "nc": "dim0"},
    {"name": "BBND_TND_dim0",   "pa": True,  "kv_l": "BBND", "q_l": "TND",  "nc": "dim0"},
    {"name": "NZ_BNSD_contig",  "pa": True,  "kv_l": "NZ",   "q_l": "BNSD", "nc": "none"},
    {"name": "NZ_BSND_contig",  "pa": True,  "kv_l": "NZ",   "q_l": "BSND", "nc": "none"},
    {"name": "NZ_TND_contig",   "pa": True,  "kv_l": "NZ",   "q_l": "TND",  "nc": "none"},
    {"name": "NonPA_BNSD_contig","pa": False,"kv_l": "none", "q_l": "BNSD", "nc": "none"},
    {"name": "NonPA_BSND_contig","pa": False,"kv_l": "none", "q_l": "BSND", "nc": "none"},
    {"name": "NonPA_TND_contig", "pa": False,"kv_l": "none", "q_l": "TND",  "nc": "none"},
]

# ==============================================================================
# 异常用例（规格外非连续 → 应报错）
# ==============================================================================
ERROR_CASES = [
    {"name": "BBND_BNSD_dim01",  "pa": True,  "kv_l": "BBND", "q_l": "BNSD", "nc": "dim01"},
    {"name": "BBND_BSND_dim01",  "pa": True,  "kv_l": "BBND", "q_l": "BSND", "nc": "dim01"},
    {"name": "NZ_BNSD_dim0",     "pa": True,  "kv_l": "NZ",   "q_l": "BNSD", "nc": "dim0"},
    {"name": "NZ_BNSD_dim01",    "pa": True,  "kv_l": "NZ",   "q_l": "BNSD", "nc": "dim01"},
    {"name": "NonPA_BNSD_dim0",  "pa": False, "kv_l": "none", "q_l": "BNSD", "nc": "dim0"},
    {"name": "NonPA_BSND_dim0",  "pa": False, "kv_l": "none", "q_l": "BSND", "nc": "dim0"},
]


# ==============================================================================
# 工具
# ==============================================================================

def _block_table(batch, seqs, bs):
    vn = [math.ceil(s / bs) for s in seqs]
    total = sum(vn)
    mx = max(vn)
    ids = list(range(total))
    random.shuffle(ids)
    t = -torch.ones(batch, mx, dtype=torch.int32)
    off = 0
    for b in range(batch):
        t[b, :vn[b]] = torch.tensor(ids[off:off+vn[b]], dtype=torch.int32)
        off += vn[b]
    return t, total


def _make_noncontiguous(tensor, nc):
    """对 4D tensor [A,B,C,D] 构造非连续视图
    nc="dim0": pad dim1 再 slice → dim0 stride 偏大
    nc="dim01": pad dim2 再 slice → dim0/dim1 stride 偏大
    nc="none": 返回原 tensor
    """
    if nc == "none":
        return tensor
    A, Bv, Cv, Dv = tensor.shape
    if nc == "dim0":
        p = torch.empty(A, Bv + 1, Cv, Dv, dtype=tensor.dtype, device=tensor.device)
        p[:, :Bv] = tensor
        return p[:, :Bv, :, :]
    elif nc == "dim01":
        p = torch.empty(A, Bv, Cv + 1, Dv, dtype=tensor.dtype, device=tensor.device)
        p[:, :, :Cv] = tensor
        return p[:, :, :Cv, :]
    else:
        raise ValueError(f"Unknown nc: {nc}")


def _kv_shape(cache_layout, total_blocks):
    if cache_layout == "BNBD":
        return (total_blocks, N_kv, BLOCK_SIZE, D)
    elif cache_layout == "BBND":
        return (total_blocks, BLOCK_SIZE, N_kv * D)
    elif cache_layout == "NZ":
        return (total_blocks, N_kv, D // 32, BLOCK_SIZE, 32)
    elif cache_layout == "none":
        return (B, N_kv, KV_SEQ, D)
    else:
        raise ValueError(f"Unknown cache_layout: {cache_layout}")


def _make_kv_cache(shape, seed):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=DTYPE)


def _q_layout_tensor(q_bnsd, q_layout, act_seq_q):
    if q_layout == "BNSD":
        return q_bnsd
    elif q_layout == "BSND":
        return q_bnsd.permute(0, 2, 1, 3).contiguous()
    elif q_layout == "TND":
        T = sum(act_seq_q)
        out = torch.zeros(T, N_q, D, dtype=DTYPE)
        t = 0
        for b in range(B):
            s = act_seq_q[b]
            out[t:t + s] = q_bnsd[b, :, :s, :].transpose(0, 1)
            t += s
        return out
    raise ValueError(f"q_layout: {q_layout}")


def _accum_seq(seqs):
    out, acc = [], 0
    for s in seqs:
        acc += s
        out.append(acc)
    return out


def _run_npu(q_npu, k_npu, v_npu, pa, q_layout, act_q, act_kv, bt_npu, scale):
    torch_npu.npu.set_device(DEVICE_ID)
    torch.npu.synchronize()
    aq = _accum_seq(act_q) if q_layout == "TND" else act_q
    akv = _accum_seq(act_kv) if q_layout == "TND" else act_kv
    r, _ = torch_npu.npu_fused_infer_attention_score(
        q_npu, k_npu, v_npu,
        actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
        num_heads=N_q, num_key_value_heads=N_kv,
        input_layout=q_layout, scale=scale,
        pre_tokens=65535, next_tokens=65535,
        block_table=bt_npu if pa else None,
        block_size=BLOCK_SIZE if pa else 0,
    )
    torch.npu.synchronize()
    return r


def _check(ref, out, tag):
    d = (out.cpu().float() - ref.cpu().float()).abs()
    mx, mn = d.max().item(), d.mean().item()
    ok = (d <= 0.05).sum().item()
    t = d.numel()
    logger.info(f"  [{tag}] max={mx:.6f} mean={mn:.6f} pass={ok}/{t} {ok/t:.4f}")
    assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=0.05, atol=0.05), \
        f"[{tag}] FAIL max={mx}"


# ==============================================================================
# 执行单个用例
# ==============================================================================

def _run_one(cfg, expect_pass):
    pa = cfg["pa"]
    kv_layout = cfg["kv_l"]
    q_layout = cfg["q_l"]
    nc = cfg["nc"]

    act_q = [Q_SEQ] * B
    act_kv = [KV_SEQ] * B
    scale = 1.0 / math.sqrt(D)

    bt, total_b = None, 0
    if pa:
        bt, total_b = _block_table(B, act_kv, BLOCK_SIZE)
    else:
        total_b = B
    bt_npu = bt.npu() if bt is not None else None

    # Q
    torch.manual_seed(42)
    q_bnsd = torch.randn(B, N_q, Q_SEQ, D, dtype=DTYPE)
    q_npu = _q_layout_tensor(q_bnsd, q_layout, act_q).npu()

    # K/V — 在目标 cache_layout 下生成数据
    kv_shape = _kv_shape(kv_layout, total_b)
    k_ref = _make_kv_cache(kv_shape, 100)
    v_ref = _make_kv_cache(kv_shape, 200)

    # 连续基准
    ref_out = _run_npu(q_npu, k_ref.npu(), v_ref.npu(), pa, q_layout, act_q, act_kv, bt_npu, scale)

    # 非连续 K/V
    k_nc = _make_noncontiguous(k_ref, nc)
    v_nc = _make_noncontiguous(v_ref, nc)
    if nc != "none":
        assert not k_nc.is_contiguous(), f"{cfg['name']}: k not non-contiguous"
        logger.info(f"  k contig={k_nc.is_contiguous()} stride={k_nc.stride()}")
    k_npu = k_nc.npu() if nc == "none" else k_nc.npu()
    v_npu = v_nc.npu() if nc == "none" else v_nc.npu()

    if not expect_pass:
        with pytest.raises(Exception):
            _run_npu(q_npu, k_npu, v_npu, pa, q_layout, act_q, act_kv, bt_npu, scale)
        logger.info(f"  [{cfg['name']}] ERROR as expected")
        return

    out = _run_npu(q_npu, k_npu, v_npu, pa, q_layout, act_q, act_kv, bt_npu, scale)
    _check(ref_out, out, cfg["name"])


# ==============================================================================
# pytest
# ==============================================================================

@pytest.mark.parametrize("cfg", NORMAL_CASES, ids=[c["name"] for c in NORMAL_CASES])
def test_normal(cfg):
    logger.info(f"\n[NORMAL] {cfg['name']}")
    _run_one(cfg, expect_pass=True)


@pytest.mark.parametrize("cfg", ERROR_CASES, ids=[c["name"] for c in ERROR_CASES])
def test_error(cfg):
    logger.info(f"\n[ERROR]  {cfg['name']}")
    _run_one(cfg, expect_pass=False)


if __name__ == '__main__':
    logger.info(f"正例={len(NORMAL_CASES)} 异常={len(ERROR_CASES)}")
    ok = True
    for c in NORMAL_CASES:
        try:
            _run_one(c, True)
            logger.info(f"  PASS")
        except Exception as e:
            logger.info(f"  FAIL: {e}")
            ok = False
    for c in ERROR_CASES:
        try:
            _run_one(c, False)
            logger.info(f"  PASS")
        except Exception as e:
            logger.info(f"  FAIL: {e}")
            ok = False
    logger.info("\n全部通过!" if ok else "\n存在失败!")
