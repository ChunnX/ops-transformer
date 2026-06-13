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
非量化 PA/非PA 非连续 kvcache 测试 (含 rope)

KV Cache 格式:
  BnNBsD = (blockNum, KV_N, blockSize, D)
  BnBsH  = (blockNum, blockSize, KV_N*D)
  NZ     = (blockNum, KV_N, D/16, blockSize, 16), D0=16
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
NZ_D0 = 16
DTYPE = torch.float16


# ==============================================================================
# 用例: d=D, dr=rope dim, dv=V dim. scale = 1/sqrt(d+dr)
# ==============================================================================
NORMAL_CASES = [
    # === 无 rope, D=128, dv=128 ===
    {"id": "BnNBsD_BNSD_dim0",  "pa": True,  "kv": "BnNBsD", "ql": "BNSD", "nc": "dim0",
     "kv_seq": 1024, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "BnNBsD_BNSD_dim01", "pa": True,  "kv": "BnNBsD", "ql": "BNSD", "nc": "dim01",
     "kv_seq": 1024, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "BnNBsD_BSND_dim0",  "pa": True,  "kv": "BnNBsD", "ql": "BSND", "nc": "dim0",
     "kv_seq": 1024, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "BnNBsD_BSND_dim01", "pa": True,  "kv": "BnNBsD", "ql": "BSND", "nc": "dim01",
     "kv_seq": 1024, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "BnBsH_BNSD_dim0",   "pa": True,  "kv": "BnBsH",  "ql": "BNSD", "nc": "dim0",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "BnBsH_BSND_dim0",   "pa": True,  "kv": "BnBsH",  "ql": "BSND", "nc": "dim0",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "NZ_BNSD_contig",    "pa": True,  "kv": "NZ",     "ql": "BNSD", "nc": "none",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "NZ_BSND_contig",    "pa": True,  "kv": "NZ",     "ql": "BSND", "nc": "none",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    # === 无 rope, 非PA ===
    {"id": "NonPA_BNSD_contig", "pa": False, "kv": "none",   "ql": "BNSD", "nc": "none",
     "kv_seq": 1024, "bs": 0, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "NonPA_BSND_contig", "pa": False, "kv": "none",   "ql": "BSND", "nc": "none",
     "kv_seq": 1024, "bs": 0, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
]

ERROR_CASES = [
    {"id": "BnBsH_BNSD_dim01",  "pa": True,  "kv": "BnBsH",  "ql": "BNSD", "nc": "dim01",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "BnBsH_BSND_dim01",  "pa": True,  "kv": "BnBsH",  "ql": "BSND", "nc": "dim01",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "NZ_BNSD_dim0",      "pa": True,  "kv": "NZ",     "ql": "BNSD", "nc": "dim0",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "NZ_BNSD_dim01",     "pa": True,  "kv": "NZ",     "ql": "BNSD", "nc": "dim01",
     "kv_seq": 2048, "bs": 512, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "NonPA_BNSD_dim0",   "pa": False, "kv": "none",   "ql": "BNSD", "nc": "dim0",
     "kv_seq": 1024, "bs": 0, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
    {"id": "NonPA_BSND_dim0",   "pa": False, "kv": "none",   "ql": "BSND", "nc": "dim0",
     "kv_seq": 1024, "bs": 0, "q_seq": 4, "d": 128, "dr": 0, "dv": 128},
]


# ==============================================================================
# 工具
# ==============================================================================

def _block_table(batch, seqs_kv, bs):
    vn = [math.ceil(s / bs) + 1 for s in seqs_kv]
    total = sum(vn)
    mx = max(vn)
    ids = list(range(total))
    random.shuffle(ids)
    tbl = -torch.ones(batch, mx, dtype=torch.int32)
    off = 0
    for b in range(batch):
        tbl[b, :vn[b]] = torch.tensor(ids[off:off + vn[b]], dtype=torch.int32)
        off += vn[b]
    return tbl, total


def _kv_shape(kv_layout, total_blocks, block_size, kv_seq, q_layout, d):
    if kv_layout == "BnNBsD":
        return (total_blocks, N_kv, block_size, d)
    elif kv_layout == "BnBsH":
        return (total_blocks, block_size, N_kv * d)
    elif kv_layout == "NZ":
        return (total_blocks, N_kv, d // NZ_D0, block_size, NZ_D0)
    elif kv_layout == "none":
        if q_layout in ("BSND", "BSH"):
            return (B, kv_seq, N_kv, d)
        return (B, N_kv, kv_seq, d)
    raise ValueError(kv_layout)


def _make_kv(shape):
    torch.manual_seed(100)
    k = torch.randn(*shape, dtype=DTYPE)
    torch.manual_seed(200)
    v = torch.randn(*shape, dtype=DTYPE)
    return k, v


def _make_nc_on_npu(tensor, nc):
    if nc == "none":
        return tensor
    pad_dim = 1 if nc == "dim0" else (2 if len(tensor.shape) == 4 else 3)
    Bv = tensor.shape[pad_dim]
    new_shape = list(tensor.shape)
    new_shape[pad_dim] = Bv + 1
    padded = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    ws = [slice(None)] * len(tensor.shape)
    ws[pad_dim] = slice(None, Bv)
    padded[tuple(ws)] = tensor
    return padded[tuple(ws)]


def _contiguous_strides(shape):
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return tuple(s)


def _q_layout(q_bnsd, q_layout, act_q):
    if q_layout == "BNSD":
        return q_bnsd
    elif q_layout == "BSND":
        return q_bnsd.permute(0, 2, 1, 3).contiguous()
    raise ValueError(q_layout)


def _accum(seqs):
    out, acc = [], 0
    for s in seqs:
        acc += s
        out.append(acc)
    return out


def _run(q_npu, k_npu, v_npu, pa, q_layout, act_q, act_kv, bt_npu, bs, scale):
    torch_npu.npu.set_device(DEVICE_ID)
    torch.npu.synchronize()
    aq = torch.tensor(act_q, dtype=torch.int64)
    akv = torch.tensor(act_kv, dtype=torch.int64)
    r, _ = torch_npu.npu_fused_infer_attention_score(
        q_npu, k_npu, v_npu,
        actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
        num_heads=N_q, num_key_value_heads=N_kv,
        input_layout=q_layout, scale=scale,
        pre_tokens=65535, next_tokens=65535,
        block_table=bt_npu if pa else None,
        block_size=bs if pa else 0)
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
# 用例执行
# ==============================================================================

def _run_one(cfg, expect_pass):
    pa = cfg["pa"]
    kv_layout = cfg["kv"]
    q_layout = cfg["ql"]
    nc = cfg["nc"]
    kv_seq = cfg["kv_seq"]
    bs = cfg["bs"]
    q_seq = cfg["q_seq"]
    d = cfg["d"]
    dr = cfg["dr"]
    dv = cfg["dv"]

    act_q = [q_seq] * B
    act_kv = [kv_seq] * B
    scale = 1.0 / math.sqrt(d + dr)

    bt, total_b = None, 0
    if pa:
        bt, total_b = _block_table(B, act_kv, bs)
    else:
        total_b = B
    bt_npu = bt.npu() if bt is not None else None

    torch.manual_seed(42)
    q_bnsd = torch.randn(B, N_q, q_seq, d, dtype=DTYPE)
    q_npu = _q_layout(q_bnsd, q_layout, act_q).npu()

    kv_shape = _kv_shape(kv_layout, total_b, bs, kv_seq, q_layout, d)
    k_ref, v_ref = _make_kv(kv_shape)

    ref_out = _run(q_npu, k_ref.npu(), v_ref.npu(), pa, q_layout, act_q, act_kv,
                   bt_npu, bs, scale)

    k_nc = _make_nc_on_npu(k_ref.npu(), nc)
    v_nc = _make_nc_on_npu(v_ref.npu(), nc)
    if nc != "none":
        cstrides = _contiguous_strides(k_nc.shape)
        assert k_nc.stride(0) != cstrides[0], f"{cfg['id']}: dim0 not non-contiguous"
        if nc == "dim01":
            assert k_nc.stride(1) != cstrides[1], f"{cfg['id']}: dim1 not non-contiguous"
        logger.info(f"  k contig={k_nc.is_contiguous()} stride={k_nc.stride()}")

    if not expect_pass:
        with pytest.raises(Exception):
            _run(q_npu, k_nc, v_nc, pa, q_layout, act_q, act_kv,
                 bt_npu, bs, scale)
        logger.info(f"  [{cfg['id']}] ERROR as expected")
        return

    out = _run(q_npu, k_nc, v_nc, pa, q_layout, act_q, act_kv,
               bt_npu, bs, scale)
    _check(ref_out, out, cfg["id"])


@pytest.mark.parametrize("cfg", NORMAL_CASES, ids=[c["id"] for c in NORMAL_CASES])
def test_normal(cfg):
    logger.info(f"\n[NORMAL] {cfg['id']}")
    _run_one(cfg, expect_pass=True)


@pytest.mark.parametrize("cfg", ERROR_CASES, ids=[c["id"] for c in ERROR_CASES])
def test_error(cfg):
    logger.info(f"\n[ERROR]  {cfg['id']}")
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
