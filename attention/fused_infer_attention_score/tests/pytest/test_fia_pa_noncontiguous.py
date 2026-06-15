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
非量化 kvcache 非连续 stride 传递测试

KV Cache 格式:
  BnNBsD = (blockNum, KV_N, blockSize, D)
  BnBsH  = (blockNum, blockSize, KV_N*D)
  NZ     = (blockNum, KV_N, D/16, blockSize, 16)

非连续规格:
  BnNBsD: dim0 / dim01
  BnBsH:  dim0 only
  NZ:     dim0 / dim01
  非PA:   不允许非连续
"""

import logging, math, random
import pytest, torch, torch_npu

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

DEVICE_ID = 0
DTYPE = torch.float16


def _block_table(batch, seqs_kv, bs):
    vn = [math.ceil(s / bs) + 1 for s in seqs_kv]
    total, mx = sum(vn), max(vn)
    ids = list(range(total)); random.shuffle(ids)
    tbl = -torch.ones(batch, mx, dtype=torch.int32)
    off = 0
    for b in range(batch):
        tbl[b, :vn[b]] = torch.tensor(ids[off:off+vn[b]], dtype=torch.int32); off += vn[b]
    return tbl, total


def _kv_shape(fmt, total_blocks, bs, d, N_kv, B, kv_seq):
    if fmt == "BnNBsD":  return (total_blocks,  N_kv, bs, d)
    if fmt == "BnBsH":   return (total_blocks,  bs, N_kv * d)
    if fmt == "NZ":      return (total_blocks,  N_kv, d // 16, bs, 16)
    if fmt == "none":    return (B, N_kv, kv_seq, d)
    raise ValueError(fmt)


def _make_kv(shape):
    torch.manual_seed(100); k = torch.randn(*shape, dtype=DTYPE)
    torch.manual_seed(200); v = torch.randn(*shape, dtype=DTYPE)
    return k, v


def _make_nc(tensor, nc):
    if nc == "none": return tensor
    pad_dim = 1 if nc == "dim0" else 2
    Bv = tensor.shape[pad_dim]
    shp = list(tensor.shape); shp[pad_dim] = Bv + 1
    p = torch.zeros(shp, dtype=tensor.dtype, device=tensor.device)
    ws = [slice(None)] * len(tensor.shape); ws[pad_dim] = slice(None, Bv)
    p[tuple(ws)] = tensor
    return p[tuple(ws)]


def _cstrides(shape):
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1): s[i] = s[i+1] * shape[i+1]
    return tuple(s)


def _run(q_npu, k_npu, v_npu, pa, q_layout, act_q, act_kv, bt_npu, bs, scale,
         q_n, kv_n, q_rope_npu=None, k_rope_npu=None, d=128, dr=0):
    torch_npu.npu.set_device(DEVICE_ID); torch.npu.synchronize()
    aq = torch.tensor(act_q, dtype=torch.int64)
    akv = torch.tensor(act_kv, dtype=torch.int64)
    if dr > 0:
        r, _ = torch_npu.npu_fused_infer_attention_score_v2(
            q_npu, k_npu, v_npu,
            query_rope=q_rope_npu, key_rope=k_rope_npu,
            actual_seq_qlen=aq, actual_seq_kvlen=akv,
            num_query_heads=q_n, num_key_value_heads=kv_n,
            input_layout=q_layout, softmax_scale=scale,
            pre_tokens=65535, next_tokens=65535,
            block_table=bt_npu if pa else None,
            block_size=bs if pa else 0, out_dtype=DTYPE)
    else:
        r, _ = torch_npu.npu_fused_infer_attention_score(
            q_npu, k_npu, v_npu,
            actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
            num_heads=q_n, num_key_value_heads=kv_n,
            input_layout=q_layout, scale=scale,
            pre_tokens=65535, next_tokens=65535,
            block_table=bt_npu if pa else None,
            block_size=bs if pa else 0)
    torch.npu.synchronize()
    return r


def _check(ref, out, tag):
    d = (out.cpu().float() - ref.cpu().float()).abs()
    mx, mn = d.max().item(), d.mean().item()
    ok = (d <= 0.05).sum().item(); t = d.numel()
    logger.info(f"  [{tag}] max={mx:.6f} mean={mn:.6f} pass={ok}/{t} {ok/t:.4f}")
    assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=0.05, atol=0.05), f"[{tag}] FAIL max={mx}"


def _run_one(cfg, expect_pass):
    B = cfg["B"]; N_q = cfg["N_q"]; N_kv = cfg["N_kv"]
    pa = cfg["pa"]; fmt = cfg["fmt"]; ql = cfg["ql"]; nc = cfg["nc"]
    kv_seq = cfg["kv_seq"]; bs = cfg["bs"]; q_seq = cfg["q_seq"]
    d = cfg["d"]; dr = cfg.get("dr", 0); dv = cfg.get("dv", d)

    act_q = [q_seq] * B; act_kv = [kv_seq] * B
    scale = 1.0 / math.sqrt(d + dr)

    bt, total_b = None, 0
    if pa: bt, total_b = _block_table(B, act_kv, bs)
    else:   total_b = B
    bt_npu = bt.npu() if bt is not None else None

    torch.manual_seed(42)
    q_bnsd = torch.randn(B, N_q, q_seq, d, dtype=DTYPE)
    if ql == "BSND": q_bnsd = q_bnsd.permute(0, 2, 1, 3).contiguous()
    q_npu = q_bnsd.npu()

    q_rope, k_rope = None, None
    if dr > 0:
        torch.manual_seed(300); qr = torch.randn(B, N_q, q_seq, dr, dtype=DTYPE)
        torch.manual_seed(400); kr = torch.randn(B, N_kv, kv_seq, dr, dtype=DTYPE)
        if ql == "BSND": qr = qr.permute(0, 2, 1, 3).contiguous()
        q_rope = qr.npu(); k_rope = kr.npu()

    kv_shape = _kv_shape(fmt, total_b, bs, d, N_kv, B, kv_seq) if pa else _kv_shape("none", total_b, bs, d, N_kv, B, kv_seq)
    k_ref, v_ref = _make_kv(kv_shape)

    ref_out = _run(q_npu, k_ref.npu(), v_ref.npu(), pa, ql, act_q, act_kv,
                   bt_npu, bs, scale, N_q, N_kv, q_rope, k_rope, d, dr)

    k_nc = _make_nc(k_ref.npu(), nc); v_nc = _make_nc(v_ref.npu(), nc)
    k_rope_nc = _make_nc(k_rope, nc) if dr > 0 and nc != "none" else k_rope
    if nc != "none":
        cs = _cstrides(k_nc.shape)
        assert k_nc.stride(0) != cs[0], f"{cfg['id']}: dim0 not non-contiguous"
        if nc == "dim01": assert k_nc.stride(1) != cs[1], f"{cfg['id']}: dim1 not non-contiguous"
        logger.info(f"  k contig={k_nc.is_contiguous()} stride={k_nc.stride()}")

    if not expect_pass:
        with pytest.raises(Exception) as ei:
            _run(q_npu, k_nc, v_nc, pa, ql, act_q, act_kv, bt_npu, bs, scale,
                 N_q, N_kv, k_rope_nc, k_rope, d, dr)
        logger.info(f"  [{cfg['id']}] ERROR: {ei.value}")
        return

    out = _run(q_npu, k_nc, v_nc, pa, ql, act_q, act_kv, bt_npu, bs, scale,
               N_q, N_kv, k_rope_nc, k_rope, d, dr)
    _check(ref_out, out, cfg["id"])


# ==============================================================================
# 用例 — 每个用例独立 B/N_q/N_kv/kv_seq/block_size/q_seq/d
# ==============================================================================
NORMAL_CASES = [
    # === BnNBsD: dim0 / dim01 ===
    {"id": "BnNBsD-GQA-contig",  "B":1, "N_q":4, "N_kv":2, "pa":True,  "fmt":"BnNBsD", "ql":"BNSD", "nc":"none",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    {"id": "BnNBsD-GQA-dim0",    "B":1, "N_q":4, "N_kv":2, "pa":True,  "fmt":"BnNBsD", "ql":"BNSD", "nc":"dim0",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    {"id": "BnNBsD-GQA-dim01",   "B":1, "N_q":4, "N_kv":2, "pa":True,  "fmt":"BnNBsD", "ql":"BNSD", "nc":"dim01",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    # === BnBsH: dim0 only ===
    {"id": "BnBsH-GQA-contig",   "B":2, "N_q":8, "N_kv":2, "pa":True,  "fmt":"BnBsH",  "ql":"BNSD", "nc":"none",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    {"id": "BnBsH-GQA-dim0",     "B":2, "N_q":8, "N_kv":2, "pa":True,  "fmt":"BnBsH",  "ql":"BNSD", "nc":"dim0",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    # === NZ: dim0 / dim01 ===
    {"id": "NZ-GQA-contig",      "B":1, "N_q":1, "N_kv":1, "pa":True,  "fmt":"NZ",     "ql":"BNSD", "nc":"none",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    {"id": "NZ-GQA-dim0",        "B":1, "N_q":1, "N_kv":1, "pa":True,  "fmt":"NZ",     "ql":"BNSD", "nc":"dim0",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    {"id": "NZ-GQA-dim01",       "B":1, "N_q":1, "N_kv":1, "pa":True,  "fmt":"NZ",     "ql":"BNSD", "nc":"dim01",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    # === 非PA: 连续 ===
    {"id": "NonPA-GQA-contig",   "B":3, "N_q":6, "N_kv":3, "pa":False, "fmt":"none",   "ql":"BNSD", "nc":"none",
     "kv_seq":1024, "bs":0, "q_seq":8, "d":128},
    # === rope_split: D=128+rope64, V=128 ===
    {"id": "BnNBsD-ROPE-dim0",   "B":1, "N_q":4, "N_kv":2, "pa":True,  "fmt":"BnNBsD", "ql":"BNSD", "nc":"dim0",
     "kv_seq":1024, "bs":512, "q_seq":4, "d":128, "dr":64, "dv":128},
    {"id": "BnNBsD-ROPE-dim01",  "B":1, "N_q":4, "N_kv":2, "pa":True,  "fmt":"BnNBsD", "ql":"BNSD", "nc":"dim01",
     "kv_seq":1024, "bs":512, "q_seq":4, "d":128, "dr":64, "dv":128},
    # === rope_combine: D=192, V=128 ===
    {"id": "BnNBsD-d192-dim0",   "B":1, "N_q":2, "N_kv":1, "pa":True,  "fmt":"BnNBsD", "ql":"BNSD", "nc":"dim0",
     "kv_seq":1024, "bs":512, "q_seq":4, "d":192, "dv":128},
]

ERROR_CASES = [
    # BBND 不支持 dim1 非连续
    {"id": "BnBsH-GQA-dim01-ERR", "B":2, "N_q":8, "N_kv":2, "pa":True,  "fmt":"BnBsH",  "ql":"BNSD", "nc":"dim01",
     "kv_seq":2048, "bs":512, "q_seq":4, "d":128},
    # 非PA 不支持非连续
    {"id": "NonPA-dim0-ERR",      "B":3, "N_q":6, "N_kv":3, "pa":False, "fmt":"none",   "ql":"BNSD", "nc":"dim0",
     "kv_seq":1024, "bs":0, "q_seq":8, "d":128},
    {"id": "NonPA-dim01-ERR",     "B":3, "N_q":6, "N_kv":3, "pa":False, "fmt":"none",   "ql":"BNSD", "nc":"dim01",
     "kv_seq":1024, "bs":0, "q_seq":8, "d":128},
]


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
        try: _run_one(c, True); logger.info(f"  PASS")
        except Exception as e: logger.info(f"  FAIL: {e}"); ok = False
    for c in ERROR_CASES:
        try: _run_one(c, False); logger.info(f"  PASS")
        except Exception as e: logger.info(f"  FAIL: {e}"); ok = False
    logger.info("\n全部通过!" if ok else "\n存在失败!")
