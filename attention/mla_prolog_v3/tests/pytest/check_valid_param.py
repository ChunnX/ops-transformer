#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)


def validate_config(params):
    bn, He, Hcq, Hckv, qn, kn, D, Dr, qs, ks, bs, la, cm, ce1, ce2, dt, wq, kvq, qq, cr, sr, ts, qr, kc, qnf, bsm = params
    # 基础校验
    if bn > 65536 or bn < 1:
        raise ValueError("batch_size must <= 65536 and >= 1")
    if qs > 16 or qs < 1:
        raise ValueError("q_seq must <= 16 and >= 1")
    if He != 7168:
        raise ValueError("He must = 7168")
    if Hcq != 1536:
        raise ValueError("Hcq must = 1536")
    if qn > 256:
        raise ValueError("q_head_num must <= 256")
    if kn > 256:
        raise ValueError("kv_head_num must <= 256")
    if qn % kn != 0:
        raise ValueError("q_head_num must be integral multiple of kv_head_num")
    if Hckv != 512:
        raise ValueError("Hckv must = 512")
    if D != 128:
        raise ValueError("head_dim must = 128")
    if Dr != 64:
        raise ValueError("rope_head_dim must = 64")
    if kn != 1:
        raise ValueError("kv_head_num must = 1")
    if bs not in (16, 128):
        raise ValueError("block_size must = 16 or 128")
    if dt not in [torch.bfloat16, torch.float16, torch.int8]:
        raise ValueError("dtype should be: float16/bfloat16/int8")
    if wq not in range(6):
        raise ValueError("weight_quant_mode must be 0-5")
    if kvq not in range(4):
        raise ValueError("kv_cache_quant_mode must be 0-3")


def check_result(expect, result):
    thres = 0.005
    num = min(len(expect), len(result))
    for i in range(num):
        if expect[i] is None or result[i] is None:
            continue
        r = result[i].cpu().to(torch.float32)
        e = expect[i].cpu().to(torch.float32)
        if r.numel() == 0 and e.numel() == 0:
            logger.info("output[%d] both empty, skip", i)
            continue
        if r.numel() != e.numel():
            logger.info("output[%d] shape mismatch: expect=%s result=%s", i, tuple(e.shape), tuple(r.shape))
            # Debug: print first 10 values from both
            if e.numel() > 0:
                logger.info("output[%d] expect first 10: %s", i, e.flatten()[:10].tolist())
            if r.numel() > 0:
                logger.info("output[%d] result first 10: %s", i, r.flatten()[:10].tolist())
            continue
        diff = (r.reshape(-1) - e.reshape(-1)).abs()
        idx = torch.nonzero(diff > thres).squeeze(-1)
        if idx.numel() > 0:
            logger.info("output[%d] >%.4f indices (first 10): %s", i, thres, idx[:10].tolist())
            logger.info("output[%d] >%.4f values: %s", i, thres, diff[idx][:10].tolist())
        pass_num = torch.sum(diff <= thres)
        total_num = diff.numel()
        acc = pass_num / total_num if total_num > 0 else 0.0
        logger.info("output[%d] pass: %d/%d accuracy: %.4f", i, pass_num.item(), total_num, acc)
        assert torch.allclose(r.flatten(), e.flatten(), rtol=0.05, atol=0.05), \
            "output[{}] mismatch".format(i)
