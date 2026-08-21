#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""精度对比 — 阈值与判定策略与 flash_attn pytests 的 utils/compare.py 一致（D23）。

单元素: |golden - dev| ≤ max(|golden| * rtol, atol)
  fp16: rtol=0.005, atol=0.000025
  bf16: rtol=0.0078125, atol=0.0001
整体:   超阈值元素占比 ≤ 0.5%（FAIL_RATIO_LIMIT）
"""

import torch

RATIO_THRESHOLD = 0.005
FAIL_RATIO_LIMIT = 0.005


def get_tolerance(output_dtype):
    # bf16: atol=0.0001, rtol=0.0078125;  fp16(默认): atol=0.000025, rtol=0.005
    if output_dtype == "bf16":
        return 0.0001, 0.0078125
    return 0.000025, 0.005


def check_result(test_name, expect, result, except_label="CPU", comp_label="NPU",
                 verbose_diff=False, atol=0.000025, rtol=0.005):
    SEP = "─" * 64
    print(f"\n┌{SEP}┐")
    print(f"│  精度报告: {test_name}  ({except_label} vs {comp_label})")
    print(f"├{SEP}┤")
    if expect.shape != result.shape:
        print(f"│  [ERROR] shape不匹配: expect={tuple(expect.shape)}  {comp_label}={tuple(result.shape)}")
        print(f"└{SEP}┘")
        return False
    ef = expect.float()
    rf = result.float()
    diff = torch.abs(ef - rf)
    ref_abs = torch.abs(ef)
    rel_err = diff / (ref_abs + 1e-6)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()
    fail_mask = ~torch.isclose(ef, rf, rtol=rtol, atol=atol)
    fail_cnt = int(fail_mask.sum().item())
    total = expect.numel()
    fail_ratio = fail_cnt / total
    passed = fail_ratio <= FAIL_RATIO_LIMIT
    print(f"│  Shape       : {tuple(expect.shape)}")
    print(f"│  MaxAbsErr   : {max_abs:.8f}")
    print(f"│  MeanAbsErr  : {mean_abs:.8f}")
    print(f"│  MaxRelErr   : {max_rel:.8f}")
    print(f"│  MeanRelErr  : {mean_rel:.8f}")
    print(f"│  FailElems   : {fail_cnt} / {total}  ({fail_ratio*100:.4f}%)")
    print(f"│  Threshold   : atol={atol}  rtol={rtol*100:.2f}%  failRatio≤{FAIL_RATIO_LIMIT*100:.2f}%")
    print(f"│  结论        : {'✓ PASS' if passed else '✗ FAIL'}")
    if fail_cnt > 0:
        print(f"├{SEP}┤")
        if verbose_diff:
            all_idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).tolist()
            print(f"│  全部 {len(all_idxs)} 个超阈值元素 (rtol={rtol*100:.2f}%):")
            print(f"│  {'idx':>10}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>12}")
            for i in all_idxs[:200]:
                print(f"│  {i:>10}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>12.6f}")
        else:
            idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)[:10].tolist()
            print(f"│  前{len(idxs)}个不通过元素:")
            print(f"│  {'idx':>8}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>10}")
            for i in idxs:
                print(f"│  {i:>8}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>10.6f}")
    print(f"└{SEP}┘")
    stats = {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "fail_cnt": fail_cnt,
        "total": total,
        "fail_ratio": fail_ratio,
    }
    return stats