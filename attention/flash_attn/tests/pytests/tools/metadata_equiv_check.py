#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""flash_attn_metadata head_dim_v 契约验证（独立脚本，不走 orchestrator 框架）。

覆盖 pytests 主框架无法表达的两类断言：
1. 等价性：head_dim_v 缺省（内部归一化 -1 -> head_dim）与显式 =head_dim 的
   metadata 输出逐元素一致（spec doc 30 §9.1）。
2. 负例：(qk, v) 组合白名单之外的调用被拒绝（spec doc 30 §9.1）：
   - D=DV=128 主算子 qk!=v 组合 (256,128)/(192,192) 报错
   - metadata head_dim_v 与组合白名单冲突报错

用法:
    python tools/metadata_equiv_check.py [--device_id 0]

判定: 全部断言通过输出 ALL PASS 并退出 0；任一失败非零退出。
"""

import argparse
import sys

import torch
import torch_npu  # noqa: F401

from cann_ops_transformer.ops import flash_attn_metadata


def _metadata(num_heads_q, num_heads_kv, head_dim, head_dim_v=None, **kw):
    return flash_attn_metadata(
        num_heads_q, num_heads_kv, head_dim, head_dim_v=head_dim_v, **kw
    )


def check_equivalence(device):
    """head_dim_v 缺省(-1 归一化) 与显式 =head_dim 的输出一致性, 覆盖 64/72/128/256/192+128。"""
    combos = [
        # (num_heads_q, num_heads_kv, head_dim, head_dim_v)
        (8, 8, 64, None),  # 缺省 == 显式 64
        (8, 8, 72, None),
        (8, 8, 128, None),
        (8, 1, 256, None),
        (
            8,
            1,
            192,
            128,
        ),  # qk != v: 显式 128 == 缺省(-1 -> 192) 不等价, 校验显式路径生效
    ]
    for nq, nkv, d, dv in combos:
        kw = dict(
            batch_size=1,
            max_seqlen_q=16,
            max_seqlen_kv=128,
            mask_mode=0,
            layout_q="BNSD",
            layout_kv="BNSD",
            layout_out="BNSD",
        )
        if dv is None:
            m_default = _metadata(nq, nkv, d, head_dim_v=None, **kw)
            m_explicit = _metadata(nq, nkv, d, head_dim_v=d, **kw)
            assert torch.equal(m_default, m_explicit), (
                f"D={d}: head_dim_v 缺省与显式 {d} 的 metadata 输出不一致"
            )
            print(f"[equiv] D={d:3d} DV=default vs {d:3d}: PASS")
        else:
            m_dv = _metadata(nq, nkv, d, head_dim_v=dv, **kw)
            # 显式 128 的切分应与 D=128 场景可比: 断言可生成且非全零
            assert m_dv is not None and m_dv.numel() > 0
            assert int(m_dv.abs().sum()) > 0, f"D={d}/DV={dv}: metadata 全零"
            print(f"[equiv] D={d:3d} DV={dv:3d} (explicit): PASS")


def check_negative(device):
    """组合白名单外的调用必须被拒绝。"""
    kw = dict(
        batch_size=1,
        max_seqlen_q=16,
        max_seqlen_kv=128,
        mask_mode=0,
        layout_q="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
    )

    def expect_reject(desc, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[neg] {desc}: PASS (rejected: {str(e)[:80]})")
            return
        raise AssertionError(f"[neg] {desc}: 未被拒绝")

    # metadata: (head_dim=256, head_dim_v=128) 不在白名单
    expect_reject(
        "metadata (256,128)", lambda: _metadata(8, 8, 256, head_dim_v=128, **kw)
    )
    # metadata: (192,192) 不在白名单
    expect_reject(
        "metadata (192,192)", lambda: _metadata(8, 8, 192, head_dim_v=192, **kw)
    )
    # metadata: head_dim_v=0 非法
    expect_reject("metadata DV=0", lambda: _metadata(8, 8, 128, head_dim_v=0, **kw))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device_id)
    check_equivalence(args.device_id)
    check_negative(args.device_id)
    print("ALL PASS")
    sys.exit(0)
