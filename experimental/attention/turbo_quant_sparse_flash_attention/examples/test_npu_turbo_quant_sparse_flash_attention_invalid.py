#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TurboQuantSparseFlashAttention 非法参数用例。

覆盖算子声明与 host 校验的支持边界：属性取值、layout 组合、dtype 与 shape。
每条用例都断言调用被拒绝，避免「声明接受但实现不支持」的组合被静默放行。
合法基线同时被断言可正常执行，防止校验收紧过头把正常用法一并拦掉。
"""

import unittest

import torch
import torch_npu  # noqa: F401

import cann_ops_transformer  # noqa: F401  注册 turbo_quant_sparse_flash_attention

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    TestCase = unittest.TestCase
    run_tests = unittest.main

from test_npu_turbo_quant_sparse_flash_attention import (  # noqa: E402
    BLOCK,
    DN,
    NHEAD,
    build_case,
)

INT64_MAX = 9223372036854775807


class TestTurboQuantSparseFlashAttentionInvalid(TestCase):
    """非法参数应被 host 校验拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.c = build_case(1, 512, 256, 9, torch.bfloat16, "normal")

    def _call(self, q=None, k=None, v=None, **over):
        c = self.c
        kw = dict(
            key_dequant_scale=c["ks"],
            value_dequant_scale=c["ks"],
            block_table=c["bt"],
            actual_seq_lengths_query=c["asq"],
            actual_seq_lengths_kv=c["ask"],
            scale_value=c["scale_value"],
            key_quant_mode=3,
            value_quant_mode=3,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=INT64_MAX,
            next_tokens=INT64_MAX,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
            return_softmax_lse=False,
        )
        kw.update(over)
        out = torch.ops.cann_ops_transformer.turbo_quant_sparse_flash_attention(
            c["q"] if q is None else q,
            c["kv"] if k is None else k,
            c["kv"] if v is None else v,
            c["si"],
            **kw,
        )
        torch.npu.synchronize()
        return out

    def _assert_rejected(self, tag, **kw):
        with self.subTest(case=tag):
            try:
                self._call(**kw)
            except Exception:  # noqa: BLE001  host 校验失败经 aclnn 抛出
                print("  [PASS] %s 已拒绝" % tag, flush=True)
                return
            self.fail("%s 未被拒绝" % tag)

    def test_baseline_is_valid(self):
        """合法基线必须能跑通，否则「全部拒绝」会让下列断言失去意义。"""
        out, _, _ = self._call()
        self.assertEqual(list(out.shape), [1, NHEAD, DN])
        print("  [PASS] 合法基线 out.shape=%s" % list(out.shape), flush=True)

    def test_invalid_attr_value(self):
        """属性取值：当前实现对每个属性只有唯一合法值。"""
        self._assert_rejected("tile_size=64", tile_size=64)
        self._assert_rejected("tile_size=256", tile_size=256)
        self._assert_rejected("rope_head_dim=32", rope_head_dim=32)
        self._assert_rejected("key_quant_mode=1", key_quant_mode=1)
        self._assert_rejected("value_quant_mode=1", value_quant_mode=1)
        self._assert_rejected("attention_mode=0", attention_mode=0)
        self._assert_rejected("quant_scale_repo_mode=0", quant_scale_repo_mode=0)

    def test_invalid_layout(self):
        """layout：query 仅 TND，KV 仅 PA_BSND / BSND。"""
        self._assert_rejected("layout_query=BSND", layout_query="BSND")
        self._assert_rejected("layout_query=BNSD", layout_query="BNSD")
        self._assert_rejected("layout_kv=TND", layout_kv="TND")
        self._assert_rejected("layout_kv=BSND", layout_kv="BSND")

    def test_invalid_dtype(self):
        """dtype：query 仅 BFLOAT16；actual_seq_lengths 仅 INT32。"""
        c = self.c
        self._assert_rejected("query=float16", q=c["q"].to(torch.float16))
        self._assert_rejected("query=float32", q=c["q"].float())
        self._assert_rejected(
            "actual_seq_lengths_kv=int64", actual_seq_lengths_kv=c["ask"].long()
        )

    def test_invalid_shape(self):
        """shape：value 须与 key 完全一致；block_table 首维须等于 batch。"""
        c = self.c
        half = c["kv"][: max(1, c["kv"].shape[0] // 2)]
        self._assert_rejected("value block 数减半", v=half)
        self._assert_rejected(
            "block_table 首维不等于 batch",
            block_table=torch.cat([c["bt"], c["bt"]], dim=0),
        )

    def test_softmax_lse_shape(self):
        """softmax_max / softmax_sum 的 shape 随 return_softmax_lse 变化。

        为 False 时算子不产出 LSE，两个输出为空张量；为 True 时 shape 为 [T, 1, N1]。
        两者的 dtype 恒为 FLOAT32，且必须始终同形，避免下游按其中之一取形出错。
        """
        c = self.c
        t_size, n1 = c["q"].shape[0], c["q"].shape[1]
        for flag in (False, True):
            with self.subTest(return_softmax_lse=flag):
                _, smax, ssum = self._call(return_softmax_lse=flag)
                self.assertEqual(smax.dtype, torch.float32)
                self.assertEqual(ssum.dtype, torch.float32)
                self.assertEqual(list(smax.shape), list(ssum.shape))
                if flag:
                    self.assertEqual(list(smax.shape), [t_size, 1, n1])
                else:
                    self.assertEqual(smax.numel(), 0)
                print(
                    "  [PASS] return_softmax_lse=%-5s smax.shape=%s"
                    % (flag, list(smax.shape)),
                    flush=True,
                )


if __name__ == "__main__":
    run_tests()
