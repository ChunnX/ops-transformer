# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""BAND regression: python test_band.py (requires matching custom package and torch extension)."""

import math
import unittest

import torch

from impl.flash_mla_with_kvcache_golden import FlashMlaWithKvcacheGolden


def make_case(q_lens, kv_lens, heads, dtype):
    generator = torch.Generator().manual_seed(1729)
    q = torch.randn(sum(q_lens), heads, 576, generator=generator).to(dtype)
    page_counts = [(n + 127) // 128 for n in kv_lens]
    # A non-identity block table catches accidental dependence on physical page order.
    pages = torch.randn(sum(page_counts), 128, 576, generator=generator).to(dtype)
    order = torch.randperm(len(pages), generator=generator)
    physical = torch.empty_like(pages)
    physical[order] = pages
    table = torch.zeros(len(q_lens), max(page_counts), dtype=torch.int32)
    offset = 0
    for b, count in enumerate(page_counts):
        table[b, :count] = order[offset : offset + count].int()
        offset += count
    k = physical.reshape(-1, 128, 36, 16).permute(0, 2, 1, 3).unsqueeze(1).contiguous()
    cu = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32
    )
    lens = torch.tensor(kv_lens, dtype=torch.int32)
    return q, k, table, cu, lens, pages


def reference(q, pages, q_lens, kv_lens, mode, left, right):
    out = torch.zeros(q.shape[1], q.shape[0], 512)
    lse = torch.full((q.shape[1], q.shape[0]), float("inf"))
    q_start = page_start = 0
    for sq, skv in zip(q_lens, kv_lens):
        page_count = (skv + 127) // 128
        kv = pages[page_start : page_start + page_count].flatten(0, 1)[:skv].float()
        qb = q[q_start : q_start + sq].transpose(0, 1).float()
        scores = qb @ kv.T / math.sqrt(576)
        pos = torch.arange(sq)[:, None] + skv - sq
        key = torch.arange(skv)[None, :]
        valid = torch.ones(sq, skv, dtype=torch.bool)
        if mode == 3:
            valid &= key <= pos
        elif mode == 4:
            if left >= 0:
                valid &= key >= pos - left
            if right >= 0:
                valid &= key <= pos + right
        scores.masked_fill_(~valid, float("-inf"))
        out[:, q_start : q_start + sq] = scores.softmax(-1).nan_to_num(0) @ kv[:, :512]
        batch_lse = scores.logsumexp(-1)
        batch_lse[:, ~valid.any(-1)] = float("inf")
        lse[:, q_start : q_start + sq] = batch_lse
        q_start += sq
        page_start += page_count
    return out, lse


class BandGoldenTest(unittest.TestCase):
    def test_inclusive_right_aligned_bounds(self):
        q = torch.arange(3)
        k = torch.arange(6)
        mask = FlashMlaWithKvcacheGolden._build_mask_mode_mask(q, k, 3, 4, 2, 1)
        self.assertEqual(
            (~mask).int().tolist(),
            [
                [0, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1],
            ],
        )

    def test_full_and_tiled_golden(self):
        q_lens, kv_lens = [3, 8], [129, 3]
        q, k, table, cu, lens, pages = make_case(q_lens, kv_lens, 64, torch.bfloat16)
        for left, right in [(0, 0), (2, 1), (-1, 0), (0, -1), (-1, -1)]:
            expected, expected_lse = reference(
                q, pages, q_lens, kv_lens, 4, left, right
            )
            for crossover in [0, 10**12]:
                with self.subTest(window=(left, right), crossover=crossover):
                    golden = FlashMlaWithKvcacheGolden()
                    golden.TILED_CROSSOVER = crossover
                    out, lse = golden.forward_golden(
                        q,
                        k,
                        block_table=table,
                        cache_seqlens=lens,
                        cu_seqlens_q=cu,
                        mask_mode=4,
                        win_left=left,
                        win_right=right,
                        layout_q="TND",
                        layout_kv="PA_NZ",
                        layout_out="NTD",
                        return_softmax_lse=True,
                        device_mode="cpu",
                    )
                    torch.testing.assert_close(
                        out.float(), expected, atol=0.035, rtol=0
                    )
                    torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=0)


class BandNpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch_npu is required")
        from cann_ops_transformer.ops import (
            flash_mla_with_kvcache,
            flash_mla_with_kvcache_metadata,
        )

        cls.mla = staticmethod(flash_mla_with_kvcache)
        cls.meta = staticmethod(flash_mla_with_kvcache_metadata)
        torch.npu.set_device(0)
        cls.mask = torch.triu(
            torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1
        )

    def run_case(self, q_lens, kv_lens, heads, dtype, mode, left=-1, right=-1):
        q, k, table, cu, lens, pages = make_case(q_lens, kv_lens, heads, dtype)
        kwargs = dict(
            cache_seqlens=lens.npu(),
            cu_seqlens_q=cu.npu(),
            mask_mode=mode,
            win_left=left,
            win_right=right,
            layout_q="TND",
        )
        metadata = self.meta(num_heads_q=heads, num_heads_kv=1, **kwargs)
        out, lse = self.mla(
            q.npu(),
            k.npu(),
            block_table=table.npu(),
            metadata=metadata,
            attn_mask=self.mask if mode else None,
            softmax_scale=576**-0.5,
            layout_kv="PA_NZ",
            layout_out="NTD",
            return_softmax_lse=True,
            **kwargs,
        )
        expected, expected_lse = reference(q, pages, q_lens, kv_lens, mode, left, right)
        torch.testing.assert_close(
            out.cpu().float(),
            expected,
            atol=0.01 if dtype == torch.float16 else 0.035,
            rtol=0,
        )
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=1e-4, rtol=0)
        return out.cpu(), lse.cpu()

    def test_band_boundaries_and_flashdecode(self):
        cases = [
            (3, 113, 0, 0),
            (5, 129, 2, 1),
            (8, 3, 0, 0),
            (8, 3, 2, 1),
            (3, 257, 111, 0),
            (3, 257, 112, 1),
            (3, 4097, 127, 3),
            (3, 4097, -1, 0),
            (3, 257, 0, -1),
            (3, 113, -1, -1),
            (1, 1, 0, 0),
            (1, 113, 15, 16),
            (1, 32768, 16384, 0),
        ]
        for dtype in [torch.float16, torch.bfloat16]:
            for i, (sq, skv, left, right) in enumerate(cases):
                with self.subTest(dtype=dtype, sq=sq, skv=skv, left=left, right=right):
                    self.run_case(
                        [sq], [skv], 96 if i % 2 == 0 else 64, dtype, 4, left, right
                    )

    def test_ragged_batches_and_mode_equivalence(self):
        for dtype in [torch.float16, torch.bfloat16]:
            for window in [(0, 0), (2, 1), (-1, 0), (-1, -1)]:
                with self.subTest(dtype=dtype, window=window):
                    result = self.run_case(
                        [2, 0, 3, 8], [0, 113, 129, 3], 64, dtype, 4, *window
                    )
                    if window[0] == -1:
                        mode = 3 if window[1] == 0 else 0
                        baseline = self.run_case(
                            [2, 0, 3, 8], [0, 113, 129, 3], 64, dtype, mode
                        )
                        for actual, expected in zip(result, baseline):
                            torch.testing.assert_close(
                                actual, expected, atol=0.01, rtol=0
                            )

    def test_invalid_windows_and_missing_mask(self):
        q, k, table, cu, lens, _ = make_case([1], [113], 96, torch.float16)
        kwargs = dict(cache_seqlens=lens.npu(), cu_seqlens_q=cu.npu(), layout_q="TND")
        metadata = self.meta(num_heads_q=96, num_heads_kv=1, mask_mode=4, **kwargs)
        for mode, left, right in [
            (4, -2, 0),
            (4, 0, -2),
            (4, 2**31, 0),
            (4, 0, 2**31),
            (0, 0, -1),
            (3, -1, 0),
            (2, -1, -1),
        ]:
            with self.subTest(mode=mode, left=left, right=right):
                with self.assertRaises(RuntimeError):
                    self.meta(
                        num_heads_q=96,
                        num_heads_kv=1,
                        mask_mode=mode,
                        win_left=left,
                        win_right=right,
                        **kwargs,
                    )
                with self.assertRaises(RuntimeError):
                    self.mla(
                        q.npu(),
                        k.npu(),
                        block_table=table.npu(),
                        metadata=metadata,
                        attn_mask=self.mask if mode else None,
                        layout_kv="PA_NZ",
                        layout_out="NTD",
                        mask_mode=mode,
                        win_left=left,
                        win_right=right,
                        **kwargs,
                    )
        with self.assertRaises(RuntimeError):
            self.mla(
                q.npu(),
                k.npu(),
                block_table=table.npu(),
                metadata=metadata,
                layout_kv="PA_NZ",
                layout_out="NTD",
                mask_mode=4,
                **kwargs,
            )


if __name__ == "__main__":
    torch.set_num_threads(10)
    unittest.main(verbosity=2)
