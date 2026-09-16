# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""Run against rebuilt host/AICPU/kernel/custom_ops on Ascend 950:

python test_layout_metadata.py

Missing torch_npu/custom_ops/device is an error, not a passing skipped test.
"""
import unittest

import torch
import torch_npu
import custom_ops  # noqa: F401 -- registers both NPU and Meta implementations


def metadata(sparse, layout, block=128, **kwargs):
    options = dict(batch_size=2, sparse_block_size_q=block, sparse_block_size_k=block,
                   quant_mode=2, mask_mode=0, layout_q=layout, layout_kv="PA_BNBD")
    options.update(kwargs)
    return torch.ops.custom.npu_quant_block_sparse_attn_metadata(sparse, 4, 2, 128, **options)


class LayoutMetadataTest(unittest.TestCase):
    def test_ge_metadata_matches_eager(self):
        # The existing golden's ACLGraph path builds metadata outside capture.
        # Use the GE backend here so the metadata converter itself is exercised.
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        if not torch_npu.npu.is_available():
            self.fail("This regression requires an Ascend device and rebuilt custom_ops")

        class MetadataModule(torch.nn.Module):
            def __init__(self, layout):
                super().__init__()
                self.layout = layout

            def forward(self, sparse):
                return metadata(sparse, self.layout)

        sparse = (torch.arange(24, dtype=torch.int32).reshape(2, 4, 3) % 3).npu()
        for layout in ("TND", "BSND", "BNSD"):
            with self.subTest(layout=layout):
                eager = metadata(sparse, layout)
                torch_npu.npu.synchronize()
                compiled = torch.compile(
                    MetadataModule(layout), fullgraph=True, dynamic=False,
                    backend=torchair.get_npu_backend(compiler_config=CompilerConfig()),
                )
                actual = compiled(sparse)
                torch_npu.npu.synchronize()
                eager, actual = eager.cpu(), actual.cpu()
                self.assertEqual(actual.dtype, torch.int32)
                self.assertEqual(actual.shape, eager.shape)
                self.assertEqual(int(actual[0]), int(eager[0]))
                used = 8 + int(eager[0]) * 36 * 8 + 72 * 8
                torch.testing.assert_close(actual[:used], eager[:used], rtol=0, atol=0)

    def test_metadata_meta_contract(self):
        sparse = torch.empty((2, 4, 3), dtype=torch.int32, device="meta")
        for layout in ("BSND", "BNSD"):
            with self.subTest(layout=layout):
                self.assertEqual(metadata(sparse, layout).shape, (8 + 2 * 4 * 36 * 8 + 72 * 8,))
                for field in ("cu_seqlens_q", "seqused_q"):
                    for size in (0, 3):
                        with self.assertRaisesRegex(RuntimeError, "requires.*None"):
                            metadata(sparse, layout, **{field: torch.empty(size, dtype=torch.int32, device="meta")})
                with self.assertRaises(RuntimeError):
                    metadata(sparse, layout, quant_mode=1)
                with self.assertRaises(RuntimeError):
                    metadata(sparse, layout, layout_kv="BSND")
                with self.assertRaises(RuntimeError):
                    metadata(torch.empty((2, 3), dtype=torch.int32, device="meta"), layout)

    def test_main_meta_shapes_and_scale_contract(self):
        for layout in ("BSND", "BNSD"):
            shape = (2, 129, 4, 128) if layout == "BSND" else (2, 4, 129, 128)
            q = torch.empty(shape, dtype=torch.float8_e4m3fn, device="meta")
            scale = torch.empty((*shape[:-1], 2, 2), dtype=torch.float8_e8m0fnu, device="meta")
            kv = torch.empty((6, 2, 128, 128), dtype=q.dtype, device="meta")
            indices = torch.empty((2, 4, 2, 3), dtype=torch.int32, device="meta")
            sparse = torch.empty((2, 4, 2), dtype=torch.int32, device="meta")

            def invoke(qscale, **kwargs):
                return torch.ops.custom.npu_quant_block_sparse_attn(
                    q, kv, kv, qscale, scale, scale, None, indices, sparse, None, 0.125, 128, 128,
                    layout_q=layout, layout_out=layout, quant_mode=2, mask_mode=0, **kwargs)

            for lse in (False, True):
                out, stats = invoke(scale, return_softmax_lse=lse)
                self.assertEqual(out.shape, shape)
                self.assertEqual(stats.shape, (2, 4, 129) if lse else (0,))
            with self.assertRaisesRegex(RuntimeError, "q_descale"):
                invoke(torch.empty((258, 4, 2, 2), dtype=scale.dtype, device="meta"))
            with self.assertRaisesRegex(RuntimeError, "contiguous"):
                invoke(scale.transpose(-1, -2))
            with self.assertRaisesRegex(RuntimeError, "q_descale"):
                invoke(torch.empty((*shape[:-1], 3, 2), dtype=scale.dtype, device="meta"))

    def test_npu_metadata_and_excess_sparse_rows(self):
        if not torch_npu.npu.is_available():
            self.fail("This regression requires an Ascend device and rebuilt custom_ops")
        torch.manual_seed(11704)
        sq, kv_len, heads, kv_heads, dim = 129, 257, 4, 2, 128
        q_cpu = torch.randn(2, sq, heads, dim) * 0.1
        for block in (64, 128):
            pages_per_batch = (kv_len + block - 1) // block
            pages = 2 * pages_per_batch
            cache = (torch.randn(pages, 2, kv_heads, block, dim) * 0.1).to(torch.float8_e4m3fn).npu()
            k, v = cache[:, 0], cache[:, 1]
            ks = torch.full((pages, kv_heads, block, 2, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
            vs = torch.full((pages, kv_heads, block // 64, dim, 2), 127,
                            dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
            table = torch.arange(pages, dtype=torch.int32).reshape(2, pages_per_batch).npu()
            used_kv = torch.tensor([kv_len, kv_len], dtype=torch.int32).npu()
            cu_q = torch.tensor([0, sq, 2 * sq], dtype=torch.int32).npu()
            qb = (sq + block - 1) // block
            counts = (torch.arange(2 * heads * qb).reshape(2, heads, qb) % 3).to(torch.int32)
            for mask_mode in (0, 3):
                mask = (torch.triu(torch.ones((2048, 2048), dtype=torch.uint8), diagonal=1).npu()
                        if mask_mode == 3 else None)
                for with_lse in (False, True):
                    def invoke(layout, extra_rows):
                        layout_counts = torch.cat((counts, torch.full((2, heads, extra_rows), 2, dtype=torch.int32)), dim=2)
                        sparse = layout_counts.npu()
                        indices = torch.arange(3, dtype=torch.int32).expand(2, heads, qb + extra_rows, 3).contiguous().npu()
                        q = q_cpu.reshape(2 * sq, heads, dim) if layout == "TND" else q_cpu
                        if layout == "BNSD":
                            q = q.transpose(1, 2).contiguous()
                        q = q.to(torch.float8_e4m3fn).npu()
                        qs = torch.full((*q.shape[:-1], 2, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
                        lengths = cu_q if layout == "TND" else None
                        meta = metadata(sparse, layout, block, cu_seqlens_q=lengths, seqused_kv=used_kv,
                                        mask_mode=mask_mode)
                        out, lse = torch.ops.custom.npu_quant_block_sparse_attn(
                            q, k, v, qs, ks, vs, None, indices, sparse, mask, 0.125, block, block,
                            cu_seqlens_q=lengths, seqused_kv=used_kv, block_table=table, metadata=meta,
                            layout_q=layout, layout_out=layout, quant_mode=2, mask_mode=mask_mode,
                            return_softmax_lse=with_lse)
                        torch_npu.npu.synchronize()
                        return out.cpu(), lse.cpu(), meta.cpu()

                    baseline, baseline_lse, baseline_meta = invoke("TND", 0)
                    for layout in ("BSND", "BNSD"):
                        for extra in (0, 5):
                            with self.subTest(layout=layout, block=block, mask=mask_mode, lse=with_lse, extra=extra):
                                out, lse, meta = invoke(layout, extra)
                                if layout == "BNSD":
                                    out = out.transpose(1, 2)
                                torch.testing.assert_close(out.reshape_as(baseline), baseline, rtol=0, atol=0)
                                if with_lse:
                                    torch.testing.assert_close(lse.transpose(1, 2).reshape_as(baseline_lse),
                                                               baseline_lse, rtol=0, atol=0)
                                if extra == 0:
                                    # Only the active section region is initialized by AICPU.
                                    used = 8 + int(meta[0]) * 36 * 8 + 72 * 8
                                    torch.testing.assert_close(meta[:used], baseline_meta[:used], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
