# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run against rebuilt host/AICPU/kernel/custom_ops on Ascend 950:

python test_layout_metadata.py

Missing torch_npu/custom_ops/device is an error, not a passing skipped test.
"""

import unittest

import torch
import torch_npu
import custom_ops  # noqa: F401 -- registers both NPU and Meta implementations


def metadata(sparse, layout, block=128, **kwargs):
    options = dict(
        batch_size=2,
        sparse_block_size_q=block,
        sparse_block_size_k=block,
        quant_mode=2,
        mask_mode=0,
        layout_q=layout,
        layout_kv="PA_BNBD",
    )
    options.update(kwargs)
    return torch.ops.custom.npu_quant_block_sparse_attn_metadata(
        sparse, 4, 2, 128, **options
    )


class LayoutMetadataTest(unittest.TestCase):
    def test_padded_layout_validation_is_not_hidden_by_metadata(self):
        """Metadata and the main op must reject their own padded-layout inputs.

        The metadata API deliberately rejects Q-length tensors for BSND/BNSD.
        A valid metadata tensor must not mask equivalent invalid inputs when the
        main operator is called directly, because the two APIs can be invoked
        independently by eager and graph callers.
        """
        sparse = torch.empty((2, 4, 2), dtype=torch.int32, device="meta")
        kv = torch.empty((4, 2, 128, 128), dtype=torch.float8_e4m3fn, device="meta")
        kv_scale = torch.empty(
            (4, 2, 128, 2, 2), dtype=torch.float8_e8m0fnu, device="meta"
        )
        value_scale = torch.empty(
            (4, 2, 2, 128, 2), dtype=torch.float8_e8m0fnu, device="meta"
        )
        indices = torch.empty((2, 4, 2, 2), dtype=torch.int32, device="meta")
        table = torch.empty((2, 2), dtype=torch.int32, device="meta")

        for layout in ("BSND", "BNSD"):
            with self.subTest(layout=layout):
                shape = (2, 129, 4, 128) if layout == "BSND" else (2, 4, 129, 128)
                q = torch.empty(shape, dtype=torch.float8_e4m3fn, device="meta")
                scale = torch.empty(
                    (*shape[:-1], 2, 2), dtype=torch.float8_e8m0fnu, device="meta"
                )

                # Metadata owns validation of the sparse metadata contract.
                for field in ("cu_seqlens_q", "seqused_q"):
                    with self.assertRaisesRegex(RuntimeError, "requires.*None"):
                        metadata(
                            sparse,
                            layout,
                            **{
                                field: torch.empty(
                                    (0,), dtype=torch.int32, device="meta"
                                )
                            },
                        )
                with self.assertRaises(RuntimeError):
                    metadata(
                        torch.empty((2, 4, 0), dtype=torch.int32, device="meta"), layout
                    )

                valid_metadata = metadata(sparse, layout)

                def invoke(query=q, q_scale=scale, **kwargs):
                    options = dict(
                        block_table=table,
                        metadata=valid_metadata,
                        layout_q=layout,
                        layout_out=layout,
                        quant_mode=2,
                        mask_mode=0,
                    )
                    options.update(kwargs)
                    return torch.ops.custom.npu_quant_block_sparse_attn(
                        query,
                        kv,
                        kv,
                        q_scale,
                        kv_scale,
                        value_scale,
                        None,
                        indices,
                        sparse,
                        None,
                        0.125,
                        128,
                        128,
                        **options,
                    )

                # A valid metadata value proves that later failures come from
                # the main-op checks, not from the metadata builder.
                out, lse = invoke()
                self.assertEqual(tuple(out.shape), shape)
                self.assertEqual(tuple(lse.shape), (0,))

                with self.assertRaisesRegex(RuntimeError, "requires.*None"):
                    invoke(
                        cu_seqlens_q=torch.empty((0,), dtype=torch.int32, device="meta")
                    )
                with self.assertRaisesRegex(RuntimeError, "requires.*None"):
                    invoke(
                        seqused_q=torch.empty((2,), dtype=torch.int32, device="meta")
                    )
                with self.assertRaisesRegex(RuntimeError, "q_descale"):
                    invoke(
                        q_scale=torch.empty(
                            (*shape[:-1], 3, 2), dtype=scale.dtype, device="meta"
                        )
                    )
                with self.assertRaises(RuntimeError):
                    invoke(
                        query=torch.empty((2, 129, 128), dtype=q.dtype, device="meta")
                    )

    def test_npu_metadata_optional_inputs(self):
        """Exercise the af49ad74f None fix without launching the attention kernel."""
        if not torch_npu.npu.is_available():
            self.fail(
                "This regression requires an Ascend device and rebuilt custom_ops"
            )
        sparse = (torch.arange(24, dtype=torch.int32).reshape(2, 4, 3) % 3).npu()
        used_kv = torch.tensor([257, 193], dtype=torch.int32).npu()
        cu_q = torch.tensor([0, 129, 258], dtype=torch.int32).npu()
        baseline = None
        for layout in ("TND", "BSND", "BNSD"):
            print(f"[METADATA BEGIN] layout={layout}", flush=True)
            torch_npu.npu.synchronize()
            actual = metadata(
                sparse,
                layout,
                cu_seqlens_q=cu_q if layout == "TND" else None,
                seqused_q=None,
                seqused_kv=used_kv,
            )
            # This is the synchronization that failed before the optional-input fix.
            torch_npu.npu.synchronize()
            print(f"[METADATA SYNC PASS] layout={layout}", flush=True)
            actual = actual.cpu()
            self.assertEqual(actual.dtype, torch.int32)
            self.assertEqual(tuple(actual.shape), (8 + 2 * 4 * 36 * 8 + 72 * 8,))
            sections = int(actual[0])
            self.assertGreater(sections, 0)
            used = 8 + sections * 36 * 8 + 72 * 8
            self.assertLessEqual(used, actual.numel())
            if baseline is None:
                baseline = actual
            else:
                self.assertEqual(sections, int(baseline[0]))
                torch.testing.assert_close(
                    actual[:used], baseline[:used], rtol=0, atol=0
                )
                # Public explicit tensors remain invalid; accepting an internal
                # null-data placeholder must not relax this interface contract.
                for field in ("cu_seqlens_q", "seqused_q"):
                    for size in (0, 2):
                        with self.assertRaisesRegex(RuntimeError, "requires.*None"):
                            metadata(
                                sparse,
                                layout,
                                **{field: torch.zeros(size, dtype=torch.int32).npu()},
                            )
                torch_npu.npu.synchronize()
            print(f"[METADATA PASS] layout={layout} sections={sections}", flush=True)

    def test_ge_metadata_matches_eager(self):
        # The existing golden's ACLGraph path builds metadata outside capture.
        # Use the GE backend here so the metadata converter itself is exercised.
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        if not torch_npu.npu.is_available():
            self.fail(
                "This regression requires an Ascend device and rebuilt custom_ops"
            )

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
                    MetadataModule(layout),
                    fullgraph=True,
                    dynamic=False,
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
                self.assertEqual(
                    metadata(sparse, layout).shape, (8 + 2 * 4 * 36 * 8 + 72 * 8,)
                )
                for field in ("cu_seqlens_q", "seqused_q"):
                    for size in (0, 3):
                        with self.assertRaisesRegex(RuntimeError, "requires.*None"):
                            metadata(
                                sparse,
                                layout,
                                **{
                                    field: torch.empty(
                                        size, dtype=torch.int32, device="meta"
                                    )
                                },
                            )
                with self.assertRaises(RuntimeError):
                    metadata(sparse, layout, quant_mode=1)
                with self.assertRaises(RuntimeError):
                    metadata(sparse, layout, layout_kv="BSND")
                with self.assertRaises(RuntimeError):
                    metadata(
                        torch.empty((2, 3), dtype=torch.int32, device="meta"), layout
                    )

    def test_main_meta_shapes_and_scale_contract(self):
        for layout in ("BSND", "BNSD"):
            shape = (2, 129, 4, 128) if layout == "BSND" else (2, 4, 129, 128)
            q = torch.empty(shape, dtype=torch.float8_e4m3fn, device="meta")
            scale = torch.empty(
                (*shape[:-1], 2, 2), dtype=torch.float8_e8m0fnu, device="meta"
            )
            kv = torch.empty((6, 2, 128, 128), dtype=q.dtype, device="meta")
            indices = torch.empty((2, 4, 2, 3), dtype=torch.int32, device="meta")
            sparse = torch.empty((2, 4, 2), dtype=torch.int32, device="meta")

            def invoke(qscale, query=q, **kwargs):
                options = dict(
                    layout_q=layout, layout_out=layout, quant_mode=2, mask_mode=0
                )
                options.update(kwargs)
                return torch.ops.custom.npu_quant_block_sparse_attn(
                    query,
                    kv,
                    kv,
                    qscale,
                    scale,
                    scale,
                    None,
                    indices,
                    sparse,
                    None,
                    0.125,
                    128,
                    128,
                    **options,
                )

            for lse in (False, True):
                out, stats = invoke(scale, return_softmax_lse=lse)
                self.assertEqual(out.shape, shape)
                self.assertEqual(stats.shape, (2, 4, 129) if lse else (0,))
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertEqual(stats.dtype, torch.float32)
            with self.assertRaisesRegex(RuntimeError, "q_descale"):
                invoke(torch.empty((258, 4, 2, 2), dtype=scale.dtype, device="meta"))
            with self.assertRaisesRegex(RuntimeError, "contiguous"):
                invoke(scale.transpose(-1, -2))
            with self.assertRaisesRegex(RuntimeError, "q_descale"):
                invoke(
                    torch.empty((*shape[:-1], 3, 2), dtype=scale.dtype, device="meta")
                )
            with self.assertRaisesRegex(RuntimeError, "q_descale"):
                invoke(scale.transpose(1, 2).contiguous())
            with self.assertRaisesRegex(RuntimeError, "contiguous"):
                invoke(scale, query=q.transpose(1, 2))
            for field in ("cu_seqlens_q", "seqused_q"):
                for size in (0, 3):
                    with self.subTest(layout=layout, field=field, size=size):
                        with self.assertRaisesRegex(RuntimeError, "requires.*None"):
                            invoke(
                                scale,
                                **{
                                    field: torch.empty(
                                        size, dtype=torch.int32, device="meta"
                                    )
                                },
                            )
            for wrong_out in ("TND", "BNSD" if layout == "BSND" else "BSND"):
                with self.assertRaisesRegex(RuntimeError, "matching layout_out"):
                    invoke(scale, layout_out=wrong_out)
            with self.assertRaisesRegex(RuntimeError, "require MXFP8"):
                invoke(scale, quant_mode=1)

    def test_npu_metadata_and_excess_sparse_rows(self):
        if not torch_npu.npu.is_available():
            self.fail(
                "This regression requires an Ascend device and rebuilt custom_ops"
            )
        torch.manual_seed(11704)
        sq, kv_len, heads, kv_heads, dim = 129, 257, 4, 2, 128
        q_cpu = torch.randn(2, sq, heads, dim) * 0.1
        # Quantize once, then move bytes only. Constant descales cannot expose an
        # incorrect batch/head/token/group address in the QScale DMA path.
        q_bytes = q_cpu.to(torch.float8_e4m3fn).view(torch.uint8)
        b = torch.arange(2).reshape(2, 1, 1, 1)
        s = torch.arange(sq).reshape(1, sq, 1, 1)
        n = torch.arange(heads).reshape(1, 1, heads, 1)
        g = torch.arange(4).reshape(1, 1, 1, 4)
        qs_bytes = (
            (125 + (3 * b + s + 2 * n + g) % 5)
            .to(torch.uint8)
            .reshape(2, sq, heads, 2, 2)
        )
        for block in (64, 128):
            pages_per_batch = (kv_len + block - 1) // block
            pages = 2 * pages_per_batch
            cache = (
                (torch.randn(pages, 2, kv_heads, block, dim) * 0.1)
                .to(torch.float8_e4m3fn)
                .npu()
            )
            k, v = cache[:, 0], cache[:, 1]
            ks = (
                torch.full((pages, kv_heads, block, 2, 2), 127, dtype=torch.uint8)
                .view(torch.float8_e8m0fnu)
                .npu()
            )
            vs = (
                torch.full(
                    (pages, kv_heads, block // 64, dim, 2), 127, dtype=torch.uint8
                )
                .view(torch.float8_e8m0fnu)
                .npu()
            )
            table = (
                torch.arange(pages, dtype=torch.int32).reshape(2, pages_per_batch).npu()
            )
            used_kv = torch.tensor([kv_len, kv_len], dtype=torch.int32).npu()
            cu_q = torch.tensor([0, sq, 2 * sq], dtype=torch.int32).npu()
            qb = (sq + block - 1) // block
            # Shift by batch and head so BOTH block sizes contain empty and
            # nonempty final rows; flat arange % 3 never empties the SB64 tail.
            counts = (
                (
                    torch.arange(2).reshape(2, 1, 1)
                    + torch.arange(heads).reshape(1, heads, 1)
                    + torch.arange(qb).reshape(1, 1, qb)
                )
                % 3
            ).to(torch.int32)
            for mask_mode in (0, 3):
                mask = (
                    torch.triu(
                        torch.ones((2048, 2048), dtype=torch.uint8), diagonal=1
                    ).npu()
                    if mask_mode == 3
                    else None
                )
                for with_lse in (False, True):

                    def invoke(layout, extra_rows, extra_count=2):
                        layout_counts = torch.cat(
                            (
                                counts,
                                torch.full(
                                    (2, heads, extra_rows),
                                    extra_count,
                                    dtype=torch.int32,
                                ),
                            ),
                            dim=2,
                        )
                        sparse = layout_counts.npu()
                        indices = (
                            torch.arange(3, dtype=torch.int32)
                            .expand(2, heads, qb + extra_rows, 3)
                            .contiguous()
                            .npu()
                        )
                        q_raw = (
                            q_bytes.reshape(2 * sq, heads, dim)
                            if layout == "TND"
                            else q_bytes
                        )
                        qs_raw = (
                            qs_bytes.reshape(2 * sq, heads, 2, 2)
                            if layout == "TND"
                            else qs_bytes
                        )
                        if layout == "BNSD":
                            q_raw = q_raw.transpose(1, 2).contiguous()
                            qs_raw = qs_raw.transpose(1, 2).contiguous()
                        q = q_raw.view(torch.float8_e4m3fn).npu()
                        qs = qs_raw.view(torch.float8_e8m0fnu).npu()
                        lengths = cu_q if layout == "TND" else None
                        meta = metadata(
                            sparse,
                            layout,
                            block,
                            cu_seqlens_q=lengths,
                            seqused_kv=used_kv,
                            mask_mode=mask_mode,
                        )
                        # Attribute asynchronous metadata failures to metadata,
                        # before any attention launch can obscure the origin.
                        torch_npu.npu.synchronize()
                        out, lse = torch.ops.custom.npu_quant_block_sparse_attn(
                            q,
                            k,
                            v,
                            qs,
                            ks,
                            vs,
                            None,
                            indices,
                            sparse,
                            mask,
                            0.125,
                            block,
                            block,
                            cu_seqlens_q=lengths,
                            seqused_kv=used_kv,
                            block_table=table,
                            metadata=meta,
                            layout_q=layout,
                            layout_out=layout,
                            quant_mode=2,
                            mask_mode=mask_mode,
                            return_softmax_lse=with_lse,
                        )
                        torch_npu.npu.synchronize()
                        return out.cpu(), lse.cpu(), meta.cpu()

                    baseline, baseline_lse, baseline_meta = invoke("TND", 0)
                    for layout in ("BSND", "BNSD"):
                        for extra, extra_count in ((0, 0), (5, 0), (5, 2)):
                            with self.subTest(
                                layout=layout,
                                block=block,
                                mask=mask_mode,
                                lse=with_lse,
                                extra=extra,
                                extra_count=extra_count,
                            ):
                                out, lse, meta = invoke(layout, extra, extra_count)
                                expected_shape = (
                                    (2, sq, heads, dim)
                                    if layout == "BSND"
                                    else (2, heads, sq, dim)
                                )
                                self.assertEqual(tuple(out.shape), expected_shape)
                                self.assertEqual(out.dtype, torch.bfloat16)
                                self.assertEqual(
                                    tuple(lse.shape),
                                    (2, heads, sq) if with_lse else (0,),
                                )
                                self.assertEqual(lse.dtype, torch.float32)
                                if layout == "BNSD":
                                    out = out.transpose(1, 2)
                                torch.testing.assert_close(
                                    out.reshape_as(baseline), baseline, rtol=0, atol=0
                                )
                                # Check empty-row semantics independently of TND,
                                # including the one-token last Q block (s=128).
                                for batch in range(2):
                                    for head in range(heads):
                                        for row in range(qb):
                                            if int(counts[batch, head, row]) == 0:
                                                start, end = (
                                                    row * block,
                                                    min((row + 1) * block, sq),
                                                )
                                                self.assertTrue(
                                                    bool(
                                                        (
                                                            out[batch, start:end, head]
                                                            == 0
                                                        ).all()
                                                    )
                                                )
                                                if with_lse:
                                                    self.assertTrue(
                                                        bool(
                                                            torch.isneginf(
                                                                lse[
                                                                    batch,
                                                                    head,
                                                                    start:end,
                                                                ]
                                                            ).all()
                                                        )
                                                    )
                                if with_lse:
                                    torch.testing.assert_close(
                                        lse.transpose(1, 2).reshape_as(baseline_lse),
                                        baseline_lse,
                                        rtol=0,
                                        atol=0,
                                    )
                                if extra == 0:
                                    # Only the active section region is initialized by AICPU.
                                    used = 8 + int(meta[0]) * 36 * 8 + 72 * 8
                                    torch.testing.assert_close(
                                        meta[:used],
                                        baseline_meta[:used],
                                        rtol=0,
                                        atol=0,
                                    )


if __name__ == "__main__":
    unittest.main()
