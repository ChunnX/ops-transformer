# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
"""Small deterministic raw-byte fixtures, independent of the golden's layout packer."""


def make_inputs(torch, layout, block=128, sq=257, pattern="mixed", extra=0, extra_count=0, seed=12177):
    batch, heads, kv_heads, dim, kv_len = 2, 4, 2, 128, 257
    rng = torch.Generator().manual_seed(seed)
    q = (torch.randn(batch, sq, heads, dim, generator=rng) * 0.1).to(torch.float8_e4m3fn).view(torch.uint8)
    b = torch.arange(batch).reshape(batch, 1, 1, 1)
    s = torch.arange(sq).reshape(1, sq, 1, 1)
    n = torch.arange(heads).reshape(1, 1, heads, 1)
    g = torch.arange(4).reshape(1, 1, 1, 4)
    scale = (125 + (3 * b + s + 2 * n + g) % 5).to(torch.uint8).reshape(batch, sq, heads, 2, 2)
    if layout == "TND":
        q, scale = q.reshape(batch * sq, heads, dim), scale.reshape(batch * sq, heads, 2, 2)
    elif layout == "BNSD":
        q, scale = q.transpose(1, 2).contiguous(), scale.transpose(1, 2).contiguous()
    pages_per_batch = (kv_len + block - 1) // block
    pages, qb = batch * pages_per_batch, (sq + block - 1) // block
    cache = (torch.randn(pages, 2, kv_heads, block, dim, generator=rng) * 0.1).to(torch.float8_e4m3fn).npu()
    counts = torch.full((batch, heads, qb), pages_per_batch, dtype=torch.int32)
    if pattern == "all":
        counts.zero_()
    elif pattern in ("first", "middle", "last"):
        row = {"first": 0, "middle": qb // 2, "last": qb - 1}[pattern]
        counts[0, 1, row] = 0
    elif pattern == "mixed":
        counts = ((torch.arange(batch).reshape(batch, 1, 1) + torch.arange(heads).reshape(1, heads, 1)
                   + torch.arange(qb).reshape(1, 1, qb)) % 3).to(torch.int32)
    elif pattern != "dense":
        raise ValueError(pattern)
    physical_counts = torch.cat((counts, torch.full((batch, heads, extra), extra_count, dtype=torch.int32)), dim=2)
    inputs = dict(
        query=q.view(torch.float8_e4m3fn).npu(), q_descale=scale.view(torch.float8_e8m0fnu).npu(),
        key=cache[:, 0], value=cache[:, 1],
        k_descale=torch.full((pages, kv_heads, block, 2, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu(),
        v_descale=torch.full((pages, kv_heads, block // 64, dim, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu(),
        p_scale=None,
        sparse_indices=torch.arange(pages_per_batch, dtype=torch.int32).expand(batch, heads, qb + extra, pages_per_batch).contiguous().npu(),
        sparse_seq_len=physical_counts.npu(), atten_mask=None,
        cu_seqlens_q=torch.tensor([0, sq, batch * sq], dtype=torch.int32).npu() if layout == "TND" else None,
        seqused_q=None, cu_seqlens_kv=None,
        seqused_kv=torch.tensor([kv_len] * batch, dtype=torch.int32).npu(),
        block_table=torch.arange(pages, dtype=torch.int32).reshape(batch, pages_per_batch).npu(),
        softmax_scale=0.125, sparse_q_block_size=block, sparse_kv_block_size=block,
        layout_q=layout, layout_out=layout, layout_kv="PA_BNBD", quant_mode=2,
        mask_mode=0, return_softmax_lse=True,
    )
    rebuild_metadata(torch, inputs)
    return inputs, counts


def rebuild_metadata(torch, inputs):
    inputs["metadata"] = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
        inputs["sparse_seq_len"], 4, 2, 128,
        cu_seqlens_q=inputs["cu_seqlens_q"], seqused_q=inputs["seqused_q"],
        seqused_kv=inputs["seqused_kv"], batch_size=2,
        sparse_block_size_q=inputs["sparse_q_block_size"], sparse_block_size_k=inputs["sparse_kv_block_size"],
        quant_mode=2, mask_mode=inputs["mask_mode"], layout_q=inputs["layout_q"], layout_kv="PA_BNBD")
    torch.npu.synchronize()  # Attribute AICPU failures before the main kernel launch.


def invoke(torch, inputs):
    out, lse = torch.ops.custom.npu_quant_block_sparse_attn(**inputs)
    torch.npu.synchronize()
    return out.cpu(), lse.cpu()


def assert_outputs(torch, inputs, outputs, sq=257):
    out, lse = outputs
    q = inputs["query"]
    assert tuple(out.shape) == tuple(q.shape)
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    if inputs["layout_q"] != "TND":
        assert tuple(lse.shape) == ((2, 4, sq) if inputs["return_softmax_lse"] else (0,))
    if inputs["layout_q"] == "BNSD":
        out = out.transpose(1, 2)
    elif inputs["layout_q"] == "TND":
        out = out.reshape(2, sq, 4, 128)
        if inputs["return_softmax_lse"]:
            lse = lse.reshape(2, sq, 4).transpose(1, 2)
    return out, lse
