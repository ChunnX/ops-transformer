"""Minimal NPU smoke test for GenericBlockSparseAttentionGradMetadata."""

import torch
import torch_npu  # noqa: F401

from cann_ops_transformer.ops import generic_block_sparse_attention_grad_metadata


def main():
    if not torch.npu.is_available():
        raise RuntimeError("NPU is required for this example")
    torch.npu.set_device(0)
    batch = 1
    num_q_heads = 8
    num_kv_heads = 8
    head_dim = 128
    max_q_seqlen = 256
    max_kv_seqlen = 256
    block_shape = [1, 256]
    kv_block_num = (max_kv_seqlen + block_shape[1] - 1) // block_shape[1]
    # BNSD layout:
    #   sparse_block_idx:    [B, N2, ceilDiv(S2, blockShapeY), S1]
    #   sparse_block_count:  [B, N2, ceilDiv(S2, blockShapeY)]
    idx = torch.zeros(
        (batch, num_kv_heads, kv_block_num, max_q_seqlen),
        dtype=torch.int32,
        device="npu",
    )
    count = torch.zeros(
        (batch, num_kv_heads, kv_block_num), dtype=torch.int32, device="npu"
    )
    metadata = generic_block_sparse_attention_grad_metadata(
        idx,
        count,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_kv_seqlen,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_shape=block_shape,
        is_packed_gqa=1,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        mask_type=1,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
    )
    torch.npu.synchronize()
    # Fixed ABI capacity = Header[32] + KTask[64][8].
    blocks_per_sparse = (block_shape[1] + 127) // 128
    full = max_kv_seqlen // block_shape[1]
    tail = max_kv_seqlen % block_shape[1]
    kv_blocks = full * blocks_per_sparse + (tail + 127) // 128
    metadata_len = 32 + 64 * 8
    assert metadata.shape == (metadata_len,), (metadata.shape, metadata_len)
    assert metadata.dtype == torch.int32

    h = metadata.cpu()
    h = h.numpy()
    # Header ABI (int32 words, offsets per generic_block_sparse_attention_grad_metadata_layout.h)
    assert int(h[0]) == 0x47425341, "magic mismatch"
    total_len = int(h[1])
    assert total_len == 32 + int(h[14]) * int(h[15]), (total_len, h[14], h[15])
    assert int(h[2]) == batch
    assert int(h[3]) == num_q_heads
    assert int(h[4]) == num_kv_heads
    assert int(h[5]) == head_dim
    assert int(h[6]) == 1, "inputLayout must be BNSD(=1)"
    assert int(h[7]) == max_q_seqlen
    assert int(h[8]) == max_kv_seqlen
    total_task_num = int(h[9])
    assert total_task_num == batch * num_kv_heads * kv_blocks, total_task_num
    assert int(h[14]) == 64
    assert int(h[15]) == 8
    print(
        f"metadata: shape={tuple(metadata.shape)}, dtype={metadata.dtype}, "
        f"magic={hex(h[0])}, totalLen={total_len}, totalTaskNum={total_task_num}"
    )


if __name__ == "__main__":
    main()
