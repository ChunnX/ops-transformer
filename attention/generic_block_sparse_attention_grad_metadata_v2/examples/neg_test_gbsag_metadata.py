"""Negative tests: the checker must reject non-conforming parameters."""

import torch
import torch_npu  # noqa: F401

from cann_ops_transformer.ops import generic_block_sparse_attention_grad_metadata


def main():
    torch.npu.set_device(0)
    B, N2, K, S1 = 1, 8, 1, 256
    idx = torch.zeros((B, N2, K, S1), dtype=torch.int32, device="npu")
    count = torch.zeros((B, N2, K), dtype=torch.int32, device="npu")

    good = dict(
        max_q_seqlen=256,
        max_kv_seqlen=256,
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=128,
        block_shape=[1, 256],
        is_packed_gqa=1,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        mask_type=1,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
    )

    def expect_fail(name, idx_t=None, count_t=None, **kwargs):
        try:
            generic_block_sparse_attention_grad_metadata(
                idx_t if idx_t is not None else idx,
                count_t if count_t is not None else count,
                **kwargs,
            )
            print(f"[FAIL] {name}: NOT rejected")
        except Exception as e:  # noqa: BLE001
            first = str(e).strip().splitlines()[0]
            print(f"[PASS] {name}: rejected -> {first[:150]}")

    kw = dict(good)
    kw["head_dim"] = 64
    expect_fail("head_dim=64", **kw)

    kw = dict(good)
    kw["block_shape"] = [2, 256]
    expect_fail("blockShapeX=2", **kw)

    kw = dict(good)
    kw["block_shape"] = [1, 100]
    expect_fail("blockShapeY=100 (not 64-aligned / <128)", **kw)

    kw = dict(good)
    kw["num_q_heads"] = 7
    expect_fail("numQHeads=7 (not divisible by numKvHeads)", **kw)

    kw = dict(good)
    kw["win_left"] = 4
    expect_fail("winLeft=4", **kw)

    kw = dict(good)
    kw["mask_type"] = 0
    expect_fail("maskType=0", **kw)

    kw = dict(good)
    kw["q_input_layout"] = "TND"
    kw["kv_input_layout"] = "TND"
    expect_fail("TND without cu_seq_lengths_q", **kw)

    idx1 = torch.zeros((1,), dtype=torch.int32, device="npu")
    expect_fail("rank-1 sparse_block_idx", idx_t=idx1, **good)

    idx64 = idx.to(torch.int64)
    expect_fail("INT64 sparse_block_idx", idx_t=idx64, **good)

    bad_meta_shape_idx = torch.zeros((B, N2, K, 128), dtype=torch.int32, device="npu")
    expect_fail("idx last dim != maxQSeqlen", idx_t=bad_meta_shape_idx, **good)

    print("negative tests done")


if __name__ == "__main__":
    main()
