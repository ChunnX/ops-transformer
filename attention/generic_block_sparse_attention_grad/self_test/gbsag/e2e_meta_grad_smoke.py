"""End-to-end smoke: metadata producer -> main GenericBlockSparseAttentionGrad kernel.

Reuses the repo self_test golden machinery for a single headDim=128 causal BNSD case.
This validates the full tiling-sunk path: AICPU writes metadata (header + per-batch +
K task table), the main kernel reads maxQ/maxK/task partition/task table from it, and
produces grads identical to the CPU golden.

Only the metadata-op-supported subset is exercised here (headDim=128, blockX=1,
blockY 64-aligned >=128, causal mask_type=1, packed GQA). The broader suite is covered
by test_gbsag.py.
"""

import os
import sys
from pathlib import Path

# ---- environment (must be set before importing the harness) ----
os.environ.setdefault("BSAG_DEVICE", "npu:0")
os.environ.setdefault("BSAG_DTYPE", "fp16")

REPO = Path(__file__).resolve().parents[4]  # ops-transformer
GBSAG_DIR = Path(__file__).resolve().parent  # .../self_test/gbsag
TORCH_EXT = REPO / "torch_extension"
for p in (str(TORCH_EXT), str(GBSAG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch_npu  # noqa: F401

import test_gbsag as H
from block_mask_to_idx import blcok_sparse_mask_to_idx
from cann_ops_transformer.ops import (
    generic_block_sparse_attention_grad,
    generic_block_sparse_attention_grad_metadata,
)


def main():
    torch.npu.set_device(0)

    # B, N1, S1_max, S2_max, D, num_kv_heads, sparsity, mha_type, format_type, block_shape,
    # mask_type, use_q_seq, use_kv_seq
    case = [1, 8, 256, 256, 128, 4, 1.0, "gqa", "bnsd", (1, 128), 1, False, False]
    case_id = 9000
    block_shape = H.normalize_block_shape(case[9])
    is_causal = case[10] == 1
    H.seed_everything(H.SEED + case_id)

    (
        query,
        key,
        value,
        d_out,
        block_sparse_mask,
        attention_out_cpu,
        softmax_lse_cpu,
        block_shape_norm,
        num_kv_heads,
        scale_value,
        S1_max,
        S2_max,
        accumulate_seq_length,
        accumulate_seq_length_kv,
        seqlen_q_list,
        seqlen_kv_list,
    ) = H.gen_data(list(case[:9]), block_shape, is_causal)

    cpu_dq, cpu_dk, cpu_dv = H.gen_golden(
        query,
        key,
        value,
        d_out,
        block_sparse_mask,
        attention_out_cpu,
        softmax_lse_cpu,
        block_shape_norm,
        scale_value,
        S1_max,
        S2_max,
        accumulate_seq_length,
        accumulate_seq_length_kv,
        "bnsd",
        is_causal,
    )

    rsvd_block_idx_cpu, rsvd_block_count_cpu = blcok_sparse_mask_to_idx(
        block_sparse_mask,
        list(block_shape_norm),
        seqlen_q_list,
        seqlen_kv_list,
        S1_max,
        S2_max,
        "kv2qIdx",
        q_head_num=case[1],
        kv_head_num=num_kv_heads,
        is_packed_gqa=True,
    )

    DEVICE = "npu:0"
    query_npu = query.to(DEVICE)
    key_npu = key.to(DEVICE)
    value_npu = value.to(DEVICE)
    d_out_npu = d_out.to(DEVICE)
    rsvd_block_idx_npu = rsvd_block_idx_cpu.to(DEVICE)
    rsvd_block_count_npu = rsvd_block_count_cpu.to(DEVICE)
    attention_out_npu = attention_out_cpu.to(DEVICE)
    softmax_lse_npu = softmax_lse_cpu.to(DEVICE)

    # ---- produce metadata (AICPU tiling-sunk producer) ----
    metadata = generic_block_sparse_attention_grad_metadata(
        rsvd_block_idx_npu,
        rsvd_block_count_npu,
        max_q_seqlen=S1_max,
        max_kv_seqlen=S2_max,
        num_q_heads=case[1],
        num_kv_heads=num_kv_heads,
        head_dim=case[4],
        block_shape=list(block_shape_norm),
        is_packed_gqa=1,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        mask_type=1,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
    )
    torch.npu.synchronize()
    hdr = metadata.cpu().numpy()
    assert int(hdr[0]) == 0x47425341, hex(hdr[0])
    print(
        f"META magic={hex(hdr[0])} totalLen={hdr[1]} batch={hdr[2]} "
        f"maxQ={hdr[7]} maxKv={hdr[8]} totalTask={hdr[9]} "
        f"perCore={hdr[10]} tail={hdr[11]}",
        flush=True,
    )

    # ---- main op consumes metadata ----
    d_query, d_key, d_value = generic_block_sparse_attention_grad(
        d_out_npu,
        query_npu,
        key_npu,
        value_npu,
        attention_out_npu,
        softmax_lse_npu,
        rsvd_block_idx_npu,
        rsvd_block_count_npu,
        metadata=metadata,
        atten_mask=None,
        block_shape=list(block_shape_norm),
        cu_seq_lengths=None,
        cu_seq_lengths_kv=None,
        actual_seq_lengths=seqlen_q_list,
        actual_seq_lengths_kv=seqlen_kv_list,
        is_packed_gqa=1,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        scale_value=scale_value,
        mask_type=1,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
    )
    torch.npu.synchronize()

    ok = H.compare_golden(cpu_dq, cpu_dk, cpu_dv, d_query, d_key, d_value)
    print("E2E_RESULT", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
