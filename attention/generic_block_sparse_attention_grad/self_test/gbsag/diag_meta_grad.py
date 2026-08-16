"""Diagnostic: dump full metadata + dq zero-pattern for the causal BNSD grad smoke case."""

import os
import sys
from pathlib import Path

os.environ.setdefault("BSAG_DEVICE", "npu:0")
os.environ.setdefault("BSAG_DTYPE", "fp16")

REPO = Path(__file__).resolve().parents[4]
GBSAG_DIR = Path(__file__).resolve().parent
TORCH_EXT = REPO / "torch_extension"
for p in (str(TORCH_EXT), str(GBSAG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch, torch_npu  # noqa
import test_gbsag as H
from block_mask_to_idx import blcok_sparse_mask_to_idx
from cann_ops_transformer.ops import (
    generic_block_sparse_attention_grad,
    generic_block_sparse_attention_grad_metadata,
)


def main():
    torch.npu.set_device(0)
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

    def T(x):
        return x.to(DEVICE)

    query_npu, key_npu, value_npu, d_out_npu = T(query), T(key), T(value), T(d_out)
    rsvd_block_idx_npu, rsvd_block_count_npu = (
        T(rsvd_block_idx_cpu),
        T(rsvd_block_count_cpu),
    )
    attention_out_npu, softmax_lse_npu = T(attention_out_cpu), T(softmax_lse_cpu)

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
    meta = metadata.cpu().numpy()
    print("META header:", meta[:32].tolist(), flush=True)
    # per-batch segment (2 words) then task table
    nb = int(meta[2])
    total_task = int(meta[9])
    per_core = int(meta[10])
    tail = int(meta[11])
    task_base = 32 + 2 * nb
    print(
        f"META summary batch={nb} totalTask={total_task} perCore={per_core} tail={tail}",
        flush=True,
    )
    print("META per-batch:", meta[32 : 32 + 2 * nb].tolist(), flush=True)
    print("META task table:")
    for t in range(total_task):
        b = task_base + 5 * t
        print(
            f"  task{t}: batch={meta[b + 0]} kvHead={meta[b + 1]} kvSeq={meta[b + 2]} preQ={meta[b + 3]} preKv={meta[b + 4]}",
            flush=True,
        )

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

    # zero-pattern analysis: which (kvHead, kBlock) tasks contributed to dq
    dq = d_query.cpu().numpy().reshape(1, 8, 256, 128)  # B,N,S,D
    gdq = cpu_dq.numpy().reshape(1, 8, 256, 128)
    gdv = cpu_dv.numpy().reshape(1, 4, 256, 128)
    dv = d_value.cpu().numpy().reshape(1, 4, 256, 128)
    print("-- per-kvHead dk/dv abs-sum (device vs golden) --")
    for hh in range(4):
        dev = float(dv[0, hh].sum())
        ref = float(gdv[0, hh].sum())
        print(
            f"  kvHead{hh}: devSum={dev:.4f} goldenSum={ref:.4f} dev_nz={int((dv[0, hh] != 0).sum())}",
            flush=True,
        )
    print("-- per-kvBlockHalf (rows 0:128 vs 128:256) dv nonzero --")
    for half in (0, 1):
        sl = slice(half * 128, half * 128 + 128)
        print(
            f"  kvRows[{half * 128}:{(half + 1) * 128}] dv_nz={int((dv[0, :, sl, :] != 0).sum())} golden_nz={int((gdv[0, :, sl, :] != 0).sum())}",
            flush=True,
        )
    print("-- dq per-qHead nonzero (device vs golden) --")
    for hh in range(8):
        print(
            f"  qHead{hh}: dev_nz={int((dq[0, hh] != 0).sum())} golden_nz={int((gdq[0, hh] != 0).sum())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
