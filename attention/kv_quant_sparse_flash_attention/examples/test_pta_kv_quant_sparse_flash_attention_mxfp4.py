#!/usr/bin/env python3
"""PTA precision test for a representative D400 MXFP4 QSFA workload."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
import torch_npu

import mxfp4_bi_precision_cases as bi


EXAMPLE_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = EXAMPLE_DIR.parent / "tests" / "pytest"
sys.path.insert(0, str(GOLDEN_DIR))
import kv_quant_sparse_flash_attention_golden as qsfa_golden


CASE_NAME = "pta_bsnd_bf16_b4_s1_8_g128_s2_135_offset32"
BATCH = 4
S1 = 8
G = 128
S2 = 135
QUERY_DIM = bi.NOPE_DIM + bi.ROPE_DIM
STORAGE_OFFSET_BYTES = 32


def make_sparse_indices():
    rows = []
    for batch in range(BATCH):
        for s1_idx in range(S1):
            valid_count = S2 - S1 + s1_idx + 1
            shift = (batch * 7 + s1_idx * 5) % valid_count
            valid = torch.roll(torch.arange(valid_count, dtype=torch.int32), shifts=shift)
            padding = torch.zeros(S2 - valid_count, dtype=torch.int32)
            rows.append(torch.cat((valid, padding)))
    return torch.stack(rows).reshape(BATCH, S1, 1, S2)


def build_inputs_and_records():
    record_case = bi.PrecisionCase(
        name=CASE_NAME,
        q_dtype="bf16",
        layout_query="BSND",
        layout_kv="BSND",
        s2=S2,
        sparse_ids=tuple(range(S2)),
        storage_offset_bytes=STORAGE_OFFSET_BYTES,
    )
    records_by_batch = [bi.make_records(record_case, token_offset=batch * (S2 + 17)) for batch in range(BATCH)]
    flat_records = [record for records in records_by_batch for record in records]

    batch_axis = torch.arange(BATCH, dtype=torch.float32).reshape(BATCH, 1, 1, 1)
    s1_axis = torch.arange(S1, dtype=torch.float32).reshape(1, S1, 1, 1)
    head_axis = torch.arange(G, dtype=torch.float32).reshape(1, 1, G, 1)
    dim_axis = torch.arange(QUERY_DIM, dtype=torch.float32).reshape(1, 1, 1, QUERY_DIM)
    query = 0.00390625 * torch.remainder(dim_axis * 3 + head_axis * 5 + batch_axis * 7 + s1_axis * 11, 29)
    query = (query - 0.00390625 * 14).to(torch.bfloat16)

    inputs = {
        "query": query,
        "key": bi.records_to_int8(flat_records).reshape(BATCH, S2, 1, bi.RECORD_BYTES),
        "value": torch.zeros((BATCH, S2, 1, bi.NOPE_DIM), dtype=torch.int8),
        "sparse_indices": make_sparse_indices(),
        "key_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "value_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "actual_seq_lengths_query": torch.full((BATCH,), S1, dtype=torch.int32),
        "actual_seq_lengths_kv": torch.full((BATCH,), S2, dtype=torch.int32),
        "block_table": None,
    }
    return inputs, records_by_batch


def build_golden(cpu_inputs, records_by_batch):
    decoded = torch.tensor(
        [[bi.decode_record(record, "bf16") for record in records] for records in records_by_batch],
        dtype=torch.float32,
    )
    decoded = decoded.to(torch.bfloat16).reshape(BATCH, 1, S2, QUERY_DIM)
    fa_param = {
        "b": BATCH,
        "numHeads": G,
        "numKeyValueHeads": 1,
        "actualSeqLengths_q": [S1] * BATCH,
        "actualSeqLengths_kv": [S2] * BATCH,
        "scaleValue": 1.0 / math.sqrt(float(QUERY_DIM)),
        "sparse_blocksize": 1,
        "sparse_blockcount": S2,
        "sparse_indices_bnsd_tensor": cpu_inputs["sparse_indices"].permute(0, 2, 1, 3).contiguous(),
        "out_shape_bnsd": [BATCH, G, S1, bi.NOPE_DIM],
        "q_bnsd_tensor": cpu_inputs["query"].permute(0, 2, 1, 3).contiguous(),
        "k_bnsd_tensor": decoded,
        "k_dequant_scale_bnsd_tensor": torch.ones((BATCH, 1, S2, 4), dtype=torch.float32),
        "v_bnsd_tensor": decoded[..., : bi.NOPE_DIM].clone(),
        "v_dequant_scale_bnsd_tensor": torch.ones((BATCH, 1, S2, 4), dtype=torch.float32),
        "q_dtype": "bfloat16",
        "sparse_mode": 3,
        "sinks_tensor": None,
    }
    return qsfa_golden._t_increattention_bnsd(fa_param).permute(0, 2, 1, 3).contiguous()


def run_once(iteration: int) -> None:
    print(f"[STEP] iteration={iteration} build_inputs", flush=True)
    cpu_inputs, records_by_batch = build_inputs_and_records()
    if cpu_inputs["query"].shape[-1] != QUERY_DIM:
        raise AssertionError(f"query D must be {QUERY_DIM}, got {cpu_inputs['query'].shape[-1]}")
    if cpu_inputs["key"].shape[-1] != bi.RECORD_BYTES:
        raise AssertionError(f"packed key D must be {bi.RECORD_BYTES}, got {cpu_inputs['key'].shape[-1]}")
    print(
        f"[INPUT] query_shape={tuple(cpu_inputs['query'].shape)} "
        f"key_shape={tuple(cpu_inputs['key'].shape)} output_shape=({BATCH},{S1},{G},{bi.NOPE_DIM})",
        flush=True,
    )
    print(f"[STEP] iteration={iteration} build_golden", flush=True)
    expected = build_golden(cpu_inputs, records_by_batch)
    print(f"[STEP] iteration={iteration} golden_ready shape={tuple(expected.shape)}", flush=True)
    print(f"[STEP] iteration={iteration} move_inputs_to_npu", flush=True)
    npu_inputs = bi.move_inputs_to_npu(cpu_inputs, STORAGE_OFFSET_BYTES)

    print(f"[STEP] iteration={iteration} call_operator", flush=True)
    actual = torch_npu.npu_kv_quant_sparse_flash_attention(
        npu_inputs["query"],
        npu_inputs["key"],
        npu_inputs["value"],
        npu_inputs["sparse_indices"],
        1.0 / math.sqrt(float(QUERY_DIM)),
        2,
        2,
        key_dequant_scale=npu_inputs["key_dequant_scale"],
        value_dequant_scale=npu_inputs["value_dequant_scale"],
        actual_seq_lengths_query=npu_inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=npu_inputs["actual_seq_lengths_kv"],
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    print(f"[STEP] iteration={iteration} synchronize", flush=True)
    torch.npu.synchronize()
    print(f"[STEP] iteration={iteration} copy_output_to_cpu", flush=True)

    actual_f32 = actual.cpu().float()
    expected_f32 = expected.float()
    close = torch.isclose(actual_f32, expected_f32, rtol=2e-2, atol=5e-2)
    ratio = float(close.float().mean())
    max_abs_error = float((actual_f32 - expected_f32).abs().max())
    if ratio < 0.99:
        raise AssertionError(
            f"iteration={iteration}, close_ratio={ratio:.6f}, max_abs_error={max_abs_error:.6f}, "
            f"actual[:8]={actual_f32.flatten()[:8].tolist()}, expected[:8]={expected_f32.flatten()[:8].tolist()}"
        )
    print(
        f"[PASS] {CASE_NAME} iteration={iteration} close_ratio={ratio:.6f} "
        f"max_abs_error={max_abs_error:.6f} output_shape={tuple(actual.shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    print(
        f"[ENV] python={sys.version.split()[0]} torch={torch.__version__} "
        f"torch_npu={getattr(torch_npu, '__version__', 'unknown')} device={args.device}",
        flush=True,
    )
    for iteration in range(args.repeat):
        run_once(iteration)
    print(f"summary: executed={args.repeat} passed={args.repeat} failed=0")


if __name__ == "__main__":
    main()
