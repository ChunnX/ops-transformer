#!/usr/bin/env python3
"""Minimal PTA precision test for D400 MXFP4 KvQuantSparseFlashAttention."""

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


CASE = bi.PrecisionCase(
    name="pta_bsnd_bf16_s2_3",
    q_dtype="bf16",
    layout_query="BSND",
    layout_kv="BSND",
    s2=3,
    sparse_ids=(0, 1, 2),
    storage_offset_bytes=0,
)


def build_golden(cpu_inputs, records):
    decoded = torch.tensor([bi.decode_record(record, "bf16") for record in records], dtype=torch.float32)
    decoded = decoded.to(torch.bfloat16).reshape(1, 1, CASE.s2, bi.NOPE_DIM + bi.ROPE_DIM)
    fa_param = {
        "b": 1,
        "numHeads": 1,
        "numKeyValueHeads": 1,
        "actualSeqLengths_q": [1],
        "actualSeqLengths_kv": [CASE.s2],
        "scaleValue": 1.0 / math.sqrt(float(bi.NOPE_DIM + bi.ROPE_DIM)),
        "sparse_blocksize": 1,
        "sparse_blockcount": len(CASE.sparse_ids),
        "sparse_indices_bnsd_tensor": cpu_inputs["sparse_indices"].permute(0, 2, 1, 3).contiguous(),
        "out_shape_bnsd": [1, 1, 1, bi.NOPE_DIM],
        "q_bnsd_tensor": cpu_inputs["query"].permute(0, 2, 1, 3).contiguous(),
        "k_bnsd_tensor": decoded,
        "k_dequant_scale_bnsd_tensor": torch.ones((1, 1, CASE.s2, 4), dtype=torch.float32),
        "v_bnsd_tensor": decoded[..., : bi.NOPE_DIM].clone(),
        "v_dequant_scale_bnsd_tensor": torch.ones((1, 1, CASE.s2, 4), dtype=torch.float32),
        "q_dtype": "bfloat16",
        "sparse_mode": 3,
        "sinks_tensor": None,
    }
    return qsfa_golden._t_increattention_bnsd(fa_param).permute(0, 2, 1, 3).contiguous()


def run_once(iteration: int) -> None:
    records = bi.make_records(CASE)
    cpu_inputs = bi.build_cpu_inputs(CASE, records)
    expected = build_golden(cpu_inputs, records)
    npu_inputs = bi.move_inputs_to_npu(cpu_inputs, CASE.storage_offset_bytes)

    actual = torch_npu.npu_kv_quant_sparse_flash_attention(
        npu_inputs["query"],
        npu_inputs["key"],
        npu_inputs["value"],
        npu_inputs["sparse_indices"],
        1.0 / math.sqrt(float(bi.NOPE_DIM + bi.ROPE_DIM)),
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
    torch.npu.synchronize()

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
    print(f"[PASS] iteration={iteration} close_ratio={ratio:.6f} max_abs_error={max_abs_error:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    for iteration in range(args.repeat):
        run_once(iteration)
    print(f"summary: executed={args.repeat} passed={args.repeat} failed=0")


if __name__ == "__main__":
    main()
