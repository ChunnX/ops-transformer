#!/usr/bin/env python3
"""Run one BSA performance case in HIF8 or high-precision mode.

The two modes deliberately share the same source tensors, block mask, layout,
actual sequence lengths, and seed.  Input conversion happens before the
operator call so that the measured section is the BSA operator itself.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Sequence

import torch
import torch_npu

from cann_ops_transformer.ops import block_sparse_attention

# Keep the HIF8 codec and mask/layout rules identical to the golden suite.
SUITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SUITE_DIR not in sys.path:
    sys.path.insert(0, SUITE_DIR)
from generate_hifloat8_data import trans_float_tensor_to_hifuint8  # noqa: E402
from test_bsa_hif8_golden import (  # noqa: E402
    HIF8_MAX,
    _bnsd_to_layout,
    build_block_sparse_mask,
)


def _arg_int(value: str, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _actual_lengths(batch: int, q_seq: int, kv_seq: int, varied: int, seed: int):
    if not varied:
        return [q_seq] * batch, [kv_seq] * batch
    # Deterministic, valid prefixes.  Keep at least one complete block so the
    # sparse mask always contains useful work.
    q_gen = torch.Generator(device="cpu").manual_seed(seed + 101)
    kv_gen = torch.Generator(device="cpu").manual_seed(seed + 102)
    q_min = min(q_seq, max(1, ((q_seq + 127) // 128 - 1) * 128 + 1))
    kv_min = min(kv_seq, max(1, ((kv_seq + 127) // 128 - 1) * 128 + 1))
    q_values = torch.randint(q_min, q_seq + 1, (batch,), generator=q_gen).tolist()
    kv_values = torch.randint(kv_min, kv_seq + 1, (batch,), generator=kv_gen).tolist()
    q_values[0] = q_seq
    kv_values[0] = kv_seq
    return [int(x) for x in q_values], [int(x) for x in kv_values]


def _source_inputs(batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, seed):
    ranges = (0.91, 0.87, 0.77)
    outputs = []
    for offset, shape, amplitude in (
        (0, (batch, q_heads, q_seq, head_dim), ranges[0]),
        (1, (batch, kv_heads, kv_seq, head_dim), ranges[1]),
        (2, (batch, kv_heads, kv_seq, head_dim), ranges[2]),
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed + offset)
        values = torch.rand(shape, dtype=torch.float16, generator=generator)
        outputs.append((values * 2 - 1).float() * amplitude)
    return outputs


def _quantize_hif8(src: torch.Tensor):
    src = src.float().contiguous()
    max_abs = float(src.abs().max())
    scale_value = max_abs / HIF8_MAX if max_abs else 1.0
    scale = torch.tensor([scale_value], dtype=torch.float32)
    encoded = trans_float_tensor_to_hifuint8(
        (src / scale_value).contiguous(), round_mode="round", over_mode=True
    )
    return encoded.contiguous(), scale


def _prepare_actual_lengths(batch, q_seq, kv_seq, varied_len, actual_available, seed):
    # When actual sequence metadata is disabled, exercise the full storage
    # shape even if the caller asks for a varied-length generator.
    return _actual_lengths(batch, q_seq, kv_seq, int(varied_len and actual_available), seed)


def _build_perf_mask(
    batch, q_heads, q_seq, kv_seq, block_shape, actual_q, actual_kv,
    keep_ratio, varied_sparsity, seed,
):
    """Build the fixed or per-query-block sparse pattern used by both modes."""
    if not varied_sparsity:
        return build_block_sparse_mask(
            batch, q_heads, q_seq, kv_seq, block_shape, "dense",
            actual_q, actual_kv, keep_ratio, seed,
        )
    block_q, block_kv = map(int, block_shape)
    q_blocks = (q_seq + block_q - 1) // block_q
    kv_blocks = (kv_seq + block_kv - 1) // block_kv
    mask = torch.zeros(batch, q_heads, q_blocks, kv_blocks, dtype=torch.int8)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for b_idx in range(batch):
        valid_q = (actual_q[b_idx] + block_q - 1) // block_q
        valid_kv = (actual_kv[b_idx] + block_kv - 1) // block_kv
        max_keep = max(1, math.ceil(valid_kv * keep_ratio))
        for head_idx in range(q_heads):
            for q_block_idx in range(valid_q):
                keep = int(torch.randint(1, max_keep + 1, (1,), generator=generator).item())
                selected = torch.randperm(valid_kv, generator=generator)[:keep]
                mask[b_idx, head_idx, q_block_idx, selected] = 1
    return mask


def run_case(argv: Sequence[str]) -> None:
    if len(argv) != 20:
        raise SystemExit(
            "usage: invoke_npu_bsa_hif8_perf.py "
            "B qS kvS qHeads kvHeads D blockQ blockKV keepRatio innerPrec "
            "qLayout kvLayout variedLen variedSparsity actualSeqAvail mode "
            "baseDtype seed device pScale"
        )

    (
        batch_s, q_seq_s, kv_seq_s, q_heads_s, kv_heads_s, dim_s,
        block_q_s, block_kv_s, keep_ratio_s, inner_prec_s, q_layout,
        kv_layout, varied_len_s, varied_sparsity_s, actual_available_s,
        mode, base_dtype, seed_s, device_s, p_scale_s,
    ) = argv
    batch = _arg_int(batch_s, "batch")
    q_seq = _arg_int(q_seq_s, "qS")
    kv_seq = _arg_int(kv_seq_s, "kvS")
    q_heads = _arg_int(q_heads_s, "qHeads")
    kv_heads = _arg_int(kv_heads_s, "kvHeads")
    head_dim = _arg_int(dim_s, "D")
    block_q = _arg_int(block_q_s, "blockQ")
    block_kv = _arg_int(block_kv_s, "blockKV")
    keep_ratio = float(keep_ratio_s)
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keepRatio must be in (0, 1], got {keep_ratio}")
    inner_prec = int(inner_prec_s)
    seed = int(seed_s)
    device = int(device_s)
    p_scale = float(p_scale_s)
    mode = mode.lower()
    q_layout = q_layout.upper()
    kv_layout = kv_layout.upper()
    if mode not in {"hif8", "highprec"}:
        raise ValueError(f"mode must be hif8 or highprec, got {mode}")
    if base_dtype.lower() in {"half", "fp16", "float16"}:
        output_dtype = torch.float16
    elif base_dtype.lower() in {"bf16", "bfloat16"}:
        output_dtype = torch.bfloat16
    else:
        raise ValueError(f"unsupported baseDtype: {base_dtype}")

    actual_q, actual_kv = _prepare_actual_lengths(
        batch, q_seq, kv_seq, int(varied_len_s), int(actual_available_s), seed
    )
    q_src, k_src, v_src = _source_inputs(
        batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, seed
    )
    mask_q_seq = max(actual_q) if q_layout == "TND" else q_seq
    mask_kv_seq = max(actual_kv) if kv_layout == "TND" else kv_seq
    block_mask = _build_perf_mask(
        batch, q_heads, mask_q_seq, mask_kv_seq, (block_q, block_kv),
        actual_q, actual_kv, keep_ratio, int(varied_sparsity_s), seed,
    )
    attention_scale = head_dim ** -0.5

    torch_npu.npu.set_device(device)
    if mode == "hif8":
        q, q_scale = _quantize_hif8(q_src)
        k, k_scale = _quantize_hif8(k_src)
        v, v_scale = _quantize_hif8(v_src)
        q = _bnsd_to_layout(q, q_layout, actual_q).contiguous().npu()
        k = _bnsd_to_layout(k, kv_layout, actual_kv).contiguous().npu()
        v = _bnsd_to_layout(v, kv_layout, actual_kv).contiguous().npu()
        kwargs = dict(
            quant_mode=5,
            q_dequant_scale=q_scale.npu(),
            k_dequant_scale=k_scale.npu(),
            v_dequant_scale=v_scale.npu(),
            p_quant_scale=torch.tensor([p_scale], dtype=torch.float32).npu(),
            attention_out_dtype=output_dtype,
        )
    else:
        q = _bnsd_to_layout(q_src.to(output_dtype), q_layout, actual_q).contiguous().npu()
        k = _bnsd_to_layout(k_src.to(output_dtype), kv_layout, actual_kv).contiguous().npu()
        v = _bnsd_to_layout(v_src.to(output_dtype), kv_layout, actual_kv).contiguous().npu()
        kwargs = {}

    block_mask = block_mask.npu()
    # Keep this call free of logging, CPU copies, golden calculation, or
    # synchronization beyond the profiler's own warm-up/measurement logic.
    if mode == "hif8":
        block_sparse_attention(
            q, k, v, block_mask, [block_q, block_kv],
            q_input_layout=q_layout, kv_input_layout=kv_layout,
            num_key_value_heads=kv_heads, scale_value=attention_scale,
            inner_precise=inner_prec,
            actual_seq_lengths=actual_q, actual_seq_lengths_kv=actual_kv,
            return_softmax_lse=False, mask_type=0, block_size=0,
            pre_tokens=2147483647, next_tokens=2147483647, dst_type_max=0.0,
            **kwargs,
        )
    else:
        # quant_mode defaults to 0, mask_type defaults to 0, and inner_precise
        # is 0 for this runner.  On 950 that combination selects the
        # SMF32_REF32_NOMASK_TRANSMM1 FP16/BF16 tiling keys.
        torch_npu.npu_block_sparse_attention(
            q, k, v, block_mask, [block_q, block_kv],
            q_input_layout=q_layout, kv_input_layout=kv_layout,
            num_key_value_heads=kv_heads, scale_value=attention_scale,
            inner_precise=inner_prec,
            actual_seq_lengths=actual_q, actual_seq_lengths_kv=actual_kv,
            softmax_lse_flag=0,
        )
    torch_npu.npu.synchronize()


if __name__ == "__main__":
    run_case(sys.argv[1:])
