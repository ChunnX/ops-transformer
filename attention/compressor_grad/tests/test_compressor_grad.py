#!/usr/bin/env python3
"""Unified compressor_grad test — merges cube and vec stage input generation
into a single file, then runs each case through the golden reference
``compressor_grad_golden.compressor_grad``.

Cases live in ``compressor_grad_cases.py`` (50 cases total: 36 cube + 14 vec).
Golden reference lives in ``compressor_grad_golden.py`` (both under tests/).
"""
import logging
import os
import sys

import torch
import torch_npu  # noqa: F401  (registers npu backend)

# Make the parent directory importable so block_cube.py resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Make this directory (tests/) importable so the golden + cases modules resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compressor_grad_golden import compressor_grad_golden
from compressor_grad_cases import CASES
from block_cube import get_device, logger


logger.setLevel(logging.INFO)

# ════════════════════════════════════════════════════════════════════════════
# Unified input generation
# ════════════════════════════════════════════════════════════════════════════
def _gen_inputs(case):
    """Generate ALL inputs required by ``compressor_grad_golden``.

    Merges cube-stage generation (x, wkv, wgate) with vec-stage generation
    (d_cmp_kv, kv, softmax_score, cu_seqlens, seqused, start_pos) so a single
    call produces the full argument set the golden reference consumes.

    Shapes (T = total tokens, C = coff * d_size, H = h_size):
        BSH (layout=0):  x (B, S, H),  d_cmp_kv (B, ceil(S/cr), D),
                         kv / sm (B, ceil(S/cr), coff*cr, D)
        TH  (layout=1):  x (T, H),     d_cmp_kv (min(T, T//cr+B), D),
                         kv / sm (min(T, T//cr+B), coff*cr, D)
        wkv / wgate:     (C, H)        [both layouts]
        cu_seqlens:       (B+1,) int32  [TH only; None for BSH]
        seqused:          (B,) int32    [if sqs is not None]
        start_pos:        (B,) int32    [if sps is not None]
    """
    seed         = case["seed"]
    grad_scale   = case["grad_scale"]
    weight_scale = case["weight_scale"]
    coff         = case["coff"]
    layout       = case["layout"]          # 0 = BSH, 1 = TH
    dataType     = case["dataType"]        # 0 = BF16, 1 = FP16
    b_size       = case["b_size"]
    s_size       = case["s_size"]
    d_size       = case["d_size"]
    h_size       = case["h_size"]
    cmp_ratio    = case["cmp_ratio"]
    sps          = case["sps"]
    sqs          = case["sqs"]

    torch_dtype = torch.bfloat16 if dataType == 0 else torch.float16
    device      = get_device()
    c_size      = coff * d_size
    sc_kv_rows  = coff * cmp_ratio

    # Resolve sps / sqs to concrete python lists
    sps_list = [0] * b_size if sps is None else list(sps)
    sqs_list = [s_size] * b_size if sqs is None else list(sqs)

    # Total tokens: BSH uses B*S; TH uses sum(sqs) (== tokenSize in vec cases)
    if layout == 0:  # BSH
        T = b_size * s_size
    else:            # TH
        T = sum(sqs_list)

    torch.manual_seed(seed)

    # ── x (BSH: (B,S,H); TH: (T,H)) ──
    if layout == 0:
        x = torch.rand((b_size, s_size, h_size), device=device,
                       dtype=torch.float32) * weight_scale * 2 - weight_scale
    else:
        x = torch.rand((T, h_size), device=device,
                       dtype=torch.float32) * weight_scale * 2 - weight_scale
    x = x.to(torch_dtype)

    # ── wkv, wgate: (C, H) ──
    wkv = (torch.rand((c_size, h_size), device=device, dtype=torch.float32)
           * weight_scale * 2 - weight_scale).to(torch_dtype)
    wgate = (torch.rand((c_size, h_size), device=device, dtype=torch.float32)
             * weight_scale * 2 - weight_scale).to(torch_dtype)

    # ── d_cmp_kv, kv, softmax_score — shape depends on layout ──
    # BSH:  d_cmp_kv (B, out_stride, D),  kv/sm (B, out_stride, sc_kv_rows, D)
    # TH:   d_cmp_kv (output_rows, D),    kv/sm (output_rows, sc_kv_rows, D)
    if layout == 0:  # BSH
        out_stride = (s_size + cmp_ratio - 1) // cmp_ratio
        d_cmp_kv = torch.randn(b_size, out_stride, d_size,
                               device=device, dtype=torch_dtype)
        kv = torch.randn(b_size, out_stride, sc_kv_rows, d_size,
                         device=device, dtype=torch.float32)
        softmax_score = torch.softmax(
            torch.randn(b_size, out_stride, sc_kv_rows, d_size, device=device),
            dim=2)
    else:            # TH
        output_rows = min(T, T // cmp_ratio + b_size)
        d_cmp_kv = torch.randn(output_rows, d_size,
                               device=device, dtype=torch_dtype)
        kv = torch.randn(output_rows, sc_kv_rows, d_size,
                         device=device, dtype=torch.float32)
        softmax_score = torch.softmax(
            torch.randn(output_rows, sc_kv_rows, d_size, device=device),
            dim=1)

    # ── cu_seqlens: (B+1,) int32, TH only (golden uses it to detect layout) ──
    if layout == 1:
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(sqs_list).cumsum(0).tolist(),
            device=device, dtype=torch.int32)
    else:
        cu_seqlens = None

    # ── seqused: (B,) int32 (only when sqs was explicitly provided) ──
    seqused = (torch.tensor(sqs_list, device=device, dtype=torch.int32)
               if sqs is not None else None)

    # ── start_pos: (B,) int32 (only when sps was explicitly provided) ──
    start_pos = (torch.tensor(sps_list, device=device, dtype=torch.int32)
                 if sps is not None else None)

    return {
        "x": x, "wkv": wkv, "wgate": wgate,
        "d_cmp_kv": d_cmp_kv, "kv": kv, "softmax_score": softmax_score,
        "cu_seqlens": cu_seqlens, "seqused": seqused, "start_pos": start_pos,
        "cmp_ratio": cmp_ratio, "coff": coff,
    }


# ════════════════════════════════════════════════════════════════════════════
# Unified case runner — calls compressor_grad_golden directly
# ════════════════════════════════════════════════════════════════════════════
def _run_case(case_name, case):
    """Generate inputs for one case and execute ``compressor_grad_golden``.

    Returns the four golden gradients (d_x, d_wkv, d_wgate, d_ape).
    """
    layout_name = "BSH" if case["layout"] == 0 else "TH"
    dtype_name  = "BF16" if case["dataType"] == 0 else "FP16"
    logger.info("*" * 60)
    logger.info(f"Run case: {case_name} "
                f"({layout_name}/{dtype_name}/coff={case['coff']}"
                f"/B={case['b_size']}/S={case['s_size']}/D={case['d_size']}"
                f"/cr={case['cmp_ratio']})")
    logger.info("*" * 60 + "\n")

    inputs = _gen_inputs(case)

    d_x, d_wkv, d_wgate, d_ape = compressor_grad_golden(
        inputs["x"],
        inputs["wkv"],
        inputs["wgate"],
        inputs["d_cmp_kv"],
        inputs["softmax_score"],
        inputs["kv"],
        inputs["cu_seqlens"],
        inputs["seqused"],
        inputs["start_pos"],
        cmp_ratio=inputs["cmp_ratio"],
        coff=inputs["coff"],
    )

    logger.info(f"[{case_name}] golden outputs: "
                f"d_x={tuple(d_x.shape)} d_wkv={tuple(d_wkv.shape)} "
                f"d_wgate={tuple(d_wgate.shape)} d_ape={tuple(d_ape.shape)}")
    return d_x, d_wkv, d_wgate, d_ape


# ════════════════════════════════════════════════════════════════════════════
# Unified runner
# ════════════════════════════════════════════════════════════════════════════
def run_all_cases():
    """Run every case in CASES through the golden reference."""
    logger.info("=" * 60)
    logger.info("PyPTO Pro compressor_grad unified test (cube + vec via golden)")
    logger.info("=" * 60 + "\n")
    for case_name, case in CASES.items():
        _run_case(case_name, case)
    logger.info("All {} cases executed through compressor_grad_golden.".format(len(CASES)))


def main():
    run_all_cases()


if __name__ == "__main__":
    main()
