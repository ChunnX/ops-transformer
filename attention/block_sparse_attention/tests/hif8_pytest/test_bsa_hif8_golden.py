#!/usr/bin/env python3
"""BSA HIF8 precision test and kernel-order CPU golden.

The golden deliberately follows the current Ascend950 BSA HIF8 kernel rather
than using one dense ``torch.softmax`` call:

* blockShape X/Y defines the sparse-mask grid;
* selected Y blocks are gathered in ascending order;
* every X block is split into 128-row Q tiles and the gathered KV stream into
  256-row KV tiles;
* MM1 applies the combined Q/K/attention scale through scalar FixPipe;
* online softmax uses ``exp(S - (m - ln(pScale)))``;
* P is rounded to HIF8 before MM2, while the denominator uses FP32 P;
* MM2 applies V scale through scalar FixPipe;
* LSE is ``m + ln(sum_scaled) - ln(pScale)``.

References used for this standalone test:

* BSA MXFP4 pytest/golden for blockShape and sparse gather semantics.
* QFA HIF8 golden for per-tensor HIF8 quantization and P casting.
* The BSA HIF8 kernel for tile sizes, operation ordering and LSE correction.
"""

import argparse
import math
import os
import struct
import unittest
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

import generate_hifloat8_data as hif8_codec

try:
    import torch_npu
    if os.environ.get("BSA_HIF8_IGNORE_VENDOR_PERM", "0") == "1":
        _real_listdir = os.listdir

        def _listdir_for_hif8(path):
            try:
                return _real_listdir(path)
            except PermissionError:
                if str(path).endswith("/opp/vendors"):
                    return []
                raise

        # The local CANN installation has a root-owned vendors directory.  The
        # custom package path is supplied separately, so skipping only this
        # unreadable optional directory is safe for the standalone golden.
        os.listdir = _listdir_for_hif8
    # torch_npu must initialize against the real CANN root.  The optional
    # user overlay is applied only before cann_ops_transformer imports its
    # OpBuilder, so it can skip an unreadable system opp/vendors directory.
    _user_cann_overlay = os.environ.get("BSA_HIF8_CANN_OVERLAY", "")
    if _user_cann_overlay and os.path.isdir(_user_cann_overlay):
        os.environ["ASCEND_HOME_PATH"] = _user_cann_overlay
        os.environ["ASCEND_TOOLKIT_HOME"] = _user_cann_overlay
    from cann_ops_transformer.ops import block_sparse_attention

    _HAS_NPU = True
    _NPU_IMPORT_ERROR = None
except ImportError as exc:  # CPU-only import keeps golden unit-testable.
    torch_npu = None
    block_sparse_attention = None
    _HAS_NPU = False
    _NPU_IMPORT_ERROR = exc

# compare_utils configures torch.npu at import time, so import it only after
# torch_npu has registered the backend (the same order as the existing smoke).
from compare_utils import check_result_printf_res


HIF8_MAX = 32768.0
Q_BASE_TILE = 128
KV_BASE_TILE = 256
MIN_FLOAT = -3.402823466e38
DEFAULT_DEVICE = int(os.environ.get("BSA_HIF8_DEVICE_ID", "0"))
# BSA HIF8 golden acceptance criterion for this test suite.  The compare
# helper's generic 99.5% threshold is intentionally tightened here: both
# Attention Out and LSE must report 100% of elements within their dtype rules.
REQUIRED_FULFILL_PERCENT = 100.0
SUPPORTED_LAYOUTS = ("BNSD", "BSND", "TND")
SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16, torch.float16)


@dataclass
class Hif8CaseResult:
    out_pct: float
    out_result: str
    lse_pct: float
    lse_result: str
    golden_out: torch.Tensor
    npu_out: torch.Tensor
    golden_lse: torch.Tensor
    npu_lse: torch.Tensor


def _float32(value) -> float:
    """Round a Python/tensor scalar exactly once to IEEE FP32."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _float32_mul(lhs, rhs) -> float:
    """Match one C++ ``float`` multiplication."""
    return _float32(_float32(lhs) * _float32(rhs))


def fp32_to_my19bit(value) -> torch.Tensor:
    """Torch equivalent of fp_test_bsa.py's scalar FixPipe-scale model."""
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(())
    bits = tensor.view(torch.int32)
    # int32 -8192 has the same bit pattern as uint32 0xFFFFE000.
    mask = torch.tensor(-8192, dtype=torch.int32, device=tensor.device)
    return torch.bitwise_and(bits, mask).view(torch.float32)


def make_mm1_fixpipe_scale(q_scale, k_scale, attention_scale) -> float:
    """Match ``float q * float k * float attentionScale`` then FixPipe."""
    combined = _float32_mul(q_scale, k_scale)
    combined = _float32_mul(combined, attention_scale)
    return fp32_to_my19bit(combined).item()


def quantize_hif8_per_tensor(src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """QFA HIF8 per-tensor quantization: return uint8 storage and scale [1]."""
    src = src.to(torch.float32)
    max_abs = float(src.abs().max())
    scale_value = max_abs / HIF8_MAX if max_abs != 0.0 else 1.0
    scale = torch.tensor([scale_value], dtype=torch.float32)
    encoded = hif8_codec.trans_float_tensor_to_hifuint8(
        (src / scale_value).contiguous(), round_mode="round", over_mode=True
    )
    return encoded.contiguous(), scale


def decode_hif8(src: torch.Tensor) -> torch.Tensor:
    return hif8_codec.trans_hifuint8_tensor_to_float(src, over_mode=True).to(torch.float32)


def cast_hif8(src: torch.Tensor) -> torch.Tensor:
    """Match the HIF8 Cast used by the vector-side P path."""
    encoded = hif8_codec.trans_float_tensor_to_hifuint8(
        src.to(torch.float32).contiguous(), round_mode="round", over_mode=True
    )
    return decode_hif8(encoded)


def _resolve_output_dtype(dtype) -> torch.dtype:
    if dtype in (torch.bfloat16, "bfloat16", "bf16"):
        return torch.bfloat16
    if dtype in (torch.float16, "float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"attention_out_dtype must be float16 or bfloat16, got {dtype}")


def _dtype_name(dtype: torch.dtype) -> str:
    return "bfloat16" if dtype == torch.bfloat16 else "float16"


def _bnsd_to_layout(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """Convert canonical [B,N,S,D] storage to a BSA physical input layout."""
    layout = layout.upper()
    if layout == "BNSD":
        return tensor.contiguous()
    bsnd = tensor.permute(0, 2, 1, 3).contiguous()
    if layout == "BSND":
        return bsnd
    if layout == "TND":
        return bsnd.flatten(0, 1).contiguous()
    raise ValueError(f"unsupported input layout: {layout}")


def _layout_to_bnsd(
    tensor: torch.Tensor,
    layout: str,
    seq_lengths: Sequence[int],
) -> torch.Tensor:
    """Convert BNSD/BSND/TND physical storage into canonical [B,N,S,D]."""
    layout = layout.upper()
    if layout == "BNSD":
        if tensor.ndim != 4:
            raise ValueError(f"BNSD tensor must be 4D, got {tensor.ndim}D")
        return tensor.contiguous()
    if layout == "BSND":
        if tensor.ndim != 4:
            raise ValueError(f"BSND tensor must be 4D, got {tensor.ndim}D")
        return tensor.permute(0, 2, 1, 3).contiguous()
    if layout != "TND":
        raise ValueError(f"unsupported input layout: {layout}")
    if tensor.ndim != 3:
        raise ValueError(f"TND tensor must be 3D, got {tensor.ndim}D")
    if tensor.shape[0] != sum(int(x) for x in seq_lengths):
        raise ValueError(
            f"TND T={tensor.shape[0]} must equal sum(seq_lengths)={sum(seq_lengths)}"
        )
    batch = len(seq_lengths)
    heads, dim = tensor.shape[1], tensor.shape[2]
    max_seq = max(int(x) for x in seq_lengths)
    result = torch.zeros((batch, heads, max_seq, dim), dtype=tensor.dtype)
    token_offset = 0
    for batch_idx, seq_len_value in enumerate(seq_lengths):
        seq_len = int(seq_len_value)
        result[batch_idx, :, :seq_len] = tensor[
            token_offset:token_offset + seq_len
        ].permute(1, 0, 2)
        token_offset += seq_len
    return result


def _bnsd_output_to_layout(
    tensor: torch.Tensor,
    layout: str,
    seq_lengths: Sequence[int],
) -> torch.Tensor:
    """Convert canonical output [B,N,S,D] to the requested physical layout."""
    layout = layout.upper()
    if layout == "BNSD":
        return tensor.contiguous()
    if layout == "BSND":
        return tensor.permute(0, 2, 1, 3).contiguous()
    if layout == "TND":
        pieces = [
            tensor[batch_idx, :, :int(seq_len)].permute(1, 0, 2)
            for batch_idx, seq_len in enumerate(seq_lengths)
        ]
        return torch.cat(pieces, dim=0).contiguous()
    raise ValueError(f"unsupported output layout: {layout}")


def _bns_lse_to_layout(
    tensor: torch.Tensor,
    layout: str,
    seq_lengths: Sequence[int],
) -> torch.Tensor:
    """Convert canonical LSE [B,N,S] to BNS1/BSN1/TN1."""
    layout = layout.upper()
    if layout == "BNSD":
        return tensor.unsqueeze(-1).contiguous()
    if layout == "BSND":
        return tensor.permute(0, 2, 1).unsqueeze(-1).contiguous()
    if layout == "TND":
        pieces = [
            tensor[batch_idx, :, :int(seq_len)].transpose(0, 1)
            for batch_idx, seq_len in enumerate(seq_lengths)
        ]
        return torch.cat(pieces, dim=0).unsqueeze(-1).contiguous()
    raise ValueError(f"unsupported LSE layout: {layout}")


def build_block_sparse_mask(
    batch: int,
    q_heads: int,
    q_seq: int,
    kv_seq: int,
    block_shape: Sequence[int],
    kind: str,
) -> torch.Tensor:
    """Build BSA mask [B,Nq,ceil(Q/X),ceil(KV/Y)]; nonzero means selected."""
    block_x, block_y = map(int, block_shape)
    ceil_q = (q_seq + block_x - 1) // block_x
    ceil_kv = (kv_seq + block_y - 1) // block_y
    if kind == "dense":
        return torch.ones(batch, q_heads, ceil_q, ceil_kv, dtype=torch.int8)

    mask = torch.zeros(batch, q_heads, ceil_q, ceil_kv, dtype=torch.int8)
    if kind == "alternating":
        for qb in range(ceil_q):
            parity = qb & 1
            selected = list(range(parity, ceil_kv, 2))
            if not selected:
                selected = [ceil_kv - 1]
            mask[:, :, qb, selected] = 1
        return mask
    if kind == "gap_sparse":
        mask[..., 0] = 1
        if ceil_kv > 1:
            mask[..., -1] = 1
        return mask
    raise ValueError(f"unsupported mask kind: {kind}")


def _gathered_kv_positions(
    mask_row: torch.Tensor, kv_seq: int, block_y: int
) -> List[int]:
    """Match mask2idx + SparseK/V load: selected Y blocks, ascending, compacted."""
    positions: List[int] = []
    for y_idx in range(mask_row.numel()):
        if int(mask_row[y_idx]) == 0:
            continue
        begin = y_idx * block_y
        end = min(begin + block_y, kv_seq)
        positions.extend(range(begin, end))
    return positions


def _mm1_raw_dn(q_tile: torch.Tensor, k_tile: torch.Tensor) -> torch.Tensor:
    """K*Q^T in Cube, returned in convenient [Q,K] semantic order."""
    # Current HIF8 path supports D=128, i.e. one L0 K stage.  Keeping the
    # explicit K*Q^T expression documents and matches the DN path.
    return torch.matmul(k_tile, q_tile.transpose(0, 1)).transpose(0, 1).contiguous()


def _mm2_raw_tiled(p_tile: torch.Tensor, v_tile: torch.Tensor) -> torch.Tensor:
    """Match MM2 K-loop: 128-wide L0A/L0B stages accumulated in FP32."""
    q_rows = p_tile.shape[0]
    out = torch.zeros((q_rows, v_tile.shape[1]), dtype=torch.float32)
    for k_begin in range(0, p_tile.shape[1], 128):
        k_end = min(k_begin + 128, p_tile.shape[1])
        part = torch.matmul(p_tile[:, k_begin:k_end], v_tile[k_begin:k_end])
        out = out + part.to(torch.float32)
    return out


def _online_softmax_hif8(
    scores: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    ln_p_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """QFA HIF8 online-softmax step, expressed in BSA FP8 benchmark form.

    The persisted maximum remains the real score maximum.  Only the temporary
    exponent baseline is shifted by ``ln(pScale)``, so both the FP32
    denominator and HIF8 P carry the same multiplicative P scale.
    """
    tile_max = torch.max(scores, dim=-1).values.to(torch.float32)
    new_max = torch.maximum(running_max, tile_max).to(torch.float32)
    exp_max = torch.exp(running_max - new_max).to(torch.float32)
    exp_baseline = (new_max - ln_p_scale).to(torch.float32)
    p_fp32 = torch.exp(scores - exp_baseline[:, None]).to(torch.float32)
    tile_sum = torch.sum(p_fp32, dim=-1).to(torch.float32)
    p_hif8 = cast_hif8(p_fp32)
    updated_sum = (exp_max * running_sum + tile_sum).to(torch.float32)
    return p_hif8, updated_sum, exp_max, new_max


def ref_flash_block_sparse_attention_hif8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_dequant_scale: torch.Tensor,
    k_dequant_scale: torch.Tensor,
    v_dequant_scale: torch.Tensor,
    p_quant_scale: torch.Tensor,
    attention_scale: float,
    *,
    debug_stages: bool = False,
    global_q_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BSA FP8-style block reference with QFA HIF8 numerical semantics.

    ``query`` is one gathered BSA X block in semantic [Q,D] order. ``key``
    and ``value`` are the selected Y blocks compacted in sparse-index order.
    The inner loops follow the HIF8 kernel base tiles: Q=128 and KV=256.
    """
    mm1_scale = make_mm1_fixpipe_scale(
        q_dequant_scale.reshape(-1)[0],
        k_dequant_scale.reshape(-1)[0],
        attention_scale,
    )
    mm2_scale = fp32_to_my19bit(
        _float32(v_dequant_scale.reshape(-1)[0])
    ).item()
    p_scale = _float32(p_quant_scale.reshape(-1)[0])
    if p_scale <= 0.0:
        raise ValueError(f"p_quant_scale must be positive, got {p_scale}")
    ln_p_scale = torch.tensor(_float32(math.log(p_scale)), dtype=torch.float32)

    q_rows = query.shape[0]
    value_dim = value.shape[1]
    block_out = torch.zeros((q_rows, value_dim), dtype=torch.float32)
    block_lse = torch.full((q_rows,), float("inf"), dtype=torch.float32)

    for q_begin in range(0, q_rows, Q_BASE_TILE):
        q_end = min(q_begin + Q_BASE_TILE, q_rows)
        q_tile = query[q_begin:q_end]
        tile_q_rows = q_end - q_begin
        running_max = torch.full((tile_q_rows,), MIN_FLOAT, dtype=torch.float32)
        running_sum = torch.zeros((tile_q_rows,), dtype=torch.float32)
        running_out = torch.zeros((tile_q_rows, value_dim), dtype=torch.float32)

        for kv_begin in range(0, key.shape[0], KV_BASE_TILE):
            kv_end = min(kv_begin + KV_BASE_TILE, key.shape[0])
            k_tile = key[kv_begin:kv_end]
            v_tile = value[kv_begin:kv_end]

            # BSA FP8 benchmark: Cube raw MM1 first, then scalar FixPipe.
            s_raw = _mm1_raw_dn(q_tile, k_tile)
            scores = (s_raw * mm1_scale).to(torch.float32)

            # QFA HIF8: P = Cast_HIF8(exp(S - (m - ln(pScale)))).
            p_hif8, new_sum, exp_max, new_max = _online_softmax_hif8(
                scores, running_max, running_sum, ln_p_scale
            )

            # BSA FP8 benchmark: 128-wide MM2 accumulation, then V FixPipe.
            tile_out_raw = _mm2_raw_tiled(p_hif8, v_tile)
            tile_out = (tile_out_raw * mm2_scale).to(torch.float32)

            if debug_stages:
                print(
                    "[BSA HIF8 GOLDEN DBG] tile "
                    f"q=[{global_q_offset + q_begin},{global_q_offset + q_end}) "
                    f"kv=[{kv_begin},{kv_end})"
                )
                # The two AIVs own Q rows [0,64) and [64,128).  Printing two
                # rows from each half lets us compare the logical values with
                # the split-M device dumps without reproducing UB padding.
                debug_rows = (0, 32, 64, 96)
                _print_debug_stage("S", scores, debug_rows)
                _print_debug_stage("P", p_hif8, debug_rows)
                _print_debug_stage("OTmp", tile_out, debug_rows)

            running_out = (
                exp_max[:, None] * running_out + tile_out
            ).to(torch.float32)
            running_sum = new_sum
            running_max = new_max

        block_out[q_begin:q_end] = running_out / running_sum[:, None]
        block_lse[q_begin:q_end] = (
            running_max + torch.log(running_sum) - ln_p_scale
        ).to(torch.float32)

    return block_out, block_lse


def _print_debug_stage(name: str, tile: torch.Tensor, row_ids: Sequence[int], col_count: int = 64) -> None:
    """Print logical stage rows that can be matched to the device dump."""
    print(f"[BSA HIF8 GOLDEN DBG] {name} shape={tuple(tile.shape)}")
    for row_id in row_ids:
        if row_id < tile.shape[0]:
            values = tile[row_id, :col_count].detach().cpu().tolist()
            print(f"[BSA HIF8 GOLDEN DBG] {name} row={row_id} values={values}")


def bsa_hif8_kernel_golden(
    query_hif8: torch.Tensor,
    key_hif8: torch.Tensor,
    value_hif8: torch.Tensor,
    block_sparse_mask: torch.Tensor,
    block_shape: Sequence[int],
    q_dequant_scale: torch.Tensor,
    k_dequant_scale: torch.Tensor,
    v_dequant_scale: torch.Tensor,
    p_quant_scale: torch.Tensor,
    attention_scale: float,
    actual_q_seq: Sequence[int] = None,
    actual_kv_seq: Sequence[int] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    input_layout: str = "BNSD",
    debug_stages: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU golden following the HIF8 kernel for BNSD, BSND or TND I/O."""
    input_layout = input_layout.upper()
    if input_layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"input_layout must be one of {SUPPORTED_LAYOUTS}, got {input_layout}")
    out_dtype = _resolve_output_dtype(out_dtype)

    if input_layout == "TND":
        if actual_q_seq is None or actual_kv_seq is None:
            raise ValueError("TND golden requires actual_q_seq and actual_kv_seq")
        actual_q_seq = [int(x) for x in actual_q_seq]
        actual_kv_seq = [int(x) for x in actual_kv_seq]
    else:
        batch_dim = query_hif8.shape[0]
        if input_layout == "BNSD":
            q_seq_dim = query_hif8.shape[2]
            kv_seq_dim = key_hif8.shape[2]
        else:
            q_seq_dim = query_hif8.shape[1]
            kv_seq_dim = key_hif8.shape[1]
        actual_q_seq = list(actual_q_seq or [q_seq_dim] * batch_dim)
        actual_kv_seq = list(actual_kv_seq or [kv_seq_dim] * batch_dim)

    query_bnsd = _layout_to_bnsd(query_hif8, input_layout, actual_q_seq)
    key_bnsd = _layout_to_bnsd(key_hif8, input_layout, actual_kv_seq)
    value_bnsd = _layout_to_bnsd(value_hif8, input_layout, actual_kv_seq)

    batch, q_heads, q_seq_padded, head_dim = query_bnsd.shape
    _, kv_heads, kv_seq_padded, value_dim = value_bnsd.shape
    if len(actual_q_seq) != batch or len(actual_kv_seq) != batch:
        raise ValueError("actual Q/KV sequence-length lists must match batch")
    if head_dim != 128 or value_dim != 128:
        raise ValueError("the current BSA HIF8 kernel path requires D=128")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    block_x, block_y = map(int, block_shape)
    if block_x < 128 or block_x % 128 or block_y < 256 or block_y % 256:
        raise ValueError("HIF8 blockShape must be X=128*n and Y=256*n")

    q_raw = decode_hif8(query_bnsd)
    k_raw = decode_hif8(key_bnsd)
    v_raw = decode_hif8(value_bnsd)

    out = torch.zeros((batch, q_heads, q_seq_padded, value_dim), dtype=torch.float32)
    lse = torch.full((batch, q_heads, q_seq_padded), float("inf"), dtype=torch.float32)
    group_size = q_heads // kv_heads

    for b_idx in range(batch):
        q_seq = int(actual_q_seq[b_idx])
        kv_seq = int(actual_kv_seq[b_idx])
        x_block_count = (q_seq + block_x - 1) // block_x
        for q_head in range(q_heads):
            kv_head = q_head // group_size
            for x_idx in range(x_block_count):
                gathered = _gathered_kv_positions(
                    block_sparse_mask[b_idx, q_head, x_idx], kv_seq, block_y
                )
                # The production kernel assumes each task has at least one
                # selected Y block.  Keep all-zero rows well-defined in golden.
                if not gathered:
                    continue
                gathered_idx = torch.tensor(gathered, dtype=torch.long)
                q_begin = x_idx * block_x
                q_end = min((x_idx + 1) * block_x, q_seq)
                q_block = q_raw[b_idx, q_head, q_begin:q_end]
                key_gathered = k_raw[b_idx, kv_head].index_select(0, gathered_idx)
                value_gathered = v_raw[b_idx, kv_head].index_select(0, gathered_idx)

                out_slice, lse_slice = ref_flash_block_sparse_attention_hif8(
                    q_block,
                    key_gathered,
                    value_gathered,
                    q_dequant_scale,
                    k_dequant_scale,
                    v_dequant_scale,
                    p_quant_scale,
                    attention_scale,
                    debug_stages=debug_stages,
                    global_q_offset=q_begin,
                )
                out[b_idx, q_head, q_begin:q_end] = out_slice
                lse[b_idx, q_head, q_begin:q_end] = lse_slice

    return (
        _bnsd_output_to_layout(out.to(out_dtype), input_layout, actual_q_seq),
        _bns_lse_to_layout(lse, input_layout, actual_q_seq),
    )


def _make_inputs(
    batch: int,
    q_heads: int,
    kv_heads: int,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    seed: int,
    data_ranges: Sequence[float],
) -> Tuple[torch.Tensor, ...]:
    """Use deterministic FP16 uniform source data for Q/K/V.

    Each entry is the positive amplitude of a symmetric interval, so
    ``data_ranges=(5, 5, 5)`` generates values in ``[-5, 5]`` for all Q/K/V.
    The larger interval is useful for exercising the FixPipe scale and
    softmax correction with non-unit input magnitudes.
    """
    if (len(data_ranges) != 3 or
            any(not math.isfinite(float(rng)) or float(rng) < 0.0 or
                float(rng) > 5.0 for rng in data_ranges)):
        raise ValueError("data_ranges must contain three amplitudes in [0, 5]")
    generators = []
    for offset in (0, 1, 2):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + offset)
        generators.append(gen)
    q_src = data_ranges[0] * (
        torch.rand((batch, q_heads, q_seq, head_dim), dtype=torch.float16, generator=generators[0]) * 2 - 1
    ).float()
    k_src = data_ranges[1] * (
        torch.rand((batch, kv_heads, kv_seq, head_dim), dtype=torch.float16, generator=generators[1]) * 2 - 1
    ).float()
    v_src = data_ranges[2] * (
        torch.rand((batch, kv_heads, kv_seq, head_dim), dtype=torch.float16, generator=generators[2]) * 2 - 1
    ).float()
    q, q_scale = quantize_hif8_per_tensor(q_src)
    k, k_scale = quantize_hif8_per_tensor(k_src)
    v, v_scale = quantize_hif8_per_tensor(v_src)
    return q, k, v, q_scale, k_scale, v_scale


def run_bsa_hif8_case(
    *,
    batch: int = 1,
    q_heads: int = 1,
    kv_heads: int = 1,
    q_seq: int = 128,
    kv_seq: int = 256,
    head_dim: int = 128,
    block_shape: Sequence[int] = (128, 256),
    mask_kind: str = "dense",
    p_scale: float = 32.0,
    attention_scale: float = None,
    seed: int = 54,
    # Keep every generated source element in [-1, 1], while retaining
    # different amplitudes for Q/K/V so the case is not an all-one fixture.
    data_ranges: Sequence[float] = (0.91, 0.87, 0.77),
    input_layout: str = "BNSD",
    attention_out_dtype: torch.dtype = torch.bfloat16,
    device: int = DEFAULT_DEVICE,
) -> Hif8CaseResult:
    """Generate, run CPU golden/NPU BSA, and compare O plus LSE."""
    if not _HAS_NPU:
        raise RuntimeError(f"NPU BSA import failed: {_NPU_IMPORT_ERROR}")
    input_layout = input_layout.upper()
    if input_layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"input_layout must be one of {SUPPORTED_LAYOUTS}, got {input_layout}")
    attention_out_dtype = _resolve_output_dtype(attention_out_dtype)
    out_dtype_name = _dtype_name(attention_out_dtype)
    attention_scale = float(attention_scale if attention_scale is not None else head_dim ** -0.5)
    q_bnsd, k_bnsd, v_bnsd, q_scale, k_scale, v_scale = _make_inputs(
        batch, q_heads, kv_heads, q_seq, kv_seq, head_dim, seed, data_ranges
    )
    q = _bnsd_to_layout(q_bnsd, input_layout)
    k = _bnsd_to_layout(k_bnsd, input_layout)
    v = _bnsd_to_layout(v_bnsd, input_layout)
    block_mask = build_block_sparse_mask(
        batch, q_heads, q_seq, kv_seq, block_shape, mask_kind
    )
    p_scale_tensor = torch.tensor([p_scale], dtype=torch.float32)

    print(
        f"[BSA HIF8 config] B={batch} Nq={q_heads} Nkv={kv_heads} "
        f"QS={q_seq} KVS={kv_seq} D={head_dim} block={list(block_shape)} "
        f"layout={input_layout} outDtype={out_dtype_name} mask={mask_kind} "
        f"pScale={p_scale} attentionScale={attention_scale} "
        f"dataRange=Q[-{data_ranges[0]},+{data_ranges[0]}],"
        f"K[-{data_ranges[1]},+{data_ranges[1]}],"
        f"V[-{data_ranges[2]},+{data_ranges[2]}]"
    )
    print(
        "[BSA HIF8 scales] "
        f"q={q_scale.item():.10g} k={k_scale.item():.10g} v={v_scale.item():.10g} "
        f"mm1Fix={make_mm1_fixpipe_scale(q_scale[0], k_scale[0], attention_scale):.10g} "
        f"vFix={fp32_to_my19bit(_float32(v_scale[0])).item():.10g}"
    )

    golden_out, golden_lse = bsa_hif8_kernel_golden(
        q, k, v, block_mask, block_shape,
        q_scale, k_scale, v_scale, p_scale_tensor, attention_scale,
        actual_q_seq=[q_seq] * batch,
        actual_kv_seq=[kv_seq] * batch,
        out_dtype=attention_out_dtype,
        input_layout=input_layout,
        debug_stages=os.environ.get("BSA_HIF8_DEBUG_STAGES", "0") == "1",
    )

    torch_npu.npu.set_device(device)
    npu_out, npu_lse = block_sparse_attention(
        q.npu(),
        k.npu(),
        v.npu(),
        block_mask.npu(),
        list(block_shape),
        q_input_layout=input_layout,
        kv_input_layout=input_layout,
        num_key_value_heads=kv_heads,
        scale_value=attention_scale,
        inner_precise=0,
        actual_seq_lengths=[q_seq] * batch,
        actual_seq_lengths_kv=[kv_seq] * batch,
        return_softmax_lse=True,
        mask_type=0,
        quant_mode=5,
        block_size=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        dst_type_max=0.0,
        q_dequant_scale=q_scale.npu(),
        k_dequant_scale=k_scale.npu(),
        v_dequant_scale=v_scale.npu(),
        p_quant_scale=p_scale_tensor.npu(),
        attention_out_dtype=attention_out_dtype,
    )
    torch_npu.npu.synchronize()
    npu_out_cpu = npu_out.cpu()
    npu_lse_cpu = npu_lse.float().cpu().reshape_as(golden_lse)

    print("[BSA HIF8] compare attention output")
    out_pct, out_result = check_result_printf_res(
        golden_out.float(), npu_out_cpu.float(), out_dtype_name
    )
    print("[BSA HIF8] compare softmax LSE")
    lse_pct, lse_result = check_result_printf_res(
        golden_lse, npu_lse_cpu, "float32"
    )
    max_out_abs = (golden_out.float() - npu_out_cpu.float()).abs().max().item()
    max_lse_abs = (golden_lse - npu_lse_cpu).abs().max().item()
    print(
        f"[BSA HIF8 summary] out={out_result}({out_pct:.6f}%) maxAbs={max_out_abs:.8g}; "
        f"lse={lse_result}({lse_pct:.6f}%) maxAbs={max_lse_abs:.8g}"
    )
    case_status = (
        "PASS" if out_result == "Pass" and lse_result == "Pass" and
        out_pct == REQUIRED_FULFILL_PERCENT and lse_pct == REQUIRED_FULFILL_PERCENT
        else "FAIL"
    )
    print(f"[BSA HIF8 CASE RESULT] {case_status} "
          f"B={batch} Nq={q_heads} Nkv={kv_heads} QS={q_seq} KVS={kv_seq} "
          f"D={head_dim} layout={input_layout} outDtype={out_dtype_name} "
          f"block={list(block_shape)} mask={mask_kind} "
          f"pScale={p_scale} seed={seed}")
    print("=" * 96)
    print()
    print()
    print()
    return Hif8CaseResult(
        out_pct, out_result, lse_pct, lse_result,
        golden_out, npu_out_cpu, golden_lse, npu_lse_cpu,
    )


class TestBsaHif8Golden(unittest.TestCase):
    """Focused cases for the current BNSD HIF8 kernel path."""

    def _run(self, **kwargs):
        if not _HAS_NPU:
            self.skipTest(f"NPU BSA unavailable: {_NPU_IMPORT_ERROR}")
        result = run_bsa_hif8_case(**kwargs)
        self.assertEqual(result.out_result, "Pass")
        self.assertEqual(result.lse_result, "Pass")
        self.assertEqual(result.out_pct, REQUIRED_FULFILL_PERCENT)
        self.assertEqual(result.lse_pct, REQUIRED_FULFILL_PERCENT)

    def test_dense_base_tile_p32(self):
        self._run(q_seq=128, kv_seq=256, block_shape=(128, 256), p_scale=32.0)

    def test_dense_tail_q127_kv250_p16(self):
        self._run(q_seq=127, kv_seq=250, block_shape=(128, 256), p_scale=16.0, seed=61)

    def test_sparse_large_blockshape(self):
        # X/Y are sparse-mask blocks. Kernel still splits them into 128/256 tiles.
        self._run(
            q_seq=250,
            kv_seq=1280,
            block_shape=(256, 512),
            mask_kind="gap_sparse",
            p_scale=32.0,
            seed=73,
        )

    def test_gqa_two_q_heads(self):
        """GQA mapping: two Q heads share one KV head."""
        self._run(
            q_heads=2,
            kv_heads=1,
            q_seq=256,
            kv_seq=512,
            block_shape=(256, 256),
            p_scale=1.0,
            seed=81,
        )

    def test_noncontiguous_sparse_multitile(self):
        """Non-contiguous selected Y blocks exercise compact gather + 256 tiles."""
        self._run(
            q_seq=384,
            kv_seq=768,
            block_shape=(128, 256),
            mask_kind="alternating",
            p_scale=2.0,
            seed=89,
        )

    def test_q191_kv511_tail(self):
        """Both Q and KV have tails while pScale is not a power-of-two."""
        self._run(
            q_seq=191,
            kv_seq=511,
            block_shape=(128, 256),
            p_scale=7.0,
            seed=97,
        )

    def test_high_dynamic_range_qkv(self):
        """Exercise source values in [-5, 5] and the scaled softmax path."""
        self._run(
            q_seq=191,
            kv_seq=257,
            block_shape=(128, 256),
            p_scale=7.0,
            seed=117,
            data_ranges=(5.0, 5.0, 5.0),
        )

    def test_batch2_mha_tail_fractional_pscale(self):
        """Batch/head indexing, two-sided tails and fractional P scale."""
        self._run(
            batch=2,
            q_heads=2,
            kv_heads=2,
            q_seq=129,
            kv_seq=257,
            block_shape=(256, 512),
            p_scale=3.25,
            attention_scale=0.071,
            seed=103,
            data_ranges=(0.63, 0.92, 0.41),
        )

    def test_gqa_four_q_two_kv_sparse(self):
        """Generalized GQA mapping with non-contiguous sparse KV blocks."""
        self._run(
            q_heads=4,
            kv_heads=2,
            q_seq=257,
            kv_seq=1025,
            block_shape=(128, 256),
            mask_kind="alternating",
            p_scale=13.0,
            seed=109,
            data_ranges=(0.52, 0.79, 0.96),
        )

    def test_generated_210_cases(self):
        """Opt-in broad smoke matrix for QS/KVS, QN/KVN and sparse tiling.

        This is intentionally opt-in because 210 device launches are not a
        normal unit-test latency.  Use ``BSA_HIF8_RUN_GENERALIZED=1`` when a
        batch smoke run is desired; every case still compares O and LSE at the
        same 100% threshold as the focused tests.
        """
        if os.environ.get("BSA_HIF8_RUN_GENERALIZED", "0") != "1":
            self.skipTest("set BSA_HIF8_RUN_GENERALIZED=1 to run 210-case smoke matrix")
        if not _HAS_NPU:
            self.skipTest(f"NPU BSA unavailable: {_NPU_IMPORT_ERROR}")

        failures = []
        for case_id, cfg in enumerate(_generated_210_cases(), 1):
            with self.subTest(case=case_id, **cfg):
                try:
                    result = run_bsa_hif8_case(**cfg)
                    if (result.out_result != "Pass" or result.lse_result != "Pass" or
                            result.out_pct != REQUIRED_FULFILL_PERCENT or
                            result.lse_pct != REQUIRED_FULFILL_PERCENT):
                        failures.append((case_id, cfg, result.out_result, result.out_pct,
                                         result.lse_result, result.lse_pct))
                except Exception as exc:  # retain all failures for batch diagnosis
                    failures.append((case_id, cfg, type(exc).__name__, str(exc)))
        self.assertFalse(failures, "generalized HIF8 failures: %s" % failures)


def _generated_210_cases():
    """Deterministic, non-all-one HIF8 smoke matrix (210 configurations).

    Most cases retain the original ``[-1, 1]``-scale inputs.  A subset uses
    amplitudes up to 5 (i.e. source values in ``[-5, 5]``) to stress the
    quantization scales, FixPipe truncation and the ``ln(pScale)`` path.
    """
    small_q = (201, 223, 257, 319, 385, 511)
    small_kv = (257, 300, 511, 640, 769, 1025)
    large_q = (769, 1025, 1537, 2047, 3071)
    large_kv = (1537, 2049, 3073, 4095, 8191)
    head_shapes = ((1, 1), (2, 1), (2, 2), (4, 2), (4, 4))
    cases = []
    for i in range(210):
        if i < 180:
            q_seq = small_q[i % len(small_q)]
            kv_seq = small_kv[(3 * i + 1) % len(small_kv)]
            q_heads, kv_heads = head_shapes[i % len(head_shapes)]
            batch = 2 if i % 17 == 0 else 1
        else:
            q_seq = large_q[(i - 180) % len(large_q)]
            kv_seq = large_kv[(2 * i + 1) % len(large_kv)]
            q_heads, kv_heads = ((1, 1), (2, 1), (2, 2))[(i - 180) % 3]
            batch = 1
        block_shape = ((256, 512) if i % 7 == 0 else
                       (128, 256) if i % 3 else (256, 256))
        mask_kind = ("dense" if i % 5 else
                     "alternating" if i % 2 else "gap_sparse")
        p_scale = (0.5, 1.0, 2.0, 3.25, 7.0, 13.0, 32.0)[i % 7]
        if i % 23 == 0:
            ranges = (5.0, 5.0, 5.0)
        elif i % 23 == 11:
            ranges = (5.0, 4.0, 3.0)
        else:
            ranges = ((0.91, 0.87, 0.77), (0.63, 0.92, 0.41),
                      (0.52, 0.79, 0.96))[i % 3]
        attention_scale = (128.0 ** -0.5) * (0.75, 1.0, 1.25)[i % 3]
        cases.append(dict(batch=batch, q_heads=q_heads, kv_heads=kv_heads,
                          q_seq=q_seq, kv_seq=kv_seq, block_shape=block_shape,
                          mask_kind=mask_kind, p_scale=p_scale,
                          attention_scale=attention_scale, seed=7000 + i,
                          data_ranges=ranges, device=0))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=int, default=DEFAULT_DEVICE,
        help="logical NPU id; with ASCEND_RT_VISIBLE_DEVICES=6 use --device 0",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--q-seq", type=int, default=128)
    parser.add_argument("--kv-seq", type=int, default=256)
    parser.add_argument("--block-x", type=int, default=128)
    parser.add_argument("--block-y", type=int, default=256)
    parser.add_argument("--mask-kind", choices=("dense", "alternating", "gap_sparse"), default="dense")
    parser.add_argument("--p-scale", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--attention-scale", type=float, default=None)
    args = parser.parse_args()
    result = run_bsa_hif8_case(
        batch=args.batch,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        q_seq=args.q_seq,
        kv_seq=args.kv_seq,
        block_shape=(args.block_x, args.block_y),
        mask_kind=args.mask_kind,
        p_scale=args.p_scale,
        seed=args.seed,
        device=args.device,
        attention_scale=args.attention_scale,
    )
    if result.out_result != "Pass" or result.lse_result != "Pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
