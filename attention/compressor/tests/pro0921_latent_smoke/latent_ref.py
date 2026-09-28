# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Independent CPU reference (pure numpy, no torch / torch_npu / cann_ops) for
# the *formal* Pro0921 Compressor Vec1 contract, plus the grouped-decode
# x_cache window derivation that the arch35 kernel implements in
# compressor_tools.h (`CrossesCompressionBoundary` / `DeriveGroupedWindow`).
#
# Why a local reference instead of importing the repo golden:
#   tests/pytest/compressor_golden.py imports torch_npu / torchair /
#   cann_ops_transformer at module import time, so it cannot run in the build
#   host.  This file re-implements the same *documented* contract in numpy so
#   the smoke cases have a golden that is independent of the kernel binary and
#   runnable on a CPU-only host.  golden_cross_check.py compares this reference
#   against the repo's canonical formula on the arithmetic it mirrors.
#
# Contract (matches compressor/README.md + the plainly readable arch35 Vec1
# ops in op_kernel/arch35/compressor_block_vec*.h):
#   kv_state    = X @ Wkv^T
#   score_state = X @ Wgate^T + Ape
#   weights     = softmax(score_window, axis = group axis)
#   cmp_kv      = sum(weights * kv_window, axis = group axis)
#   state_cache = [kv_state | score_state] written back over the window rows
#   softmax_score (grad_enabled only) = weights
#   kv            (grad_enabled only) = kv_window (raw, pre-Hadamard)
#
# Window size is coff * cmp_ratio rows.  coff=2 prepends the previous group,
# which is read back from state_cache (or, when there is no history, filled the
# way the kernel fills it: score = SOFTMAX_MIN_NUM, kv = 0).

from pathlib import Path

import numpy as np


SOFTMAX_MIN_NUM = np.float32(-3.40e38)
FP32_TINY = np.finfo(np.float32).tiny


def to_bf16(array):
    """Round a float array to the nearest bfloat16-representable float32."""
    values = np.asarray(array, dtype=np.float32)
    as_int = values.view(np.uint32)
    rounding = ((as_int >> 16) & 1) + 0x7FFF
    return ((as_int + rounding) & 0xFFFF0000).view(np.float32)


def softmax_columns(z):
    """Column-wise softmax, fp32, with the NPU denorm flush-to-zero."""
    z = np.asarray(z, dtype=np.float32)
    z_max = np.max(z, axis=0, keepdims=True)
    exp_z = np.exp((z - z_max).astype(np.float32))
    exp_z = exp_z.astype(np.float32)
    exp_z[exp_z < FP32_TINY] = 0.0
    return (exp_z / np.sum(exp_z, axis=0, keepdims=True)).astype(np.float32)


def write_state_page_cache(state, sc_new_state, b_idx, start_seq_idx, end_seq_idx,
                           block_table, cache_mode=1):
    """Port of write_state_page_cache (cache_mode 1 = linear, 2 = ring)."""
    block_size = state.shape[1]
    seq_cnt = end_seq_idx - start_seq_idx
    if seq_cnt <= 0:
        return
    finish = 0
    while finish < seq_cnt:
        cur_seq = start_seq_idx + finish
        if cache_mode == 1:
            block_id = int(block_table[b_idx][cur_seq // block_size])
        else:
            block_id = int(block_table[b_idx])
        block_row = cur_seq % block_size
        count = min(block_size - block_row, seq_cnt - finish)
        if (cache_mode == 1 and block_id != 0) or cache_mode == 2:
            state[block_id, block_row:block_row + count, :] = \
                sc_new_state[finish:finish + count, :]
        finish += count


def read_state_page_cache(state, b_idx, start_seq_idx, end_seq_idx, block_table,
                          d_start, d_end, cache_mode=1):
    """Port of read_state_page_cache (raises if a linear block is the 0 sentinel)."""
    seq_cnt = end_seq_idx - start_seq_idx
    result = np.zeros((seq_cnt, d_end - d_start), dtype=np.float32)
    if seq_cnt <= 0:
        return result
    block_size = state.shape[1]
    finish = 0
    while finish < seq_cnt:
        cur_seq = start_seq_idx + finish
        if cache_mode == 1:
            block_id = int(block_table[b_idx][cur_seq // block_size])
            if block_id == 0:
                raise ValueError(
                    "read_state_page_cache hit the unallocated block sentinel "
                    "(block_id=0) at seq %d; the case fixture is inconsistent" % cur_seq)
        else:
            block_id = int(block_table[b_idx])
        block_row = cur_seq % block_size
        count = min(block_size - block_row, seq_cnt - finish)
        result[finish:finish + count, :] = \
            state[block_id, block_row:block_row + count, d_start:d_end]
        finish += count
    return result


def compress_golden(x, wkv, wgate, state_cache, ape, block_table, cmp_ratio, coff,
                    cache_mode=1, start_pos=None, cu_seqlens=None, seqused=None,
                    grad_enabled=False):
    """Canonical Compressor contract (numpy port of tests/pytest/compressor_golden.py).

    x:          (B, S, H) float32   (BSH layout only; this harness never uses TND)
    wkv/wgate:  (coff * head_dim, H) float32
    state_cache:(block_num, block_size, 2 * coff * head_dim) float32
    ape:        (cmp_ratio, coff * head_dim) float32
    returns (cmp_kv, softmax, kv, state_cache_out); softmax/kv are None unless
    grad_enabled.
    """
    x = np.asarray(x, dtype=np.float32)
    wkv = np.asarray(wkv, dtype=np.float32)
    wgate = np.asarray(wgate, dtype=np.float32)
    state_cache = np.asarray(state_cache, dtype=np.float32)
    ape = np.asarray(ape, dtype=np.float32)
    block_table = np.asarray(block_table, dtype=np.int64)
    if cu_seqlens is not None:
        raise NotImplementedError("this harness only builds BSH (B, S, H) cases")

    B, S, _ = x.shape
    head_dim = wkv.shape[0] // coff
    assert state_cache.shape[2] == 2 * coff * head_dim, (
        "state_cache last dim must be 2*coff*head_dim, got %d" % state_cache.shape[2])
    assert ape.shape == (cmp_ratio, coff * head_dim), "ape must be (cmp_ratio, coff*head_dim)"

    half = coff * head_dim
    kv_state = state_cache[:, :, :half].copy()
    score_state = state_cache[:, :, half:].copy()

    new_kv_state = np.matmul(x, wkv.T, dtype=np.float32).reshape(B * S, -1)
    new_score_state = np.matmul(x, wgate.T, dtype=np.float32).reshape(B * S, -1)

    sr = (S + cmp_ratio - 1) // cmp_ratio
    cmp_kv = np.zeros((B, sr, head_dim), dtype=np.float32)
    if grad_enabled:
        softmax = np.zeros((B, sr, coff * cmp_ratio, head_dim), dtype=np.float32)
        kv = np.zeros((B, sr, coff * cmp_ratio, head_dim), dtype=np.float32)
    else:
        softmax = kv = None

    start_pos = [0] * B if start_pos is None else [int(v) for v in start_pos]
    seqused = None if seqused is None else [int(v) for v in seqused]

    for b in range(B):
        out_sc = 0
        b_start = start_pos[b]
        b_used = S if seqused is None else seqused[b]
        compress_seq_id = (b_start + b_used) // cmp_ratio * cmp_ratio
        seq_idx = 0
        while seq_idx < b_used:
            start_seq = b_start + seq_idx
            end_seq = start_seq // cmp_ratio * cmp_ratio + cmp_ratio
            if end_seq > (b_start + b_used):
                end_seq = b_start + b_used
            base = b * S
            start_off = base + (start_seq - b_start)
            end_off = base + (end_seq - b_start)
            in_sc = start_seq % cmp_ratio
            end_in_sc = in_sc + (end_seq - start_seq)
            new_score_state[start_off:end_off, :] += ape[in_sc:end_in_sc, :]

            save = cache_mode == 2 and start_seq >= (compress_seq_id - (coff - 1) * cmp_ratio)
            if cache_mode == 1:
                save = True
            if save:
                write_state_page_cache(kv_state, new_kv_state[start_off:end_off, :], b,
                                       start_seq, end_seq, block_table, cache_mode)
                write_state_page_cache(score_state, new_score_state[start_off:end_off, :], b,
                                       start_seq, end_seq, block_table, cache_mode)

            if start_seq < compress_seq_id:
                sc_kv = np.zeros((coff, cmp_ratio, head_dim), dtype=np.float32)
                sc_score = np.full((coff, cmp_ratio, head_dim), -np.inf, dtype=np.float32)
                filled = np.zeros((coff, cmp_ratio), dtype=bool)
                for coff_id, (d_start, d_end) in enumerate(
                        [(i * head_dim, (i + 1) * head_dim) for i in range(coff)]):
                    cnt_from_state = 0
                    if coff_id == coff - 1:
                        # current group lives in the tail d-slice
                        if b_start == start_seq:
                            cnt_from_state = b_start % cmp_ratio
                            if cnt_from_state > 0:
                                copy_start = b_start - cnt_from_state
                                sc_kv[coff_id, :cnt_from_state] = read_state_page_cache(
                                    kv_state, b, copy_start, b_start, block_table,
                                    d_start, d_end, cache_mode)
                                sc_score[coff_id, :cnt_from_state] = read_state_page_cache(
                                    score_state, b, copy_start, b_start, block_table,
                                    d_start, d_end, cache_mode)
                                filled[coff_id, :cnt_from_state] = True
                        sc_kv[coff_id, cnt_from_state:] = new_kv_state[start_off:end_off, d_start:d_end]
                        sc_score[coff_id, cnt_from_state:] = new_score_state[start_off:end_off, d_start:d_end]
                        filled[coff_id, cnt_from_state:] = True
                    else:
                        cnt_from_state = 0
                        if b_start == start_seq and b_start >= cmp_ratio:
                            cnt_from_state = cmp_ratio
                            copy_start = b_start - b_start % cmp_ratio - cmp_ratio
                            sc_kv[coff_id, :cnt_from_state] = read_state_page_cache(
                                kv_state, b, copy_start, copy_start + cnt_from_state,
                                block_table, d_start, d_end, cache_mode)
                            sc_score[coff_id, :cnt_from_state] = read_state_page_cache(
                                score_state, b, copy_start, copy_start + cnt_from_state,
                                block_table, d_start, d_end, cache_mode)
                        elif start_seq - cmp_ratio < b_start:
                            cnt_from_state = b_start % cmp_ratio
                            if cnt_from_state > 0:
                                copy_start = b_start - b_start % cmp_ratio
                                sc_kv[coff_id, :cnt_from_state] = read_state_page_cache(
                                    kv_state, b, copy_start, b_start, block_table,
                                    d_start, d_end, cache_mode)
                                sc_score[coff_id, :cnt_from_state] = read_state_page_cache(
                                    score_state, b, copy_start, b_start, block_table,
                                    d_start, d_end, cache_mode)
                        if cnt_from_state < cmp_ratio:
                            pre_start = start_off - (cmp_ratio - cnt_from_state)
                            if pre_start >= 0:
                                sc_kv[coff_id, cnt_from_state:] = new_kv_state[pre_start:start_off, d_start:d_end]
                                sc_score[coff_id, cnt_from_state:] = new_score_state[pre_start:start_off, d_start:d_end]
                                filled[coff_id, cnt_from_state:] = True
                            # pre_start < 0 means the window starts at the very first
                            # group: the kernel's ReadState `isFirst` path calls
                            # DuplicateFirstBlock instead of reading new_state, so the
                            # remaining rows stay unfilled and are set below.
                        filled[coff_id, :cnt_from_state] = True

                sc_kv = sc_kv.reshape(coff * cmp_ratio, head_dim)
                sc_score = sc_score.reshape(coff * cmp_ratio, head_dim)
                # Rows the kernel never fills come from its "no history" path:
                # DuplicateFirstBlock<IS_SCORE> writes SOFTMAX_MIN_NUM for score
                # and FLOAT_ZERO for kv, not -inf / read state.  Mirror that so a
                # slice with no previous group does not poison the softmax with
                # nan (the reference leaves them -inf; both give zero weight, but
                # the grad_enabled softmax output must match the kernel value).
                unfilled = ~filled.reshape(coff * cmp_ratio)
                if unfilled.any():
                    sc_score[unfilled] = SOFTMAX_MIN_NUM
                    sc_kv[unfilled] = 0.0
                weights = softmax_columns(sc_score)
                if grad_enabled:
                    kv[b, out_sc] = sc_kv
                    softmax[b, out_sc] = weights
                cmp_kv[b, out_sc] = (sc_kv * weights).sum(axis=0)
                out_sc += 1

            seq_idx = end_seq - b_start

    out_state = np.zeros_like(state_cache)
    out_state[:, :, :half] = kv_state
    out_state[:, :, half:] = score_state
    return cmp_kv, softmax, kv, out_state


_HIF8 = None


def hif8_utils():
    """Load the repo's CANN-provided HiFloat8 codec (no numpy hifloat8 needed).

    quant_compressor/tests/pytest/hif8_utils.py is the operator's own reference
    encoder/decoder; reusing it keeps the quant golden on the same conversion the
    operator and its pytest use instead of a BF16 shortcut.
    """
    global _HIF8
    if _HIF8 is None:
        import importlib.util
        path = (Path(__file__).resolve().parents[3]
                / "quant_compressor" / "tests" / "pytest" / "hif8_utils.py")
        spec = importlib.util.spec_from_file_location("pro0921_hif8_utils", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HIF8 = module
    return _HIF8


def decode_hif8(codes):
    return hif8_utils().hif8_to_fp32_lut(np.asarray(codes)).astype(np.float32)


def encode_hif8(values):
    return hif8_utils().fp32_to_hif8_numpy(np.asarray(values, dtype=np.float32))


def quant_compress_golden(x_codes, wkv_codes, wgate_codes, x_descale, wkv_descale,
                          wgate_descale, state_cache, ape, block_table, cmp_ratio,
                          coff, cache_mode=1, start_pos=None, seqused=None,
                          grad_enabled=False):
    """QuantCompressor contract (quant_mode=1, A8W8 A-per-tensor W-per-channel).

    x_descale is a per-tensor scalar and the weight descales are per output
    channel, so the dequant factor folds into the weight rows:
        (W @ w^T)[:, j] * (x_d * w_d[j]) == W @ (w^T * x_d * w_d[j])
    which keeps the group softmax / reduce identical to `compress_golden`.
    """
    x_fp32 = decode_hif8(x_codes)
    wkv_fp32 = decode_hif8(wkv_codes)
    wgate_fp32 = decode_hif8(wgate_codes)
    x_descale = np.asarray(x_descale, dtype=np.float32).reshape(-1)
    wkv_descale = np.asarray(wkv_descale, dtype=np.float32).reshape(-1)
    wgate_descale = np.asarray(wgate_descale, dtype=np.float32).reshape(-1)
    assert x_descale.size == 1, "x_descale must be the per-tensor scalar"
    assert wkv_descale.size == wkv_fp32.shape[0], "wkv_descale must be per output channel"
    assert wgate_descale.size == wgate_fp32.shape[0], "wgate_descale must be per output channel"
    wkv_folded = wkv_fp32 * (x_descale[0] * wkv_descale)[:, None]
    wgate_folded = wgate_fp32 * (x_descale[0] * wgate_descale)[:, None]
    return compress_golden(x_fp32, wkv_folded, wgate_folded, state_cache, ape,
                           block_table, cmp_ratio, coff, cache_mode,
                           start_pos=start_pos, seqused=seqused,
                           grad_enabled=grad_enabled)


def quant_grouped_case_golden(x_codes, wkv_codes, wgate_codes, x_descale,
                              wkv_descale, wgate_descale, state_cache, ape,
                              block_table, x_cache_codes, slot, cmp_ratio, coff,
                              cache_mode=1, start_pos=None, seqused=None,
                              x_cache_update=True, grad_enabled=False):
    """Grouped-decode quant golden: ring refresh + window decode + quant project."""
    x_codes = np.asarray(x_codes)
    flat_tokens = x_codes.reshape(-1, x_codes.shape[-1])
    x_cache = np.array(x_cache_codes, copy=True)
    rows = x_cache.shape[1]
    raw_used = 1 if seqused is None else int(seqused[0])
    if x_cache_update:
        slot = int(slot)
        if slot >= 0:
            x_cache[slot // rows, slot % rows] = flat_tokens[raw_used - 1]
    active, logical_used, window_start, _ = derive_grouped_window(
        int(start_pos[0]), raw_used, cmp_ratio, cmp_ratio)
    if not active:
        return {"cmp_kv": None, "state_cache": np.array(state_cache, copy=True),
                "x_cache": x_cache, "active": False, "window_start": window_start}
    bank = int(slot) // rows
    # keep the window as hif8 codes; quant_compress_golden decodes exactly once.
    window_codes = x_cache[bank, window_start:window_start + logical_used]
    cmp_kv, _, _, out_state = quant_compress_golden(
        window_codes[None, :, :], wkv_codes, wgate_codes, x_descale, wkv_descale,
        wgate_descale, state_cache, ape, block_table, cmp_ratio, coff, cache_mode,
        start_pos=[window_start], seqused=[logical_used], grad_enabled=grad_enabled)
    return {"cmp_kv": cmp_kv, "state_cache": out_state, "x_cache": x_cache,
            "active": True, "window_start": window_start}


def ring_row_index(slot, x_cache_rows):
    """Physical flat row index for a slot mapping entry.

    Kernel addressing (compressor_block_cube.h / compressor_block_vec.h):
        bankBase   = slot // xCacheSize * xCacheSize
        cacheStart = (windowStart + curS) % xCacheSize
    xCacheSize is tiling groupedDecodeParams.xCacheSize == x_cache.shape[1].
    """
    return int(slot) // x_cache_rows * x_cache_rows


def derive_grouped_window(raw_start_pos, raw_seq_used, cmp_ratio, read_size):
    """Port of DeriveGroupedWindow (compressor_tools.h)."""
    if cmp_ratio == 0:
        return False, 0, raw_start_pos, 0
    group_offset = raw_start_pos % cmp_ratio
    active = raw_seq_used >= cmp_ratio - group_offset
    if not active:
        return False, 0, raw_start_pos, 0
    group_end = ((raw_start_pos + raw_seq_used) // cmp_ratio) * cmp_ratio
    window_start = group_end - read_size
    current = raw_start_pos - window_start if raw_start_pos > window_start else 0
    return True, read_size, window_start, current


def grouped_case_golden(x, wkv, wgate, state_cache, ape, block_table, x_cache,
                        x_cache_slot_mapping, cmp_ratio, coff, cache_mode=1,
                        start_pos=None, seqused=None, x_cache_update=True,
                        grad_enabled=False):
    """Full grouped-decode golden, ring update included.

    1. Apply the in-place ring write exactly as UpdateXCache does: for each raw
       token index t with slot >= 0, x_cache[slot // rows, slot % rows] = x[t].
    2. Derive the completed window via DeriveGroupedWindow.
    3. If a window completed, project it with the same contract by feeding the
       window rows as the compressor input (windowStart as start_pos).

    Returns dict(cmp_kv, softmax, kv, state_cache, x_cache, active, window_start).
    """
    x = np.asarray(x, dtype=np.float32)
    B, S, _ = x.shape
    x_cache = np.array(x_cache, dtype=np.float32, copy=True)
    rows = x_cache.shape[1]
    slot_map = np.asarray(x_cache_slot_mapping).reshape(-1)
    start_pos = [0] * B if start_pos is None else [int(v) for v in start_pos]
    seqused = None if seqused is None else [int(v) for v in seqused]

    # --- step 1: in-place ring refresh (UpdateXCache) ---
    if x_cache_update:
        for b in range(B):
            raw_used = S if seqused is None else seqused[b]
            for s in range(S):
                if s >= raw_used:
                    continue
                token_idx = b * S + s
                if token_idx >= slot_map.size:
                    continue
                slot = int(slot_map[token_idx])
                if slot < 0:
                    continue
                x_cache[slot // rows, slot % rows] = x[b, s]

    # --- step 2: window derivation ---
    active, logical_used, window_start, _ = derive_grouped_window(
        start_pos[0], S if seqused is None else seqused[0], cmp_ratio, cmp_ratio)

    if not active:
        return {
            "cmp_kv": None, "softmax": None, "kv": None,
            "state_cache": np.array(state_cache, dtype=np.float32, copy=True),
            "x_cache": x_cache, "active": False, "window_start": window_start,
        }

    bank_base = ring_row_index(slot_map[0], rows)
    bank = bank_base // rows
    # The ring is a window of `rows` physical rows; the kernel addresses it as
    # cacheStart = (windowStart + curS) % xCacheSize, so wrap the window read.
    row_idx = [(window_start + k) % rows for k in range(logical_used)]
    window = x_cache[bank, row_idx].copy()
    cmp_kv, softmax, kv, out_state = compress_golden(
        window[None, :, :], wkv, wgate, state_cache, ape, block_table, cmp_ratio,
        coff, cache_mode, start_pos=[window_start], seqused=[logical_used],
        grad_enabled=grad_enabled)
    return {
        "cmp_kv": cmp_kv, "softmax": softmax, "kv": kv,
        "state_cache": out_state, "x_cache": x_cache,
        "active": True, "window_start": window_start,
    }
