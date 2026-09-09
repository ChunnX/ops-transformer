"""Fused QSLI MXFP4 implementation for TND query and PA_BBND key cache.

Vector0 gathers the discrete candidate blocks on the two AIVs.  Each AIV owns
eight candidate blocks (64 tokens) of a 128-token Cube N tile.  The gathered
packed MXFP4 K and E8M0 KScale stay quantized.  The current DSL cannot lower an
E8M0 BDN transform on UB->L1, so same-core UB Channels drain the two halves to
an internal GM tile and the AIC reloads it with the validated GM->L1
``nd2nz_fp4`` / ``mx_scale_bdn`` engines.  This fallback remains inside the
single fused kernel and performs no dequantization.

The public axes are ``Q[T1,N1,D/2]``, ``K[PA,128,1,D/2]`` and
``candidate[T1,1,C]``.  Host code only validates dtype/shape/stride and
allocates the public outputs.  The fused kernel reads sequence metadata,
maps every TND row to its batch, pads Q/QScale, gathers K/KScale, runs
Cube+Vector+streaming-TopK, and writes the public row directly.

Q/K and E8M0 scales remain quantized throughout.  Q/K staging GM is an
intentional toolchain exception: this DSL version cannot lower the required
byte-built E8M0 UB->L1 formatted copy.  Each region is private to one
persistent worker and guarded by the Vector->Cube ready lifecycle; all
computational handoffs remain Channels.
"""

from __future__ import annotations

import threading
from typing import Optional

import cannbotdsl
import torch

from cannbotdsl import ChannelKind, MemLoc, Tensor, dtypes, select as dyn_select
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_intra_wait,
    vec_sync_all,
    vec_sync_intra_arrive,
)
from cannbotdsl.tensor import make_partition_tiler, rebase_view, tile_view

TILE_M = 128
TILE_N = 128
LOGICAL_D = 128
PACKED_D = LOGICAL_D // 2
N1 = 32
N2 = 1
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
CANDIDATE_BLOCKS_PER_TILE = TILE_N // CANDIDATE_BLOCK_SIZE
CANDIDATE_BLOCKS_PER_AIV = CANDIDATE_BLOCKS_PER_TILE // 2
STAGING_DEPTH = 3
# Cross-core sync ids 0..4 are reserved by the runtime/Channel lowering.
# Keep the explicit GM staging handshakes in the validated user range.
QUERY_READY_ID = 5
VECTOR0_READY_ID = 6
AIV1_SYNC_ID_OFFSET = 16
TOPK_TRUNK_LEN = 16384
MAX_CUBE_WORKERS = 32


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _validate_pa_dim0_stride(tensor: torch.Tensor, name: str) -> int:
    """Return PA axis-0 stride after enforcing ASC's inner-contiguous scope."""
    strides = tuple(int(value) for value in tensor.stride())
    expected = [1]
    for dimension in reversed(tuple(int(value) for value in tensor.shape[1:])):
        expected.append(expected[-1] * dimension)
    expected_inner = tuple(reversed(expected[:-1]))
    if strides[1:] != expected_inner:
        raise ValueError(
            f"{name} only supports non-contiguous storage on axis 0; "
            f"expected inner strides {expected_inner}, got {strides[1:]}"
        )
    if strides[0] <= 0:
        raise ValueError(f"{name} axis-0 stride must be positive")
    return strides[0]


class QsliRawTopKWorkspace:
    """UB scratch shared by sequential Sparse and Candidate TopK stages."""

    def __init__(self, max_trunk_len: int, max_topk: int):
        self.max_trunk_len = int(max_trunk_len)
        self.max_topk = int(max_topk)
        self.topk_pad = ceil_div(self.max_topk, 256) * 256
        self.merge_len_pad = ceil_div(
            self.topk_pad + self.max_trunk_len, 128
        ) * 128
        self.current_trunk = Channel(
            MemLoc.UB,
            (1, self.max_trunk_len),
            dtypes.uint16,
            depth=1,
        )
        self.merge_key = Buffer(
            MemLoc.UB, (1, self.merge_len_pad), dtypes.uint16
        )
        self.tmp_idx = Buffer(
            MemLoc.UB, (self.merge_len_pad,), dtypes.uint16
        )
        self.histogram = Buffer(MemLoc.UB, (256,), dtypes.uint16)
        self.high_target_carrier = Buffer(
            MemLoc.UB, (64,), dtypes.int32
        )
        self.history_key = Buffer(
            MemLoc.UB, (1, self.topk_pad), dtypes.uint16
        )
        self.next_history_key = Channel(
            MemLoc.UB,
            (1, self.topk_pad),
            dtypes.uint16,
            depth=1,
        )
        self.history_idx = Buffer(
            MemLoc.UB, (self.topk_pad,), dtypes.int32
        )
        self.next_idx = Buffer(
            MemLoc.UB, (self.topk_pad,), dtypes.int32
        )


class QsliRawTopKSelector:
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    def __init__(
        self,
        tokens: int,
        topk: int,
        workspace: QsliRawTopKWorkspace,
    ):
        self.tokens = int(tokens)
        self.topk = int(topk)
        if self.tokens <= 0 or self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive tokens/topk")
        self.trunk_len = min(TOPK_TRUNK_LEN, self.tokens)
        self.full_trunk_count = self.tokens // self.trunk_len
        self.tail_len = self.tokens % self.trunk_len
        # A row up to 16K is one pass.  For longer rows, complete 16K trunks
        # stay in one rolled loop and only the final partial trunk is emitted
        # as a separate fixed-shape path, matching ASC ProcessTopK.
        self.trunk_count = self.full_trunk_count + int(self.tail_len > 0)
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(
            self.topk_pad + self.trunk_len, 128
        ) * 128
        self.workspace = workspace
        if (
            self.trunk_len > workspace.max_trunk_len
            or self.topk > workspace.max_topk
            or self.merge_len_pad > workspace.merge_len_pad
        ):
            raise ValueError("TopK selector exceeds the shared UB workspace")
        # Final V->MTE3 stages are selector-owned so Sparse and Candidate can
        # share computational scratch without racing each other's GM writes.
        self.output_idx_stage = Channel(
            MemLoc.UB, (1, self.topk), dtypes.int32, depth=1
        )
        self.output_value_bits_stage = Channel(
            MemLoc.UB, (1, self.topk), dtypes.uint16, depth=1
        )

    @jit
    def _build_merge(
        self,
        current_trunk,
        history_key,
        merge_key,
        current_len,
        current_len_pad,
    ):
        """Assemble one history+trunk input entirely on the V pipe."""
        with vf(mode="raw"):
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            for chunk in dsl_range(ceil_div(self.topk_pad, 128), unroll=1):
                offset = chunk * 128
                rr.vstore(
                    merge_key,
                    offset,
                    rr.vload(history_key, offset),
                    mask16,
                )
            for chunk in dsl_range(ceil_div(current_len, 128), unroll=1):
                source_offset = chunk * 128
                count = min(128, current_len - source_offset)
                current_mask = rr.update_mask(count, elem_bits=16)[0]
                rr.vstore(
                    merge_key,
                    self.topk_pad + source_offset,
                    rr.vload(current_trunk, source_offset),
                    current_mask,
                )
            if current_len_pad > current_len:
                pad_count = current_len_pad - current_len
                pad_mask = rr.update_mask(pad_count, elem_bits=16)[0]
                rr.vstore(
                    merge_key,
                    self.topk_pad + current_len,
                    rr.vdups(0, dtypes.uint16, mask=pad_mask),
                    pad_mask,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _select_pass(
        self,
        sort_key,
        valid_len,
        history_prefix,
        current_global_base,
        tmp_idx,
        histogram,
        high_target_carrier,
        history_idx_in,
        history_idx_out,
        next_history_key,
    ):
        """Run one ASC radix pass over one trunk or history+trunk."""
        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            h0 = rr.vdups(0, dtypes.uint16, mask=full16)
            h1 = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(ceil_div(valid_len, 128), unroll=1):
                valid_count = min(128, valid_len - chunk * 128)
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                _, high = rr.vload_deinterleave(
                    sort_key, chunk * 128, width="b8"
                )
                h0 = rr.vhistogram_accumulate(h0, high, mask=mask8, bin=0)
                h1 = rr.vhistogram_accumulate(h1, high, mask=mask8, bin=1)
            rr.vstore(histogram, 0, h0, full16)
            rr.vstore(histogram, 128, h1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            bottom_k = valid_len - self.topk + 1
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist0, bottom_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vges(hist1, bottom_k, mask=full16),
            )
            high_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(
                high_target_carrier,
                0,
                rr.vreinterpret(
                    rr.vunpack(high_target, dtypes.uint32, part="lower"),
                    dtypes.int32,
                ),
                rr.update_mask(1, elem_bits=32)[0],
            )
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=full16)
            one16 = rr.vdups(1, dtypes.uint16, mask=full16)
            bottom_k = valid_len - self.topk + 1
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            high_target8 = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b8"),
                dtypes.uint8,
            )
            prev = rr.vsub(high_target, one16, mask=full16)
            is_zero = rr.veqs(high_target, 0, mask=full16)
            prev = rr.vselect(zero16, prev, cond_mask=is_zero)
            prev_count = zero16
            for base in range_constexpr(0, 256, 128):
                hv = rr.vload(histogram, base)
                idx = rr.varange(base, dtypes.uint16)
                at_prev = rr.veq(idx, prev, mask=full16)
                picked = rr.vselect(hv, zero16, cond_mask=at_prev)
                prev_count = rr.vadd(
                    prev_count,
                    rr.vreduce_max(picked, mask=full16),
                    mask=full16,
                )
            prev_count = rr.vselect(zero16, prev_count, cond_mask=is_zero)
            next_k = rr.vsub(
                rr.vdups(bottom_k, dtypes.uint16, mask=full16),
                prev_count,
                mask=full16,
            )
            rr.vstore(tmp_idx, 0, next_k, lane0)

            l0 = rr.vdups(0, dtypes.uint16, mask=full16)
            l1 = rr.vdups(0, dtypes.uint16, mask=full16)
            for chunk in dsl_range(ceil_div(valid_len, 128), unroll=1):
                valid_count = min(128, valid_len - chunk * 128)
                mask8 = rr.update_mask(valid_count, elem_bits=8)[0]
                low, high = rr.vload_deinterleave(
                    sort_key, chunk * 128, width="b8"
                )
                eq_high = rr.veq(high, high_target8, mask=mask8)
                l0 = rr.vhistogram_accumulate(l0, low, mask=eq_high, bin=0)
                l1 = rr.vhistogram_accumulate(l1, low, mask=eq_high, bin=1)
            rr.vstore(histogram, 0, l0, full16)
            rr.vstore(histogram, 128, l1, full16)
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            lane0 = rr.update_mask(1, elem_bits=16)[0]
            next_k = rr.vload_broadcast(tmp_idx, 0)
            all_ones = rr.vdups(0xFFFF, dtypes.uint16, mask=full16)
            hist0 = rr.vload(histogram, 0)
            hist1 = rr.vload(histogram, 128)
            candidate0 = rr.vselect(
                rr.varange(0, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist0, next_k, mask=full16),
            )
            candidate1 = rr.vselect(
                rr.varange(128, dtypes.uint16),
                all_ones,
                cond_mask=rr.vge(hist1, next_k, mask=full16),
            )
            low_target = rr.vmin(
                rr.vreduce_min(candidate0, mask=full16),
                rr.vreduce_min(candidate1, mask=full16),
                mask=lane0,
            )
            rr.vstore(tmp_idx, 0, low_target, lane0)
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            full16 = rr.update_mask(128, elem_bits=16)[0]
            high_target = rr.vreinterpret_lanes(
                rr.vload_broadcast(high_target_carrier, 0, width="b16"),
                dtypes.uint16,
            )
            low_target = rr.vload_broadcast(tmp_idx, 0)
            kth = rr.vbitwise_or(
                rr.vshl(high_target, 8, mask=full16),
                low_target,
                mask=full16,
            )
            ureg = rr.vstore_unalign_begin(tmp_idx)
            for chunk in dsl_range(ceil_div(valid_len, 128), unroll=1):
                valid_count = min(128, valid_len - chunk * 128)
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(ceil_div(valid_len, 128), unroll=1):
                valid_count = min(128, valid_len - chunk * 128)
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.veq(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            rr.vstore_unalign_post(tmp_idx, 0, ureg)
            rr.vmem_bar("vst_vld")

        with vf(mode="raw"):
            # ASC FindValueOutputVFImpl keeps the compacted positions as
            # uint16 and gathers 128 values per iteration.  Do not reuse the
            # uint16->uint32 unpack used by FindRealIndexVFImpl below: that
            # unpack exposes only 64 lanes and breaks value/index pairing
            # once history is merged with another trunk.
            for chunk in dsl_range(ceil_div(self.topk, 128), unroll=1):
                count = min(128, self.topk - chunk * 128)
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                pos16 = rr.vload(tmp_idx, chunk * 128)
                selected_key = rr.vgather(
                    sort_key, pos16, mask=mask16
                )
                rr.vstore(
                    next_history_key,
                    chunk * 128,
                    selected_key,
                    mask16,
                )

            # ASC FindRealIndexVFImpl loads the same compacted positions with
            # DIST_UNPACK_B16, yielding 64 uint32 lanes for history/current
            # index mapping.  Keep this separate from the value gather above.
            full32 = rr.update_mask(64, elem_bits=32)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = min(64, self.topk - chunk * 64)
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                pos16 = rr.vload(tmp_idx, chunk * 64)
                pos32 = rr.vunpack(pos16, dtypes.uint32, part="lower")
                is_current = rr.vges(
                    pos32, history_prefix, mask=mask32
                )
                is_history = rr.vlts(
                    pos32, history_prefix, mask=mask32
                )
                old_idx = rr.vgather(
                    history_idx_in, pos32, mask=is_history
                )
                current_idx = rr.vadds(
                    rr.vreinterpret(pos32, dtypes.int32),
                    current_global_base - history_prefix,
                    mask=mask32,
                )
                global_idx = rr.vselect(
                    current_idx, old_idx, cond_mask=is_current
                )
                rr.vstore(history_idx_out, chunk * 64, global_idx, mask32)
            rr.vmem_bar("vst_vld")

    @jit
    def _commit_history(
        self, next_idx, history_idx, next_history_key, history_key
    ):
        with vf(mode="raw"):
            copy_mask32 = rr.update_mask(64, elem_bits=32)[0]
            copy_mask16 = rr.update_mask(64, elem_bits=16)[0]
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                rr.vstore(
                    history_idx,
                    chunk * 64,
                    rr.vload(next_idx, chunk * 64),
                    copy_mask32,
                )
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                rr.vstore(
                    history_key,
                    chunk * 64,
                    rr.vload(next_history_key, chunk * 64),
                    copy_mask16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        output_valid_count,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
    ):
        merge_key = self.workspace.merge_key
        current_trunk = self.workspace.current_trunk
        tmp_idx = self.workspace.tmp_idx
        histogram = self.workspace.histogram
        high_target_carrier = self.workspace.high_target_carrier
        history_key = self.workspace.history_key
        next_history_key = self.workspace.next_history_key
        history_idx = self.workspace.history_idx
        next_idx = self.workspace.next_idx
        output_idx_stage = self.output_idx_stage
        out_value_bits = self.output_value_bits_stage

        # The all-zero prefix is a strict low sentinel because VEC1 keys are
        # generated after ReLU. It makes pass zero and later history passes use
        # one rolled loop without changing the selected TopK set.
        with vf(mode="raw"):
            init_mask16 = rr.update_mask(128, elem_bits=16)[0]
            init_mask32 = rr.update_mask(64, elem_bits=32)[0]
            zero16 = rr.vdups(0, dtypes.uint16, mask=init_mask16)
            zero32 = rr.vdups(0, dtypes.int32, mask=init_mask32)
            for chunk in dsl_range(ceil_div(self.topk_pad, 128), unroll=1):
                rr.vstore(history_key, chunk * 128, zero16, init_mask16)
            for chunk in dsl_range(ceil_div(self.topk_pad, 64), unroll=1):
                rr.vstore(history_idx, chunk * 64, zero32, init_mask32)
            rr.vmem_bar("vst_vld")

        for pass_idx in dsl_range(self.full_trunk_count, unroll=1):
            trunk_offset = pass_idx * self.trunk_len
            mem_copy(
                tile_view(
                    rebase_view(current_trunk, (0, 0)),
                    (1, self.trunk_len),
                    (0, 0),
                ),
                tile_view(
                    rebase_view(gm_sort_key, (0, trunk_offset)),
                    (1, self.trunk_len),
                    (0, 0),
                ),
            )
            self._build_merge(
                current_trunk,
                history_key,
                merge_key,
                self.trunk_len,
                self.trunk_len,
            )
            self._select_pass(
                merge_key,
                self.topk_pad + self.trunk_len,
                self.topk_pad,
                trunk_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )
            self._commit_history(
                next_idx, history_idx, next_history_key, history_key
            )
            vec_sync_all()

        if self.tail_len > 0:
            tail_offset = self.full_trunk_count * self.trunk_len
            tail_len_pad = ceil_div(self.tail_len, 256) * 256
            mem_copy(
                tile_view(
                    rebase_view(current_trunk, (0, 0)),
                    (1, self.tail_len),
                    (0, 0),
                ),
                tile_view(
                    rebase_view(gm_sort_key, (0, tail_offset)),
                    (1, self.tail_len),
                    (0, 0),
                ),
            )
            self._build_merge(
                current_trunk,
                history_key,
                merge_key,
                self.tail_len,
                tail_len_pad,
            )
            self._select_pass(
                merge_key,
                self.topk_pad + tail_len_pad,
                self.topk_pad,
                tail_offset,
                tmp_idx,
                histogram,
                high_target_carrier,
                history_idx,
                next_idx,
                next_history_key,
            )
            self._commit_history(
                next_idx, history_idx, next_history_key, history_key
            )
            vec_sync_all()

        final_idx = history_idx
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = min(64, self.topk - chunk * 64)
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                indices = rr.vload(final_idx, chunk * 64)
                keys = rr.vload(history_key, chunk * 64)
                positive = rr.veq(
                    rr.vbitwise_and(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                value_bits = rr.vselect(
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    rr.vbitwise_xor(
                        keys,
                        rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                        mask=mask16,
                    ),
                    cond_mask=positive,
                )
                valid_rank32 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    output_valid_count,
                    mask=mask32,
                )
                valid_rank16 = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint16),
                    output_valid_count,
                    mask=mask16,
                )
                indices = rr.vselect(
                    indices,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    cond_mask=valid_rank32,
                )
                value_bits = rr.vselect(
                    value_bits,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    cond_mask=valid_rank16,
                )
                rr.vstore(output_idx_stage, chunk * 64, indices, mask32)
                rr.vstore(out_value_bits, chunk * 64, value_bits, mask16)
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, output_idx_stage)
        mem_copy(gm_selected_value_bits, out_value_bits)


class QsliCube:
    """AIC-owned native MXFP4 QK resources."""

    def __init__(self):
        self.q_l1 = Channel(
            MemLoc.L1, (TILE_M, LOGICAL_D), dtypes.fp4x2_e2m1,
            depth=2, data_format="nz",
        )
        self.k_l1 = Channel(
            MemLoc.L1, (TILE_N, LOGICAL_D), dtypes.fp4x2_e2m1,
            depth=2, data_format="nz",
        )
        self.q_scale_l1 = Channel(
            MemLoc.L1, (TILE_M, 4), dtypes.float8_e8m0,
            depth=2, data_format="zn",
        )
        self.k_scale_l1 = Channel(
            MemLoc.L1, (4, TILE_N), dtypes.float8_e8m0,
            depth=2, data_format="nz",
        )
        self.l0a = Channel(
            MemLoc.L0A, (TILE_M, LOGICAL_D), dtypes.fp4x2_e2m1, depth=2
        )
        self.l0b = Channel(
            MemLoc.L0B, (TILE_N, LOGICAL_D), dtypes.fp4x2_e2m1,
            depth=2, data_format="nz",
        )
        self.l0c = Channel(
            MemLoc.L0C, (TILE_M, TILE_N), dtypes.float32, depth=2
        )
        self.nd2nz_fp4 = make_copy_engine(
            format_transform="nd2nz", dtype=dtypes.fp4x2_e2m1,
            pad_value=0.0,
        )
        self.scale_a = make_copy_engine(
            format_transform="mx_scale_and", dtype=dtypes.float8_e8m0,
            pad_value=0.0,
        )
        self.scale_b = make_copy_engine(
            format_transform="mx_scale_bdn", dtype=dtypes.float8_e8m0,
            pad_value=0.0,
        )
        self.fixpipe_split_m = make_copy_engine(
            dtype=dtypes.float32, dual_dst_ctl=1, unit_flag_mode=3
        )

    def load_query(self, query, query_scale):
        mem_copy(self.q_l1, query, engine=self.nd2nz_fp4)
        mem_copy(self.q_scale_l1, query_scale, engine=self.scale_a)
        mem_copy(self.l0a, self.q_l1, mx_scale=self.q_scale_l1)

    def compute_qk(self, key_tile, key_scale_tile, qk_handoff):
        mem_copy(self.k_l1, key_tile, engine=self.nd2nz_fp4)
        mem_copy(self.k_scale_l1, key_scale_tile, engine=self.scale_b)
        mem_copy(self.l0b, self.k_l1, mx_scale=self.k_scale_l1)
        dsl_matmul(self.l0c, self.l0a, self.l0b, init=True, unit_flag=3)
        split_m = make_partition_tiler(
            (TILE_M, TILE_N), (TILE_M, TILE_N)
        )
        mem_copy(
            qk_handoff,
            self.l0c,
            engine=self.fixpipe_split_m,
            partition=split_m,
        )


class QsliVector0:
    """Prepare Q/QScale and gather sparse K/KScale on the two AIVs."""

    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.query_input_ub = Channel(
            MemLoc.UB, (N1, PACKED_D), dtypes.uint8, depth=1
        )
        self.query_scale_input_ub = Channel(
            MemLoc.UB, (N1, 4), dtypes.uint8, depth=1
        )
        self.query_stage_ub = Channel(
            MemLoc.UB, (TILE_M // 2, PACKED_D), dtypes.uint8, depth=1
        )
        self.query_scale_stage_ub = Channel(
            MemLoc.UB, (TILE_M // 2, 4), dtypes.uint8, depth=1
        )
        self.key_ub = Channel(
            MemLoc.UB, (TILE_N // 2, PACKED_D), dtypes.uint8,
            depth=2,
        )
        self.scale_ub = Channel(
            MemLoc.UB, (TILE_N // 2, 4), dtypes.uint8,
            depth=2,
        )

    @jit
    def stage_query(
        self,
        query,
        query_scale,
        query_row,
        staging_query,
        staging_query_scale,
        worker_idx,
    ):
        """Build one 64-row half of the Cube M tile on each AIV."""
        query_start = dtypes.int64(query_row) * N1
        mem_copy(
            self.query_input_ub,
            tile_view(
                rebase_view(query, (query_start, 0)),
                (N1, PACKED_D),
                (0, 0),
            ),
        )
        mem_copy(
            self.query_scale_input_ub,
            tile_view(
                rebase_view(query_scale, (query_start, 0)),
                (N1, 4),
                (0, 0),
            ),
        )
        with vf(mode="raw"):
            full8 = rr.update_mask(128, elem_bits=8)[0]
            query_chunks = N1 * PACKED_D // 128
            for chunk in range_constexpr(query_chunks):
                offset = chunk * 128
                rr.vstore(
                    self.query_stage_ub,
                    offset,
                    rr.vload(self.query_input_ub, offset),
                    full8,
                )
            padding_chunks = (TILE_M // 2 - N1) * PACKED_D // 128
            for chunk in range_constexpr(padding_chunks):
                rr.vstore(
                    self.query_stage_ub,
                    N1 * PACKED_D + chunk * 128,
                    rr.vdups(0, dtypes.uint8, mask=full8),
                    full8,
                )

            rr.vstore(
                self.query_scale_stage_ub,
                0,
                rr.vload(self.query_scale_input_ub, 0),
                full8,
            )
            rr.vstore(
                self.query_scale_stage_ub,
                N1 * 4,
                rr.vdups(127, dtypes.uint8, mask=full8),
                full8,
            )
            rr.vmem_bar("vst_vld")

        staging_row = (
            dtypes.int64(worker_idx) * TILE_M
            + dtypes.int64(self.subblock_idx) * (TILE_M // 2)
        )
        mem_copy(
            tile_view(
                rebase_view(staging_query, (staging_row, 0)),
                (TILE_M // 2, PACKED_D),
                (0, 0),
            ),
            self.query_stage_ub,
        )
        mem_copy(
            tile_view(
                rebase_view(staging_query_scale, (staging_row, 0)),
                (TILE_M // 2, 4),
                (0, 0),
            ),
            self.query_scale_stage_ub,
        )
        vec_sync_intra_arrive(PIPE.MTE3, QUERY_READY_ID)

    @jit
    def wait_query_ready(self):
        cube_sync_intra_wait(PIPE.MTE2, QUERY_READY_ID)
        cube_sync_intra_wait(
            PIPE.MTE2, QUERY_READY_ID + AIV1_SYNC_ID_OFFSET
        )

    @jit
    def _gather_half(
        self,
        half,
        key,
        key_scale,
        candidate_block_indices,
        candidate_block_length,
        block_table,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_block_table,
    ):
        key_slot = self.key_ub.acquire()
        scale_slot = self.scale_ub.acquire()
        first_candidate = (
            dtypes.int64(candidate_tile) * CANDIDATE_BLOCKS_PER_TILE
            + dtypes.int64(half) * CANDIDATE_BLOCKS_PER_AIV
        )
        valid_blocks = dtypes.int64(candidate_block_length)
        for local_block in range(CANDIDATE_BLOCKS_PER_AIV):
            candidate_pos = first_candidate + dtypes.int64(local_block)
            safe_pos = dtypes.int64(
                dyn_select(candidate_pos < valid_blocks, candidate_pos, dtypes.int64(0))
            )
            candidate_block = dtypes.int64(
                candidate_block_indices[task_query_row, 0, safe_pos]
            )
            logical_row = candidate_block * CANDIDATE_BLOCK_SIZE
            logical_page = logical_row // TILE_N
            page_offset = logical_row % TILE_N
            physical_page = logical_page
            if has_block_table:
                physical_page = dtypes.int64(
                    block_table[batch_idx, logical_page]
                )
            mem_copy(
                tile_view(
                    key_slot,
                    (CANDIDATE_BLOCK_SIZE, PACKED_D),
                    (local_block, 0),
                ),
                tile_view(
                    rebase_view(key, (physical_page, page_offset, 0)),
                    (CANDIDATE_BLOCK_SIZE, PACKED_D),
                    (0, 0),
                ),
            )
            mem_copy(
                tile_view(
                    scale_slot,
                    (CANDIDATE_BLOCK_SIZE, 4),
                    (local_block, 0),
                ),
                tile_view(
                    rebase_view(key_scale, (physical_page, page_offset, 0)),
                    (CANDIDATE_BLOCK_SIZE, 4),
                    (0, 0),
                ),
            )
        self.key_ub.commit(key_slot)
        self.scale_ub.commit(scale_slot)

        key_ready = self.key_ub.wait()
        scale_ready = self.scale_ub.wait()
        staging_row = (
            (
                dtypes.int64(task_idx) * STAGING_DEPTH
                + dtypes.int64(candidate_tile) % STAGING_DEPTH
            ) * TILE_N
            + dtypes.int64(half) * (TILE_N // 2)
        )
        mem_copy(
            tile_view(
                rebase_view(staging_key, (staging_row, 0)),
                (TILE_N // 2, PACKED_D),
                (0, 0),
            ),
            key_ready,
        )
        mem_copy(
            tile_view(
                rebase_view(staging_scale, (staging_row, 0)),
                (TILE_N // 2, 4),
                (0, 0),
            ),
            scale_ready,
        )
        self.key_ub.release(key_ready)
        self.scale_ub.release(scale_ready)
        vec_sync_intra_arrive(PIPE.MTE3, VECTOR0_READY_ID)

    @jit
    def gather(
        self,
        key,
        key_scale,
        candidate_block_indices,
        candidate_block_length,
        block_table,
        task_query_row,
        batch_idx,
        candidate_tile,
        staging_key,
        staging_scale,
        task_idx,
        has_block_table,
    ):
        self._gather_half(
            self.subblock_idx,
            key,
            key_scale,
            candidate_block_indices,
            candidate_block_length,
            block_table,
            task_query_row,
            batch_idx,
            candidate_tile,
            staging_key,
            staging_scale,
            task_idx,
            has_block_table,
        )

    @jit
    def wait_ready(self):
        cube_sync_intra_wait(PIPE.MTE2, VECTOR0_READY_ID)
        cube_sync_intra_wait(
            PIPE.MTE2, VECTOR0_READY_ID + AIV1_SYNC_ID_OFFSET
        )


class QsliVector1:
    """Compute the one-query QSLI score row and its logical token payload."""

    def __init__(self, tokens, mask_mode):
        self.tokens = int(tokens)
        self.mask_mode = int(mask_mode)
        self.weight_ub = Channel(MemLoc.UB, (1, N1), dtypes.float32, depth=1)
        # Candidate metadata is consumed repeatedly inside the same VF.  Keep
        # it as private UB state; Channels are reserved for stage hand-offs.
        self.candidate_ub = Buffer(
            MemLoc.UB, (CANDIDATE_BLOCKS_PER_TILE,), dtypes.int32
        )
        self.key_ub = Channel(
            MemLoc.UB, (1, TILE_N), dtypes.uint16, depth=1
        )
        # The payload is constructed and reread inside one VF before MTE3
        # drains it.  Pin it as private UB state to prevent Channel storage
        # reuse while the score/key stores are still in flight.
        self.logical_index_ub = Buffer(
            MemLoc.UB, (1, TILE_N), dtypes.int32
        )

    @jit
    def compute(
        self,
        qk_handoff,
        weights,
        candidate_block_indices,
        key_out,
        workspace_row,
        task_query_row,
        candidate_tile,
        candidate_block_length,
        valid_s2,
        subblock_idx,
    ):
        mem_copy(
            self.weight_ub,
            tile_view(weights, (1, N1), (task_query_row, 0)),
        )
        candidate_start = dtypes.int64(candidate_tile) * CANDIDATE_BLOCKS_PER_TILE
        mem_copy(
            self.candidate_ub,
            tile_view(
                rebase_view(
                    candidate_block_indices,
                    (task_query_row, 0, candidate_start),
                ),
                (CANDIDATE_BLOCKS_PER_TILE,),
                (0,),
            ),
        )
        with vf(mode="raw"):
            full32 = rr.full_mask()
            mask16 = rr.update_mask(TILE_N, elem_bits=16)[0]
            block_mask = rr.update_mask(
                CANDIDATE_BLOCK_SIZE, elem_bits=32
            )[0]
            block_lane = rr.varange(0, dtypes.int32)
            for block in range(CANDIDATE_BLOCKS_PER_TILE):
                block_base = rr.vmuls(
                    rr.vload_broadcast(self.candidate_ub, block),
                    CANDIDATE_BLOCK_SIZE,
                    mask=full32,
                )
                rr.vstore(
                    self.logical_index_ub,
                    block * CANDIDATE_BLOCK_SIZE,
                    rr.vadd(block_base, block_lane, mask=full32),
                    block_mask,
                )
            rr.vmem_bar("vst_vld")
            for off in (0, 64):
                acc = rr.vdups(0.0, dtypes.float32, mask=full32)
                for pair in range(N1 // 2):
                    for group in (pair * 2, pair * 2 + 1):
                        qk = rr.vload(qk_handoff, group * TILE_N + off)
                        qk = rr.vmaxs(qk, 0.0, mask=full32)

                        weight = rr.vload_broadcast(self.weight_ub, group)
                        weight_bits = rr.vreinterpret(weight, dtypes.uint32)
                        weight_lsb = rr.vbitwise_and(
                            rr.vshr(weight_bits, 16, mask=full32),
                            rr.vdups(1, dtypes.uint32, mask=full32),
                            mask=full32,
                        )
                        weight_bits = rr.vbitwise_and(
                            rr.vadd(
                                weight_bits,
                                rr.vadd(
                                    rr.vdups(0x7FFF, dtypes.uint32, mask=full32),
                                    weight_lsb,
                                    mask=full32,
                                ),
                                mask=full32,
                            ),
                            rr.vdups(0xFFFF0000, dtypes.uint32, mask=full32),
                            mask=full32,
                        )
                        weight = rr.vreinterpret(weight_bits, dtypes.float32)

                        qk_bits = rr.vreinterpret(qk, dtypes.uint32)
                        qk_lsb = rr.vbitwise_and(
                            rr.vshr(qk_bits, 16, mask=full32),
                            rr.vdups(1, dtypes.uint32, mask=full32),
                            mask=full32,
                        )
                        qk_bits = rr.vbitwise_and(
                            rr.vadd(
                                qk_bits,
                                rr.vadd(
                                    rr.vdups(0x7FFF, dtypes.uint32, mask=full32),
                                    qk_lsb,
                                    mask=full32,
                                ),
                                mask=full32,
                            ),
                            rr.vdups(0xFFFF0000, dtypes.uint32, mask=full32),
                            mask=full32,
                        )
                        qk = rr.vreinterpret(qk_bits, dtypes.float32)
                        acc = rr.vmadd(qk, weight, acc, mask=full32)

                        acc_bits = rr.vreinterpret(acc, dtypes.uint32)
                        acc_lsb = rr.vbitwise_and(
                            rr.vshr(acc_bits, 16, mask=full32),
                            rr.vdups(1, dtypes.uint32, mask=full32),
                            mask=full32,
                        )
                        acc_bits = rr.vbitwise_and(
                            rr.vadd(
                                acc_bits,
                                rr.vadd(
                                    rr.vdups(0x7FFF, dtypes.uint32, mask=full32),
                                    acc_lsb,
                                    mask=full32,
                                ),
                                mask=full32,
                            ),
                            rr.vdups(0xFFFF0000, dtypes.uint32, mask=full32),
                            mask=full32,
                        )
                        acc = rr.vreinterpret(acc_bits, dtypes.float32)
                    acc = rr.vadd(
                        acc, rr.vdups(0.0, dtypes.float32, mask=full32),
                        mask=full32,
                    )

                lane = rr.varange(off, dtypes.uint32)
                logical_index = rr.vload(self.logical_index_ub, off)
                compact_index = rr.vreinterpret(
                    rr.vadds(
                        lane,
                        dtypes.int64(candidate_tile) * TILE_N,
                        mask=full32,
                    ),
                    dtypes.int32,
                )
                candidate_valid = rr.vlts(
                    compact_index,
                    dtypes.int32(candidate_block_length) * CANDIDATE_BLOCK_SIZE,
                    mask=full32,
                )
                one = rr.vdups(1, dtypes.uint32, mask=full32)
                zero = rr.vdups(0, dtypes.uint32, mask=full32)
                valid_lanes = rr.vselect(
                    one, zero, cond_mask=candidate_valid
                )
                # ``candidate_block_length`` counts complete 8-token blocks,
                # while ``seqused_k`` may end inside the last block.  Apply
                # the token-level S2 boundary in every mask mode.  In causal
                # mode ``valid_s2`` has already been tightened to the causal
                # boundary; otherwise it is the direct seqused_k value.
                sequence_valid = rr.vlts(
                    logical_index,
                    valid_s2,
                    mask=full32,
                )
                valid_lanes = rr.vselect(
                    valid_lanes, zero, cond_mask=sequence_valid
                )
                valid = rr.veq(valid_lanes, one, mask=full32)
                acc = rr.vselect(
                    acc,
                    rr.vreinterpret(
                        rr.vdups(0xFF800000, dtypes.uint32, mask=full32),
                        dtypes.float32,
                    ),
                    cond_mask=valid,
                )
                rr.vstore(self.logical_index_ub, off, logical_index, full32)

                rounded = rr.vreinterpret(acc, dtypes.uint32)
                half = rr.vshr(rounded, 16, mask=full32)
                lsb = rr.vbitwise_and(
                    half, rr.vdups(1, dtypes.uint32, mask=full32),
                    mask=full32,
                )
                rounded = rr.vadd(
                    rounded,
                    rr.vadd(
                        rr.vdups(0x7FFF, dtypes.uint32, mask=full32),
                        lsb,
                        mask=full32,
                    ),
                    mask=full32,
                )
                bits = rr.vpack(
                    rr.vshr(rounded, 16, mask=full32),
                    dtypes.uint16,
                    part="lower",
                )
                sign = rr.vbitwise_and(
                    bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                negative = rr.veq(
                    sign, rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                positive_key = rr.vbitwise_xor(
                    bits, rr.vdups(0x8000, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                negative_key = rr.vbitwise_xor(
                    bits, rr.vdups(0xFFFF, dtypes.uint16, mask=mask16),
                    mask=mask16,
                )
                rr.vstore(
                    self.key_ub,
                    off,
                    rr.vselect(negative_key, positive_key, cond_mask=negative),
                    mask16,
                )

            rr.vmem_bar("vst_vld")

        if subblock_idx == dtypes.int64(0):
            token_offset = dtypes.int64(candidate_tile) * TILE_N
            mem_copy(
                tile_view(
                    rebase_view(key_out, (workspace_row, token_offset)),
                    (1, TILE_N),
                    (0, 0),
                ),
                self.key_ub,
            )

class QsliPositionMapper:
    """Map compact TopK positions back to sparse logical token indices."""

    def __init__(self, topk, candidate_capacity):
        self.topk = int(topk)
        self.candidate_capacity = int(candidate_capacity)
        self.position_ub = Channel(
            MemLoc.UB, (1, self.topk), dtypes.int32, depth=1
        )
        self.candidate_ub = Channel(
            MemLoc.UB, (self.candidate_capacity,), dtypes.int32, depth=1
        )
        self.output_ub = Channel(
            MemLoc.UB, (1, self.topk), dtypes.int32, depth=1
        )

    @jit
    def apply(
        self,
        selected_positions,
        candidate_block_indices,
        output_indices,
        query_row,
        valid_count,
        output_idx_offset,
    ):
        output_slot = self.output_ub.acquire()
        mem_copy(self.position_ub, selected_positions)
        mem_copy(
            self.candidate_ub,
            tile_view(
                rebase_view(candidate_block_indices, (query_row, 0, 0)),
                (self.candidate_capacity,),
                (0,),
            ),
        )
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = min(64, self.topk - chunk * 64)
                mask = rr.update_mask(count, elem_bits=32)[0]
                position = rr.vreinterpret(
                    rr.vload(self.position_ub, chunk * 64), dtypes.uint32
                )
                valid = rr.vlts(
                    rr.varange(chunk * 64, dtypes.uint32),
                    valid_count,
                    mask=mask,
                )
                # Raw TopK emits -1 positions after valid_count.  Sanitize
                # those lanes before vgather; masking only the final store is
                # too late because uint32(-1) would address beyond candidate_ub.
                position = rr.vselect(
                    position,
                    rr.vdups(0, dtypes.uint32, mask=mask),
                    cond_mask=valid,
                )
                block_position = rr.vshr(position, 3, mask=mask)
                block_id = rr.vgather(
                    self.candidate_ub, block_position, mask=mask
                )
                logical_index = rr.vadd(
                    rr.vmuls(block_id, CANDIDATE_BLOCK_SIZE, mask=mask),
                    rr.vreinterpret(
                        rr.vbitwise_and(
                            position,
                            rr.vdups(
                                CANDIDATE_BLOCK_SIZE - 1,
                                dtypes.uint32,
                                mask=mask,
                            ),
                            mask=mask,
                        ),
                        dtypes.int32,
                    ),
                    mask=mask,
                )
                logical_index = rr.vadds(
                    logical_index, output_idx_offset, mask=mask
                )
                rr.vstore(
                    output_slot,
                    chunk * 64,
                    rr.vselect(
                        logical_index,
                        rr.vdups(-1, dtypes.int32, mask=mask),
                        cond_mask=valid,
                    ),
                    mask,
                )
            rr.vmem_bar("vst_vld")
        self.output_ub.commit(output_slot)
        output_ready = self.output_ub.wait()
        mem_copy(output_indices, output_ready)
        self.output_ub.release(output_ready)


class QsliTiling:
    """Dynamic worker split plus dynamic TND row-to-batch mapping."""

    def __init__(self, has_cu, has_seqused_q):
        self.has_cu = bool(has_cu)
        self.has_seqused_q = bool(has_seqused_q)

    @jit
    def map_query(
        self, query_row, query_rows, batch_count, cu_seqlens_q, seqused_q
    ):
        batch_idx = dtypes.int64(0)
        query_begin = dtypes.int32(0)
        query_end = dtypes.int32(query_rows)
        if self.has_cu:
            for batch in dsl_range(batch_count, unroll=1):
                begin = dtypes.int32(cu_seqlens_q[batch])
                end = dtypes.int32(cu_seqlens_q[batch + 1])
                belongs_to_or_after = dtypes.int32(query_row) >= begin
                batch_idx = dtypes.int64(
                    dyn_select(
                        belongs_to_or_after,
                        dtypes.int64(batch),
                        batch_idx,
                    )
                )
                query_begin = dtypes.int32(
                    dyn_select(belongs_to_or_after, begin, query_begin)
                )
                query_end = dtypes.int32(
                    dyn_select(belongs_to_or_after, end, query_end)
                )
        query_offset = dtypes.int32(query_row) - query_begin
        used_query = query_end - query_begin
        if self.has_seqused_q:
            used_query = dtypes.int32(seqused_q[batch_idx])
        return batch_idx, query_offset, used_query


class QsliOutputWriter:
    """Initialize public rows inside the fused kernel."""

    def __init__(self, public_topk):
        self.public_topk = int(public_topk)
        self.index_ub = Channel(
            MemLoc.UB, (1, self.public_topk), dtypes.int32, depth=1
        )
        self.value_ub = Channel(
            MemLoc.UB, (1, self.public_topk), dtypes.uint16, depth=1
        )

    @jit
    def initialize(self, sparse_indices, sparse_value_bits, query_row):
        index_slot = self.index_ub.acquire()
        value_slot = self.value_ub.acquire()
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.public_topk, 64), unroll=1):
                count = min(64, self.public_topk - chunk * 64)
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                rr.vstore(
                    index_slot,
                    chunk * 64,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    mask32,
                )
                rr.vstore(
                    value_slot,
                    chunk * 64,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")
        self.index_ub.commit(index_slot)
        self.value_ub.commit(value_slot)
        index_ready = self.index_ub.wait()
        value_ready = self.value_ub.wait()
        mem_copy(
            tile_view(sparse_indices, (1, self.public_topk), (query_row, 0)),
            index_ready,
        )
        mem_copy(
            tile_view(sparse_value_bits, (1, self.public_topk), (query_row, 0)),
            value_ready,
        )
        self.index_ub.release(index_ready)
        self.value_ub.release(value_ready)


@kernel
class QsliFusedKernel:
    def __init__(
        self,
        tokens,
        fused_topk,
        public_topk,
        candidate_capacity,
        mask_mode,
        cmp_ratio,
        has_cu,
        has_seqused_q,
        has_seqused_k,
        has_residual,
        has_block_table,
        has_output_offset,
    ):
        self.tokens = int(tokens)
        self.token_tiles = self.tokens // TILE_N
        self.fused_topk = int(fused_topk)
        self.public_topk = int(public_topk)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.has_seqused_k = bool(has_seqused_k)
        self.has_residual = bool(has_residual)
        self.has_block_table = bool(has_block_table)
        self.has_output_offset = bool(has_output_offset)
        self.tiling = QsliTiling(has_cu, has_seqused_q)
        self.qk_handoff = Channel(
            MemLoc.UB,
            (TILE_M // 2, TILE_N),
            dtypes.float32,
            depth=2,
            kind=ChannelKind.CrossCore,
        )
        self.cube = QsliCube()
        self.vector0 = QsliVector0(get_subblock_id())
        self.vector1 = QsliVector1(self.tokens, self.mask_mode)
        self.output_writer = QsliOutputWriter(self.public_topk)
        self.topk_workspace = QsliRawTopKWorkspace(
            min(TOPK_TRUNK_LEN, self.tokens), self.fused_topk
        )
        self.topk = QsliRawTopKSelector(
            self.tokens, self.fused_topk, self.topk_workspace
        )
        self.position_mapper = QsliPositionMapper(
            self.fused_topk, candidate_capacity
        )

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        candidate_block_indices: Tensor,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        block_table: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_query: Tensor,
        staging_query_fp4: Tensor,
        staging_query_scale: Tensor,
        staging_query_scale_e8m0: Tensor,
        staging_key: Tensor,
        staging_key_fp4: Tensor,
        staging_scale: Tensor,
        staging_scale_e8m0: Tensor,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
    ):
        worker_idx = dtypes.int64(get_block_idx())
        subblock_idx = dtypes.int64(get_subblock_id())
        if subblock_idx == dtypes.int64(0):
            # Initialize each worker's public rows in one MTE3 phase.  The
            # single drain prevents these stores from overtaking later TopK
            # results without placing a synchronization inside the hot loop.
            for output_row in dsl_range(
                worker_idx, query_rows, worker_count, unroll=1
            ):
                self.output_writer.initialize(
                    sparse_indices, sparse_value_bits, output_row
                )
            vec_sync_all()
        for query_row in dsl_range(
            worker_idx, query_rows, worker_count, unroll=1
        ):
            batch_idx, query_offset, used_query = self.tiling.map_query(
                query_row,
                query_rows,
                batch_count,
                cu_seqlens_q,
                seqused_q,
            )
            candidate_length = dtypes.int32(
                candidate_block_length[query_row, 0]
            )
            if query_offset < used_query:
                if candidate_length > dtypes.int32(0):
                    actual_s2 = dtypes.int32(logical_k_capacity)
                    if self.has_seqused_k:
                        actual_s2 = dtypes.int32(seqused_k[batch_idx])
                    valid_s2 = actual_s2
                    if self.mask_mode == 3:
                        residual = dtypes.int32(0)
                        if self.has_residual:
                            residual = dtypes.int32(cmp_residual_k[batch_idx])
                        valid_s2 = (
                            actual_s2 * self.cmp_ratio
                            + residual
                            - used_query
                            + query_offset
                            + 1
                        ) // self.cmp_ratio
                        valid_s2 = dyn_select(
                            valid_s2 < dtypes.int32(0),
                            dtypes.int32(0),
                            valid_s2,
                        )
                        valid_s2 = dyn_select(
                            valid_s2 > actual_s2, actual_s2, valid_s2
                        )
                    # Candidate blocks are not sorted, so the visible-token
                    # count cannot be derived from min(C*8, valid_s2).  Scan
                    # the exact prefix on NPU and keep the scalar on-core;
                    # this avoids an MTE3->GM->scalar round trip per row.
                    valid_count = dtypes.int32(0)
                    if valid_s2 > dtypes.int32(0):
                        for candidate_pos in dsl_range(
                            candidate_length, unroll=1
                        ):
                            candidate_block = dtypes.int32(
                                candidate_block_indices[
                                    query_row,
                                    0,
                                    dtypes.int64(candidate_pos),
                                ]
                            )
                            visible_in_block = (
                                valid_s2
                                - candidate_block * CANDIDATE_BLOCK_SIZE
                            )
                            visible_in_block = dyn_select(
                                visible_in_block < dtypes.int32(0),
                                dtypes.int32(0),
                                visible_in_block,
                            )
                            visible_in_block = dyn_select(
                                visible_in_block
                                > dtypes.int32(CANDIDATE_BLOCK_SIZE),
                                dtypes.int32(CANDIDATE_BLOCK_SIZE),
                                visible_in_block,
                            )
                            valid_count += visible_in_block

                    # Keep the Cube/AIV control flow keyed only by sequence
                    # metadata, which is uniform across the fused core group.
                    # Candidate-derived valid_count remains an AIV TopK guard.
                    if valid_s2 > dtypes.int32(0):
                        self.vector0.stage_query(
                            query,
                            query_scale,
                            query_row,
                            staging_query,
                            staging_query_scale,
                            worker_idx,
                        )
                        self.vector0.wait_query_ready()
                        query_stage_row = worker_idx * TILE_M
                        self.cube.load_query(
                            tile_view(
                                rebase_view(
                                    staging_query_fp4, (query_stage_row, 0)
                                ),
                                (TILE_M, LOGICAL_D),
                                (0, 0),
                            ),
                            tile_view(
                                rebase_view(
                                    staging_query_scale_e8m0,
                                    (query_stage_row, 0, 0),
                                ),
                                (TILE_M, 2, 2),
                                (0, 0, 0),
                            ),
                        )
                        for candidate_tile in dsl_range(
                            self.token_tiles, unroll=1
                        ):
                            self.vector0.gather(
                                key,
                                key_scale,
                                candidate_block_indices,
                                candidate_length,
                                block_table,
                                query_row,
                                batch_idx,
                                candidate_tile,
                                staging_key,
                                staging_scale,
                                worker_idx,
                                self.has_block_table,
                            )
                            self.vector0.wait_ready()
                            staging_row = (
                                worker_idx * STAGING_DEPTH
                                + dtypes.int64(candidate_tile) % STAGING_DEPTH
                            ) * TILE_N
                            self.cube.compute_qk(
                                tile_view(
                                    rebase_view(
                                        staging_key_fp4, (staging_row, 0)
                                    ),
                                    (TILE_N, LOGICAL_D),
                                    (0, 0),
                                ),
                                tile_view(
                                    rebase_view(
                                        staging_scale_e8m0,
                                        (staging_row, 0, 0),
                                    ),
                                    (TILE_N, 2, 2),
                                    (0, 0, 0),
                                ),
                                self.qk_handoff,
                            )
                            self.vector1.compute(
                                self.qk_handoff,
                                weights,
                                candidate_block_indices,
                                key_workspace,
                                worker_idx,
                                query_row,
                                candidate_tile,
                                candidate_length,
                                valid_s2,
                                subblock_idx,
                            )

                        if subblock_idx == dtypes.int64(0):
                            vec_sync_all()
                            if valid_count > dtypes.int32(0):
                                self.topk.select_row(
                                    tile_view(
                                        key_workspace,
                                        (1, self.tokens),
                                        (worker_idx, 0),
                                    ),
                                    valid_count,
                                    tile_view(
                                        selected_position_workspace,
                                        (1, self.fused_topk),
                                        (worker_idx, 0),
                                    ),
                                    tile_view(
                                        sparse_value_bits,
                                        (1, self.fused_topk),
                                        (query_row, 0),
                                    ),
                                )
                                vec_sync_all()
                                output_offset = dtypes.int32(0)
                                if self.has_output_offset:
                                    output_offset = dtypes.int32(
                                        output_idx_offset[query_row, 0]
                                    )
                                self.position_mapper.apply(
                                    tile_view(
                                        selected_position_workspace,
                                        (1, self.fused_topk),
                                        (worker_idx, 0),
                                    ),
                                    candidate_block_indices,
                                    tile_view(
                                        sparse_indices,
                                        (1, self.fused_topk),
                                        (query_row, 0),
                                    ),
                                    query_row,
                                    valid_count,
                                    output_offset,
                                )


_COMPILED_KERNEL = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _build_compiled_fused_runner(
    worker_count,
    tokens,
    fused_topk,
    public_topk,
    candidate_capacity,
    mask_mode,
    cmp_ratio,
    has_cu,
    has_seqused_q,
    has_seqused_k,
    has_residual,
    has_block_table,
    has_output_offset,
):
    def run_fused(
        query,
        key,
        weights: Tensor,
        query_scale: Tensor,
        key_scale,
        candidate_block_indices,
        candidate_block_length: Tensor,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        block_table: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        output_idx_offset: Tensor,
        staging_query,
        staging_query_fp4,
        staging_query_scale,
        staging_query_scale_e8m0,
        staging_key,
        staging_key_fp4,
        staging_scale,
        staging_scale_e8m0,
        key_workspace: Tensor,
        selected_position_workspace: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        worker_count,
        query_rows,
        batch_count,
        logical_k_capacity,
    ):
        QsliFusedKernel(
            tokens,
            fused_topk,
            public_topk,
            candidate_capacity,
            mask_mode,
            cmp_ratio,
            has_cu,
            has_seqused_q,
            has_seqused_k,
            has_residual,
            has_block_table,
            has_output_offset,
        )[worker_count](
            query,
            key,
            weights,
            query_scale,
            key_scale,
            candidate_block_indices,
            candidate_block_length,
            cu_seqlens_q,
            seqused_q,
            block_table,
            seqused_k,
            cmp_residual_k,
            output_idx_offset,
            staging_query,
            staging_query_fp4,
            staging_query_scale,
            staging_query_scale_e8m0,
            staging_key,
            staging_key_fp4,
            staging_scale,
            staging_scale_e8m0,
            key_workspace,
            selected_position_workspace,
            sparse_indices,
            sparse_value_bits,
            worker_count,
            query_rows,
            batch_count,
            logical_k_capacity,
        )

    query_rows_dim = cannbotdsl.Dim("T1")
    physical_blocks_dim = cannbotdsl.Dim("PA")
    batch_dim = cannbotdsl.Dim("B")
    # Packed fp4x2 tensors need a static host-carrier shape in the current
    # DSL ABI.  Only this bounded launch/staging axis remains specialized.
    worker_dim = int(worker_count)
    block_table_width_dim = cannbotdsl.Dim("MAX_BLOCKS")
    key_stride_dim = cannbotdsl.Dim("K_S0")
    scale_stride_dim = cannbotdsl.Dim("KS_S0")
    fake = cannbotdsl.TensorSpec
    dummy_spec = fake((query_rows_dim, N2), dtypes.int32)
    cu_spec = (
        fake((batch_dim + 1,), dtypes.int32) if has_cu else dummy_spec
    )
    seqused_q_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_q else dummy_spec
    )
    block_table_spec = (
        fake((batch_dim, block_table_width_dim), dtypes.int32)
        if has_block_table else dummy_spec
    )
    seqused_k_spec = (
        fake((batch_dim,), dtypes.int32) if has_seqused_k else dummy_spec
    )
    residual_spec = (
        fake((batch_dim,), dtypes.int32) if has_residual else dummy_spec
    )
    offset_spec = (
        fake((query_rows_dim, N2), dtypes.int32)
        if has_output_offset else dummy_spec
    )
    compiled = jit(run_fused).compile(
        fake((query_rows_dim * N1, PACKED_D), dtypes.uint8),
        fake(
            (physical_blocks_dim, TILE_N, PACKED_D),
            dtypes.uint8,
            stride=(key_stride_dim, PACKED_D, 1),
        ),
        fake((query_rows_dim, N1), dtypes.float32),
        fake((query_rows_dim * N1, 4), dtypes.uint8),
        fake(
            (physical_blocks_dim, TILE_N, 4),
            dtypes.uint8,
            stride=(scale_stride_dim, 4, 1),
        ),
        fake(
            (query_rows_dim, N2, candidate_capacity), dtypes.int32
        ),
        fake((query_rows_dim, N2), dtypes.int32),
        cu_spec,
        seqused_q_spec,
        block_table_spec,
        seqused_k_spec,
        residual_spec,
        offset_spec,
        fake((worker_dim * TILE_M, PACKED_D), dtypes.uint8),
        fake(
            (worker_dim * TILE_M, LOGICAL_D), dtypes.fp4x2_e2m1
        ),
        fake((worker_dim * TILE_M, 4), dtypes.uint8),
        fake(
            (worker_dim * TILE_M, 2, 2), dtypes.float8_e8m0
        ),
        fake(
            (worker_dim * STAGING_DEPTH * TILE_N, PACKED_D),
            dtypes.uint8,
        ),
        fake(
            (worker_dim * STAGING_DEPTH * TILE_N, LOGICAL_D),
            dtypes.fp4x2_e2m1,
        ),
        fake(
            (worker_dim * STAGING_DEPTH * TILE_N, 4), dtypes.uint8
        ),
        fake(
            (worker_dim * STAGING_DEPTH * TILE_N, 2, 2),
            dtypes.float8_e8m0,
        ),
        fake((worker_dim, tokens), dtypes.uint16),
        fake((worker_dim, fused_topk), dtypes.int32),
        fake((query_rows_dim, public_topk), dtypes.int32),
        fake((query_rows_dim, public_topk), dtypes.uint16),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    return compiled


def _get_compiled_fused_runner(*config):
    """Compile once per constexpr algorithm branch; all shape axes are dynamic."""
    cache_key = tuple(config)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNEL.get(cache_key)
        if compiled is None:
            compiled = _build_compiled_fused_runner(*config)
            _COMPILED_KERNEL[cache_key] = compiled
        return compiled


def quant_sparse_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: Optional[torch.Tensor],
    descale_k: Optional[torch.Tensor],
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    topk: int,
    candidate_block_size: int,
    quant_mode: int = 1,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    return_value: bool = False,
) -> dict[str, torch.Tensor]:
    """Run fused QSLI without a Host-built scheduling metadata table.

    ``candidate_block_length[t, 0]`` is the exact number of valid leading
    entries in ``candidate_block_indices[t, 0]`` and must be in ``[0, 2048]``.
    Every entry in that prefix is a logical eight-token block id valid for the
    row's batch; the suffix is ignored.  These value constraints are consumed
    as the device-side contract so Host does not copy candidates or lengths
    back to CPU.  ``metadata`` remains reserved and must be ``None``.
    """
    del return_value
    if int(quant_mode) != 1:
        raise ValueError(
            "quant_sparse_lightning_indexer supports quant_mode=1 only"
        )
    if layout_q != "TND" or layout_kv != "PA_BBND":
        raise ValueError("current QSLI layouts must be TND and PA_BBND")
    if q.device.type != "npu" or k.device.type != "npu":
        raise ValueError(
            "quant_sparse_lightning_indexer requires NPU q/k tensors"
        )
    if q.dtype != torch.uint8 or k.dtype != torch.uint8:
        raise TypeError("q/k storage must be uint8 packed MXFP4")
    if q.ndim != 3 or tuple(q.shape[1:]) != (N1, PACKED_D):
        raise ValueError("q must have shape (T1,32,64)")
    if k.ndim != 4 or tuple(k.shape[1:]) != (TILE_N, N2, PACKED_D):
        raise ValueError("k must have shape (block_num,128,1,64)")
    query_rows = int(q.shape[0])
    physical_k_blocks = int(k.shape[0])
    if query_rows <= 0 or physical_k_blocks <= 0:
        raise ValueError("T1 and the physical K block count must be positive")
    _validate_pa_dim0_stride(k, "k")

    if tuple(w.shape) != (query_rows, N1) or w.dtype != torch.float32:
        raise ValueError("w must be FP32 with shape (T1,32)")
    if w.device.type != "npu":
        raise ValueError("w must be an NPU tensor")
    expected_q_scale = (query_rows, N1, LOGICAL_D // 64, 2)
    if descale_q is None or tuple(descale_q.shape) != expected_q_scale:
        raise ValueError(
            "descale_q is required with shape (T1,32,D/64,2)"
        )
    if descale_k is None or tuple(descale_k.shape) != tuple(k.shape[:3]) + (2, 2):
        raise ValueError("descale_k must have shape (block_num,128,1,2,2)")
    scale_dtypes = (torch.uint8, torch.float8_e8m0fnu)
    if descale_q.dtype not in scale_dtypes or descale_q.device.type != "npu":
        raise TypeError("descale_q must be an NPU E8M0/uint8 tensor")
    if descale_k.dtype not in scale_dtypes or descale_k.device.type != "npu":
        raise TypeError("descale_k must be an NPU E8M0/uint8 tensor")
    key_scale_stride0 = _validate_pa_dim0_stride(descale_k, "descale_k")
    if key_scale_stride0 % 2 != 0:
        raise ValueError("descale_k axis-0 stride must satisfy E8M0 pair alignment")

    if metadata is not None:
        raise ValueError("metadata is reserved and must be None")
    if int(candidate_block_size) != CANDIDATE_BLOCK_SIZE:
        raise ValueError("candidate_block_size must be 8")
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("topk must be in [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("cmp_ratio must be in [1,128]")
    if int(max_seqlen_q) < -1:
        raise ValueError("max_seqlen_q must be -1 or nonnegative")

    if (
        candidate_block_indices.ndim != 3
        or tuple(candidate_block_indices.shape[:2]) != (query_rows, N2)
        or int(candidate_block_indices.shape[2]) != CANDIDATE_CAPACITY
    ):
        raise ValueError("candidate_block_indices must have shape (T1,1,2048)")
    if (
        candidate_block_indices.dtype != torch.int32
        or candidate_block_indices.device.type != "npu"
    ):
        raise TypeError("candidate_block_indices must be an NPU int32 tensor")
    if (
        candidate_block_length.dtype != torch.int32
        or candidate_block_length.device.type != "npu"
        or int(candidate_block_length.numel()) != query_rows * N2
    ):
        raise ValueError(
            "candidate_block_length must be NPU int32 with T1*N2 entries"
        )
    candidate_length_2d = candidate_block_length.reshape(query_rows, N2)

    if cu_seqlens_q is None:
        batch = 1
    else:
        if (
            cu_seqlens_q.dtype != torch.int32
            or cu_seqlens_q.device.type != "npu"
            or cu_seqlens_q.ndim != 1
            or int(cu_seqlens_q.numel()) < 2
        ):
            raise ValueError("cu_seqlens_q must be NPU int32 with shape (B+1,)")
        batch = int(cu_seqlens_q.numel()) - 1
    if seqused_q is not None and (
        tuple(seqused_q.shape) != (batch,)
        or seqused_q.dtype != torch.int32
        or seqused_q.device.type != "npu"
    ):
        raise ValueError("seqused_q must be NPU int32 with shape (B,)")

    if block_table is None:
        if batch != 1:
            raise ValueError("block_table is required when batch is greater than one")
        logical_k_capacity = physical_k_blocks * TILE_N
    else:
        if (
            block_table.ndim != 2
            or int(block_table.shape[0]) != batch
            or block_table.dtype != torch.int32
            or block_table.device.type != "npu"
        ):
            raise ValueError("block_table must be NPU int32 with shape (B,max_blocks)")
        logical_k_capacity = int(block_table.shape[1]) * TILE_N
    if seqused_k is not None and (
        tuple(seqused_k.shape) != (batch,)
        or seqused_k.dtype != torch.int32
        or seqused_k.device.type != "npu"
    ):
        raise ValueError("seqused_k must be NPU int32 with shape (B,)")

    needs_residual = int(mask_mode) == 3 and int(cmp_ratio) != 1
    if needs_residual:
        if cmp_residual_k is None or (
            tuple(cmp_residual_k.shape) != (batch,)
            or cmp_residual_k.dtype != torch.int32
            or cmp_residual_k.device.type != "npu"
        ):
            raise ValueError(
                "cmp_residual_k must be NPU int32 with shape (B,) for causal compression"
            )
    elif cmp_residual_k is not None:
        raise ValueError(
            "cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
        )
    if output_idx_offset is not None and (
        tuple(output_idx_offset.shape) != (query_rows, N2)
        or output_idx_offset.dtype != torch.int32
        or output_idx_offset.device.type != "npu"
    ):
        raise ValueError("output_idx_offset must be NPU int32 with shape (T1,1)")

    # Static capacity supplies the rolled loop bound; dynamic lengths and
    # candidate prefixes are consumed in the kernel and never copied to CPU.
    # Compile the candidate-capacity workspace once.  ``logical_k_capacity``
    # remains a runtime scalar used by masking and position mapping, so S2 and
    # PA/block-table shapes no longer create shape-specialized binaries.
    tokens = CANDIDATE_CAPACITY * CANDIDATE_BLOCK_SIZE
    fused_topk = min(int(topk), tokens)
    worker_count = min(query_rows, MAX_CUBE_WORKERS)

    sparse_indices = q.new_empty((query_rows, N2, int(topk)), dtype=torch.int32)
    sparse_values = w.new_empty((query_rows, N2, int(topk)), dtype=torch.bfloat16)
    staging_query = q.new_empty((worker_count * TILE_M, PACKED_D))
    staging_query_scale = q.new_empty((worker_count * TILE_M, 4))
    staging_key = q.new_empty(
        (worker_count * STAGING_DEPTH * TILE_N, PACKED_D)
    )
    staging_scale = q.new_empty(
        (worker_count * STAGING_DEPTH * TILE_N, 4)
    )
    key_workspace = q.new_empty((worker_count, tokens), dtype=torch.uint16)
    selected_position_workspace = q.new_empty(
        (worker_count, fused_topk), dtype=torch.int32
    )

    # Optional inputs use an existing int32 tensor as an ABI placeholder.
    # Their constexpr flags remove every dummy access from generated device code.
    int32_dummy = candidate_length_2d
    cu_npu = cu_seqlens_q if cu_seqlens_q is not None else int32_dummy
    seqused_q_npu = seqused_q if seqused_q is not None else int32_dummy
    block_table_npu = block_table if block_table is not None else int32_dummy
    seqused_k_npu = seqused_k if seqused_k is not None else int32_dummy
    residual_npu = cmp_residual_k if cmp_residual_k is not None else int32_dummy
    offset_npu = output_idx_offset if output_idx_offset is not None else int32_dummy

    query_scale_bytes = (
        descale_q
        if descale_q.dtype == torch.uint8
        else descale_q.view(torch.uint8)
    )
    key_scale_bytes = (
        descale_k
        if descale_k.dtype == torch.uint8
        else descale_k.view(torch.uint8)
    )
    _get_compiled_fused_runner(
        worker_count,
        tokens,
        fused_topk,
        int(topk),
        CANDIDATE_CAPACITY,
        int(mask_mode),
        int(cmp_ratio),
        cu_seqlens_q is not None,
        seqused_q is not None,
        seqused_k is not None,
        cmp_residual_k is not None,
        block_table is not None,
        output_idx_offset is not None,
    )(
        q.reshape(query_rows * N1, PACKED_D),
        k.squeeze(2),
        w,
        query_scale_bytes.reshape(query_rows * N1, 4),
        key_scale_bytes.reshape(physical_k_blocks, TILE_N, 4),
        candidate_block_indices,
        candidate_length_2d,
        cu_npu,
        seqused_q_npu,
        block_table_npu,
        seqused_k_npu,
        residual_npu,
        offset_npu,
        staging_query,
        staging_query.view(torch.int8),
        staging_query_scale,
        staging_query_scale.reshape(-1, 2, 2).view(torch.float8_e8m0fnu),
        staging_key,
        staging_key.view(torch.int8),
        staging_scale,
        staging_scale.reshape(-1, 2, 2).view(torch.float8_e8m0fnu),
        key_workspace,
        selected_position_workspace,
        sparse_indices.reshape(query_rows, int(topk)),
        sparse_values.reshape(query_rows, int(topk)).view(torch.uint16),
        worker_count,
        query_rows,
        batch,
        logical_k_capacity,
    )
    return {"sparse_indices": sparse_indices, "sparse_values": sparse_values}


__all__ = ["quant_sparse_lightning_indexer"]
