"""Public QLI MXFP4 implementation.

The public ABI is the one in ``算子接口.xlsx``: q is ``uint8[T1,32,64]`` and
k is ``uint8[block_num,block_size,1,64]`` (two packed E2M1 values per byte).
The Torch-facing host layer only validates the public ABI, builds launch
metadata/staging workspaces, and allocates public outputs.  It never creates
test inputs or computes a CPU reference.  DSL kernels load native FP4 E2M1
into L1/L0 and perform the Cube MX QK product into an FP32 accumulator.
The remaining ReLU, weighted G=32 reduction, token TopK and optional
candidate-block TopK stay on NPU. The ASC MXFP4 reference uses BF16 Vector1
``MulAddDst``; the current CANNBotDSL raw ``vmadd`` lowering does not accept
BF16 operands, so only the 128-token arithmetic carrier uses FP32. It applies
explicit BF16-grid rounding at the ASC-defined boundaries and exports both
score codes and sortable keys as uint16. The fallback is documented in
``samples/quant_lightning_indexer/README.md``.
"""

from __future__ import annotations

import threading
from typing import Optional

import cannbotdsl
import torch

from cannbotdsl import ChannelKind, MemLoc, Tensor, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.spec import TensorSpec
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.matmul import matmul as dsl_matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.sync import (
    PIPE,
    cube_sync_all,
    cube_sync_intra_wait,
    vec_sync_all,
    vec_sync_intra_arrive,
)
from cannbotdsl.tensor import (
    make_partition_tiler,
    partition_view,
    rebase_view,
    tile_view,
)
from cannbotdsl.ops import reg as rr


# One M tile contains four public N1=32 query groups, matching ASC's
# M_BASIC_BLOCK_SMALL=128 path.
TILE_M = 128
TILE_N = 128
LOGICAL_D = 128
PACKED_D = LOGICAL_D // 2
N1 = 32
N2 = 1
VEC_TILE = 128
CANDIDATE_BLOCK_SIZE = 8
TOPK_TRUNK_LEN = 16384
MAX_TOKEN_CAPACITY = 128 * 1024
MAX_CUBE_WORKERS = 32
STAGING_DEPTH = 3
QUERY_READY_ID = 5
KEY_READY_ID = 6
AIV1_SYNC_ID_OFFSET = 16
KEY_STORAGE_ROW_ELEMENTS = N2 * PACKED_D
KEY_SCALE_STORAGE_ROW_ELEMENTS = N2 * 2 * 2


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _pa_axis0_storage_rows(
    tensor: torch.Tensor,
    name: str,
    expected_inner_strides: tuple[int, ...],
    storage_row_elements: int,
) -> tuple[int, int]:
    """Validate ASC's PA stride contract and expose its storage-row span.

    Only axis 0 may contain padding. The returned extent includes that
    padding without copying or repacking public input data; the fused kernel
    still selects physical pages with the original axis-0 stride.
    """
    strides = tuple(int(step) for step in tensor.stride())
    if strides[1:] != expected_inner_strides:
        raise ValueError(
            f"{name} only supports non-contiguous storage on axis 0 in "
            "PA_BBND layout"
        )
    stride0 = strides[0]
    if stride0 < 0:
        raise ValueError(f"{name} axis-0 stride must be non-negative")
    if stride0 % storage_row_elements != 0:
        raise ValueError(
            f"{name} axis-0 stride must preserve complete PA storage rows"
        )
    blocks = int(tensor.shape[0])
    if blocks <= 0:
        raise ValueError(f"{name} must contain at least one physical PA block")
    storage_rows = (
        (blocks - 1) * (stride0 // storage_row_elements)
        + int(tensor.shape[1])
    )
    return stride0, storage_rows


# ============================================================================
# 1. Device kernels: Cube QK, Vector1, raw-VF TopK
# ============================================================================


class QliRawTopKWorkspace:
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


class QliRawTopKSelector:
    """ASC Arch35 uint16 radix TopK with in-AIV trunk/history merge.

    This is not LD/S2 core splitting.  One AIV processes a row sequentially in
    16K trunks.  The first trunk creates a TopK history; every later pass runs
    the same radix selector over ``history + current trunk`` and maps its local
    uint16 positions back to global int32 token indices.
    """

    def __init__(
        self,
        topk: int,
        workspace: QliRawTopKWorkspace,
    ):
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("ASC raw TopK requires positive topk")
        # The ASC VF works in fixed 16K trunks.  The number of complete trunks
        # and the final tail length are runtime values so one compiled kernel
        # can serve every supported S2.
        self.trunk_len = TOPK_TRUNK_LEN
        # ASC reserves history in 256-element units before the next trunk.
        self.topk_pad = ceil_div(self.topk, 256) * 256
        self.merge_len_pad = ceil_div(
            self.topk_pad + TOPK_TRUNK_LEN, 128
        ) * 128
        self.workspace = workspace
        if (
            TOPK_TRUNK_LEN > workspace.max_trunk_len
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
            current_chunk_count = (current_len + 127) // 128
            for chunk in dsl_range(current_chunk_count, unroll=1):
                source_offset = chunk * 128
                count = current_len - source_offset
                if count > 128:
                    count = 128
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
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                if valid_count > 128:
                    valid_count = 128
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
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                if valid_count > 128:
                    valid_count = 128
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
            valid_chunk_count = (valid_len + 127) // 128
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                if valid_count > 128:
                    valid_count = 128
                mask16 = rr.update_mask(valid_count, elem_bits=16)[0]
                keys = rr.vload(sort_key, chunk * 128)
                idx = rr.varange(chunk * 128, dtypes.uint16)
                sq = rr.vsqueeze_and_storeunalign_init(
                    idx, mask=rr.vgt(keys, kth, mask=mask16)
                )
                rr.vsqueeze_and_storeunalign(tmp_idx, 0, sq, ureg)
            for chunk in dsl_range(valid_chunk_count, unroll=1):
                valid_count = valid_len - chunk * 128
                if valid_count > 128:
                    valid_count = 128
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
    def store_empty_row(
        self,
        gm_sparse_indices: Tensor,
        gm_selected_value_bits: Tensor,
    ):
        """Store the public empty-TopK representation without scanning GM."""
        with vf(mode="raw"):
            for chunk in dsl_range(ceil_div(self.topk, 64), unroll=1):
                count = min(64, self.topk - chunk * 64)
                mask32 = rr.update_mask(count, elem_bits=32)[0]
                mask16 = rr.update_mask(count, elem_bits=16)[0]
                rr.vstore(
                    self.output_idx_stage,
                    chunk * 64,
                    rr.vdups(-1, dtypes.int32, mask=mask32),
                    mask32,
                )
                rr.vstore(
                    self.output_value_bits_stage,
                    chunk * 64,
                    rr.vdups(0, dtypes.uint16, mask=mask16),
                    mask16,
                )
            rr.vmem_bar("vst_vld")
        mem_copy(gm_sparse_indices, self.output_idx_stage)
        mem_copy(gm_selected_value_bits, self.output_value_bits_stage)

    @jit
    def select_row(
        self,
        gm_sort_key: Tensor,
        total_tokens,
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

        full_trunk_count = total_tokens // TOPK_TRUNK_LEN
        tail_len = total_tokens % TOPK_TRUNK_LEN
        for pass_idx in dsl_range(full_trunk_count, unroll=1):
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

        if tail_len > 0:
            tail_offset = full_trunk_count * self.trunk_len
            tail_len_pad = ((tail_len + 255) // 256) * 256
            mem_copy(
                tile_view(
                    rebase_view(current_trunk, (0, 0)),
                    (1, self.trunk_len),
                    (0, 0),
                ),
                tile_view(
                    rebase_view(gm_sort_key, (0, tail_offset)),
                    (1, self.trunk_len),
                    (0, 0),
                ),
            )
            self._build_merge(
                current_trunk,
                history_key,
                merge_key,
                tail_len,
                tail_len_pad,
            )
            self._select_pass(
                merge_key,
                self.topk_pad + tail_len,
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


class QliInputStager:
    """Stage dynamic uint8 MX carriers for the AIC's static packed views.

    CANNBotDSL cannot expose a dynamic-shape fp4x2/e8m0 tensor through its
    host ABI.  Both AIVs therefore move one half-tile of the public uint8
    carriers through UB into fixed per-core GM slots.  The AIC reads aliases
    of those same slots as native fp4x2/e8m0; all work remains inside this
    single fused kernel.
    """

    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.query_ub = Channel(
            MemLoc.UB, (TILE_M // 2, PACKED_D), dtypes.uint8, depth=1
        )
        self.query_scale_ub = Channel(
            MemLoc.UB, (TILE_M // 2, 4), dtypes.uint8, depth=1
        )
        self.key_ub = Channel(
            MemLoc.UB, (TILE_N // 2, PACKED_D), dtypes.uint8, depth=2
        )
        self.key_scale_ub = Channel(
            MemLoc.UB, (TILE_N // 2, 4), dtypes.uint8, depth=2
        )

    @jit
    def stage_query(
        self,
        query,
        query_scale,
        public_row_base,
        task_span_count,
        staging_query,
        staging_query_scale,
        worker_idx,
    ):
        query_slot = self.query_ub.acquire()
        scale_slot = self.query_scale_ub.acquire()
        for row_in_subblock in range_constexpr(2):
            local_row = (
                dtypes.int32(self.subblock_idx) * 2 + row_in_subblock
            )
            source_row = dtypes.int64(public_row_base)
            if local_row < task_span_count:
                source_row = (
                    dtypes.int64(public_row_base)
                    + dtypes.int64(local_row)
                )
            mem_copy(
                tile_view(
                    query_slot,
                    (N1, PACKED_D),
                    (row_in_subblock, 0),
                ),
                tile_view(query, (N1, PACKED_D), (source_row, 0)),
            )
            mem_copy(
                tile_view(
                    scale_slot, (N1, 4), (row_in_subblock, 0)
                ),
                tile_view(query_scale, (N1, 4), (source_row, 0)),
            )
        self.query_ub.commit(query_slot)
        self.query_scale_ub.commit(scale_slot)
        query_ready = self.query_ub.wait()
        scale_ready = self.query_scale_ub.wait()
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
            query_ready,
        )
        mem_copy(
            tile_view(
                rebase_view(staging_query_scale, (staging_row, 0)),
                (TILE_M // 2, 4),
                (0, 0),
            ),
            scale_ready,
        )
        self.query_ub.release(query_ready)
        self.query_scale_ub.release(scale_ready)
        vec_sync_intra_arrive(PIPE.MTE3, QUERY_READY_ID)

    @jit
    def wait_query_ready(self):
        cube_sync_intra_wait(PIPE.MTE2, QUERY_READY_ID)
        cube_sync_intra_wait(
            PIPE.MTE2, QUERY_READY_ID + AIV1_SYNC_ID_OFFSET
        )

    @jit
    def stage_key(
        self,
        key,
        key_scale,
        physical_row,
        physical_scale_row,
        token_tile,
        staging_key,
        staging_key_scale,
        worker_idx,
    ):
        key_slot = self.key_ub.acquire()
        scale_slot = self.key_scale_ub.acquire()
        source_row = (
            dtypes.int64(physical_row)
            + dtypes.int64(self.subblock_idx) * (TILE_N // 2)
        )
        source_scale_row = (
            dtypes.int64(physical_scale_row)
            + dtypes.int64(self.subblock_idx) * (TILE_N // 2)
        )
        mem_copy(
            key_slot,
            tile_view(
                rebase_view(key, (source_row, 0)),
                (TILE_N // 2, PACKED_D),
                (0, 0),
            ),
        )
        mem_copy(
            scale_slot,
            tile_view(
                rebase_view(key_scale, (source_scale_row, 0)),
                (TILE_N // 2, 4),
                (0, 0),
            ),
        )
        self.key_ub.commit(key_slot)
        self.key_scale_ub.commit(scale_slot)
        key_ready = self.key_ub.wait()
        scale_ready = self.key_scale_ub.wait()
        staging_row = (
            (
                dtypes.int64(worker_idx) * STAGING_DEPTH
                + dtypes.int64(token_tile) % STAGING_DEPTH
            )
            * TILE_N
            + dtypes.int64(self.subblock_idx) * (TILE_N // 2)
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
                rebase_view(staging_key_scale, (staging_row, 0)),
                (TILE_N // 2, 4),
                (0, 0),
            ),
            scale_ready,
        )
        self.key_ub.release(key_ready)
        self.key_scale_ub.release(scale_ready)
        vec_sync_intra_arrive(PIPE.MTE3, KEY_READY_ID)

    @jit
    def wait_key_ready(self):
        cube_sync_intra_wait(PIPE.MTE2, KEY_READY_ID)
        cube_sync_intra_wait(
            PIPE.MTE2, KEY_READY_ID + AIV1_SYNC_ID_OFFSET
        )


class QliCube:
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
        self.q_tail_l1 = Channel(
            MemLoc.L1, (TILE_M, LOGICAL_D), dtypes.fp4x2_e2m1,
            depth=1, data_format="nz",
        )
        self.q_tail_scale_l1 = Buffer(
            MemLoc.L1,
            (TILE_M, 4),
            dtypes.float8_e8m0,
            data_format="zn",
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
            format_transform="nd2nz", dtype=dtypes.fp4x2_e2m1, pad_value=0.0
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

    @jit
    def load_query(self, query, query_scale, query_token_start, query_count):
        """Gather at most four public TND rows directly into the M=128 tile.

        Q and q_descale stay in their public xlsx layouts. The fused AIC path
        converts paired E8M0 q_descale groups to the L1 ScaleA layout. Tail
        slots duplicate the first valid query row and are discarded by the
        vector output path.
        """
        if query_count == TILE_M // N1:
            source_row = dtypes.int64(query_token_start) * N1
            mem_copy(
                self.q_l1,
                tile_view(
                    rebase_view(query, (source_row, 0)),
                    (TILE_M, LOGICAL_D),
                    (0, 0),
                ),
                engine=self.nd2nz_fp4,
            )
            mem_copy(
                self.q_scale_l1,
                tile_view(
                    rebase_view(query_scale, (source_row, 0, 0)),
                    (TILE_M, 2, 2),
                    (0, 0, 0),
                ),
                engine=self.scale_a,
            )
            mem_copy(self.l0a, self.q_l1, mx_scale=self.q_scale_l1)
        else:
            q_l1_slot = self.q_tail_l1.acquire()
            q_half_plan = make_partition_tiler(
                (TILE_M, LOGICAL_D), (TILE_M, LOGICAL_D), parts=2
            )
            for row_in_task in range_constexpr(TILE_M // N1):
                source_token = dtypes.int64(query_token_start)
                if row_in_task < query_count:
                    source_token = (
                        dtypes.int64(query_token_start)
                        + dtypes.int64(row_in_task)
                    )
                q_half = partition_view(
                    q_l1_slot, q_half_plan, row_in_task // 2
                )
                q_partition = tile_view(
                    q_half, (N1, LOGICAL_D), (row_in_task % 2, 0)
                )
                mem_copy(
                    q_partition,
                    tile_view(query, (N1, LOGICAL_D), (source_token, 0)),
                    engine=self.nd2nz_fp4,
                )
                mem_copy(
                    tile_view(
                        self.q_tail_scale_l1,
                        (N1, 4),
                        (row_in_task, 0),
                    ),
                    tile_view(
                        query_scale, (N1, 2, 2), (source_token, 0, 0)
                    ),
                    engine=self.scale_a,
                )
            self.q_tail_l1.commit(q_l1_slot)
            q_l1_ready = self.q_tail_l1.wait()
            cube_sync_all()
            mem_copy(
                self.l0a, q_l1_ready, mx_scale=self.q_tail_scale_l1
            )
            self.q_tail_l1.release(q_l1_ready)

    def compute_qk(self, key, key_scale, qk_handoff):
        mem_copy(self.k_l1, key, engine=self.nd2nz_fp4)
        mem_copy(self.k_scale_l1, key_scale, engine=self.scale_b)
        mem_copy(self.l0b, self.k_l1, mx_scale=self.k_scale_l1)
        dsl_matmul(self.l0c, self.l0a, self.l0b, init=True, unit_flag=3)
        split_m = make_partition_tiler(
            (TILE_M, TILE_N), (TILE_M, TILE_N)
        )
        mem_copy(
            qk_handoff, self.l0c,
            engine=self.fixpipe_split_m, partition=split_m,
        )

class QliVector:
    """Each AIV consumes its 64-row split and produces two QLI rows."""

    def __init__(
        self,
        subblock_idx,
        mask_mode,
        cmp_ratio,
        candidate_enabled,
        candidate_topk_count,
    ):
        self.subblock_idx = subblock_idx
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.candidate_enabled = bool(candidate_enabled)
        self.candidate_topk_count = int(candidate_topk_count)
        self.weight_ub = Channel(
            MemLoc.UB, (1, N1), dtypes.float32, depth=1
        )
        if self.candidate_enabled:
            self.candidate_source_ub = Buffer(
                MemLoc.UB, (1, TILE_N), dtypes.float32
            )
        self.key_ub = Channel(
            MemLoc.UB, (1, TILE_N), dtypes.uint16, depth=1
        )
        if self.candidate_enabled:
            self.candidate_key_lo_ub = Channel(
                MemLoc.UB,
                (1, 64 // CANDIDATE_BLOCK_SIZE),
                dtypes.uint16,
                depth=1,
            )
            self.candidate_key_hi_ub = Channel(
                MemLoc.UB,
                (1, 64 // CANDIDATE_BLOCK_SIZE),
                dtypes.uint16,
                depth=1,
            )
        self.candidate_length_ub = Channel(
            MemLoc.UB, (1,), dtypes.int32, depth=1
        )

    @jit
    def store_empty_candidate_lengths(
        self, candidate_length_out, public_row_base, task_span_count
    ):
        """Write zero Candidate length for both rows owned by this AIV."""
        for row_in_subblock in range_constexpr(2):
            local_row = dtypes.int32(
                self.subblock_idx * 2 + row_in_subblock
            )
            if local_row < task_span_count:
                length_slot = self.candidate_length_ub.acquire()
                with vf(mode="raw"):
                    length_mask = rr.update_mask(1, elem_bits=32)[0]
                    rr.vstore(
                        length_slot,
                        0,
                        rr.vdups(0, dtypes.int32, mask=length_mask),
                        length_mask,
                    )
                    rr.vmem_bar("vst_vld")
                self.candidate_length_ub.commit(length_slot)
                length_ready = self.candidate_length_ub.wait()
                mem_copy(
                    tile_view(
                        candidate_length_out,
                        (1,),
                        (
                            dtypes.int64(public_row_base)
                            + dtypes.int64(local_row),
                        ),
                    ),
                    length_ready,
                )
                self.candidate_length_ub.release(length_ready)

    @jit
    def compute_vector1(
        self,
        qk_handoff,
        weights,
        key_out,
        candidate_key_out,
        public_row_base,
        task_span_count,
        task_active_count,
        workspace_task_idx,
        token_offset,
        actual_s2,
        task_q_offset,
        task_used_q,
        cmp_residual,
        total_candidate_tokens,
        candidate_length_out,
    ):
        for row_in_subblock in range_constexpr(2):
            local_row = dtypes.int32(
                self.subblock_idx * 2 + row_in_subblock
            )
            public_row = (
                dtypes.int64(public_row_base) + dtypes.int64(local_row)
            )
            weight_row = dtypes.int64(public_row_base)
            if local_row < task_span_count:
                weight_row = public_row
            workspace_row = (
                workspace_task_idx * 4
                + self.subblock_idx * 2
                + row_in_subblock
            )
            valid_s2 = actual_s2
            if self.mask_mode == 3:
                query_in_batch = (
                    dtypes.int32(task_q_offset)
                    + dtypes.int32(self.subblock_idx) * 2
                    + dtypes.int32(row_in_subblock)
                )
                valid_s2 = (
                    actual_s2 * self.cmp_ratio
                    + cmp_residual
                    - task_used_q
                    + query_in_batch
                    + 1
                ) // self.cmp_ratio
                if valid_s2 < 0:
                    valid_s2 = 0
                if valid_s2 > actual_s2:
                    valid_s2 = actual_s2
            if local_row >= task_active_count:
                valid_s2 = 0
            mem_copy(
                self.weight_ub,
                tile_view(weights, (1, N1), (weight_row, 0)),
            )
            candidate_length_slot = self.candidate_length_ub.acquire()
            with vf(mode="raw"):
                full32 = rr.full_mask()
                mask16 = rr.update_mask(TILE_N, elem_bits=16)[0]
                length_mask = rr.update_mask(1, elem_bits=32)[0]
                rr.vstore(
                    candidate_length_slot,
                    0,
                    rr.vdups(0, dtypes.int32, mask=length_mask),
                    length_mask,
                )
                for off in range_constexpr(0, 128, 64):
                    acc = rr.vdups(0.0, dtypes.float32, mask=full32)
                    for pair in range_constexpr(N1 // 2):
                        for group in range_constexpr(pair * 2, pair * 2 + 2):
                            qk_offset = (
                                (row_in_subblock * N1 + group) * TILE_N + off
                            )
                            qk = rr.vload(qk_handoff, qk_offset)
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
                    global_token = rr.vadds(
                        rr.varange(off, dtypes.uint32),
                        token_offset,
                        mask=full32,
                    )
                    valid_token = rr.vlts(
                        global_token, valid_s2, mask=full32
                    )
                    acc = rr.vselect(
                        acc,
                        rr.vreinterpret(
                            rr.vdups(
                                0xFF800000, dtypes.uint32, mask=full32
                            ),
                            dtypes.float32,
                        ),
                        cond_mask=valid_token,
                    )
                    if self.candidate_enabled:
                        rr.vstore(
                            self.candidate_source_ub, off, acc, full32
                        )
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
                        rr.vselect(
                            negative_key, positive_key, cond_mask=negative
                        ),
                        mask16,
                    )

                if self.candidate_enabled:
                    valid_candidate_count = (
                        valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                    ) // CANDIDATE_BLOCK_SIZE
                    rr.vmem_bar("vst_vld")
                    candidate_count = TILE_N // CANDIDATE_BLOCK_SIZE
                    candidate_mask32 = rr.update_mask(
                        candidate_count, elem_bits=32
                    )[0]
                    candidate_mask16 = rr.update_mask(
                        candidate_count, elem_bits=16
                    )[0]
                    candidate_idx = rr.vmuls(
                        rr.varange(0, dtypes.uint32),
                        CANDIDATE_BLOCK_SIZE,
                        mask=candidate_mask32,
                    )
                    candidate_max = rr.vgather(
                        self.candidate_source_ub,
                        candidate_idx,
                        mask=candidate_mask32,
                    )
                    for lane in range_constexpr(1, CANDIDATE_BLOCK_SIZE):
                        candidate_max = rr.vmax(
                            candidate_max,
                            rr.vgather(
                                self.candidate_source_ub,
                                rr.vadds(
                                    candidate_idx,
                                    lane,
                                    mask=candidate_mask32,
                                ),
                                mask=candidate_mask32,
                            ),
                            mask=candidate_mask32,
                        )
                    candidate_global_idx = rr.vadds(
                        rr.varange(0, dtypes.uint32),
                        token_offset // CANDIDATE_BLOCK_SIZE,
                        mask=candidate_mask32,
                    )
                    candidate_max = rr.vselect(
                        candidate_max,
                        rr.vreinterpret(
                            rr.vdups(
                                0xFF800000,
                                dtypes.uint32,
                                mask=candidate_mask32,
                            ),
                            dtypes.float32,
                        ),
                        cond_mask=rr.vlts(
                            candidate_global_idx,
                            valid_candidate_count,
                            mask=candidate_mask32,
                        ),
                    )
                    remainder = valid_s2 % CANDIDATE_BLOCK_SIZE
                    partial_target = (
                        valid_s2 // CANDIDATE_BLOCK_SIZE
                        + (
                            (CANDIDATE_BLOCK_SIZE - remainder)
                            // CANDIDATE_BLOCK_SIZE
                        )
                        * (dtypes.int32(total_candidate_tokens) + 1)
                    )
                    candidate_max = rr.vselect(
                        rr.vreinterpret(
                            rr.vdups(
                                0x7F800000,
                                dtypes.uint32,
                                mask=candidate_mask32,
                            ),
                            dtypes.float32,
                        ),
                        candidate_max,
                        cond_mask=rr.veqs(
                            candidate_global_idx,
                            partial_target,
                            mask=candidate_mask32,
                        ),
                    )
                    candidate_bits16 = rr.vpack(
                        rr.vshr(
                            rr.vreinterpret(candidate_max, dtypes.uint32),
                            16,
                            mask=candidate_mask32,
                        ),
                        dtypes.uint16,
                        part="lower",
                    )
                    candidate_key = rr.vbitwise_xor(
                        candidate_bits16,
                        rr.vdups(
                            0x8000, dtypes.uint16, mask=candidate_mask16
                        ),
                        mask=candidate_mask16,
                    )
                    candidate_key = rr.vselect(
                        candidate_key,
                        rr.vdups(0, dtypes.uint16, mask=candidate_mask16),
                        cond_mask=rr.vlts(
                            rr.vadds(
                                rr.varange(0, dtypes.uint16),
                                token_offset // CANDIDATE_BLOCK_SIZE,
                                mask=candidate_mask16,
                            ),
                            valid_candidate_count,
                            mask=candidate_mask16,
                        ),
                    )
                    half_candidate_mask = rr.update_mask(
                        64 // CANDIDATE_BLOCK_SIZE, elem_bits=16
                    )[0]
                    rr.vstore(
                        self.candidate_key_lo_ub,
                        0,
                        candidate_key,
                        half_candidate_mask,
                    )
                    candidate_key_hi = rr.vgather_reg(
                        candidate_key,
                        rr.varange(
                            64 // CANDIDATE_BLOCK_SIZE, dtypes.uint16
                        ),
                    )
                    rr.vstore(
                        self.candidate_key_hi_ub,
                        0,
                        candidate_key_hi,
                        half_candidate_mask,
                    )
                    length_value = rr.vdups(
                        dtypes.int32(valid_candidate_count),
                        dtypes.int32,
                        mask=length_mask,
                    )
                    length_value = rr.vselect(
                        length_value,
                        rr.vdups(
                            self.candidate_topk_count,
                            dtypes.int32,
                            mask=length_mask,
                        ),
                        cond_mask=rr.vlts(
                            length_value,
                            self.candidate_topk_count,
                            mask=length_mask,
                        ),
                    )
                    rr.vstore(
                        candidate_length_slot,
                        0,
                        length_value,
                        length_mask,
                    )
                    rr.vmem_bar("vst_vld")
            self.candidate_length_ub.commit(candidate_length_slot)
            mem_copy(
                tile_view(
                    rebase_view(key_out, (workspace_row, token_offset)),
                    (1, TILE_N),
                    (0, 0),
                ),
                self.key_ub,
            )
            if self.candidate_enabled:
                mem_copy(
                    tile_view(
                        rebase_view(
                            candidate_key_out,
                            (
                                workspace_row,
                                token_offset // CANDIDATE_BLOCK_SIZE,
                            ),
                        ),
                        (1, 64 // CANDIDATE_BLOCK_SIZE),
                        (0, 0),
                    ),
                    self.candidate_key_lo_ub,
                )
                mem_copy(
                    tile_view(
                        rebase_view(
                            candidate_key_out,
                            (
                                workspace_row,
                                token_offset // CANDIDATE_BLOCK_SIZE
                                + 64 // CANDIDATE_BLOCK_SIZE,
                            ),
                        ),
                        (1, 64 // CANDIDATE_BLOCK_SIZE),
                        (0, 0),
                    ),
                    self.candidate_key_hi_ub,
                )
            candidate_length_ready = self.candidate_length_ub.wait()
            if local_row < task_span_count:
                mem_copy(
                    tile_view(candidate_length_out, (1,), (public_row,)),
                    candidate_length_ready,
                )
            self.candidate_length_ub.release(candidate_length_ready)


@kernel
class QliFusedKernel:
    def __init__(
        self,
        topk,
        candidate_enabled,
        candidate_topk,
        mask_mode,
        cmp_ratio,
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_cmp_residual_k,
        has_block_table,
    ):
        self.topk_count = int(topk)
        self.candidate_enabled = bool(candidate_enabled)
        self.candidate_topk_count = int(candidate_topk)
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.has_cu_seqlens_q = bool(has_cu_seqlens_q)
        self.has_seqused_q = bool(has_seqused_q)
        self.has_seqused_k = bool(has_seqused_k)
        self.has_cmp_residual_k = bool(has_cmp_residual_k)
        self.has_block_table = bool(has_block_table)
        self.qk_handoff = Channel(
            MemLoc.UB,
            (TILE_M // 2, TILE_N),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
        )
        self.input_stager = QliInputStager(get_subblock_id())
        self.cube = QliCube()
        self.vector = QliVector(
            get_subblock_id(),
            self.mask_mode,
            self.cmp_ratio,
            self.candidate_enabled,
            self.candidate_topk_count,
        )
        self.topk_workspace = QliRawTopKWorkspace(
            TOPK_TRUNK_LEN,
            max(
                self.topk_count,
                self.candidate_topk_count if self.candidate_enabled else 1,
            ),
        )
        self.topk = QliRawTopKSelector(
            self.topk_count, self.topk_workspace
        )
        if self.candidate_enabled:
            self.candidate_topk = QliRawTopKSelector(
                self.candidate_topk_count,
                self.topk_workspace,
            )

    @jit
    def _compute_nonempty_task(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        block_table: Tensor,
        staging_query: Tensor,
        staging_query_fp4: Tensor,
        staging_query_scale: Tensor,
        staging_query_scale_e8m0: Tensor,
        staging_key: Tensor,
        staging_key_fp4: Tensor,
        staging_key_scale: Tensor,
        staging_key_scale_e8m0: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        batch_idx,
        public_row_base,
        query_offset,
        task_span_count,
        task_active_count,
        used_q,
        actual_s2,
        cmp_residual,
        workspace_task_idx,
        tokens,
    ):
        self.input_stager.stage_query(
            query,
            query_scale,
            public_row_base,
            task_span_count,
            staging_query,
            staging_query_scale,
            workspace_task_idx,
        )
        self.input_stager.wait_query_ready()
        self.cube.load_query(
            staging_query_fp4,
            staging_query_scale_e8m0,
            workspace_task_idx * (TILE_M // N1),
            TILE_M // N1,
        )
        token_tiles = tokens // TILE_N
        candidate_tokens = tokens // CANDIDATE_BLOCK_SIZE
        for token_tile in dsl_range(token_tiles, unroll=1):
            logical_start = token_tile * TILE_N
            # PA block_size is fixed to the stable Cube N tile (128). The
            # block-table value is a physical K/K-scale page id; it has no
            # relationship to Candidate's logical groups of eight scores.
            page_in_batch = token_tile
            if dtypes.int32(logical_start) >= actual_s2:
                page_in_batch = 0
            physical_page = dtypes.int64(page_in_batch)
            if self.has_block_table:
                physical_page = dtypes.int64(
                    block_table[batch_idx, page_in_batch]
                )
            # The public PA view may contain padding only between physical
            # pages. Host passes the original element strides; page address
            # selection stays in the fused NPU kernel, matching ASC's
            # block-id * stride0 addressing rule.
            physical_row = (
                physical_page * dtypes.int64(key_stride0)
            ) // KEY_STORAGE_ROW_ELEMENTS
            physical_scale_row = (
                physical_page * dtypes.int64(key_dequant_scale_stride0)
            ) // KEY_SCALE_STORAGE_ROW_ELEMENTS
            self.input_stager.stage_key(
                key,
                key_scale,
                physical_row,
                physical_scale_row,
                token_tile,
                staging_key,
                staging_key_scale,
                workspace_task_idx,
            )
            self.input_stager.wait_key_ready()
            staging_row = (
                workspace_task_idx * STAGING_DEPTH
                + token_tile % STAGING_DEPTH
            ) * TILE_N
            self.cube.compute_qk(
                tile_view(
                    rebase_view(staging_key_fp4, (staging_row, 0)),
                    (TILE_N, LOGICAL_D),
                    (0, 0),
                ),
                tile_view(
                    rebase_view(
                        staging_key_scale_e8m0,
                        (staging_row, 0, 0),
                    ),
                    (TILE_N, 2, 2),
                    (0, 0, 0),
                ),
                self.qk_handoff,
            )
            self.vector.compute_vector1(
                self.qk_handoff,
                weights,
                key_out,
                candidate_key_out,
                public_row_base,
                task_span_count,
                task_active_count,
                workspace_task_idx,
                logical_start,
                actual_s2,
                query_offset,
                used_q,
                cmp_residual,
                candidate_tokens,
                candidate_length,
            )
        if not self.candidate_enabled:
            # Candidate is a compile-time-disabled branch. Write its public
            # zero length once per task in the fused AIV path; do not rely on
            # the per-S2-tile Candidate bookkeeping, which is eliminated.
            self.vector.store_empty_candidate_lengths(
                candidate_length, public_row_base, task_span_count
            )
        # S2 can be 32K, so the complete sortable-key history cannot remain
        # in UB.  Drain the AIV pipelines before TopK reloads the GM history;
        # this is the DSL equivalent of ASC ProcessTopK's MTE3->MTE2 event.
        vec_sync_all()
        # Reuse the verified ASC vf_topk_16 implementation through a genuine
        # JIT boundary.  It emits into this top-level kernel and does not
        # launch a second kernel.
        for row_in_subblock in range_constexpr(2):
            local_row = dtypes.int32(
                get_subblock_id() * 2 + row_in_subblock
            )
            public_row = (
                dtypes.int64(public_row_base) + dtypes.int64(local_row)
            )
            workspace_row = (
                workspace_task_idx * 4
                + get_subblock_id() * 2
                + row_in_subblock
            )
            valid_s2 = actual_s2
            if self.mask_mode == 3:
                query_in_batch = (
                    query_offset
                    + dtypes.int32(get_subblock_id()) * 2
                    + dtypes.int32(row_in_subblock)
                )
                valid_s2 = (
                    actual_s2 * self.cmp_ratio
                    + cmp_residual
                    - used_q
                    + query_in_batch
                    + 1
                ) // self.cmp_ratio
                if valid_s2 < 0:
                    valid_s2 = 0
                if valid_s2 > actual_s2:
                    valid_s2 = actual_s2
            if local_row >= task_active_count:
                valid_s2 = 0
            if local_row < task_span_count:
                self.topk.select_row(
                    tile_view(
                        key_out,
                        (1, MAX_TOKEN_CAPACITY),
                        (workspace_row, 0),
                    ),
                    tokens,
                    valid_s2,
                    tile_view(
                        sparse_indices,
                        (1, self.topk_count),
                        (public_row, 0),
                    ),
                    tile_view(
                        sparse_value_bits,
                        (1, self.topk_count),
                        (public_row, 0),
                    ),
                )
                if self.candidate_enabled:
                    valid_candidate_count = (
                        valid_s2 + CANDIDATE_BLOCK_SIZE - 1
                    ) // CANDIDATE_BLOCK_SIZE
                    self.candidate_topk.select_row(
                        tile_view(
                            candidate_key_out,
                            (
                                1,
                                MAX_TOKEN_CAPACITY
                                // CANDIDATE_BLOCK_SIZE,
                            ),
                            (workspace_row, 0),
                        ),
                        candidate_tokens,
                        valid_candidate_count,
                        tile_view(
                            candidate_indices,
                            (1, self.candidate_topk_count),
                            (public_row, 0),
                        ),
                        tile_view(
                            candidate_value_bits,
                            (1, self.candidate_topk_count),
                            (workspace_row, 0),
                        ),
                    )

    @jit
    def _compute_task(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        block_table: Tensor,
        staging_query: Tensor,
        staging_query_fp4: Tensor,
        staging_query_scale: Tensor,
        staging_query_scale_e8m0: Tensor,
        staging_key: Tensor,
        staging_key_fp4: Tensor,
        staging_key_scale: Tensor,
        staging_key_scale_e8m0: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        batch_idx,
        public_row_base,
        query_offset,
        task_span_count,
        task_active_count,
        used_q,
        actual_s2,
        cmp_residual,
        workspace_task_idx,
        tokens,
    ):
        task_valid_s2 = dtypes.int32(0)
        if task_active_count > 0:
            task_valid_s2 = actual_s2
        if self.mask_mode == 3 and task_active_count > 0:
            query_last = (
                query_offset + task_active_count - 1
            )
            task_valid_s2 = (
                actual_s2 * self.cmp_ratio
                + cmp_residual
                - used_q
                + query_last
                + 1
            ) // self.cmp_ratio
            if task_valid_s2 < 0:
                task_valid_s2 = 0
            if task_valid_s2 > actual_s2:
                task_valid_s2 = actual_s2

        if task_valid_s2 > 0:
            self._compute_nonempty_task(
                query,
                key,
                weights,
                query_scale,
                key_scale,
                key_stride0,
                key_dequant_scale_stride0,
                block_table,
                staging_query,
                staging_query_fp4,
                staging_query_scale,
                staging_query_scale_e8m0,
                staging_key,
                staging_key_fp4,
                staging_key_scale,
                staging_key_scale_e8m0,
                key_out,
                candidate_key_out,
                sparse_indices,
                sparse_value_bits,
                candidate_indices,
                candidate_value_bits,
                candidate_length,
                batch_idx,
                public_row_base,
                query_offset,
                task_span_count,
                task_active_count,
                used_q,
                actual_s2,
                cmp_residual,
                workspace_task_idx,
                tokens,
            )
        else:
            for row_in_subblock in range_constexpr(2):
                local_row = dtypes.int32(
                    get_subblock_id() * 2 + row_in_subblock
                )
                public_row = (
                    dtypes.int64(public_row_base) + dtypes.int64(local_row)
                )
                workspace_row = (
                    workspace_task_idx * 4
                    + get_subblock_id() * 2
                    + row_in_subblock
                )
                if local_row < task_span_count:
                    self.topk.store_empty_row(
                        tile_view(
                            sparse_indices,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                        tile_view(
                            sparse_value_bits,
                            (1, self.topk_count),
                            (public_row, 0),
                        ),
                    )
                    if self.candidate_enabled:
                        self.candidate_topk.store_empty_row(
                            tile_view(
                                candidate_indices,
                                (1, self.candidate_topk_count),
                                (public_row, 0),
                            ),
                            tile_view(
                                candidate_value_bits,
                                (1, self.candidate_topk_count),
                                (workspace_row, 0),
                            ),
                        )
            self.vector.store_empty_candidate_lengths(
                candidate_length, public_row_base, task_span_count
            )

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        weights: Tensor,
        query_scale: Tensor,
        key_scale: Tensor,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        staging_query: Tensor,
        staging_query_fp4: Tensor,
        staging_query_scale: Tensor,
        staging_query_scale_e8m0: Tensor,
        staging_key: Tensor,
        staging_key_fp4: Tensor,
        staging_key_scale: Tensor,
        staging_key_scale_e8m0: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
    ):
        workspace_task_idx = get_block_idx()
        for batch_idx in dsl_range(batch_count, unroll=1):
            query_begin = dtypes.int32(0)
            query_end = dtypes.int32(total_q)
            if self.has_cu_seqlens_q:
                query_begin = dtypes.int32(cu_seqlens_q[batch_idx])
                query_end = dtypes.int32(cu_seqlens_q[batch_idx + 1])
            query_span = query_end - query_begin
            used_q = query_span
            if self.has_seqused_q:
                used_q = dtypes.int32(seqused_q[batch_idx])
            actual_s2 = dtypes.int32(tokens)
            if self.has_seqused_k:
                actual_s2 = dtypes.int32(seqused_k[batch_idx])
            cmp_residual = dtypes.int32(0)
            if self.has_cmp_residual_k:
                cmp_residual = dtypes.int32(cmp_residual_k[batch_idx])

            query_task_count = (
                query_span + (TILE_M // N1) - 1
            ) // (TILE_M // N1)
            for query_task in dsl_range(query_task_count, unroll=1):
                task_owner = (
                    batch_idx * max_tasks_per_batch
                    + dtypes.int64(query_task)
                ) % worker_count
                if task_owner == workspace_task_idx:
                    query_offset = query_task * (TILE_M // N1)
                    task_span_count = query_span - query_offset
                    if task_span_count > TILE_M // N1:
                        task_span_count = TILE_M // N1
                    task_active_count = used_q - query_offset
                    if task_active_count < 0:
                        task_active_count = 0
                    if task_active_count > task_span_count:
                        task_active_count = task_span_count
                    self._compute_task(
                        query,
                        key,
                        weights,
                        query_scale,
                        key_scale,
                        key_stride0,
                        key_dequant_scale_stride0,
                        block_table,
                        staging_query,
                        staging_query_fp4,
                        staging_query_scale,
                        staging_query_scale_e8m0,
                        staging_key,
                        staging_key_fp4,
                        staging_key_scale,
                        staging_key_scale_e8m0,
                        key_out,
                        candidate_key_out,
                        sparse_indices,
                        sparse_value_bits,
                        candidate_indices,
                        candidate_value_bits,
                        candidate_length,
                        batch_idx,
                        query_begin + query_offset,
                        query_offset,
                        task_span_count,
                        task_active_count,
                        used_q,
                        actual_s2,
                        cmp_residual,
                        workspace_task_idx,
                        tokens,
                    )


_COMPILED_KERNEL: dict[tuple[object, ...], object] = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _build_compiled_fused_runner(
    topk,
    candidate_enabled,
    candidate_topk,
    mask_mode,
    cmp_ratio,
    has_cu_seqlens_q,
    has_seqused_q,
    has_seqused_k,
    has_cmp_residual_k,
    has_block_table,
):
    """Compile one launcher whose data-dependent shape axes stay symbolic."""
    topk = int(topk)
    candidate_enabled = bool(candidate_enabled)
    candidate_topk = int(candidate_topk)
    mask_mode = int(mask_mode)
    cmp_ratio = int(cmp_ratio)

    def run_fused_cv(
        query,
        key,
        weights: Tensor,
        query_scale,
        key_scale,
        key_stride0,
        key_dequant_scale_stride0,
        cu_seqlens_q: Tensor,
        seqused_q: Tensor,
        seqused_k: Tensor,
        cmp_residual_k: Tensor,
        block_table: Tensor,
        staging_query: Tensor,
        staging_query_fp4: Tensor,
        staging_query_scale: Tensor,
        staging_query_scale_e8m0: Tensor,
        staging_key: Tensor,
        staging_key_fp4: Tensor,
        staging_key_scale: Tensor,
        staging_key_scale_e8m0: Tensor,
        key_out: Tensor,
        candidate_key_out: Tensor,
        sparse_indices: Tensor,
        sparse_value_bits: Tensor,
        candidate_indices: Tensor,
        candidate_value_bits: Tensor,
        candidate_length: Tensor,
        worker_count,
        total_q,
        batch_count,
        tokens,
        max_tasks_per_batch,
    ):
        QliFusedKernel(
            topk,
            candidate_enabled,
            candidate_topk,
            mask_mode,
            cmp_ratio,
            has_cu_seqlens_q,
            has_seqused_q,
            has_seqused_k,
            has_cmp_residual_k,
            has_block_table,
        )[worker_count](
            query,
            key,
            weights,
            query_scale,
            key_scale,
            key_stride0,
            key_dequant_scale_stride0,
            cu_seqlens_q,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            staging_query,
            staging_query_fp4,
            staging_query_scale,
            staging_query_scale_e8m0,
            staging_key,
            staging_key_fp4,
            staging_key_scale,
            staging_key_scale_e8m0,
            key_out,
            candidate_key_out,
            sparse_indices,
            sparse_value_bits,
            candidate_indices,
            candidate_value_bits,
            candidate_length,
            worker_count,
            total_q,
            batch_count,
            tokens,
            max_tasks_per_batch,
        )

    query_rows_dim = cannbotdsl.Dim("T1")
    key_storage_rows_dim = cannbotdsl.Dim("K_ROWS")
    key_scale_storage_rows_dim = cannbotdsl.Dim("KS_ROWS")
    batch_dim = cannbotdsl.Dim("B")
    block_table_width_dim = cannbotdsl.Dim("MAX_BLOCKS")
    workspace_rows = MAX_CUBE_WORKERS * (TILE_M // N1)
    fake = TensorSpec
    dummy_i32_spec = fake((1,), dtypes.int32)
    cu_spec = (
        fake((batch_dim + 1,), dtypes.int32)
        if has_cu_seqlens_q
        else dummy_i32_spec
    )
    seqused_q_spec = (
        fake((batch_dim,), dtypes.int32)
        if has_seqused_q
        else dummy_i32_spec
    )
    seqused_k_spec = (
        fake((batch_dim,), dtypes.int32)
        if has_seqused_k
        else dummy_i32_spec
    )
    residual_spec = (
        fake((batch_dim,), dtypes.int32)
        if has_cmp_residual_k
        else dummy_i32_spec
    )
    block_table_spec = (
        fake((batch_dim, block_table_width_dim), dtypes.int32)
        if has_block_table
        else fake((1, 1), dtypes.int32)
    )
    candidate_key_spec = (
        fake(
            (
                workspace_rows,
                MAX_TOKEN_CAPACITY // CANDIDATE_BLOCK_SIZE,
            ),
            dtypes.uint16,
        )
        if candidate_enabled
        else fake((workspace_rows, 1), dtypes.uint16)
    )
    candidate_index_spec = (
        fake((query_rows_dim, candidate_topk), dtypes.int32)
        if candidate_enabled
        else fake((1, 1), dtypes.int32)
    )

    return jit(run_fused_cv).compile(
        fake(
            (query_rows_dim * N1, PACKED_D), dtypes.uint8
        ),
        fake(
            (key_storage_rows_dim, PACKED_D), dtypes.uint8
        ),
        fake((query_rows_dim, N1), dtypes.float32),
        fake(
            (query_rows_dim * N1, 4), dtypes.uint8
        ),
        fake(
            (key_scale_storage_rows_dim, 4), dtypes.uint8
        ),
        dtypes.int64,
        dtypes.int64,
        cu_spec,
        seqused_q_spec,
        seqused_k_spec,
        residual_spec,
        block_table_spec,
        fake(
            (MAX_CUBE_WORKERS * TILE_M, PACKED_D), dtypes.uint8
        ),
        fake(
            (MAX_CUBE_WORKERS * TILE_M, LOGICAL_D),
            dtypes.fp4x2_e2m1,
        ),
        fake((MAX_CUBE_WORKERS * TILE_M, 4), dtypes.uint8),
        fake(
            (MAX_CUBE_WORKERS * TILE_M, 2, 2),
            dtypes.float8_e8m0,
        ),
        fake(
            (
                MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N,
                PACKED_D,
            ),
            dtypes.uint8,
        ),
        fake(
            (
                MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N,
                LOGICAL_D,
            ),
            dtypes.fp4x2_e2m1,
        ),
        fake(
            (
                MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N,
                4,
            ),
            dtypes.uint8,
        ),
        fake(
            (
                MAX_CUBE_WORKERS * STAGING_DEPTH * TILE_N,
                2,
                2,
            ),
            dtypes.float8_e8m0,
        ),
        fake((workspace_rows, MAX_TOKEN_CAPACITY), dtypes.uint16),
        candidate_key_spec,
        fake((query_rows_dim, topk), dtypes.int32),
        fake((query_rows_dim, topk), dtypes.uint16),
        candidate_index_spec,
        fake((workspace_rows, candidate_topk), dtypes.uint16),
        fake((query_rows_dim,), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )


def _get_compiled_fused_runner(*config):
    """Compile once per constexpr QLI branch, independently of B/T/S2."""
    cache_key = tuple(config)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNEL.get(cache_key)
        if compiled is None:
            compiled = _build_compiled_fused_runner(*config)
            _COMPILED_KERNEL[cache_key] = compiled
        return compiled


# ============================================================================
# 3. Torch interface
# ============================================================================


def quant_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    topk: int,
    quant_mode: int = 1,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    return_value: bool = False,
    candidate_topk_blocks: int = -1,
    candidate_block_size: int = -1,
) -> dict[str, torch.Tensor]:
    """Run the xlsx QLI MXFP4 contract on NPU tensors.

    ``cu_seqlens_q`` maps TND query rows to batches, ``block_table`` maps each
    batch's logical K pages to physical PA_BBND blocks, and ``seqused_k[b]``
    gives that batch's direct (non-cumulative) usable S2 length. Internally,
    up to four query rows from one batch form one M=128 SplitCore task; every
    such task keeps its full S2 traversal on one Cube core (LD/S2 split off).
    """
    # Keep internal terminology descriptive while exposing the exact xlsx
    # argument names at the public Python boundary.
    query, key, weights = q, k, w
    query_dequant_scale, key_dequant_scale = q_descale, k_descale
    if int(quant_mode) != 1:
        raise ValueError("quant_lightning_indexer public contract supports quant_mode=1 only")
    if layout_q != "TND":
        raise ValueError("layout_q must be TND under the current xlsx contract")
    if layout_kv != "PA_BBND":
        raise ValueError("layout_kv must be PA_BBND under the current xlsx contract")
    if query.device.type != "npu" or key.device.type != "npu":
        raise ValueError("quant_lightning_indexer requires NPU q/k tensors")
    if query.dtype != torch.uint8 or key.dtype != torch.uint8:
        raise TypeError("public q/k storage must be uint8 packed MXFP4")
    if query.ndim != 3 or key.ndim != 4:
        raise ValueError("q must be [T1,32,64], k must be [block_num,block_size,1,64]")
    if query.shape[1] != N1 or key.shape[2] != N2 or query.shape[2] != PACKED_D or key.shape[3] != PACKED_D:
        raise ValueError("public static axes require N1=32, N2=1 and logical D=128")
    if tuple(weights.shape) != tuple(query.shape[:2]) or weights.dtype != torch.float32:
        raise ValueError("w must be FP32 with shape (T1,N1)")
    if query_dequant_scale is None:
        raise ValueError("q_descale is a required xlsx input")
    expected_q_scale = (*query.shape[:2], LOGICAL_D // 64, 2)
    if tuple(query_dequant_scale.shape) != expected_q_scale:
        raise ValueError(
            "q_descale must have corrected xlsx shape "
            "(T1,N1,D/64,2)"
        )
    if key_dequant_scale is None:
        raise ValueError("k_descale is a required xlsx input")
    expected = tuple(key.shape[:3]) + (2, 2)
    if tuple(key_dequant_scale.shape) != expected:
        raise ValueError("k_descale must have xlsx shape (block_num,block_size,N2,2,2)")
    key_stride0, key_storage_rows = _pa_axis0_storage_rows(
        key,
        "k",
        (PACKED_D, PACKED_D, 1),
        KEY_STORAGE_ROW_ELEMENTS,
    )
    (
        key_dequant_scale_stride0,
        key_dequant_scale_storage_rows,
    ) = _pa_axis0_storage_rows(
        key_dequant_scale,
        "k_descale",
        (KEY_SCALE_STORAGE_ROW_ELEMENTS, 2 * 2, 2, 1),
        KEY_SCALE_STORAGE_ROW_ELEMENTS,
    )
    # CANNBotDSL's packed-FP4 host carrier is compact-only. Expose the
    # allocation span, including axis-0 gaps, as a zero-copy compact carrier;
    # the original page strides are passed separately and consumed on NPU.
    key_storage = torch.as_strided(
        key,
        (key_storage_rows, PACKED_D),
        (KEY_STORAGE_ROW_ELEMENTS, 1),
    )
    key_dequant_scale_storage = torch.as_strided(
        key_dequant_scale,
        (key_dequant_scale_storage_rows, 4),
        (KEY_SCALE_STORAGE_ROW_ELEMENTS, 1),
    )
    if metadata is not None:
        raise ValueError("metadata is reserved by the current xlsx constraint and must be None")
    if candidate_topk_blocks == -1:
        if candidate_block_size != -1:
            raise ValueError("candidate_block_size must be -1 when candidate is disabled")
    elif candidate_topk_blocks <= 0 or candidate_block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError("candidate requires positive topk_blocks and block_size=8")
    # Keep this as a Python/JIT specialization constant. Disabled launches
    # compile out Candidate block-max, key generation, TopK and UB resources.
    candidate_enabled = int(candidate_topk_blocks) > 0
    if int(key.shape[1]) != TILE_N:
        raise ValueError(
            "current fused PA implementation requires k block_size=128"
        )
    if int(topk) < 1 or int(topk) > 8192:
        raise ValueError("topk must be in ASC-supported range [1,8192]")
    if int(mask_mode) not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3")
    if int(cmp_ratio) < 1 or int(cmp_ratio) > 128:
        raise ValueError("cmp_ratio must be in [1,128]")
    if query_dequant_scale.dtype != torch.uint8:
        raise TypeError("q_descale must use uint8 E8M0 storage")
    if key_dequant_scale.dtype != torch.uint8:
        raise TypeError("k_descale must use uint8 E8M0 storage")
    for name, tensor in (
        ("w", weights),
        ("q_descale", query_dequant_scale),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not query.is_contiguous():
        raise ValueError("q must be contiguous")

    t = int(query.shape[0])
    if t <= 0:
        raise ValueError("q must contain at least one T1 row")
    has_cu_seqlens_q = cu_seqlens_q is not None
    batch = int(cu_seqlens_q.numel()) - 1 if has_cu_seqlens_q else 1
    if batch <= 0:
        raise ValueError("cu_seqlens_q must have shape (B+1,)")

    def validate_metadata(name, tensor, shape):
        if tensor is None:
            return
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the same NPU as q")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if has_cu_seqlens_q:
        validate_metadata("cu_seqlens_q", cu_seqlens_q, (batch + 1,))
    validate_metadata("seqused_q", seqused_q, (batch,))
    validate_metadata("seqused_k", seqused_k, (batch,))
    if int(mask_mode) == 3 and int(cmp_ratio) != 1:
        if cmp_residual_k is None:
            raise ValueError(
                "cmp_residual_k is required for mask_mode=3 and cmp_ratio!=1"
            )
        validate_metadata("cmp_residual_k", cmp_residual_k, (batch,))
    elif cmp_residual_k is not None:
        raise ValueError(
            "cmp_residual_k must be None unless mask_mode=3 and cmp_ratio!=1"
        )
    if block_table is not None:
        if block_table.ndim != 2 or int(block_table.shape[0]) != batch:
            raise ValueError(
                "block_table must have shape (B,max_num_blocks_per_seq)"
            )
        if block_table.dtype != torch.int32:
            raise TypeError("block_table must be int32")
        if block_table.device != query.device:
            raise ValueError("block_table must be on the same NPU as q")
        if not block_table.is_contiguous():
            raise ValueError("block_table must be contiguous")

    has_seqused_q = seqused_q is not None
    has_seqused_k = seqused_k is not None
    has_cmp_residual_k = cmp_residual_k is not None
    has_block_table = block_table is not None
    tokens_pad = (
        int(block_table.shape[1]) * TILE_N
        if has_block_table
        else int(key.shape[0]) * TILE_N
    )
    if tokens_pad <= 0:
        raise ValueError("K storage must expose at least one 128-token PA page")
    if tokens_pad > MAX_TOKEN_CAPACITY:
        raise ValueError(
            "current dynamic QLI kernel supports S2 capacity up to 128K"
        )

    # Optional xlsx tensors remain optional at the public boundary.  One
    # uninitialized carrier is sufficient because the JIT specialization
    # compiles out every load when the corresponding tensor is absent.
    dummy_i32 = query.new_empty((1,), dtype=torch.int32)
    dummy_table = query.new_empty((1, 1), dtype=torch.int32)
    cu_kernel = cu_seqlens_q if has_cu_seqlens_q else dummy_i32
    seqused_q_kernel = seqused_q if has_seqused_q else dummy_i32
    seqused_k_kernel = seqused_k if has_seqused_k else dummy_i32
    cmp_residual_kernel = (
        cmp_residual_k if has_cmp_residual_k else dummy_i32
    )
    block_table_kernel = block_table if has_block_table else dummy_table

    # Public outputs and GM scratch are allocation-only Torch calls.  The
    # fused kernel writes every public row, including invalid/padded rows.
    sparse_indices_2d = query.new_empty((t, int(topk)), dtype=torch.int32)
    sparse_values_2d = weights.new_empty(
        (t, int(topk)), dtype=torch.bfloat16
    )
    candidate_count = max(0, int(candidate_topk_blocks))
    candidate_indices_2d = query.new_empty(
        (t, candidate_count), dtype=torch.int32
    )
    candidate_length_1d = query.new_empty((t,), dtype=torch.int32)

    max_query_length = int(max_seqlen_q) if int(max_seqlen_q) > 0 else t
    # Packed fp4 host carriers require static shapes in the current DSL ABI.
    # Launch the ASC-sized 32-core grid for every dynamic shape; idle workers
    # simply own no task. This keeps B/T/S2 out of the compilation key.
    worker_count = MAX_CUBE_WORKERS
    workspace_row_count = worker_count * (TILE_M // N1)
    staging_query = query.new_empty(
        (worker_count * TILE_M, PACKED_D)
    )
    staging_query_scale = query.new_empty(
        (worker_count * TILE_M, 4)
    )
    staging_key = query.new_empty(
        (worker_count * STAGING_DEPTH * TILE_N, PACKED_D)
    )
    staging_key_scale = query.new_empty(
        (worker_count * STAGING_DEPTH * TILE_N, 4)
    )
    key_workspace = query.new_empty(
        (workspace_row_count, MAX_TOKEN_CAPACITY), dtype=torch.uint16
    )
    candidate_key_workspace = query.new_empty(
        (
            workspace_row_count,
            # The unified launcher keeps a Tensor argument in both variants;
            # the disabled variant never accesses this one-element carrier.
            (
                MAX_TOKEN_CAPACITY // CANDIDATE_BLOCK_SIZE
                if candidate_enabled
                else 1
            ),
        ),
        dtype=torch.uint16,
    )
    kernel_candidate_count = candidate_count if candidate_enabled else 1
    candidate_indices_kernel = (
        candidate_indices_2d
        if candidate_enabled
        else query.new_empty((1, 1), dtype=torch.int32)
    )
    candidate_value_workspace = query.new_empty(
        (workspace_row_count, kernel_candidate_count), dtype=torch.uint16
    )

    # Public Q/K and scales stay uint8. Fixed staging tensors have alias views
    # for the native fp4/e8m0 AIC contract; these are metadata-only views and
    # launch no Torch/NPU compute kernels.
    query_storage = query.reshape(t * N1, PACKED_D)
    query_scale_storage = query_dequant_scale.reshape(t * N1, 4)

    max_tasks_per_batch = ceil_div(
        max_query_length, TILE_M // N1
    )
    _get_compiled_fused_runner(
        int(topk),
        candidate_enabled,
        kernel_candidate_count,
        int(mask_mode),
        int(cmp_ratio),
        has_cu_seqlens_q,
        has_seqused_q,
        has_seqused_k,
        has_cmp_residual_k,
        has_block_table,
    )(
        query_storage,
        key_storage,
        weights,
        query_scale_storage,
        key_dequant_scale_storage,
        key_stride0,
        key_dequant_scale_stride0,
        cu_kernel,
        seqused_q_kernel,
        seqused_k_kernel,
        cmp_residual_kernel,
        block_table_kernel,
        staging_query,
        staging_query.view(torch.int8),
        staging_query_scale,
        staging_query_scale.reshape(-1, 2, 2).view(
            torch.float8_e8m0fnu
        ),
        staging_key,
        staging_key.view(torch.int8),
        staging_key_scale,
        staging_key_scale.reshape(-1, 2, 2).view(
            torch.float8_e8m0fnu
        ),
        key_workspace,
        candidate_key_workspace,
        sparse_indices_2d,
        sparse_values_2d.view(torch.uint16),
        candidate_indices_kernel,
        candidate_value_workspace,
        candidate_length_1d,
        worker_count,
        t,
        batch,
        tokens_pad,
        max_tasks_per_batch,
    )

    return {
        "sparse_indices": sparse_indices_2d.view(t, N2, int(topk)),
        "sparse_values": sparse_values_2d.view(t, N2, int(topk)),
        "candidate_block_indices": candidate_indices_2d.view(
            t, N2, candidate_count
        ),
        "candidate_block_length": candidate_length_1d.view(t, N2),
    }


__all__ = ["quant_lightning_indexer"]
