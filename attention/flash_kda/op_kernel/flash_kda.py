# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import dataclasses
import os
import threading
from typing import NamedTuple, Optional, Tuple

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import cannbotdsl
import torch
from cannbotdsl import dtypes
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.arena import _current_channel_arena, channel_rewind
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.types.delay_line import DelayLineGroup
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    mask_xor,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    vexp,
    vexp_sub,
    vload,
    vload_broadcast,
    vload_unpack,
    vmax,
    vmem_bar,
    vmin,
    vmins,
    vmul,
    vmuls,
    vreduce_max,
    vreduce_min,
    vreduce_sum,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
    vsqrt,
)
from cannbotdsl.ops.reg import vselect as vselect_raw
from cannbotdsl.ops.sync import (
    cube_fill_l1_zero,
    cube_raw_l1_to_l0a,
    cube_raw_l1_to_l0b,
    cube_sync_all,
    cube_sync_pipe,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import (
    idx2crd,
    partition_view,
    local_slice,
    tile_view,
    make_layout,
    make_partition_tiler,
)
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl import ChannelKind, MemLoc, PIPE, Tensor
from cannbotdsl.lang import vf

CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = 128
HALF_CHUNK_SIZE = CHUNK_SIZE // 2
D_HALF = SUPPORTED_HEAD_DIM // 2
LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32
# 及时释放无复用价值的数据，把 L2 留给真正有局部性的访问
_L2_CACHE_LAST_USE = 1
_L2_CACHE_DISABLE = 4


@jit
def _prefix_sum_rows(dst, src):
    # Equivalent to the retired cumsum(axis=0): keep sequential FP32 additions.
    rows, cols = dst.shape
    with vf(mode="raw"):
        for col in range(0, cols, 64):
            mask, _ = update_mask(cols - col, elem_bits=32)
            prefix = vload(src, col)
            vstore(dst, col, prefix, mask)
            for row in range(1, rows):
                prefix = vadd(prefix, vload(src, row * src.physical_stride[0] + col), mask=mask)
                vstore(dst, row * dst.physical_stride[0] + col, prefix, mask)


@jit
def _cast_state_bf16(dst, src):
    # The original state cast is FP32 -> BF16; preserve both physical strides.
    rows, cols = dst.shape
    with vf(mode="raw"):
        for row in range(rows):
            for col in range(0, cols, 64):
                mask, _ = update_mask(cols - col, elem_bits=32)
                value = vload(src, row * src.physical_stride[0] + col)
                converted = vcast(value, dtypes.bfloat16, mask=mask)
                vstore_pack(dst, row * dst.physical_stride[0] + col, converted, mask,
                            pack_mode=PackMode.B32_TO_B16)


def _valid_rows_view(full_tile, valid_rows, cols: int):
    return full_tile[:valid_rows, :cols]


def _o_head_chunk_tile(gm_O, seq_idx, chunk_idx, dv_base, dv_idx, valid_rows):
    nv = gm_O.shape[1]
    batch_idx = seq_idx // nv
    head_idx = seq_idx - batch_idx * nv
    full_chunk = tile_view(gm_O[batch_idx, head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
        (chunk_idx, 0))
    column_tile = tile_view(full_chunk, (CHUNK_SIZE, dv_base), (0, dv_idx))
    return _valid_rows_view(column_tile, valid_rows, dv_base)


class _Stage1Constants(NamedTuple):
    lower_mask_tiles: torch.Tensor
    identity16: torch.Tensor


_STAGE1_CONSTANTS_CACHE: dict[tuple[str, int | None, torch.dtype, torch.dtype], _Stage1Constants] = {}


@dataclasses.dataclass
class StageOneWorkspace:
    Q_decayed: torch.Tensor
    Mqk: torch.Tensor
    K_restored: torch.Tensor
    gamma_C: torch.Tensor
    U_pre: torch.Tensor
    W: torch.Tensor
    identity16: torch.Tensor
    lower_mask_tiles: torch.Tensor


def _empty_like_device(
    shape: tuple[int, ...],
    *,
    ref: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=ref.device)


def allocate_qk_exchange_workspace(*, active_block_num: int, ref: torch.Tensor) -> torch.Tensor:
    assert active_block_num > 0, "active_block_num must be positive"
    return _empty_like_device((active_block_num, 2, CHUNK_SIZE, D_HALF), ref=ref, dtype=HIGH_DTYPE)


def qk_exchange_slices(workspace: torch.Tensor, *, block_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert workspace.dtype == HIGH_DTYPE
    assert workspace.dim() == 4 and tuple(workspace.shape[1:]) == (2, CHUNK_SIZE, D_HALF)
    assert 0 <= block_idx < workspace.shape[0], "block_idx is outside the exchange workspace"
    return workspace[block_idx, 0], workspace[block_idx, 1]


def _resolved_device_key(ref: torch.Tensor) -> tuple[str, int | None]:
    device = ref.device
    device_type = getattr(device, "type", "cpu")
    device_index = getattr(device, "index", None)
    if device_type in {"npu", "privateuseone"} and device_index is None:
        npu = getattr(torch, "npu", None)
        current_device = getattr(npu, "current_device", None) if npu is not None else None
        if current_device is not None:
            device_index = int(current_device())
    return device_type, device_index


def _stage1_constants_for(
    ref: torch.Tensor,
    *,
    mask_dtype: torch.dtype = HIGH_DTYPE,
    identity_dtype: torch.dtype = torch.float16,
) -> _Stage1Constants:
    device_type, device_index = _resolved_device_key(ref)
    key = (device_type, device_index, mask_dtype, identity_dtype)
    cached = _STAGE1_CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached

    mask_cols = CHUNK_SIZE // 4
    strict_zero = torch.triu(torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool, device=ref.device), diagonal=0)
    casual_zero = torch.triu(torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.bool, device=ref.device), diagonal=1)
    strict_bytemask = strict_zero.contiguous().view(torch.float32)
    casual_bytemask = casual_zero.contiguous().view(torch.float32)
    lower_mask_tiles = torch.empty((CHUNK_SIZE, mask_cols * 2), dtype=torch.float32, device=ref.device)
    lower_mask_tiles[:, :mask_cols] = strict_bytemask
    lower_mask_tiles[:, mask_cols:] = casual_bytemask
    constants = _Stage1Constants(lower_mask_tiles=lower_mask_tiles.contiguous(),
        identity16=torch.eye(16, dtype=identity_dtype, device=ref.device).contiguous())
    _STAGE1_CONSTANTS_CACHE[key] = constants
    return constants


def allocate_stage1_workspace(
    batch: int,
    heads: int,
    eff_group: int,
    dim: int,
    *,
    ref: torch.Tensor,
) -> StageOneWorkspace:
    # 临时工作区按 chunk group 复用，不随完整序列长度增长。
    bn_chunks = batch * heads * eff_group
    constants = _stage1_constants_for(ref)
    return StageOneWorkspace(Q_decayed=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        Mqk=_empty_like_device((bn_chunks * CHUNK_SIZE, CHUNK_SIZE), ref=ref, dtype=LOW_DTYPE),
        K_restored=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        gamma_C=_empty_like_device((bn_chunks * dim, 1), ref=ref, dtype=HIGH_DTYPE),
        U_pre=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        W=_empty_like_device((bn_chunks * CHUNK_SIZE, dim), ref=ref, dtype=LOW_DTYPE),
        identity16=constants.identity16, lower_mask_tiles=constants.lower_mask_tiles)


ELEM_BYTES = 4
VL = 64  # 一个 256B 向量寄存器的 f32/b32 lane 数（raw VF 步长）；此处恰等于 CHUNK_SIZE，一行 = 一寄存器

NEUMANN_BLOCK_SIZE = 16
NEUMANN_DIAG_BLOCK_NUM = 4
NEUMANN_FRACTAL_ELEMS = 256                                                     # 一个 16x16 块的元素数。
NEUMANN_DIAG_SRC_STRIDE = NEUMANN_BLOCK_SIZE * CHUNK_SIZE + NEUMANN_BLOCK_SIZE  # ND 源矩阵相邻对角块起点间隔。
NEUMANN_DIAG_DST_STRIDE = (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS  # packed/NZ 目标相邻对角块起点间隔。

class StageOneMatmul:
    def __init__(self):
        # Neumann packed64 求逆工作区。
        self.tile_power_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.packed_inv64_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.tile_inv_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.scratch_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.neumann_power_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1, kind=ChannelKind.CrossCore)
        self.packed_identity_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1)
        self.zero_l1 = Buffer(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16)
        self.resident_inv_l1 = Channel(MemLoc.L1, (CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1)
        # 主矩阵乘使用双缓冲，D 轴 16×16 矩阵乘复用独立 L0 槽。
        self.l0a_db = Channel(MemLoc.L0A, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.l0b_db = Channel(MemLoc.L0B, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.l0c_db = Channel(MemLoc.L0C, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.float32, depth=2)
        self.d_axis_left_k_l0a = Channel(MemLoc.L0A, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.d_axis_left_q_l0a = Channel(MemLoc.L0A, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.d_axis_right_k_l0b = Channel(MemLoc.L0B, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=2)
        self.raw_l0a_db = Channel(MemLoc.L0A, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=2)
        self.raw_l0b_db = Channel(MemLoc.L0B, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=2)
        self.raw_l0c_main = Channel(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1)
        self.raw_l0c_aux = Channel(MemLoc.L0C, (CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1)
        self.gm2l1_identity16_diag = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16,
            pad_value=0.0, nd_num=NEUMANN_DIAG_BLOCK_NUM, src_nd_stride=0,
            dst_nd_stride=NEUMANN_DIAG_DST_STRIDE, dst_c0_stride=CHUNK_SIZE)
        self.packed64_diag_l12l0 = make_copy_engine(dtype=dtypes.float16, src_c0_stride=NEUMANN_DIAG_BLOCK_NUM + 1,
            dst_c0_stride=NEUMANN_DIAG_BLOCK_NUM + 1)
        self.fixpipe_engine = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)
        self.fixpipe_l0c2packed_l1 = make_copy_engine(dtype=dtypes.float32, unit_flag_mode=3)

    def load_k_decayed(self, k_decayed_l1):
        mem_copy(self.l0a_db, k_decayed_l1)

    def load_q_decayed(self, q_decayed_l1):
        mem_copy(self.l0a_db, q_decayed_l1)

    def load_k_inv(self, k_inv_l1):
        mem_copy(self.l0b_db, k_inv_l1)

    def matmul_kkt(self):
        # KKT = K_decayed @ K_inv^T；K_inv 的 L0B 槽留给 Mqk 复用。
        matmul(local_slice(self.l0c_db, (CHUNK_SIZE, CHUNK_SIZE), offset=0), self.l0a_db, self.l0b_db, init=True)
        return self.l0b_db

    def matmul_mqk(self, k_inv_l0b):
        # Mqk = Q_decayed @ K_inv^T，复用 KKT 留下的 K_inv L0B 槽。
        matmul(local_slice(self.l0c_db, (CHUNK_SIZE, CHUNK_SIZE), offset=0), self.l0a_db, k_inv_l0b, init=True)

    def load_d_axis_left_k(self, l1_channel):
        mem_copy(self.d_axis_left_k_l0a, l1_channel)

    def load_d_axis_left_q(self, l1_channel):
        mem_copy(self.d_axis_left_q_l0a, l1_channel)

    def load_d_axis_right_k(self, l1_channel):
        mem_copy(self.d_axis_right_k_l0b, l1_channel)

    def matmul_d_axis_kkt(self, row_tile: int, col_tile: int):
        kkt_tile = tile_view(self.raw_l0c_main, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            (row_tile, col_tile))
        matmul(kkt_tile, self.d_axis_left_k_l0a, self.d_axis_right_k_l0b, init=True)

    def matmul_d_axis_mqk(self, row_tile: int, col_tile: int):
        mqk_tile = tile_view(self.raw_l0c_aux, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            (row_tile, col_tile))
        matmul(mqk_tile, self.d_axis_left_q_l0a, self.d_axis_right_k_l0b, init=True)

    def store_d_axis_kkt_to_ub(self, ub_kkt):
        mem_copy(local_slice(ub_kkt, (HALF_CHUNK_SIZE, CHUNK_SIZE), offset=0),
            local_slice(self.raw_l0c_main, (CHUNK_SIZE, CHUNK_SIZE), offset=0),
            engine=self.fixpipe_engine)

    def store_d_axis_mqk_to_ub(self, ub_mqk):
        mem_copy(local_slice(ub_mqk, (HALF_CHUNK_SIZE, CHUNK_SIZE), offset=0),
            local_slice(self.raw_l0c_aux, (CHUNK_SIZE, CHUNK_SIZE), offset=0),
            engine=self.fixpipe_engine)

    def store_l0c_to_gm(self, gm_tile, *, shape=(CHUNK_SIZE, CHUNK_SIZE)):
        mem_copy(gm_tile, local_slice(self.l0c_db, shape, offset=0))

    def packed64_tile(self, l1_matrix, row_tile: int, col_tile: int):
        # 64×64 NZ 紧凑矩阵中的逻辑 16×16 tile 按分形偏移定位。
        offset_elems = col_tile * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
        offset_elems += row_tile * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
        tile = local_slice(l1_matrix, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            stride=(CHUNK_SIZE, 1), offset=offset_elems * 2)
        return tile, l1_matrix, row_tile, col_tile

    def _packed64_tile_offset_elems(self, row_tile: int, col_tile: int):
        return col_tile * CHUNK_SIZE * NEUMANN_BLOCK_SIZE + row_tile * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE

    def l0_tile_offset_elems(self, row_tile: int, col_tile: int):
        return (row_tile * NEUMANN_DIAG_BLOCK_NUM + col_tile) * NEUMANN_FRACTAL_ELEMS

    def c64_diag_nz_offset_elems(self, diag_tile: int):
        return diag_tile * (NEUMANN_DIAG_BLOCK_NUM + 1) * NEUMANN_FRACTAL_ELEMS

    def c64_odd_even_pair_dst_gap_fractals(self):
        return (self.c64_diag_nz_offset_elems(2) // NEUMANN_FRACTAL_ELEMS) - 1

    def _raw_l0a_full(self, l0a_slot):
        return local_slice(l0a_slot, (CHUNK_SIZE, CHUNK_SIZE), offset=0)

    def _raw_l0b_full(self, l0b_slot):
        return local_slice(l0b_slot, (CHUNK_SIZE, CHUNK_SIZE), offset=0)

    def _raw_l0a_tile(self, l0a_slot, row_tile: int, col_tile: int):
        return local_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            offset=self.l0_tile_offset_elems(row_tile, col_tile) * 2)

    def _raw_l0b_tile(self, l0b_slot, row_tile: int, col_tile: int):
        return local_slice(l0b_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE),
            offset=self.l0_tile_offset_elems(row_tile, col_tile) * 2)

    def _load_l1_tile_to_l0a(self, l1_tile, l0a_slot, row_tile: int, col_tile: int):
        # 将一个 packed L1 16×16 分形装入 full64 L0A 的指定 tile。
        _, l1_parent, _, _ = l1_tile
        l0_tile = self._raw_l0a_tile(l0a_slot, row_tile, col_tile)
        l1_offset = col_tile * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
        l1_offset += row_tile * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
        cube_raw_l1_to_l0a(l0_tile, l1_parent, l1_offset_elems=l1_offset, m_start=0, k_start=0,
            m_step=1, k_step=1, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=1, transpose=False)

    def _load_l1_tile_to_l0b(self, l1_tile, l0b_slot, row_tile: int, col_tile: int):
        # 将一个 packed L1 16×16 分形转置装入 full64 L0B 的指定 tile。
        _, l1_parent, _, _ = l1_tile
        l0_tile = self._raw_l0b_tile(l0b_slot, row_tile, col_tile)
        l1_offset = col_tile * CHUNK_SIZE * NEUMANN_BLOCK_SIZE
        l1_offset += row_tile * NEUMANN_BLOCK_SIZE * NEUMANN_BLOCK_SIZE
        cube_raw_l1_to_l0b(l0_tile, l1_parent, l1_offset_elems=l1_offset, m_start=0, k_start=0,
            m_step=1, k_step=1, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=1, transpose=True)

    def _load_packed32_tile_to_full64_l0b(self, l1_matrix, l0b_slot, *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = True,
    ):
        src_stride = (src_packed_size + NEUMANN_BLOCK_SIZE - 1) // NEUMANN_BLOCK_SIZE
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_tile_offset_elems((dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE)
            l0_tile = local_slice(l0b_slot, (NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l0_offset * 2)
            cube_raw_l1_to_l0b(l0_tile, l1_matrix, l1_offset_elems=l1_offset, m_start=0, k_start=0,
                m_step=1, k_step=2, src_stride=src_stride, dst_stride=1, transpose=transpose)

    def _load_packed32_tile_to_full64_l0a(self, l1_matrix, l0a_slot, *,
        src_row: int,
        src_col: int,
        dst_row: int,
        dst_col: int,
        src_packed_size: int = CHUNK_SIZE,
        transpose: bool = False,
    ):
        # 将 packed32 L1 tile 装入 full64 L0A。
        src_stride = (src_packed_size + NEUMANN_BLOCK_SIZE - 1) // NEUMANN_BLOCK_SIZE
        for row_offset in (0, NEUMANN_BLOCK_SIZE):
            l1_offset = (src_col // NEUMANN_BLOCK_SIZE) * src_packed_size * NEUMANN_BLOCK_SIZE
            l1_offset += (src_row + row_offset) * NEUMANN_BLOCK_SIZE + (src_col % NEUMANN_BLOCK_SIZE)
            l0_offset = self.l0_tile_offset_elems((dst_row + row_offset) // NEUMANN_BLOCK_SIZE, dst_col // NEUMANN_BLOCK_SIZE)
            l0_offset -= 6 * NEUMANN_FRACTAL_ELEMS
            l0_tile = local_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, HALF_CHUNK_SIZE), offset=l0_offset * 2)
            cube_raw_l1_to_l0a(l0_tile, l1_matrix, l1_offset_elems=l1_offset, m_start=0, k_start=0,
                m_step=1, k_step=2, src_stride=src_stride, dst_stride=1, transpose=transpose)

    def _load_packed16_pair_to_full64_l0a_with_gap(self, l1_matrix, l0a_slot, *,
        src0_row_tile: int,
        src0_col_tile: int,
        src1_row_tile: int,
        src1_col_tile: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        # 将一对 packed16 L1 tile 装入 full64 L0A，目标 tile 之间保留间隔。
        dst0_tile = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_tile = dst0_tile + dst_gap_fractals + 1
        for src_row_tile, src_col_tile, dst_tile in (
            (src0_row_tile, src0_col_tile, dst0_tile),
            (src1_row_tile, src1_col_tile, dst1_tile),
        ):
            src_offset = self._packed64_tile_offset_elems(src_row_tile, src_col_tile)
            # 3510 L0A 的 16×16 tile 地址需要补偿一个 full64 M 跨度。
            dst_offset = (dst_tile - (NEUMANN_DIAG_BLOCK_NUM - 1)) * NEUMANN_FRACTAL_ELEMS
            l0_tile = local_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, NEUMANN_BLOCK_SIZE), offset=dst_offset * 2)
            cube_raw_l1_to_l0a(l0_tile, l1_matrix, l1_offset_elems=src_offset, m_start=0, k_start=0,
                m_step=1, k_step=1, src_stride=NEUMANN_DIAG_BLOCK_NUM, dst_stride=1, transpose=False)

    def _load_packed16_pair_to_full64_l0b_trans_with_gap(self, l1_matrix, l0b_slot, *,
        src0_row_tile: int,
        src0_col_tile: int,
        src1_row_tile: int,
        src1_col_tile: int,
        dst_offset_elems: int,
        dst_gap_fractals: int,
    ):
        # 将一对 packed16 L1 tile 转置装入 full64 L0B，目标 tile 之间保留间隔。
        dst0_tile = dst_offset_elems // NEUMANN_FRACTAL_ELEMS
        dst1_tile = dst0_tile + dst_gap_fractals + 1
        self._load_l1_tile_to_l0b(self.packed64_tile(l1_matrix, src0_row_tile, src0_col_tile), l0b_slot,
            dst0_tile // NEUMANN_DIAG_BLOCK_NUM, dst0_tile % NEUMANN_DIAG_BLOCK_NUM)
        self._load_l1_tile_to_l0b(self.packed64_tile(l1_matrix, src1_row_tile, src1_col_tile), l0b_slot,
            dst1_tile // NEUMANN_DIAG_BLOCK_NUM, dst1_tile % NEUMANN_DIAG_BLOCK_NUM)

    def zero_half_l1(self, l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    def zero_packed64_l0_inputs(self, l0a_slot, l0b_slot):
        mem_copy(self._raw_l0a_full(l0a_slot), self.zero_l1)
        mem_copy(self._raw_l0b_full(l0b_slot), self.zero_l1)

    def load_packed64_diag_inputs_to_l0(self, left_l1, right_l1, l0a_slot, l0b_slot, *, clear_l0: bool = True):
        if clear_l0:
            self.zero_packed64_l0_inputs(l0a_slot, l0b_slot)
        mem_copy(local_slice(l0a_slot, (NEUMANN_BLOCK_SIZE, CHUNK_SIZE), offset=0),
            local_slice(left_l1, (NEUMANN_BLOCK_SIZE, CHUNK_SIZE), offset=0),
            engine=self.packed64_diag_l12l0)
        mem_copy(local_slice(l0b_slot, (NEUMANN_BLOCK_SIZE, CHUNK_SIZE), offset=0),
            local_slice(right_l1, (NEUMANN_BLOCK_SIZE, CHUNK_SIZE), offset=0),
            engine=self.packed64_diag_l12l0, transpose=True)

    def load_packed64_diag_inputs_to_l0_db(self, left_l1, right_l1, *, clear_l0: bool = True):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.load_packed64_diag_inputs_to_l0(left_l1, right_l1, l0a_write, l0b_write, clear_l0=clear_l0)

    def matmul_raw_l0ab_db_packed64(self, c_tile, *, init_c: bool, unit_flag: int):
        # 发起 full64 raw MMAD，并显式指定初始化或累加。
        l0a_read = self.raw_l0a_db
        l0b_read = self.raw_l0b_db
        matmul(c_tile, self._raw_l0a_full(l0a_read), self._raw_l0b_full(l0b_read), init=init_c, unit_flag=unit_flag)

    def matmul_packed64_diag_only(self, left_l1, right_l1, c_tile, *, init_c: bool, unit_flag: int, clear_l0: bool = True):
        self.load_packed64_diag_inputs_to_l0_db(left_l1, right_l1, clear_l0=clear_l0)
        self.matmul_raw_l0ab_db_packed64(c_tile, init_c=init_c, unit_flag=unit_flag)

    def matmul_packed64_diag_pair(self, left_l10, right_l10, left_l11, right_l11, c_tile, *, init_c0: bool, unit_flag0: int,
                                            init_c1: bool, unit_flag1: int, clear_l0: bool = True):
        self.load_packed64_diag_inputs_to_l0_db(left_l10, right_l10, clear_l0=clear_l0)
        self.load_packed64_diag_inputs_to_l0_db(left_l11, right_l11, clear_l0=clear_l0)
        self.matmul_raw_l0ab_db_packed64(c_tile, init_c=init_c0, unit_flag=unit_flag0)
        self.matmul_raw_l0ab_db_packed64(c_tile, init_c=init_c1, unit_flag=unit_flag1)

    def fixpipe_packed64_full_to_half_l1(self, dst_channel, c_tile):
        mem_copy(dst_channel, c_tile, engine=self.fixpipe_l0c2packed_l1)

    def fixpipe_packed64_full_to_bf16_l1(self, dst_channel, c_tile):
        mem_copy(dst_channel, c_tile, engine=self.fixpipe_l0c2packed_l1)

    def matmul_half_l1_power_packed64_diag(self, left_l1, right_l1, dst_channel, *, clear_l0: bool):
        c_power = self.raw_l0c_aux
        self.matmul_packed64_diag_only(left_l1, right_l1, c_power, init_c=True, unit_flag=3, clear_l0=clear_l0)
        c_power_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(dst_channel, c_power_read)

    def make_packed64_identity_l1(self, gm_identity16):
        # 构造 packed 64×64 单位矩阵。
        identity_write = self.packed_identity_l1
        self.zero_half_l1(identity_write)
        cube_sync_pipe(PIPE.MTE2)
        mem_copy(identity_write, gm_identity16, engine=self.gm2l1_identity16_diag)
        return self.packed_identity_l1

    def load_negative_t_identity_to_l0(self, negative_t, identity):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        mem_copy(self._raw_l0a_full(l0a_write), negative_t)
        mem_copy(self._raw_l0b_full(l0b_write), identity)

    def load_identity_pair_to_l0(self, identity):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        mem_copy(self._raw_l0a_full(l0a_write), identity)
        mem_copy(self._raw_l0b_full(l0b_write), identity)

    def make_inv_init_from_negative_t(self, negative_t, identity):
        # Cube 在 L0C 构造 scratch_l1 = (-T) @ I + I @ I。
        self.load_negative_t_identity_to_l0(negative_t, identity)
        self.load_identity_pair_to_l0(identity)
        c_inv_init = self.raw_l0c_main
        self.matmul_raw_l0ab_db_packed64(c_inv_init, init_c=True, unit_flag=2)
        self.matmul_raw_l0ab_db_packed64(c_inv_init, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.scratch_l1, c_inv_read)
        return self.scratch_l1

    def load_packed64_diagonal_inputs(self, neumann_power_l1, gm_identity16):
        # Cube 消费 AIV 生成的 -T，并构造 I + (-T)。
        identity = self.make_packed64_identity_l1(gm_identity16)
        negative_t = neumann_power_l1
        scratch = self.make_inv_init_from_negative_t(negative_t, identity)
        return negative_t, identity, scratch

    def make_neumann_diag_power2(self, negative_t, identity, scratch):
        self.matmul_half_l1_power_packed64_diag(negative_t, negative_t, self.tile_inv_l1, clear_l0=True)
        power2 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.matmul_packed64_diag_pair(scratch, identity, scratch, power2, c_inv, init_c0=True,
            unit_flag0=2, init_c1=False, unit_flag1=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power2, self.packed_inv64_l1

    def make_neumann_diag_power4(self, power2, packed_inv):
        self.matmul_half_l1_power_packed64_diag(power2, power2, self.tile_power_l1, clear_l0=False)
        power4 = self.tile_power_l1

        c_inv = self.raw_l0c_main
        self.matmul_packed64_diag_only(packed_inv, power4, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return power4, self.packed_inv64_l1

    def make_neumann_diag_power8(self, power4, packed_inv):
        self.matmul_half_l1_power_packed64_diag(power4, power4, self.tile_inv_l1, clear_l0=False)
        power8 = self.tile_inv_l1

        c_inv = self.raw_l0c_main
        self.matmul_packed64_diag_only(packed_inv, power8, c_inv, init_c=False, unit_flag=3, clear_l0=False)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def load_odd_lower_even_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_inputs(l0a_write, l0b_write)
        self._load_packed16_pair_to_full64_l0a_with_gap(scratch, l0a_write, src0_row_tile=1, src0_col_tile=0,
            src1_row_tile=3, src1_col_tile=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals())
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(packed_inv, l0b_write,
            src0_row_tile=0, src0_col_tile=0, src1_row_tile=2, src1_col_tile=2,
            dst_offset_elems=self.c64_diag_nz_offset_elems(0),
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals())

    def load_odd_inv_odd_tmp_full64(self, packed_inv, tmp64):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_inputs(l0a_write, l0b_write)
        self._load_l1_tile_to_l0a(self.packed64_tile(packed_inv, 1, 1), l0a_write, 1, 1)
        self._load_l1_tile_to_l0a(self.packed64_tile(packed_inv, 3, 3), l0a_write, 3, 3)
        self._load_packed16_pair_to_full64_l0b_trans_with_gap(tmp64, l0b_write, src0_row_tile=1,
            src0_col_tile=0, src1_row_tile=3, src1_col_tile=2,
            dst_offset_elems=NEUMANN_DIAG_BLOCK_NUM * NEUMANN_FRACTAL_ELEMS,
            dst_gap_fractals=self.c64_odd_even_pair_dst_gap_fractals())

    def make_odd_even_lower16_to32_inverse(self, scratch, packed_inv):
        # 组合两个 32×32 对角 tile 内部的下三角逆矩阵。
        self.load_odd_lower_even_inv_full64(scratch, packed_inv)
        c_tmp = self.raw_l0c_aux
        self.matmul_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.load_odd_inv_odd_tmp_full64(packed_inv, tmp64)
        c_inv = self.raw_l0c_main
        self.matmul_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_half_l1(self.packed_inv64_l1, c_inv_read)
        return self.packed_inv64_l1

    def load_odd32_lower_even32_inv_full64(self, scratch, packed_inv):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_inputs(l0a_write, l0b_write)
        mem_copy(self._raw_l0a_full(l0a_write), scratch)
        self._load_packed32_tile_to_full64_l0b(packed_inv, l0b_write, src_row=0, src_col=0,
            dst_row=0, dst_col=0, transpose=True)

    def load_odd32_inv_to_l0(self, packed_inv):
        l0a_write = self.raw_l0a_db
        l0b_write = self.raw_l0b_db
        self.zero_packed64_l0_inputs(l0a_write, l0b_write)
        mem_copy(self._raw_l0a_full(l0a_write), packed_inv)
        return l0a_write, l0b_write

    def load_odd32_tmp_to_l0(self, tmp64, l0a_write, l0b_write):
        self._load_packed32_tile_to_full64_l0b(tmp64, l0b_write, src_row=HALF_CHUNK_SIZE, src_col=0,
            dst_row=HALF_CHUNK_SIZE, dst_col=0, transpose=True)

    def make_odd_even_lower32_to64_inverse(self, scratch, packed_inv):
        # 组合 64×64 左下 32×32 下三角逆矩阵并生成 BF16 常驻逆矩阵。
        self.load_odd32_lower_even32_inv_full64(scratch, packed_inv)
        second_l0a, second_l0b = self.load_odd32_inv_to_l0(packed_inv)
        c_tmp = self.raw_l0c_aux
        self.matmul_raw_l0ab_db_packed64(c_tmp, init_c=True, unit_flag=3)
        c_tmp_read = self.raw_l0c_aux
        self.fixpipe_packed64_full_to_half_l1(self.tile_inv_l1, c_tmp_read)
        tmp64 = self.tile_inv_l1

        self.load_odd32_tmp_to_l0(tmp64, second_l0a, second_l0b)
        c_inv = self.raw_l0c_main
        self.matmul_raw_l0ab_db_packed64(c_inv, init_c=False, unit_flag=3)
        c_inv_read = self.raw_l0c_main
        self.fixpipe_packed64_full_to_bf16_l1(self.resident_inv_l1, c_inv_read)

    def _load_resident_inv_to_l0a(self):
        resident_inv = self.resident_inv_l1
        l0a_write = self.l0a_db
        cube_raw_l1_to_l0a(local_slice(l0a_write, (CHUNK_SIZE, CHUNK_SIZE), offset=0), resident_inv,
            l1_offset_elems=0, m_start=0, k_start=0, m_step=NEUMANN_DIAG_BLOCK_NUM,
            k_step=NEUMANN_DIAG_BLOCK_NUM, src_stride=NEUMANN_DIAG_BLOCK_NUM,
            dst_stride=NEUMANN_DIAG_BLOCK_NUM, transpose=False)
        return self.l0a_db

    def _load_beta_to_l0b(self, beta_channel):
        beta_read = beta_channel
        l0b_write = self.l0b_db
        cube_raw_l1_to_l0b(l0b_write, beta_read, l1_offset_elems=0, m_start=0, k_start=0,
            m_step=NEUMANN_DIAG_BLOCK_NUM, k_step=SUPPORTED_HEAD_DIM // NEUMANN_BLOCK_SIZE,
            src_stride=NEUMANN_DIAG_BLOCK_NUM,
            dst_stride=SUPPORTED_HEAD_DIM // NEUMANN_BLOCK_SIZE, transpose=True)
        return self.l0b_db

    def matmul_resident_inv_beta(self, l0a_resident, l0b_beta):
        l0c_write = self.l0c_db
        matmul(l0c_write, local_slice(l0a_resident, (CHUNK_SIZE, CHUNK_SIZE), offset=0), l0b_beta, init=True)

    def matmul_u_pre_and_w_from_resident_inv_beta(
        self,
        gm_U_pre_scratch,
        gm_W_scratch,
        linear_idx,
        v_beta_l1,
        k_decayed_beta_l1,
    ):
        # β·K 先产出，故先算 W；W 的 MMAD 和写回与后续 β·V 交叠。
        l0a_resident = self._load_resident_inv_to_l0a()

        l0b_beta = self._load_beta_to_l0b(k_decayed_beta_l1)
        self.matmul_resident_inv_beta(l0a_resident, l0b_beta)
        gm_W_chunk = tile_view(gm_W_scratch, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (linear_idx, 0))
        self.store_l0c_to_gm(gm_W_chunk, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM))

        l0b_beta = self._load_beta_to_l0b(v_beta_l1)
        self.matmul_resident_inv_beta(l0a_resident, l0b_beta)
        u_pre_part = tile_view(gm_U_pre_scratch, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (linear_idx, 0))
        self.store_l0c_to_gm(u_pre_part, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM))


class StageOneVector:
    def __init__(self, *, scale_value, subblock_idx):
        self.scale_value = scale_value
        self.subblock_idx = subblock_idx
        self.ub_q = Channel(MemLoc.UB, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.ub_k = Channel(MemLoc.UB, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1)
        self.ub_qk_exchange = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32)
        self.ub_g_raw = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1)
        self.ub_gate_activated = Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32)
        self.ub_dt_bias = Channel(MemLoc.UB, (D_HALF,), dtypes.float32, depth=1)
        self.ub_alpha = Channel(MemLoc.UB, (1,), dtypes.float32, depth=1)
        self.ub_beta_raw = Channel(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.bfloat16, depth=1)
        self.ub_beta = Buffer(MemLoc.UB, (CHUNK_SIZE, 1), dtypes.float32)
        self.ub_q_bf16 = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_k_bf16 = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_v_raw = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1)
        self.ub_gamma_c = Channel(MemLoc.UB, (1, D_HALF), dtypes.float32, depth=2)
        self.ub_cs_snapshot = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.float32) for _ in range(2)]
        self.ub_mqk = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=2, kind=ChannelKind.CrossCore)
        self.ub_bf16_nz = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=2, data_format="nz", n1_pad=16)
        self.ub_cast_bf16 = Channel(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_kkt_fp16 = Buffer(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float16)
        self.ub_mqk_bf16 = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.bfloat16, depth=1, data_format="nd")
        self.ub_raw_k = [Buffer(MemLoc.UB, (CHUNK_SIZE, D_HALF), dtypes.bfloat16) for _ in range(2)]
        self.ub_kkt = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float32, depth=1, kind=ChannelKind.CrossCore)
        self.ub_kkt_nz = Channel(MemLoc.UB, (HALF_CHUNK_SIZE, CHUNK_SIZE), dtypes.float16, depth=1, data_format="nz", n1_pad=16)
        self.ub_d_axis_nd = Channel(MemLoc.UB, (NEUMANN_BLOCK_SIZE, D_HALF), dtypes.bfloat16, depth=2)
        self.ub_d_axis_nz = Channel(MemLoc.UB, (NEUMANN_BLOCK_SIZE, D_HALF), dtypes.bfloat16, depth=2, data_format="nz", n1_pad=16)
        # 三槽 L1 流水依次传递目标行的左 K/Q 和各列右 K，Cube 将左 K/Q 常驻在独立 L0A 槽。
        self.d_axis_l1 = Channel(MemLoc.L1, (NEUMANN_BLOCK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=3, kind=ChannelKind.CrossCore)
        self.l1_V_beta = Channel(MemLoc.L1, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16, depth=1, kind=ChannelKind.CrossCore)
        self.l1_K_decayed_beta_for_W = Channel(MemLoc.L1, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtypes.bfloat16,
            depth=1, kind=ChannelKind.CrossCore)
        self.full_d_column_partition = make_partition_tiler((CHUNK_SIZE, SUPPORTED_HEAD_DIM),
            (CHUNK_SIZE, SUPPORTED_HEAD_DIM), axis=1)
        self.d_axis_partition = make_partition_tiler((NEUMANN_BLOCK_SIZE, D_HALF * 2),
            (NEUMANN_BLOCK_SIZE, D_HALF * 2), axis=1)
        self.full_square_partition = make_partition_tiler((CHUNK_SIZE, CHUNK_SIZE), (CHUNK_SIZE, CHUNK_SIZE))
        self.ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.d_axis_ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.fp16_ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16, pad_value=0.0)

    def scratch_half_tile(self, gm_scratch, block_idx, cols: int):
        return tile_view(gm_scratch, (HALF_CHUNK_SIZE, cols), (block_idx * 2 + self.subblock_idx, 0))

    def scratch_dhalf_tile(self, gm_scratch, block_idx):
        return tile_view(gm_scratch, (CHUNK_SIZE, D_HALF), (block_idx, self.subblock_idx))

    @jit
    def publish_normalized_qk_norm(self, norm_exchange, outgoing_norm):
        # 在后续门控计算前发布本 AIV 的 64 个逆范数。
        vec_sync_notify(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, QK_V_TO_MTE3_EVENT_ID)
        mem_copy(norm_exchange, outgoing_norm)
        vec_sync_block_arrive(PIPE.MTE3, QK_PUBLISH_FLAG_ID, mode=1)

    @jit
    def issue_normalized_qk_norm(self, norm_exchange, incoming_norm):
        # 提前搬入另一 AIV 的 64 个逆范数，使其与独立门控计算交叠。
        vec_sync_block_wait(PIPE.MTE2, QK_PUBLISH_FLAG_ID, mode=1)
        mem_copy(incoming_norm, norm_exchange)
        vec_sync_notify(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)

    @jit
    def finish_issued_normalized_qk_half(self):
        # 等待另一 AIV 的逆范数搬运，并完成跨 AIV 交接。
        vec_sync_wait(PIPE.MTE2, PIPE.V, QK_MTE2_TO_V_EVENT_ID)
        vec_sync_block_arrive(PIPE.MTE2, QK_CONSUME_FLAG_ID, mode=1)
        vec_sync_block_wait(PIPE.MTE3, QK_CONSUME_FLAG_ID, mode=1)

    @jit
    def normalize_owned_full_d_bf16(self, full_tile, local_half, subblock_idx: int, inverse_norms):
        # 计算每行完整 D 轴范数，并归一化本 AIV 负责的半轴。
        with vf(mode="raw"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * SUPPORTED_HEAD_DIM
                first_unpacked = vload_unpack(full_tile, offset, unpack_mode=UnpackMode.B16_TO_B32)
                first = vcast(first_unpacked, dtypes.float32, mask=lane_mask)
                second_unpacked = vload_unpack(full_tile, offset + D_HALF, unpack_mode=UnpackMode.B16_TO_B32)
                second = vcast(second_unpacked, dtypes.float32, mask=lane_mask)
                first_square = vmul(first, first, mask=lane_mask)
                second_square = vmul(second, second, mask=lane_mask)
                square_sum = vadd(first_square, second_square, mask=lane_mask)
                sum_squares = vreduce_sum(square_sum, mask=lane_mask)
                sum_squares_broadcast = vdup(sum_squares, mask=lane_mask)
                stabilized_sum_squares = vadds(sum_squares_broadcast, 1.0e-6, mask=lane_mask)
                norm = vsqrt(stabilized_sum_squares, mask=lane_mask)
                inverse_norm = vdiv(one, norm, mask=lane_mask)
                owned_values = first
                if const_expr(subblock_idx == 1):
                    owned_values = second
                local_offset = row * D_HALF
                normalized = vmul(owned_values, inverse_norm, mask=lane_mask)
                normalized_bf16 = vcast(normalized, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(local_half, local_offset, normalized_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
                vstore_first(inverse_norms, row, inverse_norm)
            vmem_bar("vst_vld")

    @jit
    def normalize_peer_half_bf16(self, local_half, inverse_norms):
        # 使用收到的逆范数归一化另一 AIV 负责的半轴。
        with vf(mode="raw"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                unpacked_value = vload_unpack(local_half, offset, unpack_mode=UnpackMode.B16_TO_B32)
                value = vcast(unpacked_value, dtypes.float32, mask=lane_mask)
                inverse_norm = vload_broadcast(inverse_norms, row)
                normalized = vmul(value, inverse_norm, mask=lane_mask)
                normalized_bf16 = vcast(normalized, dtypes.bfloat16, mask=lane_mask)
                vstore_pack(local_half, offset, normalized_bf16, lane_mask, pack_mode=PackMode.B32_TO_B16)
            vmem_bar("vst_vld")

    @jit
    def issue_normalized_qk_full_prefetch(self, gm_K, gm_Q, batch_idx, qk_head_idx, chunk_idx, valid_rows):
        # 当前完整 Q/K 被消费后，复用原 UB 槽预取下一任务的完整 Q/K。
        q_chunk = tile_view(gm_Q[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
        k_chunk = tile_view(gm_K[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
        q_chunk = _valid_rows_view(q_chunk, valid_rows, SUPPORTED_HEAD_DIM)
        k_chunk = _valid_rows_view(k_chunk, valid_rows, SUPPORTED_HEAD_DIM)
        if self.subblock_idx == 0:
            mem_copy(self.ub_q, q_chunk)
        else:
            mem_copy(self.ub_k, k_chunk)

    @jit
    def issue_normalized_qk_local_half(self, gm_K, gm_Q, gm_qk_exchange, batch_idx, qk_head_idx,
        chunk_idx, block_idx, valid_rows):
        # 在上一任务计算前发射完整 Q/K 和本地保存的另一半轴。
        q_to_k = gm_qk_exchange[block_idx, 0, None, None]
        k_to_q = gm_qk_exchange[block_idx, 1, None, None]
        q_chunk = tile_view(gm_Q[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
        k_chunk = tile_view(gm_K[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk_idx, 0))
        q_chunk = _valid_rows_view(q_chunk, valid_rows, SUPPORTED_HEAD_DIM)
        k_chunk = _valid_rows_view(k_chunk, valid_rows, SUPPORTED_HEAD_DIM)
        q_local_half = tile_view(gm_Q[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        k_local_half = tile_view(gm_K[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        q_local_half = _valid_rows_view(q_local_half, valid_rows, D_HALF)
        k_local_half = _valid_rows_view(k_local_half, valid_rows, D_HALF)

        if self.subblock_idx == 0:
            mem_copy(self.ub_q, q_chunk)
            mem_copy(self.ub_k_bf16, k_local_half)
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        else:
            mem_copy(self.ub_k, k_chunk)
            mem_copy(self.ub_q_bf16, q_local_half)
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)

        return q_to_k, k_to_q

    @jit
    def issue_normalized_qk_peer_half(self, gm_K, gm_Q, gm_qk_exchange, batch_idx, qk_head_idx,
        chunk_idx, block_idx, valid_rows):
        # 完整 Q/K 已预取时只搬入另一 AIV 所需的本地半轴。
        q_to_k = gm_qk_exchange[block_idx, 0, None, None]
        k_to_q = gm_qk_exchange[block_idx, 1, None, None]
        q_local_half = tile_view(gm_Q[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        k_local_half = tile_view(gm_K[batch_idx, qk_head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        q_local_half = _valid_rows_view(q_local_half, valid_rows, D_HALF)
        k_local_half = _valid_rows_view(k_local_half, valid_rows, D_HALF)
        if self.subblock_idx == 0:
            mem_copy(self.ub_k_bf16, k_local_half)
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        else:
            mem_copy(self.ub_q_bf16, q_local_half)
            vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
        return q_to_k, k_to_q

    @jit
    def finish_normalized_qk_local_half(self, q_to_k, k_to_q, valid_rows):
        # 完成本 AIV 的完整 D 轴范数，并仅发布 64 个逆范数。
        inverse_norms = local_slice(self.ub_qk_exchange, (1, CHUNK_SIZE), offset=0)
        if self.subblock_idx == 0:
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
            if valid_rows < CHUNK_SIZE:
                self._fill_bf16_invalid_rows(self.ub_q, valid_rows=valid_rows, cols=SUPPORTED_HEAD_DIM, value=0.0)
                self._fill_bf16_invalid_rows(self.ub_k_bf16, valid_rows=valid_rows, cols=D_HALF, value=0.0)
            self.normalize_owned_full_d_bf16(self.ub_q, self.ub_q_bf16, 0, inverse_norms)
            q_norm_exchange = tile_view(q_to_k, (1, CHUNK_SIZE), (0, 0))
            self.publish_normalized_qk_norm(q_norm_exchange, inverse_norms)
        else:
            vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_SUBBLOCK1_EVENT_ID)
            if valid_rows < CHUNK_SIZE:
                self._fill_bf16_invalid_rows(self.ub_k, valid_rows=valid_rows, cols=SUPPORTED_HEAD_DIM, value=0.0)
                self._fill_bf16_invalid_rows(self.ub_q_bf16, valid_rows=valid_rows, cols=D_HALF, value=0.0)
            self.normalize_owned_full_d_bf16(self.ub_k, self.ub_k_bf16, 1, inverse_norms)
            k_norm_exchange = tile_view(k_to_q, (1, CHUNK_SIZE), (0, 0))
            self.publish_normalized_qk_norm(k_norm_exchange, inverse_norms)

    @jit
    def issue_normalized_qk_exchange(self, q_to_k, k_to_q):
        # 开始接收另一 AIV 的 64 个逆范数。
        incoming_norms = local_slice(self.ub_qk_exchange, (1, CHUNK_SIZE), offset=0)
        if self.subblock_idx == 0:
            k_norm_exchange = tile_view(k_to_q, (1, CHUNK_SIZE), (0, 0))
            self.issue_normalized_qk_norm(k_norm_exchange, incoming_norms)
        else:
            q_norm_exchange = tile_view(q_to_k, (1, CHUNK_SIZE), (0, 0))
            self.issue_normalized_qk_norm(q_norm_exchange, incoming_norms)

    @jit
    def finish_normalized_qk_exchange(self):
        # 使用收到的逆范数归一化另一 AIV 负责的半轴。
        self.finish_issued_normalized_qk_half()
        inverse_norms = local_slice(self.ub_qk_exchange, (1, CHUNK_SIZE), offset=0)
        if self.subblock_idx == 0:
            self.normalize_peer_half_bf16(self.ub_k_bf16, inverse_norms)
        else:
            self.normalize_peer_half_bf16(self.ub_q_bf16, inverse_norms)
        return self.ub_k_bf16, self.ub_q_bf16

    def _copy_bf16_gm_half_tile_to_ub(self, gm_tile, ub_channel, *, cols: int = SUPPORTED_HEAD_DIM):
        bf16_valid = local_slice(ub_channel, (HALF_CHUNK_SIZE, cols), offset=0)
        mem_copy(bf16_valid, gm_tile)

    @jit
    def _fill_bf16_invalid_rows(self, ub_tile, *, valid_rows, cols: int, value: float):
        # 使用 BF16 常量初始化 [valid_rows, 64) 的无效行。
        with vf(mode="raw"):
            lane_mask, _ = update_mask(cols, elem_bits=16)
            fill_bf16 = vdups(value, dtypes.bfloat16, mask=lane_mask)
            for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=4):
                vstore(ub_tile, row * cols, fill_bf16, lane_mask)
            vmem_bar("vst_vld")

    @jit
    def issue_raw_gate(self, gm_g, gm_A_log, gm_dt_bias, batch_idx, head_idx, chunk_idx, valid_rows):
        # 在 Q/K 归一化和交换前发射门控输入。
        g_chunk = tile_view(gm_g[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        g_chunk = _valid_rows_view(g_chunk, valid_rows, D_HALF)
        bias_half = tile_view(gm_dt_bias[head_idx, None], (D_HALF,), (self.subblock_idx,))
        alpha = tile_view(gm_A_log, (1,), (head_idx,))
        mem_copy(self.ub_g_raw, g_chunk)
        mem_copy(self.ub_dt_bias, bias_half)
        mem_copy(self.ub_alpha, alpha)
        vec_sync_notify(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)

    @jit
    def activate_issued_gate(self, lower_bound, valid_rows, buffer_parity: int):
        # 等待门控输入，完成激活后计算累积和。
        vec_sync_wait(PIPE.MTE2, PIPE.V, GATE_MTE2_TO_V_EVENT_ID)
        g_slot = self.ub_g_raw
        rows_to_activate = CHUNK_SIZE
        if valid_rows < CHUNK_SIZE:
            # gate 只计算有效行；将 [valid_rows, 64) 清零后送入 cumsum，
            # 使无效行成为尾块的身份行。
            with vf(mode="raw"):
                lane_mask, _ = update_mask(D_HALF, elem_bits=32)
                zero = vdups(0.0, dtypes.float32, mask=lane_mask)
                for row in dsl_range(valid_rows, CHUNK_SIZE, 1, unroll=4):
                    vstore(self.ub_gate_activated, row * D_HALF, zero, lane_mask)
                vmem_bar("vst_vld")
            rows_to_activate = valid_rows
        with vf(mode="raw"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            bound = vdups(lower_bound, dtypes.float32, mask=lane_mask)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            alpha = vload_broadcast(self.ub_alpha, 0)
            alpha_exp = vexp(alpha, mask=lane_mask)
            neg_exp_a = vmuls(alpha_exp, -1.0, mask=lane_mask)
            for row in dsl_range(0, rows_to_activate, 1, unroll=4):
                offset = row * D_HALF
                unpacked_g = vload_unpack(g_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                raw_g = vcast(unpacked_g, dtypes.float32, mask=lane_mask)
                dt_bias = vload(self.ub_dt_bias, 0)
                biased_g = vadd(raw_g, dt_bias, mask=lane_mask)
                gate_logit = vmul(neg_exp_a, biased_g, mask=lane_mask)
                exp_gate_logit = vexp(gate_logit, mask=lane_mask)
                gate_denominator = vadd(one, exp_gate_logit, mask=lane_mask)
                gate = vdiv(bound, gate_denominator, mask=lane_mask)
                vstore(self.ub_gate_activated, offset, gate, lane_mask)
            vmem_bar("vst_vld")
        vec_sync_all()
        # 延迟任务已消费对应奇偶槽，累积和可直接写入常驻快照，省去一次完整 FP32 tile 复制。
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        _prefix_sum_rows(gate_cumsum_snapshot, self.ub_gate_activated)
        return gate_cumsum_snapshot

    @jit
    def issue_raw_beta(self, gm_beta, batch_idx, head_idx, chunk_idx, valid_rows):
        # 在 D 轴分块计算前发射一块 BF16 beta logit。
        beta_chunk = tile_view(gm_beta[batch_idx, head_idx, None, None], (CHUNK_SIZE, 1), (chunk_idx, 0))
        beta_chunk = _valid_rows_view(beta_chunk, valid_rows, 1)
        mem_copy(self.ub_beta_raw, beta_chunk)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)

    @jit
    def activate_issued_beta(self, valid_rows):
        # 等待 beta logit 并计算 sigmoid。
        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_EVENT_ID)
        beta_slot = self.ub_beta_raw
        with vf(mode="raw"):
            lane_mask, _ = update_mask(VL, elem_bits=32)
            one = vdups(1.0, dtypes.float32, mask=lane_mask)
            unpacked_beta = vload_unpack(beta_slot, 0, unpack_mode=UnpackMode.B16_TO_B32)
            raw_beta = vcast(unpacked_beta, dtypes.float32, mask=lane_mask)
            neg_beta = vmuls(raw_beta, -1.0, mask=lane_mask)
            exp_neg_beta = vexp(neg_beta, mask=lane_mask)
            sigmoid_denominator = vadd(one, exp_neg_beta, mask=lane_mask)
            beta = vdiv(one, sigmoid_denominator, mask=lane_mask)
            if valid_rows < CHUNK_SIZE:
                zero = vdups(0.0, dtypes.float32, mask=lane_mask)
                valid_mask, _ = update_mask(valid_rows, elem_bits=32)
                invalid_mask = mask_xor(lane_mask, valid_mask, exec_mask=lane_mask)
                vstore(self.ub_beta, 0, beta, valid_mask)
                vstore(self.ub_beta, 0, zero, invalid_mask)
            else:
                vstore(self.ub_beta, 0, beta, lane_mask)
            vmem_bar("vst_vld")
        return self.ub_beta

    @jit
    def apply_beta_and_finish_t_for_neumann(self, beta_slot, neumann_power_l1, subblock_base: int, buffer_parity: int):
        row_base = subblock_base * HALF_CHUNK_SIZE  # 本半块全局行起点为 0 或 32。
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                row_off = row * CHUNK_SIZE
                t_true = vload(self.ub_kkt, row_off)
                beta_brc = vload_broadcast(beta_slot, row_base + row)  # β[row] 广播到整行。
                t_beta = vmul(t_true, beta_brc, mask=row_mask)  # T[row, :] *= β[row]。
                # 严格下三角：全局行 i 保留前 i 列(对角也清)；update_mask(i) 的前 i 个 lane = 保留谓词。
                tril_mask, _ = update_mask(row_base + row, elem_bits=32)
                t_tril = vselect_raw(t_beta, zero_reg, cond_mask=tril_mask)  # lane < i 取 t_beta，其余取零。
                neg_t = vmuls(t_tril, -1.0, mask=row_mask)
                neg_t_fp16 = vcast(neg_t, dtypes.float16, mask=row_mask)
                vstore_pack(self.ub_kkt_fp16, row_off, neg_t_fp16, row_mask, pack_mode="b32_to_b16")
        # ND→NZ 转换放在 raw VF 外，同一 V 流水按程序序保证读写依赖。
        with vf(outputs=[self.ub_kkt_nz]):
            mem_copy(self.ub_kkt_nz, self.ub_kkt_fp16, engine=self.fp16_ub2l1)
        self.store_neumann_tmp_to_l1(neumann_power_l1, subblock_base)

    @jit
    def _finish_beta_weighted_v_and_k(
        self, v_slot, beta_slot, raw_k_snapshot, gate_cumsum_snapshot, valid_rows,
    ):
        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                neg_beta = vmuls(beta, -1.0, mask=row_mask)
                cs_value = vload(gate_cumsum_snapshot, offset)
                raw_k = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(raw_k, dtypes.float32, mask=row_mask)
                decay = vexp(cs_value, mask=row_mask)
                k_decayed = vmul(k_value, decay, mask=row_mask)
                k_decayed_beta = vmul(k_decayed, neg_beta, mask=row_mask)
                k_decayed_beta_bf16 = vcast(k_decayed_beta, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, k_decayed_beta_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        with vf(outputs=[self.ub_bf16_nz]):
            mem_copy(self.ub_bf16_nz, cast_read, engine=self.ub2l1)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_K_decayed_beta_for_W, self.ub_bf16_nz)

        vec_sync_wait(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)
        if valid_rows < CHUNK_SIZE:
            self._fill_bf16_invalid_rows(v_slot, valid_rows=valid_rows, cols=D_HALF, value=0.0)

        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                beta = vload_broadcast(beta_slot, row)
                raw_v = vload_unpack(v_slot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                v_value = vcast(raw_v, dtypes.float32, mask=row_mask)
                v_beta = vmul(v_value, beta, mask=row_mask)
                v_beta_bf16 = vcast(v_beta, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, v_beta_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        with vf(outputs=[self.ub_bf16_nz]):
            mem_copy(self.ub_bf16_nz, cast_read, engine=self.ub2l1)
        self._store_dhalf_nz_ub_to_full_l1(self.l1_V_beta, self.ub_bf16_nz)

    @jit
    def issue_beta_weighted_v(self, gm_V, batch_idx, head_idx, chunk_idx, valid_rows):
        # 提前一任务发射 V，使搬运与当前任务 C3 向量计算交叠。
        v_half = tile_view(gm_V[batch_idx, head_idx, None, None], (CHUNK_SIZE, D_HALF), (chunk_idx, self.subblock_idx))
        v_half = _valid_rows_view(v_half, valid_rows, D_HALF)
        self._copy_bf16_gm_half_tile_to_ub(v_half, self.ub_v_raw, cols=D_HALF)
        vec_sync_notify(PIPE.MTE2, PIPE.V, MTE2_TO_V_V_STAGING_EVENT_ID)

    @jit
    def make_beta_weighted_v_and_k(self, beta_slot, buffer_parity: int, valid_rows):
        # 上一任务预取 V 完成后构造 C3 所需的 β·V 和 β·K。
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        self._finish_beta_weighted_v_and_k(self.ub_v_raw, beta_slot, raw_k_snapshot,
            gate_cumsum_snapshot, valid_rows)

    @jit
    def prepare_normalized_chunk_inputs(self, normalized_k_dhalf, normalized_q_dhalf, gate_cumsum_dhalf, buffer_parity: int):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        log_gamma_c_write = self.ub_gamma_c
        self.save_gamma_c_row(gate_cumsum_dhalf, log_gamma_c_write)
        gate_cumsum_tile = local_slice(gate_cumsum_dhalf, (CHUNK_SIZE, D_HALF), offset=0)

        mem_copy(raw_k_snapshot, normalized_k_dhalf)

        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                gamma = vload(gate_cumsum_tile, offset)
                exp_value = vexp(gamma, mask=row_mask)
                raw_q = vload_unpack(normalized_q_dhalf, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q_value = vcast(raw_q, dtypes.float32, mask=row_mask)
                q_decayed = vmul(q_value, exp_value, mask=row_mask)
                q_decayed = vmuls(q_decayed, self.scale_value, mask=row_mask)
                q_decayed_bf16 = vcast(q_decayed, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, q_decayed_bf16, row_mask, pack_mode="b32_to_b16")
        return self.ub_cast_bf16, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf

    def _store_dhalf_nz_ub_to_full_l1(self, l1_dst, ub_nz_src):
        l1_part = partition_view(l1_dst, self.full_d_column_partition, self.subblock_idx)
        mem_copy(l1_part, ub_nz_src)

    def store_neumann_tmp_to_l1(self, l1_dst, subblock_base: int):
        l1_part = partition_view(l1_dst, self.full_square_partition, subblock_base)
        mem_copy(l1_part, self.ub_kkt_nz)

    @jit
    def _store_masked_mqk_to_bf16_gm(self, gm_tile, buffer_parity: int):
        # Mqk 仅保留含对角下三角，再转为 BF16 写回 GM。
        mqk_slot = self.ub_mqk
        bf16_write = self.ub_mqk_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            zero_reg = vdups(0.0, dtypes.float32, mask=row_mask)
            for row in dsl_range(0, HALF_CHUNK_SIZE, 1, unroll=4):
                off = row * CHUNK_SIZE
                keep, _ = update_mask(self.subblock_idx * HALF_CHUNK_SIZE + row + 1, elem_bits=32)
                mqk = vload(mqk_slot, off)
                mqk_tril = vselect_raw(mqk, zero_reg, cond_mask=keep)  # 保留 lane <= i。
                mqk_bf16 = vcast(mqk_tril, dtypes.bfloat16, mask=row_mask)
                vstore_pack(bf16_write, off, mqk_bf16, row_mask, pack_mode="b32_to_b16")
        bf16_read = self.ub_mqk_bf16
        mem_copy(gm_tile, local_slice(bf16_read, (HALF_CHUNK_SIZE, CHUNK_SIZE), offset=0))

    def finish_mqk(self, gm_Mqk_scratch, block_idx, subblock_base: int, buffer_parity: int):
        Mqk_part = self.scratch_half_tile(gm_Mqk_scratch, block_idx, CHUNK_SIZE)
        self._store_masked_mqk_to_bf16_gm(Mqk_part, buffer_parity)
        return self.ub_gamma_c

    def store_q_decayed_scratch(self, gm_Q_decayed_scratch, block_idx, q_decayed_slot):
        # Q_decayed 已转为 BF16 并保留在原 UB 槽，直接写回可省去一次重复类型转换。
        Q_decayed_slot_part = self.scratch_dhalf_tile(gm_Q_decayed_scratch, block_idx)
        mem_copy(Q_decayed_slot_part, q_decayed_slot)

    def save_gamma_c_row(self, g_slot, gamma_c_write):
        src_pre = (CHUNK_SIZE - 1) * D_HALF  # 累积和最后一行的元素起点。
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            gamma_c = vload(g_slot, src_pre)
            vstore(gamma_c_write, 0, gamma_c, row_mask)

    @jit
    def store_restored_state_inputs(
        self, gm_K_restored_scratch, gm_gamma_C_scratch, block_idx, gamma_c_slot, buffer_parity: int,
    ):
        raw_k_snapshot = self.ub_raw_k[buffer_parity]
        gate_cumsum_snapshot = self.ub_cs_snapshot[buffer_parity]
        K_restored_dhalf = self.scratch_dhalf_tile(gm_K_restored_scratch, block_idx)
        cast_write = self.ub_cast_bf16
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            cs_last = vload(gamma_c_slot, 0)
            for row in dsl_range(0, CHUNK_SIZE, 1, unroll=4):
                offset = row * D_HALF
                raw_k = vload_unpack(raw_k_snapshot, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k_value = vcast(raw_k, dtypes.float32, mask=row_mask)
                gate_cumsum = vload(gate_cumsum_snapshot, offset)
                restore_scale = vexp_sub(cs_last, gate_cumsum, mask=row_mask)
                restored = vmul(k_value, restore_scale, mask=row_mask)
                restored_bf16 = vcast(restored, dtypes.bfloat16, mask=row_mask)
                vstore_pack(cast_write, offset, restored_bf16, row_mask, pack_mode="b32_to_b16")
        cast_read = self.ub_cast_bf16
        mem_copy(K_restored_dhalf, cast_read)

        # gamma_C 下溢为零是正确的状态衰减值，后续不再与 K_inv 相乘。
        with vf(mode="raw"):
            row_mask, _ = update_mask(VL, elem_bits=32)
            gamma_c = vload(gamma_c_slot, 0)
            gamma_c_exp = vexp(gamma_c, mask=row_mask)
            vstore(gamma_c_slot, 0, gamma_c_exp, row_mask)
        gamma_C_gm = tile_view(gm_gamma_C_scratch, (D_HALF, 1), (block_idx * 2 + self.subblock_idx, 0))
        mem_copy(gamma_C_gm, gamma_c_slot)

    @jit
    def store_d_axis_tile_to_l1(self, full_channel, nd_slot):
        mem_copy(self.ub_d_axis_nz, nd_slot, engine=self.d_axis_ub2l1)
        full_part = partition_view(full_channel, self.d_axis_partition, self.subblock_idx)
        mem_copy(full_part, self.ub_d_axis_nz)

    @jit
    def make_d_axis_left_k(self, normalized_k_dhalf, gate_cumsum_dhalf, row_tile: int, left_k_l1):
        k_read = local_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), offset=row_tile * NEUMANN_BLOCK_SIZE * D_HALF * 2)
        decay_scratch = local_slice(self.ub_qk_exchange, (NEUMANN_BLOCK_SIZE, D_HALF), offset=0)
        ref_offset = row_tile * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_d_axis_nd
        with vf(mode="raw"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (row_tile * NEUMANN_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(gate_cumsum, ref, mask=mask)
                vstore(decay_scratch, offset, decay, mask)
                k_decayed = vmul(k, decay, mask=mask)
                k_decayed_bf16 = vcast(k_decayed, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, k_decayed_bf16, mask, pack_mode="b32_to_b16")
            vmem_bar("vst_vld")
        nd_read = self.ub_d_axis_nd
        self.store_d_axis_tile_to_l1(left_k_l1, nd_read)

    @jit
    def make_d_axis_left_q(self, normalized_q_dhalf, row_tile: int, left_q_l1):
        q_read = local_slice(normalized_q_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), offset=row_tile * NEUMANN_BLOCK_SIZE * D_HALF * 2)
        decay_scratch = local_slice(self.ub_qk_exchange, (NEUMANN_BLOCK_SIZE, D_HALF), offset=0)
        nd_write = self.ub_d_axis_nd
        with vf(mode="raw"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                unpacked_q = vload_unpack(q_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                q = vcast(unpacked_q, dtypes.float32, mask=mask)
                decay = vload(decay_scratch, offset)
                q_decayed = vmul(q, decay, mask=mask)
                scaled_q = vmuls(q_decayed, self.scale_value, mask=mask)
                scaled_q_bf16 = vcast(scaled_q, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, scaled_q_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_d_axis_nd
        self.store_d_axis_tile_to_l1(left_q_l1, nd_read)

    @jit
    def make_d_axis_right_k(self, normalized_k_dhalf, gate_cumsum_dhalf, row_tile: int, col_tile: int, right_k_l1):
        k_read = local_slice(normalized_k_dhalf, (NEUMANN_BLOCK_SIZE, D_HALF), offset=col_tile * NEUMANN_BLOCK_SIZE * D_HALF * 2)
        ref_offset = row_tile * NEUMANN_BLOCK_SIZE * D_HALF

        nd_write = self.ub_d_axis_nd
        with vf(mode="raw"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            ref = vload(gate_cumsum_dhalf, ref_offset)
            for row in dsl_range(0, NEUMANN_BLOCK_SIZE, 1, unroll=2):
                offset = row * D_HALF
                g_offset = (col_tile * NEUMANN_BLOCK_SIZE + row) * D_HALF
                unpacked_k = vload_unpack(k_read, offset, unpack_mode=UnpackMode.B16_TO_B32)
                k = vcast(unpacked_k, dtypes.float32, mask=mask)
                gate_cumsum = vload(gate_cumsum_dhalf, g_offset)
                decay = vexp_sub(ref, gate_cumsum, mask=mask)
                restored_k = vmul(k, decay, mask=mask)
                restored_k_bf16 = vcast(restored_k, dtypes.bfloat16, mask=mask)
                vstore_pack(nd_write, offset, restored_k_bf16, mask, pack_mode="b32_to_b16")
        nd_read = self.ub_d_axis_nd
        self.store_d_axis_tile_to_l1(right_k_l1, nd_read)

@jit
def _run_stage1_body(
    tail_group,
    tail_rows,
    gm_K,
    gm_V,
    gm_Q,
    gm_beta_brc,
    gm_g,
    gm_qk_exchange,
    gm_A_log,
    gm_dt_bias,
    gm_q_decayed,
    gm_Mqk,
    gm_k_restored,
    gm_gamma,
    gm_U_pre,
    gm_w,
    gm_lower_mask_tiles,
    gm_identity16,
    batch: int,  # 兼容参数；BN 合轴后由 head_base 和 head_count 表达。
    head_count: int,
    chunks_total: int,
    seq_len: int,
    head_base: int,
    group_start: int,
    group_count: int,
    eff_group: int,
    active_core_num: int,
    task_start: int,
    task_end: int,
    workspace_task_base: int,
    scale_value: float,
    lower_bound: float,
):
    # Stage1 使用 Neumann 展开求解块内三角系统，并生成 Stage2 所需工作区。
    subblock_idx = get_subblock_id()
    block_idx = get_block_idx()
    batch_count = gm_K.shape[0]
    value_head_count = gm_V.shape[1]
    key_head_count = gm_K.shape[1]
    heads_per_key_head = value_head_count // key_head_count


    if block_idx < active_core_num:
        # 严格和含对角下三角均由 raw VF 谓词裁剪，不再搬入常驻 byte mask。
        matmul = StageOneMatmul()
        vector = StageOneVector(scale_value=scale_value, subblock_idx=subblock_idx)
        # zero_l1 是 packed64 L0 清零的只读源，每次 Stage1 仅初始化一次。
        matmul.zero_half_l1(matmul.zero_l1)
        task_delay_line = DelayLineGroup(2, "idx", "valid_rows")
        pipeline_tick = 0
        issued_task_count = 0
        full_qk_prefetched = False
        for local_task_index in range(task_start, task_end):
            current_local_head_index = local_task_index // group_count
            current_group_chunk_index = (
                local_task_index - current_local_head_index * group_count
            )
            current_chunk_index = group_start + current_group_chunk_index
            current_global_head_index = head_base + current_local_head_index
            current_batch_index, current_value_head_index = idx2crd(current_global_head_index, [batch_count, value_head_count])
            current_query_key_head_index = (
                current_value_head_index // heads_per_key_head
            )
            current_valid_rows = CHUNK_SIZE
            if tail_group:
                if current_group_chunk_index + 1 == group_count:
                    current_valid_rows = tail_rows

            # 在上一任务的 Vector/Cube 尾部前发射当前任务 MTE2 搬运。
            vector.issue_raw_gate(gm_g, gm_A_log, gm_dt_bias, current_batch_index,
                current_value_head_index, current_chunk_index, current_valid_rows)
            q_to_k_exchange = gm_qk_exchange[block_idx, 0, None, None]
            k_to_q_exchange = gm_qk_exchange[block_idx, 1, None, None]
            if not full_qk_prefetched:
                vector.issue_normalized_qk_local_half(gm_K, gm_Q, gm_qk_exchange, current_batch_index,
                    current_query_key_head_index, current_chunk_index, block_idx, current_valid_rows)
            else:
                vector.issue_normalized_qk_peer_half(gm_K, gm_Q, gm_qk_exchange, current_batch_index,
                    current_query_key_head_index, current_chunk_index, block_idx, current_valid_rows)

            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_index = task_delay_line.idx.tap(1)
                local_head_index = delayed_task_index // group_count
                group_chunk_index = (
                    delayed_task_index - local_head_index * group_count
                )
                chunk_index = group_start + group_chunk_index
                global_head_index = head_base + local_head_index
                batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
                buffer_parity = delayed_task_index % 2
                delayed_valid_rows = task_delay_line.valid_rows.tap(1)
                vector.apply_beta_and_finish_t_for_neumann(vector.ub_beta, matmul.neumann_power_l1, subblock_idx, buffer_parity)
                vector.make_beta_weighted_v_and_k(vector.ub_beta, buffer_parity, delayed_valid_rows)
                negative_lower_matrix, identity_matrix, neumann_scratch = (
                    matmul.load_packed64_diagonal_inputs(matmul.neumann_power_l1, gm_identity16)
                )
                second_power, packed_inverse = matmul.make_neumann_diag_power2(negative_lower_matrix, identity_matrix, neumann_scratch)
                fourth_power, packed_inverse = matmul.make_neumann_diag_power4(second_power, packed_inverse)
                packed_inverse = matmul.make_neumann_diag_power8(fourth_power, packed_inverse)
                packed_inverse = matmul.make_odd_even_lower16_to32_inverse(neumann_scratch, packed_inverse)
                matmul.make_odd_even_lower32_to64_inverse(neumann_scratch, packed_inverse)

            vector.issue_beta_weighted_v(gm_V, current_batch_index, current_value_head_index, current_chunk_index, current_valid_rows)
            task_delay_line.push(idx=local_task_index, valid_rows=current_valid_rows)
            local_head_index = current_local_head_index
            group_chunk_index = current_group_chunk_index
            chunk_index = current_chunk_index
            global_head_index = current_global_head_index
            batch_index = current_batch_index
            value_head_index = current_value_head_index
            workspace_task_index = (
                workspace_task_base + global_head_index * eff_group + group_chunk_index
            )
            buffer_parity = local_task_index % 2

            vector.finish_normalized_qk_local_half(q_to_k_exchange, k_to_q_exchange, current_valid_rows)
            vector.issue_normalized_qk_exchange(q_to_k_exchange, k_to_q_exchange)
            gate_cumsum_dhalf = vector.activate_issued_gate(lower_bound, current_valid_rows, buffer_parity)
            normalized_k_dhalf, normalized_q_dhalf = vector.finish_normalized_qk_exchange()
            vector.issue_raw_beta(gm_beta_brc, batch_index, value_head_index, chunk_index, current_valid_rows)
            decayed_q_dhalf, gate_cumsum_dhalf, normalized_k_dhalf, normalized_q_dhalf = (
                vector.prepare_normalized_chunk_inputs(normalized_k_dhalf, normalized_q_dhalf,
                    gate_cumsum_dhalf, buffer_parity)
            )
            vector.store_q_decayed_scratch(gm_q_decayed, workspace_task_index, decayed_q_dhalf)

            # 当前完整 Q/K 在归一化后失效，复用原 UB 槽预取下一任务的完整 Q/K。
            if local_task_index + 1 < task_end:
                next_local_task_index = local_task_index + 1
                next_local_head_index = next_local_task_index // group_count
                next_group_chunk_index = next_local_task_index - next_local_head_index * group_count
                next_chunk_index = group_start + next_group_chunk_index
                next_global_head_index = head_base + next_local_head_index
                next_batch_index, next_value_head_index = idx2crd(next_global_head_index, [batch_count, value_head_count])
                next_query_key_head_index = next_value_head_index // heads_per_key_head
                next_valid_rows = CHUNK_SIZE
                if tail_group:
                    if next_group_chunk_index + 1 == group_count:
                        next_valid_rows = tail_rows
                vector.issue_normalized_qk_full_prefetch(gm_K, gm_Q, next_batch_index,
                    next_query_key_head_index, next_chunk_index, next_valid_rows)
                full_qk_prefetched = True
            else:
                full_qk_prefetched = False

            # D 轴分块只计算 col_tile <= row_tile，跳过上三角 6 对 KKT 和 Mqk。
            vector.make_d_axis_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 0, vector.d_axis_l1)
            matmul.load_d_axis_left_k(vector.d_axis_l1)
            vector.make_d_axis_left_q(normalized_q_dhalf, 0, vector.d_axis_l1)
            matmul.load_d_axis_left_q(vector.d_axis_l1)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 0, 0, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(0, 0)
            matmul.matmul_d_axis_mqk(0, 0)

            vector.make_d_axis_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 1, vector.d_axis_l1)
            matmul.load_d_axis_left_k(vector.d_axis_l1)
            vector.make_d_axis_left_q(normalized_q_dhalf, 1, vector.d_axis_l1)
            matmul.load_d_axis_left_q(vector.d_axis_l1)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 1, 0, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(1, 0)
            matmul.matmul_d_axis_mqk(1, 0)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 1, 1, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(1, 1)
            matmul.matmul_d_axis_mqk(1, 1)

            vector.make_d_axis_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 2, vector.d_axis_l1)
            matmul.load_d_axis_left_k(vector.d_axis_l1)
            vector.make_d_axis_left_q(normalized_q_dhalf, 2, vector.d_axis_l1)
            matmul.load_d_axis_left_q(vector.d_axis_l1)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 2, 0, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(2, 0)
            matmul.matmul_d_axis_mqk(2, 0)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 2, 1, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(2, 1)
            matmul.matmul_d_axis_mqk(2, 1)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 2, 2, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(2, 2)
            matmul.matmul_d_axis_mqk(2, 2)

            vector.make_d_axis_left_k(normalized_k_dhalf, gate_cumsum_dhalf, 3, vector.d_axis_l1)
            matmul.load_d_axis_left_k(vector.d_axis_l1)
            vector.make_d_axis_left_q(normalized_q_dhalf, 3, vector.d_axis_l1)
            matmul.load_d_axis_left_q(vector.d_axis_l1)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 3, 0, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(3, 0)
            matmul.matmul_d_axis_mqk(3, 0)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 3, 1, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(3, 1)
            matmul.matmul_d_axis_mqk(3, 1)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 3, 2, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(3, 2)
            matmul.matmul_d_axis_mqk(3, 2)
            vector.make_d_axis_right_k(normalized_k_dhalf, gate_cumsum_dhalf, 3, 3, vector.d_axis_l1)
            matmul.load_d_axis_right_k(vector.d_axis_l1)
            matmul.matmul_d_axis_kkt(3, 3)
            matmul.matmul_d_axis_mqk(3, 3)
            matmul.store_d_axis_kkt_to_ub(vector.ub_kkt)

            vector.activate_issued_beta(current_valid_rows)
            matmul.store_d_axis_mqk_to_ub(vector.ub_mqk)

            issued_task_count = issued_task_count + 1
            if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
                delayed_task_index = task_delay_line.idx.tap(1)
                local_head_index = delayed_task_index // group_count
                group_chunk_index = (
                    delayed_task_index - local_head_index * group_count
                )
                workspace_task_index = (
                    head_base + local_head_index
                ) * eff_group + group_chunk_index + workspace_task_base
                buffer_parity = delayed_task_index % 2
                log_gamma_c_dhalf = vector.finish_mqk(gm_Mqk, workspace_task_index, subblock_idx, buffer_parity)
                vector.store_restored_state_inputs(gm_k_restored, gm_gamma, workspace_task_index, log_gamma_c_dhalf, buffer_parity)
                matmul.matmul_u_pre_and_w_from_resident_inv_beta(gm_U_pre, gm_w, workspace_task_index,
                    vector.l1_V_beta, vector.l1_K_decayed_beta_for_W)
            task_delay_line.advance()
            pipeline_tick = pipeline_tick + 1

        if pipeline_tick >= 1 and pipeline_tick - 1 < issued_task_count:
            delayed_task_index = task_delay_line.idx.tap(1)
            local_head_index = delayed_task_index // group_count
            group_chunk_index = delayed_task_index - local_head_index * group_count
            chunk_index = group_start + group_chunk_index
            global_head_index = head_base + local_head_index
            batch_index, value_head_index = idx2crd(global_head_index, [batch_count, value_head_count])
            workspace_task_index = (
                workspace_task_base + global_head_index * eff_group + group_chunk_index
            )
            buffer_parity = delayed_task_index % 2
            delayed_valid_rows = task_delay_line.valid_rows.tap(1)
            vector.apply_beta_and_finish_t_for_neumann(vector.ub_beta, matmul.neumann_power_l1, subblock_idx, buffer_parity)
            vector.make_beta_weighted_v_and_k(vector.ub_beta, buffer_parity, delayed_valid_rows)
            negative_lower_matrix, identity_matrix, neumann_scratch = (
                matmul.load_packed64_diagonal_inputs(matmul.neumann_power_l1, gm_identity16)
            )
            second_power, packed_inverse = matmul.make_neumann_diag_power2(negative_lower_matrix, identity_matrix, neumann_scratch)
            fourth_power, packed_inverse = matmul.make_neumann_diag_power4(second_power, packed_inverse)
            packed_inverse = matmul.make_neumann_diag_power8(fourth_power, packed_inverse)
            packed_inverse = matmul.make_odd_even_lower16_to32_inverse(neumann_scratch, packed_inverse)
            matmul.make_odd_even_lower32_to64_inverse(neumann_scratch, packed_inverse)
            log_gamma_c_dhalf = vector.finish_mqk(gm_Mqk, workspace_task_index, subblock_idx, buffer_parity)
            vector.store_restored_state_inputs(gm_k_restored, gm_gamma, workspace_task_index, log_gamma_c_dhalf, buffer_parity)
            matmul.matmul_u_pre_and_w_from_resident_inv_beta(gm_U_pre, gm_w, workspace_task_index,
                vector.l1_V_beta, vector.l1_K_decayed_beta_for_W)

class StageTwoMatmul:

    def __init__(self, *, dv_base: int):
        self.dv_base = dv_base
        self.l0a_chunk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
        self.l0a_kr_t = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1, data_format="zn")
        self.l0a_mqk = Channel(MemLoc.L0A, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=1)
        self.l0b_u = Channel(MemLoc.L0B, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.bfloat16, depth=1)
        self.l0b_state = Channel(MemLoc.L0B, shape=(dv_base, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=1)
        self.l0b_kr = Channel(MemLoc.L0B, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=1)
        self.l0c_u = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1)
        self.l0c_o = Channel(MemLoc.L0C, shape=(CHUNK_SIZE, dv_base), dtype=dtypes.float32, depth=1)
        self.l0c_state = Channel(MemLoc.L0C, shape=(dv_base, SUPPORTED_HEAD_DIM), dtype=dtypes.float32, depth=1)
        self.gm2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)
        self.fixpipe_state = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=2)
        self.fixpipe_u = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)

    def load_w_to_l1(self, w_l1, gm_tile):
        mem_copy(w_l1, gm_tile, engine=self.gm2l1)

    def load_q_decayed_to_l1(self, q_decayed_l1, gm_tile):
        mem_copy(q_decayed_l1, gm_tile, engine=self.gm2l1)

    def load_k_restored_to_l1(self, k_restored_l1, gm_tile):
        mem_copy(k_restored_l1, gm_tile, engine=self.gm2l1, l2_cache_ctl=_L2_CACHE_LAST_USE)

    def load_mqk_to_l1(self, mqk_l1, gm_tile):
        mem_copy(mqk_l1, gm_tile, engine=self.gm2l1, l2_cache_ctl=_L2_CACHE_LAST_USE)

    def load_state(self, state_l1):
        mem_copy(self.l0b_state, state_l1, transpose=False)

    def matmul_w_state(self, w_l1):
        mem_copy(self.l0a_chunk, w_l1)
        matmul(self.l0c_u, self.l0a_chunk, self.l0b_state, init=True)

    def matmul_q_state(self, q_decayed_l1):
        mem_copy(self.l0a_chunk, q_decayed_l1)
        matmul(self.l0c_o, self.l0a_chunk, self.l0b_state, init=True)

    def load_k_restored_to_l0(self, k_restored_l1):
        mem_copy(self.l0b_kr, k_restored_l1, transpose=True)

    def load_mqk_to_l0(self, mqk_l1):
        mem_copy(self.l0a_mqk, mqk_l1)

    def matmul_state_delta(self, u_slot):
        u_src = local_slice(u_slot, (CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0a_kr_t, u_src, transpose=True)
        matmul(self.l0c_state, self.l0a_kr_t, self.l0b_kr, init=True)

    def matmul_mqk_u(self, u_slot):
        u_tile = local_slice(u_slot, (CHUNK_SIZE, self.dv_base), offset=0)
        mem_copy(self.l0b_u, u_tile, transpose=True)
        matmul(self.l0c_o, self.l0a_mqk, self.l0b_u, init=False)

    def store_u_delta(self, ub):
        mem_copy(ub, local_slice(self.l0c_u, (CHUNK_SIZE, self.dv_base), offset=0), engine=self.fixpipe_u)

    def store_output(self, gm_tile):
        # FIXPIPE 按目标 GM 的实际行数写入逻辑尾块。
        mem_copy(gm_tile, local_slice(self.l0c_o, (CHUNK_SIZE, self.dv_base), offset=0),
            l2_cache_ctl=_L2_CACHE_DISABLE)

    def store_state_delta(self, ub):
        shape = (self.dv_base, SUPPORTED_HEAD_DIM)
        split_n = make_partition_tiler(shape, shape, axis=1)
        mem_copy(ub, local_slice(self.l0c_state, shape, offset=0), engine=self.fixpipe_state, partition=split_n)


class StageTwoVector:
    def __init__(self, *, u_rows_per_subblock: int, dv_base: int):
        self.dv_base = dv_base
        self.u_rows_per_subblock = u_rows_per_subblock
        self.half_dk = D_HALF
        self.ub_state_fp32 = Buffer(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32)
        self.tmp_state_bf16 = Buffer(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16)
        self.tmp_u_bf16 = Buffer(MemLoc.UB, (u_rows_per_subblock, dv_base), dtypes.bfloat16)
        self.state_nd = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1)
        self.state_nz = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16)
        self.u_nz = Channel(MemLoc.UB, (u_rows_per_subblock, dv_base), dtypes.bfloat16, depth=1, data_format="nz", n1_pad=16)
        self.state_in = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1)
        self.state_out = Channel(MemLoc.UB, (self.dv_base, self.half_dk), dtypes.float32, depth=1)
        self.u_in = Channel(MemLoc.UB, (u_rows_per_subblock, dv_base), dtypes.bfloat16, depth=1)
        self.ub2l1 = make_copy_engine(format_transform="nd2nz", dtype=dtypes.bfloat16, pad_value=0.0)

    def load_gamma_col(self, gamma_ub, gamma_col):
        slot = gamma_ub
        ub_gamma = local_slice(slot, (self.half_dk, 1), offset=0)
        mem_copy(ub_gamma, gamma_col)

    def _produce_state_l1(self, state_l1, subblock_idx):
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        nd_slot = self.state_nd
        with vf(outputs=[self.state_nd]):
            _cast_state_bf16(nd_slot, ub_state)
        nd_read = self.state_nd
        nz_slot = self.state_nz
        with vf():
            mem_copy(nz_slot, nd_read, engine=self.ub2l1)
        nz_read = self.state_nz
        slot = state_l1
        state_l1_rows = tile_view(slot, (self.dv_base, self.half_dk), (0, subblock_idx))
        mem_copy(state_l1_rows, nz_read)

    def load_initial_state_and_write_l1(self, current_state, state_l1, subblock_idx):
        in_slot = self.state_in
        mem_copy(in_slot, current_state)
        in_read = self.state_in
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        with vf():
            mem_copy(ub_state, in_read)
        self._produce_state_l1(state_l1, subblock_idx)

    def write_state_l1_snapshot(self, state_l1, subblock_idx):
        # 将累积状态写入 L1，供下一次 Cube 计算读取。
        self._produce_state_l1(state_l1, subblock_idx)

    @jit
    def scale_and_add_delta(self, tmp_delta, gamma_ub):
        # 在一次完整状态遍历中融合状态衰减和增量累加。
        gamma_read = gamma_ub
        ub_gamma = local_slice(gamma_read, (1, self.half_dk), offset=0)
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        delta = tmp_delta
        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            gamma_vec = vload(ub_gamma, 0)
            for base in dsl_range(0, self.dv_base * self.half_dk, VL, unroll=4):
                state_v = vload(ub_state, base)
                delta_v = vload(delta, base)
                state_scaled = vmul(state_v, gamma_vec, mask=mask)
                state_next = vadd(state_scaled, delta_v, mask=mask)
                vstore(ub_state, base, state_next, mask)

    def issue_u_rows(self, gm_u_tile, subblock_idx):
        gm_u_rows = tile_view(gm_u_tile, (self.u_rows_per_subblock, self.dv_base), (subblock_idx, 0))
        mem_copy(self.u_in, gm_u_rows)

    @jit
    def finish_u_delta_to_l1(self, u_l1, tmp_u_delta, subblock_idx):
        u_l1_rows = tile_view(u_l1, (self.u_rows_per_subblock, self.dv_base), (subblock_idx, 0))
        tmp_u_bf16 = local_slice(self.tmp_u_bf16, (self.u_rows_per_subblock, self.dv_base), offset=0)

        # 算术 VF 直接消费 MTE2 Channel，避免一次完整 UB 中转复制。
        u_in_read = self.u_in
        delta = tmp_u_delta
        with vf(mode="raw"):
            mask, _ = update_mask(VL, elem_bits=32)
            for base in range(0, self.u_rows_per_subblock * self.dv_base, 2 * VL):
                u_raw_pre = vload_unpack(u_in_read, base, unpack_mode=UnpackMode.B16_TO_B32)
                u_raw_post = vload_unpack(u_in_read, base + VL, unpack_mode=UnpackMode.B16_TO_B32)
                u_fp32_pre = vcast(u_raw_pre, dtypes.float32, mask=mask)
                u_fp32_post = vcast(u_raw_post, dtypes.float32, mask=mask)
                delta_pre = vload(delta, base)
                delta_post = vload(delta, base + VL)
                u_next_pre = vadd(u_fp32_pre, delta_pre, mask=mask)
                u_next_post = vadd(u_fp32_post, delta_post, mask=mask)
                u_bf16_pre = vcast(u_next_pre, dtypes.bfloat16, mask=mask)
                u_bf16_post = vcast(u_next_post, dtypes.bfloat16, mask=mask)
                vstore_pack(tmp_u_bf16, base, u_bf16_pre, mask, pack_mode=PackMode.B32_TO_B16)
                vstore_pack(tmp_u_bf16, base + VL, u_bf16_post, mask, pack_mode=PackMode.B32_TO_B16)

        u_nz_slot = self.u_nz
        with vf():
            mem_copy(u_nz_slot, tmp_u_bf16, engine=self.ub2l1)
        u_nz_read = self.u_nz
        mem_copy(u_l1_rows, u_nz_read)

    def store_state_to_gm(self, gm_state_rows):
        # 末态累加在 V 流水完成，通过 Channel 交给 MTE3 写回 FP32 GM。
        ub_state = local_slice(self.ub_state_fp32, (self.dv_base, self.half_dk), offset=0)
        out_slot = self.state_out
        with vf():
            mem_copy(out_slot, ub_state)
        out_read = self.state_out
        mem_copy(gm_state_rows, out_read)


@jit
def _run_stage2_body(
    gm_k_restored,
    gm_gamma,
    gm_q_decayed,
    gm_w,
    gm_U,
    gm_O,
    gm_state_in,
    gm_state_out,
    gm_Mqk,
    *,
    dv_base: int,
    head_base: int,
    head_count: int,
    chunks_total: int,
    seq_len: int,
    group_start: int,
    group_count: int,
    eff_group: int,
    logical_core_num: int,
    item_start: int,
    item_end: int,
    workspace_task_base: int,
):

    block_idx = get_block_idx()
    subblock_idx = get_subblock_id()
    num_dv_base = SUPPORTED_HEAD_DIM // dv_base
    half_dk = D_HALF
    u_rows_per_subblock = CHUNK_SIZE // 2

    q_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
    w_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2)
    k_restored_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2, data_format="nz")
    mqk_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtypes.bfloat16, depth=2)
    state_l1 = Channel(MemLoc.L1, shape=(dv_base, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16,
        depth=2, data_format="nz", kind=ChannelKind.CrossCore)
    u_l1 = Channel(MemLoc.L1, shape=(CHUNK_SIZE, SUPPORTED_HEAD_DIM), dtype=dtypes.bfloat16, depth=2, kind=ChannelKind.CrossCore)
    s2_matmul = StageTwoMatmul(dv_base=dv_base)
    gamma_ub = Channel(MemLoc.UB, shape=(half_dk, 1), dtype=dtypes.float32, depth=2)
    tmp_delta = Channel(MemLoc.UB, shape=(dv_base, half_dk), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore)
    tmp_u_delta = Channel(MemLoc.UB, shape=(u_rows_per_subblock, dv_base), dtype=dtypes.float32, depth=1, kind=ChannelKind.CrossCore)

    s2_vector = StageTwoVector(u_rows_per_subblock=u_rows_per_subblock, dv_base=dv_base)


    if block_idx < logical_core_num:
        for item in range(item_start, item_end):
            head_id, dv_idx = idx2crd(item, [head_count, num_dv_base])
            seq_idx = head_base + head_id
            write_state_ref = tile_view(gm_state_out[0, seq_idx, None, None], (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), (0, 0))

            first_linear = workspace_task_base + seq_idx * eff_group
            first_w_chunk = tile_view(gm_w, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_q_chunk = tile_view(gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_k_restored_chunk = tile_view(gm_k_restored, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (first_linear, 0))
            first_mqk_tile = tile_view(gm_Mqk, (CHUNK_SIZE, CHUNK_SIZE), (first_linear, 0))
            first_gamma_col_full = tile_view(gm_gamma, (SUPPORTED_HEAD_DIM, 1), (first_linear, 0))
            first_gamma_col = tile_view(first_gamma_col_full, (half_dk, 1), (subblock_idx, 0))
            s2_matmul.load_w_to_l1(w_l1, first_w_chunk)
            s2_matmul.load_q_decayed_to_l1(q_l1, first_q_chunk)
            s2_matmul.load_k_restored_to_l1(k_restored_l1, first_k_restored_chunk)
            s2_matmul.load_mqk_to_l1(mqk_l1, first_mqk_tile)
            if group_start == 0:
                in_state_ref = tile_view(gm_state_in[0, seq_idx, None, None], (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), (0, 0))
                first_state_rows = tile_view(in_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            else:
                first_state_rows = tile_view(write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                s2_vector.load_initial_state_and_write_l1(first_state_rows, state_l1, subblock_idx)
            s2_vector.load_gamma_col(gamma_ub, first_gamma_col)

            for chunk_id in range(0, group_count):
                k = group_start + chunk_id
                linear = workspace_task_base + seq_idx * eff_group + chunk_id
                u_tile_full = tile_view(gm_U, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (linear, 0))
                u_tile = tile_view(u_tile_full, (CHUNK_SIZE, dv_base), (0, dv_idx))
                valid_rows = CHUNK_SIZE
                if k + 1 == chunks_total:
                    valid_rows = seq_len - k * CHUNK_SIZE
                o_chunk = _o_head_chunk_tile(gm_O, seq_idx, k, dv_base, dv_idx, valid_rows)
                mqk_tile = tile_view(gm_Mqk, (CHUNK_SIZE, CHUNK_SIZE), (linear, 0))

                s2_vector.issue_u_rows(u_tile, subblock_idx)
                s2_matmul.load_state(state_l1)
                s2_matmul.matmul_w_state(w_l1)
                s2_matmul.matmul_q_state(q_l1)
                s2_matmul.load_k_restored_to_l0(k_restored_l1)
                s2_matmul.load_mqk_to_l0(mqk_l1)
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_q_chunk = tile_view(gm_q_decayed, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_w_chunk = tile_view(gm_w, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_gamma_col_full = tile_view(gm_gamma, (SUPPORTED_HEAD_DIM, 1), (next_linear, 0))
                    next_gamma_col = tile_view(next_gamma_col_full, (half_dk, 1), (subblock_idx, 0))
                    s2_matmul.load_q_decayed_to_l1(q_l1, next_q_chunk)
                    s2_matmul.load_w_to_l1(w_l1, next_w_chunk)
                    s2_vector.load_gamma_col(gamma_ub, next_gamma_col)
                s2_matmul.store_u_delta(tmp_u_delta)

                s2_vector.finish_u_delta_to_l1(u_l1, tmp_u_delta, subblock_idx)

                s2_matmul.matmul_state_delta(u_l1)
                s2_matmul.store_state_delta(tmp_delta)

                s2_matmul.matmul_mqk_u(u_l1)
                if chunk_id + 1 < group_count:
                    next_linear = linear + 1
                    next_k_restored_chunk = tile_view(gm_k_restored, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (next_linear, 0))
                    next_mqk_tile = tile_view(gm_Mqk, (CHUNK_SIZE, CHUNK_SIZE), (next_linear, 0))
                    s2_matmul.load_k_restored_to_l1(k_restored_l1, next_k_restored_chunk)
                    s2_matmul.load_mqk_to_l1(mqk_l1, next_mqk_tile)
                s2_matmul.store_output(o_chunk)

                s2_vector.scale_and_add_delta(tmp_delta, gamma_ub)
                if chunk_id + 1 < group_count:
                    s2_vector.write_state_l1_snapshot(state_l1, subblock_idx)
                else:
                    state_rows = tile_view(write_state_ref, (dv_base, half_dk), (dv_idx, subblock_idx))
                    s2_vector.store_state_to_gm(state_rows)


# mode=0 表示全栅格同步；metadata 将启动核数限制为常驻 Cube 核数，保证所有核均能到达栅栏。
STAGE1_BOUNDARY_FLAG_ID = 0
STAGE2_BOUNDARY_FLAG_ID = 1
_STAGE_BOUNDARY_AIC_FLAG_OFFSET = 4
_STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET = 8
QK_PUBLISH_FLAG_ID = 10
QK_CONSUME_FLAG_ID = 11
MTE2_TO_V_EVENT_ID = 0
MTE2_TO_V_V_STAGING_EVENT_ID = 1
MTE2_TO_V_SUBBLOCK1_EVENT_ID = 4
QK_V_TO_MTE3_EVENT_ID = 2
QK_MTE2_TO_V_EVENT_ID = 3
GATE_MTE2_TO_V_EVENT_ID = 5


def _rewind_buffers():
    channel_rewind(reset_sync_id=False)


def _stage_boundary_reset(flag_id: int):
    cube_sync_all()
    vec_sync_all()

    aiv_ready_flag_id = flag_id
    aic_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_FLAG_OFFSET
    aic_aiv_ready_flag_id = flag_id + _STAGE_BOUNDARY_AIC_AIV_FLAG_OFFSET

    vec_sync_block_arrive(PIPE.MTE3, aiv_ready_flag_id, mode=2)
    cube_sync_block_wait(PIPE.S, aiv_ready_flag_id, mode=2)

    cube_sync_block_arrive(PIPE.FIXPIPE, aic_ready_flag_id, mode=0)
    cube_sync_block_wait(PIPE.S, aic_ready_flag_id, mode=0)

    cube_sync_block_arrive(PIPE.MTE3, aic_aiv_ready_flag_id, mode=2)
    vec_sync_block_wait(PIPE.S, aic_aiv_ready_flag_id, mode=2)
    _rewind_buffers()


from ...flash_kda_metadata.op_kernel.flash_kda_metadata import (
    HEADER_CORE_NUM,
    HEADER_STAGE12_ROUND_NUM,
    HEADER_STATUS,
    HEADER_WORDS,
    MAX_AIC_CORES,
    STATUS_OK,
    WORKSPACE_SLOTS,
    _device_block_num,
)


LOW_DTYPE = torch.bfloat16
HIGH_DTYPE = torch.float32


def _reset_stage2_channel_plan(sync_id_base: int) -> None:
    # 四个互斥分支从同一片上地址和同步 ID 起点规划 Channel。
    arena = _current_channel_arena()
    arena.rewind(reset_sync_id=True)
    arena._sync_id = int(sync_id_base)


@kernel
class flash_kda_kernel:
    def __call__(
        self,
        gm_K: Tensor,
        gm_V: Tensor,
        gm_Q: Tensor,
        gm_beta: Tensor,
        gm_g: Tensor,
        gm_qk_exchange: Tensor,
        gm_A_log: Tensor,
        gm_dt_bias: Tensor,
        gm_k_restored: Tensor,
        gm_gamma: Tensor,
        gm_q_decayed: Tensor,
        gm_w: Tensor,
        gm_Mqk: Tensor,
        gm_U: Tensor,
        gm_O: Tensor,
        gm_state_in: Tensor,
        gm_state_out: Tensor,
        gm_lower_mask_tiles: Tensor,
        gm_identity16: Tensor,
        gm_metadata: Tensor,
        scale_value: float,
        lower_bound: float,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        metadata_status = dtypes.int64(gm_metadata[HEADER_STATUS])
        metadata_core_num = dtypes.int64(gm_metadata[HEADER_CORE_NUM])
        if metadata_status == dtypes.int64(STATUS_OK) and metadata_core_num == block_num:
            stage12_round_num = dtypes.int64(gm_metadata[HEADER_STAGE12_ROUND_NUM])
            for global_round_idx in range(dtypes.int64(0), stage12_round_num):
                record = dtypes.int64(gm_metadata[dtypes.int64(HEADER_WORDS) + global_round_idx])
                active_batch_num = dtypes.int64(gm_metadata[record + dtypes.int64(1)])
                batch_idx_base = record + dtypes.int64(2)
                storage_batch_idx_base = batch_idx_base + active_batch_num
                token_start_base = storage_batch_idx_base + active_batch_num
                valid_seq_len_base = token_start_base + active_batch_num
                group_num_index = valid_seq_len_base + active_batch_num
                chunk_start_base = group_num_index + dtypes.int64(1)
                chunk_num_base = chunk_start_base + active_batch_num
                stage1_task_prefix_base = chunk_num_base + active_batch_num
                stage1_core_ranges_base = stage1_task_prefix_base + active_batch_num + dtypes.int64(1)
                dv_splits_num_base = stage1_core_ranges_base + dtypes.int64(MAX_AIC_CORES * 2)
                stage2_task_prefix_base = dv_splits_num_base + active_batch_num
                stage2_core_ranges_base = stage2_task_prefix_base + active_batch_num + dtypes.int64(1)

                stage1_core_range_word = stage1_core_ranges_base + block_idx * dtypes.int64(2)
                stage1_core_start = dtypes.int64(gm_metadata[stage1_core_range_word])
                stage1_core_end = dtypes.int64(gm_metadata[stage1_core_range_word + dtypes.int64(1)])
                for active_batch_idx in range(dtypes.int64(0), active_batch_num):
                    prefix_start = dtypes.int64(gm_metadata[stage1_task_prefix_base + active_batch_idx])
                    prefix_end = dtypes.int64(gm_metadata[stage1_task_prefix_base + active_batch_idx + dtypes.int64(1)])
                    intersection_start = stage1_core_start
                    if intersection_start < prefix_start:
                        intersection_start = prefix_start
                    intersection_end = stage1_core_end
                    if intersection_end > prefix_end:
                        intersection_end = prefix_end
                    task_start = dtypes.int64(0)
                    task_end = dtypes.int64(0)
                    if intersection_start < intersection_end:
                        task_start = intersection_start - prefix_start
                        task_end = intersection_end - prefix_start

                    logical_batch = dtypes.int64(gm_metadata[batch_idx_base + active_batch_idx])
                    storage_batch = dtypes.int64(gm_metadata[storage_batch_idx_base + active_batch_idx])
                    token_start = dtypes.int64(gm_metadata[token_start_base + active_batch_idx])
                    seq_len = dtypes.int64(gm_metadata[valid_seq_len_base + active_batch_idx])
                    group_start = dtypes.int64(gm_metadata[chunk_start_base + active_batch_idx])
                    chunk_num_per_group = dtypes.int64(gm_metadata[chunk_num_base + active_batch_idx])
                    chunks_total = (seq_len + dtypes.int64(CHUNK_SIZE - 1)) // dtypes.int64(CHUNK_SIZE)
                    tail_rows = seq_len - (chunks_total - dtypes.int64(1)) * dtypes.int64(CHUNK_SIZE)
                    tail_group = group_start + chunk_num_per_group == chunks_total

                    token_stop = token_start + seq_len
                    k = gm_K[storage_batch, None, token_start:token_stop, None]
                    v = gm_V[storage_batch, None, token_start:token_stop, None]
                    q = gm_Q[storage_batch, None, token_start:token_stop, None]
                    beta = gm_beta[storage_batch, None, token_start:token_stop, None]
                    gate = gm_g[storage_batch, None, token_start:token_stop, None]
                    workspace_slot_start = prefix_start
                    value_heads = v.shape[1]
                    _run_stage1_body(tail_group, tail_rows, k, v, q, beta, gate, gm_qk_exchange,
                        gm_A_log, gm_dt_bias, gm_q_decayed, gm_Mqk, gm_k_restored, gm_gamma, gm_U, gm_w,
                        gm_lower_mask_tiles, gm_identity16, value_heads, value_heads, chunks_total,
                        seq_len, dtypes.int64(0), group_start, chunk_num_per_group,
                        chunk_num_per_group, block_num, task_start, task_end, workspace_slot_start, scale_value, lower_bound)

                _stage_boundary_reset(STAGE1_BOUNDARY_FLAG_ID)

                stage2_core_range_word = stage2_core_ranges_base + block_idx * dtypes.int64(2)
                stage2_core_start = dtypes.int64(gm_metadata[stage2_core_range_word])
                stage2_core_end = dtypes.int64(gm_metadata[stage2_core_range_word + dtypes.int64(1)])
                round_dv_splits_num = dtypes.int64(gm_metadata[dv_splits_num_base])
                stage2_sync_id_base = _current_channel_arena()._sync_id
                for active_batch_idx in range(dtypes.int64(0), active_batch_num):
                    prefix_start = dtypes.int64(gm_metadata[stage2_task_prefix_base + active_batch_idx])
                    prefix_end = dtypes.int64(gm_metadata[stage2_task_prefix_base + active_batch_idx + dtypes.int64(1)])
                    intersection_start = stage2_core_start
                    if intersection_start < prefix_start:
                        intersection_start = prefix_start
                    intersection_end = stage2_core_end
                    if intersection_end > prefix_end:
                        intersection_end = prefix_end
                    item_start = dtypes.int64(0)
                    item_end = dtypes.int64(0)
                    if intersection_start < intersection_end:
                        item_start = intersection_start - prefix_start
                        item_end = intersection_end - prefix_start

                    logical_batch = dtypes.int64(gm_metadata[batch_idx_base + active_batch_idx])
                    storage_batch = dtypes.int64(gm_metadata[storage_batch_idx_base + active_batch_idx])
                    token_start = dtypes.int64(gm_metadata[token_start_base + active_batch_idx])
                    seq_len = dtypes.int64(gm_metadata[valid_seq_len_base + active_batch_idx])
                    group_start = dtypes.int64(gm_metadata[chunk_start_base + active_batch_idx])
                    chunk_num_per_group = dtypes.int64(gm_metadata[chunk_num_base + active_batch_idx])
                    chunks_total = (seq_len + dtypes.int64(CHUNK_SIZE - 1)) // dtypes.int64(CHUNK_SIZE)

                    token_stop = token_start + seq_len
                    o = gm_O[storage_batch, None, token_start:token_stop, None]
                    state_in = gm_state_in[logical_batch, None, None, None]
                    state_out = gm_state_out[logical_batch, None, None, None]
                    workspace_slot_start = dtypes.int64(gm_metadata[stage1_task_prefix_base + active_batch_idx])
                    if round_dv_splits_num == 1:
                        _reset_stage2_channel_plan(stage2_sync_id_base)
                        _run_stage2_body(gm_k_restored, gm_gamma, gm_q_decayed, gm_w, gm_U, o, state_in, state_out, gm_Mqk,
                            dv_base=128, head_base=dtypes.int64(0), head_count=gm_V.shape[1],
                            chunks_total=chunks_total, seq_len=seq_len, group_start=group_start,
                            group_count=chunk_num_per_group, eff_group=chunk_num_per_group,
                            logical_core_num=block_num, item_start=item_start, item_end=item_end,
                            workspace_task_base=workspace_slot_start)
                    elif round_dv_splits_num == 2:
                        _reset_stage2_channel_plan(stage2_sync_id_base)
                        _run_stage2_body(gm_k_restored, gm_gamma, gm_q_decayed, gm_w, gm_U, o, state_in, state_out, gm_Mqk,
                            dv_base=64, head_base=dtypes.int64(0), head_count=gm_V.shape[1],
                            chunks_total=chunks_total, seq_len=seq_len, group_start=group_start,
                            group_count=chunk_num_per_group, eff_group=chunk_num_per_group,
                            logical_core_num=block_num, item_start=item_start, item_end=item_end,
                            workspace_task_base=workspace_slot_start)
                    elif round_dv_splits_num == 4:
                        _reset_stage2_channel_plan(stage2_sync_id_base)
                        _run_stage2_body(gm_k_restored, gm_gamma, gm_q_decayed, gm_w, gm_U, o, state_in, state_out, gm_Mqk,
                            dv_base=32, head_base=dtypes.int64(0), head_count=gm_V.shape[1],
                            chunks_total=chunks_total, seq_len=seq_len, group_start=group_start,
                            group_count=chunk_num_per_group, eff_group=chunk_num_per_group,
                            logical_core_num=block_num, item_start=item_start, item_end=item_end,
                            workspace_task_base=workspace_slot_start)
                    else:
                        _reset_stage2_channel_plan(stage2_sync_id_base)
                        _run_stage2_body(gm_k_restored, gm_gamma, gm_q_decayed, gm_w, gm_U, o, state_in, state_out, gm_Mqk,
                            dv_base=16, head_base=dtypes.int64(0), head_count=gm_V.shape[1],
                            chunks_total=chunks_total, seq_len=seq_len, group_start=group_start,
                            group_count=chunk_num_per_group, eff_group=chunk_num_per_group,
                            logical_core_num=block_num, item_start=item_start, item_end=item_end,
                            workspace_task_base=workspace_slot_start)

                if global_round_idx + dtypes.int64(1) < stage12_round_num:
                    _stage_boundary_reset(STAGE2_BOUNDARY_FLAG_ID)


class FlashKDA:
    @jit
    def run(
        self,
        K: Tensor,
        V: Tensor,
        Q: Tensor,
        beta: Tensor,
        g: Tensor,
        qk_exchange: Tensor,
        A_log: Tensor,
        dt_bias: Tensor,
        kr: Tensor,
        gamma: Tensor,
        qd: Tensor,
        w: Tensor,
        mqk: Tensor,
        u: Tensor,
        out: Tensor,
        state_in: Tensor,
        state_out: Tensor,
        masks: Tensor,
        identity: Tensor,
        metadata: Tensor,
        scale: float,
        lower_bound: float,
        core_num: int,
    ):
        op = flash_kda_kernel()
        op[core_num](K, V, Q, beta, g, qk_exchange, A_log, dt_bias, kr, gamma, qd, w, mqk, u,
            out, state_in, state_out, masks, identity, metadata, scale, lower_bound)


_COMPILED_KERNEL = None
_COMPILED_KERNEL_LOCK = threading.Lock()


def _sequence_spec(dtype, physical_batches, length, heads, width, stride_prefix):
    stride_0 = cannbotdsl.Dim(f"{stride_prefix}_S0")
    stride_1 = cannbotdsl.Dim(f"{stride_prefix}_S1")
    stride_2 = cannbotdsl.Dim(f"{stride_prefix}_S2")
    return cannbotdsl.TensorSpec((physical_batches, heads, length, width), dtype, stride=(stride_0, stride_1, stride_2, 1))


def _get_compiled_kernel():
    global _COMPILED_KERNEL
    with _COMPILED_KERNEL_LOCK:
        if _COMPILED_KERNEL is not None:
            return _COMPILED_KERNEL

        physical_batches = cannbotdsl.Dim("P")
        storage_length = cannbotdsl.Dim("L")
        output_length = cannbotdsl.Dim("OL")
        batch = cannbotdsl.Dim("B")
        n_qk = cannbotdsl.Dim("NQK")
        n_v = cannbotdsl.Dim("NV")
        metadata_capacity = cannbotdsl.Dim("META")
        core_num = cannbotdsl.Dim("CORE_NUM", min=1, max=MAX_AIC_CORES)
        scratch_tokens = WORKSPACE_SLOTS * CHUNK_SIZE
        gamma_rows = WORKSPACE_SLOTS * SUPPORTED_HEAD_DIM
        fake = cannbotdsl.TensorSpec
        qk_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_qk, SUPPORTED_HEAD_DIM, "QK")
        v_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_v, SUPPORTED_HEAD_DIM, "V")
        beta_spec = _sequence_spec(dtypes.bfloat16, physical_batches, storage_length, n_v, 1, "BETA")
        output_spec = _sequence_spec(dtypes.bfloat16, physical_batches, output_length, n_v, SUPPORTED_HEAD_DIM, "OUT")
        state_spec = fake((batch, n_v, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtypes.float32)
        _COMPILED_KERNEL = FlashKDA().run.compile(qk_spec, v_spec, qk_spec, beta_spec, v_spec,
            fake((core_num, 2, CHUNK_SIZE, D_HALF), dtypes.float32),
            fake((n_v,), dtypes.float32),
            fake((n_v, SUPPORTED_HEAD_DIM), dtypes.float32),
            fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
            fake((gamma_rows, 1), dtypes.float32),
            fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
            fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
            fake((scratch_tokens, CHUNK_SIZE), dtypes.bfloat16),
            fake((scratch_tokens, SUPPORTED_HEAD_DIM), dtypes.bfloat16),
            output_spec,
            state_spec,
            state_spec,
            fake((CHUNK_SIZE, CHUNK_SIZE // 2), dtypes.float32),
            fake((16, 16), dtypes.float16),
            fake((metadata_capacity,), dtypes.int32), dtypes.float32, dtypes.float32, dtypes.int64)
        return _COMPILED_KERNEL


def flash_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    layout_qkv: str,
    metadata: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert layout_qkv in ("TND", "BNSD", "BSND"), f"layout_qkv must be TND, BNSD, or BSND, got {layout_qkv!r}"
    assert initial_state.dim() == 4, "initial_state must be rank-4"
    batch, n_v, state_dv, state_dk = initial_state.shape
    assert state_dv == state_dk == SUPPORTED_HEAD_DIM, f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"

    if layout_qkv == "TND":
        assert q.dim() == k.dim() == v.dim() == g.dim() == 3, "TND q/k/v/g must be rank-3"
        storage_length, n_qk, dim = q.shape
        assert k.shape == q.shape, "q/k shape does not match TND layout"
        assert v.shape == g.shape == (storage_length, n_v, dim), "v/g shape does not match TND layout"
        assert beta.dim() == 2 and beta.shape == (storage_length, n_v), "beta must be [T, Nv] for TND"
        q_logical = torch.as_strided(q, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1))
        k_logical = torch.as_strided(k, (1, n_qk, storage_length, dim), (storage_length * n_qk * dim, dim, n_qk * dim, 1))
        v_logical = torch.as_strided(v, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        g_logical = torch.as_strided(g, (1, n_v, storage_length, dim), (storage_length * n_v * dim, dim, n_v * dim, 1))
        beta_logical = torch.as_strided(beta, (1, n_v, storage_length, 1), (storage_length * n_v, 1, n_v, 1))
        padded_length = ((storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
        out_stage_storage = torch.empty(padded_length, n_v, dim, dtype=LOW_DTYPE, device=q.device)
        out = out_stage_storage[:storage_length]
        out_stage = torch.as_strided(out_stage_storage, (1, n_v, padded_length, dim), (padded_length * n_v * dim, dim, n_v * dim, 1))
    else:
        assert q.dim() == k.dim() == v.dim() == g.dim() == 4, "BNSD/BSND q/k/v/g must be rank-4"
        if layout_qkv == "BNSD":
            physical_batches, n_qk, storage_length, dim = q.shape
            assert k.shape == q.shape, "q/k shape does not match BNSD layout"
            assert v.shape == g.shape == (physical_batches, n_v, storage_length, dim), "v/g shape does not match BNSD layout"
            assert beta.dim() == 3 and beta.shape == (physical_batches, n_v, storage_length), "beta must be [B, Nv, S] for BNSD"
            q_logical, k_logical, v_logical, g_logical = q, k, v, g
            beta_logical = beta.unsqueeze(-1)
        else:
            physical_batches, storage_length, n_qk, dim = q.shape
            assert k.shape == q.shape, "q/k shape does not match BSND layout"
            assert v.shape == g.shape == (physical_batches, storage_length, n_v, dim), "v/g shape does not match BSND layout"
            assert beta.dim() == 3 and beta.shape == (physical_batches, storage_length, n_v), "beta must be [B, S, Nv] for BSND"
            q_logical = q.transpose(1, 2)
            k_logical = k.transpose(1, 2)
            v_logical = v.transpose(1, 2)
            g_logical = g.transpose(1, 2)
            beta_logical = beta.unsqueeze(-1).transpose(1, 2)
        assert physical_batches == batch, "padded storage batch must match initial_state"
        padded_length = ((storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
        if layout_qkv == "BNSD":
            out_stage_storage = torch.empty(batch, n_v, padded_length, dim, dtype=LOW_DTYPE, device=q.device)
            out = out_stage_storage[:, :, :storage_length, :]
            out_stage = out_stage_storage
        else:
            out_stage_storage = torch.empty(batch, padded_length, n_v, dim, dtype=LOW_DTYPE, device=q.device)
            out = out_stage_storage[:, :storage_length, :, :]
            out_stage = out_stage_storage.transpose(1, 2)

    assert dim == SUPPORTED_HEAD_DIM, f"FlashKDA only supports D={SUPPORTED_HEAD_DIM}"
    assert n_v % n_qk == 0, f"GQA requires Nv % Nqk == 0, got Nv={n_v}, Nqk={n_qk}"
    assert storage_length > 0, "storage sequence length must be positive"
    assert q.dtype == k.dtype == v.dtype == g.dtype == beta.dtype == LOW_DTYPE, "q/k/v/g/beta must be bf16"
    assert initial_state.dtype == HIGH_DTYPE, "initial_state must be fp32"
    assert A_log.dtype == HIGH_DTYPE and A_log.shape == (n_v,), f"A_log must be fp32 with shape ({n_v},)"
    assert dt_bias.dtype == HIGH_DTYPE and dt_bias.shape == (n_v, dim), f"dt_bias must be fp32 with shape ({n_v}, {dim})"
    assert -5.0 <= float(lower_bound) <= 0.0, "lower_bound must be in [-5, 0]"
    tensors = (q, k, v, g, beta, initial_state, A_log, dt_bias)
    assert all(tensor.device == q.device for tensor in tensors), "all inputs must be on the same device"
    assert all(tensor.is_contiguous() for tensor in tensors), "all public inputs must be contiguous"
    assert metadata.dtype == torch.int32, "metadata must be int32"
    assert metadata.dim() == 1, "metadata must be rank-1"
    assert metadata.is_contiguous(), "metadata must be contiguous"
    assert metadata.device == q.device, "all inputs must be on the same device"

    core_num = _device_block_num(q)
    workspace = allocate_stage1_workspace(1, 1, WORKSPACE_SLOTS, SUPPORTED_HEAD_DIM, ref=q)
    qk_exchange = allocate_qk_exchange_workspace(active_block_num=core_num, ref=q)
    final_state = torch.empty(initial_state.shape, dtype=initial_state.dtype, device=initial_state.device)
    fn = _get_compiled_kernel()
    fn(k_logical, v_logical, q_logical, beta_logical, g_logical, qk_exchange, A_log, dt_bias,
        workspace.K_restored, workspace.gamma_C, workspace.Q_decayed, workspace.W, workspace.Mqk,
        workspace.U_pre, out_stage, initial_state, final_state, workspace.lower_mask_tiles,
        workspace.identity16, metadata, float(scale), float(lower_bound), core_num)
    return out, final_state


# Native publication only compiles the existing dynamic entry; it does not launch it.
from cannbotdsl.package.native import register


@register("flash_kda")
def export_flash_kda():
    global _COMPILED_KERNEL
    try:
        _get_compiled_kernel()
    finally:
        with _COMPILED_KERNEL_LOCK:
            if _COMPILED_KERNEL is not None:
                _COMPILED_KERNEL.close()
                _COMPILED_KERNEL = None


def build_native(output, target):
    from pathlib import Path
    from cannbotdsl.package.native import build

    return build(collectors=Path(__file__).resolve().parents[2], operators=["flash_kda"],
                 pattern="flash_kda.py", target=target, output=output)
