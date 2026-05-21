#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# flash_attn_metadata Python Simulator
# Replicates the AICPU SectionStreamK load-balance algorithm for KT simulation mode.
# ======================================================================================================================

"""
flash_attn_metadata_sim.py
===========================
Pure-Python implementation of the AICPU `npu_flash_attn_metadata` operator.
Generates the same metadata buffer layout as the hardware AICPU kernel,
enabling KT simulation (仿真模式) without NPU hardware.

Usage:
    from flash_attn_metadata_sim import generate_metadata_cpu
    metadata_tensor = generate_metadata_cpu(...)
"""

import math
import numpy as np
from typing import List, Optional, Tuple

# ============ Constants (matching flash_attn_metadata.h) ============
AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
FA_METADATA_SIZE = 16  # uint32 slots per AIC core
FD_METADATA_SIZE = 16  # uint32 slots per AIV core
HEADER_SIZE = 16       # uint32 slots for header

# FA Metadata Index
FA_BN2_START_INDEX = 0
FA_M_START_INDEX = 1
FA_S2_START_INDEX = 2
FA_BN2_END_INDEX = 3
FA_M_END_INDEX = 4
FA_S2_END_INDEX = 5
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6

# FD Metadata Index
FD_BN2_IDX_INDEX = 0
FD_M_IDX_INDEX = 1
FD_WORKSPACE_IDX_INDEX = 2
FD_WORKSPACE_NUM_INDEX = 3
FD_M_START_INDEX = 4
FD_M_NUM_INDEX = 5

# Tiling constants
SOUTER_32 = 32
SOUTER_64 = 64
SINNER_128 = 128
SINNER_256 = 256
DSIZE_128 = 128
DSIZE_256 = 256

# SparseMode
SPARSE_DEFAULT_MASK = 0
SPARSE_ALL_MASK = 1
SPARSE_LEFT_UP_CAUSAL = 2
SPARSE_RIGHT_DOWN_CAUSAL = 3
SPARSE_BAND = 4
SPARSE_BUTT = 5  # no mask


# ============ AdjustSinnerAndSouter ============
def adjust_sinner_and_souter(v_head_dim: int, s1_size: int, s2_size: int,
                             mask_mode: int, win_left: int, win_right: int,
                             q_layout: int) -> Tuple[int, int]:
    """Determine sOuterFactor and sInnerFactor based on operator parameters.
    Returns (sOuterFactor, sInnerFactor).
    """
    s_outer_factor = SOUTER_64
    s_inner_factor = SINNER_128

    check_query_and_value_s = (s1_size <= SOUTER_64) and (s2_size > SINNER_128)

    if v_head_dim <= DSIZE_128:
        win_left_tmp = win_left
        win_right_tmp = win_right
        if mask_mode == 0:
            win_left_tmp = 0 if win_left_tmp > 0 else win_left_tmp
        elif mask_mode == 4:
            win_right_tmp = 0 if win_right_tmp > 0 else win_right_tmp
        check_sparse_mode = (mask_mode != 2) and (win_left_tmp + win_right_tmp > 128)
        if check_query_and_value_s and check_sparse_mode:
            s_outer_factor = SOUTER_32
            s_inner_factor = SINNER_256
    elif v_head_dim > DSIZE_128 and s1_size != 1:
        if check_query_and_value_s and v_head_dim <= DSIZE_256:
            s_outer_factor = SOUTER_32
            s_inner_factor = SINNER_256
        else:
            s_outer_factor = SOUTER_64
            s_inner_factor = SINNER_128
    elif s1_size == 1 and v_head_dim > DSIZE_128:
        s_outer_factor = SOUTER_64
        s_inner_factor = SINNER_128

    return s_outer_factor, s_inner_factor


# ============ Cost function ============
def calc_cost(basic_m: int, basic_s2: int) -> int:
    """Default cost function: 6 * ceil(M/16) + 10 * ceil(S2/64)"""
    align_m = (basic_m + 15) >> 4
    align_s2 = (basic_s2 + 63) >> 6
    return 6 * align_m + 10 * align_s2


def ceil_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    return (a + b - 1) // b


def num_to_index(val: int) -> int:
    return val - 1


# ============ BaseInfo helper ============
class BaseInfoSim:
    """Simulates the C++ BaseInfo/IBaseInfo interface."""

    def __init__(self, batch_size: int, query_head_num: int, kv_head_num: int,
                 head_dim: int, max_seqlen_q: int, max_seqlen_kv: int,
                 actual_query_seq: Optional[List[int]], actual_kv_seq: Optional[List[int]],
                 is_cumulative_q: bool, is_cumulative_kv: bool,
                 sparse_mode: int, pre_token: int, next_token: int,
                 layout_q: str, layout_kv: str):
        self.batch_size = batch_size
        self.query_head_num = query_head_num
        self.kv_head_num = kv_head_num if kv_head_num > 0 else query_head_num
        self.head_dim = head_dim
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.is_cumulative_q = is_cumulative_q
        self.is_cumulative_kv = is_cumulative_kv
        self.sparse_mode = sparse_mode
        self.pre_token = pre_token
        self.next_token = next_token
        self.layout_q = layout_q
        self.layout_kv = layout_kv

        # Actual seq sizes
        if actual_query_seq is not None:
            self.actual_query_seq = list(actual_query_seq)
        else:
            self.actual_query_seq = [max_seqlen_q] * batch_size

        if actual_kv_seq is not None:
            self.actual_kv_seq = list(actual_kv_seq)
        else:
            self.actual_kv_seq = [max_seqlen_kv] * batch_size

    @property
    def group_size(self) -> int:
        if self.kv_head_num == 0:
            return 1
        return self.query_head_num // self.kv_head_num

    def get_query_seq_size(self, batch_idx: int) -> int:
        if not self.actual_query_seq:
            return self.max_seqlen_q
        if not self.is_cumulative_q:
            return self.actual_query_seq[batch_idx]
        if batch_idx == 0:
            return self.actual_query_seq[0]
        return self.actual_query_seq[batch_idx] - self.actual_query_seq[batch_idx - 1]

    def get_kv_seq_size(self, batch_idx: int) -> int:
        if not self.actual_kv_seq:
            return self.max_seqlen_kv
        if not self.is_cumulative_kv:
            return self.actual_kv_seq[batch_idx]
        if batch_idx == 0:
            return self.actual_kv_seq[0]
        return self.actual_kv_seq[batch_idx] - self.actual_kv_seq[batch_idx - 1]

    def get_sparse_mode(self) -> int:
        """Returns SPARSE_BUTT if no mask."""
        if self.sparse_mode > SPARSE_BAND:
            return SPARSE_BUTT
        return self.sparse_mode

    def get_pre_token_left_up(self, query_seq: int, kv_seq: int) -> int:
        mode = self.get_sparse_mode()
        if mode == SPARSE_BAND:
            return query_seq - kv_seq + self.pre_token
        return self.pre_token

    def get_next_token_left_up(self, query_seq: int, kv_seq: int) -> int:
        mode = self.get_sparse_mode()
        if mode in (SPARSE_DEFAULT_MASK, SPARSE_ALL_MASK, SPARSE_LEFT_UP_CAUSAL):
            return self.next_token
        elif mode == SPARSE_RIGHT_DOWN_CAUSAL:
            return kv_seq - query_seq
        elif mode == SPARSE_BAND:
            return kv_seq - query_seq + self.next_token
        return self.next_token

    @property
    def is_s1g(self) -> bool:
        return self.layout_q in ("TND", "BSH", "BSND")


# ============ SectionStreamK Implementation ============
class SectionStreamKSim:
    """Pure-Python reimplementation of SectionStreamKImpl."""

    def __init__(self, m_base_size: int, s2_base_size: int, fd_on: bool = True,
                 fa_tolerance_ratio: int = 2, fd_tolerance: int = 0, v0_cost: int = 0):
        self.m_base_size = m_base_size
        self.s2_base_size = s2_base_size
        self.fd_on = fd_on
        self.fa_tolerance_ratio = fa_tolerance_ratio
        self.fd_tolerance = fd_tolerance
        self.v0_cost = v0_cost

    def compute(self, aic_core_num: int, aiv_core_num: int, base_info: BaseInfoSim):
        """Main entry: returns (fa_result, fd_result, is_fd)."""
        grid_info = self._calc_grid_info(base_info)

        # All-empty case
        if grid_info['is_empty']:
            fa_result = self._empty_fa_result(aic_core_num, base_info)
            fd_result = self._empty_fd_result(aiv_core_num)
            return fa_result, fd_result, False

        cost_info = self._calc_cost_info(base_info, grid_info)
        if cost_info['block_num'] == 0:
            fa_result = self._empty_fa_result(aic_core_num, base_info)
            fd_result = self._empty_fd_result(aiv_core_num)
            return fa_result, fd_result, False

        return self._schedule_section(aic_core_num, aiv_core_num, base_info, grid_info, cost_info)

    def _empty_fa_result(self, aic_core_num, base_info):
        fa = {
            'used_core_num': 1,
            'bN2End': [0] * aic_core_num,
            'gS1End': [0] * aic_core_num,
            's2End': [0] * aic_core_num,
            'firstFdDataWorkspaceIdx': [0] * aic_core_num,
        }
        fa['bN2End'][0] = num_to_index(base_info.batch_size * base_info.query_head_num)
        return fa

    def _empty_fd_result(self, aiv_core_num):
        return {
            'used_vec_num': 0,
            'bN2Idx': [0] * AIC_CORE_NUM,
            'gS1Idx': [0] * AIC_CORE_NUM,
            'workspaceIdx': [0] * AIC_CORE_NUM,
            's2SplitNum': [0] * AIC_CORE_NUM,
            'taskIdx': [0] * aiv_core_num,
            'mStart': [0] * aiv_core_num,
            'mLen': [0] * aiv_core_num,
        }

    def _calc_grid_info(self, base_info: BaseInfoSim) -> dict:
        grid = {
            'is_empty': True,
            'bn2_idx': base_info.batch_size * base_info.kv_head_num,
            'm_base_num': [],
            's2_base_num': [],
            'm_tail_size': [],
            's2_tail_size': [],
        }
        for b in range(base_info.batch_size):
            s1 = base_info.get_query_seq_size(b)
            s2 = base_info.get_kv_seq_size(b)
            m_num = ceil_div(s1 * base_info.group_size, self.m_base_size)
            m_tail = (s1 * base_info.group_size) % self.m_base_size
            s2_num = ceil_div(s2, self.s2_base_size)
            s2_tail = s2 % self.s2_base_size
            grid['m_base_num'].append(m_num)
            grid['s2_base_num'].append(s2_num)
            grid['m_tail_size'].append(m_tail)
            grid['s2_tail_size'].append(s2_tail)
            if m_num != 0 and s2_num != 0:
                grid['is_empty'] = False
        return grid

    def _calc_s2_range(self, s1g_idx: int, base_info: BaseInfoSim, b_idx: int,
                       s1_size: int, s2_size: int, pre_token_lu: int, next_token_lu: int) -> Tuple[int, int]:
        if s1_size == 0 or s2_size == 0:
            return 0, 0

        # No mask
        if base_info.get_sparse_mode() == SPARSE_BUTT:
            return 0, ceil_div(s2_size, self.s2_base_size)

        s1g_first_token = s1g_idx * self.m_base_size
        s1g_last_token = min(s1g_first_token + self.m_base_size,
                             num_to_index(s1_size * base_info.group_size))

        if base_info.is_s1g:
            s1_first = s1g_first_token // base_info.group_size
            s1_last = s1g_last_token // base_info.group_size
        else:
            if s1g_first_token // s1_size == s1g_last_token // s1_size:
                s1_first = s1g_first_token % s1_size
                s1_last = s1g_last_token % s1_size
            else:
                s1_first = 0
                s1_last = s1_size

        s2_first_token = s1_first - pre_token_lu
        s2_last_token = s1_last + next_token_lu

        if s2_first_token >= s2_size or s2_last_token < 0 or s2_last_token < s2_first_token:
            return 0, 0

        s2_first_token = max(0, min(s2_first_token, num_to_index(s2_size)))
        s2_last_token = max(0, min(s2_last_token, num_to_index(s2_size)))
        s2_start = s2_first_token // self.s2_base_size
        s2_end = s2_last_token // self.s2_base_size + 1  # ToOpenInterval

        return s2_start, s2_end

    def _calc_cost_table(self, s1_normal: int, s2_normal: int, s1_tail: int, s2_tail: int):
        """Returns 2x2 table [m_type][s2_type]"""
        table = [[0, 0], [0, 0]]
        table[0][0] = calc_cost(s1_normal, s2_normal)
        table[1][0] = calc_cost(s1_tail, s2_normal) if s1_tail != 0 else 0
        table[0][1] = calc_cost(s1_normal, s2_tail) if s2_tail != 0 else 0
        table[1][1] = calc_cost(s1_tail, s2_tail) if (s1_tail != 0 and s2_tail != 0) else 0
        return table

    def _calc_s1g_cache(self, s1g_idx: int, base_info: BaseInfoSim, grid_info: dict, b_idx: int):
        s1_size = base_info.get_query_seq_size(b_idx)
        s2_size = base_info.get_kv_seq_size(b_idx)
        pre_token_lu = base_info.get_pre_token_left_up(s1_size, s2_size)
        next_token_lu = base_info.get_next_token_left_up(s1_size, s2_size)

        s2_start, s2_end = self._calc_s2_range(s1g_idx, base_info, b_idx, s1_size, s2_size,
                                                pre_token_lu, next_token_lu)

        if s2_start >= s2_end or grid_info['m_base_num'][b_idx] == 0:
            return {'s2_start': s2_start, 's2_end': s2_end, 'block': 0, 'cost': 0,
                    'last_block_cost': 0, 'normal_block_cost': 0}

        block = s2_end - s2_start
        # tail detection
        has_s2_tail = (grid_info['s2_tail_size'][b_idx] != 0 and s2_end == grid_info['s2_base_num'][b_idx])
        tail_s2_num = 1 if has_s2_tail else 0
        normal_s2_num = block - tail_s2_num

        cost_table = self._calc_cost_table(self.m_base_size, self.s2_base_size,
                                            grid_info['m_tail_size'][b_idx], grid_info['s2_tail_size'][b_idx])

        is_m_tail = (s1g_idx == grid_info['m_base_num'][b_idx] - 1 and grid_info['m_tail_size'][b_idx] != 0)
        m_type = 1 if is_m_tail else 0

        cost = cost_table[m_type][0] * normal_s2_num + cost_table[m_type][1] * tail_s2_num
        last_block_cost = cost_table[m_type][1] if tail_s2_num > 0 else cost_table[m_type][0]
        normal_block_cost = cost_table[m_type][0]

        if block == 1:
            last_block_cost += self.v0_cost
        cost += self.v0_cost

        return {'s2_start': s2_start, 's2_end': s2_end, 'block': block, 'cost': cost,
                'last_block_cost': last_block_cost, 'normal_block_cost': normal_block_cost}

    def _calc_cost_info(self, base_info: BaseInfoSim, grid_info: dict) -> dict:
        cost_info = {
            'bn2_cost': [],
            'bn2_block': [],
            'bn2_last_block_cost': [],
            'block_num': 0,
            'total_cost': 0,
        }
        for b in range(base_info.batch_size):
            bn2_cost = 0
            bn2_block = 0
            bn2_last = 0
            if base_info.get_query_seq_size(b) > 0 and base_info.get_kv_seq_size(b) > 0:
                for s1g in range(grid_info['m_base_num'][b]):
                    cache = self._calc_s1g_cache(s1g, base_info, grid_info, b)
                    bn2_cost += cache['cost']
                    bn2_block += cache['block']
                    if cache['block'] > 0:
                        bn2_last = cache['last_block_cost']
            cost_info['bn2_cost'].append(bn2_cost)
            cost_info['bn2_block'].append(bn2_block)
            cost_info['bn2_last_block_cost'].append(bn2_last)

        # Total across all BN2
        for bn2_idx in range(grid_info['bn2_idx']):
            b_idx = bn2_idx // base_info.kv_head_num
            cost_info['block_num'] += cost_info['bn2_block'][b_idx]
            cost_info['total_cost'] += cost_info['bn2_cost'][b_idx]

        return cost_info

    def _is_within_tolerance(self, limit: int, tolerance: int, value: int) -> bool:
        return limit + tolerance >= value

    def _schedule_fa(self, core_num: int, fd_on: bool, base_info: BaseInfoSim,
                     grid_info: dict, cost_info: dict):
        """Run FA scheduling for a given core count. Returns result dict."""
        result = {
            'max_cost': 0,
            'used_core_num': 0,
            'bN2End': [0] * AIC_CORE_NUM,
            'gS1End': [0] * AIC_CORE_NUM,
            's2End': [0] * AIC_CORE_NUM,
            'firstFdDataWorkspaceIdx': [0] * AIC_CORE_NUM,
            # FD task info
            'fd_task_num': 0,
            'fd_bN2Idx': [0] * AIC_CORE_NUM,
            'fd_mIdx': [0] * AIC_CORE_NUM,
            'fd_workspaceIdx': [0] * AIC_CORE_NUM,
            'fd_s2SplitNum': [0] * AIC_CORE_NUM,
            'fd_mSize': [0] * AIC_CORE_NUM,
        }

        if core_num == 0:
            return result

        # State
        cur_b_idx = 0
        cur_bn2_idx = 0
        cur_s1g_idx = 0
        unassigned_cost = cost_info['total_cost']
        bn2_cost = cost_info['bn2_cost'][0]
        bn2_block = cost_info['bn2_block'][0]
        cur_kv_split_part = 1
        pre_fd_data_num = 0
        is_finished = False

        s1g_cache = self._calc_s1g_cache(cur_s1g_idx, base_info, grid_info, cur_b_idx)
        cur_s2_idx = s1g_cache['s2_start']

        for i in range(core_num):
            if is_finished or unassigned_cost <= 0:
                break

            result['firstFdDataWorkspaceIdx'][i] = pre_fd_data_num + cur_kv_split_part - 1

            core_cost_limit = unassigned_cost // (core_num - i)
            # Adjust limit
            if not fd_on:
                core_cost_limit = max(core_cost_limit, s1g_cache['cost'])
            else:
                cur_cost = s1g_cache['normal_block_cost']
                if cur_s2_idx == s1g_cache['s2_end'] - 1:
                    cur_cost = s1g_cache['last_block_cost']
                if cur_s2_idx == 0 and s1g_cache['block'] != 1:
                    cur_cost += self.v0_cost
                core_cost_limit = max(core_cost_limit, cur_cost)

            core_cost = 0
            core_block = 0

            # AssignByBatch
            while True:
                tol = cost_info['bn2_last_block_cost'][cur_b_idx] // self.fa_tolerance_ratio if self.fa_tolerance_ratio != 0 else 0
                if bn2_cost != 0 and not self._is_within_tolerance(core_cost_limit, tol, core_cost + bn2_cost):
                    break
                core_cost += bn2_cost
                core_block += bn2_block
                cur_bn2_idx += 1

                if cur_bn2_idx == grid_info['bn2_idx']:
                    cur_s1g_idx = 0
                    cur_s2_idx = 0
                    is_finished = True
                    break

                if cur_bn2_idx // base_info.kv_head_num != cur_b_idx:
                    cur_b_idx = cur_bn2_idx // base_info.kv_head_num

                bn2_cost = cost_info['bn2_cost'][cur_b_idx]
                bn2_block = cost_info['bn2_block'][cur_b_idx]
                cur_s1g_idx = 0
                s1g_cache = self._calc_s1g_cache(cur_s1g_idx, base_info, grid_info, cur_b_idx)
                cur_s2_idx = s1g_cache['s2_start']

            if is_finished:
                result['bN2End'][i] = cur_bn2_idx
                result['gS1End'][i] = cur_s1g_idx
                result['s2End'][i] = cur_s2_idx
                result['max_cost'] = max(result['max_cost'], core_cost)
                unassigned_cost -= core_cost
                result['used_core_num'] = i + 1

                # Check FD record
                if fd_on and i > 0 and cur_kv_split_part > 1:
                    if not (cur_bn2_idx == result['bN2End'][i-1] and cur_s1g_idx == result['gS1End'][i-1]):
                        self._record_fd_info(result, i, base_info, grid_info, cur_kv_split_part, pre_fd_data_num)
                        pre_fd_data_num += cur_kv_split_part
                        cur_kv_split_part = 1
                break

            # AssignByRow
            while not is_finished:
                tol = s1g_cache['last_block_cost'] // self.fa_tolerance_ratio if self.fa_tolerance_ratio != 0 else 0
                if not self._is_within_tolerance(core_cost_limit, tol, core_cost + s1g_cache['cost']):
                    break
                core_cost += s1g_cache['cost']
                core_block += s1g_cache['block']
                bn2_cost = max(0, bn2_cost - s1g_cache['cost'])
                bn2_block = max(0, bn2_block - s1g_cache['block'])

                # Next row (skip empty rows)
                while True:
                    cur_s1g_idx += 1
                    s1g_cache = self._calc_s1g_cache(cur_s1g_idx, base_info, grid_info, cur_b_idx)
                    if s1g_cache['block'] != 0:
                        break
                cur_s2_idx = s1g_cache['s2_start']

            # AssignByBlock (only if FD on)
            if fd_on and not is_finished:
                cur_block_cost = s1g_cache['normal_block_cost']
                if cur_s2_idx == s1g_cache['s2_end'] - 1:
                    cur_block_cost = s1g_cache['last_block_cost']
                real_cost = cur_block_cost
                if cur_s2_idx == 0 and s1g_cache['block'] != 1:
                    real_cost += self.v0_cost

                while True:
                    tol = real_cost // self.fa_tolerance_ratio if self.fa_tolerance_ratio != 0 else 0
                    if not self._is_within_tolerance(core_cost_limit, tol, core_cost + real_cost):
                        break
                    core_cost += real_cost
                    core_block += 1
                    cur_s2_idx += 1
                    bn2_cost -= real_cost
                    s1g_cache['cost'] -= real_cost
                    bn2_block -= 1
                    s1g_cache['block'] -= 1

                    real_cost = cur_block_cost
                    if cur_s2_idx == 0 and s1g_cache['block'] != 1:
                        real_cost += self.v0_cost

            result['bN2End'][i] = cur_bn2_idx
            result['gS1End'][i] = cur_s1g_idx
            result['s2End'][i] = cur_s2_idx
            result['max_cost'] = max(result['max_cost'], core_cost)
            unassigned_cost -= core_cost

            # FD record check
            if fd_on and i > 0 and cur_kv_split_part > 1:
                if not (cur_bn2_idx == result['bN2End'][i-1] and cur_s1g_idx == result['gS1End'][i-1]):
                    self._record_fd_info(result, i, base_info, grid_info, cur_kv_split_part, pre_fd_data_num)
                    pre_fd_data_num += cur_kv_split_part
                    cur_kv_split_part = 1

            if cur_s2_idx > s1g_cache['s2_start'] and cur_s2_idx <= s1g_cache['s2_end']:
                cur_kv_split_part += 1

            result['used_core_num'] = i + 1

        return result

    def _record_fd_info(self, result: dict, core_idx: int, base_info: BaseInfoSim,
                        grid_info: dict, kv_split_part: int, pre_fd_data_num: int):
        split_bn2 = result['bN2End'][core_idx - 1]
        split_s1g = result['gS1End'][core_idx - 1]
        split_b_idx = split_bn2 // base_info.kv_head_num
        s1_size = base_info.get_query_seq_size(split_b_idx)

        if split_s1g == num_to_index(grid_info['m_base_num'][split_b_idx]):
            fd_s1g_size = s1_size * base_info.group_size - split_s1g * self.m_base_size
        else:
            fd_s1g_size = self.m_base_size

        idx = result['fd_task_num']
        result['fd_bN2Idx'][idx] = split_bn2
        result['fd_mIdx'][idx] = split_s1g
        result['fd_workspaceIdx'][idx] = pre_fd_data_num
        result['fd_s2SplitNum'][idx] = kv_split_part
        result['fd_mSize'][idx] = fd_s1g_size
        result['fd_task_num'] += 1

    def _schedule_fd(self, aiv_num: int, result: dict):
        """Assign FD tasks to AIV cores."""
        fd_result = {
            'used_vec_num': 0,
            'bN2Idx': result['fd_bN2Idx'],
            'gS1Idx': result['fd_mIdx'],
            'workspaceIdx': result['fd_workspaceIdx'],
            's2SplitNum': result['fd_s2SplitNum'],
            'taskIdx': [0] * aiv_num,
            'mStart': [0] * aiv_num,
            'mLen': [0] * aiv_num,
        }

        fd_task_num = result['fd_task_num']
        if fd_task_num == 0:
            return fd_result

        total_load = sum(result['fd_s2SplitNum'][i] * result['fd_mSize'][i] for i in range(fd_task_num))
        avg_load = ceil_div(total_load, aiv_num)

        cur_core = 0
        for i in range(fd_task_num):
            cur_vec_num = (result['fd_s2SplitNum'][i] * result['fd_mSize'][i]) // avg_load
            cur_vec_num = max(cur_vec_num, 1)
            cur_avg_m = ceil_div(result['fd_mSize'][i], cur_vec_num)
            cur_vec_num = ceil_div(result['fd_mSize'][i], cur_avg_m)

            for vid in range(cur_vec_num):
                if cur_core >= aiv_num:
                    break
                fd_result['taskIdx'][cur_core] = i
                fd_result['mStart'][cur_core] = vid * cur_avg_m
                fd_result['mLen'][cur_core] = cur_avg_m if vid < cur_vec_num - 1 else \
                    (result['fd_mSize'][i] - vid * cur_avg_m)
                cur_core += 1

        fd_result['used_vec_num'] = cur_core
        return fd_result

    def _schedule_section(self, aic_core_num: int, aiv_core_num: int, base_info: BaseInfoSim,
                          grid_info: dict, cost_info: dict):
        block_num = cost_info['block_num']
        max_core = min(aic_core_num, block_num)
        min_core = max(1, round(math.sqrt(block_num + 0.25) + 0.5))
        min_core = min(min_core, max_core)

        best_no_fd = None
        best_fd = None

        for core_num in range(min_core, max_core + 1):
            # Without FD
            res = self._schedule_fa(core_num, False, base_info, grid_info, cost_info)
            if best_no_fd is None or res['max_cost'] < best_no_fd['max_cost']:
                best_no_fd = res

            # With FD
            if self.fd_on:
                res_fd = self._schedule_fa(core_num, True, base_info, grid_info, cost_info)
                if best_fd is None or res_fd['max_cost'] < best_fd['max_cost']:
                    best_fd = res_fd

        # Choose best
        if best_fd is not None:
            fd_result = self._schedule_fd(aiv_core_num, best_fd)
            # Compare: if no_fd + tolerance <= fd, choose no_fd
            if best_no_fd['max_cost'] + self.fd_tolerance <= best_fd['max_cost']:
                fd_result_empty = self._empty_fd_result(aiv_core_num)
                return best_no_fd, fd_result_empty, False
            return best_fd, fd_result, fd_result['used_vec_num'] > 0
        else:
            fd_result_empty = self._empty_fd_result(aiv_core_num)
            return best_no_fd, fd_result_empty, False


# ============ Main API ============
def generate_metadata_cpu(
    cu_seqlens_q: Optional[List[int]] = None,
    cu_seqlens_kv: Optional[List[int]] = None,
    seqused_q: Optional[List[int]] = None,
    seqused_kv: Optional[List[int]] = None,
    num_heads_q: int = 1,
    num_heads_kv: int = 1,
    head_dim: int = 128,
    batch_size: int = 1,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    mask_mode: int = 1,
    win_left: int = -1,
    win_right: int = -1,
    layout_q: str = "BNSD",
    layout_kv: str = "BNSD",
    layout_out: str = "BNSD",
) -> np.ndarray:
    """
    Generate flash_attn metadata buffer (as uint32 numpy array) matching AICPU output.

    Parameters match torch.ops.npu_ops_transformer.npu_flash_attn_metadata().
    Returns numpy uint32 array of shape (HEADER + AIC*FA + AIV*FD,).
    """
    # ---- Determine actual seq lengths ----
    is_cumulative_q = layout_q in ("TND", "NTD")
    is_cumulative_kv = layout_kv in ("TND", "NTD")

    actual_query_seq = None
    actual_kv_seq = None

    if is_cumulative_q and cu_seqlens_q is not None:
        batch_size = len(cu_seqlens_q) - 1
        actual_query_seq = [cu_seqlens_q[i+1] for i in range(batch_size)]  # cumulative
        for i in range(batch_size):
            max_seqlen_q = max(max_seqlen_q, cu_seqlens_q[i+1] - cu_seqlens_q[i])

    if seqused_q is not None:
        batch_size = len(seqused_q)
        actual_query_seq = list(seqused_q)
        if is_cumulative_q:
            # Convert to cumulative
            cum = [0]
            for s in seqused_q:
                cum.append(cum[-1] + s)
            actual_query_seq = cum[1:]  # cumulative end positions
        for s in seqused_q:
            max_seqlen_q = max(max_seqlen_q, s)

    if is_cumulative_kv and cu_seqlens_kv is not None:
        actual_kv_seq = [cu_seqlens_kv[i+1] for i in range(len(cu_seqlens_kv)-1)]
        for i in range(len(cu_seqlens_kv)-1):
            max_seqlen_kv = max(max_seqlen_kv, cu_seqlens_kv[i+1] - cu_seqlens_kv[i])

    if seqused_kv is not None:
        actual_kv_seq = list(seqused_kv)
        if is_cumulative_kv:
            cum = [0]
            for s in seqused_kv:
                cum.append(cum[-1] + s)
            actual_kv_seq = cum[1:]
        for s in seqused_kv:
            max_seqlen_kv = max(max_seqlen_kv, s)

    # ---- Sparse mode conversion ----
    # mask_mode from op: 0=no_mask→BUTT, 1=default, 2=causal(left_up), 3=all_mask, 4=band
    sparse_mode = SPARSE_BUTT
    pre_token = 0
    next_token = 0
    if mask_mode == 0:
        sparse_mode = SPARSE_BUTT  # No mask → no clipping
        pre_token = 0xFFFFFFFF
        next_token = 0xFFFFFFFF
    elif mask_mode == 1:
        sparse_mode = SPARSE_DEFAULT_MASK
        pre_token = win_left if win_left != -1 else 0xFFFFFFFF
        next_token = win_right if win_right != -1 else 0xFFFFFFFF
    elif mask_mode == 2:
        sparse_mode = SPARSE_LEFT_UP_CAUSAL
        pre_token = 0xFFFFFFFF
        next_token = 0xFFFFFFFF
    elif mask_mode == 3:
        sparse_mode = SPARSE_ALL_MASK
        pre_token = win_left if win_left != -1 else 0xFFFFFFFF
        next_token = win_right if win_right != -1 else 0xFFFFFFFF
    elif mask_mode == 4:
        sparse_mode = SPARSE_BAND
        pre_token = win_left if win_left != -1 else 0xFFFFFFFF
        next_token = win_right if win_right != -1 else 0xFFFFFFFF

    # ---- Determine mBaseSize, s2BaseSize ----
    LAYOUT_BNSD = 1
    LAYOUT_BSH = 0
    LAYOUT_TND = 2
    q_layout_id = LAYOUT_BNSD
    if layout_q in ("BSH", "BSND"):
        q_layout_id = LAYOUT_BSH
    elif layout_q == "TND":
        q_layout_id = LAYOUT_TND

    s_outer, s_inner = adjust_sinner_and_souter(head_dim, max_seqlen_q, max_seqlen_kv,
                                                 mask_mode, pre_token, next_token, q_layout_id)
    cv_ratio = AIV_CORE_NUM // AIC_CORE_NUM  # = 2
    m_base_size = s_outer * cv_ratio
    s2_base_size = s_inner

    # ---- Build BaseInfo ----
    base_info = BaseInfoSim(
        batch_size=batch_size,
        query_head_num=num_heads_q,
        kv_head_num=num_heads_kv,
        head_dim=head_dim,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        actual_query_seq=actual_query_seq,
        actual_kv_seq=actual_kv_seq,
        is_cumulative_q=is_cumulative_q,
        is_cumulative_kv=is_cumulative_kv,
        sparse_mode=sparse_mode,
        pre_token=pre_token,
        next_token=next_token,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )

    # ---- Run SectionStreamK ----
    scheduler = SectionStreamKSim(m_base_size=m_base_size, s2_base_size=s2_base_size, fd_on=True)
    fa_result, fd_result, is_fd = scheduler.compute(AIC_CORE_NUM, AIV_CORE_NUM, base_info)

    # ---- Build output buffer ----
    # Buffer size: header(16) + AIC(36*16) + AIV(72*16) = 16 + 576 + 1152 = 1744 uint32
    total_size = HEADER_SIZE + AIC_CORE_NUM * FA_METADATA_SIZE + AIV_CORE_NUM * FD_METADATA_SIZE
    buf = np.zeros(total_size, dtype=np.uint32)

    # Header
    buf[0] = 1  # sectionNum = 1
    buf[1] = 1 if (is_fd and fd_result['used_vec_num'] > 0) else 0  # isFD
    buf[2] = m_base_size
    buf[3] = s2_base_size

    # FA metadata
    fa_base = HEADER_SIZE
    for core in range(fa_result['used_core_num']):
        off = fa_base + core * FA_METADATA_SIZE
        if core > 0:
            buf[off + FA_BN2_START_INDEX] = fa_result['bN2End'][core - 1]
            buf[off + FA_M_START_INDEX] = fa_result['gS1End'][core - 1]
            buf[off + FA_S2_START_INDEX] = fa_result['s2End'][core - 1]
        buf[off + FA_BN2_END_INDEX] = fa_result['bN2End'][core]
        buf[off + FA_M_END_INDEX] = fa_result['gS1End'][core]
        buf[off + FA_S2_END_INDEX] = fa_result['s2End'][core]
        buf[off + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = fa_result['firstFdDataWorkspaceIdx'][core]

    # FD metadata
    fd_base = HEADER_SIZE + AIC_CORE_NUM * FA_METADATA_SIZE
    for core in range(fd_result['used_vec_num']):
        off = fd_base + core * FD_METADATA_SIZE
        task_idx = fd_result['taskIdx'][core]
        buf[off + FD_BN2_IDX_INDEX] = fd_result['bN2Idx'][task_idx]
        buf[off + FD_M_IDX_INDEX] = fd_result['gS1Idx'][task_idx]
        buf[off + FD_WORKSPACE_IDX_INDEX] = fd_result['workspaceIdx'][task_idx]
        buf[off + FD_WORKSPACE_NUM_INDEX] = fd_result['s2SplitNum'][task_idx]
        buf[off + FD_M_START_INDEX] = fd_result['mStart'][core]
        buf[off + FD_M_NUM_INDEX] = fd_result['mLen'][core]

    return buf


def generate_metadata_tensor(
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    num_heads_q=1, num_heads_kv=1,
    head_dim=128, batch_size=1,
    max_seqlen_q=0, max_seqlen_kv=0,
    mask_mode=1, win_left=-1, win_right=-1,
    layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD",
):
    """Generate metadata as a PyTorch int32 tensor (CPU), same format as NPU output."""
    import torch

    # Convert tensor inputs to lists
    def _to_list(v):
        if v is None:
            return None
        if hasattr(v, 'tolist'):
            return v.tolist()
        return list(v) if v is not None else None

    buf = generate_metadata_cpu(
        cu_seqlens_q=_to_list(cu_seqlens_q),
        cu_seqlens_kv=_to_list(cu_seqlens_kv),
        seqused_q=_to_list(seqused_q),
        seqused_kv=_to_list(seqused_kv),
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
    )
    return torch.from_numpy(buf.astype(np.int32))


# ============ Quick self-test ============
if __name__ == "__main__":
    print("=== flash_attn_metadata_sim self-test ===")
    # Simple case: B=1, N1=N2=12, S1=S2=1024, D=128, BNSD, no mask
    meta = generate_metadata_cpu(
        num_heads_q=12, num_heads_kv=12,
        head_dim=128, batch_size=1,
        max_seqlen_q=1024, max_seqlen_kv=1024,
        mask_mode=0,
        layout_q="BNSD", layout_kv="BNSD",
    )
    print(f"Buffer size: {len(meta)} uint32")
    print(f"Header: sectionNum={meta[0]}, isFD={meta[1]}, mBaseSize={meta[2]}, s2BaseSize={meta[3]}")

    # Print active FA cores
    fa_base = HEADER_SIZE
    active_cores = 0
    for core in range(AIC_CORE_NUM):
        off = fa_base + core * FA_METADATA_SIZE
        bn2_end = meta[off + FA_BN2_END_INDEX]
        if bn2_end != 0 or core == 0:
            active_cores += 1
            if core < 5 or bn2_end != 0:
                print(f"  AIC{core:02d}: BN2[{meta[off+0]}-{meta[off+3]}] "
                      f"M[{meta[off+1]}-{meta[off+4]}] S2[{meta[off+2]}-{meta[off+5]}] "
                      f"FD_WS={meta[off+6]}")
    print(f"Active FA cores: {active_cores}")

    # TND case
    print("\n--- TND case: B=2, S1=[512,256], S2=[1024,512] ---")
    meta2 = generate_metadata_cpu(
        cu_seqlens_q=[0, 512, 768],
        cu_seqlens_kv=[0, 1024, 1536],
        num_heads_q=12, num_heads_kv=12,
        head_dim=128, batch_size=2,
        max_seqlen_q=512, max_seqlen_kv=1024,
        mask_mode=1, win_left=2147483647, win_right=2147483647,
        layout_q="TND", layout_kv="TND",
    )
    print(f"Header: sectionNum={meta2[0]}, isFD={meta2[1]}, mBaseSize={meta2[2]}, s2BaseSize={meta2[3]}")
    print("PASS - metadata generated successfully")
