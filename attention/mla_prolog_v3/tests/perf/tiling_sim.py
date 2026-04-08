#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compat shim for the original tiling_sim.py.

The implementation has been split into the perf.tiling package. This module
re-exports the public API so existing imports continue to work unchanged.
New code should prefer the package paths:

    from perf.tiling.config import compute_tiling, TilingConfig
    from perf.tiling.split_kn_search import search_split_kn
"""

from perf.tiling.config import (
    TilingConfig, compute_tiling, search_best_tiling,
    ceil_div, calc_single_core_n,
    BLOCK_SIZE, GROUP_COMPUTE_CUBE_NUM_PER_GROUP, HIGH_THROUGHPUT_D_SIZE,
    GROUP_COMPUTE_T_SIZE, GROUP_COMPUTE_N_SIZE,
    GROUP_COMPUTE_MIN_AIC_NUM, GROUP_COMPUTE_MIN_AIV_NUM,
    NUM_BYTES_BF16, NUM_BYTES_INT32,
)
from perf.tiling.block_search import (
    search_matmul_block_sizes, search_all_matmul_blocks,
    SEARCH_BASE_M, SEARCH_BASE_N, SEARCH_BASE_K,
)
from perf.tiling.mm2_split_k import MM2SplitKResult, search_mm2_split_k
from perf.tiling.split_kn_search import (
    SplitKNResult, search_split_kn, _gen_kn_candidates,
)
from perf.tiling.split_m_search import (
    SplitMResult, search_split_m, _get_split_m_block_specs,
)
from perf.tiling.prefill_search import PrefillResult, search_prefill_optimal
