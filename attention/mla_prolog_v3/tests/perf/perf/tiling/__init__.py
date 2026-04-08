"""perf.tiling - tiling computation and search algorithms.

Public API re-exports.
"""

from .config import (
    TilingConfig, compute_tiling, search_best_tiling,
    ceil_div, calc_single_core_n,
    BLOCK_SIZE, GROUP_COMPUTE_CUBE_NUM_PER_GROUP, HIGH_THROUGHPUT_D_SIZE,
    GROUP_COMPUTE_T_SIZE, GROUP_COMPUTE_N_SIZE,
    GROUP_COMPUTE_MIN_AIC_NUM, GROUP_COMPUTE_MIN_AIV_NUM,
    NUM_BYTES_BF16, NUM_BYTES_INT32,
)
from .block_search import (
    search_matmul_block_sizes, search_all_matmul_blocks,
    SEARCH_BASE_M, SEARCH_BASE_N, SEARCH_BASE_K,
)
from .mm2_split_k import MM2SplitKResult, search_mm2_split_k
from .split_kn_search import SplitKNResult, search_split_kn, _gen_kn_candidates
from .split_m_search import SplitMResult, search_split_m, _get_split_m_block_specs
from .prefill_search import PrefillResult, search_prefill_optimal
