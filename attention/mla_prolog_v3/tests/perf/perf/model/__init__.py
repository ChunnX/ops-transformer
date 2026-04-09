"""perf.model - hardware specs, matmul/vector timing models, orchestration.

Public API re-exports for easier imports.
"""

from .hardware import (
    AscendHWSpec, CacheSpec,
    ASCEND_910B, ASCEND_950, CHIP_REGISTRY,
)
from .quant import (
    QuantMode, resolve_quant_mode, dtype_size,
    is_full_quant, is_int8_quant, is_mxfp8,
)
from .timing_types import MatMulTiming, VectorOpTiming
from .operator_params import OperatorParams
from .block_spec import MatmulBlockSpec, check_cache_fit, derive_stepK
from .matmul_legacy import estimate_matmul
from .matmul_detailed import estimate_matmul_detailed
from .matmul_split_kn import (
    SplitKNSpec, estimate_matmul_split_kn, estimate_mm2_split_k_cross_core,
)
from .matmul_split_m import SplitMSpec, estimate_matmul_split_m
from .matmul_k_outer import estimate_matmul_k_outer
from .vector_ops import estimate_vector_op
from .dequant_opt import (
    DequantOptStageTiming, DequantOptBudget,
    estimate_dequant_opt_pipeline, is_dequant_opt_applicable, compute_budget,
)
from .split_kn_fused import (
    SplitKnFusedTiming, SplitKnFusedBudget,
    estimate_split_kn_fused, STATIC_AIV_UB_BYTES,
)
from .orchestration import (
    get_default_block_specs,
    estimate_all_stages,
    estimate_all_stages_split_m,
    estimate_all_stages_k_outer,
)
