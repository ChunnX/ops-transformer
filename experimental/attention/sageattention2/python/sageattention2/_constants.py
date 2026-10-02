"""SageAttention2 Unified Constants and Configuration Specifications."""
from typing import Any

import torch

# 布局枚举
LAYOUT_HND = "HND"
LAYOUT_NHD = "NHD"
VALID_TENSOR_LAYOUTS = (LAYOUT_HND, LAYOUT_NHD)

# 量化粒度枚举
GRAN_PER_WARP = "per_warp"
GRAN_PER_THREAD = "per_thread"
VALID_QK_QUANT_GRANS = (GRAN_PER_WARP, GRAN_PER_THREAD)
DEFAULT_QK_QUANT_GRAN = GRAN_PER_THREAD

# PV 累加档位枚举
ACCUM_FP32 = "fp32"
ACCUM_FP32_FP32 = "fp32+fp32"
ACCUM_FP32_FP16 = "fp32+fp16"
VALID_PV_ACCUM_DTYPES = (ACCUM_FP32, ACCUM_FP32_FP32, ACCUM_FP32_FP16)
DEFAULT_PV_ACCUM_DTYPE = ACCUM_FP32_FP16

# 环境变量名称
ENV_PV_ACCUM_DTYPE = "SAGEATTN_PV_ACCUM_DTYPE"
ENV_QK_QUANT_GRAN = "SAGEATTN_QK_QUANT_GRAN"

# Head Dim 对齐分桶档位（0 < D <= 64 补至 64；64 < D <= 128 补至 128）
PAD_BUCKETS = (64, 128)
MAX_HEAD_DIM = PAD_BUCKETS[-1]

# 浮点上界常数
FP32_MAX = torch.finfo(torch.float32).max

# 微架构量化常数
PER_THREAD_SEG = 16
MX_GROUP_TOKENS = 32
E4M3_MAX = 448.0
E8M0_MIN_EXPONENT = -127.0
QUERY_TILE_ROWS = 64
KV_TILE_ROWS = 128

# 概率域 scale_max 分档：κ = exp 域上界，p = exp(S - m') <= κ
KAPPA: dict[str, float] = {
    ACCUM_FP32: 448.0,
    ACCUM_FP32_FP32: 448.0,
    ACCUM_FP32_FP16: 2.25,
}

# 官方任务标准 Benchmark 规格矩阵（P-01 ~ P-07）
BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "P-01": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 4096,
        "sk": 4096,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S1",
        "desc": "[1,32,4096,128] FP16 non-causal (Standard 4K Context)",
    },
    "P-02": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 8192,
        "sk": 8192,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S1",
        "desc": "[1,32,8192,128] FP16 non-causal (Standard 8K Context)",
    },
    "P-03": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 8192,
        "sk": 8192,
        "d": 128,
        "is_causal": True,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S2",
        "desc": "[1,32,8192,128] FP16 causal (8K Causal Masking)",
    },
    "P-04": {
        "batch": 2,
        "hq": 32,
        "hk": 32,
        "sq": 8192,
        "sk": 8192,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S1",
        "desc": "[2,32,8192,128] FP16 non-causal (Batch=2 8K Context)",
    },
    "P-05": {
        "batch": 1,
        "hq": 32,
        "hk": 8,
        "sq": 4096,
        "sk": 4096,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S3",
        "desc": "[1,32:8,4096,128] FP16 GQA (Grouped Query 32:8, S=4096)",
    },
    "P-06": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 4096,
        "sk": 4096,
        "d": 128,
        "is_causal": False,
        "dtype": torch.bfloat16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S4",
        "desc": "[1,32,4096,128] BF16 non-causal (BFloat16 Mixed Precision)",
    },
    "P-07": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 16384,
        "sk": 16384,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S5",
        "desc": "[1,32,16384,128] FP16 long-sequence (Ultra-long 16K Context)",
    },
}

STANDARD_CASE_IDS: list[str] = ["P-01", "P-02", "P-03", "P-04", "P-05", "P-06", "P-07"]
SHORT_CASE_IDS: list[str] = ["P-00-S128", "P-00-S512", "P-00-Decode"]
EXTENDED_CASE_IDS: list[str] = [
    "P-08", "P-09", "P-10", "X-01", "X-02", "X-03", "X-04", "X-05", "X-06", "X-07"
]

EXTENDED_BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "P-00-S128": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 128,
        "sk": 128,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S0",
        "desc": "[1,32,128,128] FP16 short-sequence (S=128 Micro-Prompt)",
    },
    "P-00-S512": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 512,
        "sk": 512,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S0",
        "desc": "[1,32,512,128] FP16 short-sequence (S=512 Short Prompt)",
    },
    "P-00-Decode": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 1,
        "sk": 4096,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S0",
        "desc": "[1,32,1,128] x [1,32,4096,128] FP16 Decode Step (Sq=1, Sk=4096)",
    },
    "P-08": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 32768,
        "sk": 32768,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S5",
        "desc": "[1,32,32768,128] FP16 ultra-long (32K Context)",
    },
    "P-09": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 65536,
        "sk": 65536,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S5",
        "desc": "[1,32,65536,128] FP16 ultra-long (64K Context)",
    },
    "P-10": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 131072,
        "sk": 131072,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "S5",
        "desc": "[1,32,131072,128] FP16 extreme-long (128K Context)",
    },
    "X-01": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 2048,
        "sk": 2048,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "SX",
        "desc": "[1,32,2048,128] FP16 medium (2K Context)",
    },
    "X-02": {
        "batch": 1,
        "hq": 32,
        "hk": 8,
        "sq": 2048,
        "sk": 2048,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "SX",
        "desc": "[1,32:8,2048,128] FP16 GQA (32:8, 2K Context)",
    },
    "X-03": {
        "batch": 1,
        "hq": 32,
        "hk": 8,
        "sq": 8192,
        "sk": 8192,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "SX",
        "desc": "[1,32:8,8192,128] FP16 GQA (32:8, 8K Context)",
    },
    "X-04": {
        "batch": 1,
        "hq": 32,
        "hk": 8,
        "sq": 16384,
        "sk": 16384,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "SX",
        "desc": "[1,32:8,16384,128] FP16 GQA (32:8, 16K Context)",
    },
    "X-05": {
        "batch": 2,
        "hq": 32,
        "hk": 32,
        "sq": 4096,
        "sk": 4096,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "SX",
        "desc": "[2,32,4096,128] FP16 non-causal (Batch=2 4K Context)",
    },
    "X-06": {
        "batch": 4,
        "hq": 32,
        "hk": 32,
        "sq": 2048,
        "sk": 2048,
        "d": 128,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "SX",
        "desc": "[4,32,2048,128] FP16 non-causal (Batch=4 2K Context)",
    },
    "X-07": {
        "batch": 1,
        "hq": 32,
        "hk": 32,
        "sq": 4096,
        "sk": 4096,
        "d": 64,
        "is_causal": False,
        "dtype": torch.float16,
        "pv_accum_dtype": ACCUM_FP32_FP16,
        "category": "SX",
        "desc": "[1,32,4096,64] FP16 native head_dim=64",
    },
}

ALL_BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    **BENCHMARK_SPECS,
    **EXTENDED_BENCHMARK_SPECS,
}

