# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Ascend950 CatCCOS UDMA fused operators for PyTorch."""

from . import _C as _C  # noqa: F401 - loading the DSO registers torch.ops schemas
from .ops import all_to_allv_grouped_mat_mul, grouped_mat_mul_all_to_allv
from .routing import (
    build_global_route_matrix,
    build_rank_route_tables,
    build_rank_route_tensors,
)
from .runtime import (
    UdmaConfig,
    destroy_udma_group,
    empty_symmetric,
    get_udma_config,
    init_udma_group,
    udma_group,
)

__all__ = [
    "UdmaConfig",
    "all_to_allv_grouped_mat_mul",
    "build_global_route_matrix",
    "build_rank_route_tables",
    "build_rank_route_tensors",
    "destroy_udma_group",
    "empty_symmetric",
    "get_udma_config",
    "grouped_mat_mul_all_to_allv",
    "init_udma_group",
    "udma_group",
]

__version__ = "0.2.0"
