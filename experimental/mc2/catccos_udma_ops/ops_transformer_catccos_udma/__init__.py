# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Ascend950 CatCCOS UDMA fused operators for PyTorch."""

import sys as _sys
import threading as _threading
_load_lock = _sys.__dict__.setdefault("_ops_transformer_catccos_load_lock", _threading.RLock())
with _load_lock:
    if (getattr(_sys, "_ops_transformer_catccos_backend", None) == "mte" or
            "ops_transformer_catccos_mte._C" in _sys.modules):
        raise RuntimeError(
            "UDMA and MTE extensions cannot be loaded in the same process: SHMEM global state "
            "and library symbols are shared. Use separate torchrun workers."
        )
    from . import _C as _C  # noqa: F401 - loading the DSO registers torch.ops schemas
    _sys._ops_transformer_catccos_backend = "udma"
from ._build_versions import CATCCOS_COMMIT, SHMEM_COMMIT, SHMEM_REF, SHMEM_VERSION
from ._version import __version__
from .ops import all_to_allv_grouped_mat_mul, grouped_mat_mul_all_to_allv
from .routing import (
    build_global_route_matrix,
    build_rank_route,
    build_rank_route_tables,
    build_rank_route_tensors,
)
from .runtime import (
    UdmaConfig,
    destroy_udma_group,
    empty_symmetric,
    get_udma_config,
    init_udma_group,
    poison_udma_workspace_for_testing,
    udma_group,
)

__all__ = [
    "UdmaConfig",
    "CATCCOS_COMMIT",
    "SHMEM_COMMIT",
    "SHMEM_REF",
    "SHMEM_VERSION",
    "all_to_allv_grouped_mat_mul",
    "build_global_route_matrix",
    "build_rank_route",
    "build_rank_route_tables",
    "build_rank_route_tensors",
    "destroy_udma_group",
    "empty_symmetric",
    "get_udma_config",
    "grouped_mat_mul_all_to_allv",
    "init_udma_group",
    "poison_udma_workspace_for_testing",
    "udma_group",
]
