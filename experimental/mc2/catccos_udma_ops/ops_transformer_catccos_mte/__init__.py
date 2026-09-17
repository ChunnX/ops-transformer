# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Ascend950 MTE operators; CPU planning imports do not load the NPU runtime."""

from importlib import import_module
from ._version import __version__
from ._build_versions import CATCCOS_COMMIT, SHMEM_COMMIT, SHMEM_VERSION

_EXPORTS = {
    "all_to_allv_grouped_mat_mul": "ops",
    "grouped_mat_mul_all_to_allv": "ops",
    "build_global_route_matrix": "routing",
    "build_rank_route": "routing",
    "build_rank_route_tables": "routing",
    "build_rank_route_tensors": "routing",
    "MteConfig": "runtime",
    "init_mte_group": "runtime",
    "destroy_mte_group": "runtime",
    "get_mte_config": "runtime",
    "empty_symmetric": "runtime",
    "mte_group": "runtime",
}
__all__ = list(_EXPORTS) + ["__version__", "CATCCOS_COMMIT", "SHMEM_COMMIT", "SHMEM_VERSION"]


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module("." + module_name, __name__), name)
    globals()[name] = value
    return value
