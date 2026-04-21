"""Verify compilation artifacts — zero torch dependency."""

from __future__ import annotations

import glob
import os

from npukernelbench.models.task import BuildHandle


def verify_build(build: BuildHandle) -> tuple[bool, bool, bool, bool]:
    """Check compilation outputs. Returns (overall, ops_so, custom_opapi, kernel_o)."""
    src = build.paths.src_dir
    build_dir = build.paths.build_dir

    # Check kernel_gen_ops.*.so
    so_pattern = os.path.join(build_dir, "kernel_gen_ops.*.so")
    ops_so_exists = len(glob.glob(so_pattern)) > 0

    # Check libcust_opapi.so
    custom_opapi_path = _find_custom_lib_path(src)
    custom_opapi_exists = custom_opapi_path is not None

    # Check kernel .o files
    if custom_opapi_exists:
        kernel_pattern = os.path.join(custom_opapi_path, "**", f"{build.metadata.op_name}*.o")
        kernel_exists = len(glob.glob(kernel_pattern, recursive=True)) > 0
    else:
        kernel_exists = False

    overall = ops_so_exists and custom_opapi_exists and kernel_exists
    return overall, ops_so_exists, custom_opapi_exists, kernel_exists


def op_artifacts_exist(build: BuildHandle) -> bool:
    """Check if operator compilation artifacts exist (without pybind .so).

    Returns True if libcust_opapi.so and kernel .o files are present,
    meaning the slow operator compilation can be skipped.
    """
    src = build.paths.src_dir
    custom_opapi_path = _find_custom_lib_path(src)
    if not custom_opapi_path:
        return False
    kernel_pattern = os.path.join(custom_opapi_path, "**", f"{build.metadata.op_name}*.o")
    kernel_exists = len(glob.glob(kernel_pattern, recursive=True)) > 0
    return kernel_exists


def _find_custom_lib_path(search_dir: str) -> str | None:
    """Find the prefix path containing op_api/lib/libcust_opapi.so."""
    target = "libcust_opapi.so"
    target_pattern = os.path.join("op_api", "lib", target)

    for root, dirs, files in os.walk(search_dir):
        if target in files:
            full_path = os.path.join(root, target)
            idx = full_path.find(os.path.join("op_api", "lib"))
            if idx > 0:
                return full_path[:idx - 1]
    return None
