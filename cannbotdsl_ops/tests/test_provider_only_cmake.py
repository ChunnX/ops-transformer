# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""CANN 9.2 integration checks for the explicit Provider-only CMake graph."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest


_REPOSITORY = Path(__file__).resolve().parents[2]
_RUN_INTEGRATION = os.environ.get("CANNBOTDSL_RUN_PROVIDER_ONLY_CMAKE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason="set CANNBOTDSL_RUN_PROVIDER_ONLY_CMAKE_TESTS=1 under CANN 9.2",
)


def _run_configure(build_dir: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(_REPOSITORY),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DCANNBOTDSL_PROVIDER_ONLY=ON",
            *arguments,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            (
                "-DENABLE_CANNBOTDSL_PROVIDER_ACLNN=OFF",
                "-DASCEND_COMPUTE_UNIT=ascend950",
                "-DASCEND_OP_NAME=flash_kda",
            ),
            "ENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON",
        ),
        (
            (
                "-DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON",
                "-DASCEND_COMPUTE_UNIT=ascend910b",
                "-DASCEND_OP_NAME=flash_kda",
            ),
            "ASCEND_COMPUTE_UNIT=ascend950",
        ),
        (
            (
                "-DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON",
                "-DASCEND_COMPUTE_UNIT=ascend950",
                "-DASCEND_OP_NAME=",
            ),
            "exactly one ASCEND_OP_NAME=flash_kda",
        ),
        (
            (
                "-DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON",
                "-DASCEND_COMPUTE_UNIT=ascend950",
                "-DASCEND_OP_NAME=ALL",
            ),
            "exactly one ASCEND_OP_NAME=flash_kda",
        ),
        (
            (
                "-DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON",
                "-DASCEND_COMPUTE_UNIT=ascend950",
                "-DASCEND_OP_NAME=chunk_kda",
            ),
            "exactly one ASCEND_OP_NAME=flash_kda",
        ),
        (
            (
                "-DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON",
                "-DASCEND_COMPUTE_UNIT=ascend950",
                "-DASCEND_OP_NAME=flash_kda,chunk_kda",
            ),
            "exactly one ASCEND_OP_NAME=flash_kda",
        ),
    ],
)
def test_provider_only_rejects_invalid_configurations(
    tmp_path: Path, arguments: tuple[str, ...], message: str
):
    result = _run_configure(tmp_path / "build", *arguments)

    assert result.returncode != 0
    assert message in result.stdout + result.stderr


def _required_setting(name: str, default: Path | None = None) -> Path:
    raw = os.environ.get(name)
    result = Path(raw).resolve() if raw else default
    if result is None or not result.exists():
        pytest.fail(f"{name} must name an existing CANN 9.2 path")
    return result


def _valid_configuration_arguments(
    tmp_path: Path,
    *,
    provider_only: bool,
    operator: str,
    include_traditional_option_requests: bool = True,
) -> tuple[str, ...]:
    cann_path = _required_setting(
        "CANNBOTDSL_TASK6_CANN_PATH",
        Path(os.environ["ASCEND_HOME_PATH"]).resolve()
        if os.environ.get("ASCEND_HOME_PATH")
        else None,
    )
    cannbotdsl_source = _required_setting(
        "CANNBOTDSL_TASK6_SOURCE_DIR", _REPOSITORY.parent / "cannbot-dsl"
    )
    runtime_root = _required_setting("CANNBOTDSL_TASK6_RUNTIME_ROOT")
    opbase_source = _required_setting("CANNBOTDSL_TASK6_OPBASE_SOURCE_DIR")
    opbase_library = _required_setting("CANNBOTDSL_TASK6_OPBASE_LIBRARY")
    third_party = _required_setting("CANNBOTDSL_TASK6_THIRD_PARTY_PATH")
    protobuf_headers = _required_setting(
        "CANNBOTDSL_TASK6_PROTOBUF_HEADERS",
        third_party / "lib_cache/protobuf_static/include",
    )
    host_protobuf = _required_setting(
        "CANNBOTDSL_TASK6_HOST_PROTOBUF_LIBRARY",
        cann_path / "x86_64-linux/lib64/libhost_ascend_protobuf.a",
    )
    acceptance_run = tmp_path / "flash-kda-provider-only.run"
    arguments = [
        f"-DCANNBOTDSL_PROVIDER_ONLY={'ON' if provider_only else 'OFF'}",
        "-DBUILD_OPEN_PROJECT=ON",
        "-DENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON",
        "-DASCEND_COMPUTE_UNIT=ascend950",
        f"-DASCEND_OP_NAME={operator}",
        "-DCANNBOTDSL_BUILD_FLASH_KDA_ACLNN_SMOKE=ON",
        f"-DCANNBOTDSL_SOURCE_DIR={cannbotdsl_source}",
        f"-DCANNBOTDSL_RUNTIME_ROOT={runtime_root}",
        f"-DCANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR={opbase_source}",
        f"-DCANNBOTDSL_PROVIDER_OPBASE_LIBRARY={opbase_library}",
        f"-DCANNBOTDSL_FLASH_KDA_ACCEPTANCE_RUN={acceptance_run}",
        f"-DCUSTOM_ASCEND_CANN_PACKAGE_PATH={cann_path}",
        f"-DCANN_3RD_LIB_PATH={third_party}",
        f"-DHOST_PROTOBUF_STATIC_LIB={host_protobuf}",
        f"-DHOST_PROTOBUF_STATIC_INCLUDE={protobuf_headers}",
        f"-DCMAKE_INSTALL_PREFIX={tmp_path / 'install'}",
    ]
    if include_traditional_option_requests:
        arguments.extend(
            (
                "-DENABLE_OPS_HOST=ON",
                "-DENABLE_OPS_KERNEL=ON",
                "-DENABLE_TILING_SINK=ON",
            )
        )
    return tuple(arguments)


def _cache_value(build_dir: Path, name: str) -> str:
    prefix = f"{name}:"
    for line in (build_dir / "CMakeCache.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1]
    raise AssertionError(f"{name} is missing from CMakeCache.txt")


def _acceptance_commands(build_dir: Path) -> str:
    return subprocess.run(
        [
            "ninja",
            "-C",
            str(build_dir),
            "-t",
            "commands",
            "cannbotdsl_flash_kda_acceptance",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_provider_off_mixed_configure_preserves_options_and_acceptance_mode(
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    result = _run_configure(
        build_dir,
        *_valid_configuration_arguments(
            tmp_path, provider_only=False, operator="chunk_kda_fwd"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "CANNBOTDSL_PROVIDER_ONLY requires" not in result.stdout + result.stderr
    assert _cache_value(build_dir, "ENABLE_OPS_HOST") == "ON"
    assert _cache_value(build_dir, "ENABLE_OPS_KERNEL") == "ON"
    assert _cache_value(build_dir, "ENABLE_TILING_SINK") == "ON"
    acceptance_commands = _acceptance_commands(build_dir)
    assert "--package-mode mixed" in acceptance_commands
    assert "--expected-traditional-kernel-count 8" in acceptance_commands


def test_provider_only_cache_does_not_force_options_off_after_mode_is_disabled(
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    provider_only_result = _run_configure(
        build_dir,
        *_valid_configuration_arguments(
            tmp_path, provider_only=True, operator="flash_kda"
        ),
    )
    assert provider_only_result.returncode == 0, (
        provider_only_result.stdout + provider_only_result.stderr
    )
    assert _cache_value(build_dir, "ENABLE_OPS_HOST") == "ON"
    assert _cache_value(build_dir, "ENABLE_OPS_KERNEL") == "ON"
    assert _cache_value(build_dir, "ENABLE_TILING_SINK") == "ON"

    provider_off_result = _run_configure(
        build_dir,
        *_valid_configuration_arguments(
            tmp_path,
            provider_only=False,
            operator="chunk_kda_fwd",
            include_traditional_option_requests=False,
        ),
    )

    assert provider_off_result.returncode == 0, (
        provider_off_result.stdout + provider_off_result.stderr
    )
    assert _cache_value(build_dir, "ENABLE_OPS_HOST") == "ON"
    assert _cache_value(build_dir, "ENABLE_OPS_KERNEL") == "ON"
    assert _cache_value(build_dir, "ENABLE_TILING_SINK") == "ON"
    assert "--package-mode mixed" in _acceptance_commands(build_dir)


def test_provider_only_generates_only_provider_and_infrastructure_targets(
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    result = _run_configure(
        build_dir,
        *_valid_configuration_arguments(
            tmp_path, provider_only=True, operator="flash_kda"
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    target_help = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for target in (
        "cannbotdsl_provider_aclnn_registry",
        "ophost_transformer_opapi_obj",
        "cust_opapi",
        "ops_aclnn",
        "cust_proto",
        "cust_opmaster",
        "optiling_compat",
        "cannbotdsl_flash_kda_aclnn_smoke",
        "cannbotdsl_flash_kda_acceptance",
        "package",
    ):
        assert re.search(rf"\b{re.escape(target)}\b", target_help), target_help

    target_graph = subprocess.run(
        [
            "ninja",
            "-C",
            str(build_dir),
            "-t",
            "graph",
            "cust_opapi",
            "cust_proto",
            "cust_opmaster",
            "cannbotdsl_flash_kda_aclnn_smoke",
            "cannbotdsl_flash_kda_acceptance",
            "package",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "provider_only_stub.cpp" in target_graph
    acceptance_commands = _acceptance_commands(build_dir)
    assert "--package-mode provider-only" in acceptance_commands
    assert "--expected-traditional-kernel-count 0" in acceptance_commands

    prohibited = (
        "chunk_kda",
        "ChunkKda",
        "opbuild_gen_",
        "generate_transformer_adapt_py",
        "generate_compile_cmd",
        "generate_ops_info",
        "ops_transformer_kernel",
        "all_gather_matmul",
        "flash_attention",
        "sparse_flash",
    )
    for name in prohibited:
        assert name not in target_help
        assert name not in target_graph
