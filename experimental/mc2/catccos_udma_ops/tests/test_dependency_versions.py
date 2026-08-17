# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import re
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
SHMEM_COMMIT = "aee374a396b389626f68fab74ec8c555e18978a7"


def test_package_version_has_one_source_of_truth():
    version_file = (
        PROJECT_ROOT / "ops_transformer_catccos_udma" / "_version.py"
    )
    package_version = runpy.run_path(str(version_file))["__version__"]
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", package_version)

    setup_source = (PROJECT_ROOT / "setup.py").read_text()
    package_source = (
        PROJECT_ROOT / "ops_transformer_catccos_udma" / "__init__.py"
    ).read_text()
    cmake_source = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    assert 'PACKAGE_NAME, "_version.py"' in setup_source
    assert "from ._version import __version__" in package_source
    assert "ops_transformer_catccos_udma/_version.py" in cmake_source
    assert "VERSION \"${OPS_TRANSFORMER_CATCCOS_UDMA_VERSION}\"" in cmake_source
    assert "VERSION 0." not in cmake_source


def test_wheel_build_cleans_and_verifies_distribution_metadata():
    source = (PROJECT_ROOT / "scripts" / "build_wheel.sh").read_text()
    clean = source.index("python3 setup.py clean")
    build = source.index("python3 -m build --wheel --no-isolation")
    verify = source.index("Verified Wheel metadata:")
    assert clean < build < verify
    assert 'name.endswith(".dist-info/METADATA")' in source
    assert "if len(metadata_paths) != 1" in source
    assert 'metadata.get("Version") != expected_version' in source


def test_prepare_dependencies_pins_official_shmem_v1_6_0():
    source = (PROJECT_ROOT / "scripts" / "prepare_dependencies.sh").read_text()
    assert "SHMEM_REPOSITORY=https://gitcode.com/cann/shmem.git" in source
    assert f"SHMEM_COMMIT={SHMEM_COMMIT}" in source
    assert 'SHMEM_DIR=${SHMEM_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/shmem"}' in source


def test_cmake_rejects_a_mismatched_shmem_checkout():
    source = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    assert f'set(SHMEM_TAG_ID "{SHMEM_COMMIT}"' in source
    assert "SHMEM revision mismatch" in source
    assert ".ops_transformer_source_commit" in source
    assert "${SHMEM_SOURCE_PATH}/include" in source
    assert "${CATCCOS_SHMEM_SOURCE_DIR}/include" not in source


def test_wheel_records_bundled_shmem_revision():
    source = (
        PROJECT_ROOT
        / "ops_transformer_catccos_udma"
        / "_build_versions.py"
    ).read_text()
    assert 'SHMEM_VERSION = "1.6.0"' in source
    assert f'SHMEM_COMMIT = "{SHMEM_COMMIT}"' in source
