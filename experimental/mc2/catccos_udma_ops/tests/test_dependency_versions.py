# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
SHMEM_COMMIT = "aee374a396b389626f68fab74ec8c555e18978a7"


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
