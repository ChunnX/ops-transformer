#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PROJECT_DIR}/../../.." && pwd)
export CATCCOS_SOURCE_PATH=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/catccos"}
export SHMEM_SOURCE_PATH=${SHMEM_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/shmem"}
export SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-"${SHMEM_SOURCE_PATH}/install"}
EXPECTED_SHMEM_COMMIT=0132a6f504c2035d9c1a04100e0d7254fd2968e4

if [[ ! -f "${SHMEM_SOURCE_PATH}/include/shmem.h" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem_utils.so" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/.ops_transformer_source_commit" ]]; then
    echo "error: SHMEM UDMA library was not found; run scripts/prepare_dependencies.sh first" >&2
    exit 1
fi
ACTUAL_SHMEM_COMMIT=$(git -C "${SHMEM_SOURCE_PATH}" rev-parse HEAD 2>/dev/null || true)
BUILT_SHMEM_COMMIT=$(tr -d '[:space:]' < "${SHMEM_HOME_PATH}/shmem/.ops_transformer_source_commit")
if [[ "${ACTUAL_SHMEM_COMMIT}" != "${EXPECTED_SHMEM_COMMIT}" ||
      "${BUILT_SHMEM_COMMIT}" != "${EXPECTED_SHMEM_COMMIT}" ]]; then
    echo "error: SHMEM source or library is not the pinned official release/v1.6.0 snapshot" >&2
    echo "expected: ${EXPECTED_SHMEM_COMMIT}" >&2
    echo "source:   ${ACTUAL_SHMEM_COMMIT:-missing}" >&2
    echo "library:  ${BUILT_SHMEM_COMMIT:-missing}" >&2
    echo "run scripts/prepare_dependencies.sh to rebuild the dependency" >&2
    exit 1
fi
SHMEM_WORKTREE_STATUS=$(git -C "${SHMEM_SOURCE_PATH}" status --porcelain --untracked-files=normal)
if [[ -n "${SHMEM_WORKTREE_STATUS}" ]]; then
    echo "error: SHMEM source has local changes and no longer matches its build provenance" >&2
    printf '%s\n' "${SHMEM_WORKTREE_STATUS}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
# A previous package version can leave an old dist-info directory under the
# setuptools bdist staging tree. Start from a clean local package build so one
# Wheel cannot accidentally contain metadata for two versions.
python3 setup.py clean
python3 -m build --wheel --no-isolation

EXPECTED_PACKAGE_VERSION=$(python3 - <<'PY'
import runpy

print(runpy.run_path(
    "ops_transformer_catccos_udma/_version.py"
)["__version__"])
PY
)
export EXPECTED_PACKAGE_VERSION
python3 - <<'PY'
import os
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

expected_version = os.environ["EXPECTED_PACKAGE_VERSION"]
wheels = list(Path("dist").glob("*.whl"))
if len(wheels) != 1:
    raise SystemExit(
        "error: expected exactly one Wheel under dist, found {}: {}".format(
            len(wheels), ", ".join(str(path) for path in wheels)
        )
    )

wheel = wheels[0]
expected_prefix = "ops_transformer_catccos_udma-{}-".format(expected_version)
if not wheel.name.startswith(expected_prefix):
    raise SystemExit(
        "error: Wheel filename {!r} does not contain package version {}".format(
            wheel.name, expected_version
        )
    )

with ZipFile(wheel) as archive:
    metadata_paths = [
        name for name in archive.namelist()
        if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_paths) != 1:
        raise SystemExit(
            "error: Wheel must contain exactly one dist-info/METADATA, found {}: {}".format(
                len(metadata_paths), ", ".join(metadata_paths)
            )
        )
    metadata_path = metadata_paths[0]
    expected_metadata_path = (
        "ops_transformer_catccos_udma-{}.dist-info/METADATA".format(
            expected_version
        )
    )
    if metadata_path != expected_metadata_path:
        raise SystemExit(
            "error: unexpected Wheel metadata path {!r}; expected {!r}".format(
                metadata_path, expected_metadata_path
            )
        )
    metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))

distribution_name = (metadata.get("Name") or "").lower().replace("-", "_")
if distribution_name != "ops_transformer_catccos_udma":
    raise SystemExit(
        "error: unexpected Wheel distribution name {!r}".format(
            metadata.get("Name")
        )
    )
if metadata.get("Version") != expected_version:
    raise SystemExit(
        "error: Wheel metadata version {!r} does not match expected {}".format(
            metadata.get("Version"), expected_version
        )
    )

print("Verified Wheel metadata: {} {}".format(
    metadata.get("Name"), metadata.get("Version")
))
PY
echo "Wheel written under ${PROJECT_DIR}/dist"
