#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

BACKEND=${CATCCOS_BACKEND:-udma}
WITH_TUNING=${CATCCOS_MTE_WITH_TUNING:-0}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || { echo "error: --backend requires udma or mte" >&2; exit 2; }
            BACKEND=$2; shift 2 ;;
        --with-tuning) WITH_TUNING=1; shift ;;
        --help|-h) echo "Usage: $0 [--backend udma|mte] [--with-tuning]"; exit 0 ;;
        *) echo "error: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ "$BACKEND" == "udma" || "$BACKEND" == "mte" ]] || { echo "error: invalid backend $BACKEND" >&2; exit 2; }
case "${WITH_TUNING,,}" in
    1|on|true|yes) WITH_TUNING=1 ;;
    ""|0|off|false|no) WITH_TUNING=0 ;;
    *) echo "error: CATCCOS_MTE_WITH_TUNING must be boolean" >&2; exit 2 ;;
esac
if [[ "$WITH_TUNING" != "0" && "$BACKEND" != "mte" ]]; then
    echo "error: --with-tuning requires --backend mte" >&2; exit 2
fi
export CATCCOS_BACKEND=$BACKEND
if [[ "$BACKEND" == "mte" ]]; then
    export CATCCOS_DIST_DIR="dist/mte"
else
    export CATCCOS_DIST_DIR="dist"
fi
export CATCCOS_MTE_WITH_TUNING=$WITH_TUNING

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${PROJECT_DIR}/../../.." && pwd)
export CATCCOS_SOURCE_PATH=${CATCCOS_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/catccos"}
if [[ "$BACKEND" == "mte" ]]; then
    DEFAULT_SHMEM_PATH="${REPO_ROOT}/3rdparty/shmem-mte"
else
    DEFAULT_SHMEM_PATH="${REPO_ROOT}/3rdparty/shmem"
fi
export SHMEM_SOURCE_PATH=${SHMEM_SOURCE_PATH:-"${DEFAULT_SHMEM_PATH}"}
export SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-"${SHMEM_SOURCE_PATH}/install"}
EXPECTED_SHMEM_COMMIT=7fa19662da0093587d9a332e740b91d6c929cdcf
if [[ "$BACKEND" == "mte" ]]; then
    EXPECTED_SHMEM_COMMIT=aee374a396b389626f68fab74ec8c555e18978a7
fi

if [[ ! -f "${SHMEM_SOURCE_PATH}/include/shmem.h" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/lib/libshmem_utils.so" ||
      ! -f "${SHMEM_HOME_PATH}/shmem/.ops_transformer_source_commit" ]]; then
    echo "error: SHMEM ${BACKEND} library was not found; run scripts/prepare_dependencies.sh first" >&2
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
python3 -m build --wheel --no-isolation --outdir "${CATCCOS_DIST_DIR}"

EXPECTED_PACKAGE_VERSION=$(python3 - <<'PY'
import runpy
import os

print(runpy.run_path(
    "ops_transformer_catccos_{}/_version.py".format(os.environ["CATCCOS_BACKEND"])
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
backend = os.environ["CATCCOS_BACKEND"]
package = "ops_transformer_catccos_" + backend
wheels = list(Path(os.environ["CATCCOS_DIST_DIR"]).glob("*.whl"))
if len(wheels) != 1:
    raise SystemExit(
        "error: expected exactly one Wheel under dist, found {}: {}".format(
            len(wheels), ", ".join(str(path) for path in wheels)
        )
    )

wheel = wheels[0]
expected_prefix = "{}-{}-".format(package, expected_version)
if not wheel.name.startswith(expected_prefix):
    raise SystemExit(
        "error: Wheel filename {!r} does not contain package version {}".format(
            wheel.name, expected_version
        )
    )

with ZipFile(wheel) as archive:
    opposite = "ops_transformer_catccos_" + ("udma" if backend == "mte" else "mte") + "/"
    if any(name.startswith(opposite) or (backend == "mte" and name.startswith("validation/"))
           for name in archive.namelist()):
        raise SystemExit("error: Wheel contains another backend or an unnamespaced validation package")
    if package + "/_C.abi3.so" not in archive.namelist():
        raise SystemExit("error: selected-backend extension is missing from Wheel")
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
        "{}-{}.dist-info/METADATA".format(package, expected_version)
    )
    if metadata_path != expected_metadata_path:
        raise SystemExit(
            "error: unexpected Wheel metadata path {!r}; expected {!r}".format(
                metadata_path, expected_metadata_path
            )
        )
    metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))

distribution_name = (metadata.get("Name") or "").lower().replace("-", "_")
if distribution_name != package:
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
echo "Wheel written under ${PROJECT_DIR}/${CATCCOS_DIST_DIR}"
