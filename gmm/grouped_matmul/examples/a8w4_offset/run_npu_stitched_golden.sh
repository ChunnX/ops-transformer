#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/build_install_run_custom.sh"
DUMP_DIR="${A8W4_STITCHED_DUMP_DIR:-${REPO_ROOT}/build_out/a8w4_npu_stitched_case}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export A8W4_DEQUANT_MODE="${A8W4_DEQUANT_MODE:-pergroup}"
export A8W4_WEIGHT_FORMAT="${A8W4_WEIGHT_FORMAT:-ND}"
export A8W4_OUTPUT_DTYPE="${A8W4_OUTPUT_DTYPE:-fp16}"
export A8W4_DUMP_DIR="${DUMP_DIR}"
case "${A8W4_DEQUANT_MODE}" in
  perchannel|pergroup) ;;
  *) echo "A8W4_DEQUANT_MODE must be perchannel or pergroup" >&2; exit 2 ;;
esac
case "${A8W4_WEIGHT_FORMAT}" in
  ND) expected_weight_nz=0 ;;
  NZ) expected_weight_nz=1 ;;
  *) echo "A8W4_WEIGHT_FORMAT must be ND or NZ" >&2; exit 2 ;;
esac
if [[ -n "${A8W4_WEIGHT_NZ+x}" && "${A8W4_WEIGHT_NZ}" != "${expected_weight_nz}" ]]; then
  echo "A8W4_WEIGHT_NZ=${A8W4_WEIGHT_NZ} conflicts with A8W4_WEIGHT_FORMAT=${A8W4_WEIGHT_FORMAT}" >&2
  exit 2
fi
export A8W4_WEIGHT_NZ="${expected_weight_nz}"
if [[ "${A8W4_API:-V5}" == "WeightNz" && "${A8W4_WEIGHT_FORMAT}" != "NZ" ]]; then
  echo "A8W4_API=WeightNz requires A8W4_WEIGHT_FORMAT=NZ" >&2
  exit 2
fi

mkdir -p "${DUMP_DIR}"
if [[ "${A8W4_STITCHED_REBUILD:-1}" == "1" ]]; then
  bash "${RUNNER}" test-build
fi

set +e
bash "${RUNNER}" run-only
formal_rc=$?
set -e

echo "formal_test_exit=${formal_rc}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/npu_stitched_golden.py" "${DUMP_DIR}" \
  --device "${A8W4_NPU_DEVICE:-npu:0}" \
  --save-npu "${DUMP_DIR}/npu_golden.f32.bin"