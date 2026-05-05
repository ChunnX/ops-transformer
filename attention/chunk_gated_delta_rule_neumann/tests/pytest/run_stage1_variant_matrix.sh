#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPS_ROOT="$(cd "$OP_DIR/../.." && pwd)"
INSTALL_PATH="/home/developer/Ascend/cann-9.0.0/opp"
RUN_PKG="$OPS_ROOT/build_out/cann-ops-transformer-custom_linux-aarch64.run"

VARIANTS=(
  C16_FP32
  C16_FP16
  C64_1616_FP32
  C64_1616_FP16
  C64_3232_FP32
  C64_3232_FP16
)

selected_chunk() {
  case "$1" in
    C16_*) echo 16 ;;
    *) echo 64 ;;
  esac
}

select_variant() {
  local suite="$1"
  local variant="$2"
  python3 "$SCRIPT_DIR/select_stage1_variant.py" --suite "$suite" --variant "$variant"
}

build_and_install() {
  (cd "$OP_DIR" && bash ../../build.sh --pkg --soc=ascend910_93 \
      --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16)
  "$RUN_PKG" --quiet --install-path="$INSTALL_PATH"
}

run_precision() {
  local variant="$1"
  local chunk
  chunk="$(selected_chunk "$variant")"
  set +u
  source "$INSTALL_PATH/vendors/custom_transformer/bin/set_env.bash"
  set -u
  (cd "$OP_DIR" && CGDR_GOLDEN_CHUNK_SIZE="$chunk" TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext \
      bash tests/pytest/test_run.sh compare)
}

run_perf_selected_s() {
  local suite="$1"
  local variant="$2"
  local out_dir="$OP_DIR/tests/pytest/prof_out/stage1_${suite}_${variant}_s_selected"
  set +u
  source "$INSTALL_PATH/vendors/custom_transformer/bin/set_env.bash"
  set -u
  (cd "$OP_DIR/tests/pytest" && TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext \
      python3 run_chunk_gdr_performance_msprof.py \
      --output "$out_dir" \
      --case prefill_b8_s16_d64_h16_hkv4 \
      --case prefill_b8_s64_d64_h16_hkv4 \
      --case prefill_b8_s125_d64_h16_hkv4 \
      --case prefill_b8_s2048_d128_h16_hkv4)
}

usage() {
  cat <<EOF
Usage:
  $0 precision <suite>
  $0 performance <suite> <variant>

Suites: baseline, halflocal, fulllocal
EOF
}

mode="${1:-}"
case "$mode" in
  precision)
    suite="${2:-}"
    if [[ -z "$suite" ]]; then
      usage
      exit 1
    fi
    for variant in "${VARIANTS[@]}"; do
      select_variant "$suite" "$variant"
      build_and_install
      run_precision "$variant"
    done
    ;;
  performance)
    suite="${2:-}"
    variant="${3:-}"
    if [[ -z "$suite" || -z "$variant" ]]; then
      usage
      exit 1
    fi
    select_variant "$suite" "$variant"
    build_and_install
    run_perf_selected_s "$suite" "$variant"
    ;;
  *)
    usage
    exit 1
    ;;
esac
