#!/usr/bin/env bash
# MQSMLA DSL: paramset/Excel -> CPU input + golden -> installed torch.ops.
set -euo pipefail

TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

usage() {
    cat <<'EOF'
Usage: bash test.sh <command> [options] [-- pytest-options]
  single       Generate inputs/golden and run current paramset (or --excel).
  batch_save   Save CPU inputs/golden only; no NPU or installed wheel required.
  batch_exec   Replay saved .pt files without Excel, input generation or golden.
  batch        Save then replay each case; keep all saved data.
Options:
  --excel FILE       Workbook (batch_save/batch default: whitebox cases).
  --sheet NAME       Sheet name (default: decode).
  --pt-dir DIR       Data directory (default: tests/mqsmla_testcase).
  --device-id ID     NPU index (default: 0).
  --                 Forward remaining arguments to pytest, e.g. -k SHAPE_S1_03.
Load the CANN/Python environment first. Paths supplied by the caller are relative
to the caller's working directory. PYTHON selects the Python executable.
Saved data is kept; repeated saves replace the same case's file atomically.
batch_exec replays every .pt in its directory in filename order; use a separate
directory for each suite. DSL files use their own format, not mainline .pt format.
EOF
}

case "$command" in
    help|-h|--help) usage; exit 0 ;;
    single) data_mode=run ;;
    batch_save) data_mode=save ;;
    batch_exec) data_mode=replay ;;
    batch) data_mode=save-run ;;
    *) echo "Unknown command: $command" >&2; usage >&2; exit 2 ;;
esac

excel="${MQSMLA_EXCEL:-}"
sheet="${MQSMLA_SHEET:-decode}"
pt_dir="${MQSMLA_PT_DIR:-$TEST_DIR/mqsmla_testcase}"
device_id=0
pytest_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --excel|--sheet|--pt-dir|--device-id)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2; exit 2
            fi
            case "$1" in
                --excel) excel="$2" ;;
                --sheet) sheet="$2" ;;
                --pt-dir) pt_dir="$2" ;;
                --device-id) device_id="$2" ;;
            esac
            shift 2 ;;
        --) shift; pytest_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1; put pytest options after --" >&2; exit 2 ;;
    esac
done
if [[ ! "$device_id" =~ ^[0-9]+$ ]]; then
    echo '--device-id must be a non-negative integer' >&2; exit 2
fi
if [[ "$command" == batch_save || "$command" == batch ]]; then
    excel="${excel:-$TEST_DIR/whitebox/mqsmla_whitebox_cases.xlsx}"
fi
args=(-q -s --rootdir "$TEST_DIR" "$TEST_DIR/test_mqsmla.py" --data-mode "$data_mode"
      --pt-dir "$pt_dir" --device-id "$device_id")
if [[ "$data_mode" != replay && -n "$excel" ]]; then
    args+=(--excel "$excel" --sheet "$sheet")
fi
exec "$PYTHON_BIN" -m pytest "${args[@]}" "${pytest_args[@]}"
