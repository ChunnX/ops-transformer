#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the generated case directory}"
DEVICE_ID="${DEVICE_ID:-0}"
SUITE="${1:-smoke}"

python3 "${ROOT_DIR}/check_case_coverage.py"
python3 "${ROOT_DIR}/generate_case.py" --suite "${SUITE}" --output-root "${DATA_ROOT}"

python3 - "${ROOT_DIR}" "${DATA_ROOT}" "${SUITE}" "${DEVICE_ID}" <<'PY'
import sys
from pathlib import Path

root, data_root, suite, device_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
sys.path.insert(0, str(root))
from case_utils import load_case_file

for case in load_case_file(root / f"cases_{suite}.yaml"):
    print(f"RUN {case['id']}", flush=True)
    code = __import__("subprocess").run(
        [sys.executable, str(root / "run_case.py"), str(data_root / case["id"]), "--device-id", device_id],
        check=False,
    ).returncode
    if code:
        raise SystemExit(code)
PY
