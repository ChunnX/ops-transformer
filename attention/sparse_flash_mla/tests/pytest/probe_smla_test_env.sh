#!/usr/bin/env bash
# 探测 SMLA pytest 环境并回写 smla_test_env.json 中的 probe_status / 可自动填充字段。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_JSON="${SCRIPT_DIR}/smla_test_env.json"

if [[ ! -f "${ENV_JSON}" ]]; then
  echo "missing ${ENV_JSON}" >&2
  exit 1
fi

SET_ENV_SCRIPT="$(python3 - <<'PY' "${ENV_JSON}"
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
print(cfg["ascend"]["set_env_script"])
PY
)"

CONDA_ENV="$(python3 - <<'PY' "${ENV_JSON}"
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
print(cfg["conda"]["env_name"])
PY
)"

# shellcheck disable=SC1090
source "${SET_ENV_SCRIPT}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || true
fi

PROBE_PY="$(mktemp)"
cat > "${PROBE_PY}" <<'PY'
import json, os, subprocess, sys
from datetime import datetime, timezone

env_json = sys.argv[1]
with open(env_json, "r", encoding="utf-8") as f:
    cfg = json.load(f)

status = cfg.setdefault("probe_status", {})
status["last_probed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# npu-smi
try:
    out = subprocess.check_output(["npu-smi", "info"], stderr=subprocess.STDOUT, text=True, timeout=30)
    status["npu_smi_ok"] = "910B" in out or "910" in out
except Exception as e:
    status["npu_smi_ok"] = False
    status["last_error"] = f"npu-smi: {e}"

# python / conda
cfg["conda"]["python_executable"] = sys.executable
cfg["conda"]["python_version"] = sys.version.split()[0]
status["conda_env_found"] = os.environ.get("CONDA_DEFAULT_ENV") == cfg["conda"]["env_name"]

# packages
for pkg, key in [("torch", "torch"), ("torch_npu", "torch_npu"), ("numpy", "numpy"), ("pytest", "pytest")]:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        cfg["python_packages"][key] = ver
    except Exception:
        cfg["python_packages"][key] = None

status["torch_import_ok"] = cfg["python_packages"]["torch"] is not None
status["torch_npu_import_ok"] = cfg["python_packages"]["torch_npu"] is not None

try:
    import torch
    import torch_npu
    torch_npu.npu.set_device(0)
    cfg["ascend"]["chip_name"] = torch_npu.npu.get_device_name(0)
    status["operator_call_ok"] = None
except Exception as e:
    status["last_error"] = f"torch_npu: {e}"

try:
    import torch
    if hasattr(torch.ops, "cann_ops_transformer"):
        status["operator_call_ok"] = hasattr(torch.ops.cann_ops_transformer, "sparse_flash_mla")
    else:
        status["operator_call_ok"] = False
except Exception as e:
    status["operator_call_ok"] = False
    status["last_error"] = f"operator: {e}"

cfg["ascend"]["cann_toolkit_home"] = os.environ.get("ASCEND_TOOLKIT_HOME", cfg["ascend"].get("cann_toolkit_home"))
cfg["ascend"]["ascend_opp_path"] = os.environ.get("ASCEND_OPP_PATH", cfg["ascend"].get("ascend_opp_path"))

with open(env_json, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(json.dumps(status, indent=2, ensure_ascii=False))
PY

python3 "${PROBE_PY}" "${ENV_JSON}"
rm -f "${PROBE_PY}"
echo "updated ${ENV_JSON}"
