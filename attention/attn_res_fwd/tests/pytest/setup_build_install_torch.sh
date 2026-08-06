#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# 在任意机器上：编译 + 安装 AttnResFwd 自定义算子包，并准备 Torch 适配（PYTHONPATH）。
#
# 用法：
#   # 1) 配置 CANN（任选其一）
#   export ASCEND_HOME_PATH=/path/to/cann-x.y   # 推荐：已含 set_env 的 toolkit 根
#   # 或：
#   export CANN_SET_ENV=/path/to/ascend-toolkit/set_env.sh
#
#   # 2) 编译安装
#   bash setup_build_install_torch.sh
#   bash setup_build_install_torch.sh --oom          # 开启 kernel OOM 检测
#   bash setup_build_install_torch.sh --soc=ascend910b -j16
#   bash setup_build_install_torch.sh --skip-build   # 只装已有 .run / 只配环境说明
#   bash setup_build_install_torch.sh --print-only   # 只打印 attenres_lite 命令
#
# 环境变量（均可选）：
#   ASCEND_HOME_PATH / CANN_SET_ENV / CANN_PKG_DIR
#   CUSTOM_INSTALL_PATH   默认：<ops-transformer>/custompkg_install
#   SOC                   默认 ascend910b
#   JOBS                  默认 8
#   OPS                   默认 attn_res_fwd
#   CONDA_ENV_NAME        默认 kdy311（仅用于打印后续命令）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .../attention/attn_res_fwd/tests/pytest → ops-transformer 根
OPS_TRANSFORMER_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TORCH_EXT_ROOT="${OPS_TRANSFORMER_ROOT}/torch_extension"
BUILD_SH="${OPS_TRANSFORMER_ROOT}/build.sh"

SOC="${SOC:-ascend910b}"
JOBS="${JOBS:-8}"
OPS="${OPS:-attn_res_fwd}"
CUSTOM_INSTALL_PATH="${CUSTOM_INSTALL_PATH:-${OPS_TRANSFORMER_ROOT}/custompkg_install}"
ENABLE_OOM=0
SKIP_BUILD=0
PRINT_ONLY=0
FORCE_INSTALL=1

usage() {
  sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

for arg in "$@"; do
  case "${arg}" in
    -h|--help) usage ;;
    --oom) ENABLE_OOM=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --print-only) PRINT_ONLY=1; SKIP_BUILD=1 ;;
    --soc=*) SOC="${arg#*=}" ;;
    -j*) JOBS="${arg#-j}" ;;
    --jobs=*) JOBS="${arg#*=}" ;;
    --install-path=*) CUSTOM_INSTALL_PATH="${arg#*=}" ;;
    --ops=*) OPS="${arg#*=}" ;;
    --no-force) FORCE_INSTALL=0 ;;
    *)
      echo "ERROR: unknown arg: ${arg}" >&2
      usage
      ;;
  esac
done

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup][ERROR] %s\n' "$*" >&2; exit 1; }

resolve_cann_set_env() {
  if [[ -n "${CANN_SET_ENV:-}" && -f "${CANN_SET_ENV}" ]]; then
    echo "${CANN_SET_ENV}"
    return 0
  fi
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    local cands=(
      "${ASCEND_HOME_PATH}/../ascend-toolkit/set_env.sh"
      "${ASCEND_HOME_PATH}/set_env.sh"
      "${ASCEND_HOME_PATH}/ascend-toolkit/set_env.sh"
    )
    local p
    for p in "${cands[@]}"; do
      if [[ -f "${p}" ]]; then
        echo "$(cd "$(dirname "${p}")" && pwd)/$(basename "${p}")"
        return 0
      fi
    done
  fi
  if [[ -n "${CANN_PKG_DIR:-}" && -f "${CANN_PKG_DIR}/ascend-toolkit/set_env.sh" ]]; then
    echo "${CANN_PKG_DIR}/ascend-toolkit/set_env.sh"
    return 0
  fi
  # 常见安装位置
  local p
  for p in \
    "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
    "/usr/local/Ascend/cann/set_env.sh" \
    "${HOME}/Ascend/ascend-toolkit/set_env.sh"
  do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

print_run_commands() {
  local custom_setenv="${CUSTOM_INSTALL_PATH}/vendors/custom_transformer/bin/set_env.bash"
  cat <<EOF

======== 后续在本机执行 attenres_lite.py ========

# 进入测试目录
cd ${SCRIPT_DIR}

# Python（按本机 conda 环境改名）
# source env_kdy.sh
# 或：conda activate \${CONDA_ENV_NAME:-kdy311}

# CANN + 自定义 OPP + Torch 适配 PYTHONPATH
# precision_compare.py 已在本目录，无需额外 ANALYSE_PATH
source $(resolve_cann_set_env 2>/dev/null || echo /path/to/ascend-toolkit/set_env.sh)
source ${custom_setenv}
export PYTHONPATH="${TORCH_EXT_ROOT}:\${PYTHONPATH:-}"

# 清 JIT 缓存（改了 torch_extension/csrc 或换包后建议）
rm -rf "\${HOME}/.cache/torch_extensions"/*/attn_res_fwd 2>/dev/null || true

# 选卡
export DEVICE_ID=\${DEVICE_ID:-0}
# export FORCE_AUTO_DEVICE=1

# --- 常用命令 ---
# 默认生产 shape：T=544 N=12 H=7168，精度+profiler
python3 attenres_lite.py

# 仅精度
python3 attenres_lite.py --no-perf

# T=16000 N=2 H=7168（示例）
python3 attenres_lite.py --num-tokens 16000 --num-blocks 2 --hidden-size 7168

# 只对比 golden 与自定义融合算子
python3 attenres_lite.py --compare attn_res_fwd_cann --baseline apply_attn_res_cust

# 列出实现名
python3 attenres_lite.py --list

# 或一键：
# bash test_run.sh lite --num-tokens 16000 --num-blocks 2 --hidden-size 7168

===============================================
EOF
}

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  print_run_commands
  exit 0
fi

[[ -f "${BUILD_SH}" ]] || die "build.sh not found: ${BUILD_SH}（请在 ops-transformer 仓内使用本脚本）"
[[ -d "${TORCH_EXT_ROOT}/cann_ops_transformer" ]] || die "torch_extension 缺失: ${TORCH_EXT_ROOT}"

CANN_SET_ENV_FILE="$(resolve_cann_set_env)" || die "找不到 CANN set_env.sh。请设置 ASCEND_HOME_PATH 或 CANN_SET_ENV"
log "OPS_TRANSFORMER_ROOT=${OPS_TRANSFORMER_ROOT}"
log "CANN_SET_ENV=${CANN_SET_ENV_FILE}"
log "CUSTOM_INSTALL_PATH=${CUSTOM_INSTALL_PATH}"
log "SOC=${SOC} OPS=${OPS} JOBS=${JOBS} OOM=${ENABLE_OOM}"

# shellcheck disable=SC1090
set +u
source "${CANN_SET_ENV_FILE}"
set -u
[[ -n "${ASCEND_HOME_PATH:-}" ]] || die "source set_env 后 ASCEND_HOME_PATH 仍为空"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  log "building custom package..."
  cd "${OPS_TRANSFORMER_ROOT}"
  BUILD_ARGS=(--pkg --soc="${SOC}" --ops="${OPS}" -j"${JOBS}")
  if [[ "${ENABLE_OOM}" -eq 1 ]]; then
    BUILD_ARGS+=(--oom)
  fi
  bash build.sh "${BUILD_ARGS[@]}"

  RUN_PKG="$(ls -1t "${OPS_TRANSFORMER_ROOT}/build_out"/cann-ops-transformer-custom_linux-*.run 2>/dev/null | head -1 || true)"
  [[ -n "${RUN_PKG}" && -f "${RUN_PKG}" ]] || die "未找到 build_out 下的 .run 安装包"
  log "installing ${RUN_PKG} -> ${CUSTOM_INSTALL_PATH}"
  mkdir -p "${CUSTOM_INSTALL_PATH}"
  INSTALL_ARGS=(--quiet --install-path="${CUSTOM_INSTALL_PATH}")
  if [[ "${FORCE_INSTALL}" -eq 1 ]]; then
    INSTALL_ARGS+=(--force)
  fi
  bash "${RUN_PKG}" "${INSTALL_ARGS[@]}"
else
  log "skip build (--skip-build)"
fi

CUSTOM_SETENV="${CUSTOM_INSTALL_PATH}/vendors/custom_transformer/bin/set_env.bash"
[[ -f "${CUSTOM_SETENV}" ]] || die "自定义包 set_env 不存在: ${CUSTOM_SETENV}（请先成功 --pkg 安装）"

# shellcheck disable=SC1090
set +u
# set_env.bash 可能依赖 ASCEND_CUSTOM_OPP_PATH 已定义
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
source "${CUSTOM_SETENV}"
set -u

export PYTHONPATH="${TORCH_EXT_ROOT}:${PYTHONPATH:-}"
# 清理本算子 JIT，避免旧 .so
rm -rf "${HOME}/.cache/torch_extensions"/py*/attn_res_fwd 2>/dev/null || true
rm -rf "${HOME}/.cache/torch_extensions"/py*/*/attn_res_fwd 2>/dev/null || true

log "verify python import (meta only)..."
python3 - <<'PY'
import os, sys
print("ASCEND_HOME_PATH=", os.environ.get("ASCEND_HOME_PATH"))
print("ASCEND_CUSTOM_OPP_PATH=", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
print("PYTHONPATH head=", (os.environ.get("PYTHONPATH") or "").split(os.pathsep)[:3])
try:
    from cann_ops_transformer.ops import attn_res_fwd as m
    print("OK: cann_ops_transformer.ops.attn_res_fwd loaded, builder=", m.attn_res_fwd_op_builder.name)
except Exception as e:
    print("WARN: import attn_res_fwd failed (torch/torch_npu 未就绪时可能正常):", e)
PY

print_run_commands
log "done."
