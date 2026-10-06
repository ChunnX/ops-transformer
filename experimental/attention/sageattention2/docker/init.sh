#!/usr/bin/env bash
# init.sh — 容器一次性置备（幂等；由 ./docker/sage init 显式触发，run.sh 不调用）。
# 做四件事：置备 gitignore 目录与 .env、克隆 CATLASS 固定 commit、拉取缺失的基础镜像、
# 可选复制 golden/prototypes 测试数据（已存在则绝不改写/重生成）。
#
# 用法：bash docker/init.sh [--golden <path>] [--prototypes <path>]
# 可覆盖项（环境变量）：CATLASS_REPO / CATLASS_COMMIT / CANN_IMAGE / GOLDEN_SRC / PROTOS_SRC
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CATLASS_REPO="${CATLASS_REPO:-https://gitcode.com/cann/catlass.git}"
CATLASS_COMMIT="${CATLASS_COMMIT:-e420328d38945dea53ac8ee56795215db20b943b}"
CATLASS_DIR="${DIR}/third_party/catlass"
GOLDEN_SRC="${GOLDEN_SRC:-}"
PROTOS_SRC="${PROTOS_SRC:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --golden)   GOLDEN_SRC="${2:?--golden 需要路径参数}"; shift 2 ;;
        --golden=*)   GOLDEN_SRC="${1#*=}"; shift ;;
        --prototypes)   PROTOS_SRC="${2:?--prototypes 需要路径参数}"; shift 2 ;;
        --prototypes=*) PROTOS_SRC="${1#*=}"; shift ;;
        -h|--help)
            echo "usage: bash docker/init.sh [--golden <path>] [--prototypes <path>]"
            echo "env: CATLASS_REPO / CATLASS_COMMIT / CANN_IMAGE / GOLDEN_SRC / PROTOS_SRC"
            exit 0
            ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

echo "== 目录与 .env =="
mkdir -p "${DIR}/cache" "${DIR}/out" "${DIR}/third_party"
[ -f "${DIR}/.env" ] || cp "${DIR}/.env.example" "${DIR}/.env"

echo "== CATLASS（官方 cann/catlass, 固定 commit）=="
if [ ! -d "${CATLASS_DIR}/.git" ]; then
    git clone --quiet "${CATLASS_REPO}" "${CATLASS_DIR}"
fi
current_commit="$(git -C "${CATLASS_DIR}" rev-parse HEAD)"
if [ "${current_commit}" != "${CATLASS_COMMIT}" ]; then
    git -C "${CATLASS_DIR}" fetch --quiet origin "${CATLASS_COMMIT}" \
        || git -C "${CATLASS_DIR}" fetch --quiet origin
    git -C "${CATLASS_DIR}" checkout --quiet "${CATLASS_COMMIT}"
fi

echo "== 基础镜像 =="
# 镜像名与 docker-compose.yml 同源（compose 消费 .env / CANN_IMAGE）
CANN_IMAGE_EFF="${CANN_IMAGE:-$(grep -E '^CANN_IMAGE=' "${DIR}/.env" 2>/dev/null | tail -n1 | cut -d= -f2- || true)}"
CANN_IMAGE_EFF="${CANN_IMAGE_EFF:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.1-950-ubuntu22.04-py3.10-devel}"
docker image inspect "${CANN_IMAGE_EFF}" >/dev/null 2>&1 || docker pull --quiet "${CANN_IMAGE_EFF}"

echo "== golden 数据集（docker/golden, 交付基线, 不自动重生成）=="
if [ -f "${DIR}/golden/manifest.json" ]; then
    echo "  已就绪 ($(du -sh "${DIR}/golden" | cut -f1))"
elif [ -n "${GOLDEN_SRC}" ] && [ -f "${GOLDEN_SRC}/manifest.json" ]; then
    mkdir -p "${DIR}/golden"
    cp -R "${GOLDEN_SRC}/." "${DIR}/golden/"
    find "${DIR}/golden" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
    echo "  已复制 $(du -sh "${DIR}/golden" | cut -f1)"
else
    echo "[WARN] 未提供 --golden <path>, golden 门禁在置备前不可用" >&2
fi

echo "== prototypes 数值参考（docker/third_party/prototypes, test_quant 消费）=="
if [ -d "${DIR}/third_party/prototypes" ]; then
    echo "  已就绪 ($(du -sh "${DIR}/third_party/prototypes" | cut -f1))"
elif [ -n "${PROTOS_SRC}" ] && [ -f "${PROTOS_SRC}/tilelang_sageattn_prototype.py" ]; then
    mkdir -p "${DIR}/third_party/prototypes"
    cp -R "${PROTOS_SRC}/." "${DIR}/third_party/prototypes/"
    find "${DIR}/third_party/prototypes" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
    echo "  已复制"
else
    echo "[WARN] 未提供 --prototypes <path>, test_quant 的 TileLang 数值参考门禁在置备前不可用" >&2
fi

echo "== init 完成 =="
