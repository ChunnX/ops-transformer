#!/usr/bin/env bash
# init.sh — SageAttention2 算子门禁容器一键初始化（幂等，可重复执行）。
#
# 职责（独立仓库交付契约：所有外部依赖内聚到算子 docker/ 子目录，不引用仓库外宿主路径）：
#   1. 校验宿主 docker + compose 插件；
#   2. 创建 docker/cache、docker/out、docker/third_party（均为 gitignore 目录）；
#   3. 首次运行时生成 docker/.env（自 .env.example）；
#   4. 拉取官方 CATLASS（cann/catlass）固定 commit 到 docker/third_party/catlass
#      （唯一进入编译链的外部组件；已存在则校验/对齐 commit）；
#   5. 校验 CANN 基础镜像存在性（缺失则拉取默认镜像，可用 .env 覆盖）；
#   6. 可选：--golden <路径>（或 GOLDEN_SRC 环境变量）把 golden 数据集复制到
#      仓库根 golden/（位级比对基线，属交付数据；仓库根已有则跳过，绝不改写）。
#
# 用法：
#   bash docker/init.sh                    # 全量初始化（CATLASS + 镜像 + .env）
#   bash docker/init.sh --golden <path>    # 同上, 并置备 golden 数据集
#   GOLDEN_SRC=<path> bash docker/init.sh  # 等价 --golden
# 可覆盖项（环境变量）：CATLASS_REPO / CATLASS_COMMIT / CANN_IMAGE
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(cd "${DIR}/.." && pwd)"
REPO_ROOT="$(cd "${OP_DIR}/../../.." && pwd)"

CATLASS_REPO="${CATLASS_REPO:-https://gitcode.com/cann/catlass.git}"
CATLASS_COMMIT="${CATLASS_COMMIT:-e420328d38945dea53ac8ee56795215db20b943b}"
CATLASS_DIR="${DIR}/third_party/catlass"
GOLDEN_SRC="${GOLDEN_SRC:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --golden)
            GOLDEN_SRC="${2:?--golden 需要路径参数}"
            shift 2
            ;;
        --golden=*)
            GOLDEN_SRC="${1#*=}"
            shift
            ;;
        -h|--help)
            sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "未知参数: $1（用法见 bash docker/init.sh --help）" >&2
            exit 2
            ;;
    esac
done

echo "== [1/5] 校验 docker 与 compose 插件 =="
command -v docker >/dev/null 2>&1 || { echo "[ERROR] 未找到 docker，宿主唯一依赖即 Docker（含 compose 插件）。" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "[ERROR] 未找到 docker compose 插件。" >&2; exit 1; }

echo "== [2/5] 创建容器工作目录（cache/out/third_party, 均 gitignore）=="
mkdir -p "${DIR}/cache" "${DIR}/out" "${DIR}/third_party"

echo "== [3/5] .env（首次生成）=="
if [ ! -f "${DIR}/.env" ] && [ -f "${DIR}/.env.example" ]; then
    cp "${DIR}/.env.example" "${DIR}/.env"
    echo "  已从 .env.example 生成 .env（可按需修改）"
else
    echo "  .env 已存在或无模板，跳过"
fi

echo "== [4/5] CATLASS（官方 cann/catlass, 固定 commit）=="
if [ ! -d "${CATLASS_DIR}/.git" ]; then
    echo "  克隆 ${CATLASS_REPO} ..."
    git clone --quiet "${CATLASS_REPO}" "${CATLASS_DIR}"
fi
current_commit="$(git -C "${CATLASS_DIR}" rev-parse HEAD)"
if [ "${current_commit}" != "${CATLASS_COMMIT}" ]; then
    echo "  对齐到固定 commit ${CATLASS_COMMIT}（当前 ${current_commit}）..."
    git -C "${CATLASS_DIR}" fetch --quiet origin "${CATLASS_COMMIT}" 2>/dev/null \
        || git -C "${CATLASS_DIR}" fetch --quiet origin
    git -C "${CATLASS_DIR}" checkout --quiet "${CATLASS_COMMIT}"
fi
test -f "${CATLASS_DIR}/include/catlass/catlass.hpp" \
    || { echo "[ERROR] ${CATLASS_DIR} 缺少 include/catlass/catlass.hpp，不是官方 cann/catlass checkout。" >&2; exit 1; }
echo "  CATLASS 就绪: ${CATLASS_DIR} @ $(git -C "${CATLASS_DIR}" rev-parse --short HEAD)"

echo "== [5/5] CANN 基础镜像存在性 =="
# 与 docker-compose.yml 保持同一默认值（唯二引用点：compose 默认 + 此处校验）
CANN_IMAGE_EFF="${CANN_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.1-950-ubuntu22.04-py3.10-devel}"
if [ -f "${DIR}/.env" ]; then
    # .env 中的 CANN_IMAGE 优先（compose 实际消费的就是它）
    CANN_IMAGE_EFF="$(grep -E '^CANN_IMAGE=' "${DIR}/.env" | tail -n1 | cut -d= -f2- || true)"
    CANN_IMAGE_EFF="${CANN_IMAGE_EFF:-${CANN_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.1-950-ubuntu22.04-py3.10-devel}}"
fi
if docker image inspect "${CANN_IMAGE_EFF}" >/dev/null 2>&1; then
    echo "  基础镜像已存在: ${CANN_IMAGE_EFF}"
else
    echo "  本地缺失, 拉取 ${CANN_IMAGE_EFF} ..."
    docker pull --quiet "${CANN_IMAGE_EFF}"
fi

# 可选：golden 数据集置备（复制到仓库根 golden/，测试向上搜索即命中；不重生成、不改写）
if [ -n "${GOLDEN_SRC}" ]; then
    echo "== 附加: golden 数据集置备 =="
    GOLDEN_DST="${REPO_ROOT}/golden"
    if [ -f "${GOLDEN_DST}/manifest.json" ]; then
        echo "  仓库根 golden/ 已存在, 跳过（绝不改写既有 golden）"
    elif [ -f "${GOLDEN_SRC}/manifest.json" ]; then
        cp -R "${GOLDEN_SRC}" "${GOLDEN_DST}"
        find "${GOLDEN_DST}" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
        echo "  已复制 $(du -sh "${GOLDEN_DST}" | cut -f1) golden 数据到 ${GOLDEN_DST}"
    else
        echo "[WARN] ${GOLDEN_SRC} 下没有 manifest.json, 跳过 golden 置备" >&2
    fi
fi

echo "== init 完成 =="
echo "  后续: bash run.sh host|kernel|python|opp|all （自动包装进容器, 容器内统一 BiSheng）"
echo "  交互: ./docker/sage exec"
