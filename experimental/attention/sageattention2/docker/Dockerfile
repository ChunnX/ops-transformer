# SageAttention2 算子门禁容器（Atlas 950PR / BiSheng 统一编译）
# 默认国内网络: 华为云 SWR 基础镜像 + 华为云 apt/PyPI 源 + 阿里云 torch CPU 轮子源
# 镜像源自 ./docker/Dockerfile 配置，指令序列保持一致以复用本地构建层缓存。
# 海外覆盖示例:
#   CANN_IMAGE=quay.io/ascend/cann:9.2.0-beta.1-950-ubuntu22.04-py3.10-devel
# tag 规则 <CANN版本>-<芯片>-<OS>-<py>[-devel], 见 github.com/Ascend/cann-container-image
# 默认 tag 与验收环境 CANN 9.2.0-beta.1 对齐(950 系 devel, 含 bic/毕昇编译器与 googletest)
ARG CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.1-950-ubuntu22.04-py3.10-devel
FROM ${CANN_IMAGE}

ARG APT_MIRROR=http://mirrors.huaweicloud.com/ubuntu
RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
      [ -f "$f" ] && sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
        -e "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" "$f" || true; \
    done \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      git cmake ninja-build curl unzip bash-completion \
 && rm -rf /var/lib/apt/lists/*

# 挂载宿主工作区后避免 git dubious ownership
RUN git config --system --add safe.directory '*'

# PyPI 源: uv 与 pip 统一指向华为云镜像
ENV PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
    UV_DEFAULT_INDEX=https://mirrors.huaweicloud.com/repository/pypi/simple \
    UV_PYTHON_PREFERENCE=only-system

# 先单独安装 uv，后续依赖均通过 uv pip 安装
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache packaging numpy pytest

# torch 配套(可覆盖): 官方配套表 https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md
# 表内最新为 CANN 9.1.X <-> torch-npu 2.12.0; CANN 9.2 beta 阶段默认沿用, 验收有指定时覆盖
# torch 取 CPU 轮子(避免拉入 CUDA 依赖), 阿里云镜像; 海外可改
# TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC=torch==2.12.0
ARG TORCH_NPU_SPEC=torch-npu==2.12.0
ARG TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cpu/
RUN if [ -n "$TORCH_SPEC" ]; then \
      uv pip install --system --no-cache --index-url "${TORCH_INDEX}" ${TORCH_SPEC} \
      || uv pip install --system --no-cache ${TORCH_SPEC}; \
    fi \
 && if [ -n "$TORCH_NPU_SPEC" ]; then uv pip install --system --no-cache ${TORCH_NPU_SPEC}; fi

# CANN 环境兜底注入 bashrc
RUN set_env=/usr/local/Ascend/ascend-toolkit/set_env.sh; \
    [ -f "$set_env" ] && echo "source $set_env" >> /root/.bashrc || true

WORKDIR /work
CMD ["bash"]
