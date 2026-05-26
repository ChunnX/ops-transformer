# ops-transformer 一键构建与运行指南

## 目录结构约定

```
${CURRENT_DIR}/
├── ops-transformer/      # 算子工程
└── shmem/                # 通信库
```

## 前置依赖

| 依赖 | 说明 |
|------|------|
| CANN | 已解压并安装到 `CANN_PATH` 指向的目录，包含 `ascend-toolkit/` |
| TileXR | 已部署到 `TileXR_PATH`，提供 `cmake`、`ccache`、`mpich`、`set_proxy.sh` 等工具链 |

## 环境变量

```bash
export CANN_PATH=/home/z00699601/cann/
export TileXR_PATH=/home/z00699601/TileXR/
```

## 源码拉取（one_build.sh）

```bash
#!/bin/bash
set -e

source ${TileXR_PATH}/set_proxy.sh

CURRENT_DIR=$(pwd)

# 拉取算子工程
git clone https://gitcode.com/zhu-wangyi/ops-transformer.git
cd ops-transformer
git checkout feature/AIV_URMA_master_2N16P
OPTS_PATH=${CURRENT_DIR}/ops-transformer/

# 拉取 shmem
cd ${CURRENT_DIR}
git clone https://gitcode.com/zhu-wangyi/shmem.git
cd shmem
git checkout feature/CrossPOD_fix_2_mix_topo_ops_transformer
```

> **⚠️ 拉取完成后，请手动修改 `ops-transformer` 中的 IP 地址配置**，替换为实际两台机器的 IP。

## 一键构建与运行（one_run.sh）

```bash
#!/bin/bash
set -e

CURRENT_DIR=$(pwd)

# ---------- 基础环境 ----------
export PATH=${TileXR_PATH}/env/util/cmake/bin:${PATH}
export PATH=${TileXR_PATH}/env/util/ccache:${PATH}
export MPI_TEST_HOME=${TileXR_PATH}/env/util/mpich
export SHMEM_HOME=${CURRENT_DIR}/shmem/

source ${TileXR_PATH}/set_proxy.sh
source ${CANN_PATH}/ascend-toolkit/set_env.sh

# ---------- 构建 shmem ----------
cd ${SHMEM_HOME}
bash scripts/build.sh -examples -soc_type Ascend950

# ---------- 构建 ops-transformer ----------
cd ${CURRENT_DIR}/ops-transformer
time bash build.sh --pkg -j384 \
    -p ${CANN_PATH}/cann/ \
    --soc=ascend950 \
    --ops=moe_distribute_dispatch_v2

# ---------- 安装自定义算子包 ----------
bash ./build_out/cann-ops*.run \
    --install-path=${CANN_PATH}/cann/ \
    --force

source ${CANN_PATH}/cann/vendors/custom_transformer/bin/set_env.bash

# ---------- 运行示例 ----------
bash build.sh --run_example moe_distribute_dispatch_v2 eager cust \
    -p ${CANN_PATH}/cann/ \
    --soc=ascend950
```

## 完整执行步骤

```bash
# 1. 设置环境变量
export CANN_PATH=/home/z00699601/cann/
export TileXR_PATH=/home/z00699601/TileXR/

# 2. 进入工作目录
mkdir -p ~/work_dir && cd ~/work_dir

# 3. 拉取源码
bash one_build.sh

# 4. 修改 ops-transformer 中的 IP 配置（按需）

# 5. 一键构建并运行
bash one_run.sh
```