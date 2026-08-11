# vllm-ascend 容器环境搭建完整步骤

## 步骤一：宿主机清除代理

```bash
export http_proxy=""
export https_proxy=""
export no_proxy="localhost,127.0.0.1"
```

## 步骤二：基于镜像创建容器

```bash
docker run --name tyy_b120 -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /dev:/dev \
    -v /dev/nvme1n1p2:/dev/nvme1n1p2 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy= \
    -e https_proxy= \
    quay.io/ascend/vllm-ascend:v0.20.2rc1
```

> 注：原指南镜像 `ddfa90673e90` 不存在，选用了 `quay.io/ascend/vllm-ascend:v0.20.2rc1`

## 步骤三：进入容器

```bash
docker exec -it tyy_b120 bash
```

## 步骤四：配置代理与 pip 镜像源

```bash
# 配置代理（按需）
HOST_IP=xxx
export http_proxy="http://${HOST_IP}:3128"
export https_proxy=${http_proxy}
export ftp_proxy=${http_proxy}

# 清除 git 代理和 insteadOf 重写
git config --global http.sslVerify false
git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global --unset url.https://gh-proxy.org/https://github.com/.insteadof
git config --global --unset url.https://mirror.ghproxy.com/https://github.com/.insteadof

# 配置 pip 镜像源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 步骤五：卸载镜像自带的 vllm 和 vllm-ascend

```bash
pip uninstall vllm vllm_ascend -y
```

## 步骤六：克隆代码仓库

```bash
cd /home
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/ader47/vllm-ascend.git
git clone https://gitcode.com/yrewzjsx/memcache.git
git clone https://gitcode.com/Ascend/memfabric_hybrid
```

> vllm-ascend 克隆超时，改用 `git clone --depth 1`，后台 nohup 克隆成功

## 步骤七：切换 vllm 到 v0.19.0

```bash
cd /home/vllm
git fetch --depth 1 origin tag v0.19.0
git checkout v0.19.0
```

## 步骤八：源码安装 vllm（可编辑模式）

```bash
cd /home/vllm
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e .
```

## 步骤九：源码安装 vllm-ascend（可编辑模式）

```bash
cd /home/vllm-ascend
pip install -v -e .
```

### 编译失败排查与修复（关键问题）

**问题1**：`TILINGKEY_PAR_COMPILE=1` 导致 opc 并行编译冲突 `Another process is using this dir`

**修复**：修改 `build.sh` 限制并行度 + 禁用 `TILINGKEY_PAR_COMPILE`

```bash
# 限制编译并行度为1（完全串行）
sed -i 's/JOB_NUM="-j${CPU_NUM}"/JOB_NUM="-j1"/' /home/vllm-ascend/csrc/build.sh

# 禁用 TILINGKEY_PAR_COMPILE
find /home/vllm-ascend/csrc -maxdepth 2 -name 'CMakeLists.txt' -o -name '*.cmake' | \
    xargs grep -l 'TILINGKEY_PAR_COMPILE' | while read f; do
    sed -i 's/TILINGKEY_PAR_COMPILE=1/TILINGKEY_PAR_COMPILE=0/g' "$f"
done

# 彻底清理 build 目录后重新安装
rm -rf /home/vllm-ascend/csrc/build
cd /home/vllm-ascend
pip install -v -e . > /tmp/install_vllm_ascend_j1.log 2>&1
```

## 步骤十：监控编译进度

```bash
# 检查安装是否完成
grep INSTALL_EXIT_CODE /tmp/install_vllm_ascend_j1.log

# 查看编译进度
grep -o '\[[0-9]*/533\]' /tmp/install_vllm_ascend_j1.log | tail -1

# 检查错误数
grep -c "Another process" /tmp/install_vllm_ascend_j1.log
grep -c "FAILED" /tmp/install_vllm_ascend_j1.log

# 查看最新日志
tail -3 /tmp/install_vllm_ascend_j1.log
```

## 步骤六：安装 memfabric_hybrid 和 memcache

> 必须先安装 memfabric_hybrid，再安装 memcache（memcache 依赖 memfabric）

### 6.1 安装 memfabric_hybrid

```bash
# 下载代码（切换到 v1.0.0 分支）
cd /home
git clone -b v1.0.0 https://gitcode.com/Ascend/memfabric_hybrid
cd memfabric_hybrid
git clean -xdf
git reset --hard

# 拉取第三方库
git submodule update --recursive --init

# pybind11 版本不能超过 3.0.1（否则编译报错：def_property family does not currently support call_guard）
pip install pybind11==3.0.1

# 编译
# build_and_pack_run.sh 支持三个参数：<build_mode> <build_python> <xpu_type>
# build_mode: RELEASE 或 DEBUG，默认 RELEASE
# build_python: ON 或 OFF，默认 ON
# xpu_type: NPU 或 GPU，默认 NPU
bash script/build_and_pack_run.sh --build_mode RELEASE --build_python ON --xpu_type NPU

# 安装，编译完后会生成 run 包到 output 目录，格式为 memfabric-hybrid-${version}_${os}_${arch}.run
cd output
bash memfabric_hybrid-1.0.0_linux_aarch64.run
source /usr/local/memfabric_hybrid/set_env.sh

# 检查 whl 包是否安装成功，未安装则手动安装
pip show memfabric_hybrid
cd /usr/local/memfabric_hybrid/1.0.0/aarch64-linux/wheel
pip install memfabric_hybrid-1.0.0-cp311-cp311-linux_aarch64.whl
```

#### memfabric_hybrid 安装目录结构

```
/usr/local/memfabric_hybrid
├── 1.0.0
│   ├── aarch64-linux
│   │   ├── bin
│   │   ├── include
│   │   │   ├── hybm
│   │   │   │   ├── hybm.h
│   │   │   │   ├── hybm_big_mem.h
│   │   │   │   ├── hybm_data_op.h
│   │   │   │   └── hybm_def.h
│   │   │   └── smem
│   │   │       ├── device
│   │   │       │   ├── smem_shm_aicore_base_api.h
│   │   │       │   ├── smem_shm_aicore_base_copy.h
│   │   │       │   ├── smem_shm_aicore_base_define.h
│   │   │       │   ├── smem_shm_aicore_base_meta.h
│   │   │       │   └── smem_shm_aicore_base_rdma.h
│   │   │       └── host
│   │   │           ├── smem.h
│   │   │           ├── smem_bm.h
│   │   │           ├── smem_bm_def.h
│   │   │           ├── smem_shm.h
│   │   │           ├── smem_shm_def.h
│   │   │           ├── smem_trans.h
│   │   │           └── smem_trans_def.h
│   │   ├── lib64
│   │   │   ├── libmf_hybm_core.so
│   │   │   └── libmf_smem.so
│   │   ├── script
│   │   │   ├── certs
│   │   │   │   ├── generate_client_cert.py
│   │   │   │   ├── generate_crl.py
│   │   │   │   ├── generate_root_cert.py
│   │   │   │   └── generate_server_cert.py
│   │   │   └── mock_server
│   │   └── wheel
│   │       ├── mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl
│   │       └── memfabric_hybrid-1.0.0-cp311-cp311-linux_aarch64.whl
│   ├── uninstall.sh
│   └── version.info
├── latest -> 1.0.0
└── set_env.sh
```

### 6.2 安装 memcache

```bash
# 下载代码（切换到 v1.0.0 分支）
cd /home
git clone -b v1.0.0 https://gitcode.com/Ascend/memcache
cd memcache
git clean -xdf
git reset --hard

# 拉取第三方库
git submodule update --recursive --init
git submodule update --remote 3rdparty/memfabric_hybrid

# 编译
# build_and_pack_run.sh 支持两个参数：<build_mode> <build_python>
# build_mode: RELEASE 或 DEBUG
# build_python: ON 或 OFF
bash script/build_and_pack_run.sh --build_mode RELEASE

# 安装，编译完后会生成 run 包到 output 目录，格式为 memcache_hybrid-${version}_${os}_${arch}.run
cd output
bash memcache_hybrid-1.0.0_linux_aarch64.run
source /usr/local/memcache_hybrid/set_env.sh

# 检查 whl 包是否安装成功，未安装则手动安装
pip show memcache_hybrid
cd /usr/local/memcache_hybrid/1.0.0/aarch64-linux/wheel
pip install memcache_hybrid-1.0.0-cp311-cp311-linux_aarch64.whl
```

#### memcache 安装目录结构

```
/usr/local/memcache_hybrid/
├── 1.0.0
│   ├── aarch64-linux
│   │   ├── bin
│   │   │   └── mmc_meta_service
│   │   ├── include
│   │   │   └── memcache
│   │   │       ├── cpp
│   │   │       │   └── mmcache.h
│   │   │       ├── mmc.h
│   │   │       ├── mmc_client.h
│   │   │       ├── mmc_def.h
│   │   │       └── mmc_service.h
│   │   ├── lib64
│   │   │   ├── _pymmc.cpython-311-aarch64-linux-gnu.so
│   │   │   └── libmf_memcache.so
│   │   ├── logs
│   │   │   ├── mmc-meta-audit.log
│   │   │   └── mmc-meta.log
│   │   ├── script
│   │   │   ├── ha
│   │   │   │   └── test-mmc-meta-ha.py
│   │   │   ├── k8s_deploy
│   │   │   │   ├── account-role-demo.yaml
│   │   │   │   ├── local-pods-demo.yaml
│   │   │   │   ├── meta-cluster-ip-demo.yaml
│   │   │   │   ├── meta-lease-lock-demo.yaml
│   │   │   │   └── meta-pods-demo.yaml
│   │   │   └── mock_server
│   │   │       ├── server.py
│   │   │       └── smem_bm_server.py
│   │   └── wheel
│   │       └── memcache_hybrid-1.0.0-cp311-cp311-linux_aarch64.whl
│   ├── config
│   │   ├── mmc-local.conf
│   │   └── mmc-meta.conf
│   ├── uninstall.sh
│   └── version.info
├── latest -> 1.0.0
└── set_env.sh
```

### 6.3 编译问题排查

**问题**：pybind11 版本过高导致编译失败

```
error: static assertion failed: def_property family does not currently support call_guard. Use a py::cpp_function instead.
```

**修复**：降级 pybind11 到 3.0.1

```bash
pip install pybind11==3.0.1
```

然后清理 build 目录重新编译：

```bash
cd /home/memfabric_hybrid  # 或 /home/memcache
git clean -xdf
git reset --hard
git submodule update --recursive --init
bash script/build_and_pack_run.sh --build_mode RELEASE --build_python ON --xpu_type NPU
```

## 环境信息

- 服务器：121.37.53.41 (SSH端口 10001)
- NPU：910B3 x 8
- 容器：tyy_b120
- 基础镜像：quay.io/ascend/vllm-ascend:v0.20.2rc1
- Python：3.11.15
- vllm：0.19.0+empty (源码可编辑安装)
- vllm-ascend：0.1.dev1 (源码可编辑安装)
- memfabric_hybrid：1.0.8
- memcache_hybrid：1.0.4
- pybind11：3.0.1 (降级以兼容 memfabric 编译)
