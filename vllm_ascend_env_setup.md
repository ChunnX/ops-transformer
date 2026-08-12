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

## 步骤七：拉起 memcache 后端服务

### 7.1 修改配置文件

```bash
# 拷贝配置文件到自己的目录
mkdir -p /home/lsy/layerwise
cp /usr/local/memcache_hybrid/latest/config/mmc-meta.conf /home/lsy/layerwise/mmc-meta.conf
cp /usr/local/memcache_hybrid/latest/config/mmc-local.conf /home/lsy/layerwise/mmc-local.conf

# 修改 mmc-local.conf
# 1. protocol 改为 device_sdma（适合A3）
sed -i 's/ock.mmc.local_service.protocol = host_rdma/ock.mmc.local_service.protocol = device_sdma/' /home/lsy/layerwise/mmc-local.conf

# 2. dram.size 改为 10GB
sed -i 's/ock.mmc.local_service.dram.size = 1GB/ock.mmc.local_service.dram.size = 10GB/' /home/lsy/layerwise/mmc-local.conf
```

### 7.2 拉起 MetaService 服务

```bash
# 设置环境变量
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/home/lsy/layerwise/mmc-meta.conf

# 方法一：二进制方式拉起（后台运行）
nohup /usr/local/memcache_hybrid/latest/aarch64-linux/bin/mmc_meta_service > /tmp/mmc_meta_service.log 2>&1 &
echo $! > /tmp/mmc_meta_pid.txt

# 方法二：Python 方式拉起
# python3 -c "from memcache_hybrid import MetaService; MetaService.main()"
```

### 7.3 验证 MetaService 服务

```bash
# 检查进程
ps -p $(cat /tmp/mmc_meta_pid.txt) -o pid,stat,etime,args

# 检查端口（5000=MetaService, 6000=ConfigStore, 8000=HTTP）
cat /proc/net/tcp | awk '{print $2}' | grep -i ":1388"  # port 5000
cat /proc/net/tcp | awk '{print $2}' | grep -i ":1770"  # port 6000

# 查看日志
tail -5 /var/log/memcache_hybrid/logs/mmc-meta.log
```

### 7.4 常见报错

**端口被占用**：
```
address in use for bind listen on 127.0.0.1:5000
```
解决：修改 mmc-meta.conf 和 mmc-local.conf 中的端口号（如改为 5001）

## 步骤八：版本兼容性修复（关键）

> vllm-ascend 与 vllm 版本必须严格匹配，否则会出现各种 ImportError

### 8.1 版本兼容性问题说明

| vllm 版本 | vllm-ascend 版本 | 兼容性 |
|-----------|------------------|--------|
| v0.19.0 | fork main | ❌ 缺少 `_approximate_gcd` |
| latest main | fork main | ❌ triton 3.5 gluon 不兼容 |
| v0.20.2 | fork main | ❌ `FusedMoE` 改名为 `FusedMoEFactory` |
| latest main | 官方 v0.20.2rc | ❌ `Glm47MoeModelToolParser._extract_tool_call_regions` 不存在 |
| latest main | 官方 v0.20.2rc | ❌ `get_total_cp_world_size` 不存在 |
| **v0.20.2** | **官方 v0.20.2rc** | ✅ **兼容** |

### 8.2 正确的版本组合安装

```bash
# 1. 安装 vllm v0.20.2
cd /home/vllm
git checkout v0.20.2
pip uninstall vllm -y
VLLM_TARGET_DEVICE=empty pip install -v -e .

# 2. 克隆官方 vllm-ascend v0.20.2rc 分支
cd /home
git clone -b releases/v0.20.2rc --depth 1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-official

# 3. 安装 vllm-ascend
cd /home/vllm-ascend-official
pip install -v -e .

# 4. 验证
pip show vllm | grep Version   # 0.20.2+empty
pip show vllm_ascend | grep Version  # 0.1.dev1+gface36594
python3 -c "import vllm_ascend; print('OK')"
```

### 8.3 triton 兼容性修复

triton-ascend 3.2.1 自带 triton 3.2.0，可能缺少 `next_power_of_2` 函数：

```bash
# 检查
python3 -c "import triton; print(hasattr(triton, 'next_power_of_2'))"

# 如果缺失，手动添加到 triton __init__.py
TRITON_INIT=$(python3 -c "import triton; import os; print(os.path.join(os.path.dirname(triton.__file__), '__init__.py'))")
if ! grep -q "next_power_of_2" "$TRITON_INIT"; then
    echo '' >> "$TRITON_INIT"
    echo 'def next_power_of_2(n):' >> "$TRITON_INIT"
    echo '    if n <= 0:' >> "$TRITON_INIT"
    echo '        return 1' >> "$TRITON_INIT"
    echo '    return 1 << (n - 1).bit_length()' >> "$TRITON_INIT"
fi
```

### 8.4 vllm-ascend 编译问题修复（TILINGKEY_PAR_COMPILE）

```bash
# 问题：opc 并行编译冲突 "Another process is using this dir"
# 修复：限制编译并行度 + 禁用 TILINGKEY_PAR_COMPILE

# 限制编译并行度为1（完全串行）
sed -i 's/JOB_NUM="-j${CPU_NUM}"/JOB_NUM="-j1"/' /home/vllm-ascend-official/csrc/build.sh

# 禁用 TILINGKEY_PAR_COMPILE
find /home/vllm-ascend-official/csrc -maxdepth 2 -name 'CMakeLists.txt' -o -name '*.cmake' | \
    xargs grep -l 'TILINGKEY_PAR_COMPILE' | while read f; do
    sed -i 's/TILINGKEY_PAR_COMPILE=1/TILINGKEY_PAR_COMPILE=0/g' "$f"
done

# 清理 build 目录后重新安装
rm -rf /home/vllm-ascend-official/csrc/build
cd /home/vllm-ascend-official
pip install -v -e .
```

## 步骤九：拉起 vllm 实例

### 9.1 创建 run.sh 脚本

```bash
cat > /home/lsy/run.sh << 'EOF'
#!/bin/bash
rm -rf /root/ascend/log/*

echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern

export PYTHONPATH=$PYTHONPATH:/home/vllm
export PYTHONPATH=$PYTHONPATH:/home/vllm-ascend-official

nic_name="enp189s0f0"
local_ip="172.28.1.10"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export HCCL_NPU_SOCKET_PORT_RANGE=20000-25000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export VLLM_VERSION=0.20.2

source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

export MMC_LOCAL_CONFIG_PATH=/home/lsy/layerwise/mmc-local.conf

vllm serve /mnt/sfs_turbo/weights/Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 8 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name qwen3 \
    --enable-expert-parallel \
    --max-num-seqs 100 \
    --max-model-len 70000 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.25 \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false}}' > /home/lsy/log_p.log 2>&1
EOF
chmod +x /home/lsy/run.sh
```

### 9.2 执行 run.sh

```bash
cd /home/lsy
bash run.sh

# 查看日志
tail -f log_p.log
```

### 9.3 带 memcache KV transfer 的 run.sh（可选）

如果需要使用 memcache 作为 KV transfer backend：

```bash
vllm serve /mnt/sfs_turbo/weights/Qwen3.5-35B-A3B \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 8 \
    --enforce-eager \
    --seed 1024 \
    --served-model-name qwen3 \
    --enable-expert-parallel \
    --max-num-seqs 100 \
    --max-model-len 70000 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --gpu-memory-utilization 0.25 \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": false,"mooncake_rpc_port":"0"}}'
```

### 9.4 验证服务接口

```bash
# 新窗口执行
unset http_proxy
unset https_proxy
curl -X POST -s http://127.0.0.1:8004/v1/completions -H "Content-Type: application/json" -d '{
"model": "qwen3",
"prompt": "The sun dipped below the horizon painting the sky in hues of orange and purple.",
"max_tokens": 200,
"temperature": 0
}'
```

### 9.5 常见启动问题排查

**问题1：HCCL 端口冲突**
```
Communication_Error_Bind_IP_Port(EI0020): Failed to enable listening for the NPU network adapter socket.
```
解决：添加环境变量 `export HCCL_NPU_SOCKET_PORT_RANGE=20000-25000`

**问题2：NPU 显存不足**
```
ValueError: Free memory on device (27.62/60.96 GiB) on startup is less than desired GPU memory utilization (0.9, 54.86 GiB).
```
解决：降低 `--gpu-memory-utilization`（如 0.25），或释放被其他进程占用的 NPU 显存

**问题3：Hybrid KV cache manager 冲突**
```
ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.
```
解决：添加 `--no-disable-hybrid-kv-cache-manager` 参数（使用 kv-transfer-config 时需要）

**问题4：triton next_power_of_2 缺失**
```
RuntimeError: Worker failed with error 'module 'triton' has no attribute 'next_power_of_2''
```
解决：参考步骤 8.3 手动添加 `next_power_of_2` 函数

**问题5：DP=2 + TP=8 设备分配失败**
```
set default device id failed, ret:107001
```
解决：DP=2 + TP=8 总共需要 16 个 Worker，但只有 8 张 NPU。改为 DP=1 + TP=8

## 环境信息

- 服务器：121.37.53.41 (SSH端口 10001)
- NPU：910B3 x 8（注意：NPU 1 和 2 被其他进程占用约 37GB 显存）
- 容器：tyy_b120
- 基础镜像：quay.io/ascend/vllm-ascend:v0.20.2rc1
- Python：3.11.15
- vllm：0.20.2+empty (源码可编辑安装, /home/vllm)
- vllm-ascend：0.1.dev1+gface36594 (官方 releases/v0.20.2rc 分支, /home/vllm-ascend-official)
- memfabric_hybrid：1.0.8
- memcache_hybrid：1.0.4
- pybind11：3.0.1
- triton-ascend：3.2.1 (triton 3.2.0 + next_power_of_2 patch)
- 模型：Qwen3.5-35B-A3B (/mnt/sfs_turbo/weights/Qwen3.5-35B-A3B, bfloat16, 66.97 GiB)
- MetaService：运行中 (端口 5000/6000/8000)

### 模型加载状态

- ✅ vllm v0.20.2 + vllm-ascend v0.20.2rc 安装成功
- ✅ MetaService 后端服务运行中
- ✅ 模型权重加载成功（14/14 shards, 237秒, 每卡 8.36GB）
- ✅ HCCL 分布式通信初始化成功（8 Worker, world_size=8）
- ⚠️ KV cache profiling 阶段 Worker 崩溃（显存不足，需释放 NPU 1/2 上的其他进程或降低 max-model-len）
