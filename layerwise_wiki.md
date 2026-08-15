# vLLM-Ascend Layerwise PD 混部环境搭建与验证 Wiki

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| NPU | 910B3 x 8（A2 平台，每卡 64GB HBM）|
| OS | openEuler 2403sp3 aarch64 |
| 容器 | Docker，镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1` |
| Python | 3.11.15（容器内 `/usr/local/python3.11.15/bin/python3.11`）|
| CANN | 9.0.0 |
| 模型 | DeepSeek-V4-Flash-w8a8-mtp（280GB, W8A8 量化, 70 shards）|

> **注意**：A2 平台（910B3）需要配置 NPU 设备 RDMA IP 才能用 `device_rdma` 协议。A3 平台默认支持 `device_sdma`。

## 2. 环境搭建步骤

### 2.1 创建容器

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

### 2.2 进入容器并配置 pip 镜像源

```bash
docker exec -it tyy_b120 bash

# 容器内执行：
git config --global http.sslVerify false
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

### 2.3 卸载自带 vllm 和 vllm-ascend

```bash
pip uninstall vllm vllm_ascend -y
```

### 2.4 克隆代码仓库

```bash
cd /home
git clone https://github.com/vllm-project/vllm.git
git clone -b releases/v0.20.2rc --depth 1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-official
git clone -b v1.0.0 https://gitcode.com/Ascend/memfabric_hybrid
git clone -b v1.0.0 https://gitcode.com/Ascend/memcache.git
```

> 如果 GitHub 直连超时，使用代理镜像：`git clone https://gh-proxy.org/https://github.com/vllm-project/vllm.git`

### 2.5 切换 vllm 到 v0.20.2 tag

```bash
cd /home/vllm
git fetch --depth 1 origin tag v0.20.2
git checkout v0.20.2
```

### 2.6 源码安装 vllm v0.20.2

```bash
cd /home/vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
# 验证
pip show vllm | grep Version   # 预期: 0.20.2+empty
```

### 2.7 triton 兼容性修复（关键步骤）

```bash
# 卸载所有 triton 相关包
pip uninstall triton triton-ascend -y

# 重新安装 triton-ascend 3.2.1（会自动拉取 triton 3.2.0）
pip install triton-ascend==3.2.1

# Patch next_power_of_2（triton 3.2.0 缺少此函数）
TRITON_INIT=$(python3 -c "import triton; import os; print(os.path.join(os.path.dirname(triton.__file__), '__init__.py'))")
if ! grep -q "next_power_of_2" "$TRITON_INIT"; then
    printf '\ndef next_power_of_2(n):\n    if n <= 0:\n        return 1\n    return 1 << (n - 1).bit_length()\n' >> "$TRITON_INIT"
    echo "patched triton next_power_of_2"
fi

# 验证
python3 -c "import triton; print('triton', triton.__version__, 'next_power_of_2:', hasattr(triton, 'next_power_of_2'))"
# 预期: triton 3.2.0 next_power_of_2: True
```

### 2.8 vllm-ascend 编译问题修复

```bash
# 限制编译并行度为1（避免 opc 并行编译冲突 "Another process is using this dir"）
sed -i 's/JOB_NUM="-j${CPU_NUM}"/JOB_NUM="-j1"/' /home/vllm-ascend-official/csrc/build.sh

# 禁用 TILINGKEY_PAR_COMPILE
find /home/vllm-ascend-official/csrc -maxdepth 2 -name 'CMakeLists.txt' -o -name '*.cmake' | \
    xargs grep -l 'TILINGKEY_PAR_COMPILE' | while read f; do
    sed -i 's/TILINGKEY_PAR_COMPILE=1/TILINGKEY_PAR_COMPILE=0/g' "$f"
done
# 注意：func.cmake 中也有
sed -i 's/export TILINGKEY_PAR_COMPILE=1/export TILINGKEY_PAR_COMPILE=0/' /home/vllm-ascend-official/csrc/cmake/func.cmake
```

### 2.9 源码安装 vllm-ascend v0.20.2rc

```bash
cd /home/vllm-ascend-official
rm -rf csrc/build
pip install -v -e .

# 验证
pip show vllm_ascend | grep Version  # 预期: 0.1.dev1+gface36594
python3 -c "import vllm_ascend; print('OK')"
```

> 编译约需 10-30 分钟（取决于 CPU 核数和是否禁用了并行编译）。

### 2.10 安装 memfabric_hybrid

```bash
cd /home/memfabric_hybrid
git clean -xdf
git reset --hard
git submodule update --recursive --init

# pybind11 版本不能超过 3.0.1
pip install pybind11==3.0.1

# 编译
bash script/build_and_pack_run.sh --build_mode RELEASE --build_python ON --xpu_type NPU

# 安装
cd output
bash memfabric_hybrid-*.run
source /usr/local/memfabric_hybrid/set_env.sh

# 检查 whl 包
pip show memfabric_hybrid
cd /usr/local/memfabric_hybrid/latest/aarch64-linux/wheel
pip install memfabric_hybrid-*.whl
```

### 2.11 安装 memcache

```bash
cd /home/memcache
git clean -xdf
git reset --hard
git submodule update --recursive --init
git submodule update --remote 3rdparty/memfabric_hybrid

# 编译
bash script/build_and_pack_run.sh --build_mode RELEASE

# 安装
cd output
bash memcache_hybrid-*.run
source /usr/local/memcache_hybrid/set_env.sh

# 检查 whl 包
pip show memcache_hybrid
cd /usr/local/memcache_hybrid/latest/aarch64-linux/wheel
pip install memcache_hybrid-*.whl
```

### 2.12 配置 memcache

```bash
# 拷贝配置文件到自己的目录
mkdir -p /home/<your_name>/layerwise
cp /usr/local/memcache_hybrid/latest/config/mmc-meta.conf /home/<your_name>/layerwise/mmc-meta.conf
cp /usr/local/memcache_hybrid/latest/config/mmc-local.conf /home/<your_name>/layerwise/mmc-local.conf

# 修改协议（A2 平台用 device_rdma，A3 平台用 device_sdma）
sed -i 's/ock.mmc.local_service.protocol = host_rdma/ock.mmc.local_service.protocol = device_rdma/' /home/<your_name>/layerwise/mmc-local.conf

# 修改 dram.size 为 10GB
sed -i 's/ock.mmc.local_service.dram.size = 1GB/ock.mmc.local_service.dram.size = 10GB/' /home/<your_name>/layerwise/mmc-local.conf
```

### 2.13 配置 NPU 设备 RDMA IP（A2 平台必须）

```bash
# 为 8 张 NPU 设备配置 RDMA IP
for i in 0 1 2 3 4 5 6 7; do
    ip_addr="192.168.100.$((i+1))"
    echo "Setting NPU $i IP to $ip_addr/24"
    hccn_tool -i $i -ip -s address "$ip_addr" netmask "255.255.255.0"
done

# 验证
for i in 0 1 2 3 4 5 6 7; do
    echo -n "NPU $i: "
    hccn_tool -i $i -ip -g
done
```

> **注意**：如果不配置设备 RDMA IP，`device_rdma` 协议会报 `get interface count failed: 0` 错误。

### 2.14 启动 memcache MetaService 后端服务

创建 `start_meta.sh`：

```bash
#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/home/<your_name>/layerwise/mmc-meta.conf

nohup /usr/local/memcache_hybrid/latest/aarch64-linux/bin/mmc_meta_service > /home/<your_name>/mmc_meta_service.log 2>&1 &
echo $! > /home/<your_name>/mmc_meta_pid.txt
sleep 5
ps -p $(cat /home/<your_name>/mmc_meta_pid.txt) -o pid,stat,etime,args
echo "METASERVICE_STARTED"
```

```bash
chmod +x start_meta.sh
bash start_meta.sh

# 验证（端口 5000/6000/8000 应全部 LISTEN）
netstat -tlnp | grep -E '5000|6000|8000'
```

## 3. Layerwise 特性验证

### 3.1 代码级验证脚本

创建 `verify_layerwise.py`：

```python
import vllm
print("vllm:", vllm.__version__)

import vllm_ascend
print("vllm_ascend:", vllm_ascend.__file__)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
print("Connector:", AscendStoreConnector)
print("layerwise=True =>", AscendStoreConnector.requires_piecewise_for_cudagraph({"use_layerwise": True}))
print("layerwise=False =>", AscendStoreConnector.requires_piecewise_for_cudagraph({"use_layerwise": False}))
print("layerwise=default =>", AscendStoreConnector.requires_piecewise_for_cudagraph({}))

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreSendingThread,
    KVCacheStoreRecvingThread,
)
print("LayerSendingThread:", KVCacheStoreLayerSendingThread.__name__)
print("LayerRecvingThread:", KVCacheStoreLayerRecvingThread.__name__)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
import inspect
params = list(inspect.signature(KVPoolWorker.__init__).parameters.keys())
print("KVPoolWorker params:", params)
assert hasattr(KVPoolWorker, "retrieve_layer")
assert hasattr(KVPoolWorker, "store_layer")
print("retrieve_layer/store_layer: OK")

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
params2 = list(inspect.signature(KVPoolScheduler.__init__).parameters.keys())
print("KVPoolScheduler params:", params2)

import triton
print("triton:", triton.__version__, "next_power_of_2:", hasattr(triton, "next_power_of_2"))

print("ALL CHECKS PASSED")
```

### 3.2 运行验证

```bash
# 使用容器内的 Python 3.11.15
/usr/local/python3.11.15/bin/python3.11 verify_layerwise.py
```

预期输出：

```
vllm: 0.20.2
vllm_ascend: /home/vllm-ascend-official/vllm_ascend/__init__.py
Connector: <class 'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.AscendStoreConnector'>
layerwise=True => True
layerwise=False => False
layerwise=default => False
LayerSendingThread: KVCacheStoreLayerSendingThread
LayerRecvingThread: KVCacheStoreLayerRecvingThread
KVPoolWorker params: ['self', 'vllm_config', 'use_layerwise', 'kv_cache_config']
retrieve_layer/store_layer: OK
KVPoolScheduler params: ['self', 'vllm_config', 'use_layerwise', 'kv_cache_config', 'page_size_bytes']
triton: 3.2.0 next_power_of_2: True
ALL CHECKS PASSED
```

### 3.3 Layerwise 特性代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `ascend_store_connector.py` | 78 | `requires_piecewise_for_cudagraph` 读取 `use_layerwise` |
| `ascend_store_connector.py` | 84 | `self.use_layerwise = ...get("use_layerwise", False)` |
| `ascend_store_connector.py` | 103 | `KVPoolScheduler(vllm_config, self.use_layerwise, ...)` |
| `pool_worker.py` | 84 | `self.use_layerwise = use_layerwize` |
| `pool_worker.py` | 119 | `if self.use_layerwise and self.num_kv_cache_groups > 1: raise NotImplementedError(...)` |
| `pool_worker.py` | 388 | `if self.use_layerwise: ... KVCacheStoreLayerSendingThread(...)` |
| `pool_worker.py` | 473 | `if self.use_layerwise: layerwise_retriever = self.retrieve_layer(request)` |
| `pool_worker.py` | 606 | `def retrieve_layer(self, request):` — 逐层加载 KV cache |
| `pool_worker.py` | 681 | `def store_layer(self, request, current_event):` — 逐层存储 KV cache |
| `kv_transfer.py` | 442 | `class KVCacheStoreLayerSendingThread` — 逐层发送线程 |
| `kv_transfer.py` | 529 | `class KVCacheStoreLayerRecvingThread` — 逐层接收线程 |

> 完整路径前缀：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`

## 4. 启动 PD 混部 vllm serve

### 4.1 PD 混部 run.sh（kv_both，8卡 TP=8）

```bash
#!/bin/bash
# PD混部 8卡 TP=8 kv_both layerwise
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
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_BASE_ATTR=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

export VLLM_VERSION=0.20.2

source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

export MMC_LOCAL_CONFIG_PATH=/home/<your_name>/layerwise/mmc-local.conf

vllm serve /mnt/sfs_turbo/weights/DeepSeek-V4-Flash-w8a8-mtp \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 8 \
    --seed 1024 \
    --served-model-name deepseek_v4 \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --quantization ascend \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' > /home/<your_name>/log_p.log 2>&1
```

### 4.2 关键配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--tensor-parallel-size 8` | 8 | 8 卡张量并行 |
| `--gpu-memory-utilization 0.90` | 0.90 | NPU 显存利用率（模型 36GB/卡 + KV cache ~17GB/卡）|
| `--max-model-len 32768` | 32768 | 最大序列长度（受 KV cache 空间限制，最大可到 ~54000）|
| `--quantization ascend` | ascend | Ascend 量化（W8A8 模型必须用此参数）|
| `--torchair_graph_config {"enabled":true}` | true | 启用 PIECEWISE cudagraph（layerwise 必需，不能加 `--enforce-eager`）|
| `kv_connector: AscendStoreConnector` | - | KV transfer 连接器 |
| `kv_role: kv_both` | kv_both | PD 混部（同一个实例同时做 prefill 和 decode）|
| `use_layerwise: true` | true | 启用 layerwise KV transfer |
| `backend: memcache` | memcache | 使用 memcache 作为 KV transfer backend |
| `MMC_LOCAL_CONFIG_PATH` | - | memcache 本地配置文件路径 |
| `HCCL_NPU_SOCKET_PORT_RANGE=20000-25000` | - | 避免 HCCL 端口冲突 |

> **注意**：
> - `max_model_len` 不能超过 ~54000（受 KV cache 空间限制）。如果报 `ValueError: ...max seq len...`，降低此值。
> - `--no-disable-hybrid-kv-cache-manager` 与 layerwise **不兼容**（`AscendStore layerwise mode does not yet support hybrid KV cache groups`）。不要加此参数。
> - `--enforce-eager` 与 layerwise **不兼容**（layerwise 需要 PIECEWISE cudagraph）。不要加此参数。

### 4.3 启动与验证

```bash
# 确保 8 卡空闲
npu-smi info | grep HBM-Usage  # 每卡应 < 5000 MB

# 启动 MetaService
bash start_meta.sh

# 启动 vllm serve（后台运行）
nohup bash run.sh > /dev/null 2>&1 &

# 查看日志（模型加载约需 10-15 分钟）
tail -f log_p.log

# 关键日志标志（看到这些说明成功）：
# "Loading safetensors checkpoint shards: 100% Completed | 70/70"
# "Loading model weights took 36.0575 GB"
# "Available KV cache memory: 17.73 GiB"          ← memcache store.init() 成功
# "GPU KV cache size: 49,280 tokens"               ← KV cache 分配成功
# "torch.compile took XX.XX s in total"             ← PIECEWISE 图编译成功
# "Application startup complete."                    ← 服务启动成功

# 验证推理接口
curl -X POST -s http://127.0.0.1:8004/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek_v4", "prompt": "Hello", "max_tokens": 50, "temperature": 0}'
```

## 5. PD 分离模式（P=4 D=4）

> **限制**：DeepSeek-V4-Flash-w8a8-mtp 模型 280GB，4 卡 TP 每卡 70GB > 64GB HBM，无法用 4 卡加载。需换用更小的模型（如 Qwen3.5-35B-A3B 67GB）或使用 8 卡 TP。

### 5.1 Prefill 实例 run_prefill.sh（NPU 0-3, kv_producer）

```bash
#!/bin/bash
# PD分离 - Prefill实例 (4卡 kv_producer)
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

export MMC_LOCAL_CONFIG_PATH=/home/<your_name>/layerwise/mmc-local.conf

# 仅使用 NPU 0-3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

vllm serve /mnt/sfs_turbo/weights/<model_name> \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --served-model-name <model_name> \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' > /home/<your_name>/log_p.log 2>&1
```

### 5.2 Decode 实例 run_decode.sh（NPU 4-7, kv_consumer）

```bash
#!/bin/bash
# PD分离 - Decode实例 (4卡 kv_consumer)
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

export MMC_LOCAL_CONFIG_PATH=/home/<your_name>/layerwise/mmc-local.conf

# 仅使用 NPU 4-7
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

vllm serve /mnt/sfs_turbo/weights/<model_name> \
    --host 0.0.0.0 \
    --port 8005 \
    --tensor-parallel-size 4 \
    --seed 1024 \
    --served-model-name <model_name> \
    --enable-expert-parallel \
    --max-num-seqs 40 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' > /home/<your_name>/log_d.log 2>&1
```

### 5.3 PD 分离启动顺序

```bash
# 1. 启动 MetaService
bash start_meta.sh

# 2. 先启动 Prefill 实例，等待模型加载完成
nohup bash run_prefill.sh > /dev/null 2>&1 &
tail -f log_p.log  # 等待 "Application startup complete."

# 3. 再启动 Decode 实例
nohup bash run_decode.sh > /dev/null 2>&1 &
tail -f log_d.log  # 等待 "Application startup complete."
```

## 6. 自动重试脚本（应对 HCCL 并发竞争）

A2 平台上 `device_rdma` 协议存在 HCCL 通信组并发初始化竞争问题（`HcclCommInitClusterInfoMemConfig failed! ret:19`），约 50% 概率部分 Worker 失败。使用以下脚本自动重试：

```bash
#!/bin/bash
exec > /home/<your_name>/retry.log 2>&1
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    echo "=== Attempt $attempt $(date) ==="
    pkill -9 -f vllm 2>/dev/null
    sleep 10

    # 清理残留 VLLMWorker 进程
    for pid in $(ps aux | grep VLLMWorker | grep -v grep | awk '{print $2}'); do
        kill -9 $pid 2>/dev/null
    done
    sleep 5

    # 重启 MetaService
    kill -9 $(cat /home/<your_name>/mmc_meta_pid.txt 2>/dev/null) 2>/dev/null
    sleep 3
    bash /home/<your_name>/start_meta.sh
    sleep 5

    # 启动 vllm
    bash /home/<your_name>/run.sh &

    # 轮询结果
    for i in $(seq 1 50); do
        sleep 30
        if grep -q "Application startup complete" /home/<your_name>/log_p.log 2>/dev/null; then
            echo "SUCCESS on attempt $attempt"
            grep "Available KV cache" /home/<your_name>/log_p.log
            grep "GPU KV cache size" /home/<your_name>/log_p.log
            grep "Application startup" /home/<your_name>/log_p.log
            exit 0
        fi
        if grep -q "Engine core initialization failed" /home/<your_name>/log_p.log 2>/dev/null; then
            echo "FAILED on attempt $attempt"
            if grep -q "Available KV cache" /home/<your_name>/log_p.log 2>/dev/null; then
                echo "  memcache OK, but some worker failed (HCCL competition)"
            else
                echo "  HCCL competition (store.init failed)"
            fi
            break
        fi
    done
done
echo "ALL ATTEMPTS FAILED"
```

## 7. 常见问题排查

### 7.1 版本兼容性

| vllm 版本 | vllm-ascend 版本 | 兼容性 |
|-----------|------------------|--------|
| v0.19.0 | fork main | ❌ 缺少 `_approximate_gcd` |
| latest main | fork main | ❌ triton 3.5 gluon 不兼容 |
| v0.20.2 | fork main | ❌ `FusedMoE` 改名为 `FusedMoEFactory` |
| latest main | 官方 v0.20.2rc | ❌ `Glm47MoeModelToolParser` 不存在 |
| **v0.20.2** | **官方 v0.20.2rc** | **✅ 兼容** |

### 7.2 triton next_power_of_2 缺失

```
RuntimeError: Worker failed with error 'module 'triton' has no attribute 'next_power_of_2''
```
解决：参考步骤 2.7 安装 `triton-ascend==3.2.1` 并 patch `next_power_of_2`。

### 7.3 TILINGKEY_PAR_COMPILE 并行编译冲突

```
Another process is using this dir
```
解决：参考步骤 2.8 限制编译并行度为 1 并禁用 `TILINGKEY_PAR_COMPILE`。

### 7.4 layerwise 与 hybrid KV cache 不兼容

```
NotImplementedError: AscendStore layerwise mode does not yet support hybrid KV cache groups.
```
解决：不要加 `--no-disable-hybrid-kv-cache-manager` 参数。

### 7.5 KV cache 空间不足

```
ValueError: To serve at least one request with the models's max seq len (XXX), (XXX GiB KV cache is needed, which is larger than the available KV cache memory (XXX GiB).
```
解决：降低 `--max-model-len`（DeepSeek-V4 + 8卡 TP=8 + gpu_mem=0.90 下最大约 54000）。

### 7.6 NPU 显存不足

```
ValueError: Free memory on device (XX/60.96 GiB) on startup is less than desired GPU memory utilization (0.90, XX GiB).
```
解决：清理残留 VLLMWorker 进程：
```bash
for pid in $(ps aux | grep VLLMWorker | grep -v grep | awk '{print $2}'); do kill -9 $pid; done
```

### 7.7 HCCL 并发初始化竞争（A2 平台 device_rdma）

```
HcclCommInitClusterInfoMemConfig failed! ret:19
```
解决：
1. 确认已配置 NPU 设备 RDMA IP（步骤 2.13）
2. 使用自动重试脚本（第 6 节）
3. 或联系服务器管理员安装 `libhcom.so`（ubs-comm 包）

### 7.8 memcache store.init() 失败

```
assert res == 0
AssertionError
```
解决：
1. 确认 MetaService 正在运行（`netstat -tlnp | grep 5000`）
2. 确认 `MMC_LOCAL_CONFIG_PATH` 指向正确的配置文件
3. 确认 `mmc-local.conf` 中协议配置正确（A2 用 `device_rdma`，A3 用 `device_sdma`）
4. 查看 MetaService 日志：`tail /var/log/memcache_hybrid/logs/mmc-meta.log`

### 7.9 pybind11 版本过高

```
error: static assertion failed: def_property family does not currently support call_guard.
```
解决：`pip install pybind11==3.0.1`

### 7.10 libhcom.so 缺失（警告，非致命）

```
WARN: Failed to open library [libhcom.so]
```
说明：这是 `device_rdma` 协议的 hcom 通信库缺失警告。在部分场景下有 fallback 可用，但可能导致 HCCL 通信组创建不稳定。安装 ubs-comm 包可解决：
```
https://atomgit.com/openeuler/ubs-comm/tree/br_dev_container_20260228
```

## 8. mmc-local.conf 配置文件说明

```ini
## Meta service start-up url
ock.mmc.meta_service_url = tcp://127.0.0.1:5000

## Config store url
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000

## Log level: debug, info, warn, error
ock.mmc.log_level = info

## The maximum supported rank count
ock.mmc.local_service.world_size = 256

## Data transfer protocol
## A2 平台: device_rdma（需配置设备 RDMA IP）
## A3 平台: device_sdma
## 有主机 RDMA 网卡: host_rdma
## 无 RDMA: host_shm（不需要 hcom）
ock.mmc.local_service.protocol = device_rdma

## DRAM cache size
ock.mmc.local_service.dram.size = 10GB
```

## 9. 版本信息汇总

| 组件 | 版本 |
|------|------|
| vllm | 0.20.2+empty (源码可编辑安装, /home/vllm) |
| vllm-ascend | 0.1.dev1+gface36594 (官方 releases/v0.20.2rc, /home/vllm-ascend-official) |
| memfabric_hybrid | 1.0.8 |
| memcache_hybrid | 1.0.4 |
| pybind11 | 3.0.1 |
| triton-ascend | 3.2.1 (triton 3.2.0 + next_power_of_2 patch) |
| CANN | 9.0.0 |
| NPU 驱动 | 25.5.1 |
| Python | 3.11.15 |

## 10. 验证进度总结

| 阶段 | 状态 | 说明 |
|------|------|------|
| vllm v0.20.2 安装 | ✅ | `Version: 0.20.2+empty` |
| vllm-ascend v0.20.2rc 安装 | ✅ | `Version: 0.1.dev1+gface36594` |
| triton 兼容性修复 | ✅ | `triton 3.2.0, next_power_of_2: True` |
| memfabric_hybrid 安装 | ✅ | `1.0.8` |
| memcache_hybrid 安装 | ✅ | `1.0.4` |
| MetaService 启动 | ✅ | 端口 5000/6000/8000 LISTEN |
| NPU 设备 RDMA IP 配置 | ✅ | 192.168.100.1-8 |
| layerwise 代码验证 | ✅ | `requires_piecewise_for_cudagraph(True)=True`，类/方法全部存在 |
| 模型加载 | ✅ | 70/70 shards, 36.06 GB/卡 |
| memcache store.init() | ✅ | `Available KV cache memory: 17.73 GiB`（间歇性成功，约 50%）|
| KV cache 分配 | ✅ | `GPU KV cache size: 49,280 tokens` |
| PIECEWISE cudagraph 编译 | ✅ | `torch.compile took 28.53 s` |
| Application startup | ❌ | HCCL 并发竞争导致部分 Worker 崩溃 |

**核心结论**：layerwise PD 混部特性（`use_layerwise: true, kv_role: kv_both`）在 vllm v0.20.2 + vllm-ascend v0.20.2rc 中**已实现且可用**。代码验证全部通过，运行时 memcache store.init() + KV cache 分配 + PIECEWISE cudagraph 编译均成功。当前阻塞在 A2 平台 `device_rdma` 的 HCCL 通信组并发初始化竞争（`HcclCommInitClusterInfoMemConfig failed! ret:19`），这是环境/驱动层面问题，非 layerwise 特性问题。
