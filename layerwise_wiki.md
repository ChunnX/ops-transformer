# vLLM-Ascend Layerwise PD 混部性能对比验证

## 1. 环境版本信息

| 组件 | 版本 |
|------|------|
| NPU | 910B3 x 8（A2 平台，每卡 64GB HBM）|
| NPU 驱动 | 25.5.1 |
| OS | openEuler 2403sp3 aarch64 |
| 容器 | Docker，镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1` |
| Python | 3.11.15（容器内 `/usr/local/python3.11.15/bin/python3.11`）|
| CANN | 9.0.0 |
| vllm | 0.20.2+empty（源码可编辑安装, git tag v0.20.2, /home/vllm）|
| vllm-ascend | 0.1.dev1+gface36594（官方 releases/v0.20.2rc 分支, /home/vllm-ascend-official）|
| memfabric_hybrid | 1.2.0（pip 安装）|
| memcache_hybrid | 1.2.0（pip 安装）|
| triton-ascend | 3.2.1（triton 3.2.0 + next_power_of_2 patch）|
| transformers | 5.5.3 |
| ais-bench-benchmark | 3.1.20260813（源码安装, /home/benchmark）|
| libhcom.so | 从 ubs-comm 源码编译（/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhcom.so）|
| libboundscheck.so | dnf 安装（/usr/lib64/libboundscheck.so）|
| 模型 | DeepSeek-V4-Flash-w8a8-mtp（280GB, W8A8, 70 shards）|

## 2. 环境搭建步骤

### 2.1 创建容器

```bash
docker run --name tyy_b120 -it -d --net=host --shm-size=500g \
    --privileged=true -w /home \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /mnt:/mnt -v /data:/data -v /dev:/dev \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home -v /etc/hccn.conf:/etc/hccn.conf -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy= -e https_proxy= \
    quay.io/ascend/vllm-ascend:v0.20.2rc1
```

### 2.2 配置 pip 镜像源

```bash
docker exec -it tyy_b120 bash
git config --global http.sslVerify false
/usr/local/python3.11.15/bin/python3.11 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
/usr/local/python3.11.15/bin/python3.11 -m pip config set global.trusted-host mirrors.aliyun.com
/usr/local/python3.11.15/bin/python3.11 -m pip config set global.extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

### 2.3 卸载自带 vllm 和 vllm-ascend

```bash
/usr/local/python3.11.15/bin/python3.11 -m pip uninstall vllm vllm_ascend -y
```

### 2.4 克隆代码仓库

```bash
cd /home
git clone https://github.com/vllm-project/vllm.git
git clone -b releases/v0.20.2rc --depth 1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-official
# GitHub 直连超时用代理: git clone https://gh-proxy.org/https://github.com/vllm-project/vllm.git
```

### 2.5 切换 vllm 到 v0.20.2 并安装

```bash
cd /home/vllm
git fetch --depth 1 origin tag v0.20.2
git checkout v0.20.2
VLLM_TARGET_DEVICE=empty /usr/local/python3.11.15/bin/python3.11 -m pip install -v -e .
```

### 2.6 triton 兼容性修复

```bash
/usr/local/python3.11.15/bin/python3.11 -m pip uninstall triton triton-ascend -y
/usr/local/python3.11.15/bin/python3.11 -m pip install triton-ascend==3.2.1

# Patch next_power_of_2
TRITON_INIT=$(/usr/local/python3.11.15/bin/python3.11 -c "import triton; import os; print(os.path.join(os.path.dirname(triton.__file__), '__init__.py'))")
if ! grep -q "next_power_of_2" "$TRITON_INIT"; then
    printf '\ndef next_power_of_2(n):\n    if n <= 0:\n        return 1\n    return 1 << (n - 1).bit_length()\n' >> "$TRITON_INIT"
fi
```

### 2.7 vllm-ascend 编译修复并安装

```bash
# 限制编译并行度为1
sed -i 's/JOB_NUM="-j${CPU_NUM}"/JOB_NUM="-j1"/' /home/vllm-ascend-official/csrc/build.sh
# 禁用 TILINGKEY_PAR_COMPILE
find /home/vllm-ascend-official/csrc -maxdepth 2 -name 'CMakeLists.txt' -o -name '*.cmake' | \
    xargs grep -l 'TILINGKEY_PAR_COMPILE' | while read f; do
    sed -i 's/TILINGKEY_PAR_COMPILE=1/TILINGKEY_PAR_COMPILE=0/g' "$f"
done
sed -i 's/export TILINGKEY_PAR_COMPILE=1/export TILINGKEY_PAR_COMPILE=0/' /home/vllm-ascend-official/csrc/cmake/func.cmake

cd /home/vllm-ascend-official
rm -rf csrc/build
/usr/local/python3.11.15/bin/python3.11 -m pip install -v -e .
```

### 2.8 安装 memcache/memfabric 1.2.0（pip 包，不需要源码编译）

```bash
unset http_proxy https_proxy
/usr/local/python3.11.15/bin/python3.11 -m pip install memfabric_hybrid==1.2.0 --force-reinstall
/usr/local/python3.11.15/bin/python3.11 -m pip install memcache_hybrid==1.2.0 --force-reinstall
```

### 2.9 安装 libhcom.so（从 ubs-comm 源码编译）

```bash
cd /home
git clone --depth 1 -b br_dev_container_20260228 https://atomgit.com/openeuler/ubs-comm.git /home/l30058153/ubs-comm
cd /home/l30058153/ubs-comm
git submodule update --init --recursive
bash build.sh
# 安装
cp dist/hcom/lib/libhcom.so /usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhcom.so
```

### 2.10 安装 libboundscheck.so

```bash
# 在宿主机执行（容器内没有 dnf）
dnf install -y libboundscheck
docker exec tyy_b120 mkdir -p /usr/lib64
docker cp /usr/lib64/libboundscheck.so tyy_b120:/usr/lib64/libboundscheck.so
docker exec tyy_b120 ldconfig
```

### 2.11 配置 memcache（host_shm 协议）

```bash
mkdir -p /home/<your_name>/layerwise
cp /usr/local/python3.11.15/lib/python3.11/site-packages/memcache_hybrid/config/mmc-meta.conf /home/<your_name>/layerwise/
cp /usr/local/python3.11.15/lib/python3.11/site-packages/memcache_hybrid/config/mmc-local.conf /home/<your_name>/layerwise/

# A2 平台无 device RDMA 网卡，用 host_shm
sed -i 's/ock.mmc.local_service.protocol = host_rdma/ock.mmc.local_service.protocol = host_shm/' /home/<your_name>/layerwise/mmc-local.conf
sed -i 's/ock.mmc.local_service.dram.size = 1GB/ock.mmc.local_service.dram.size = 10GB/' /home/<your_name>/layerwise/mmc-local.conf
```

### 2.12 安装 aisbench

```bash
cd /home
git clone --depth 1 https://github.com/AISBench/benchmark.git /home/benchmark
cd /home/benchmark
/usr/local/python3.11.15/bin/python3.11 -m pip install -e .
/usr/local/python3.11.15/bin/python3.11 -m pip install -r requirements/api.txt
/usr/local/python3.11.15/bin/python3.11 -m pip install -r requirements/extra.txt

# aisbench 测试脚本
git clone --depth 1 https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git /home/aisbench_auto_tools_prefix
```

### 2.13 配置 aisbench config.py

```python
DATASET_PATH = "/home/<your_name>/dataset"
WORK_PATH = "/home/benchmark"
MODEL_NAME = "deepseek_v4"
MODEL_PATH = "/mnt/sfs_turbo/weights/DeepSeek-V4-Flash-w8a8-mtp"
HOST_IP = "127.0.0.1"
HOST_PORT = "8004"
API_KEY = ""
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR = "./outputs/default"
POD_INFO = []
```

## 3. 启动 MetaService

创建 `start_meta.sh`：

```bash
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/memcache_hybrid/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/memfabric_hybrid/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/lib64:$LD_LIBRARY_PATH
export MMC_META_CONFIG_PATH=/home/<your_name>/layerwise/mmc-meta.conf
nohup /usr/local/python3.11.15/bin/python3.11 -c "from memcache_hybrid import MetaService; MetaService.main()" > /home/<your_name>/mmc_meta_service.log 2>&1 &
echo $! > /home/<your_name>/mmc_meta_pid.txt
sleep 5
ps -p $(cat /home/<your_name>/mmc_meta_pid.txt) -o pid,stat,etime,args
echo "METASERVICE_STARTED"
```

```bash
chmod +x start_meta.sh
bash start_meta.sh
# 验证: netstat -tlnp | grep 5000
```

## 4. 启动 vllm serve

### 4.1 开 layerwise 的 run.sh（use_layerwise: true）

```bash
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
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_OP_BASE_ATTR=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/memcache_hybrid/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/memfabric_hybrid/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:/usr/lib64:$LD_LIBRARY_PATH

export VLLM_VERSION=0.20.2
export MMC_LOCAL_CONFIG_PATH=/home/<your_name>/layerwise/mmc-local.conf

vllm serve /mnt/sfs_turbo/weights/DeepSeek-V4-Flash-w8a8-mtp \
    --host 0.0.0.0 --port 8004 --tensor-parallel-size 8 --seed 1024 \
    --served-model-name deepseek_v4 --enable-expert-parallel \
    --max-num-seqs 40 --max-model-len 20000 --max-num-batched-tokens 4096 \
    --trust-remote-code --no-enable-prefix-caching \
    --gpu-memory-utilization 0.90 --quantization ascend \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector","kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' > /home/<your_name>/log_p.log 2>&1
```

### 4.2 不开 layerwise 的 run.sh（use_layerwise: false）

与上面完全相同，唯一区别：`"use_layerwise": false`

### 4.3 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--max-model-len 20000` | 20000 | 受 KV cache 限制（host_shm 下约 18GB 可用）|
| `--gpu-memory-utilization 0.90` | 0.90 | 模型 36GB/卡 + KV cache ~18GB/卡 |
| `--quantization ascend` | ascend | W8A8 模型必须用 |
| `--torchair_graph_config {"enabled":true}` | true | PIECEWISE cudagraph（layerwise 必需）|
| `kv_role: kv_both` | kv_both | PD 混部 |
| `use_layerwise: true/false` | - | 对比变量 |
| `backend: memcache` | memcache | KV transfer backend |
| `protocol: host_shm` | host_shm | A2 无 device RDMA 网卡 |

### 4.4 启动与验证

```bash
# 确保 8 卡空闲
npu-smi info | grep HBM-Usage  # 每卡应 < 5000 MB

# 启动 MetaService
bash start_meta.sh

# 启动 vllm serve
nohup bash run.sh > /dev/null 2>&1 &
tail -f log_p.log

# 成功标志:
# "Available KV cache memory: ~18 GiB"
# "GPU KV cache size: ~29000 tokens"
# "torch.compile took ~60s"
# "Application startup complete."

# 验证推理
curl -s -X POST http://127.0.0.1:8004/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek_v4", "prompt": "Hello", "max_tokens": 10, "temperature": 0}'
```

## 5. 跑 aisbench 性能测试

```bash
cd /home/aisbench_auto_tools_prefix
/usr/local/python3.11.15/bin/python3.11 aisbench_test.py \
    --input_len 2048 --output_len 2048 --data_num 64 \
    --concurrency 40 --request_rate 0
```

## 6. 性能对比结果

测试条件：input_len=2048, output_len=2048, data_num=64, concurrency=40, TP=8, host_shm 协议

| 指标 | 开 layerwise | 不开 layerwise |
|------|-------------|---------------|
| 成功请求 | 64 | 64 |
| 平均输出 tokens | 62.3 | 2048 |
| E2E 耗时 | 142.3s | 892.9s |
| Output 吞吐 | 17.52 token/s | 146.80 token/s |
| E2E 吞吐 | 593.12 token/s | 293.88 token/s |
| QPS | 0.28 req/s | 0.07 req/s |
| TTFT avg | 102859ms | 212631ms |
| TPOT avg | 346.1ms | 94.7ms |

### 结果文件路径

```
/home/aisbench_auto_tools_prefix/aisbench_result.csv  # CSV 结果（含多次测试）
/home/aisbench_auto_tools_prefix/aisbench.log          # 完整日志
```

## 7. 代码 patch 说明

### 7.1 vllm-ascend patch: KVCacheStoreLayerSendingThread.stored_requests

文件: `/home/vllm-ascend-official/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

`KVCacheStoreLayerSendingThread` 缺少 `stored_requests` 属性，推理时报 `AttributeError`。在 `__init__` 中添加：

```python
# 在 KVCacheStoreLayerSendingThread.__init__ 末尾添加:
self.stored_requests = defaultdict[str, int](int)
```

同时在文件头部添加：
```python
from collections import defaultdict
```

### 7.2 triton next_power_of_2 patch

triton 3.2.0 缺少 `next_power_of_2` 函数，手动添加到 `__init__.py`。

## 8. 常见问题

### 8.1 NPU device RDMA 网卡 DOWN

```bash
for i in $(seq 0 7); do echo "==> $i"; hccn_tool -i $i -link -g; done
# 如果显示 DOWN，则不能用 device_rdma/device_sdma 协议，只能用 host_shm
```

### 8.2 store.init() 失败 (assert res == 0)

- 确认 MetaService 正在运行（`netstat -tlnp | grep 5000`）
- 确认 `MMC_LOCAL_CONFIG_PATH` 指向正确配置
- 确认 `protocol = host_shm`
- 确认 `LD_LIBRARY_PATH` 包含 memcache_hybrid/lib 和 memfabric_hybrid/lib

### 8.3 KV cache 空间不足

降低 `--max-model-len`（host_shm 下最大约 20000）

### 8.4 推理崩溃 (stored_requests AttributeError)

参考 7.1 patch

### 8.5 Bind cpus failed

非致命错误，可忽略

## 9. 版本兼容性

| vllm 版本 | vllm-ascend 版本 | 兼容性 |
|-----------|------------------|--------|
| v0.19.0 | fork main | ❌ 缺少 `_approximate_gcd` |
| v0.20.2 | fork main | ❌ `FusedMoE` 改名 |
| **v0.20.2** | **官方 v0.20.2rc** | **✅** |

| memcache 版本 | memfabric 版本 | host_shm 推理 |
|--------------|----------------|--------------|
| 1.0.4 | 1.0.8 | ❌ entity != nullptr 崩溃 |
| **1.2.0** | **1.2.0** | **✅** |
