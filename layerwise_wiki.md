# vLLM-Ascend Layerwise PD 混部环境搭建与验证

## 1. 创建容器

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

## 2. 配置 pip 镜像源

```bash
docker exec -it tyy_b120 bash

git config --global http.sslVerify false
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

## 3. 卸载自带 vllm 和 vllm-ascend

```bash
pip uninstall vllm vllm_ascend -y
```

## 4. 克隆代码仓库

```bash
cd /home
git clone https://github.com/vllm-project/vllm.git
git clone -b releases/v0.20.2rc --depth 1 https://github.com/vllm-project/vllm-ascend.git vllm-ascend-official
git clone -b v1.0.0 https://gitcode.com/Ascend/memfabric_hybrid
git clone -b v1.0.0 https://gitcode.com/Ascend/memcache.git
# GitHub 直连超时用代理: git clone https://gh-proxy.org/https://github.com/vllm-project/vllm.git
```

## 5. 切换 vllm 到 v0.20.2 并安装

```bash
cd /home/vllm
git fetch --depth 1 origin tag v0.20.2
git checkout v0.20.2
VLLM_TARGET_DEVICE=empty pip install -v -e .
pip show vllm | grep Version   # 预期: 0.20.2+empty
```

## 6. triton 兼容性修复

```bash
pip uninstall triton triton-ascend -y
pip install triton-ascend==3.2.1

# Patch next_power_of_2（triton 3.2.0 缺少此函数）
TRITON_INIT=$(python3 -c "import triton; import os; print(os.path.join(os.path.dirname(triton.__file__), '__init__.py'))")
if ! grep -q "next_power_of_2" "$TRITON_INIT"; then
    printf '\ndef next_power_of_2(n):\n    if n <= 0:\n        return 1\n    return 1 << (n - 1).bit_length()\n' >> "$TRITON_INIT"
fi
python3 -c "import triton; print('triton', triton.__version__, 'next_power_of_2:', hasattr(triton, 'next_power_of_2'))"
# 预期: triton 3.2.0 next_power_of_2: True
```

## 7. vllm-ascend 编译修复并安装

```bash
# 限制编译并行度为1（避免 opc 并行编译冲突）
sed -i 's/JOB_NUM="-j${CPU_NUM}"/JOB_NUM="-j1"/' /home/vllm-ascend-official/csrc/build.sh

# 禁用 TILINGKEY_PAR_COMPILE
find /home/vllm-ascend-official/csrc -maxdepth 2 -name 'CMakeLists.txt' -o -name '*.cmake' | \
    xargs grep -l 'TILINGKEY_PAR_COMPILE' | while read f; do
    sed -i 's/TILINGKEY_PAR_COMPILE=1/TILINGKEY_PAR_COMPILE=0/g' "$f"
done
sed -i 's/export TILINGKEY_PAR_COMPILE=1/export TILINGKEY_PAR_COMPILE=0/' /home/vllm-ascend-official/csrc/cmake/func.cmake

# 安装
cd /home/vllm-ascend-official
rm -rf csrc/build
pip install -v -e .
pip show vllm_ascend | grep Version  # 预期: 0.1.dev1+gface36594
```

## 8. 安装 memfabric_hybrid

```bash
cd /home/memfabric_hybrid
git clean -xdf && git reset --hard
git submodule update --recursive --init
pip install pybind11==3.0.1
bash script/build_and_pack_run.sh --build_mode RELEASE --build_python ON --xpu_type NPU
cd output && bash memfabric_hybrid-*.run
source /usr/local/memfabric_hybrid/set_env.sh
pip show memfabric_hybrid  # 预期: 1.0.8
```

## 9. 安装 memcache

```bash
cd /home/memcache
git clean -xdf && git reset --hard
git submodule update --recursive --init
git submodule update --remote 3rdparty/memfabric_hybrid
bash script/build_and_pack_run.sh --build_mode RELEASE
cd output && bash memcache_hybrid-*.run
source /usr/local/memcache_hybrid/set_env.sh
pip show memcache_hybrid  # 预期: 1.0.4
```

## 10. 配置 memcache

```bash
mkdir -p /home/<your_name>/layerwise
cp /usr/local/memcache_hybrid/latest/config/mmc-meta.conf /home/<your_name>/layerwise/
cp /usr/local/memcache_hybrid/latest/config/mmc-local.conf /home/<your_name>/layerwise/

# A2 平台用 device_rdma，A3 平台用 device_sdma
sed -i 's/ock.mmc.local_service.protocol = host_rdma/ock.mmc.local_service.protocol = device_rdma/' /home/<your_name>/layerwise/mmc-local.conf
sed -i 's/ock.mmc.local_service.dram.size = 1GB/ock.mmc.local_service.dram.size = 10GB/' /home/<your_name>/layerwise/mmc-local.conf
```

## 11. 配置 NPU 设备 RDMA IP（A2 平台必须）

```bash
for i in 0 1 2 3 4 5 6 7; do
    hccn_tool -i $i -ip -s address "192.168.100.$((i+1))" netmask "255.255.255.0"
done
# 验证
for i in 0 1 2 3 4 5 6 7; do echo -n "NPU $i: "; hccn_tool -i $i -ip -g; done
```

## 12. 启动 MetaService

```bash
cat > /home/<your_name>/start_meta.sh << 'EOF'
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
EOF
chmod +x /home/<your_name>/start_meta.sh
bash /home/<your_name>/start_meta.sh
# 验证: netstat -tlnp | grep -E '5000|6000|8000'
```

## 13. Layerwise 特性验证

```bash
cat > /home/<your_name>/verify_layerwise.py << 'EOF'
import vllm
print("vllm:", vllm.__version__)
import vllm_ascend
print("vllm_ascend:", vllm_ascend.__file__)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
print("layerwise=True =>", AscendStoreConnector.requires_piecewise_for_cudagraph({"use_layerwise": True}))
print("layerwise=False =>", AscendStoreConnector.requires_piecewise_for_cudagraph({"use_layerwise": False}))
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreLayerSendingThread, KVCacheStoreLayerRecvingThread
print("LayerSendingThread:", KVCacheStoreLayerSendingThread.__name__)
print("LayerRecvingThread:", KVCacheStoreLayerRecvingThread.__name__)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
import inspect
print("KVPoolWorker params:", list(inspect.signature(KVPoolWorker.__init__).parameters.keys()))
assert hasattr(KVPoolWorker, "retrieve_layer")
assert hasattr(KVPoolWorker, "store_layer")
print("retrieve_layer/store_layer: OK")
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
print("KVPoolScheduler params:", list(inspect.signature(KVPoolScheduler.__init__).parameters.keys()))
import triton
print("triton:", triton.__version__, "next_power_of_2:", hasattr(triton, "next_power_of_2"))
print("ALL CHECKS PASSED")
EOF
/usr/local/python3.11.15/bin/python3.11 /home/<your_name>/verify_layerwise.py
```

## 14. 启动 PD 混部 vllm serve

```bash
cat > /home/<your_name>/run.sh << 'EOF'
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
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export VLLM_VERSION=0.20.2

source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

export MMC_LOCAL_CONFIG_PATH=/home/<your_name>/layerwise/mmc-local.conf

# kv_both=PD混部, use_layerwise=true 启用 layerwise KV transfer
# max_model_len 受 KV cache 限制最大约 54000，报 ValueError 时降低此值
# 不能加 --no-disable-hybrid-kv-cache-manager（与 layerwise 不兼容）
# 不能加 --enforce-eager（layerwise 需要 PIECEWISE cudagraph）
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
EOF
chmod +x /home/<your_name>/run.sh

# 确保 8 卡空闲
npu-smi info | grep HBM-Usage  # 每卡应 < 5000 MB

# 启动
nohup bash /home/<your_name>/run.sh > /dev/null 2>&1 &
tail -f /home/<your_name>/log_p.log

# 成功标志:
# "Loading model weights took 36.0575 GB"
# "Available KV cache memory: 17.73 GiB"       ← memcache store.init() 成功
# "GPU KV cache size: 49,280 tokens"            ← KV cache 分配成功
# "torch.compile took XX.XX s in total"         ← PIECEWISE 图编译成功
# "Application startup complete."               ← 服务启动成功
```

## 15. 自动重试脚本（应对 HCCL 并发竞争）

```bash
cat > /home/<your_name>/retry.sh << 'EOF'
#!/bin/bash
exec > /home/<your_name>/retry.log 2>&1
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    echo "=== Attempt $attempt $(date) ==="
    pkill -9 -f vllm 2>/dev/null
    sleep 10
    for pid in $(ps aux | grep VLLMWorker | grep -v grep | awk '{print $2}'); do kill -9 $pid 2>/dev/null; done
    sleep 5
    kill -9 $(cat /home/<your_name>/mmc_meta_pid.txt 2>/dev/null) 2>/dev/null
    sleep 3
    bash /home/<your_name>/start_meta.sh
    sleep 5
    bash /home/<your_name>/run.sh &
    for i in $(seq 1 50); do
        sleep 30
        if grep -q "Application startup complete" /home/<your_name>/log_p.log 2>/dev/null; then
            echo "SUCCESS on attempt $attempt"
            grep "Available KV cache" /home/<your_name>/log_p.log
            grep "GPU KV cache size" /home/<your_name>/log_p.log
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
EOF
chmod +x /home/<your_name>/retry.sh
nohup bash /home/<your_name>/retry.sh > /dev/null 2>&1 &
tail -f /home/<your_name>/retry.log
```

## 16. PD 分离模式（P=4 D=4）

```bash
# Prefill 实例 (NPU 0-3, kv_producer)
cat > /home/<your_name>/run_prefill.sh << 'EOF'
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
export MMC_LOCAL_CONFIG_PATH=/home/<your_name>/layerwise/mmc-local.conf
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# 注意: DeepSeek-V4 280GB/4卡=70GB > 64GB HBM，需换更小模型
vllm serve /mnt/sfs_turbo/weights/<model_name> \
    --host 0.0.0.0 --port 8004 --tensor-parallel-size 4 --seed 1024 \
    --served-model-name <model_name> --enable-expert-parallel \
    --max-num-seqs 40 --max-model-len 1048576 --max-num-batched-tokens 4096 \
    --trust-remote-code --no-enable-prefix-caching --gpu-memory-utilization 0.85 \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector","kv_role": "kv_producer",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' > /home/<your_name>/log_p.log 2>&1
EOF

# Decode 实例 (NPU 4-7, kv_consumer)
cat > /home/<your_name>/run_decode.sh << 'EOF'
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
export MMC_LOCAL_CONFIG_PATH=/home/<your_name>/layerwise/mmc-local.conf
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
vllm serve /mnt/sfs_turbo/weights/<model_name> \
    --host 0.0.0.0 --port 8005 --tensor-parallel-size 4 --seed 1024 \
    --served-model-name <model_name> --enable-expert-parallel \
    --max-num-seqs 40 --max-model-len 1048576 --max-num-batched-tokens 4096 \
    --trust-remote-code --no-enable-prefix-caching --gpu-memory-utilization 0.85 \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector","kv_role": "kv_consumer",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}' > /home/<your_name>/log_d.log 2>&1
EOF

# 启动顺序: 先 Prefill 再 Decode
# nohup bash run_prefill.sh > /dev/null 2>&1 &
# 等 "Application startup complete." 后
# nohup bash run_decode.sh > /dev/null 2>&1 &
```
