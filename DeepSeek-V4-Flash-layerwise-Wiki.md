# DeepSeek-V4-Flash-w8a8-mtp PD混部 layerwise 性能验证 Wiki

## 1. 环境概览

| 组件 | 版本 | 备注 |
|------|------|------|
| 服务器 | 121.37.53.41:10001 (root) | openEuler 2403sp3, aarch64, 8×910B3 |
| 容器 | quay.io/ascend/vllm-ascend:v0.20.2rc1 | 容器名 tyy_b120 |
| Python | 3.11.15 | /usr/local/python3.11.15/bin/python3 |
| CANN | 9.0.0 | /usr/local/Ascend/cann-9.0.0 |
| ATB | 9.1.T1 (cxx_abi_1) | /usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1 |
| torch | 2.10.0+cpu | pip 安装 |
| torch_npu | 2.10.0 | pip 安装 |
| transformers | 5.14.1 | 需手动升级（容器自带 5.5.3 不够） |
| triton | 3.5.0 | HAS_TRITON=False，Ascend 不使用 triton kernels |
| triton-ascend | 3.2.0 | pip 安装 |
| vllm | **v0.27.1** (6e448d0) | /home/vllm，PYTHONPATH 方式加载 |
| vllm-ascend | **main** (1aad874) | /home/vllm-ascend，PYTHONPATH 方式加载 |
| memcache | **1.2.0** | release/1.2 分支构建安装 |
| memfabric | **1.2.0** | release/1.2 分支构建安装 |
| 模型 | DeepSeek-V4-Flash-w8a8-mtp | /mnt/sfs_turbo/weights/DeepSeek-V4-Flash-w8a8-mtp |

## 2. 环境搭建完整步骤

### 2.1 SSH 连接

```bash
# SSH config (~/.ssh/config)
Host npu
    HostName 121.37.53.41
    Port 10001
    User root
    IdentityFile C:\Users\Administrator\.ssh\id_ed25519
    IdentitiesOnly yes
```

### 2.2 构建 memfabric_hybrid 1.2.0（宿主机）

```bash
# 拉取代码
cd /home/memfabric_hybrid
git checkout release/1.2    # VERSION=1.2.0
git submodule update --recursive --init

# 编译
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
rm -rf build
bash script/build_and_pack_run.sh --build_mode RELEASE --build_python ON --xpu_type NPU --build_test OFF

# 安装
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash /home/memfabric_hybrid/output/memfabric_hybrid-1.2.0_linux_aarch64.run --quiet

# 验证
cat /usr/local/memfabric_hybrid/latest/version.info   # Version:1.2.0
pip show memfabric-hybrid | grep Version               # 1.2.0
```

### 2.3 构建 memcache 1.2.0（宿主机）

```bash
# 拉取代码
cd /home/memcache
git checkout release/1.2    # VERSION=1.2.0
git submodule update --recursive --init

# 编译
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
rm -rf build
bash script/build_and_pack_run.sh --build_mode RELEASE

# 安装
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
bash /home/memcache/output/memcache_hybrid-1.2.0_linux_aarch64.run --quiet

# 验证
cat /usr/local/memcache_hybrid/latest/version.info     # Version:1.2.0
pip show memcache-hybrid | grep Version                # 1.2.0
```

### 2.4 vllm v0.27.1 + vllm-ascend main 代码准备（宿主机）

```bash
# vllm v0.27.1
cd /home/vllm
git describe --tags    # v0.27.1

# vllm-ascend main（需更新到最新，旧版 verified commit 不匹配 v0.27.1）
cd /home/vllm-ascend
git fetch origin
git checkout main
git pull origin main   # HEAD=1aad874, verified vllm commit=58d3918e3 (v0.26.1-based)
```

### 2.5 容器内依赖安装

```bash
# 进入容器
docker exec -it tyy_b120 bash

# 升级 transformers（容器自带 5.5.3，vllm-ascend main 需要 5.14.1）
pip install transformers==5.14.1

# 安装 vllm 依赖（容器内 pip 无 vllm，用 PYTHONPATH 方式）
pip install -r /home/vllm/requirements/common.txt

# 安装 vllm-ascend 额外依赖
pip install msgpack quart numba arctic-inference torchvision==0.25.0 pandas-stubs

# 降级 numpy（triton-ascend 要求 1.26.4）
pip install numpy==1.26.4
```

### 2.6 修复 vllm v0.27.1 pyc 缺失问题

vllm v0.27.1 部分模块只有 `__pycache__/*.pyc` 没有 `.py` 源文件，需复制 pyc 到包目录：

```bash
# 在容器内执行
VLLM_DIR="/home/vllm/vllm"
find "$VLLM_DIR" -type d -name "__pycache__" | while read cache_dir; do
    pkg_dir=$(dirname "$cache_dir")
    for pyc in "$cache_dir"/*.cpython-311.pyc; do
        [ -f "$pyc" ] || continue
        module=$(basename "$pyc" .cpython-311.pyc)
        if [ ! -f "$pkg_dir/${module}.py" ] && [ ! -f "$pkg_dir/${module}.pyc" ]; then
            cp "$pyc" "$pkg_dir/${module}.pyc"
        fi
    done
done
```

### 2.7 修复 vllm cp_utils.py API 缺失

vllm-ascend main 调用 `get_total_cp_world_size`，但 vllm v0.27.1 没有此函数，需手动添加：

```bash
# 在 /home/vllm/vllm/v1/worker/cp_utils.py 末尾追加
cat >> /home/vllm/vllm/v1/worker/cp_utils.py << 'PYEOF'


def get_total_cp_world_size() -> int:
    """Get total context parallel world size (pcp_size * dcp_size).
    Returns 1 when context parallel is not used.
    """
    try:
        from vllm.distributed import get_pcp_group, get_dcp_group
        pcp_size = get_pcp_group().world_size
        dcp_size = get_dcp_group().world_size
        return pcp_size * dcp_size
    except Exception:
        return 1
PYEOF
```

### 2.8 Patch layerwise_cache_layout.py（支持 DeepSeek-V4 多 cache spec）

**问题：** DeepSeek-V4 的 DSA 稀疏注意力层有 5 个 cache spec（compressor.state_cache、indexer.k_cache、indexer.compressor.state_cache、swa_cache、attn），原始代码要求每层只能有 1 个 main spec + 1 个 indexer.k_cache spec。

**Patch 文件：** `/home/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py`

```python
#!/usr/bin/env python3
"""Patch layerwise_cache_layout.py to support DeepSeek-V4's multi cache spec"""
import sys

FILE = "/home/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py"

with open(FILE) as f:
    content = f.read()

if "extra_main_specs" in content:
    print("Already patched, skipping")
    sys.exit(0)

# 1. 给 LayerwiseLayerCacheSpecs 添加 extra_main_specs 字段
old_dataclass = """    indexer: NamedKVCacheSpec | None = None"""
new_dataclass = """    indexer: NamedKVCacheSpec | None = None
    extra_main_specs: tuple[NamedKVCacheSpec, ...] = ()"""
content = content.replace(old_dataclass, new_dataclass, 1)

# 2. 修改检查逻辑：允许多个 main specs
old_check = """        indexer_specs = [spec for spec in named_specs if spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        main_specs = [spec for spec in named_specs if not spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        if len(main_specs) != 1 or len(indexer_specs) != 1:
            raise ValueError(
                f"Physical layer {physical_layer} with multiple cache specs must have "
                f"exactly one main spec and one '{_INDEXER_CACHE_SUFFIX}' spec; "
                f"got {[spec.layer_name for spec in named_specs]}."
            )
        layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(
            main=main_specs[0],
            indexer=indexer_specs[0],
        )"""

new_check = """        indexer_specs = [spec for spec in named_specs if spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        main_specs = [spec for spec in named_specs if not spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        if len(main_specs) < 1:
            raise ValueError(
                f"Physical layer {physical_layer} has no main cache spec; "
                f"got {[spec.layer_name for spec in named_specs]}."
            )
        # Select '.attn' as main spec, rest as extra
        main_spec = next((s for s in main_specs if s.layer_name.endswith(".attn")), main_specs[0])
        extra_specs = tuple(s for s in main_specs if s is not main_spec)
        indexer_spec = indexer_specs[0] if indexer_specs else None
        layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(
            main=main_spec,
            indexer=indexer_spec,
            extra_main_specs=extra_specs,
        )"""

if old_check not in content:
    print("ERROR: Could not find the check block to replace")
    sys.exit(1)

content = content.replace(old_check, new_check, 1)

with open(FILE, "w") as f:
    f.write(content)

print("Patch applied successfully")
```

**执行 patch（在容器内）：**
```bash
python3 /tmp/patch_layerwise_v2.py
```

> **注意：** 此 patch 仅对 layerwise ON 需要。layerwise OFF 不需要 patch。

### 2.9 MemCache 配置文件

**文件：** `/home/lsy/layerwise/mmc-local.conf`

```
## Meta service start-up url
ock.mmc.meta_service_url = tcp://127.0.0.1:5000

## Config store url
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000

## Log level: debug, info, warn, error
ock.mmc.log_level = info

## The maximum supported rank count
ock.mmc.local_service.world_size = 256

## Data transfer protocol: host_shm for same-node inter-process communication
ock.mmc.local_service.protocol = host_shm

## DRAM space usage
ock.mmc.local_service.dram.size = 1GB
```

## 3. 三种测试场景的启动脚本

### 3.1 通用启动脚本 `start_vllm_container.sh`

**文件路径：** 容器内 `/home/lsy/start_vllm_container.sh`

```bash
#!/bin/bash
# DeepSeek-V4-Flash PD混部性能测试启动脚本
# vllm v0.27.1 + vllm-ascend main + memcache 1.2.0

# 环境变量
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
source /usr/local/memcache_hybrid/set_env.sh 2>/dev/null
source /usr/local/memfabric_hybrid/set_env.sh 2>/dev/null

export PYTHONPATH=/home/vllm:/home/vllm-ascend:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export MMC_LOCAL_CONFIG_PATH=/home/lsy/layerwise/mmc-local.conf

export HCCL_IF_IP=172.28.1.10
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=200
export HCCL_NPU_SOCKET_PORT_RANGE=20000-25000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

LAYERWISE=$1
if [ "$LAYERWISE" = "off" ]; then
    # 场景1: layerwise OFF
    KV_CONFIG='{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","use_layerwise":false,"mooncake_rpc_port":"0"}}'
else
    # 场景2/3: layerwise ON
    KV_CONFIG='{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","use_layerwise":true,"mooncake_rpc_port":"0"}}'
    # 场景3 加 prefetch: 将上面替换为:
    # KV_CONFIG='{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","use_layerwise":true,"layerwise_prefetch_layers":3,"mooncake_rpc_port":"0"}}'
fi

echo "=== use_layerwise: $LAYERWISE ==="
echo "=== KV_CONFIG: $KV_CONFIG ==="

python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/sfs_turbo/weights/DeepSeek-V4-Flash-w8a8-mtp \
    --port 8004 --tensor-parallel-size 8 --seed 1024 \
    --served-model-name deepseek_v4 \
    --enable-expert-parallel \
    --max-num-seqs 40 --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --gpu-memory-utilization 0.85 \
    --quantization ascend \
    --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":true}}' \
    --kv-transfer-config "$KV_CONFIG"
```

### 3.2 三种场景的 KV_CONFIG 差异

| 场景 | use_layerwise | layerwise_prefetch_layers | gpu_memory_utilization | max_model_len |
|------|:---:|:---:|:---:|:---:|
| 场景1: layerwise OFF | false | - | 0.85 | 16384 |
| 场景2: layerwise ON (无prefetch) | true | - | 0.85 | 16384 |
| 场景3: layerwise ON + prefetch:3 | true | 3 | 0.79* | 8192* |

> *场景3 因 NPU 被其他进程占用约 12GB/卡，降低 gpu_memory_utilization 和 max_model_len 以适配。场景1/2 跑在 NPU 全空时。

**场景1 KV_CONFIG:**
```json
{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","use_layerwise":false,"mooncake_rpc_port":"0"}}
```

**场景2 KV_CONFIG:**
```json
{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","use_layerwise":true,"mooncake_rpc_port":"0"}}
```

**场景3 KV_CONFIG:**
```json
{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"backend":"memcache","use_layerwise":true,"layerwise_prefetch_layers":3,"mooncake_rpc_port":"0"}}
```

### 3.3 后台启动命令

```bash
# 清理旧进程 + 启动（容器内执行）
# clean_and_launch.sh
#!/bin/bash
for pid in $(pgrep -f "VLLM::" 2>/dev/null); do kill -9 $pid 2>/dev/null; done
for pid in $(pgrep -f "vllm.entrypoints" 2>/dev/null); do kill -9 $pid 2>/dev/null; done
for pid in $(pgrep -f "start_vllm_container" 2>/dev/null); do kill -9 $pid 2>/dev/null; done
for pid in $(pgrep -x python3 2>/dev/null); do
    if [ "$pid" -eq 483203 ]; then continue; fi  # 跳过 memcache MetaService
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmdline" | grep -q "multiprocessing"; then kill -9 $pid 2>/dev/null; fi
done
sleep 3
rm -f /home/lsy/log_p_layerwise_on.log   # 或 log_p_layerwise_off.log
setsid bash /home/lsy/start_vllm_container.sh on > /home/lsy/log_p_layerwise_on.log 2>&1 &
echo "launched PID=$!"
```

```bash
# 宿主机执行
docker exec tyy_b120 bash /tmp/clean_and_launch.sh

# 等待约 15-20 分钟模型加载完成，检查端口
docker exec tyy_b120 bash -c "netstat -tlnp 2>/dev/null | grep 8004"
```

## 4. 压测脚本

**文件：** 容器内 `/home/lsy/bench_pd.py`

```python
#!/usr/bin/env python3
"""简单压测脚本 - PD混部性能测试"""
import json
import time
import asyncio
import aiohttp

API_URL = "http://localhost:8004/v1/completions"
MODEL = "deepseek_v4"
NUM_PROMPTS = 50
CONCURRENCY = 20
INPUT_LEN = 512   # prefill 长度
OUTPUT_LEN = 128  # decode 长度

async def send_request(session, idx):
    payload = {
        "model": MODEL,
        "prompt": "Hello " * INPUT_LEN,
        "max_tokens": OUTPUT_LEN,
        "temperature": 0,
        "stream": False,
    }
    start = time.time()
    try:
        async with session.post(API_URL, json=payload, timeout=300) as resp:
            data = await resp.json()
            latency = time.time() - start
            output_tokens = data.get("usage", {}).get("completion_tokens", OUTPUT_LEN)
            return {"idx": idx, "latency": latency, "output_tokens": output_tokens, "success": True}
    except Exception as e:
        return {"idx": idx, "error": str(e), "success": False, "latency": time.time() - start}

async def main():
    print(f"=== Benchmark: {NUM_PROMPTS} prompts, concurrency={CONCURRENCY} ===")
    print(f"=== Input len: {INPUT_LEN} tokens, Output len: {OUTPUT_LEN} tokens ===")
    print("Warmup...")
    async with aiohttp.ClientSession() as session:
        await send_request(session, -1)
    print("Starting benchmark...")
    start_time = time.time()
    sem = asyncio.Semaphore(CONCURRENCY)
    async def bounded_send(session, idx):
        async with sem:
            return await send_request(session, idx)
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[bounded_send(session, i) for i in range(NUM_PROMPTS)])
    total_time = time.time() - start_time
    success = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    total_output_tokens = sum(r.get("output_tokens", 0) for r in success)
    print(f"\n=== Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success: {len(success)}/{NUM_PROMPTS}")
    print(f"Failed: {len(failed)}")
    print(f"Request throughput: {len(success)/total_time:.2f} req/s")
    print(f"Output token throughput: {total_output_tokens/total_time:.2f} tokens/s")
    if success:
        latencies = [r["latency"] for r in success]
        print(f"Avg latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"P50 latency: {sorted(latencies)[len(latencies)//2]:.2f}s")
        print(f"P99 latency: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}s")
    result_file = f"/home/lsy/bench_result_{int(time.time())}.json"
    with open(result_file, "w") as f:
        json.dump({"config": {"num_prompts": NUM_PROMPTS, "concurrency": CONCURRENCY,
                   "input_len": INPUT_LEN, "output_len": OUTPUT_LEN},
                   "total_time": total_time, "success_count": len(success),
                   "total_output_tokens": total_output_tokens,
                   "request_throughput": len(success)/total_time,
                   "output_token_throughput": total_output_tokens/total_time,
                   "latencies": [r.get("latency", 0) for r in success]}, f, indent=2)
    print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    asyncio.run(main())
```

**执行压测：**
```bash
docker exec tyy_b120 python3 /home/lsy/bench_pd.py
```

## 5. 性能对比结果

### 5.1 压测参数

| 参数 | 值 |
|------|-----|
| 请求数 | 50 |
| 并发数 | 20 |
| 输入长度 | 512 tokens |
| 输出长度 | 128 tokens |
| 采样方式 | temperature=0 (greedy) |

### 5.2 结果汇总

| 指标 | 场景1: layerwise OFF | 场景2: ON (无prefetch) | 场景3: ON + prefetch:3 |
|------|:---:|:---:|:---:|
| 总耗时 | 44.75s | 191.44s | 119.71s |
| 成功率 | 50/50 (100%) | 50/50 (100%) | 50/50 (100%) |
| Request throughput | 1.12 req/s | 0.26 req/s | 0.42 req/s |
| Output token throughput | 143.01 tok/s | 33.43 tok/s | 53.46 tok/s |
| Avg latency | 16.43s | 63.41s | 39.75s |
| P50 latency | 15.49s | 64.15s | 39.96s |
| P99 latency | 22.25s | 66.68s | 41.12s |
| Min latency | 7.28s | 58.42s | 38.59s |
| Max latency | 22.25s | 66.68s | 41.12s |

### 5.3 可视化对比

```
Output token throughput (tokens/s)
┌──────────────────────────────────────────────────────────┐
│ 场景1 OFF           ████████████████████████      143.01 │
│ 场景2 ON(无prefetch) █████                          33.43 │
│ 场景3 ON+prefetch:3  ████████                       53.46 │
└──────────────────────────────────────────────────────────┘

Avg latency (s)
┌──────────────────────────────────────────────────────────┐
│ 场景1 OFF           █████                          16.43  │
│ 场景2 ON(无prefetch) ████████████████████          63.41  │
│ 场景3 ON+prefetch:3  ██████████                     39.75  │
└──────────────────────────────────────────────────────────┘

Request throughput (req/s)
┌──────────────────────────────────────────────────────────┐
│ 场景1 OFF           ███████████                     1.12  │
│ 场景2 ON(无prefetch) ██                              0.26  │
│ 场景3 ON+prefetch:3  ████                            0.42  │
└──────────────────────────────────────────────────────────┘
```

### 5.4 场景间对比

| 对比 | throughput 变化 | 延迟变化 |
|------|:---:|:---:|
| 场景2 vs 场景1 (ON无prefetch vs OFF) | -76.6% | +286% |
| 场景3 vs 场景2 (加prefetch vs 不加) | +59.9% | -37.3% |
| 场景3 vs 场景1 (ON+prefetch vs OFF) | -62.6% | +142% |

## 6. 关键注意事项

1. **Patch 必需性：** vllm-ascend main 原始代码不支持 DeepSeek-V4 的 DSA 多 cache spec + layerwise，必须打 patch（见 2.8 节）。layerwise OFF 不需要 patch。

2. **版本兼容性：** vllm-ascend main 的官方 verified vllm commit 是 `58d3918e3`（基于 v0.26.1），与 vllm v0.27.1 有少量 API 差异，需手动修复（见 2.6、2.7 节）。

3. **gpu_memory_utilization 调整：** NPU 被其他进程占用时需降低此值。场景1/2 在 NPU 全空时用 0.85，场景3 因他人占用约 12GB/卡 降到 0.79 + max_model_len 降到 8192。

4. **layerwise 对 cudagraph 的影响：** 开启 layerwise 会强制将 cudagraph_mode 从 FULL_AND_PIECEWISE 降级为 PIECEWISE，损失部分图编译优化。

5. **layerwise_prefetch_layers 效果：** 设置为 3 后相比无 prefetch 提升约 60% throughput、降低 37% 延迟，但相比 OFF 仍有明显差距。

6. **环境变量注意事项：**
   - 容器通过 `docker exec` 启动时，环境变量可能不继承，需在脚本中 `source` 所有 set_env.sh
   - `setsid` 用于后台运行，避免 SSH 断开导致进程退出
   - 清理旧进程时需通过 `pgrep -f "VLLM::"` 精确匹配，避免误杀其他用户的进程
