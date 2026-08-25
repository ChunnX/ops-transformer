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

---
# 附录A：npu2 服务器复现手册（2026-08-25 实测复现）

> 本附录是在另一台服务器 npu2（121.37.53.41:10007，8×910B3，driver 26.0.rc1）上从零复现本文档全流程的完整手册，
> 含环境迁移/容器创建/依赖修复/MetaService/三场景压测/踩坑记录，并附 npu2 实测结果（与本文档结果一致，偏差 <4%）。
> 原始手册位于 npu2:/home/lsy/npu2_pd_setup_guide.md。


> 服务器：npu2（121.37.53.41:10007，内网 172.28.1.220，openEuler 24.03 LTS-SP3, aarch64, 8×910B3）
> 容器：`lsy_pd`（基于 `quay.io/ascend/vllm-ascend:v0.20.2rc1`）
> 撰写日期：2026-08-25
>
> 对应 Wiki：https://gitcode.com/cann/ops-transformer/pull/10470 （DeepSeek-V4-Flash-layerwise-Wiki.md）

---

## 目录

1. [环境总览](#1-环境总览)
2. [环境迁移（一次性，已完成）](#2-环境迁移一次性已完成)
    - [2.0 从零获取四个仓库（全新机器搭建时用）](#20-从零获取四个仓库全新机器搭建时用)
3. [容器创建](#3-容器创建)
4. [容器内依赖修复](#4-容器内依赖修复)
5. [启动 MetaService（每次复跑前必做）](#5-启动-metaservice每次复跑前必做)
6. [三场景启动与压测（日常复跑核心）](#6-三场景启动与压测日常复跑核心)
7. [启动脚本与配置文件详解](#7-启动脚本与配置文件详解)
8. [性能结果](#8-性能结果)
9. [device_rdma 尝试记录（硬件不具备）](#9-device_rdma-尝试记录硬件不具备)
10. [踩坑记录与注意事项](#10-踩坑记录与注意事项)
11. [常用排查命令速查](#11-常用排查命令速查)

---

## 1. 环境总览

| 组件 | 版本 | 位置 |
|---|---|---|
| vllm | v0.27.1 (commit 6e448d0) | `/home/vllm`（PYTHONPATH 加载，含 pyc/cp_utils 补丁） |
| vllm-ascend | main (commit 1aad874) | `/home/vllm-ascend`（PYTHONPATH 加载，含 layerwise 多 cache spec 补丁） |
| memcache | 1.2.0 (release/1.2) | 容器内 pip 包 + 宿主机 `/usr/local/memcache_hybrid` |
| memfabric | 1.2.0 (release/1.2) | 容器内 pip 包 + 宿主机 `/usr/local/memfabric_hybrid` |
| transformers | 5.14.1 | 容器内 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 | 容器内 |
| CANN | 9.0.0 | 容器内 |
| Python | 3.11.15 | `/usr/local/python3.11.15/bin/python3`（容器内） |
| 模型 | DeepSeek-V4-Flash-w8a8-mtp（280GB） | `/mnt/weights/DeepSeek-V4-Flash-w8a8-mtp` |
| NPU 驱动 | 26.0.rc1 | 宿主机 |

关键目录：

```
/home/vllm                    # vllm v0.27.1 源码（已打补丁）
/home/vllm-ascend             # vllm-ascend main 源码（已打补丁）
/home/lsy/                    # 工作目录
├── start_pd_npu2.sh          # 启动脚本（off / on / on_prefetch 三模式）
├── bench_pd.py               # 压测脚本
├── layerwise/mmc-local.conf  # memcache 本地服务配置（host_shm 协议）
├── npu2_log_layerwise_*.log  # 三场景服务日志
└── bench_result_*.json       # 压测结果
/home/wheels/                 # memcache/memfabric 1.2.0 的 wheel 与 run 包备份
```

---

## 2. 环境迁移（一次性，已完成）

以下命令仅作记录，日常复跑**不需要**再执行。

### 2.0 从零获取四个仓库（全新机器搭建时用）

> 本手册部署所用代码的准确来源与版本。若不迁移而是全新 clone，按本节操作；
> 注意 vllm / vllm-ascend 的两个补丁（Wiki 2.6/2.7/2.8 节）需要自行重打。

```bash
# ① vllm v0.27.1（官方仓库，commit 6e448d0ea，tag v0.27.1）
cd /home
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout v0.27.1
git describe --tags        # 确认输出 v0.27.1

# ② vllm-ascend main（commit 1aad874，verified vllm commit=58d3918e3）
#    注意：本环境用的是 npu 服务器上已有的 fork（ader47/vllm-ascend），HEAD=1aad874
cd /home
git clone https://github.com/vader47/vllm-ascend.git    # 或官方仓：https://gitee.com/ascend/vllm-ascend.git
cd vllm-ascend && git checkout 1aad8745e4a3e02644c76d8a44bcbbd0a4113f59

# ③ memcache release/1.2（VERSION=1.2.0，commit ca5f69cf）
cd /home
git clone -b release/1.2 https://gitcode.com/yrewzjsx/memcache.git
cd memcache && git rev-parse HEAD    # ca5f69cf4ba3ae936d9318bf9daa1c34c6916582
git submodule update --recursive --init

# ④ memfabric_hybrid release/1.2（VERSION=1.2.0，commit 999034e9）
cd /home
git clone -b release/1.2 https://gitcode.com/Ascend/memfabric_hybrid.git
cd memfabric_hybrid && git rev-parse HEAD   # 999034e9320abf27bbde7be83a32d3bf4d0f7940
git submodule update --recursive --init
```

克隆后需编译 memcache / memfabric（Wiki 2.2/2.3 节，需要 CANN 环境）：

```bash
# memfabric 编译
cd /home/memfabric_hybrid
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
rm -rf build
bash script/build_and_pack_run.sh --build_mode RELEASE --build_python ON --xpu_type NPU --build_test OFF
# 产物: output/memfabric_hybrid-1.2.0_linux_aarch64.run + wheel/

# memcache 编译（依赖 memfabric 的 set_env）
cd /home/memcache
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
rm -rf build
bash script/build_and_pack_run.sh --build_mode RELEASE
# 产物: output/memcache_hybrid-1.2.0_linux_aarch64.run + wheel/
```

> 提示：在**容器内**编译的 wheel 是 manylinux 兼容版（GLIBC_2.25），
> 在宿主机编译的 wheel 依赖 GLIBC_2.38，装进容器会报错（详见 4.2 节）。
> 全新搭建建议直接把编译放在容器内进行，或参照 4.2 节做替换。

### 2.1 配置 npu2 → npu 免密

```bash
# npu2 上生成密钥（已有则跳过）
ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 -q
cat /root/.ssh/id_ed25519.pub
# 将公钥追加到 npu:/root/.ssh/authorized_keys

# 验证
ssh -p 10001 root@172.28.1.10 "echo OK"   # 应免密输出 OK
```

### 2.2 迁移代码目录

```bash
# npu(172.28.1.10:10001) → npu2，内网直连约 110MB/s
rsync -a -e "ssh -p 10001 -o StrictHostKeyChecking=no" \
  root@172.28.1.10:/home/vllm \
  root@172.28.1.10:/home/vllm-ascend \
  root@172.28.1.10:/home/lsy /home/

# 迁移 memcache/memfabric 1.2.0 构建产物（wheel + run 安装包）
rsync -a -e "ssh -p 10001" \
  root@172.28.1.10:/home/memcache/output/memcache/wheel/memcache_hybrid-1.2.0-cp311-cp311-linux_aarch64.whl \
  root@172.28.1.10:/home/memfabric_hybrid/output/memfabric_hybrid/wheel/memfabric_hybrid-1.2.0-cp311-cp311-linux_aarch64.whl \
  root@172.28.1.10:/home/memcache/output/memcache_hybrid-1.2.0_linux_aarch64.run \
  root@172.28.1.10:/home/memfabric_hybrid/output/memfabric_hybrid-1.2.0_linux_aarch64.run \
  /home/wheels/
```

> **注意**：`/home/vllm` 与 `/home/vllm-ascend` 里已包含 Wiki 2.6/2.7/2.8 节的全部补丁
> （pyc 复制、`cp_utils.py` 补 `get_total_cp_world_size`、`layerwise_cache_layout.py` 多 cache spec 支持），
> 迁移即继承，无需重打。

### 2.3 迁移 Docker 镜像（17.9GB）

```bash
# npu2 上流式传输（docker save | docker load，约 5 分钟）
nohup bash -c 'ssh -p 10001 -o StrictHostKeyChecking=no root@172.28.1.10 \
  "docker save quay.io/ascend/vllm-ascend:v0.20.2rc1" | docker load' \
  > /root/docker_image_transfer.log 2>&1 &

# 验证
docker images quay.io/ascend/vllm-ascend   # 应看到 v0.20.2rc1
```

### 2.4 宿主机安装 memfabric + memcache（顺序不能反）

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 先 memfabric
bash /home/wheels/memfabric_hybrid-1.2.0_linux_aarch64.run --quiet
# 再 memcache（依赖 memfabric 的 set_env）
source /usr/local/memfabric_hybrid/set_env.sh
bash /home/wheels/memcache_hybrid-1.2.0_linux_aarch64.run --quiet

# 验证
cat /usr/local/memcache_hybrid/latest/version.info    # Version:1.2.0
cat /usr/local/memfabric_hybrid/latest/version.info   # Version:1.2.0
pip show memcache-hybrid | grep Version               # 1.2.0
```

> 宿主机安装的 run 包主要用于提供 `mmc_meta_service` 二进制和库文件；
> **容器内 python 用的 wheel 另行处理**（见第 4 节）。

### 2.5 模型迁移（已完成，280GB）

```bash
# 当时命令（已执行完毕，结果已校验 md5 一致）
rsync -a --partial --info=progress2 -e "ssh -p 10001" \
  root@172.28.1.10:/mnt/sfs_turbo/weights/DeepSeek-V4-Flash-w8a8-mtp /mnt/weights/
```

---

## 3. 容器创建

### 3.1 第一次创建（失败教训：--ipc=shareable 导致 /dev/shm 只有 64MB）

```bash
docker run -d --name lsy_pd --privileged --network host --ipc=shareable ...   # ✗ 不可用
```

容器内 vllm 报 `Insufficient space in /dev/shm: 160 MiB required, 64 MiB free`，
且容器内 remount 扩容对 vllm 子进程不生效。

### 3.2 正确做法：commit 后用 --ipc=host 重建

```bash
# 1) 提交容器当前状态（保留所有 pip 安装）为镜像
docker commit lsy_pd lsy_pd:ready

# 2) 删旧容器，用 --ipc=host 重建（共享宿主机 945G /dev/shm）
docker rm -f lsy_pd
docker run -d --name lsy_pd --privileged --network host --ipc=host --restart no \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /dev:/dev \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  -v /home:/home \
  -v /tmp:/tmp \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /mnt:/mnt \
  -v /usr/local/dcmi:/usr/local/dcmi \
  lsy_pd:ready sleep infinity
```

挂载说明（复刻 npu 上 tyy_b120 容器）：

| 挂载 | 用途 |
|---|---|
| `/dev:/dev` | 全量设备（davinci0-7、infiniband uverbs 等） |
| `/usr/local/Ascend/driver` | NPU 驱动（宿主机 26.0.rc1） |
| `/home:/home` | vllm/vllm-ascend 源码与脚本（宿主机直接可改） |
| `/mnt:/mnt` | 模型权重 |
| `/etc/hccn.conf` | RoCE 网络配置 |
| `--network host` | 直用宿主机网络（HCCL、MetaService 127.0.0.1 直达） |
| `--ipc=host` | 共享内存（vllm 多进程 + memcache host_shm 必需） |
| `--privileged` | NPU 设备完全访问 |

---

## 4. 容器内依赖修复

```bash
docker exec -it lsy_pd bash
PIP=/usr/local/python3.11.15/bin/pip
```

### 4.1 transformers 升级 + vllm 0.27.1 依赖

```bash
$PIP install transformers==5.14.1          # 镜像自带 5.5.3 不够
$PIP install -r /home/vllm/requirements/common.txt   # xgrammar>=0.2.1, fastapi>=0.133, starlette>=1.0.1 等
```

> 不装 requirements 会报：`ImportError: cannot import name 'normalize_tool_choice' from 'xgrammar'`

### 4.2 memcache/memfabric wheel 的 GLIBC 坑（重要）

宿主机编译的 wheel（`/home/wheels/*.whl`）链接 GLIBC_2.38，而容器系统是 Ubuntu glibc 2.35，
直接安装后 `import memcache_hybrid` 报：
`OSError: libc.so.6: version 'GLIBC_2.38' not found (required by libmf_hybm_core.so)`

**解决**：从 npu 的 tyy_b120 容器里打包 manylinux 版（GLIBC_2.25 兼容）直接覆盖：

```bash
# 在 npu 上打包（已执行，产物在 npu2:/tmp/mf_pkgs.tar.gz）
docker exec tyy_b120 bash -c 'cd /usr/local/python3.11.15/lib/python3.11/site-packages && \
  tar czf /tmp/mf_pkgs.tar.gz \
  memfabric_hybrid memfabric_hybrid-1.2.0.dist-info memfabric_hybrid.libs \
  memcache_hybrid memcache_hybrid-1.2.0.dist-info memcache_hybrid.libs'

# 在 npu2 容器内解压替换
SP=/usr/local/python3.11.15/lib/python3.11/site-packages
rm -rf $SP/memfabric_hybrid* $SP/memcache_hybrid*
tar xzf /tmp/mf_pkgs.tar.gz -C $SP/

# 验证
python3 -c "import memcache_hybrid, memfabric_hybrid; print('OK')"
```

### 4.3 环境自检（容器内）

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH=/home/vllm:/home/vllm-ascend
python3 -c "import vllm; print(vllm.__version__)"              # 0.27.1
python3 -c "import memcache_hybrid; print('OK')"               # OK
python3 -c "import torch, torch_npu; print(torch.__version__)" # 2.10.0+cpu
```

---

## 5. 启动 MetaService（每次复跑前必做）

memcache backend 需要独立的 MetaService 进程（监听 127.0.0.1:5000/6000/8000），
**在宿主机运行**（容器 host 网络可直接访问 127.0.0.1）：

```bash
# 宿主机执行
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
nohup /usr/local/memcache_hybrid/latest/aarch64-linux/bin/mmc_meta_service \
  > /root/mmc_meta_service.log 2>&1 &

# 验证端口
netstat -tlnp | grep -E ':5000 |:6000 '   # 两个端口都在 LISTEN 即 OK
```

**卡死/重启方法**（进程有锁文件）：

```bash
pkill -9 -f mmc_meta_service
rm -f /tmp/mmc_meta_service.lock /var/log/memcache_hybrid/*.lock
# 然后重新 nohup 启动
```

> MetaService 保存了 rank 配置一致性状态（maxDRAMSize 等）。如果改过
> `dram.size`/`max.dram.size` 后初始化报 `CheckRankConfigConsistency` 不一致，
> 重启 MetaService 即可清掉旧状态。

---

## 6. 三场景启动与压测（日常复跑核心）

### 6.1 场景定义

| 场景 | start_pd_npu2.sh 参数 | use_layerwise | prefetch_layers |
|---|---|---|---|
| 场景1：OFF | `off` | false | - |
| 场景2：ON | `on` | true | - |
| 场景3：ON+prefetch:3 | `on_prefetch` | true | 3 |

### 6.2 启动流程（容器内执行）

```bash
# ① 清理旧进程（每次必做，否则端口/设备占用）
for pid in $(pgrep -f "vllm.entrypoints"); do kill -9 $pid; done
for pid in $(pgrep -f "start_pd_npu2");    do kill -9 $pid; done
for pid in $(pgrep -f "VLLM::");           do kill -9 $pid; done
sleep 10
# 确认 HBM 释放（每卡回到 ~3400MB 基线）
npu-smi info | grep "HBM-Usage"

# ② 后台启动（三选一：off / on / on_prefetch）
setsid bash /home/lsy/start_pd_npu2.sh off > /home/lsy/launcher.log 2>&1 &

# ③ 等待就绪（模型加载约 15~18 分钟）
watch -n 60 "tail -3 /home/lsy/npu2_log_layerwise_off.log; netstat -tln | grep 8004"
# 看到 "Application startup complete" 且 8004 端口 LISTEN 即就绪
```

### 6.3 压测（容器内执行）

```bash
docker exec lsy_pd /usr/local/python3.11.15/bin/python3 /home/lsy/bench_pd.py
```

压测参数（bench_pd.py 内置，与 Wiki 一致）：50 请求、并发 20、输入 512 tokens、输出 128 tokens、temperature=0。

结果 JSON 自动保存到 `/home/lsy/bench_result_<时间戳>.json`。

### 6.4 三场景串行完整流程

```bash
for MODE in off on on_prefetch; do
    # 清理
    for pid in $(pgrep -f "vllm.entrypoints"); do kill -9 $pid; done; sleep 10
    # 启动
    setsid bash /home/lsy/start_pd_npu2.sh $MODE > /home/lsy/launcher.log 2>&1 &
    # 等就绪（轮询 8004）
    while ! netstat -tln | grep -q ':8004 '; do sleep 60; done; sleep 30
    # 压测
    /usr/local/python3.11.15/bin/python3 /home/lsy/bench_pd.py | tee /home/lsy/bench_${MODE}.txt
done
```

---

## 7. 启动脚本与配置文件详解

### 7.1 `/home/lsy/start_pd_npu2.sh`（核心启动脚本）

关键内容（完整脚本见服务器）：

```bash
#!/bin/bash
MODE=${1:-on}   # off / on / on_prefetch

# ---- 环境 ----
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=/home/vllm:/home/vllm-ascend:${PYTHONPATH}
export MMC_LOCAL_CONFIG_PATH=/home/lsy/layerwise/mmc-local.conf   # memcache 配置

# ---- 网络（npu2 专属 IP）----
export HCCL_IF_IP=172.28.1.220
export GLOO_SOCKET_IFNAME=enp189s0f0
export TP_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=200
export HCCL_NPU_SOCKET_PORT_RANGE=20000-25000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

# ---- 三场景 KV_CONFIG ----
# off:         {"kv_connector":"AscendStoreConnector","kv_role":"kv_both",
#               "kv_connector_extra_config":{"backend":"memcache","use_layerwise":false,"mooncake_rpc_port":"0"}}
# on:          同上，use_layerwise=true
# on_prefetch: 同上，use_layerwise=true，另加 "layerwise_prefetch_layers":3

/usr/local/python3.11.15/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /mnt/weights/DeepSeek-V4-Flash-w8a8-mtp \
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
    --kv-transfer-config "$KV_CONFIG" > ${LOG_FILE} 2>&1
```

### 7.2 `/home/lsy/layerwise/mmc-local.conf`（memcache 本地服务配置）

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = host_shm      # ← 当前使用协议
ock.mmc.local_service.dram.size = 1GB
```

### 7.3 `/home/lsy/bench_pd.py`

50 并发 20 的 aiohttp 压测脚本，POST `http://localhost:8004/v1/completions`，
prompt 为 "Hello "×512，max_tokens=128，greedy。输出吞吐/延迟统计与 JSON 落盘。

---

## 8. 性能结果

压测参数：50 请求、并发 20、输入 512 / 输出 128 tokens、greedy。

| 指标 | 场景1: OFF | 场景2: ON | 场景3: ON+prefetch:3 |
|---|---|---|---|
| 总耗时 | 43.63s | 119.04s | 115.58s |
| 成功率 | 50/50 | 50/50 | 50/50 |
| Request throughput | 1.15 req/s | 0.42 req/s | 0.43 req/s |
| Output token throughput | **146.70 tok/s** | **53.76 tok/s** | **55.37 tok/s** |
| Avg latency | 16.23s | 39.94s | 38.64s |
| P50 latency | 12.66s | 38.59s | 38.57s |
| P99 latency | 29.69s | 42.97s | 39.62s |

结果文件：

- 场景1：`/home/lsy/bench_result_1787571782.json`
- 场景2：`/home/lsy/bench_result_1787573055.json`
- 场景3：`/home/lsy/bench_result_1787574395.json`
- 服务日志：`/home/lsy/npu2_log_layerwise_{off,on,on_prefetch}.log`

与 Wiki（npu 原机）对比：OFF 146.70 vs 143.01（+2.6%）、ON+prefetch 55.37 vs 53.46（+3.6%），基本复现。

---

## 9. device_rdma 尝试记录（硬件不具备）

结论：**npu2（以及 npu）的 8 卡 RoCE 物理链路全部 DOWN，无法使用 device_rdma 协议。**

尝试过程（记录备查）：

```bash
# ① 为 8 张 NPU 配置 RoCE IP（已写入 /etc/hccn.conf，重启保留）
for i in 0 1 2 3 4 5 6 7; do
    hccn_tool -i $i -ip -s address 192.168.100.$((11+i)) netmask 255.255.255.0
done

# ② 新建 device_rdma 配置 /home/lsy/layerwise/mmc-local-device-rdma.conf
#    ock.mmc.local_service.protocol = device_rdma
#    ock.mmc.local_service.dram.size = 8GB（device 协议自动对齐 1GB）

# ③ 单进程初始化验证
#    SMEM/hybm/smem_bm_init 全部成功，但最终 RaRdevInit failed：
#    [HYBM device_rdma_transport_manager.cpp:564] RaRdevInit failed.
#    根因：HCCP RoCE 设备初始化需要物理链路

# ④ 链路状态证据
hccn_tool -i <N> -link -g    # 8 张卡全部 "link status: DOWN"
rdma link show               # hns_0/1/3 DOWN，hns_2 ACTIVE 但走的是业务网线(1Gbps)
hccn_tool -i 0 -ping -g address 192.168.100.12   # 100% 丢包
```

如后续机房接好 RoCE 线（100G 口入交换机），复跑步骤：
1. 确认 `hccn_tool -i N -link -g` 为 UP、卡间 ping 通
2. 启动 MetaService（同第 5 节）
3. `export MMC_LOCAL_CONFIG_PATH=/home/lsy/layerwise/mmc-local-device-rdma.conf` 后按第 6 节流程启动压测

其他备选协议：`host_rdma` / `host_tcp`（需 hcom 库）、`device_sdma`（HCCS 域内 SDMA，不依赖外部网络）。

---

## 10. 踩坑记录与注意事项

| # | 坑 | 现象 | 解决 |
|---|---|---|---|
| 1 | 容器 `--ipc=shareable` | `Insufficient space in /dev/shm: 160 MiB required, 64 MiB free` | `docker commit` 后用 `--ipc=host` 重建 |
| 2 | 宿主机编译的 wheel GLIBC 过高 | `GLIBC_2.38 not found (libmf_hybm_core.so)` | 用 npu 容器内 manylinux 版打包替换 |
| 3 | vllm 0.27.1 依赖缺失 | `ImportError: normalize_tool_choice from xgrammar` | `pip install -r /home/vllm/requirements/common.txt` |
| 4 | MetaService 未启动 | SMEM `Initialize attempt 5/5 failed`（connect 127.0.0.1:6000 失败） | 宿主机启动 `mmc_meta_service`（第 5 节） |
| 5 | MetaService 锁文件 | 重启报 `process lock file is locked` | `pkill -9` + 删 `/tmp/mmc_meta_service.lock` |
| 6 | rank 配置一致性 | `CheckRankConfigConsistency: maxDRAMSize 1G != 8G` | 配置文件与代码里 dram/max_dram 保持一致，或重启 MetaService |
| 7 | PowerShell→ssh 引号破坏 | 命令里字母 `r` 被吃掉、`|` 变管道 | 复杂命令写入脚本文件，`cat xx.sh \| ssh "tr -d '\r' \| bash"` 执行 |
| 8 | 场景切换残留 | 上一场景 vllm 进程占 HBM/端口 | 每次启动前 kill + 等 10s + 核对 npu-smi |
| 9 | layerwise 强制降级 cudagraph | 日志 WARNING `Overriding cudagraph_mode from FULL_AND_PIECEWISE to PIECEWISE` | 正常现象，非错误 |
| 10 | sshd 禁用 TcpForwarding | VS Code Remote 报 `administratively prohibited` | 修改 `/etc/ssh/sshd_config` 的 `AllowTcpForwarding yes` 后 `systemctl reload sshd` |

---

## 11. 常用排查命令速查

```bash
# NPU 状态 / HBM 占用
npu-smi info | grep -E "HBM-Usage|Process"

# vllm 进程
pgrep -f "vllm.entrypoints" | wc -l

# 服务日志（实时）
tail -f /home/lsy/npu2_log_layerwise_on.log

# 端口就绪
netstat -tlnp | grep 8004

# MetaService 状态
ps -ef | grep mmc_meta | grep -v grep
netstat -tlnp | grep -E ':5000 |:6000 '

# 手动发一条请求验证
curl -s http://127.0.0.1:8004/v1/models | head -c 300

# 容器内快速自检
docker exec lsy_pd bash -c '
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh;
export PYTHONPATH=/home/vllm:/home/vllm-ascend;
/usr/local/python3.11.15/bin/python3 -c "import vllm, memcache_hybrid; print(\"ENV OK\")"'

# 全部停止（释放 NPU）
docker exec lsy_pd bash -c '
for pid in $(pgrep -f "vllm.entrypoints"); do kill -9 $pid; done'
```
