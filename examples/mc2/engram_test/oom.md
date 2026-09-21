# EngramFetch OOM 问题分析报告

## 1. 概述

本文档总结了 EngramFetch 算子测试过程中遇到的所有 OOM（Out of Memory）问题，包括历史 OOM 事件和最新的 16x tensor 场景 OOM 根因分析。

**测试环境**：
- 服务器：141.61.33.14 / 141.63.88.80
- NPU：Ascend950DT，16 卡，每卡 91.75 GB HBM
- CANN：9.2.0
- 通信库：HCCL
- PyTorch：torch_npu，ACL Graph 模式（`is_graph=1`, `dynamic=False`）

---

## 2. 历史 OOM 事件

### 2.1 Ninja 编译 OOM（CPU 内存，exit code 137）

**场景**：`torch_extensions` JIT 编译 C++ 扩展（如 `block_sparse_attention`）时，进程被 OOM killer 杀死（exit code 137）。

**根因**：ninja 并行编译多个 `.o` 文件时 CPU 内存不足。编译被 kill 后残留 `.lock` 文件，导致下次运行重复触发编译，形成 OOM 死循环。

**关键信号**：
- `FAILED: [code=137] block_sparse_attention.o`
- `ninja: build stopped: subcommand failed.`
- `~/.cache/torch_extensions/<ext_name>/lock` 文件残留

**解决方案**：
1. 删除 stale lock：`rm -f ~/.cache/torch_extensions/py311_cpu/*/lock`
2. 低负载下预编译：`MAX_JOBS=1 python -c "import ..."` 确保编译完成
3. 确认 `.so` 文件生成后再启动测试

### 2.2 HCCL 端口冲突（非 OOM，但导致测试失败）

**场景**：前一次测试异常退出后残留进程占用 HCCL 通信端口（MASTER_PORT=50008），导致后续测试 `Acquire HCCL channel failed, ret: 4`。

**解决方案**：启动前 `pkill -f <test_script>` 清理残留进程，或使用不同 MASTER_PORT。

### 2.3 NPU Stream 创建失败（error 507899，共享卡资源耗尽）

**场景**：多用户共享 NPU 时，ACL Graph 编译创建 Stream 失败，报 `halSqCqAllocate failed, drvRetCode=-1`。

**根因**：NPU 驱动对每张卡的 Stream/CQ 数量有上限，多用户或大量图编译导致驱动资源耗尽。

**解决方案**：切换到空闲卡，或减少不同 shape 的图编译次数。

---

## 3. 16x Tensor NPU OOM 根因分析（本次重点）

### 3.1 故障现象

**测试用例**：`engram_tensor` sheet，Case 5-6（16x tensor）

| 参数 | 值 |
|------|-----|
| input_tensor_shape | `[[321125, 9216]]` |
| input_tensor_dtype | `fp8_e4m3fn`（1 byte/element） |
| fetch_indicies | `[16, 32, ..., 8388608, 16777216]`（21 种 token count） |
| world_size | 4 |
| is_graph | 1（ACL Graph 模式） |
| use_fp8 | 1 |
| fetch_mode | local / sliced（均失败） |

**错误信息**：
```
NPU out of memory. Tried to allocate 72.00 GiB
(NPU 0; 91.75 GiB total capacity;
 76.94 GiB already allocated;
 76.94 GiB current active;
 13.03 GiB free;
 77.47 GiB reserved in total by PyTorch)

NPUGraph: ERROR — capture failed: OutOfMemoryError
```

所有 4 个 rank 同时 OOM，产生 `core.python.*` core dump 文件。

### 3.2 测试用例对比

三种 tensor shape 的表总元素数相同（~2.96B），但 hidden 维度不同：

| Case | numEntries | hiddenSize | 表大小 = numEntries × hidden × 1B | 峰值 HBM | 状态 |
|------|-----------|------------|----------------------------------|----------|------|
| 1x | 5,138,000 | 576 | 2.96 GB | 19,569 MB | SUCCESS |
| 4x | 12,845,000 | 2,304 | 29.6 GB | 79,768 MB | SUCCESS |
| 16x | 321,125 | 9,216 | 2.96 GB | — | **OOM** |

表大小相同但 16x OOM，说明问题不在表存储，而在**输出张量**。

### 3.3 OOM 触发点

**文件**：`ops-transformer2/mc2/common/torch_extension/csrc/elastic_buffer.cpp:1894`

```cpp
auto fetched = at::empty({numTokens, hiddenSize},
    at::TensorOptions().dtype(dtype).device(indices.device()));
```

**计算公式**：

```
fetched_bytes = numTokens × hiddenSize × elementSize(dtype)
```

- `numTokens`：当前迭代的 fetch token count
- `hiddenSize`：表的隐藏维度
- `elementSize(fp8_e4m3fn)` = 1 byte

**具体计算**（numTokens=8,388,608 时）：

| Case | hiddenSize | 计算 | fetched 大小 |
|------|------------|------|-------------|
| 1x | 576 | 8,388,608 × 576 × 1 = 4,831,838,208 B | **4.5 GiB** |
| 4x | 2,304 | 8,388,608 × 2,304 × 1 = 19,327,352,832 B | **18.0 GiB** |
| 16x | 9,216 | 8,388,608 × 9,216 × 1 = 77,309,411,328 B | **72.0 GiB** |

OOM 时 NPU 仅有 13.03 GiB 空闲，无法满足 72 GiB 的分配请求。

### 3.4 ACL Graph 累积效应

测试使用 `is_graph=1` 且 `dynamic=False`，意味着**每个 token size 独立捕获一个 ACL Graph**。之前 20 个较小 token size 的 graph pool 不会释放，累积占用约 77 GiB：

```
已分配 76.94 GiB = 表存储(~3GB) + sf_table(~0.1GB)
                 + 前 20 个 token size 的 graph pool 累积(~70GB+)
尝试分配 72.00 GiB = fetched 输出张量（token=8M）
─────────────────────────────────────────────
合计 ≈ 149 GiB > 91.75 GiB 总容量 → OOM
```

### 3.5 所有随 numTokens 增长的 NPU 显存分配

| # | 文件:行号 | 张量 | 大小公式 | dtype | 16x token=8M 时大小 |
|---|----------|------|---------|-------|---------------------|
| 1 | `elastic_buffer.cpp:1894` | `fetched`（推理输出） | `numTokens × hiddenSize × 1B` | fp8 | **72.0 GiB** |
| 2 | `elastic_buffer.py:893` | `fetched_sf`（scaling factor 输出） | `numTokens × ⌈hiddenSize/32⌉ × 1B` | e8m0 | **2.25 GiB** |
| 3 | `aclnn_common.h:1080` | ACLNN workspace（内核工作空间） | 不透明，上限≈`numTokens × hiddenSize × dtypeSize` | bytes | **~72 GiB**（上限） |
| 4 | `egram_test.py:617` | `a = torch.mul(indices, 1)` | `numTokens × 4B` | int32 | 32 MiB |
| 5 | `egram_test.py:709` | `indices_npu`（输入索引） | `numTokens × 4B` | int32 | 32 MiB |

**训练路径额外分配**（当前测试不走训练，但列出供参考）：

| # | 文件:行号 | 张量 | 大小公式 | dtype |
|---|----------|------|---------|-------|
| 6 | `elastic_buffer.cpp:1915` | `fetched`（训练前向输出） | `numTokens × hiddenSize × dtypeSize` | user dtype |
| 7 | `elastic_buffer.cpp:1921` | `perm`（排列索引） | `numTokens × 4B` | int32 |
| 8 | `elastic_buffer.cpp:1960` | `gradUnique`（梯度去重） | `(numMaxTokensPerRank × rankSize) × hiddenSize × dtypeSize` | user dtype |
| 9 | `elastic_buffer.cpp:1961` | `uniqueLocalEntry` | `(numMaxTokensPerRank × rankSize) × 4B` | int32 |

**关键发现**：
- `fetched`（72GB）是 OOM 主因，单次分配就几乎占满整卡
- `fetched_sf`（2.25GB）是 fp8 专用，随 `numTokens × hiddenSize/32` 增长，比 `fetched` 小 32 倍但不可忽略
- ACLNN workspace 大小不透明，上限与 `fetched` 相当，在 Graph 模式下被 pool 捕获并驻留
- 训练路径的 `gradUnique` 同样是 hidden 维度放大的重灾区

### 3.6 16x 各 token size 的 `fetched` 大小

| numTokens | fetched 大小 (hidden=9216, fp8) | 累积 graph pool 估算 |
|-----------|-------------------------------|---------------------|
| 16 | 1.4 MiB | ~3 GB（表+sf） |
| 32 | 2.8 MiB | ~3 GB |
| 64 | 5.6 MiB | ~3 GB |
| 128 | 11.3 MiB | ~3 GB |
| 256 | 22.5 MiB | ~3 GB |
| 512 | 45.0 MiB | ~3.1 GB |
| 1,024 | 90.0 MiB | ~3.2 GB |
| 2,048 | 180.0 MiB | ~3.4 GB |
| 4,096 | 360.0 MiB | ~3.8 GB |
| 8,192 | 720.0 MiB | ~4.6 GB |
| 16,384 | 1.4 GiB | ~6.2 GB |
| 32,768 | 2.8 GiB | ~9.4 GB |
| 65,536 | 5.6 GiB | ~15.8 GB |
| 131,072 | 11.3 GiB | ~28.6 GB |
| 262,144 | 22.5 GiB | ~53.6 GB |
| 524,288 | 45.0 GiB | ~102 GB → **已超限** |
| 1,048,576 | 90.0 GiB | OOM |
| 2,097,152 | 180.0 GiB | OOM |
| 4,194,304 | 360.0 GiB | OOM |
| 8,388,608 | **72.0 GiB** | OOM（实际触发点） |
| 16,777,216 | 144.0 GiB | OOM |

> 注：实际 OOM 在 token=8,388,608 时触发（72GB），但 token=524,288 时累积 graph pool 已可能超限。Graph pool 中除了 `fetched` 还包含 workspace 和其他临时张量，实际累积比表中估算更大。

---

## 4. 修改建议

### 4.1 C++ 层预检查（`elastic_buffer.cpp`, line 1894 之前）

```cpp
// 在分配 fetched 之前检查可用显存
int64_t fetchedBytes = numTokens * hiddenSize * at::elementSize(dtype);
size_t freeBytes = 0, totalBytes = 0;
aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes);
TORCH_CHECK(fetchedBytes < static_cast<int64_t>(freeBytes * 0.8),
    "engram_fetch output tensor (", fetchedBytes / 1024.0 / 1024.0 / 1024.0,
    " GiB) would exceed 80% of available NPU memory (",
    freeBytes / 1024.0 / 1024.0 / 1024.0,
    " GiB free). Consider reducing numTokens or hiddenSize.");
```

### 4.2 Python 层预检查（`elastic_buffer.py`, `_check_engram_fetch_ready` 中）

```python
fetched_bytes = indices.size(0) * self._engram_hidden_size * self._engram_dtype.itemsize
free_bytes, total_bytes = torch.npu.mem_get_info()
if fetched_bytes > free_bytes * 0.8:
    raise RuntimeError(
        f"engram_fetch output ({fetched_bytes / 1024**3:.1f} GiB) would exceed "
        f"80% of available NPU memory ({free_bytes / 1024**3:.1f} GiB free). "
        f"numTokens={indices.size(0)}, hidden={self._engram_hidden_size}, "
        f"dtype={self._engram_dtype}. Reduce numTokens or hiddenSize."
    )
```

### 4.3 分块 fetch（Chunked Fetch）

对大 token count，将 indices 分块处理，逐块 fetch 后拼接，避免单次超大分配：

```python
def engram_fetch_chunked(self, indices, max_chunk_bytes=4 * 1024**3):
    """分块 fetch，避免单次分配超过 max_chunk_bytes"""
    elem_bytes = self._engram_dtype.itemsize
    max_tokens = max_chunk_bytes // (self._engram_hidden_size * elem_bytes)
    if indices.size(0) <= max_tokens:
        return self.engram_fetch(indices)
    chunks = []
    for start in range(0, indices.size(0), max_tokens):
        chunk = indices[start:start + max_tokens]
        chunks.append(self.engram_fetch(chunk))
    return torch.cat(chunks, dim=0)
```

### 4.4 ACL Graph 显存管理

- 在不同 token size 之间**释放前一个 graph pool**，避免累积：
  ```python
  torch.npu.empty_cache()
  ```
- 或使用 `dynamic=True` 让多个 token size 共享同一个 graph，减少 pool 数量

### 4.5 测试配置调整

根据 hiddenSize 和可用显存动态计算最大可测试 token count：

```python
free_bytes, _ = torch.npu.mem_get_info()
max_tokens = int(free_bytes * 0.7 / (hidden_size * dtype_size))
# 过滤掉超过 max_tokens 的 token size
valid_tokens = [t for t in token_sizes if t <= max_tokens]
```

---

## 5. 附录：性能数据

### 5.1 成功用例性能摘要

| Tensor Shape | Mode | 小 token (<512) | 中 token (1K~65K) | 大 token (>131K) | 峰值 HBM |
|-------------|------|-----------------|------------------|-----------------|----------|
| 1x (5.1M×576) | Local vs Sliced | Local 快 (2.3x) | 持平 (0.85~1.0x) | Sliced 快 (0.84~0.92x) | 19,569 MB |
| 4x (12.8M×2304) | Local vs Sliced | Local 快 (2.4x) | Sliced 快 (0.55~0.91x) | Local 快 (1.0~1.28x) | 79,768 MB |

### 5.2 图表文件

所有图表保存于 `D:\cann\engram2\engram_tensor\images\`：

| 文件 | 内容 |
|------|------|
| `perf_comparison_by_tensor.png` | 各 tensor shape 的 Local vs Sliced 性能对比 |
| `speedup_analysis.png` | Sliced/Local 加速比柱状图 |
| `memory_usage_over_time.png` | HBM 内存随时间变化 |
| `peak_memory_comparison.png` | 峰值内存柱状对比 |
| `perf_overview.png` | Local/Sliced 模式跨 shape 汇总 |
| `hbm_vs_token_by_tensor.png` | HBM vs token count（按 shape 分组） |
| `hbm_vs_token_local_vs_sliced.png` | HBM vs token count（Local vs Sliced） |
| `hbm_vs_token_all_cases.png` | 全部用例 HBM 汇总 |
| `engram_perf_comparison_1x.png` | 1x 4 面板详细对比 |
| `engram_perf_comparison_4x.png` | 4x 4 面板详细对比 |
| `cross_tensor_comparison.png` | 跨 shape 同 token 对比 |
| `perf_data_summary.json` | 结构化性能数据 |
