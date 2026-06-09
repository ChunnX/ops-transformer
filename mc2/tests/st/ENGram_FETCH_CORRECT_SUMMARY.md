# EngramFetch 测试用例生成 - 正确版本总结

## 一、关键修正：Input/Output 定义

### 1.1 第一版的问题

**第一版错误的定义**：
```python
input_tensor_shape: [[num_tokens], [num_tokens, hidden]]
input_tensor_dtype: ['int32', 'bf16']
input_tensor_range: [(0, max_index), (-5, 5)]
```

**问题**：将 `indices` 和 `fetched` 都当作输入tensor

### 1.2 根据文档的正确理解

从 `tests/elastic/fanchuan.md` 第2.1章节：

**ACLNN接口定义**：
```
aclnnEngramFetchGetWorkspaceSize
  入参: bufferHandle, indices, fetched
  出参: workspaceSize, executor

其中：
- bufferHandle: aclElasticBufferHandle (不透明host指针，不是tensor)
- indices: aclTensor* → 输入tensor
- fetched: aclTensor* → 输出tensor（fetch落地buffer）
```

**正确理解**：
- **输入tensor**：`indices` (num_tokens,) int32 - 全局索引
- **输出tensor**：`fetched` (num_tokens, hidden) bf16 - RDMA获取的数据
- **bufferHandle**：构造参数，不是tensor

### 1.3 正确版本的定义

**正确版本**：
```python
# Input tensor (仅indices)
input_tensor_shape: [[num_tokens]]
input_tensor_dtype: ['int32']
input_tensor_range: [(0, num_entries * world_size - 1)]

# Output tensor (fetched)
output_tensor_shape: [[num_tokens, hidden]]
output_tensor_dtype: ['bf16']
```

**indices的range解释**：
- `(0, num_entries × world_size - 1)` 
- 全局索引范围：覆盖所有rank的所有entry
- 例如：world_size=8, num_entries=524288 → range=(0, 4194303)

## 二、完整的xlsx列字段定义

### 2.1 标准列（按xlsx-format-spec.md）

| 字段名 | 编码格式 | 说明 | 示例 |
|--------|----------|------|------|
| **input_tensor_shape** | Python list of lists | indices的shape | `[[1024]]` |
| **input_tensor_dtype** | Python list of strings | indices的dtype | `['int32']` |
| **input_tensor_range** | Python list of tuples | indices的取值范围 | `[(0, 4194303)]` |
| **output_tensor_shape** | Python list of lists | fetched的shape | `[[1024, 256]]` |
| **output_tensor_dtype** | Python list of strings | fetched的dtype | `['bf16']` |

### 2.2 构造参数列（六维TilingKey）

| 字段名 | 类型 | 说明 | 取值范围 |
|--------|------|------|----------|
| **num_qps** | int | NUM_QPS_TIER | {1,4,8,16,32} |
| **hidden** | int | HIDDEN_TIER | {64..4096} |
| **num_peers** | int | NUM_PEERS_TIER | {1,2,4,8,16} |
| **ranks_per_peer** | int | RANKS_PER_PEER_TIER | {1,2,4,8} |
| **world_size** | int | 计算值 | num_peers × ranks_per_peer |
| **num_entries** | int | ENTRIES_TIER | {2^12..2^20} |
| **num_tokens** | int | indices数量 | {32..8192} |
| **hybrid_mode** | int | HYBRID_MODE | {0,1} |

### 2.3 测试分类列

| 字段名 | 说明 | 取值 |
|--------|------|------|
| **template_type** | 模板分类 | template_1_flat, template_2_hybrid, template_3_large_hidden |
| **test_level** | 测试层级 | level0, level1 |
| **test_type** | 测试类型 | single_factor, two_factor_combo, random_combo |
| **factor_name** | 因子名称 | 如 NUM_QPS_TIER, HIDDEN × HYBRID_MODE |
| **seed** | 随机种子 | 整数 |

## 三、生成文件对比

### 3.1 三个版本的xlsx文件

| 文件 | Input定义 | 问题 | 状态 |
|------|----------|------|------|
| `engram_fetch.xlsx` | indices + fetched作为input | ❌ 错误 | 已废弃 |
| `engram_fetch_tiling.xlsx` | num_peers/ranks_per_peer作为参数 | ⚠️ 缺少input/output tensor定义 | 已废弃 |
| **`engram_fetch_correct.xlsx`** | 仅indices作为input，fetched作为output | ✅ 正确 | **推荐使用** |

### 3.2 正确版本的生成结果

**文件**：`result/excel/engram_fetch_correct.xlsx`

**统计**：
- Level0: 38条（≤200 ✓）
- Level1: 500条（500~700 ✓）

**模板分布**：
- template_1_flat: Level0=34, Level1=164
- template_2_hybrid: Level0=1, Level1=156
- template_3_large_hidden: Level0=3, Level1=180

## 四、关键改进点总结

### 4.1 Input/Output Tensor 正确性

✅ **第一版错误**：将fetched当作输入tensor
✅ **正确版本**：indices是输入，fetched是输出

### 4.2 indices的range定义

✅ **正确理解**：
```python
indices_range = (0, num_entries × world_size - 1)
```

**推导**：
- 全局表总行数 = num_ranks × num_entries
- world_size = num_peers × ranks_per_peer
- 最大索引 = num_entries × world_size - 1

**示例**：
- world_size=8, num_entries=524288
- indices ∈ [0, 4194303]

### 4.3 六维TilingKey全覆盖

✅ **保持了全覆盖**：
- NUM_QPS_TIER: 5档 → Level0全覆盖
- HIDDEN_TIER: 7档 → Level0全覆盖
- NUM_PEERS_TIER: 5档 → Level0全覆盖
- RANKS_PER_PEER_TIER: 4档 → Level0全覆盖
- ENTRIES_TIER: 8档 → Level0全覆盖
- HYBRID_MODE: 2档 → Level0全覆盖
- NUM_TOKENS: 7档 → Level0全覆盖

### 4.4 三种模板全覆盖

✅ **模板验证**：
- template_1_flat (HYBRID_MODE=0): 已覆盖
- template_2_hybrid (HYBRID_MODE=1): 已覆盖
- template_3_large_hidden (HIDDEN≥1024): 已覆盖

### 4.5 符合xlsx-format-spec.md规范

✅ **列字段规范**：
- input_tensor_shape: 嵌套Python list
- input_tensor_dtype: 单引号string list
- input_tensor_range: tuple list
- output_tensor_shape: 嵌套Python list
- output_tensor_dtype: 单引号string list

## 五、用例示例

### 5.1 Level0典型用例

```python
case_name: 'qps8_h256_tok1024_e65536_ID000'

# Input tensor
input_tensor_shape: [[1024]]
input_tensor_dtype: ['int32']
input_tensor_range: [(0, 524287)]

# Output tensor
output_tensor_shape: [[1024, 256]]
output_tensor_dtype: ['bf16']

# 构造参数
num_qps: 8
hidden: 256
num_peers: 4
ranks_per_peer: 2
world_size: 8
num_entries: 65536
num_tokens: 1024
hybrid_mode: 0

template_type: 'template_1_flat'
test_level: 'level0'
test_type: 'single_factor'
factor_name: 'NUM_QPS_TIER'
```

**解释**：
- indices: 1024个随机索引，范围[0, 524287]（8 ranks × 65536 entries）
- fetched: 1024行 × 256维，bf16类型
- 8个QP，flat模式（HYBRID_MODE=0）

### 5.2 Level1典型用例（模板切换）

```python
case_name: 'h1024_hybrid0_template_ID014'

# Input
input_tensor_shape: [[1024]]
input_tensor_dtype: ['int32']
input_tensor_range: [(0, 131071)]

# Output
output_tensor_shape: [[1024, 1024]]
output_tensor_dtype: ['bf16']

# 参数
num_qps: 8
hidden: 1024  # ≥1024触发模板三
num_peers: 4
ranks_per_peer: 2
world_size: 8
num_entries: 65536
num_tokens: 1024
hybrid_mode: 0

template_type: 'template_3_large_hidden'
test_level: 'level1'
test_type: 'two_factor_combo'
factor_name: 'HIDDEN × HYBRID_MODE'
```

**解释**：
- HIDDEN=1024触发模板三（大hidden）
- 验证每warp降为1token的切分逻辑
- indices范围扩大：[0, 131071]（更大entries）

## 六、测试执行验证

### 6.1 Input Tensor 验证

**indices tensor生成**：
```python
indices = torch.randint(0, num_entries * world_size, 
                        (num_tokens,), device='cuda', dtype=torch.int)
```

**示例**：
- num_entries=65536, world_size=8, num_tokens=1024
- indices = torch.randint(0, 524288, (1024,), device='cuda', dtype=torch.int)
- 生成1024个随机全局索引

### 6.2 Output Tensor 验证

**fetched tensor**：
```python
fetched = buffer.engram_fetch(indices)()
# fetched.shape = (num_tokens, hidden) = (1024, 256)
# fetched.dtype = torch.bfloat16
```

**正确性验证**：
```python
global_storage = torch.empty((world_size * num_entries, hidden), 
                             dtype=torch.bfloat16, device='cuda')
ref_fetched = global_storage[indices]
assert torch.equal(ref_fetched, fetched), "Data inconsistency"
```

## 七、与第一版对比表

| 对比项 | 第一版（错误） | 正确版本 |
|--------|-------------|---------|
| **input_tensor_shape** | [[num_tokens], [num_tokens, hidden]] | [[num_tokens]] |
| **input_tensor_dtype** | ['int32', 'bf16'] | ['int32'] |
| **input_tensor_range** | [(0, max), (-5, 5)] | [(0, max)] |
| **output_tensor_shape** | 无定义 | [[num_tokens, hidden]] |
| **output_tensor_dtype** | ['bf16'] | ['bf16'] |
| **indices理解** | 输入tensor（正确） | 输入tensor（正确） |
| **fetched理解** | ❌当作输入 | ✅当作输出 |
| **tensor数量** | 2个input | 1个input + 1个output |

## 八、总结

### 8.1 核心修正

✅ **最关键的修正**：
- **indices**：输入tensor (num_tokens,) int32
- **fetched**：输出tensor (num_tokens, hidden) bf16
- **不再将fetched当作输入**

### 8.2 符合规范

✅ **符合xlsx-format-spec.md**：
- input_tensor_* 字段仅包含indices
- output_tensor_* 字段包含fetched
- 维度一致性：1个输入tensor + 1个输出tensor

### 8.3 保持全覆盖

✅ **保持六维TilingKey全覆盖**：
- 所有31个档位Level0全覆盖
- 三种模板全部触发验证
- 关键组合Level1覆盖

### 8.4 推荐使用

**推荐使用文件**：
- ✅ `result/excel/engram_fetch_correct.xlsx`（正确版本）
- ❌ `result/excel/engram_fetch.xlsx`（第一版，已废弃）
- ❌ `result/excel/engram_fetch_tiling.xlsx`（缺少input/output定义）

---

**生成时间**: 2026-06-08
**生成脚本**: `result/generate_engram_fetch_correct.py`
**输出文件**: `result/excel/engram_fetch_correct.xlsx`
**关键修正**: Input/Output tensor定义正确，符合文档和规范