# EngramWrite 算子测试设计总结

## 一、算子定位

**算子名称**：engram_write

**核心功能**：将本 rank 的表分片写入 host pinned 共享段，作为 engram_fetch 的前置操作。

**重要说明**：engram_write 是 **host runtime 操作**，**无独立 ACLNN 接口**。

## 二、算子特点

### 2.1 执行流程

```
engram_write(storage):
    ① HeavyBarrier(with_cpu_sync=True)
       ├─ aclrtSynchronizeDevice()     ← 等本 device 所有 stream 静止
       ├─ HcclBarrier(comm, stream)     ← 跨 rank barrier kernel
       └─ aclrtSynchronizeDevice()      ← CPU 阻塞至 barrier 完成
    ② memcpy(pinned_dst + cpu_offset, host_src, nbytes)  ← H→H
       cpu_offset = hybrid ? scaleup_rank_idx × num_cpu_buffer_bytes : 0
    ③ HeavyBarrier(with_cpu_sync=True)   ← 同 ①
```

### 2.2 关键特性

| 特性 | 说明 |
|------|------|
| **无独立 kernel** | 纯内存拷贝，计算密度 = 0 |
| **H→H memcpy** | host → host pinned DRAM |
| **两道 Barrier** | pre-barrier + post-barrier |
| **cpu_offset 计算** | hybrid 模式下按 scaleup_rank_idx 偏移 |
| **可跳过 memcpy** | 权重直接加载到 pinned 段时，仅 barrier |

### 2.3 为什么无独立 ACLNN 接口？

1. 计算密度 = 0，纯内存拷贝
2. 无需 tiling 切分（一次 memcpy 即可）
3. `aclrtMemcpyAsync` + `HcclBarrier` 已是公开 API
4. 封装 ACLNN 无额外价值

## 三、参数分析

### 3.1 Tensor 参数

| 参数名 | Shape | Dtype | Memory Location | 说明 |
|--------|-------|-------|----------------|------|
| storage | (num_entries, hidden) | bf16 | host memory | 本 rank 的表分片 |

### 3.2 构造参数（影响算子行为）

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|----------|------|
| num_entries | int | [4096, ..., 524288] | 每个 rank 的表行数 |
| hidden | int | [64, 128, ..., 4096] | 隐藏层维度 |
| world_size | int | [2, 4, 8, 16, 32] | 通信域大小 |
| allow_hybrid_mode | bool | 0/1 | 是否使用 hybrid 模式（影响 cpu_offset） |
| copy_flag | bool | 0/1 | 是否执行 memcpy（0=跳过，仅 barrier） |

### 3.3 输出

engram_write **无输出 tensor**，数据写入到 ElasticBuffer 的 host pinned 段。

## 四、约束分析

### 4.1 数据类型约束

- **dtype_constraint**：storage 仅支持 bfloat16

### 4.2 Shape 约束

- storage shape = (num_entries, hidden)

### 4.3 内存布局约束

- **cpu_offset 计算**：
  - flat 模式：cpu_offset = 0
  - hybrid 模式：cpu_offset = scaleup_rank_idx × num_cpu_buffer_bytes

### 4.4 并发约束

- **Write 必须先于第一次 Fetch**
- **Write 进行中不允许未完成的 Fetch**
- pre-barrier 保证在飞的 fetch 已读完 pinned 段
- post-barrier 保证全网写完才放行后续 fetch

### 4.5 存在性约束

- 权重可直接加载到已注册的 pinned 段（跳过 memcpy）
- 这种情况下 engram_write 退化为仅前后两道 barrier

## 五、测试因子提取

### 5.1 主要测试因子

| 因子类型 | 来源 | Level0 覆盖 | Level1 覆盖 |
|----------|------|------------|------------|
| dtype_factor | storage dtype | bf16 全覆盖 | bf16 × shape 组合 |
| shape_factor | num_entries/hidden | 边界值 | 两两组合 |
| world_size_factor | 通信域大小 | 主要值 | world_size × hybrid 组合 |
| hybrid_mode_factor | bool 标志 | True/False | hybrid × cpu_offset 组合 |
| copy_flag_factor | bool 标志 | True/False | copy × hybrid 组合 |

### 5.2 边界值定义

**num_entries 边界**：
- 最小值：4096 (2^12)
- 最大值：524288 (2^19)
- 中间值：16384, 65536, 262144

**hidden 边界**：
- 最小值：64
- 最大值：4096
- 中间值：128, 256, 512, 1024, 2048

**world_size 边界**：
- 最小值：2
- 典型值：4, 8, 16
- 大值：32

**hybrid_mode 边界**：
- 0: flat 模式（cpu_offset = 0）
- 1: hybrid 模式（cpu_offset 按节点内 rank 偏移）

**copy_flag 边界**：
- 0: 跳过 memcpy，仅 barrier
- 1: 正常 memcpy + barrier

## 六、测试用例设计

### 6.1 Level0 用例（基础覆盖）

**设计原则**：单因子独立覆盖，用例数 ≤ 200

**当前设计**：43 条用例

**覆盖内容**：
1. storage dtype 覆盖（bf16 基础组合） - 16 条
2. num_entries 边界值覆盖 - 3 条
3. hidden 边界值覆盖 - 7 条
4. world_size 主要值覆盖 - 5 条
5. hybrid_mode 标志覆盖 - 2 条
6. copy_flag 标志覆盖 - 2 条
7. world_size × hybrid_mode 组合（cpu_offset 验证） - 8 条

### 6.2 Level1 用例（两两组合）

**设计原则**：两两因子组合覆盖，用例数 500~700

**当前设计**：500 条用例

**组合重点**：
1. **num_entries × hidden 组合**（存储容量验证） - 35 条
   - 验证不同存储容量的写入正确性
2. **world_size × allow_hybrid_mode 组合**（cpu_offset 计算验证） - 10 条
   - 验证不同分布式拓扑下的 offset 计算
3. **num_entries × world_size 组合**（分布式规模验证） - 25 条
   - 验证大规模分布式写入的正确性
4. **allow_hybrid_mode × copy_flag 组合**（关键场景） - 4 条
   - 验证 hybrid 模式下跳过 memcpy的场景
5. **hidden × world_size 组合**（性能验证） - 35 条
   - 验证不同 hidden 维度的性能表现

### 6.3 特殊用例

**关键场景**：
- **copy_flag=0**：权重直接加载到 pinned 段，仅 barrier
- **hybrid_mode=1**：验证 cpu_offset 的正确计算
- **world_size × hybrid_mode 组合**：验证跨节点拓扑的适配

## 七、列字段定义

### 7.1 Tensor 列字段

| 字段名 | 编码格式 | 说明 |
|--------|----------|------|
| storage_shape | Python list of lists | [[num_entries, hidden]] |
| storage_dtype | Python list of strings | ['bf16'] |
| storage_range | Python list of tuples | [(-5, 5)] |

### 7.2 算子特定列字段

| 字段名 | 编码格式 | 说明 |
|--------|----------|------|
| num_entries | 整数 | 每个 rank 的表行数 |
| hidden | 整数 | 隐藏层维度 |
| world_size | 整数 | 通信域大小 |
| allow_hybrid_mode | 0/1 | hybrid 模式标志 |
| copy_flag | 0/1 | 是否执行 memcpy |
| seed | 整数 | 随机种子 |

## 八、测试执行建议

### 8.1 测试策略

由于 engram_write 无独立 ACLNN 接口，测试主要通过：

**方式一：engram_write + engram_fetch 组合测试**
```python
# 1. 构造 buffer
buffer = ElasticBuffer(...)

# 2. 写入数据
buffer.engram_write(storage)

# 3. 验证写入正确性（通过 fetch）
indices = torch.arange(0, num_entries, device='cuda', dtype=torch.int)
hook = buffer.engram_fetch(indices)
fetched = hook()

# 4. 验证数据一致性
assert torch.equal(storage, fetched), "engram_write failed"
```

**方式二：直接验证 host pinned 段数据**
```python
# 1. 构造 buffer
buffer = ElasticBuffer(...)

# 2. 写入数据
buffer.engram_write(storage)

# 3. 直接读取 host pinned 段
pinned_data = buffer.get_host_pinned_data()

# 4. 验证数据一致性
assert torch.equal(storage, pinned_data), "engram_write failed"
```

### 8.2 Barrier 正确性验证

```python
# 验证 pre-barrier 和 post-barrier 的正确性

# 测试1：并发 write 不允许
buffer1.engram_write(storage1)
# 此时 buffer2.engram_write(storage2) 应被阻塞

# 测试2：write 进行中不允许 fetch
buffer.engram_write(storage)
# 此时 buffer.engram_fetch(indices) 应被阻塞
```

### 8.3 cpu_offset 验证

```python
# hybrid 模式下验证 cpu_offset 计算

# 获取每个 rank 的 cpu_offset
for rank_idx in range(num_scaleup_ranks):
    expected_offset = rank_idx * num_cpu_buffer_bytes
    actual_offset = buffer.get_cpu_offset(rank_idx)
    assert actual_offset == expected_offset, f"cpu_offset error at rank {rank_idx}"
```

## 九、检查清单验证

生成 xlsx 后已验证：

- [x] 包含 level0 和 level1 sheets
- [x] Level0 用例数 ≤ 200（实际：43）
- [x] Level1 用例数 500~700（实际：500）
- [x] 所有 shape/dtype/range 可 eval() 解析
- [x] case_name 遵循命名规范
- [x] 布尔值使用 0/1 编码

## 十、与 engram_fetch 的协同测试

### 10.1 协同测试场景

**场景一：基础协同**
```python
# 1. write
buffer.engram_write(storage)

# 2. fetch（验证 write 正确性）
indices = torch.randint(0, num_entries * world_size, (num_tokens,), device='cuda')
hook = buffer.engram_fetch(indices)
fetched = hook()

# 3. 验证
ref_fetched = global_storage[indices]
assert torch.equal(ref_fetched, fetched), "write-fetch协同失败"
```

**场景二：hybrid 模式协同**
```python
# hybrid 模式下验证跨节点写入和读取

buffer = ElasticBuffer(allow_hybrid_mode=True)
buffer.engram_write(storage)

# 从不同节点 rank fetch 数据
indices = torch.randint(0, num_entries * world_size, (num_tokens,), device='cuda')
fetched = buffer.engram_fetch(indices)()
```

**场景三：跳过 memcpy 协同**
```python
# 权重直接加载到 pinned 段
pinned_storage = buffer.get_host_pinned_segment()
load_weights_to_pinned(pinned_storage)

# 仅执行 barrier
buffer.engram_write(None, copy_flag=False)

# 立即 fetch
fetched = buffer.engram_fetch(indices)()
```

### 10.2 协同测试用例（可选）

可创建额外的协同测试 sheet：

| Sheet名 | 用途 | 用例数 |
|----------|------|--------|
| write_fetch_combo | write+fetch 协同测试 | 100~200 |

## 十一、输出文件

**文件路径**：`result/excel/engram_write.xlsx`

**生成时间**：2026-06-08

**用例统计**：
- Level0: 43 条
- Level1: 500 条

---

**设计依据**：tests/elastic/fanchuan.md（EngramWrite 算子设计文档）

**参考规范**：
- xlsx-format-spec.md
- operator-design-patterns.md

**生成工具**：result/generate_engram_write_cases.py

**重要提示**：
- engram_write 是 host runtime 操作，无独立 ACLNN 接口
- 测试需要通过 engram_write + engram_fetch 的组合验证
- 重点验证 Barrier 正确性和 cpu_offset 计算