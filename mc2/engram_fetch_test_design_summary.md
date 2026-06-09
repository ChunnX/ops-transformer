# EngramFetch 算子测试设计总结

## 一、算子定位

**算子名称**：engram_fetch

**核心功能**：从分布式 ElasticBuffer 中通过 RDMA one-sided Get 远程抓取指定数据行到本地 device HBM。

**应用场景**：大规模 Engram 表（100B~1T params）分布式推理，支持跨节点 RDMA 数据获取。

## 二、参数分析

### 2.1 Tensor 参数

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| indices | (num_tokens,) | int32 | 全局索引，范围 ∈ [0, world_size × num_entries_per_rank) |
| fetched | (num_tokens, hidden) | bf16 | fetch 落地 buffer，位于 GPU HBM |

### 2.2 构造参数（影响算子行为）

| 参数名 | 类型 | 取值范围 | 说明 |
|--------|------|----------|------|
| num_entries | int | [4096, ..., 524288] | 每个 rank 的表行数 |
| hidden | int | [64, 128, ..., 4096] | 隐藏层维度 |
| num_tokens | int | [0, ..., 8192] | fetch 的 token 数量 |
| num_qps | int | [1, 4, 8, 16, 32] | QP 数量（构造时固定） |
| world_size | int | [2, 4, 8, 16, 32] | 通信域大小 |
| allow_hybrid_mode | bool | 0/1 | 是否使用 hybrid 模式 |

### 2.3 输出

| 参数名 | Shape | Dtype | 说明 |
|--------|-------|-------|------|
| fetched | (num_tokens, hidden) | bf16 | RDMA 获取的数据 |

## 三、约束分析

### 3.1 数据类型约束

- **dtype_constraint**：fetched/storage 仅支持 bfloat16，indices 为 int32
- **精度要求**：二进制一致（bit-exact），零容差

### 3.2 Shape 约束

- indices shape = (num_tokens,)
- fetched shape = (num_tokens, hidden)
- num_tokens 可为 0（合法边界值）

### 3.3 值约束

- indices ∈ [0, world_size × num_entries_per_rank - 1]
- 越界值行为未定义（调用方负责）

### 3.4 并发约束

- 同句柄不允许并发 fetch
- Wait 与 fetch 需在同一 stream
- Write 必须先于 Fetch

### 3.5 存在性约束

- Write 必须先于第一次 Fetch
- Write 进行中不允许未完成的 Fetch

## 四、测试因子提取

### 4.1 主要测试因子

| 因子类型 | 来源 | Level0 覆盖 | Level1 覆盖 |
|----------|------|------------|------------|
| dtype_factor | dtype 支持列表 | bf16 全覆盖 | bf16 × shape 组合 |
| shape_factor | num_entries/hidden/num_tokens | 边界值 | 两两组合 |
| num_qps_factor | QP 数量 | 全枚举 | QPs × hybrid 组合 |
| world_size_factor | 通信域大小 | 主要值 | world_size × hybrid 组合 |
| hybrid_mode_factor | bool 标志 | True/False | hybrid × 其他因子 |
| value_factor | indices 取值范围 | 边界值 | 值域交叉 |

### 4.2 边界值定义

**num_entries 边界**：
- 最小值：4096 (2^12)
- 最大值：524288 (2^19)
- 中间值：16384, 65536, 262144

**hidden 边界**：
- 最小值：64
- 最大值：4096
- 中间值：128, 256, 512, 1024, 2048

**num_tokens 边界**：
- 最小值：0（特殊边界）
- 小值：32, 128
- 中值：512, 1024, 2048
- 大值：4096, 8192

**num_qps 边界**：
- 最小值：1
- 典型值：4, 8, 16
- 最大值：32

**world_size 边界**：
- 最小值：2
- 典型值：4, 8, 16
- 大值：32

## 五、测试用例设计

### 5.1 Level0 用例（基础覆盖）

**设计原则**：单因子独立覆盖，用例数 ≤ 200

**当前设计**：46 条用例

**覆盖内容**：
1. dtype 覆盖（bf16 基础组合） - 16 条
2. num_entries 边界值覆盖 - 3 条
3. hidden 边界值覆盖 - 7 条
4. num_tokens 边界值覆盖 - 7 条
5. num_qps 全枚举覆盖 - 5 条
6. world_size 主要值覆盖 - 5 条
7. hybrid_mode 标志覆盖 - 2 条
8. num_tokens=0 特殊边界 - 1 条

### 5.2 Level1 用例（两两组合）

**设计原则**：两两因子组合覆盖，用例数 500~700

**当前设计**：500 条用例

**组合重点**：
1. **hidden × num_tokens 组合**（最重要） - 49 条
   - 验证不同 hidden 和 token 数量的性能表现
2. **num_entries × world_size 组合** - 25 条
   - 验证分布式规模对 RDMA 的影响
3. **num_qps × hybrid_mode 组合** - 10 条
   - 验证 QP 数量与 hybrid 模式的协同
4. **world_size × hybrid_mode 组合** - 10 条
   - 验证分布式拓扑与 hybrid 模式的适配
5. **随机组合补充** - 达到 500 条

### 5.3 特殊用例

**边界用例**：
- num_tokens = 0：验证空数据场景
- indices 边界值：验证索引范围的正确性

**典型场景**：
- 大规模推理：num_entries=524288, hidden=1024, num_tokens=4096
- 小规模测试：num_entries=4096, hidden=128, num_tokens=32

## 六、列字段定义

### 6.1 通用列字段

| 字段名 | 编码格式 | 说明 |
|--------|----------|------|
| case_name | 纯字符串 | 用例命名：{num_tokens}_{hidden}_{dtype}_engram_{feature}_ID{序号} |
| input_tensor_shape | Python list of lists | [[num_tokens], [num_tokens, hidden]] |
| input_tensor_range | Python list of tuples | [(indices_min, indices_max), (-5, 5)] |
| input_tensor_dtype | Python list of strings | ['int32', 'bf16'] |
| output_tensor_dtype | Python list of strings | ['bf16'] |

### 6.2 算子特定列字段

| 字段名 | 编码格式 | 说明 |
|--------|----------|------|
| num_entries | 整数 | 每个 rank 的表行数 |
| hidden | 整数 | 隐藏层维度 |
| num_tokens | 整数 | fetch 的 token 数量 |
| num_qps | 整数 | QP 数量 |
| world_size | 整数 | 通信域大小 |
| allow_hybrid_mode | 0/1 | hybrid 模式标志 |
| seed | 整数 | 随机种子 |

## 七、测试执行建议

### 7.1 单算子模式

```bash
python3 mc2_test.py --xlsx result/excel/engram_fetch.xlsx --sheet level0 --num 1 --scene npu
```

### 7.2 图模式

```bash
python3 mc2_test.py --xlsx result/excel/engram_fetch.xlsx --sheet level1 --num 1 --scene graph
```

### 7.3 性能验证

根据文档第5节，建议验证以下性能指标：

- **带宽目标**：≥ 70% 线速（RoCE 400Gbps, 4096 tokens, hidden=1024）
- **延迟目标**：≤ 50 µs（4096 tokens, hidden=128, world=8）
- **AIV 占用**：≤ numQps 核 × launch 期

## 八、检查清单验证

生成 xlsx 后已验证：

- [x] 包含 level0 和 level1 sheet
- [x] level0 用例数 ≤ 200（实际：46）
- [x] level1 用例数 500~700（实际：500）
- [x] 所有 shape/dtype/range 可 eval() 解析
- [x] case_name 遵循命名规范
- [x] 布尔值使用 0/1 编码

## 九、迭代优化建议

### 9.1 Level0 优化

当前 Level0 用例数 46，可以增加以下覆盖：
- 增加 hidden × num_tokens 的基础组合（约 20 条）
- 增加 value_range 的边界测试（约 10 条）
- 总数控制在 150~200 条

### 9.2 Level1 优化

当前 Level1 用例数 500，可以优化为：
- 重点组合：hidden × num_tokens × world_size（三维组合）
- 性能测试：大 hidden (1024, 2048, 4096) × 大 tokens (4096, 8192)
- 总数控制在 600~700 条

### 9.3 Exception Sheet（可选）

可添加异常用例：
- indices 越界（超出 world_size × num_entries）
- num_tokens 为负数（非法）
- 未先执行 engram_write（约束违反）

## 十、输出文件

**文件路径**：`result/excel/engram_fetch.xlsx`

**生成时间**：2026-06-08

**用例统计**：
- Level0: 46 条
- Level1: 500 条

---

**设计依据**：tests/elastic/fanchuan.md（EngramFetch 算子设计文档）

**参考规范**：
- xlsx-format-spec.md
- operator-design-patterns.md

**生成工具**：result/generate_engram_fetch_cases.py