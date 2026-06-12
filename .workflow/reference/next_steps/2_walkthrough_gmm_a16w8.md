# 任务 2：走读 GMM A16W8 代码实现

## 背景

GMM A16W4 pergroup ND 的实现将参照现有的 GMM A16W8 多多多实现。需要完整走读 kernel 和 tiling 两部分的代码，理解现有的计算流程、空间划分、地址管理等方式，为后续适配 A16W4 提供充分的代码理解基础。

## 任务详情

### 目标

产出完整的走读文档，覆盖 GMM A16W8 多多多的 tiling 实现和 kernel 实现，重点记录可复用的部分和需要适配的部分。

### 走读范围

#### Part 1: Tiling 实现

| 文件 | 走读重点 |
|------|----------|
| `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h` | tiling 类定义、接口 |
| `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp` | tiling 计算逻辑：如何根据输入 shape 计算 baseM/baseN/baseK、mList/kList/nList、groupSize 等 |
| `gmm/grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h` | `GMMWeightQuantParam` 结构定义：所有 tiling 字段的含义和用途 |

**走读要点**：
- `GMMWeightQuantParam` 中 `groupSize` 字段的设置逻辑
- `singleX`/`singleWeight`/`singleY` 如何影响 tiling 参数
- `groupType`/`groupListType` 如何影响 mList/kList/nList 的填充
- baseM/baseN/baseK 的计算方式（当前 A16W8 是 256*256，kL1=256）
- tiling 参数如何传递到 kernel（通过 GM_ADDR tiling）

#### Part 2: Kernel 入口实现

| 文件 | 走读重点 |
|------|----------|
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h` | **核心参照文件**：Controller 的 Init/Process/SetMKN/SetGmAddr 完整流程 |

**走读要点**：
- `Init()` 参数传递链：`groupSize` 如何从 tiling 传递到 `wqmmBasicBlock_.Init()`
- `Process()` 主循环：group 遍历 → SetMKN → SetGmAddr → baseM/baseN 动态调整 → ComputeBasicBlock
- `SetMKN()` 中 groupType=0（单单单）和 groupType=-1（多多多）的 MKN 设置差异
- `SetGmAddr()` 中各 tensor 的地址偏移计算：
  - x/weight/y 的偏移公式
  - antiquantScale/antiquantOffset 的偏移公式（**perchannel vs pergroup 的核心差异点**）
  - `int4b_t` weight 的偏移特殊处理（`>> 1`）
- `GetSplitValueFromGroupList()` 中 groupListType=0/1 的处理

#### Part 3: Kernel 中 L1/UB 空间划分方案

| 文件 | 走读重点 |
|------|----------|
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h` | BasicBlock 的 Init/ComputeBasicBlock/End 流程，L1/UB buffer 分配 |
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h` | `WqmmConfig` 配置结构、`VecAntiQuantConfig`、已有的配置项 |

**走读要点**：
- L1 buffer 划分：aL1/bL1 的大小和数量
- UB buffer 划分：weight UB、scale UB、output UB 的分配方式
- `VecAntiQuantConfig` 中 `ubMte2BufferNum` 和 `ubMte2InnerSize` 的作用
- `WeightQuantMatmulBasicBlock` 的模板参数如何影响计算路径

#### Part 4: Kernel Cube 计算与向量计算调度

| 文件 | 走读重点 |
|------|----------|
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h` | cube 计算：matmul 调用方式 |
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h` | 向量计算调度：反量化 VF 函数的调用链和分发逻辑 |

**走读要点**：
- cube 计算流程：A/B 矩阵加载 → matmul → C 写回
- cube 计算与向量计算的协作方式（流水线、同步机制）
- 向量计算中反量化的调度流程：
  - 如何根据 `antiQuantType`（PER_GROUP/PER_CHANNEL/PER_TENSOR）分发到不同的 VF 函数
  - 反量化 VF 函数的调用接口（输入参数、输出参数）
  - scale/offset 的加载和传递方式
- **注意**：VF 函数的内部实现不在本任务走读范围内，将在任务 3 中走读 ops-nn 仓库的 wqbmmv2 反量化实现方案（`AntiquantComputeW4NKPerGroup`/`AntiquantComputeKNPerGroup`）

### 产出

走读文档输出到 `.workflow/raw/docs/` 目录下，建议分章节编写。

### 完成标准

- [ ] tiling 走读文档：覆盖 `GMMWeightQuantParam` 所有字段含义、tiling 计算流程、groupSize 设置逻辑
- [ ] kernel 走读文档：覆盖 Controller 完整流程、SetGmAddr 地址偏移公式、L1/UB 空间划分、Cube/Vec 计算调度
- [ ] 明确标注 A16W4 pergroup 需要适配的代码点（Controller 地址偏移、tiling 参数、Vec 调度分支等）
- [ ] 明确标注可复用的代码和需要新增的代码

## 任务计划

1. 走读 tiling 实现（Part 1）→ 产出 tiling 走读文档
2. 走读 kernel Controller（Part 2）→ 记录核心流程和地址偏移公式
3. 走读 L1/UB 空间划分（Part 3）→ 记录 buffer 分配方式
4. 走读 kernel Cube/Vec 计算调度（Part 4）→ 记录 matmul 调用方式和反量化调度流程
5. 汇总产出 kernel 走读文档
