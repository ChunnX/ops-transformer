# 任务 4：编写 SRS 算子开发实现方案

## 背景

基于任务 1（算子规格）、任务 2（GMM A16W8 走读）、任务 3（wqbmmv2 ND 反量化走读）的产出，编写 GMM A16W4 pergroup ND 的 SRS 算子开发实现方案。该方案需要尽可能详细，作为后续编码的直接依据。

**本任务不进行编码**，仅产出设计方案文档，编码方案有任何不一致的地方以 `.workflow/raw/spec/gmm_a16w4nd_op.md` 为准。

## 文档组织原则

方案文档**按模块分章节、独立成文件**，方便精确检索和独立评审。每个文档聚焦一个独立关注点，文档间通过交叉引用关联。

## 产出文档清单

| 序号 | 文档 | 文件名 | 内容聚焦 |
|:----:|------|--------|----------|
| 01 | 交付件清单 | `01_deliverables.md` | 完整项目交付件清单：新增/修改的文件、注册点、编译配置 |
| 02 | 与硬件无关的实现方案 | `02_platform_independent_design.md` | 数据流总体设计、地址偏移公式推导、pergroup 量化模型、兼容性策略 |
| 03 | Tiling 拦截校验与切分方案 | `03_tiling_design.md` | tiling data、tiling 计算、参数校验、shape list 切分、tiling key 注册 |
| 04 | Kernel 实现方案 | `04_kernel_design.md` | Controller 适配、BasicBlock 适配、VF 反量化函数、L1/UB 空间划分、Cube 计算 |

所有文档输出到 `.workflow/raw/design/` 目录下，按编号顺序阅读。

## 文档 01：交付件清单（01_deliverables.md）

### 目标

明确完整项目交付的交付件清单，覆盖从算子定义注册到 kernel 实现的全部新增/修改文件。

### 应包含的内容

#### 1.1 新增 Tiling 模板
- 是否需要新增 tiling 类（继承 `GroupedQmmTiling` 或独立类）
- 还是复用现有 `GroupedWeightQuantBatchMatmulTiling` 并扩展
- 新增 tiling 类的文件路径和类名

#### 1.2 Tiling 模板注册与分发
- `TilingGMM()` 中的分发逻辑是否需要修改（当前伪量化走 `GroupedWeightQuantBatchMatmulTiling.SetTiling()` 直接调用）
- 是否需要新增 `REGISTER_OPS_TILING_TEMPLATE` 注册
- tiling key 参数是否需要新增（`weight_quant_tiling_key.h` 中的 `ASCENDC_TPL_ARGS_SEL` 新增组合）

#### 1.3 grouped_matmul_def.cpp 适配
- `config950` 中是否需要新增 dtype/format 组合
- 当前 A16W4 INT4 ND pergroup 的 dtype 组合是否已在 config950 中注册
- 需要确认 x=FLOAT16/BFLOAT16 + weight=INT4 + antiquantScale=FLOAT16/BFLOAT16 的组合是否存在

#### 1.4 grouped_matmul_apt.cpp kernel 入口适配
- `V310_GMM_ANTI_QUANT` 分支中是否需要新增 `if constexpr` 分支
- 新增分支的模板参数匹配条件（`W_TYPE`、`W_QUANT_TYPE`、`WQ_B_TRANS` 等）
- `INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL` 宏调用时使用的 `WqmmConfig` 和 `VecAntiQuantConfig`

#### 1.5 Kernel 代码实现
- 新增/修改的 kernel 文件清单
- 新增 `WqmmConfig` 实例
- 新增 VF 函数（`basic_block_vf_nd.h`）
- 修改的 Controller 代码（`SetGmAddr` 偏移公式）

#### 1.6 测试交付件
- tiling UT 新增用例
- kernel UT 新增用例

#### 1.7 编译配置
- CMakeLists.txt 是否需要修改
- 编译宏是否需要新增

### 依赖的输入

| 文档 | 来源 |
|------|------|
| 算子规格文档 | `.workflow/raw/spec/gmm_a16w4nd_op.md` |
| GMM A16W8 tiling 走读 | `.workflow/raw/docs/gmm_a16w8_tiling_walkthrough.md` |
| GMM A16W8 kernel 走读 | `.workflow/raw/docs/gmm_a16w8_kernel_walkthrough.md` |
| 架构分析 | explore agent 产出的注册/分发架构分析 |
| **02 与硬件无关方案** | `.workflow/raw/design/02_platform_independent_design.md` |

### 完成标准

- [ ] 交付件清单覆盖所有新增/修改的文件
- [ ] 每个交付件有明确的文件路径和修改内容摘要
- [ ] tiling 注册和 kernel 入口注册的代码位置和修改方式明确
- [ ] grouped_matmul_def.cpp 的 dtype 组合确认完整
- [ ] 编译配置变更明确

---

## 文档 03：Tiling 拦截校验与切分方案（03_tiling_design.md）

### 目标

设计 A16W4 pergroup ND 场景的 tiling 参数校验、拦截逻辑和切分策略。

### 应包含的内容

#### 2.1 Tiling Data 设计
- `GMMWeightQuantParam` 结构是否需要新增字段
- `groupSize` 字段的设置方式（32/64/128/256 如何从输入参数获取）
- `GMMWeightQuantTilingData` 完整结构定义

#### 2.2 Tiling 拦截校验
- `IsA16W4ND()` 判断逻辑（当前如何区分 A16W4 ND pergroup 和 A16W8 perchannel）
- `CheckGroupSize()` 校验逻辑（新增 32/64/128/256 校验）
- `CheckTensorDim*()` 校验适配（pergroup antiquantScale 为 3 维 `(E, G, N)` vs perchannel 为 2 维 `(E, N)`）
- `CheckTensorShape()` 校验适配（K % groupSize == 0、pergroup 整除等）

#### 2.3 Tiling 计算方案
- 参照 A16W8 多多多的 tiling 策略（x: 256*256, weight: 256*256, kL1=256）
- A16W4 pergroup 场景下 baseM/baseN/baseK 的计算方式
- `SetAntiquantGroupSize()` 中 groupSize 的推导逻辑
- `SetTilingKey()` 中 `antiquantType` 设置为 `PER_GROUP`

#### 2.4 Shape List 切分
- `SetShapeList()` 单单单/多多多两条路径的适配
- mList/kList/nList 的填充逻辑
- `CalcResplitTiling()` 尾块重切分在 A16W4 pergroup 场景下的适用性

#### 2.5 Tiling Key 注册
- `weight_quant_tiling_key.h` 中需要新增的 `ASCENDC_TPL_ARGS_SEL` 组合
- 新增组合的模板参数值（`W_TYPE`、`W_QUANT_TYPE`、`C_QUANT_TYPE`、`WQ_B_TRANS`、`ALGORITHM_SUB_CATEGORY` 等）
- 转置和非转置场景的 tiling key 差异

### 依赖的输入

| 文档 | 来源 |
|------|------|
| 算子规格文档 | `.workflow/raw/spec/gmm_a16w4nd_op.md` |
| GMM A16W8 tiling 走读 | `.workflow/raw/docs/gmm_a16w8_tiling_walkthrough.md` |
| **01 交付件清单** | `.workflow/raw/design/01_deliverables.md` |
| **02 与硬件无关方案** | `.workflow/raw/design/02_platform_independent_design.md` |

### 完成标准

- [ ] tiling data 结构定义完整
- [ ] 拦截校验覆盖所有约束条件（group_size、K 对齐、shape 匹配等）
- [ ] tiling 计算方案覆盖单单单/多多多、转置/非转置
- [ ] tiling key 新增组合明确，与 kernel 入口的 `if constexpr` 匹配

---

## 文档 04：Kernel 实现方案（04_kernel_design.md）

### 目标

设计 A16W4 pergroup ND 场景的 kernel 实现方案，覆盖 Controller、BasicBlock、VF 函数各层。

### 应包含的内容

#### 3.1 Controller 层适配
- `GMMWeightQuantBasicController` 的修改点
- `SetGmAddr()` 中 pergroup 地址偏移公式设计：
  - perchannel（A16W8 现有）：`antiquantParamsBaseOffset += nSize`
  - pergroup（A16W4 新增）：`antiquantParamsBaseOffset += (kSize / groupSize) * nSize`
  - 转置场景下的偏移公式
- `int4b_t` weight 偏移的 `>> 1` 处理在 pergroup 场景下的正确性验证

#### 3.2 BasicBlock 层适配
- `WqmmConfig` 新增配置项设计（参照 `S8S4_NZKN_G` 新增 ND 版本）
- `WeightQuantMatmulBasicBlock` 模板参数适配
- `ComputeBasicBlockAivNdNkNzKn` 中的 pergroup 分支设计

#### 3.3 L1/UB 空间划分方案
- L1 buffer 划分：aL1/bL1 的大小和数量（参照 A16W8）
- UB buffer 划分：
  - weight UB buffer（int4 unpack 后的 fp16/bf16 空间）
  - scale UB buffer（pergroup scale 的加载空间，shape 为 `(k/gs, n)`）
  - offset UB buffer（可选）
  - output UB buffer
- 参照 A16W8 的 UB 划分，明确 A16W4 pergroup 需要的调整
- `VecAntiQuantConfig` 参数设置
- `GetBufferConfig()` 中 pergroup scale UB 空间的计算

#### 3.4 Vec 反量化函数实现
- 新增 NK 排布的 pergroup ND 反量化 VF 函数（参照 wqbmmv2 `AntiquantComputeW4NKPerGroup`）
- 新增 KN 排布的 pergroup ND 反量化 VF 函数（参照 wqbmmv2 `AntiquantComputeKNPerGroup`）
- 函数接口与 GMM `basic_block_vf_nd.h` 已有函数的对齐方式
- INT4 weight unpack 和 cast 流程设计
- pergroup scale/offset 加载方式设计
- 按 groupSize 分支处理的策略（运行时分支 vs 统一处理）

#### 3.5 Cube 计算
- 确认 cube 计算流程与 A16W8 多多多一致，无需修改
- 如有差异，明确修改点

#### 3.6 核心代码适配点汇总
- 列出所有需要修改的文件和函数
- 每个修改点的具体修改内容
- 新增文件和函数的清单
- 不需要修改的部分（确认可复用）

### 依赖的输入

| 文档 | 来源 |
|------|------|
| GMM A16W8 kernel 走读 | `.workflow/raw/docs/gmm_a16w8_kernel_walkthrough.md` |
| wqbmmv2 ND 反量化走读 | `.workflow/raw/docs/wqbmmv2_nd_antiquant_walkthrough.md` |
| **01 交付件清单** | `.workflow/raw/design/01_deliverables.md` |
| **02 与硬件无关方案** | `.workflow/raw/design/02_platform_independent_design.md` |

### 完成标准

- [ ] Controller 层修改方案明确（地址偏移公式推导验证）
- [ ] WqmmConfig 新增配置项定义明确
- [ ] L1/UB 空间划分方案明确（pergroup scale UB 空间计算）
- [ ] VF 函数接口设计明确，与 wqbmmv2 和 GMM 已有接口对齐
- [ ] 核心代码适配点汇总完整（文件级清单）

---

## 文档 02：与硬件无关的实现方案（02_platform_independent_design.md）

### 目标

描述 A16W4 pergroup ND 数据流的总体设计，不依赖具体硬件实现细节，作为跨平台参考。

### 应包含的内容

#### 4.1 数据流总体设计
- A16W4 pergroup 伪量化数据流的计算公式
- 反量化模型：`(weight + offset) × scale`，scale shape 为 `(k/gs, n)`
- 与 perchannel 量化的本质差异（反量化粒度：K 轴按 gs 分段 vs 整个 K 轴共用）

#### 4.2 地址偏移公式推导
- 单单单场景下各 tensor 的 GM 地址偏移公式推导
- 多多多场景下各 tensor 的 GM 地址偏移公式推导
- pergroup scale/offset 的偏移公式推导（与 perchannel 的对比）
- INT4 weight 的字节偏移特殊处理

#### 4.3 兼容性策略
- 如何确保不破坏已有 A2/A3 A16W4 功能
- 如何确保不破坏已有 Ascend 950 A16W8 perchannel 功能
- 代码分支隔离策略（编译期 `if constexpr` vs 运行时分支）

#### 4.4 测试方案
- tiling UT 用例设计（覆盖 group_size=32/64/128/256、转置/非转置、单单单/多多多）
- kernel UT 用例设计（覆盖各种 shape 组合）
- 兼容性验证方案

### 依赖的输入

| 文档 | 来源 |
|------|------|
| 算子规格文档 | `.workflow/raw/spec/gmm_a16w4nd_op.md` |
| 脑图文档 | `.workflow/raw/development/gmm_a16w4_pergroup_nd_mindmap.md` |
| **01 交付件清单** | `.workflow/raw/design/01_deliverables.md` |

### 完成标准

- [ ] 地址偏移公式经过数学推导验证
- [ ] 兼容性策略明确
- [ ] 测试方案覆盖所有需要验证的场景

---

## 任务执行顺序

按文档编号顺序执行：

1. **01 交付件清单**（`01_deliverables.md`）：先明确交付件范围，作为其他文档的输入
2. **02 与硬件无关方案**（`02_platform_independent_design.md`）：先确定总体设计和公式推导
3. **03 Tiling 方案**（`03_tiling_design.md`）：基于交付件和总体设计，细化 tiling 方案
4. **04 Kernel 方案**（`04_kernel_design.md`）：基于交付件和总体设计，细化 kernel 方案

## 完成标准（全局）

- [ ] 4 个文档全部产出（01~04）
- [ ] 文档间交叉引用正确（使用编号引用）
- [ ] 交付件清单覆盖所有新增/修改的文件
- [ ] tiling 拦截校验覆盖算子规格文档中的所有约束
- [ ] kernel 实现方案覆盖 Controller/BasicBlock/VF 各层
- [ ] 地址偏移公式经过推导验证
- [ ] 测试方案覆盖所有需要验证的场景
