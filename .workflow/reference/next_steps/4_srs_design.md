# 任务 4：编写 SRS 算子开发实现方案

## 背景

基于任务 1（算子规格）、任务 2（GMM A16W8 走读）、任务 3（wqbmmv2 ND 反量化走读）的产出，编写 GMM A16W4 pergroup ND 的 SRS 算子开发实现方案。该方案需要尽可能详细，作为后续编码的直接依据。

**本任务不进行编码**，仅产出设计方案文档。

## 任务详情

### 目标

产出 `.workflow/raw/design/gmm_a16w4_950_design.md`，覆盖 tiling data、tiling 计算方案、L1/UB 空间划分、kernel 实现方案、核心代码适配点。

### 方案应包含的内容

#### 1. Tiling Data 设计
- `GMMWeightQuantParam` 结构是否需要新增字段
- `groupSize` 字段的设置方式（32/64/128/256 如何从输入参数获取）
- 新增字段（如有）的定义和用途

#### 2. Tiling 计算方案
- 参照 A16W8 多多多的 tiling 策略（x: 256*256, weight: 256*256, kL1=256）
- A16W4 pergroup 场景下 baseM/baseN/baseK 的计算方式
- mList/kList/nList 的填充逻辑（单单单 vs 多多多）
- `groupSize` 在 tiling 计算中的使用方式
- 转置/非转置场景的 tiling 差异

#### 3. L1/UB 空间划分方案
- L1 buffer 划分：aL1/bL1 的大小和数量（参照 A16W8）
- UB buffer 划分：
  - weight UB buffer（int4 unpack 后的 fp16/bf16 空间）
  - scale UB buffer（pergroup scale 的加载空间，shape 为 `(k/gs, n)`）
  - offset UB buffer（可选，pergroup offset 的加载空间）
  - output UB buffer
- 参照 A16W8 的 UB 划分，明确 A16W4 pergroup 需要的调整
- `VecAntiQuantConfig` 参数设置

#### 4. Kernel 实现方案

##### 4.1 Controller 层适配
- `GMMWeightQuantBasicController` 的修改点
- `SetGmAddr()` 中 pergroup 地址偏移公式设计：
  - perchannel（A16W8 现有）：`antiquantParamsBaseOffset += nSize`
  - pergroup（A16W4 新增）：`antiquantParamsBaseOffset += (kSize / groupSize) * nSize`
  - 转置场景下的偏移公式
- `int4b_t` weight 偏移的 `>> 1` 处理在 pergroup 场景下的正确性验证

##### 4.2 BasicBlock 层适配
- `WqmmConfig` 新增配置项设计（参照 `S8S4_NZKN_G` 新增 ND 版本）
- `WeightQuantMatmulBasicBlock` 模板参数适配
- `ComputeBasicBlockAivNdNkNzKn` 中的 pergroup 分支设计

##### 4.3 Vec 反量化函数实现
- 新增 NK 排布的 pergroup ND 反量化 VF 函数（参照 wqbmmv2 `AntiquantComputeW4NKPerGroup`）
- 新增 KN 排布的 pergroup ND 反量化 VF 函数（参照 wqbmmv2 `AntiquantComputeKNPerGroup`）
- 函数接口与 GMM `basic_block_vf_nd.h` 已有函数的对齐方式
- INT4 weight unpack 和 cast 流程设计
- pergroup scale/offset 加载方式设计

##### 4.4 Cube 计算
- 确认 cube 计算流程与 A16W8 多多多一致，无需修改
- 如有差异，明确修改点

#### 5. 核心代码适配点汇总
- 列出所有需要修改的文件和函数
- 每个修改点的具体修改内容
- 新增文件和函数的清单
- 不需要修改的部分（确认可复用）

#### 6. 测试方案
- tiling UT 用例设计（覆盖 group_size=32/64/128/256、转置/非转置、单单单/多多多）
- kernel UT 用例设计（覆盖各种 shape 组合）
- 兼容性验证方案

### 依赖的输入文档

| 文档 | 来源 |
|------|------|
| 算子规格文档 | 任务 1 产出：`.workflow/raw/spec/gmm_a16w4nd_op.md` |
| GMM A16W8 走读文档 | 任务 2 产出：`.workflow/raw/docs/gmm_a16w8_*_walkthrough.md` |
| wqbmmv2 ND 反量化走读文档 | 任务 3 产出：`.workflow/raw/docs/wqbmmv2_nd_antiquant_walkthrough.md` |

### 完成标准

- [ ] 方案覆盖 tiling data、tiling 计算、L1/UB 空间划分、kernel 实现四个维度
- [ ] 每个修改点有明确的代码位置和修改内容
- [ ] pergroup 地址偏移公式经过推导验证
- [ ] VF 函数接口设计明确，与 wqbmmv2 和 GMM 已有接口对齐
- [ ] 新增 WqmmConfig 配置项定义明确
- [ ] 测试方案覆盖所有需要验证的场景

## 任务计划

1. 确认任务 1/2/3 的产出文档已完成
2. 设计 Tiling Data 和 Tiling 计算方案
3. 设计 L1/UB 空间划分方案
4. 设计 Kernel 实现方案（Controller → BasicBlock → VF 函数）
5. 汇总核心代码适配点
6. 设计测试方案
7. 整合产出 SRS 文档
