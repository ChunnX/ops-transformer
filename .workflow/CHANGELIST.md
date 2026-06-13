# CHANGELIST — 变更记录

> 记录项目演进和变更，便于后期回溯和审查。
> **只追加，不删除已有条目。**

---

## 变更记录格式

```
### YYYY-MM-DD | 变更标题

**类型**: 🆕 新功能 | 🔧 修改 | 🐛 修复 | 🗑️ 删除 | 📝 文档
**范围**: 影响范围

#### 变更内容
- 具体变更了什么

#### 关键决策
- 做了什么决策，为什么

#### 影响范围
- 对现有功能的影响
```

---

<!-- 在此追加变更记录 -->

### 2026-06-11 | 重写规划文档 — 四任务结构，初期不编码

**类型**: 🔧 修改
**范围**: .workflow/ 全目录

#### 变更内容
- 重写 GOAL.md：补充实现方案要点（tiling 策略、kernel 参照、vec 反量化参照、UB 划分、地址偏移）
- 重写 PLAN.md：三阶段结构（规划设计 → 编码实现 → 验证），当前阶段为规划设计
- 重写 NEXT_STEPS.md：简化为任务索引，详细任务说明移至 reference/next_steps/
- 新增 4 个详细任务说明文档：
  - `reference/next_steps/1_op_spec.md` — 算子原型描述文档
  - `reference/next_steps/2_walkthrough_gmm_a16w8.md` — 走读 GMM A16W8 代码
  - `reference/next_steps/3_walkthrough_wqbmmv2.md` — 走读 wqbmmv2 ND 反量化
  - `reference/next_steps/4_srs_design.md` — SRS 算子开发实现方案
- 更新 raw/INDEX.md：登记新增的算子规格文档

#### 关键决策
- 初期方案阶段不进行编码，仅产出规格文档、走读文档和设计方案
- 详细任务说明从 NEXT_STEPS.md 拆分到 reference/next_steps/ 目录，NEXT_STEPS.md 仅保留索引
- 走读范围明确包含 tiling 实现、tiling data、kernel 入口、L1/UB 空间划分、kernel 实现代码
- vec 反量化参照 ops-nn 仓库的 wqbmmv2 实现（`AntiquantComputeW4NKPerGroup`/`AntiquantComputeKNPerGroup`）

#### 影响范围
- 仅规划文档变更，不涉及代码

---

### 2026-06-12 | 完成 GMM A16W8 代码走读文档

**类型**: 📝 文档
**范围**: `.workflow/raw/docs/` 走读文档

#### 变更内容
- 新增 `gmm_a16w8_tiling_walkthrough.md`：Tiling 走读文档
  - 覆盖 `GMMWeightQuantParam` 所有字段含义
  - 覆盖 `SetTiling()` 完整流程（AnalyzeAttr → CalcResplitTiling → SetBaseTiling → SetMatMulTiling → SetTilingKey → SetCustomParam）
  - 覆盖 `SetShapeList()` 单单单/多多多两条路径
  - 覆盖 `SetAntiquantGroupSize()` groupSize 设置逻辑
  - 覆盖 `IsA16W4ND()` 判断逻辑及其在各校验函数中的使用
  - 明确标注 A16W4 pergroup 需要适配的代码点（6处修改 + 3处新增）
- 新增 `gmm_a16w8_kernel_walkthrough.md`：Kernel 走读文档
  - 覆盖 Controller 完整流程（Init → Process → SetMKN → SetGmAddr）
  - 覆盖 SetGmAddr 地址偏移公式（含 perchannel vs pergroup 核心差异）
  - 覆盖 L1/UB 空间划分方案（L1 512KB 分配、UB 248KB 分配）
  - 覆盖 Cube 计算调度（LaunchMatmul、AIC 核循环）
  - 覆盖 Vec 反量化调度流程（CopyGmToUb → CopyAntiQuantParamsGmToUb → WeightAntiQuantCompute → AntiQuantProcessNd → VF 函数）
  - 覆盖 AIC/AIV 同步机制
  - 明确标注 A16W4 pergroup 需要适配的代码点（Controller 1处、Config 2处、Vec 2处、VF 6处、Tiling 3处）
  - 明确标注可复用代码（15+ 函数）和需要新增的代码（6项）

#### 关键决策
- 发现 `CopyAntiQuantParamsGmToUb()` 中 `PER_GROUP` 分支已存在（用于 S8S4 NZ 场景），但 ND 场景的 VF 函数（`basic_block_vf_nd.h`）中缺少 `PER_GROUP` 分支
- 确认 A16W4 pergroup 的核心新增工作在 `basic_block_vf_nd.h` 中的 `NdNkLoadScaleOffset()` 和 `NdKnLoadScaleOffset()` 需要新增 PER_GROUP 分支
- 确认 Controller 的 `SetGmAddr()` 中 `antiquantParamsBaseOffset` 更新公式需要区分 perchannel（`+= nSize`）和 pergroup（`+= (kSize/groupSize) * nSize`）

#### 影响范围
- 仅文档变更，不涉及代码
- 为后续任务 3（走读 wqbmmv2 ND 反量化）和任务 4（SRS 实现方案）提供充分的代码理解基础

---

### 2026-06-12 | 完成 WQBMMv2 ND 反量化走读文档

**类型**: 📝 文档
**范围**: `.workflow/raw/docs/wqbmmv2_nd_antiquant_walkthrough.md`

#### 变更内容
- 新增 `wqbmmv2_nd_antiquant_walkthrough.md`（939 行）：WQBMMv2 ND INT4 PerGroup 反量化走读文档
  - 覆盖 `AntiquantComputeW4NKPerGroup` 完整实现（按 groupSize 分 6+ 个专用 VF 函数分支）
  - 覆盖 `AntiquantComputeKNPerGroup` 完整实现（统一 VF 函数 + groupTail 处理）
  - 覆盖 VF 底层函数实现详解（MicroAPI 调用序列、数据搬运、计算流程）
  - 覆盖 NK vs KN 排布差异对比表（12 维度对比）
  - 覆盖 NZ 格式 pergroup 对比参照
  - 覆盖 GMM `basic_block_vf_nd.h` 已有接口分析
  - 覆盖迁移适配方案（可复用部分 + 5 项需适配部分 + 步骤建议）
  - 覆盖与 GMM 接口对齐方式（新增函数签名建议 + 与已有接口一致性对比）
  - 每个核心函数均有伪代码 + ASCII 数据流图

#### 关键决策
- **NK 排布复杂度高**：按 groupSize 分 6+ 个专用 VF 函数（gs=32/64/128/256/odd/gt128），scale 用广播加载
- **KN 排布统一简洁**：单一 VF 函数 `AntiquantPerGroupKN`，K 方向逐行步进，scale 用直接加载
- **GMM 迁移建议**：保持 GMM 的 `__local_mem__` 指针传递风格，复用 wqbmmv2 的 VF 核心逻辑，groupSize 通过运行时参数传入
- **参数传递方案**：推荐方案 A（函数参数传递，灵活性高），而非方案 B（扩展 WqmmConfig 编译期固定）

#### 影响范围
- 仅文档变更，不涉及代码
- 为后续任务 4（SRS 实现方案）提供 wqbmmv2 反量化实现的完整参考和迁移方案

---

### 2026-06-13 | 完成 SRS 算子开发实现方案（任务 4）

**类型**: 📝 文档
**范围**: `.workflow/raw/design/` 设计文档

#### 变更内容
- 新增 `01_deliverables.md`（交付件清单）：
  - 明确 11 个需修改文件 + 2 个无需修改文件
  - 明确 8 个新增 tiling key 组合（pergroup × offset/bias × 转置/非转置）
  - 明确 config950 需新增 INT4 ND dtype 组合
  - 明确 grouped_matmul_apt.cpp 需新增 8 个 if constexpr 分支
- 新增 `02_platform_independent_design.md`（与硬件无关方案）：
  - 完成 pergroup 地址偏移公式推导（含数值验证）
  - 发现 bias 偏移需与 antiquantScale 偏移分离（关键设计发现）
  - 明确兼容性策略（编译期 if constexpr 隔离）
  - 设计 tiling UT 19 个用例 + kernel UT 15 个用例 + 兼容性验证 5 项
- 新增 `03_tiling_design.md`（Tiling 方案）：
  - 确认 GMMWeightQuantParam 无需新增字段（groupSize 已有）
  - 设计 IsA16W4NDPergroup() 判断函数
  - 设计 CheckGroupSize() 新增 32/64/128/256 校验
  - 设计多多多场景 antiquantScale 2 维 pergroup 判断逻辑
  - 确认 baseM/baseN/baseK 复用 A16W8 默认值（256/256/64）
  - 设计 8 个 ASCENDC_TPL_ARGS_SEL 新增组合
- 新增 `04_kernel_design.md`（Kernel 方案）：
  - 设计 SetGmAddr() pergroup 偏移公式 + bias 偏移分离方案
  - 新增 4 个 WqmmConfig 实例（A16W4_NDKN_PG/A16W4_NDNK_PG 各含有/无 offset）
  - 设计 GetNdPerGroupBufferInfo()（scale UB 从 4KB 增大到 8KB）
  - 设计 AntiQuantInt4NdNkPerGroup/AntiQuantInt4NdKnPerGroup VF 函数接口
  - 确认 Cube 计算完全复用 A16W8（无需修改）
  - 汇总 14 个修改点 + 7 个新增项 + 20+ 个可复用项

#### 关键决策
- **bias 偏移分离**：发现现有代码中 bias 与 antiquantScale 共用偏移变量，pergroup 场景下两者偏移量不同，需新增独立 biasBaseOffset
- **UB buffer 调整**：pergroup scale UB 从 4KB 增大到 8KB（highBitData 从 66KB 缩减到 62KB），总量不变
- **tiling key 组合数**：8 个（4 种 offset/bias 组合 × 2 种转置状态），全部使用 N_FIRST_BASIC_BLOCK
- **VF 函数策略**：NK 排布按 groupSize 运行时分支（与 wqbmmv2 一致），KN 排布统一处理
- **config950 适配**：需新增 weight=INT4 + FORMAT_ND 的 dtype/format 组合条目

#### 影响范围
- 仅文档变更，不涉及代码
- 4 个设计文档作为后续编码实现（任务 5/6）的直接依据

---

### 2026-06-13 | 新增任务 5：设计 GMM A16W4 PerGroup ND 测试用例

**类型**: 📝 文档
**范围**: `.workflow/` 规划文档 + `reference/next_steps/`

#### 变更内容
- 更新 `NEXT_STEPS.md`：新增任务 5 索引（设计测试用例）
- 更新 `PLAN.md`：阶段一新增任务 5，当前进度更新
- 新增 `reference/next_steps/5_test_case_design.md`：详细任务说明

#### 关键决策
- **任务 5 定位**：在编码之前完成测试用例设计，明确验证目标和参数设置
- **用例类型**：5 种（tiling 异常拦截、精度、功能、性能、泛化），兼容性用例暂不考虑
- **产出形式**：用例设计文档 + Excel 表格（多 sheet）+ Python 生成脚本
- **参数要求**：每个用例需给出完整的 dtype/format/shape/group_size 等具体参数，不使用模糊描述

#### 影响范围
- 仅规划文档变更，不涉及代码
- 任务 5 产出将作为后续编码验证的直接依据
