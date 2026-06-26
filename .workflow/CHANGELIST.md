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

---

### 2026-06-13 | 任务 5 完成：GMM A16W4 PerGroup ND 测试用例设计

**类型**: 📝 文档
**范围**: 测试用例设计文档 + Excel 表格 + 生成脚本

#### 变更内容
- 新增 `raw/design/05_test_case_design.md`：完整测试用例设计文档（120 个用例）
- 新增 `raw/design/test_cases_gmm_a16w4_pergroup_nd.xlsx`：6 sheet Excel 表格
- 新增 `raw/design/generate_test_cases.py`：Python 生成脚本（pandas + openpyxl）

#### 用例统计
| 类型 | 数量 | 编号范围 |
|------|------|----------|
| Tiling 异常拦截 | 44 | T-01 ~ T-44 |
| 精度验证 | 9 | P-01 ~ P-09 |
| 功能验证 | 27 | F-01 ~ F-27 |
| 性能验证 | 6 | PERF-01 ~ PERF-06 |
| 泛化验证 | 12 | G-01 ~ G-12 |
| Kernel 打桩验证 | 22 | KS-01 ~ KS-22 |
| **总计** | **120** | |

#### 关键决策
- **Kernel 打桩验证用例**：新增 KS-01~KS-22，覆盖 MTE 搬运、VF 计算、地址偏移、tiling 参数传递、matmul 计算 5 个关键阶段
- **N=1 约束**：因 weight 最后一维必须为偶数，极小 N 用例使用 N=2 替代 N=1
- **精度容差**：FP16 使用 rtol=1e-3/atol=1e-3，BF16 使用 rtol=1e-2/atol=1e-2
- **bias 偏移分离**：KS-16 专门验证 bias 独立偏移（pergroup 场景下 bias 与 scale 偏移不同）

#### 影响范围
- 为后续编码实现提供明确的验证目标
- Excel 文件可直接用于测试框架的参数化驱动

---

### 2026-06-14 | 补充 Tiling/Kernel 白盒用例

**类型**: 📝 文档
**范围**: 测试用例设计文档 + Excel 表格 + 生成脚本

#### 变更内容
- 更新 `raw/design/05_test_case_design.md`：新增 §8 Tiling 白盒用例（TW-01~TW-20）和 §9 Kernel 白盒用例（KW-01~KW-22）
- 更新 `raw/design/generate_test_cases.py`：新增 `generate_tiling_whitebox_cases()` 和 `generate_kernel_whitebox_cases()` 函数
- 重新生成 `raw/design/test_cases_gmm_a16w4_pergroup_nd.xlsx`：8 个 sheet，162 个用例

#### 新增用例统计
| 类型 | 数量 | 编号范围 | 测试方法 |
|------|------|----------|----------|
| Tiling 白盒 | 20 | TW-01 ~ TW-20 | 白盒（端到端级） |
| Kernel 白盒 | 22 | KW-01 ~ KW-22 | 白盒（端到端级） |
| **新增小计** | **42** | | |

#### 累计用例统计
| 类型 | 数量 | 测试方法 |
|------|------|----------|
| Tiling 异常拦截 | 44 | 黑盒 |
| 精度验证 | 9 | 黑盒 |
| 功能验证 | 27 | 黑盒 |
| 性能验证 | 6 | 黑盒 |
| 泛化验证 | 12 | 黑盒 |
| Kernel 打桩验证 | 22 | 白盒（插桩级） |
| Tiling 白盒 | 20 | 白盒（端到端级） |
| Kernel 白盒 | 22 | 白盒（端到端级） |
| **总计** | **162** | |

#### 关键决策
- **白盒用例设计原则**：设计依据是代码内部路径（函数名 + 分支条件），而非规格约束
- **Tiling 白盒覆盖路径**：IsA16W4NDPergroup 判断、SetAntiquantGroupSize、SetTilingKey、SetShapeList、baseM/baseN/baseK 计算、CalcResplitTiling、GMMWeightQuantParam 字段完整性
- **Kernel 白盒覆盖路径**：if constexpr 分支、ComputeBasicBlock 分发、GetBufferConfig 分支、VF groupSize 运行时分支、CopyAntiQuantParamsGmToUb PER_GROUP 分支、SetGmAddr pergroup 偏移、AIC/AIV 协作
- **端到端执行**：白盒用例不需要插桩，通过检查最终输出或 tiling 结构体字段值来验证

#### 影响范围
- 测试用例总数从 120 增加到 162
- Excel 文件从 6 个 sheet 增加到 8 个 sheet
- 为编码阶段提供更完整的代码路径覆盖验证目标

---

### 2026-06-14 | 新增任务 6：总结归纳设计文档

**类型**: 📝 文档
**范围**: `.workflow/` 规划文档 + `reference/next_steps/` + `raw/development/`

#### 变更内容
- 更新 `NEXT_STEPS.md`：新增任务 6 索引
- 更新 `PLAN.md`：阶段一新增任务 6，修正阶段二/三任务编号（5→7, 6→8, 7→9, 8→10）
- 新增 `reference/next_steps/6_summary_docs.md`：详细任务说明

#### 任务 6 产出（3 个文档，均在 `raw/development/` 下）
1. **更新脑图** `gmm_a16w4_pergroup_nd_mindmap.md`：对齐用例设计文档，标注黑盒/白盒和用例条数
2. **STC 用例设计方案** `gmm_a16w4_pergroup_nd_stc.md`：记录用例设计方法论和数量统计，泛化用例预留
3. **SRS 算子实现方案** `gmm_a16w4_pergroup_nd_srs.md`：含 tiling 切分 SVG、buffer 分配 SVG、软件算法到硬件映射 SVG（核心，放在计算流程伪代码之后）、A2/A3→950 关键修改点

#### 关键决策
- 任务 6 定位为**总结归纳**，不新增设计内容，而是提炼已有文档的核心信息
- 脑图作为用例总览图，不展开全部用例细节
- STC 文档只记录设计方案（方法论），不列出具体用例
- SRS 文档面向他人理解，省略细枝末节，图表优先
- **核心 SVG**：软件算法到硬件映射图，展示 GMM A16W4 数据流在 Ascend 950 硬件上的完整执行过程，放在计算流程伪代码之后
- 阶段二/三任务编号重新排列，避免与阶段一任务 5/6 冲突

#### 影响范围
- 仅规划文档变更，不涉及代码
- 阶段二任务编号从 5/6 改为 7/8，阶段三从 7/8 改为 9/10

---

### 2026-06-14 | 任务 6 完成：总结归纳设计文档

**类型**: 📝 文档
**范围**: `.workflow/raw/development/` 下 3 个文档

#### 变更内容
- 更新 `gmm_a16w4_pergroup_nd_mindmap.md`：UT/ST 部分对齐 05_test_case_design.md，标注黑盒/白盒和用例条数（162 个用例总览）
- 新增 `gmm_a16w4_pergroup_nd_stc.md`：STC 用例设计方案文档，记录测试策略、设计方法论、数量统计（8 类 162 个用例），泛化用例预留
- 新增 `gmm_a16w4_pergroup_nd_srs.md`：SRS 算子实现方案文档，包含：
  - Tiling 切分策略 SVG（M/N/K 三维切分 + 多 Core 分配）
  - Buffer 空间分配 SVG（L1 512KB + UB 248KB）
  - 软件算法到硬件映射 SVG（核心，展示数据流在 Ascend 950 上的执行过程）
  - 计算流程伪代码
  - 地址偏移对比表（perchannel vs pergroup）
  - A2/A3 → Ascend 950 关键修改点汇总表

#### 关键决策
- SRS 文档命名为 `_srs.md`（标准名称），而非 `_design_summary.md`
- 核心 SVG 为"软件算法到硬件映射图"，放在计算流程伪代码之后，将软件算法功能与 Ascend 950 硬件架构做对应
- STC 文档仅记录方法论和数量统计，不列出具体用例
- 脑图作为用例总览图，不展开全部用例细节

#### 影响范围
- 阶段一全部 6 个任务完成
- 3 个总结文档作为留档查看的参考资料

---

### 2026-06-14 | 修正 SRS tiling 切分和 buffer 分配章节

**类型**: 🔧 修改
**范围**: `raw/design/03_tiling_design.md` + `raw/development/gmm_a16w4_pergroup_nd_srs.md` + SVG 图

#### 变更内容
- **代码走读**：走读 `GroupedWeightQuantBatchMatmulTiling` 和 `GMMWeightQuantBasicController` 代码，获取实际切分策略和 buffer 分配方案
- **更新 `03_tiling_design.md`**：
  - 新增 §3.0 SetTiling() 6 步主流程
  - 新增 §3.1.1 Kernel 侧 Basic Controller 分核算法（动态 baseM/baseN 调整 + round-robin 伪代码）
  - 新增 §3.4 尾块重切分（使能条件 + 三分支逻辑）
  - 新增 §3.5 A16W4 PerGroup 适配点（当前代码 vs 用户确认策略的差异对比）
  - 新增 §4.2.1 单单单 vs 多多多 M/K/N 来源差异表
- **重写 `gmm_a16w4_pergroup_nd_srs.md` §2**：
  - 重写为 8 个子节（2.1~2.8）：切分常量 → Host 侧流程 → Kernel 侧分核算法 → 尾块重切分 → 单单单 vs 多多多 → A16W4 适配点 → Tiling Key → groupSize 传递
- **重写 `gmm_a16w4_pergroup_nd_srs.md` §3**：
  - 重写为 5 个子节（3.1~3.5）：L1 分配 → UB 分配 → Buffer 与基本块关系 → PerChannel vs PerGroup 差异 → A16W4 适配
- **新建 SVG 图**：
  - `svg/tiling_split_strategy.svg`：M/N 二维切分 + 8 核 round-robin 分配 + K 方向不切分 + 多多多 vs 单单单对比
  - `svg/buffer_allocation.svg`：L1（512KB）有/无 bias 两种布局 + UB（248KB）4 buffer 布局 + 数据流箭头 + PerGroup 适配分析

#### 关键发现
- **当前 A16W8 代码**：M 和 N 方向都会切分，round-robin 分核
- **用户确认的 A16W4 策略**：多多多仅按 expert 分核（不切分），单单单仅 M 方向切分
- **尾块重切分**：仅 NZ 或 B 转置时使能，ND 非转置不使能
- **Buffer 精确数值**：WeightL1=32KB×2, AL1=220KB×2(有bias)/224KB×2(无bias), UB 固定 248KB
- **GetNdPerGroupBufferInfo() 不存在**：需要新建，pergroup 的 antiQuantScale 在 groupSize=32 + NdNk 场景下 4KB 不够

#### 影响范围
- 修正了 SRS 文档中 tiling 和 buffer 章节的不准确描述
- 为后续编码实现提供了准确的参考依据

---

### 2026-06-15 | 移除 antiquantOffset 支持（A16W4 pergroup ND 不支持 offset 输入）

**类型**: 🔧 修改
**范围**: `raw/spec/gmm_a16w4nd_op.md` + `raw/development/gmm_a16w4_pergroup_nd_srs.md` + `raw/design/03_tiling_design.md` + `raw/design/04_kernel_design.md` + SVG 图

#### 变更内容
- **`gmm_a16w4nd_op.md`**：
  - 移除 antiquantOffset 作为可选输入，改为"不使用"
  - 简化计算公式为 `y = x × weight × scale + bias`
  - 移除 antiquantOffset shape 约束
  - 更新量化模式约束为"仅 pergroup，不支持 antiquant_offset"
- **`gmm_a16w4_pergroup_nd_srs.md`**：
  - 移除 offset 数据流描述和计算公式
  - tiling key 组合从 8 个减少到 4 个（移除 2 个 offset 状态）
  - UB buffer 表移除 antiQuantOffset 行
  - Controller 地址偏移表移除 offset 行
  - VF 反量化步骤移除 "add offset" 步骤
  - 修改点汇总移除 offset 相关文件和兼容性条目
- **`03_tiling_design.md`**：
  - 移除 antiquantOffset shape 描述和校验
  - tiling key 组合从 8 个减少到 4 个
  - Buffer 结论移除 offset 引用
- **`04_kernel_design.md`**：
  - kernel 入口分支从 8 个减少到 4 个
  - WqmmConfig 从 4 个减少到 2 个（移除 _OFF 变体）
  - UB buffer 配置移除 antiQuantOffset
  - VF 函数模板参数 hasAntiQuantOffset 固定为 false
  - 反量化步骤移除 offset 加法
  - 修改点汇总和兼容性保护移除 offset 条目
- **`svg/buffer_allocation.svg`**：
  - 移除 antiQuantOffset buffer 块
  - UB 总计从 248KB 改为 244KB（剩余 4KB）
  - 数据流箭头和适配说明移除 offset 引用
  - 图例移除 antiQuantOffset 条目
- **`svg/dataflow.svg`**：
  - GM 区域移除 offset 数据
  - UB 区域移除 offset buffer
  - VF 反量化公式移除 "(+ offset)"
  - MTE2 箭头移除 offset 搬运

#### 关键决策
- A16W4 pergroup ND 数据流**不支持 antiquantOffset 输入**，仅支持对称量化（scale only）
- tiling key 组合从 2×2×2=8 减少到 2×2=4（transB × bias）
- WqmmConfig.hasAntiQuantOffset 固定为 false
- **UB 空间分配逻辑不变**：`antiQuantOffset` 的 4KB 保留（复用 `GetNdBufferInfo()`），运行时不使用但不修改分配代码

#### 影响范围
- 所有设计文档和 SVG 图均已更新
- 后续编码实现不需要处理 offset 相关逻辑
- UB buffer 分配代码无需修改

---

### 2026-06-15 | SRS 文档新增 VF 反量化详细代码实现方案

**类型**: 📝 文档
**范围**: `raw/development/gmm_a16w4_pergroup_nd_srs.md`

#### 变更内容
- **新增 §4.4 VF 反量化详细代码实现方案**（约 200 行），基于 ops-nn wqbmmv2 源码走读：
  - §4.4.1 KN 路径（`bTrans=false`）：统一 VF 函数 `AntiquantPerGroupKN` 的完整参数构造、三层循环结构（nLoop → group → k）、关键指令说明（DIST_UNPACK4_B8 / DIST_NORM / Cast / Mul / DATA_BLOCK_COPY）、输出 NZ 格式布局
  - §4.4.2 NK 路径（`bTrans=true`）：按 groupSize 分派 4 个专用 VF 函数的详细实现：
    - gs=128：`AntiquantW4Pergroup128NK`（最简单，DIST_BRC_B16 广播 scale）
    - gs=256：`AntiquantW4Pergroup256NK`（2 次 weight 加载）
    - gs=64：`AntiquantW4Pergroup64NK`（Select 拼接相邻 group scale）
    - gs=32：`AntiquantW4Pergroup32NK`（DIST_E2B_B16 + Interleave 扩展 scale）
  - §4.4.3 KN vs NK 路径对比表（7 个维度）
  - §4.4.4 迁移到 GMM 的 5 个适配要点（函数签名、UB 地址映射、hasAntiQuantOffset=false、useVag 参数、输出 NZ 格式）
- 原 §4.4 Cube 复用重编号为 §4.5

#### 关键决策
- 所有 VF 函数描述中 `hasAntiQuantOffset` 通过 `WqmmConfig` 做编译期分支，offset 相关操作在 `hasAntiQuantOffset=false` 时由编译器消除
- `useVag` 参数建议初始实现设为 `false`（直接地址计算），后续可优化
- 输出 NZ 格式与 GMM 现有 MTE3 流程一致，无需修改

#### 影响范围
- 仅 SRS 文档更新，为后续编码实现提供完整的 VF 反量化参考

---

### 2026-06-15 | 恢复 antiquantOffset 为可选参数（pergroup 支持非对称量化）

**类型**: 🔧 修改
**范围**: `raw/spec/gmm_a16w4nd_op.md` + `raw/design/03_tiling_design.md` + `raw/design/04_kernel_design.md` + `raw/development/gmm_a16w4_pergroup_nd_srs.md` + SVG 图

#### 变更内容
- **原因**：A2/A3 文档描述错误，pergroup 实际支持非对称量化，antiquantOffset 是可选参数
- **`gmm_a16w4nd_op.md`**：
  - antiquantOffsetOptional 恢复为"可选"，dtype FLOAT16/BFLOAT16/null
  - 恢复完整计算公式（含 offset 和不含 offset 两种）
  - 恢复 antiquantOffset shape 约束表
  - 量化模式约束改为"支持对称和非对称量化"
- **`03_tiling_design.md`**：
  - 恢复 antiquantOffset shape 描述（单单单 3D、多多多 2D）
  - 恢复 shape 校验（antiquantOffset shape == antiquantScale shape）
  - tiling key 组合从 4 个恢复为 8 个（2 transB × 2 offset × 2 bias）
  - OFFSET_OR_BIAS_EXIT 恢复为 0~3（4 种组合）
  - UB buffer 结论恢复 offset 引用
- **`04_kernel_design.md`**：
  - kernel 入口分支从 4 个恢复为 8 个
  - WqmmConfig 从 2 个恢复为 4 个（新增 _OFF 变体）
  - SetGmAddr 恢复 pergroup scale/offset 偏移
  - VF 反量化步骤恢复 offset 加法
  - hasAntiQuantOffset 恢复为编译期分支（非固定 false）
  - scale/offset UB 布局恢复
  - 兼容性保护恢复"有/无 offset"条目
- **`gmm_a16w4_pergroup_nd_srs.md`**：
  - 数据流描述恢复 offset
  - 计算公式恢复 offset
  - pergroup 反量化模型恢复 offset
  - tiling key 组合从 4 个恢复为 8 个
  - Controller 地址偏移表恢复 offset 行
  - VF 反量化步骤恢复 offset
  - VF 详细代码实现方案恢复 offset 加载和加法操作
  - hasAntiQuantOffset 恢复为编译期分支
  - 修改点汇总恢复 8 个分支和 scale/offset 引用
  - 兼容性恢复"有/无 offset"条目
- **SVG 图**：
  - `dataflow.svg`：GM 区域恢复 offset 数据和 MTE2 箭头，VF 公式恢复 "(+ offset)"
  - `buffer_allocation.svg`：antiQuantOffset buffer 恢复正常显示（非半透明），适配说明恢复 scale/offset

#### 关键决策
- pergroup 支持非对称量化（有 offset）和对称量化（无 offset），通过 `hasAntiQuantOffset` 编译期分支控制
- tiling key 组合为 2×2×2=8（transB × offset × bias）
- WqmmConfig 需要 4 个实例（非转置/转置 × 有/无 offset）
- UB 空间分配逻辑不变，antiQuantOffset 4KB 保留

#### 影响范围
- 所有设计文档和 SVG 图均已更新
- 后续编码实现需要处理 offset 相关逻辑（编译期分支）

---

### 2026-06-16 | 修正 SRS 中 NK 反量化 scale/offset 布局方案

**类型**: 📝 文档
**范围**: `raw/development/gmm_a16w4_pergroup_nd_srs.md` + `CHECKLIST.md`

#### 变更内容
- 在 SRS 中明确 `antiquantScale`/`antiquantOffset` 的 GM 输入布局不随 weight 转置变化，始终为 `[G,N]`（单单单为 `[E,G,N]`）
- 修正 NK（`bTrans=true`）VF 方案：不能原样使用 ops-nn 的 `[N,G]` scale/offset 索引，需要按 GMM 固定 `[G,N]` 输入使用 `gLocal*nTileAlign+nLocal` 地址公式
- 明确 NK 路径中 scale/offset 通过 `DIST_BRC_B16` 广播，并在 `groupSize=32/64` 时使用 `Select` 拼接跨 group 的寄存器
- 明确 scale/offset 必须按 VF tile 粒度搬运，不能按完整 MTE2 weight tile 大块搬运，否则 `groupSize=32` 时可能超过 1KB 单缓冲
- 明确 `PER_GROUP` 且 `hasAntiQuantOffset=true` 时需要和 scale 一样搬运 offset 到 UB
- 在 `CHECKLIST.md` 增加 VF tile 搬运、NK `[G,N]` 地址公式、broadcast/select、PER_GROUP offset 搬运检查项

#### 关键决策
- KN 路径可继续按 ops-nn `AntiquantComputeKNPerGroup` 思路迁移
- NK 路径只复用 ops-nn 的 weight unpack/cast/output NZ 和 groupSize 分派结构；scale/offset 加载采用 GMM 专用实现
- 不在 UB 中预转置 scale/offset，优先在 VF 内通过广播和 Select 构造正确寄存器，减少额外 buffer 和搬运复杂度

#### 影响范围
- 仅文档和检查清单更新，不涉及代码
- 后续编码需重点修改 `weight_quant_vec_compute.h` 与 `basic_block_vf_nd.h` 的 A16W4 pergroup ND 路径

---

### 2026-06-16 | Task 7 Tiling 实现规划完成

**类型**: 📝 文档
**范围**: `PLAN.md` + `NEXT_STEPS.md` + `reference/next_steps/7_tiling_impl.md`

#### 变更内容
- **`PLAN.md`**：
  - 阶段一标记为完成 ✅
  - 阶段二标记为当前阶段
  - 任务 7 拆分为 3 个子阶段（7.1 UT 框架搭建、7.2 Tiling 代码实现、7.3 UT 验证迭代）
  - 当前进度更新
- **`NEXT_STEPS.md`**：
  - 新增任务 7 条目，包含参考文档、目标文件、编译/UT 命令、链接到详细任务说明
- **`reference/next_steps/7_tiling_impl.md`**（新建）：
  - 完整的任务目标、参考文档、现有代码结构分析
  - 3 阶段执行计划：
    - 7.1 CSV 文件（26 列格式定义）+ CSV 驱动测试代码 + ~60 个用例覆盖
    - 7.2 tiling 代码 9 个修改点（含关键代码片段）
    - 7.3 验证步骤和迭代策略
  - 风险与应对、交付件清单、验收标准

#### 关键决策
- 采用 TDD 方式：先写测试（阶段 7.1），再实现代码（阶段 7.2），最后验证（阶段 7.3）
- 新建独立 CSV 文件和测试类（`test_grouped_weight_quant_arch35_pergroup_tiling.*`），不影响现有测试
- CSV 格式在现有 24 列基础上扩展 2 列（hasAntiquantOffset、groupSize）
- 所有 tiling 修改通过 `IsA16W4NDPergroup()` 隔离，确保 A16W8 perchannel 回归安全

#### 影响范围
- 仅工作流文档更新，不涉及代码
- 下一步按 7.1 → 7.2 → 7.3 顺序执行编码

---

### 2026-06-16 | Task 7 Tiling 实现 + UT 验证完成

**类型**: 🆕 新功能
**范围**: tiling 代码 + UT 测试 + tiling key 注册

#### 变更内容
- **`grouped_weight_quant_batch_matmul_tiling.h`**：
  - 新增 `bool IsA16W4NDPergroup() const;` 声明
- **`grouped_weight_quant_batch_matmul_tiling.cpp`**（9 个修改点）：
  - `IsA16W4NDPergroup()` 实现：`IsA16W4ND() && groupSize_ > 0`
  - `CheckGroupSize()` 新增 A16W4 pergroup 分支：校验 32/64/128/256
  - `SetAntiquantGroupSize()` 3D 分支：`IsA16W4ND()` 时 G 始终取 `GetDim(1)`（不受 transB 影响）
  - `SetAntiquantGroupSize()` 2D 分支：多多多 pergroup `groupSize_ = kList_[0] / scaleDim0`
  - `SetTilingKey()` antiquantType：新增 `PER_GROUP` 分支
  - `SetTilingKey()` optionInputSituation：pergroup 时编码 offset+bias（4 种组合）
  - `CheckTensorDimSingleXSingleWeightSingleY()`：scale/offset 接受 2D 或 3D
  - `CheckTensorDimMultiXMultiWeightMultiY()`：scale/offset 接受 1D 或 2D
  - `SetBaseTiling()`：已有 `groupSize` 字段传播，无需修改
- **`weight_quant_tiling_key.h`**：
  - 新增 8 个 `ASCENDC_TPL_ARGS_SEL` 组合（2 transB × 4 offset/bias），全部使用 `WQGMM_PER_GROUP` + `WQGMM_ND` + `WQGMM_N_FIRST_BASIC_BLOCK` + `WQGMM_VECTOR_ANTIQUANT`
- **`test_grouped_weight_quant_arch35_pergroup_tiling.cpp`**（新增，339 行）：
  - CSV 驱动参数化测试，参考 `test_grouped_quant_arch35_tiling.cpp` 模式
  - 支持单单单（groupType=0）和多多多（groupType=-1）两种场景的 shape 构造
- **`test_grouped_weight_quant_arch35_pergroup_tiling.csv`**（新增，37 个用例）：
  - 正常通过：单单单 12 个 + 多多多 9 个 + 边界 5 个
  - 异常拦截：gs 非法 3 个 + K%gs≠0 1 个 + dtype 非法 2 个 + transA 1 个 + NZ 1 个 + groupType 非法 2 个 + K=0 1 个

#### 关键决策
- 采用 TDD 方式：先写测试（7.1），再实现代码（7.2），最后验证（7.3）
- 所有 tiling 修改通过 `IsA16W4NDPergroup()` 隔离，确保 A16W8 perchannel 回归安全
- pergroup scale 布局固定为 `[E, G, N]`（单单单）或 `[G_i, N_i]`（多多多），不受 transB 影响
- optionInputSituation 对 pergroup 编码 offset+bias（4 种），对非 pergroup 保持原有 offset-only（2 种）

#### 验证结果
- 编译通过 ✅
- pergroup UT 37/37 通过 ✅
- 原有 WeightQuant UT 13/13 无回归 ✅
- LSP diagnostics 零错误 ✅

#### 影响范围
- 新增 A16W4 PerGroup ND 数据流的 tiling 支持
- 原有 A16W8 perchannel、A8W4、S8S4 路径不受影响
- 下一步：Task 8 Kernel 适配

---

### 2026-06-17 | 任务 8 Kernel 适配详细规划

**类型**: 📝 文档
**范围**: `PLAN.md` + `NEXT_STEPS.md` + `reference/next_steps/8_kernel_impl.md`

#### 变更内容
- **新建 `reference/next_steps/8_kernel_impl.md`**（~750 行）：任务 8 详细任务说明
  - 现有代码结构分析（apt.cpp 6 个分支、WqmmConfig 3 个实例、Controller/VecCompute/VF 现状）
  - 6 个子任务拆分（8.1~8.6），按依赖顺序排列
  - 每个子任务的修改文件、代码修改方案、验收标准
  - 完整的参考代码索引（GMM + ops-nn wqbmmv2）
  - 风险分析与应对
- **更新 `NEXT_STEPS.md`**：新增任务 8 索引条目
- **更新 `PLAN.md`**：任务 8 拆分为 6 个子任务，当前进度更新

#### 关键决策
- **config950 需新增 4 个 dtype 组合**：现有 3 个 A16W4 ND 条目均含 antiquant_offset，需补充 4 个 antiquant_offset=null 的组合（FP16: bias=FP16/null，BF16: bias=BF16/FLOAT）
- **编译命令**：使用 `bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8`（当前环境无法运行 UT，仅编译验证）
- **子任务拆分策略**：按代码层次自底向上（配置层 → Controller → VecCompute → VF KN → VF NK → 编译验证），每个子任务可独立编译验证
- **VF 函数策略**：KN 路径统一 VF（所有 groupSize 共用），NK 路径按 groupSize 分派 4 个专用 VF
- **Buffer 配置**：复用 `GetNdBufferInfo()`，不新增 pergroup buffer 配置

#### 影响范围
- 仅工作流文档更新，不涉及代码
- 下一步按 8.1 → 8.2 → 8.3 → 8.4 → 8.5 → 8.6 顺序执行编码

---

### 2026-06-17 | Task 8 Kernel 适配完成 — GMM A16W4 PerGroup ND

**类型**: 🆕 新功能
**范围**: 6 个文件修改，新增 A16W4 INT4 pergroup ND 数据流的 kernel 支持

#### 变更内容

**修改文件**（6 个）：

| 文件 | 修改内容 |
|------|----------|
| `basic_block_config.h` | 新增 4 个 WqmmConfig 实例（A16W4_NDKN_PG/OFF, A16W4_NDNK_PG/OFF） |
| `grouped_matmul_apt.cpp` | 新增 8 个 if constexpr 分支（2 transB × 2 offset × 2 bias） |
| `grouped_matmul_def.cpp` | config950 新增 4 个 dtype 组合（antiquant_offset=null），154→158 条目 |
| `grouped_matmul_weight_quant_basic_controller.h` | SetGmAddr pergroup 偏移（kSize/groupSize × nSize）+ biasBaseOffset_ 独立偏移 |
| `weight_quant_vec_compute.h` | CalLocalAddrForVf PER_GROUP 地址 + CopyAntiQuantParamsGmToUb offset 搬运 + AntiQuantProcessNd pergroup 分发 |
| `basic_block_vf_nd.h` | NdKnLoadScaleOffset PER_GROUP 分支 + AntiQuantInt4PerGroupNdKn（统一 VF）+ AntiQuantInt4PerGroupNdNk 分派器 + 4 个 gs 专用 VF（Gs32/Gs64/Gs128/Gs256） |

#### 关键决策

1. **Buffer 配置复用**：PER_GROUP + ND 路由到 `GetNdBufferInfo()`，不新增 pergroup buffer 配置
2. **bias 偏移独立**：新增 `biasBaseOffset_` 成员变量，bias 地址不再与 antiquantScale/offset 共用偏移
3. **NK scale 地址公式**：GMM 固定 `[G,N]` 布局，地址 = `groupIdx * VEC_MAX_ELEM_B16 + nBubIdx`，不使用 ops-nn 的 `[N,G]` 索引
4. **gs=32 NK 实现**：8 次 DIST_BRC_B16 + 4 次 Interleave + 2 次 Select（GMM [G,N] 布局下不能用 DIST_E2B_B16）
5. **KN 统一 VF**：所有 groupSize 共用一个函数，通过 groupNum/groupTail 处理尾部

#### 编译验证

- 编译命令：`bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8`
- 结果：**0 errors**，1574 pre-existing warnings
- 最终审查：42/42 检查项全部通过

#### 影响范围

- 新增 A16W4 PerGroup ND 数据流支持
- 原有 A16W8 perchannel、A8W4、S8S4 路径不受影响（`if constexpr` 隔离）
- 下一步：阶段三验证（UT + 兼容性回归，需 NPU 硬件环境）

---

### 2026-06-24 | 修复 A16W4 Pergroup ND NK 反量化

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 转置（NK）路径

#### 变更内容
- 修正 gs=64/128/256 NK 路径 weight N 行步长，使用 `vecConfig.ubMte2InnerSize >> 1` 匹配 `CopyWeightGmToUb` 的 UB 行布局
- 修正 NK 输出 NZ block stride，统一使用 `WEIGHT_F16_UB_NZ_STRIDE`，避免按 `bubNLen` 动态计算导致 L1 搬运源布局不一致
- 修正 gs=32 scale/offset 寄存器拼接，改为 8 次 `DIST_BRC_B16` 后用 `Select` 组合 32 元素分组
- 移除 group 循环内 `maskLen` 重置，保留 `UpdateMask` 跨 group 递减处理 K 尾块

#### 关键决策
- GMM NK pergroup scale/offset UB 布局固定为 `[G_tile, N_tile_align]`，仍按 `groupIdx * VEC_MAX_ELEM_B16 + nBubIdx` 寻址
- weight UB 的 N 行 stride 来源于 MTE2 搬运配置，不由当前 VF 的 `bubKLen` 推导

#### 影响范围
- 仅影响 A16W4 pergroup ND 且 `bTrans=true` 的 NK 反量化路径
- 原有 perchannel、KN 路径和非 INT4 路径不受影响
- 编译验证：`bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8` 通过（exit code 0）

---

### 2026-06-24 | 修复 A16W4 Pergroup ND NK gs128 双段反量化

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 转置（NK）、`groupSize=128` 路径

#### 变更内容
- 修正 `AntiQuantInt4PerGroupNdNkGs128`，移除第二组 128K weight 加载、反量化和写回，避免 `K=128` 时误读下一段 UB 数据
- 保留 group 维循环：`K=128/groupSize=128` 时只迭代 1 次，`K>128` 时按 group 逐段处理
- 明确 `CopyWeightGmToUb` 中 `vecConfig.ubMte2InnerSize=256` 在 NK 场景作为 N 行 stride 使用，不表示当前 VF 的有效 K 必须为 256

#### 关键决策
- gs128 每个 group 只对应一个 128 元素 MicroAPI 向量，不应像 gs256 一样处理两个 128K 向量
- 本次仅收敛 gs128 NK 反量化数据流，不调整 MTE2 搬运或 scale/offset 布局

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=true` 且 `groupSize=128` 的反量化精度
- 原有 perchannel、KN 路径和非 INT4 路径不受影响
- 编译验证：`bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8` 通过（exit code 0）

---

### 2026-06-24 | 重实现 A16W4 Pergroup ND NK gs32 反量化

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 转置（NK）、`groupSize=32` 路径

#### 变更内容
- 重实现 `AntiQuantInt4PerGroupNdNkGs32`，按 256K inner 分块处理 8 个 32K group
- 按 GMM `[G,N]` scale/offset UB 布局使用逐 group `DIST_BRC_B16` 加 `Select` 拼接 32-lane scale/offset 段
- 对 `K<256` 尾块避免读取不存在的第二个 128K weight 向量和 group 4-7 scale/offset

#### 关键决策
- 未复用 WQBMMv2 的 `DIST_E2B_B16`，因为 GMM scale/offset 在 UB 中按 group 行存放，8 个 group 对同一 N 不连续
- 保持 NK weight 行 stride 为 `vecConfig.ubMte2InnerSize >> 1`，输出 stride 使用 `WEIGHT_F16_UB_NZ_STRIDE`

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=true` 且 `groupSize=32` 的反量化精度
- 原有 perchannel、KN 路径和非 INT4 路径不受影响
- 编译验证：`bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8` 通过（exit code 0）

---

### 2026-06-24 | 重写 A16W4 Pergroup ND NK gs128 反量化

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 转置（NK）、`groupSize=128` 路径

#### 变更内容
- 重写 `AntiQuantInt4PerGroupNdNkGs128`，按单个 128K vector 完成 weight 加载、反量化和 NZ 写回
- 按 GMM `[G,N]` scale/offset UB 布局使用 `groupIdx * VEC_MAX_ELEM_B16 + nBubIdx` 寻址
- 保持 NK weight N 行步长为 `vecConfig.ubMte2InnerSize >> 1`，输出 stride 使用 `WEIGHT_F16_UB_NZ_STRIDE`

#### 关键决策
- gs128 每个 group 恰好等于一个 128-lane MicroAPI 向量，不复用 gs256 的第二段加载/写回逻辑
- `UpdateMask` 保持 POST_UPDATE 语义，跨 group 递减处理 K 尾块，避免尾块写回越界
- 本次仅修改 VF 层 gs128 函数，不调整 MTE2 搬运、scale/offset UB 布局或其他 groupSize 分支

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=true` 且 `groupSize=128` 的反量化精度
- 原有 perchannel、KN 路径、gs32/64/256 路径和非 INT4 路径不受影响
- 编译验证：`bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8` 通过（exit code 0）

---

### 2026-06-24 | 调整 A16W4 Pergroup ND NK gs128 为 group-first 实现

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 转置（NK）、`groupSize=128` 路径

#### 变更内容
- 将 `AntiQuantInt4PerGroupNdNkGs128` 改为 `groupNum -> N` 双层 `for` 循环，避免在 `__VEC_SCOPE__` 中使用运行期 `if`
- weight 按转置 `[N,K]` UB 布局读取：`nBubIdx * (vecConfig.ubMte2InnerSize >> 1) + groupIdx * 64`
- scale/offset 按 GMM 非转置 `[K//GS,N]` UB 布局读取：`groupIdx * VEC_MAX_ELEM_B16 + nBubIdx`
- 输出继续使用 `WEIGHT_F16_UB_NZ_STRIDE`，每个 group 完成 `bubNLen` 行后跳到下一个 128K NZ block

#### 关键决策
- 与 wqbmmv2 gs128 的 `group -> N` 数据流保持一致，仅调整 scale/offset 地址公式以适配 GMM `[G,N]` 布局
- 保持 `UpdateMask` 跨 group 递减，覆盖 `bubKLen=128/256` 以及 K 尾块
- 本轮不进行编译，按用户要求仅做静态检查

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=true` 且 `groupSize=128` 的反量化精度
- 原有 perchannel、KN 路径、gs32/64/256 路径和非 INT4 路径不受影响
- 静态检查：`git diff --check` 通过；未编译

---

### 2026-06-24 | 修复 A16W4 Pergroup ND NK gs64 反量化

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 转置（NK）、`groupSize=64` 路径

#### 变更内容
- 修正 `AntiQuantInt4PerGroupNdNkGs64`，每个 `innerIdx` 只处理一个 128-lane weight 寄存器，不再同时处理第二个 128K 段
- 保持 scale/offset 构造方式：按 GMM `[K//GS,N]` UB 布局加载相邻两个 group，再用 `Select(maskH)` 拼成前 64 lane / 后 64 lane
- weight 读取保持转置 `[N,K]` packed INT4 布局：`outerIdx * (vecConfig.ubMte2InnerSize >> 1) + innerIdx * 64`
- 输出保持 `WEIGHT_F16_UB_NZ_STRIDE`，每个 128K inner 块写一个 NZ block

#### 关键决策
- gs64 与 wqbmmv2 的 `inner(K/128) -> outer(N)` 结构保持一致；GMM 只调整 scale/offset 地址公式为 `[G,N]`
- 删除原实现中第二个 128K weight/output 路径，避免 `K[128:256]` 使用 `group0/1` 的 scale/offset 重复反量化
- 本轮不进行编译，按用户要求仅做静态检查

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=true` 且 `groupSize=64` 的反量化精度
- 原有 perchannel、KN 路径、gs32/128/256 路径和非 INT4 路径不受影响
- 静态检查：`git diff --check` 通过；未编译

---

### 2026-06-24 | 修复 A16W4 Pergroup ND NK gs32 反量化

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 转置（NK）、`groupSize=32` 路径

#### 变更内容
- 重构 `AntiQuantInt4PerGroupNdNkGs32`，每个 `innerIdx` 只处理一个 128-lane weight 寄存器，对应 4 个 32K group
- 按 GMM `[K//GS,N]` UB 布局加载 4 个 group 的 scale/offset：`(groupBase + i) * VEC_MAX_ELEM_B16 + n`
- 通过 `Select(mask32/mask96/maskH)` 构造 `[A|B|C|D]` 四段 32-lane scale/offset 寄存器
- 删除 gs32 参数加载 helper 中的运行期 `if` 和原实现中的 `hasSecondVector` 运行期分支，保持 `__VEC_SCOPE__` 内为固定 `for` 数据流

#### 关键决策
- gs32 与 gs64/gs128 保持同一抽象：每次 inner 处理一个 128K 向量，输出一个 NZ block
- GMM scale/offset 固定 `[G,N]` 布局，不能复用 wqbmmv2 NK 的 `DIST_E2B_B16` 连续 group 加载
- 本轮不进行编译，按用户要求仅做静态检查

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=true` 且 `groupSize=32` 的反量化精度
- 原有 perchannel、KN 路径、gs64/128/256 路径和非 INT4 路径不受影响
- 静态检查：`git diff --check` 通过；未编译

---

### 2026-06-26 | 修复 A16W4 Pergroup ND KN gs32 scale/offset stride

**类型**: 🐛 修复
**范围**: arch35 A16W4 INT4 pergroup ND、weight 非转置（KN）、`groupSize=32` 泛化精度

#### 变更内容
- `CopyAntiQuantParamsGmToUb` 中 pergroup KN 的 scale/offset 目标行步长改为当前 N tile 对齐值，保持 UB 布局为 `[G_tile, N_tile_align]`
- `CalLocalAddrForVf` 和 `AntiQuantInt4PerGroupNdKn` 统一使用同一个 `antiQuantParamNStride_` 作为 group 行步长
- 新增 `{bufferNum=2, ubMte2InnerSize=256}` 的 tiling key 枚举/宏，A16W4 ND pergroup host tiling 与 kernel 分支显式使用该 key

#### 关键决策
- KN 路径继续复用统一 VF 实现，不按 groupSize 新增 gs32 专用逻辑；问题根因是搬运 stride 与 VF 寻址 stride 不一致
- perchannel/MX 仍保留原 `{bufferNum=4, ubMte2InnerSize=256}` key；A16W4 ND pergroup 单独走 `{2,256}`
- 本轮不进行编译，按用户要求仅做静态检查

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=false` 的 scale/offset UB 布局，重点修复 `groupSize=32` 泛化精度
- A16W4 pergroup ND `bTrans=true` 继续使用 `{2,256}` 配置；perchannel、MX、NZ 路径不受影响
- 静态检查：`git diff --check` 通过；未编译

---

### 2026-06-26 | 重写 A16W4 Pergroup ND KN INT4 反量化

**类型**: 🐛 修复
**范围**: `basic_block_vf_nd.h` 中 arch35 A16W4 INT4 pergroup ND、weight 非转置（KN）路径

#### 变更内容
- 重写 `AntiQuantInt4PerGroupNdKn`，参考 wqbmmv2 `AntiquantComputeKNPerGroup` / `AntiquantPerGroupKN` 的 `nLoop -> groupNum -> groupTail` 数据流
- 使用 `CreateAddrReg` 加载 `[G_tile, N_tile_align]` scale/offset，weight 按 KN UB 布局使用 `nIdx / groupIdx / kIdx` 地址寄存器读取
- 输出改为单 `weightOutBaseAddr` 的 `DATA_BLOCK_COPY + POST_MODE_UPDATE` 方式，每个 N 向量块处理完 main groups 和 tail group 后按 `weightOutStride` 跳转
- `__VEC_SCOPE__` 内仅保留 `for` 和 `if constexpr`，移除运行时 `if (groupTail > 0)` 分支

#### 关键决策
- 函数参数显式区分当前 VF 的实际 `bubNLen` 和 scale/offset 的 `antiQuantParamNStride`，避免将搬运 stride 当作本次 N 长度
- weight 行跨度继续使用 GMM 的 `vecConfig.ubMte2InnerSize >> 1`，适配统一 `{bufferNum=2, ubMte2InnerSize=256}` 配置
- 本轮不进行编译，按用户要求仅做静态检查

#### 影响范围
- 影响 A16W4 pergroup ND `bTrans=false` 的 INT4 pergroup 反量化，覆盖 groupSize 32/64/128/256 统一路径
- A16W4 pergroup ND `bTrans=true` 的 gs 专用 NK 路径、perchannel、MX、NZ 路径不受影响
- 静态检查：`git diff --check` 通过；未编译
