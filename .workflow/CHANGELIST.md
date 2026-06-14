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
