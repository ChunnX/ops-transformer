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
