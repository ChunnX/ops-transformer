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
