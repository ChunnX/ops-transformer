# AGENTS.md — 算子开发工作流

> **Agent 编排 + 文件驱动** 的开发工作流模板。主 agent 编排规划，subagent 执行任务，通过 `.workflow/` 文件实现跨会话、跨 agent 的信息持久化与协同。

## 项目特定指导（当前任务：GMM A16W4 INT4 ND for Ascend 950）

### 范围边界

- **本任务范围**：仅 tiling + kernel（arch35 平台）
- **不做**：Aclnn 通路（API checker、infershape）— 由其他人完成
- **参照实现**：`GMMWeightQuantBasicController`（A16W8 多多多），不是 A2/A3 的 antiquant 体系

## 工作流架构

```
主 Agent（编排者/项目管理）
  ├── 读取 .workflow/ → 掌握全局状态
  ├── 制定/调整 PLAN.md → 规划任务
  ├── 分发任务 → subagent 执行（开发、测试、评审等）
  └── 更新 .workflow/ → 记录进展与变更

Subagent（执行者）
  ├── 接收精准上下文 → 完成具体任务
  ├── 最小化上下文 → 提高可靠性
  └── 产出回写 .workflow/ → 保证信息不丢失
```

**核心理念**: 文件是 agent 间的通信机制。每个 agent 上下文独立且有限，通过文件传递状态，确保复杂任务的连贯性。

## .workflow/ 目录结构

| 文件/目录 | 职责 | 更新频率 |
|-----------|------|----------|
| [GOAL.md](.workflow/GOAL.md) | 总体目标与方向，确保不偏离 | 目标进展时 |
| [PLAN.md](.workflow/PLAN.md) | 多阶段任务规划与拆解 | 阶段变更时 |
| [NEXT_STEPS.md](.workflow/NEXT_STEPS.md) | 用户任务索引：概要 + 状态 + 链接到详细任务说明 | 用户更新时 |
| [CHECKLIST.md](.workflow/CHECKLIST.md) | 各阶段质量检查项 | 发现新检查项时 |
| [CHANGELIST.md](.workflow/CHANGELIST.md) | 重要变更记录（追加式） | 每次重要变更后 |
| [reference/](.workflow/reference/) | 参考文档：方法论、详细任务说明（`next_steps/`）等 | 按需 |
| [raw/](.workflow/raw/) | 原始数据：需求规格（`spec/`）+ 参考文档（`reference/`），详见 [raw/INDEX.md](.workflow/raw/INDEX.md) | 按需追加，**不可修改** |

## 会话生命周期

### 启动新会话

按顺序读取（渐进式披露，宏观→微观）：

1. **本文档** → 了解工作流方法论
2. **GOAL.md** → 总体目标和方向
3. **PLAN.md** → 当前阶段任务规划
4. **NEXT_STEPS.md** → 用户下一步指示
5. 如有歧义 → **raw/INDEX.md** 检索所需原始文档

### 工作期间

| 场景 | 行为 |
|------|------|
| 做重大决策前 | 读取 CHECKLIST.md 逐项检查 |
| 发现新信息 | 立即写入对应文件 |
| 不确定原始意图 | 查阅 raw/INDEX.md 定位原始文档 |
| 收到用户新指示 | 更新 NEXT_STEPS.md，调整 PLAN.md |

### 完成工作后

1. 按 CHECKLIST.md 逐项验证
2. 在 CHANGELIST.md 追加变更记录（**严格按格式**）
3. 更新 PLAN.md 和 NEXT_STEPS.md 的任务状态
4. 如目标进度变更，更新 GOAL.md

## 核心原则

### 一：关键信息必须落盘
**不依赖上下文记忆。** 有价值的信息 → 立即写入文件。上下文是临时的，文件是持久的。

### 二：原始信息完整保留
用户原始输入、设计意图以**未压缩、未改写**的形式保存在 `raw/` 中。保持原始语言，不翻译不改写。raw/ 是 **truth source**，其他文件是衍生物。

### 三：主 agent 编排，subagent 执行
主 agent 是项目管理者 — 规划任务、分发工作、验证结果。subagent 用最小精准上下文完成具体任务，降低上下文污染风险。

---

## CHANGELIST 记录格式

```markdown
### YYYY-MM-DD | 变更标题

**类型**: 🆕 新功能 | 🔧 修改 | 🐛 修复 | 🗑️ 删除 | 📝 文档
**范围**: 影响范围

#### 变更内容
- 具体变更

#### 关键决策
- 决策及原因

#### 影响范围
- 对现有功能的影响
```

**只追加，不删除已有条目。**
