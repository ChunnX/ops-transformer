# AIC InitInverseMatrix 实施计划 - 2026-04-28

**日期**: 2026-04-28  
**目标**: 开始实现 Neumann 初始矩阵初始化改造，解决 `TASK_RATIO=2` 下 AIC 读取不完整 `16×16` 矩阵的问题。

---

## 背景

当前 `ProcessOneChunk()` 的 Stage1 数据流仍是：

1. AIV 写 `queryContinousGm_` / `keyContinousGm_`。
2. AIC 计算 `kkWsGm_ = key @ key.T`。
3. AIV `KKBetaCompute()` 计算半块 `L`。
4. AIV `InverseCompute()` 写半块 `L_powWsGm_` 和 `INVWsGm_`。
5. AIV 通过 flag `0x7` 通知 AIC。
6. AIC 立刻执行 `NeumannLpowCompute()` / `NeumannMmulCompute()`。

问题是 `TASK_RATIO=2` 下两个 AIV subBlock 各处理 `8×16`，AIC 只等待一个 `0x7` 事件就可能开始读完整 `16×16`，存在读到半初始化矩阵的风险。

---

## 实施原则

| 原则 | 说明 |
|---|---|
| 最小改造 | 先修复 `16×16` 初始化完整性和同步，不重构 Stage2/Stage3 |
| 保持 `TASK_RATIO=2` | 不改变现有 AIC/AIV 分工和 core 映射 |
| 保留 Neumann 三轮流程 | 继续使用 `L_pow = L@L`、`tmp = INV@L_pow`、`INV += tmp` |
| 明确 AIC/AIV边界 | AIV 负责 beta/gamma/mask 等逐元素处理；AIC 负责矩阵乘 |

---

## 技术方案

### 方案 A：严格 AIC 初始化（目标方案）

新增 `InitInverseMatrix()`，由 AIC 生成完整 `L_powWsGm_ = L` 和 `INVWsGm_ = I-L`。

实施前提：
- AIC 侧需要可用的数据源包含完整 `L`。
- 如果 AIC 不能直接执行逐元素 beta/gamma/mask/对角线写入，则需要 AIV 先把完整 `L` 写入中间 workspace，再由 AIC 消费。

风险：
- 当前 AIC 分支未初始化 vector UB/queue，不能假设可直接使用 `Duplicate`、`Muls`、`DataCopyPad` 这类 vector 路径。

### 方案 B：同步修复优先（可落地的第一步）

保留 AIV 生成半块 `L` / `I-L`，但修改同步协议，使 AIC 确认两个 AIV subBlock 都完成后再启动 Neumann MMA。

预期改造：
- subBlock 0 完成后设置一个 flag。
- subBlock 1 完成后设置另一个 flag。
- AIC 等待两个 flag 后再执行 `NeumannLpowCompute()`。
- 保留 `NeumannAddCompute()`、`GBKCompute()`、`VBetaCompute()`、`QPrimeCompute()`。

该方案不违背最终目标：它先保证完整 `16×16` 输入，再为后续把初始化进一步搬到 AIC 留出边界。

---

## 当前推荐路径

1. 第一阶段采用方案 B，修复 AIC 过早读取的问题，降低 kernel 挂起风险。
2. 第二阶段评估 AIC 侧是否能安全执行矩阵初始化所需的 vector 操作。
3. 第三阶段若 AIC 能执行初始化，则将 `InverseCompute()` 中的初始化职责迁移为 AIC `InitInverseMatrix()`。

原因：
- 当前代码中 `L` 的完整定义依赖 `beta`、`g`、`stageOneMask_`，这些都在 AIV vector 路径中生成。
- 直接在 AIC 写 `I-L` 可能会引入不可编译或不可执行的 vector 指令。
- 同步修复是最小、可验证、能直接解除“读半块”的实施步骤。

---

## 关键修改点

| 文件 | 修改点 | 说明 |
|---|---|---|
| `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` | `ProcessOneChunk()` | 调整 AIC 等待两个 AIV subBlock 初始化完成 |
| `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` | `InverseCompute()` | 第一阶段保留，继续写 `L_powWsGm_` / `INVWsGm_` 对应半块 |
| `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` | `NeumannAddCompute()` | 保留，继续执行三轮 `INV += tmp` |
| `docs/2026-04-28/TODOLIST.md` | 实施跟踪 | 逐项记录状态和验证结果 |

---

## 验证计划

1. 静态检查同步 flag 是否成对出现，避免 AIC/AIV 互等。
2. 编译 `chunk_gated_delta_rule_neumann`。
3. 安装自定义算子包。
4. 运行 pytest 单算子测试。
5. 若仍挂起，按 `SKILL/kernel-debugging` 记录 flag 位置和 dump 数据。

