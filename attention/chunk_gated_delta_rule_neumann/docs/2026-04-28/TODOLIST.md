# TODO - 2026-04-28

**目标**: 开始实现 Stage1 Neumann 初始化同步修复，并为 AIC `InitInverseMatrix()` 迁移做准备。

---

## 实施 TODO

| ID | 状态 | 任务 | 文件 |
|---|---|---|---|
| T1 | 完成 | 启动 sub-agents 分析同步流和可复用实现模式 | N/A |
| T2 | 完成 | 创建实施计划文档 | `docs/2026-04-28/AIC_INIT_INVERSE_IMPLEMENTATION_PLAN.md` |
| T3 | 完成 | 修复 Stage1 Neumann 初始化半块写入和 full-matrix 累加职责 | `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` |
| T4 | 完成 | 静态核对 flag 成对关系，避免死锁 | `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` |
| T5 | 待开始 | 编译验证 | `build.sh` |
| T6 | 待开始 | pytest 精度验证 | `tests/pytest/test_run.sh` |
| T7 | 完成 | 更新工作日志、进度文档、Debug History | `docs/2026-04-28/*`, `AGENTS.md` |

---

## 子代理任务

| 子代理 | 状态 | 任务 |
|---|---|---|
| Turing | 完成 | 分析 Stage1 AIC/AIV 同步和最小改造方案 |
| Anscombe | 完成 | 查找相邻算子可复用模式 |

---

## 当前阻塞和风险

| 风险 | 状态 | 处理 |
|---|---|---|
| AIC 不能直接执行 vector 初始化 | 已确认风险 | 第一阶段不强行迁移逐元素初始化到 AIC |
| AIC 读取半初始化矩阵 | 缓解中 | 修复半块对角线写入，保留 mode=0x2 同步语义 |
| flag 复用造成死锁 | 已静态核对 | 保持三轮 Neumann 原 flag 不变 |

---

## 本次代码改动

| 改动 | 说明 |
|---|---|
| 修复 `I-L` 对角线写入 | 第二个 AIV subBlock 的 UB 行号应使用本地行 `i`，不是全局行 `subOffset_ + i` |
| 限定 `NeumannAddCompute()` 写者 | `INV += tmp` 是完整 `16×16` 矩阵累加，只允许 `subBlockIdx_ == 0` 写回，避免两个 AIV subBlock 同写 `INVWsGm_` |
| 保留当前同步协议 | `0x1/0x2`、`0x3/0x4`、`0x1/0x2` 三轮不改，降低死锁风险 |

---

## 子代理结论吸收

| 结论 | 处理 |
|---|---|
| 完整 `L` 不能只来自 `kkWsGm_`，还需要 `beta`、`gamma/decay`、mask | 计划中不再假设 AIC 直接从 `kkWsGm_` 初始化 |
| `0x7` 应保留为 AIV→AIC 的“`L_powWsGm_` 就绪”语义 | 本轮保留 `0x7` 方向 |
| AIC 分支未初始化 vector UB/queue | 本轮不直接实现 AIC vector 版 `InitInverseMatrix()` |
| `NeumannAddCompute()` 只应一个 subBlock 写回，但两个 subBlock 继续参与 ack | 本轮已限定写回者为 `subBlockIdx_ == 0` |

---

## 下一轮 TODO

1. 编译 `chunk_gated_delta_rule_neumann`，确认新增代码可通过 Ascend C 编译。
2. 运行单算子 pytest，观察是否仍有 507015 挂起。
3. 若仍挂起，继续拆分 flag：把初始化完成和 `INV += tmp` 完成的 subBlock 参与关系显式化。
4. 编译通过后再评估真正 AIC 版 `InitInverseMatrix()` 是否需要新增 AIC 侧 vector buffer。
