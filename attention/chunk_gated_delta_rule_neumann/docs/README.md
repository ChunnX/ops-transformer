# chunk_gated_delta_rule_neumann 文档索引

**最后更新**: 2026-04-29

---

## 目录规范

- 文档按真实日期归档到 `docs/YYYY-MM-DD/`。
- 当日进度使用 `PROGRESS_AND_ISSUES.md`。
- 当日工作记录使用 `WORK_LOG.md`。
- 被后续方案替代但仍有参考价值的文档放到对应日期的 `archive/`。

---

## 当前有效文档

| 日期 | 文档 | 用途 |
|---|---|---|
| 2026-04-29 | `docs/2026-04-29/PROGRESS_AND_ISSUES.md` | 当前最新状态与下一步 |
| 2026-04-29 | `docs/2026-04-29/WORK_LOG.md` | 今日文档整理和状态确认记录 |
| 2026-04-28 | `docs/2026-04-28/DIRECT_ACLNN_PRECISION_FIX_PLAN.md` | 单算子直调精度修复与验证结论 |
| 2026-04-28 | `docs/2026-04-28/PROGRESS_AND_ISSUES.md` | 2026-04-28 最终进度记录 |
| 2026-04-28 | `docs/2026-04-28/AIC_INIT_INVERSE_IMPLEMENTATION_PLAN.md` | AIC 初始化长期方案和风险分析 |
| 2026-04-28 | `docs/2026-04-28/AIC_INVERSE_GENERATION_PLAN.md` | AIC 生成初始矩阵原始方案 |
| 2026-04-27 | `docs/2026-04-27/CHUNK16_NEUMANN_MIGRATION_PLAN.md` | CHUNK=16 + Neumann 初始迁移计划 |
| 2026-04-27 | `docs/2026-04-27/PROJECT_EXECUTION_LOG.md` | 克隆、构建和早期调试历史 |

---

## 归档文档

| 文档 | 原因 |
|---|---|
| `docs/2026-04-28/archive/EXECUTION_PLAN_SUPERSEDED.md` | 已被 `DIRECT_ACLNN_PRECISION_FIX_PLAN.md` 和最新进度替代 |
| `docs/2026-04-28/archive/PROGRESS_AND_ISSUES_INITIAL.md` | 初始进度，已被同日最终进度替代 |
| `docs/2026-04-28/archive/WORK_LOG_AIC_PLAN.md` | AIC 方案确认阶段日志，已合并到最终工作记录 |

---

## 当前状态快照

- 当前 chunk：`16`。
- 当前求逆：Neumann 三步倍增，`L²/L⁴/L⁸` 通过 AIC FP32 matmul，`INV += tmp` 通过 AIV FP32 add。
- 当前初始化：AIV 两个 subBlock 写完整 `L_powWsGm_` / `INVWsGm_`，AIC 读取完整矩阵做 Neumann；纯 AIC `InitInverseMatrix()` 尚未落地。
- 当前验证：aclnn 单算子直调已通过默认、变量长度、短尾块、GQA、`dk=dv=32` 用例。
