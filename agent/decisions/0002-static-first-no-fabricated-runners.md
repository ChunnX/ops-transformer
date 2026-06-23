# 0002: Static First, No Fabricated Runners

## Status

Accepted.

## Context

`ops-math` 当前工作树已经拥有：

- `scripts/st_accuracy/run_st_accuracy.py`
- `scripts/agent_context/validate_agent_context.py`
- `agent_ops/<op>/` 单算子账本
- `.agents/skills/`

`ops-transformer` 最初静态扫描没有这些基建。当前已迁移裁剪 `scripts/st_accuracy` 作为两个重点 attention 算子的局部能力，并补充对应 host golden compare；当前已有 `.agents/skills` 作为 repo-local performance skill 入口。首轮真实性能闭环后，当前已新增 `agent_ops/lightning_indexer/` 和 `agent_ops/recurrent_gated_delta_rule/` 两个轻量账本；仍没有 `scripts/agent_context`。

## Decision

`ops-transformer` 文档只能引用当前仓库真实存在的脚本、目录和验证入口。可以把 `ops-math` 的治理结构迁移为文档组织方式，但不能把不存在或尚未覆盖的 ST runner、agent context 校验、单算子账本写成已具备能力。`.agents/skills` 可以作为当前已存在能力记录，但其中从 `ops-math` 迁移的案例必须按本仓库重新验证。

## Consequences

- 精度入口记录为 C++ UT、example、部分 pytest、部分 ST 资产、两个重点 attention 算子的局部 `scripts/st_accuracy` runner 和 CI UT 脚本。
- 性能入口记录为 example 二进制加 `msprof`。
- 文档校验使用 `find`、`rg`、算子表计数等静态方式。
- 后续如扩大 ST runner 覆盖、继续新增 `agent_ops/<op>/` 或新增 skill 触发规则，必须重新更新 `current-state.md`、`INDEX.md` 和相关 runbook。
