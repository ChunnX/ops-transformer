# Agent Change Boundary

本文记录后续工作中建议保持分离的变更边界。当前任务只涉及 agent 文档重构，不涉及算子源码、构建产物或 profiling raw 证据。

## Boundary A: Agent Context

适合放在同一提交：

- `AGENTS.md`
- `agent/INDEX.md`
- `agent/current-state.md`
- `agent/comparison-ops-math.md`
- `agent/policies/`
- `agent/runbooks/`
- `agent/decisions/`
- `agent/lessons/`
- 既有 `agent/*-flow.md`、`agent/*-structure.md` 等 reference 文档的维护性修订

不应包含：

- 算子源码改动。
- `build/`、`output/`、`build_out/`。
- raw `msprof` dump。
- `.o`、`.bin`、完整 CSV 或长日志。

## Boundary B: Operator Source Changes

适合按单算子或小范围公共依赖分提交：

- `<domain>/<op>/op_kernel/`
- `<domain>/<op>/op_host/`
- `<domain>/<op>/op_host_aclnn*`
- `<domain>/<op>/op_graph/`
- `<domain>/<op>/tests/`
- `<domain>/<op>/examples/`

如果修改公共目录，例如 `attention/common/`、`gmm/common/`、`mc2/common/`、`mc2/3rd/`、`moe/3rd/`，提交说明必须列出影响范围和已验证算子。

## Boundary C: Evidence Artifacts

raw 证据不应直接进入 agent 知识库。

适合长期记录：

- 命令。
- summary。
- 关键数字。
- manifest。
- hash。
- 结论。

适合临时或 artifact 边界：

- `msprof` 原始目录。
- simulator dump。
- `.o`、`.bin`。
- 大 CSV。
- 长日志。

## Current Worktree

当前文档重构产生的是 agent context 变更。后续若开始真实性能优化，应先完成 baseline，再为单算子改动建立独立变更边界。
