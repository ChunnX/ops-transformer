# Agent Knowledge Index

`AGENTS.md` 是启动入口，只放必须总是加载的硬规则和路标。需要细节时按本索引读取对应文件，避免一次性加载全部上下文。

## 当前状态

- [current-state.md](current-state.md)：当前环境、静态范围、已确认 AscendC 算子数量、910B 候选和已知缺口。
- [comparison-ops-math.md](comparison-ops-math.md)：`ops-transformer` 与 `ops-math` 的仓库结构、文档结构和能力差异。
- [change-boundary.md](change-boundary.md)：agent 文档、未来算子源码改动和 raw 证据产物的推荐边界。
- [known-risks-and-gaps.md](known-risks-and-gaps.md)：首轮静态分析不能证明的事项和容易误判的位置。
- [performance-optimization-method.md](performance-optimization-method.md)：本仓库 AscendC AI Core 性能优化 gate、闭环方法和记录格式。

## 硬性策略

- [policies/accuracy-integrity.md](policies/accuracy-integrity.md)：精度验证诚信边界。
- [policies/performance-integrity.md](policies/performance-integrity.md)：性能优化诚信边界。
- [policies/evidence-retention.md](policies/evidence-retention.md)：性能证据保留、归档和删除规则。

## Runbooks

- [runbooks/build.md](runbooks/build.md)：`build.sh`、CMake、UT 和 package 的常用路径。
- [runbooks/accuracy.md](runbooks/accuracy.md)：example、C++ UT、pytest、ST 资产和 CI UT 脚本。
- [runbooks/performance.md](runbooks/performance.md)：eager 样例、`msprof op`、性能矩阵和采样口径。
- [runbooks/package-deploy.md](runbooks/package-deploy.md)：`--opkernel`、`--pkg`、vendor 安装和运行目录同步边界。

## Decisions

- [decisions/0001-910b-b3-arch22.md](decisions/0001-910b-b3-arch22.md)：默认优化目标是 Ascend 910B B3、`--soc=ascend910b`、`arch22`。
- [decisions/0002-static-first-no-fabricated-runners.md](decisions/0002-static-first-no-fabricated-runners.md)：本仓库当前只记录静态可见入口，不借用 `ops-math` 的 ST/agent_context 基建。
- [decisions/0003-opkernel-vs-pkg.md](decisions/0003-opkernel-vs-pkg.md)：何时可以用 `--opkernel` 快速路径，何时必须完整 `--pkg`。

## Lessons

- [lessons/pitfalls.md](lessons/pitfalls.md)：环境、构建、安装、example、MC2 和 profiling 常见坑。
- [lessons/performance-optimization.md](lessons/performance-optimization.md)：面向 transformer 类算子的性能优化经验模板和反例提醒。

## Repo-Local Skills

`.agents/skills/` 是当前仓库的 Codex skill 入口。启动 Codex 时若技能发现机制扫描仓库 skill，应能看到以下性能相关 skill：

- `.agents/skills/ascendc-operator-performance-optim/`：端到端 AscendC 算子性能优化闭环。
- `.agents/skills/ascendc-perf-optimize/`：从 tiling、pipeline、profiling 和 bound 诊断制定优化策略。
- `.agents/skills/ascendc-performance-best-practices/`：可复用 AscendC 性能优化实践、成功模板和失败反例。
- `.agents/skills/ops-profiling/`：`msprof op` / `msprof` 采集、归档、矩阵和回归分析。

这些 skills 已从 `ops-math` 经验中裁剪，默认必须按 `ops-transformer` 的 `ascend910b -> arch22`、`custom_transformer`、主线目录和本仓库精度入口重新验证。

## Reference

- [ascendc-operators.md](ascendc-operators.md)：117 个确认存在 AscendC `op_kernel` 源码的主线算子列表。
- [build-flow.md](build-flow.md)：端到端编译流程和模块输入输出。
- [accuracy-test-flow.md](accuracy-test-flow.md)：精度验证、UT、pytest、ST 和 example 执行链路。
- [performance-test-flow.md](performance-test-flow.md)：性能测试入口、profiling 流程、输出产物和缺口。
- [environment.md](environment.md)：支持的 CANN、SoC、硬件、系统依赖和环境变量。
- [repo-structure.md](repo-structure.md)：仓库目录结构和主要职责边界。
- [operator-development-pattern.md](operator-development-pattern.md)：单个算子的源码组织、host/kernel/API/graph/test 约定。
- [optimization-entrypoints.md](optimization-entrypoints.md)：后续优化时优先关注的源码、参数、编译选项和测量闭环。

## Future Ledger

当前仓库已开始建立 `agent_ops/<op>/` 单算子轻量账本。已存在：

- [../agent_ops/lightning_indexer/README.md](../agent_ops/lightning_indexer/README.md)：LightningIndexer 首轮有效核数收窄优化。
- [../agent_ops/recurrent_gated_delta_rule/README.md](../agent_ops/recurrent_gated_delta_rule/README.md)：RecurrentGatedDeltaRule 首轮 `b * nv` 有效核数收窄优化。

账本只放摘要、命令、关键指标、manifest 和 hash；raw profiling dump 不长期放入 `agent/`。
