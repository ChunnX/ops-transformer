# ops-transformer 与 ops-math 对照

本文件记录本轮参考 `/root/.config/superpowers/worktrees/ops-math/perf-collection-adapter` 后，对当前 `ops-transformer` 文档体系的重构依据。

## 总体结论

两个仓库整体框架相近，但不能视为完全同构。

相近之处：

- 都是 CANN 官方开源算子仓。
- 都以 `build.sh` 作为统一入口。
- 都使用 CMake、`opbuild`、AscendC binary compile 和 custom package。
- 都存在 `op_host`、`op_kernel`、`op_graph`、`examples`、`tests` 等常见目录模式。
- 都可以通过 `--pkg`、`--opkernel`、`-u`、`--run_example` 建立构建/验证/profiling 闭环。

关键差异：

- `ops-math` 主域是 `math/`、`conversion/`、`random/`；`ops-transformer` 主域是 `attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`。
- `ops-transformer` 有大量 attention、MoE、GMM、MC2 通信融合算子，MC2 引入 HCCL、多卡和拓扑变量。
- `ops-math` 当前工作树已有 `scripts/st_accuracy`、`scripts/agent_context`、`agent_ops`、`.agents/skills` 等长期优化基础设施；`ops-transformer` 当前已迁移裁剪 `scripts/st_accuracy`，已有 `.agents/skills`，并为两个已闭环 attention 算子新增轻量 `agent_ops` 账本，但仍没有 `scripts/agent_context`。
- `ops-math` 的当前环境记录是 CANN 9.0.0、910B3、`arch35` 默认；`ops-transformer` 本轮用户明确给出的目标是 910B B3、aarch64，仓库静态映射是 `ascend910b -> arch22`。
- `ops-transformer` 的 `scripts/st_accuracy` 当前只覆盖 `lightning_indexer` 与 `recurrent_gated_delta_rule`，但这两个算子已接入 native ACLNN 调用和 host golden compare；不能继承 `ops-math` 的全量 coverage 结论。
- `ops-math` 的 `build.sh` 支持 `--dump_cce`、`--mssanitizer` 等调试参数；`ops-transformer` 静态可见主要是 `--oom`、`--kernel_template_input`、`--bisheng_flags`、`--op_debug_config`、`--ops-compile-options`。

## 文档结构对照

| 维度 | ops-math | ops-transformer 重构结果 |
| --- | --- | --- |
| 入口 | `AGENTS.md` 只放硬规则和路标 | 采用同样原则，压缩根入口 |
| 总索引 | `agent/INDEX.md` | 新增 `agent/INDEX.md` |
| 当前状态 | `agent/current_state.md` | 新增 `agent/current-state.md`，不伪造 coverage |
| 硬性策略 | `agent/policies/*` | 新增 accuracy/performance/evidence policies |
| runbook | `agent/runbooks/*` | 新增 build/accuracy/performance/package-deploy |
| decisions | `agent/decisions/*` | 新增 910B/静态优先/opkernel-vs-pkg 决策 |
| lessons | `agent/lessons/*` | 新增 pitfalls/performance-optimization |
| 单算子账本 | `agent_ops/<op>/` 已存在 | 已为两个完成闭环的 attention 算子引入轻量账本，后续逐算子扩展 |
| 校验脚本 | `scripts/agent_context/validate_agent_context.py` | 不存在，使用静态 `find/rg` 校验 |

## 可迁移内容

已经迁移为 transformer 版本：

- `AGENTS.md` 只放硬约束、默认范围、关键路径、常用命令、部署边界和索引。
- 精度诚信边界：不能改 golden、case、阈值或 runner 逻辑来掩盖失败。
- 性能诚信边界：不能删计算、改语义、挑有利 case 或混用旧产物来制造提升。
- `--opkernel` 与 `--pkg` 部署边界：kernel-only 不等于运行时加载新实现。
- 证据保留原则：长期保留 summary/manifest/命令/结论，不把 raw profiling dump 堆进 `agent/`。
- 工作台层与 reference 层分离：入口轻，细节按需读。

## 不可直接迁移内容

以下 `ops-math` 内容不能直接套用到 `ops-transformer`：

- `ops-math` 的全量 `scripts/st_accuracy/run_st_accuracy.py` coverage 结论；transformer 当前只迁移两个 attention 算子的局部 runner，不继承 math 仓库的历史结论。
- `scripts/agent_context/validate_agent_context.py` 校验命令。
- `ops-math` 的 `agent_ops/<op>/` 中的单算子状态、性能结论和 manifest。
- `.agents/skills/` 中的 ops-math 单算子结论不能直接迁移；当前 transformer 仓库已有对应目录，但 skill 内容必须按 `arch22`、`custom_transformer` 和本仓库验证入口裁剪。
- CANN 9.0.0、`arch35`、ACL logical device `0`、physical NPU id `5` 等当前机器事实。
- `custom_math` vendor 路径，应改为 transformer 的 custom package/vendor 语义。
- `--dump_cce`、`--mssanitizer` 等 `ops-transformer` 当前 `build.sh` 未暴露的参数。

## 对后续优化的影响

`ops-transformer` 后续性能优化要比 `ops-math` 多关注：

- attention 类算子的 seq/head/cache/softmax/quant 分支。
- GMM/MoE 的 expert 分布、routing、token layout 和 small batch。
- MC2 的 HCCL、多卡通信、rank、all-reduce/all-gather/reduce-scatter/all-to-all 与 matmul overlap。
- `arch22` 分支和 910B CI list，而不是 `ops-math` 当前文档中的 `arch35` 默认。

因此，本仓库文档采用 `ops-math` 的治理形态，但保留 transformer 自己的算子域、命令、缺口和硬件默认。
