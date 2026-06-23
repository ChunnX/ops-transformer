# ops-transformer Agent Entry

本文件是 Codex 进入 `ops-transformer` 仓库后的启动入口，只保留必须总是加载的规则和路标。详细静态分析、runbook、决策和经验按需读取 [agent/INDEX.md](agent/INDEX.md)。

注意：当前仓库的通用分析目录名是 `agent/`，不是 `agents/`。`.agents/skills/` 是本仓库的 Codex skill 入口，当前包含 AscendC 性能优化、profiling 和最佳实践相关 skills。文档主体来自静态分析和对 `ops-math` 文档体系的对照重构；当前额外建立了两个重点 attention 算子的局部 ST 精度 runner。当前环境已用 `--pkg --ops=lightning_indexer,recurrent_gated_delta_rule --vendor_name=custom --soc=ascend910b` 构建并安装 `custom_transformer`，两个算子的 18 个 ST case 已完成设备侧 golden compare，并完成首轮有效核数收窄性能优化闭环。单算子摘要账本见 `agent_ops/lightning_indexer/` 和 `agent_ops/recurrent_gated_delta_rule/`。

## 硬性约束

- 后续默认只优化最终运行在 AI Core 上的 AscendC 算子；不把 `op_kernel_aicpu/`、AICPU-only 调度、样例模板、测试代码或 third-party 目录计入主线优化对象，除非用户明确改变范围。
- 不得为了让精度通过而无依据修改 golden、参考结果、case JSON、比较阈值或 runner 逻辑。可以修复真实 runner bug、补充合法 dtype/range/shape/package gate，或把超出官方文档/example/UT 可确定验证域的本地 ST case 调整回可验证输入域；必须记录原因和验证命令。
- 不得为了制造性能提升而删减必要计算、改变算子语义、只测有利 case、遗漏回退 case、混用旧安装产物、用单次最优样本代表结果或跨环境直接比较。
- 性能优化只有同时满足精度验证正常、目标 case 提升、代表性矩阵无稳定回退、profiling 确认加载新产物，才可写成有效优化。
- `build.sh --opkernel` 只代表编译 kernel binary，不代表运行时 vendor 目录、op_api、op_host tiling、op info 和 kernel config 已更新。改动涉及 tiling/opdef/op_api/metadata 时默认重新 `--pkg` 并安装。
- 当前仓库已有从 `ops-math` 迁移并裁剪的 `scripts/st_accuracy`，当前只优先覆盖 `attention/lightning_indexer` 与 `attention/recurrent_gated_delta_rule`。native runner 对这两个算子包含 host golden compare；真实设备侧 `run` 必须强制重新 `--pkg` 构建并安装 selected ops 的 `custom_transformer` package，`--no-execute`、`coverage` 和 `taxonomy` 不能写成设备侧精度通过。当前已有 `.agents/skills` 性能优化 skills；仍没有 `scripts/agent_context` 基建。当前已为两个优化算子建立轻量 `agent_ops/` 账本，账本只记录摘要、命令、hash 和结论，不长期保存 raw profiling dump。
- 后续算子性能开发、调试和文档账本更新默认基于 `br_tang_dev`；某个算子完成闭环后，只把核心算子代码、必要测试和最小可复核说明合入 `master`，再从 `master` 提交到主仓库。

## 性能优化默认闭环

真实算子性能优化默认按以下顺序推进，不能跳步写结论：

1. 候选 gate：确认目标位于主线 AscendC AI Core 范围，优先选择 910B CI 覆盖、存在 example/UT/ST 或真实业务输入的算子。
2. package/load gate：确认 `--pkg --soc=ascend910b --vendor_name=custom` 生成并安装 `custom_transformer`，profiling 加载当前 op_api、op_impl、kernel binary 和 config。
3. 精度 gate：先运行可用 UT、pytest、example golden 或 `scripts/st_accuracy`，且 selected case 不能有 `failed/error/unsupported`。
4. baseline：固定代表性矩阵，采集同一环境下的 baseline；矩阵至少覆盖目标 shape、小 shape、边界 shape、常见中等 shape 和相关 dtype/attrs。
5. candidate：每次改动后重新构建/安装，复跑精度，再用同一矩阵 profiling；每个 case 至少 3 次采样，默认用 median。
6. 判定：只有目标 case 提升、代表性矩阵无稳定回退、profiling 映射正确且加载新产物可证明，才可保留优化。
7. 归档：按 [agent/policies/evidence-retention.md](agent/policies/evidence-retention.md) 记录 summary、命令、关键指标和 hash；raw dump 不长期堆入 `agent/`。
8. 反哺：形成可复用成功模式、失败反例或 gate 规则时，更新 [agent/lessons/performance-optimization.md](agent/lessons/performance-optimization.md) 或 `.agents/skills/`。

## 默认范围

- 仓库类别：`attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`。
- 当前目标硬件：Ascend 910B 系列 B3。
- 当前系统架构：`aarch64`。
- 默认编译参数：`--soc=ascend910b`。
- CMake 架构目录：`ascend910b -> arch22`。
- 当前已静态确认主线 AscendC 算子：117 个，见 [agent/ascendc-operators.md](agent/ascendc-operators.md)。
- 当前 910B CI 列表覆盖主线 AscendC 算子：99 个，优先作为 B3 性能优化候选。

## 关键路径

优化 AscendC AI Core 算子时重点看：

- `<domain>/<op>/op_kernel/`：设备侧 AscendC kernel 实现。
- `<domain>/<op>/op_kernel/arch22/`：当前 910B/B3 优先关注的架构特化路径。
- `<domain>/<op>/op_host/*tiling*`：tiling key、block dim、workspace、TilingData、shape/dtype/format 分支。
- `<domain>/<op>/op_host/config/`：SoC、binary、runtime metadata 线索。
- `<domain>/<op>/op_host_aclnn*`：eager/aclnn API 定义和封装。
- `<domain>/<op>/op_graph/`：graph 模式注册和接入。
- `<domain>/<op>/examples/`：可生成 smoke/profiling 二进制的样例。
- `<domain>/<op>/tests/`：UT、pytest、ST 资产。
- `common/`、`attention/common/`、`gmm/common/`、`mc2/common/`、`mc2/3rd/`、`moe/3rd/`：公共依赖，修改会扩大影响面。

## 常用命令

以下命令来自静态脚本和文档梳理。当前已执行两个重点 attention 算子的 package build/install、`scripts/st_accuracy` 设备侧精度和默认性能矩阵 profiling；尚未执行全仓库 ST。

编译并安装单算子 custom 包：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

只编译 kernel binary：

```bash
bash build.sh --opkernel --soc=ascend910b --ops=<op_name> -j16 -O3
```

运行 host / op_api / op_kernel UT：

```bash
bash build.sh -u --ophost_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh -u --opapi_test --soc=ascend910b --ops=<op_name> -j16
bash build.sh -u --opkernel_test --soc=ascend910b --ops=<op_name> -j16
```

运行 eager / graph 样例：

```bash
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b
bash build.sh --run_example <op_name> graph cust --vendor_name=custom --soc=ascend910b
```

采集性能：

```bash
cd build
msprof op --application=./test_aclnn_<op_name> --output=../artifacts/<op_name>/msprof/<tag>
```

## 部署边界

- 首次建立 baseline、修改 `op_host/*tiling*`、`op_host/*def*`、`op_host_aclnn*`、op info、dtype/format、workspace 或 tiling key 语义时，默认用 `--pkg` 生成并安装 custom package。
- 仅改 `op_kernel/` 或 binary config 时，`--opkernel` 可以作为快速编译路径；但必须显式同步新 binary/config 到运行时 vendor 目录，并在记录中写清同步路径。
- custom package 默认 vendor 示例是 `custom_transformer`。确认 profiling 加载的是当前包，而不是 CANN 原始实现或旧 custom 产物。
- MC2 算子常涉及 HCCL、多卡、rank 和通信拓扑。单卡 example 或 simulator 不能代表真实性能。

## 文档入口

- [agent/INDEX.md](agent/INDEX.md)：知识库总索引。
- [agent/current-state.md](agent/current-state.md)：当前静态状态、环境、范围和已知缺口。
- [agent/comparison-ops-math.md](agent/comparison-ops-math.md)：与 `ops-math` 的结构和文档体系对照。
- [agent/change-boundary.md](agent/change-boundary.md)：agent 文档、未来算子源码改动和 raw 证据产物的推荐边界。
- [agent/performance-optimization-method.md](agent/performance-optimization-method.md)：本仓库 AscendC 性能优化通用方法和 gate。
- [agent/policies/accuracy-integrity.md](agent/policies/accuracy-integrity.md)：精度验证诚信边界。
- [agent/policies/performance-integrity.md](agent/policies/performance-integrity.md)：性能优化诚信边界。
- [agent/policies/evidence-retention.md](agent/policies/evidence-retention.md)：证据保留和归档规则。
- [agent/runbooks/build.md](agent/runbooks/build.md)：构建、UT 和 package runbook。
- [agent/runbooks/accuracy.md](agent/runbooks/accuracy.md)：example、UT、pytest、ST 资产 runbook。
- [agent/runbooks/performance.md](agent/runbooks/performance.md)：`msprof op` 和性能矩阵 runbook。
- [agent/runbooks/package-deploy.md](agent/runbooks/package-deploy.md)：`--opkernel`、`--pkg`、vendor 和运行时加载边界。
- [agent/decisions/](agent/decisions/)：长期决策记录。
- [agent/lessons/](agent/lessons/)：可复用踩坑和性能优化经验。
- [agent_ops/](agent_ops/)：已进入真实闭环的单算子轻量账本，只放摘要、命令、关键指标和 hash。

## 经验沉淀

- `AGENTS.md` 只维护入口规则和索引，不记录长日志、全量算子表或历史流水。
- 当前动态事实写入 [agent/current-state.md](agent/current-state.md)。
- 详细静态分析保留在 `agent/*-flow.md`、`agent/*-structure.md`、`agent/ascendc-operators.md` 等 reference 文档。
- 通用规则写入 `agent/policies/`、`agent/runbooks/`、`agent/decisions/` 或 `agent/lessons/`。
- 可复用性能优化流程写入 `.agents/skills/`；从 `ops-math` 迁移来的案例只能作为假设来源，必须在 `ops-transformer` 的 package、精度和 profiling 闭环中重新验证。
- 已进入真实性能闭环的单算子记录写入 `agent_ops/<op>/`；只保留可复核摘要和 manifest，不长期堆 raw profiling dump。

## 完整性检查

当前仓库没有 `ops-math` 的 `scripts/agent_context/validate_agent_context.py`。文档维护使用静态检查：

```bash
find AGENTS.md agent agent_ops .agents/skills -maxdepth 4 -type f | sort
rg -n 'TO[D]O|TB[D]|FIX[M]E|待[补]|待[确]认' AGENTS.md agent agent_ops .agents/skills
rg -c '^\| `[^`]+` \| `[^`]+/[^`]+` \|' agent/ascendc-operators.md
```
