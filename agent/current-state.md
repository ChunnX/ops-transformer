# Current Agent State

本文是当前动态事实源。`AGENTS.md` 只引用本文件，不复制长段算子表、覆盖率或候选队列。

## Environment

- 用户声明硬件：Ascend 910B 系列 B3。
- 用户声明系统架构：`aarch64`。
- 仓库默认编译 SoC：`ascend910b`。
- CMake 架构映射：`ascend910b -> arch22`。
- 仓库版本静态声明：`ops-transformer` `9.1.0`。
- CANN build/run 依赖静态声明：`>=8.5`。
- 当前迁移开发分支：`br_tang_dev`。后续性能开发默认在该分支进行，完成单算子闭环后再将核心代码合入 `master`。

已完成两个重点 attention 算子的设备侧精度和性能采集闭环。后续扩展到新算子时仍需记录 `npu-smi info`、driver/firmware、CANN toolkit/opp、`bisheng`、`msprof`、Python、gcc、cmake 和关键环境变量。

## Scope

当前静态确认的主线 AscendC AI Core 算子：117 个。

统计口径：

- 目录位于 `attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`。
- 存在 `op_kernel/`。
- `op_kernel/` 中有 C/C++ 源文件。
- 源码命中 AscendC 特征，例如 `__aicore__`、`AscendC::`、`using namespace AscendC`、`extern "C" __global__` 或 `KERNEL_TASK_TYPE_AICORE`。
- 排除 `tests/`、`examples/`、`common/`、`3rd/`、`experimental/`、模板目录和纯 AICPU 调度目录。

分类数量：

| 域 | 主线 AscendC 算子数 | 910B CI 覆盖 |
| --- | ---: | ---: |
| `attention` | 40 | 38 |
| `ffn` | 5 | 4 |
| `gmm` | 7 | 5 |
| `mc2` | 27 | 14 |
| `moe` | 27 | 27 |
| `posembedding` | 11 | 11 |
| 合计 | 117 | 99 |

详细列表见 [ascendc-operators.md](ascendc-operators.md)。

## Current Documentation State

已经建立两层文档：

- 工作台层：`AGENTS.md`、`agent/INDEX.md`、`agent/current-state.md`、`agent/policies/`、`agent/runbooks/`、`agent/decisions/`、`agent/lessons/`。
- Skill 层：`.agents/skills/`，当前包含 AscendC 性能优化、profiling、策略和最佳实践相关 skill。
- 静态 reference 层：`agent/ascendc-operators.md`、`agent/build-flow.md`、`agent/accuracy-test-flow.md`、`agent/performance-test-flow.md`、`agent/environment.md`、`agent/repo-structure.md`、`agent/operator-development-pattern.md`、`agent/optimization-entrypoints.md`、`agent/known-risks-and-gaps.md`。

对 `ops-math` 的可迁移结构已经吸收进工作台层；当前已迁移裁剪 `scripts/st_accuracy` 作为两个 attention 算子的局部 ST 精度入口，并补充这两个算子的 host golden compare。当前已为 `lightning_indexer` 和 `recurrent_gated_delta_rule` 创建 `agent_ops/` 单算子账本，因为这两个算子已有 package、精度、profiling 和矩阵判定证据。

当前 repo-local performance skills：

| skill | 用途 |
| --- | --- |
| `ascendc-operator-performance-optim` | 端到端 AscendC 算子性能优化闭环，包含排查、修改、精度和性能判定。 |
| `ascendc-perf-optimize` | 根据 tiling、pipeline、profiling 和 bound 诊断制定优化策略。 |
| `ascendc-performance-best-practices` | 查询或更新可复用性能实践、成功模板和失败反例。 |
| `ops-profiling` | 使用 `msprof op` / `msprof` 采集、归档和分析性能数据。 |

这些 skills 的默认事实必须以本仓库为准：主线目录是 `attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`；910B/B3 架构目录是 `arch22`；custom package vendor 是 `custom_transformer`。

## Build State

静态入口：

- 主入口：`build.sh`
- CMake 入口：`CMakeLists.txt`
- 自定义 package：`bash build.sh --pkg --soc=ascend910b --ops=<op> --vendor_name=custom`
- kernel-only：`bash build.sh --opkernel --soc=ascend910b --ops=<op>`
- UT：`bash build.sh -u --ophost_test|--opapi_test|--opgraph_test|--opkernel_test --ops=<op> --soc=ascend910b`
- example：`bash build.sh --run_example <op> eager|graph cust --vendor_name=custom --soc=ascend910b`

当前已执行两个重点算子的 custom package build：

```bash
bash build.sh --pkg --soc=ascend910b --ops=lightning_indexer,recurrent_gated_delta_rule --vendor_name=custom -j16 -O3
```

生成并安装了 `build_out/cann-ops-transformer-custom_linux-aarch64.run`，安装目录为 `/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer`。

本次迁移到 `br_tang_dev` 后已在目标仓重新执行 package build/install：

- package sha256: `4d8f2567a0f2e9b8422e1d098db12f8b8dc9b0c50bba7a2a79d97af5742eca0c`
- installed op_api sha256: `7da7964e25974ba1af76afdd0cbdcd4836e34c157c3a16d29878f0266b2b0f30`

## Accuracy State

静态可见验证入口：

- C++ UT：`build.sh -u ...`
- example：`build.sh --run_example ...`
- pytest：集中在部分 attention 算子。
- ST 资产：部分 attention、mc2、posembedding 算子有 `tests/st/`；当前新增两个重点 attention 算子的 ST JSON。
- 本地 ST runner：`scripts/st_accuracy/run_st_accuracy.py`，当前优先支持 `lightning_indexer` 与 `recurrent_gated_delta_rule`。
- CI UT 批量脚本：`scripts/ci/ut_make.sh`。

当前状态：

- `scripts/st_accuracy` 已迁移为两个重点 attention 算子的局部 runner，不代表全仓库 ST coverage 已建立。
- 当前 native runner 对 `LightningIndexer` 比较 `sparseIndicesOut` 和可选 `sparseValuesOut`；对 `RecurrentGatedDeltaRule` 比较 `out` 和原地更新后的 `stateRef`。
- `coverage --ops lightning_indexer,recurrent_gated_delta_rule`：18 executable、0 unsupported。
- 已安装 `custom_transformer` 后，本次迁移验证 `run --ops lightning_indexer,recurrent_gated_delta_rule --out build/st_accuracy/migration_device`：18 passed、0 failed、0 error、0 unsupported。
- 根因记录：最初设备执行失败是因为当前 CANN OPP 只存在 `custom_math`，没有 `/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so`；构建并安装本仓库 custom package 后解决。
- `RecurrentGatedDeltaRule` 默认 ST 矩阵使用唯一 `ssmStateIndices`；重复 state index 会引入 `stateRef` 原地更新的覆盖/顺序依赖，不纳入默认批量通过矩阵。
- example 是否做 golden 对比取决于单个算子源码。

## Performance State

静态可见性能入口：

- 先构建/安装 custom package。
- 通过 `build.sh --run_example` 生成可执行二进制。
- 使用 `msprof op` 或 `msprof --application` 采集。
- simulator 和 `cannsim` 只在特定场景使用。

当前已完成的真实性能闭环：

- `lightning_indexer`：host `blockDim` 按 shape-implied base-block 上界收窄。ST 矩阵 22.72% 到 51.59% 提升；README PA large 为 +0.39%，按噪声区间视作持平；无稳定回退。账本见 [../agent_ops/lightning_indexer/README.md](../agent_ops/lightning_indexer/README.md)。
- `recurrent_gated_delta_rule`：host `vectorCoreNum` 按 `b * nv` 收窄。ST 矩阵 14.33% 到 34.82% 提升；无稳定回退。账本见 [../agent_ops/recurrent_gated_delta_rule/README.md](../agent_ops/recurrent_gated_delta_rule/README.md)。
- 共同证据：`artifacts/perf_candidate/li_rgdr_blockdim_20260617_0442/SUMMARY.md`，package sha256 `9cadc00a497ed7acc41cbb7112eb3f529fbcf139d6813041879f4384c36e0f07`，installed op_api sha256 `96bf2f2af0840292e7ceb3ecd149af57021ac0f44af12bd0bbe3820e8b1cd3ab`。

当前缺口：

- 没有统一 `build.sh --perf`。
- 没有统一 benchmark schema。
- `agent_ops/<op>/` 账本刚建立，只覆盖上述两个 attention 算子。
- MC2 性能需要 HCCL、多卡和拓扑信息，不能只看单卡 smoke。

## Next Useful Work

扩展到下一个算子前，建议先选一个 910B CI 覆盖、存在 example 和 UT 的算子，建立最小闭环：

1. `--pkg` 构建并安装 custom package。
2. `--run_example` 生成 eager 二进制并确认加载 custom package。
3. 跑对应 UT 或 pytest/ST 资产。
4. `msprof op` 采集 baseline。
5. 如开始反复实验，再创建或更新 `agent_ops/<op>/` 账本记录 summary、manifest 和证据边界。
