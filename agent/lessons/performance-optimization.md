# Performance Optimization Lessons

本文记录 `ops-transformer` 中已验证或待验证的性能优化经验。`ops-math` 迁移经验只能作为假设来源；已验证结论必须来自本仓库 package、精度和 profiling 闭环。

## Candidate Gate

优先选择：

- 位于 [../ascendc-operators.md](../ascendc-operators.md) 主列表。
- 在 910B CI list 中覆盖。
- 有 `examples/` 可生成 eager 二进制。
- 有 `tests/ut`、pytest 或 ST 资产。
- 业务提供真实 shape/dtype/format。

不优先选择：

- 仅存在于 `experimental/` 的目录，除非用户明确要求。
- 只存在样例或模板源码的目录。
- AICPU-only 或 scheduler-only 目录。
- 缺少 public 调用路径或 custom package 加载路径的目录。

候选不等于可 profiling。进入性能采集前还需要 package/load gate：`custom_transformer` op_api、op_impl、kernel binary 和 config 必须能证明来自当前 build/install。

## Keep Or Reject Rule

可保留优化需要同时满足：

- 精度验证通过。
- 目标 case 有稳定提升。
- 通用矩阵无稳定回退。
- baseline/candidate 同环境同输入同 package 口径。
- profiling 证明加载新产物。

应拒绝或继续优化：

- 只有单个 shape 提升，但小/中/边界 shape 稳定回退。
- 收益低于约 1% 且没有重复 A/B 证明。
- 只减少理论指令或 GM 字节数，但实测回退。
- 通过改变语义、跳过写回或减少必要计算得到提升。
- `unsupported` 或 runner 跳过被误读成精度通过。
- `msprof op --launch-count` 不足导致只采集 batch 前缀。
- 混合 kernel suffix、tiling key 或 ACLNN 多 task 链时 case 映射不可信。

## Migrated Historical Patterns

以下来自 `ops-math` 的经验只能作为假设来源，不能直接作为 `ops-transformer` 的性能结论：

- 目标大 shape 分支优化可以保留，但必须保留小 shape 旧路径或证明小/边界/中等矩阵无回退。
- 减少 GM payload 不必然更快；大量小 DMA 描述、标量抽取或同步开销可能让 `Task Duration(us)` 回退。
- 初始化或清零如果是语义必需，不能为了性能跳过；只能优化实现方式并用精度和矩阵证明。
- 同族算子经验不能直接套用。tiling key、host TilingData、kernel suffix 和 API 支持范围必须逐算子核对。
- APT/opFile 路径切换先作为 compile/package gate 验证；编译失败或 API 不兼容时立即回退，不要继续叠加性能改动。
- TPipe/TQue 生命周期、double buffer、UB tile 放大等 API cleanup 需要性能矩阵判定；aggregate 变快但代表 case 回退时应拒绝。

## Transformer-Specific Focus

attention：

- softmax pipeline、mask、KV cache、seq/head 分支。
- quant/dequant、scale、layout、cache 写回。

GMM/MoE：

- expert/group 分布、token layout、routing、small batch。
- swiglu/activation/quant 融合。

MC2：

- matmul 与 all-reduce/all-gather/reduce-scatter/all-to-all 的 overlap。
- HCCL、rank、topology 和通信 task。

posembedding：

- rope/rotary 向量化、sin/cos cache、kvcache 连续写入。

## Verified Pattern: Effective Core Count

首轮已在两个 attention 算子中验证一个通用 tiling 模式：当 host 默认按物理核数或 TSch 混合核数发射，而 kernel 的独立任务数明显更少时，先收窄 host 发射核数，避免大量空核调度开销。

适用条件：

- 能从 shape 或 attrs 推导不会低估的独立任务数上界。
- kernel 已按实际任务数分发工作，额外发射的核只会空转或只做调度开销。
- 收窄不改变 tiling key、dtype/format 支持范围或语义。
- 大 shape 下有效任务数大于等于平台核数时保持原平台核数。

已验证案例：

- `lightning_indexer`：用 `B * N2 * ceil(S1 / s1Base) * ceil(S2 / s2Base)` 的 shape 上界收窄 host `blockDim`。ST 矩阵提升 22.72% 到 51.59%；README PA large 为 +0.39%，视作持平。账本见 `agent_ops/lightning_indexer/`。
- `recurrent_gated_delta_rule`：用 `b * nv` 收窄 `vectorCoreNum`。ST 矩阵提升 14.33% 到 34.82%。账本见 `agent_ops/recurrent_gated_delta_rule/`。

使用边界：

- 不得用 runtime tensor 的实际值做 host 静态低估；host 只能用 shape 上界，runtime exact split 仍应留给 kernel。
- `sparse_count`、mask、actual seq length、varlen 等会改变实际 work count 的路径必须纳入矩阵。
- 接近 1% 的收益或回退按噪声处理，需要补采或同窗口 A/B 才能写成结论。
- 该模式主要消除 launch/scheduling overhead；对大 shape 或 kernel 内部算力/访存瓶颈未必有效。

## Experiment Record Template

```text
operator:
api:
hypothesis:
changed files:
shape/dtype/format:
attrs:
soc/arch:
build/install:
correctness:
profiling:
baseline:
candidate:
matrix summary:
decision:
follow-up:
```

## Skill Feedback

真实闭环结束后：

- 成功保留的模式写入 `.agents/skills/ascendc-performance-best-practices/` 或对应 reference。
- profiling 采集、case 映射、launch-count、vendor 污染等流程经验写入 `.agents/skills/ops-profiling/`。
- 单算子局部事实优先写入 `agent_ops/<op>/` 账本，不要塞进 `AGENTS.md`。
