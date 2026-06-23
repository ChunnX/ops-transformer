---
name: ascendc-performance-best-practices
description: Use when looking up or updating reusable ops-transformer Ascend C operator performance practices, operator-family optimization examples, migrated ops-math historical patterns, or failed optimization counterexamples.
---

# AscendC 性能优化最佳实践

本 skill 汇总可复用性能优化知识。默认用于 `ops-transformer`，历史 `ops-math` 案例只作为假设来源，不能直接当成本仓库优化结论。

## Repo Defaults

- SoC：`ascend910b`
- 架构目录：`arch22`
- custom vendor：`custom_transformer`
- 主线目录：`attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`

真实闭环结束后，如果形成可复用模式、失败反例或 gate 规则，更新本 skill 或其 reference。单算子局部日志优先写入 `agent_ops/<op>/` 账本或 `agent/lessons/`，不要把 raw dump 写进 skill。

## 查询方式

输入：

```text
operator:
operator family:
suspected bound:
optimization type:
evidence available:
```

查询流程：

1. 先判断算子族或 bound 类型。
2. 读取对应 reference。
3. 将 reference 中的经验转成候选假设。
4. 按 `agent/performance-optimization-method.md` 设计验证矩阵。

## Operator Families And References

| 类别 | 典型场景 | Reference |
| --- | --- | --- |
| MatMul / Cube | matmul、batch matmul、GMM、Moe matmul 变体 | `reference/matmul/guide.md` |
| RadixSort / TopK | sort、topk、arg 系列 | `reference/sort/radix_sort.md` |
| Scalar bound | scalar-heavy、常量折叠、循环控制占比高 | `reference/scalar/guide.md` |
| Reduction / Softmax | reduce、softmax、layernorm 类模式 | `reference/softmax/state_resident_design.md`, `reference/softmax/online_softmax_design.md` |
| Elementwise | Abs、Sin、Cos、Exp、简单逐元素或 broadcast | `reference/elementwise/double_buffer_design.md`, `reference/elementwise/vector_efficiency_design.md` |
| SIMT / Irregular | 条件分支、离散索引、不规则访问 | `reference/simt/optimization-guide.md` |
| Tiling / Core split | host `blockDim`、`vectorCoreNum`、有效任务数小于物理核数 | `reference/tiling/effective_core_count.md` |
| Common Data Movement | tail、DataCopy、UB resident、bank conflict | `reference/common/tail_block_design.md`, `reference/common/datacopy_optimization_design.md`, `reference/common/ub_resident_design.md` |

未收录的算子族返回“该算子族优化知识暂未收录”，然后按通用 tiling/DataCopy/memory/pipeline 方法设计实验。

## Migrated Historical Patterns

这些案例来自 `ops-math` 历史经验，使用时必须重新验证：

- 大 shape dtype/shape 分支可能有效，但需要证明小/边界/中等矩阵不回退。
- UB tile 放大、double buffer、TPipe/TQue 生命周期调整都可能因 scalar、queue 或同步开销导致矩阵回退。
- 减少 GM payload 不保证更快；大量小 DMA 描述和标量抽取可能让目标 case 变慢。
- 语义必需的输出初始化、清零、mask、state 更新不能删除。
- 同族迁移只能作为假设。tiling key、kernel suffix、host TilingData 和 API 支持范围必须逐算子核对。
- APT/opFile 路径切换先当 compile/package gate；失败后回退，不继续叠加优化。

## Keep Or Reject

可保留：

- 精度正常。
- 目标 case 提升。
- 代表性矩阵无稳定回退。
- package/load evidence 有效。
- case 映射可信。

必须拒绝或继续迭代：

- 只测有利 case。
- 单次最优样本代表结果。
- 收益低于约 1% 且无同窗口 A/B 证明。
- 目标大 shape 变快但通用矩阵回退。
- 改变语义、跳过必要写回或削减支持范围。

## Updating This Skill

更新时只写可复用规则：

```text
pattern:
applies when:
does not apply when:
required evidence:
known failure mode:
ops-transformer validation:
```

不要写长日志、完整 CSV、raw profiling dump 或单次不可复核结论。
