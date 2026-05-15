# Chunk GDR Tail Padding A131 案例

这是 `kernel-optimization-discipline` 的算子案例证据。不要把这里的阶段号、shape 集合或安装路径直接变成通用规则；遇到 tail padding、UB 对齐、build/install artifact 污染或 tail 性能验证时再读取。

## 背景

A131 将 `ApplyTailPaddingToAttn(validLen)` 从逐元素 `SetValue` tail loop 改为 `Duplicate` 补零。目标是删除按元素标量写造成的长周期冗余，同时保持 tail correctness。

## 可复用教训

- tail padding 不能用逐元素 `SetValue` 当正式优化方案。它会把每个 tail lane 变成标量循环，长 shape 下固定开销会累积。
- `Duplicate` 补零前先确认起始地址对齐。row tail 不能从未对齐地址直接 `Duplicate`，`S=5` 这类 shape 可能触发 AIV UB unaligned。
- 对 row tail 的安全写法是从 row start 做 masked `Duplicate`，或改成整块/整行 `Duplicate`，让起始地址保持 UB 对齐。
- tail padding 属于独立优化。不要同时改 main-block 路径、final RHS、workspace layout 或 profiling wrapper，否则失败时无法判断是 tail 写法还是其他数据流变化。

## 对齐检查

修改 tail padding 前先回答：

- `validLen` 后的 tail 是按 row 处理，还是跨 row 连续处理？
- `Duplicate` 的 dst 指针是否来自 row start、aligned block start 或其他已证明对齐的位置？
- mask 是否只覆盖本 row 需要清零的 tail lane？
- `S < blockSize`、`S == blockSize`、`S == blockSize + 1` 是否走同一套安全语义？

如果唯一的起点是 `rowStart + validLen`，且 `validLen` 不是硬件要求的对齐倍数，不要直接从该地址 `Duplicate`。

## Artifact 防污染

A131 曾遇到 build cache 被 `chunk_gated_delta_rule_C128` 污染。遇到“源码看起来对、运行结果不像当前 Neumann”的情况，先证明运行对象：

1. 查 `CMakeCache.txt` 或等价 build cache，确认 `chunk_gated_delta_rule_neumann` 目标和源码路径没有被旧 variant 污染。
2. 查构建包内 Neumann 的 object/json，确认包里包含当前 Neumann kernel artifact，而不是只看源码时间戳。
3. 查 top-level OPP 安装路径的 op_impl object/json hash，确认安装后的运行对象来自刚构建的包。
4. 若 OPP 下存在嵌套 `vendors/custom_transformer` 残留，不以内层残留为准；验证 top-level `op_impl` 路径和当前 `ASCEND_CUSTOM_OPP_PATH` 实际加载面。

这一步属于验证面清洁，不是性能优化本身。artifact 未证明前，tail smoke 或 profiling 结果都不能作为候选结论。

## Correctness Gate

本案例的 correctness gate：

- tail smoke：`S=5/16/31/32/33/34/48/49/63/65/129`
- aligned smoke：`S=64/256/1024`
- full compare

只有上述 gate 全部通过后，才进入性能 profiling。tail smoke 通过但 aligned smoke 或 full compare 失败时，不能把候选描述为 target path PASS。

## Profiling Gate

精度 gate 通过后再做 8-shape profiling。最终性能结论仍按 `performance-profiling-msprof` 的规则读取 measurement phase 的 `msprof --task-time=on` 结果。

记录 profiling 结论时至少说明：

- `SetValue` tail loop 是否被完全移除；
- `Duplicate` 是否从对齐 row start 或整块/整行起点发起；
- tail shapes 与 aligned shapes 是否都使用同一个已安装 artifact；
- 8-shape 中短 shape 收益是否伴随中长 shape 回退。
