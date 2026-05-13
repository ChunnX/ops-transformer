# 2026-05-06 Fast Inverse 参考分析

## 参考对象

- 文档：<https://github.com/huawei-csl/gdn-tri-inverse/blob/0.1.0/markdown/fast_inverse_blog/fast_inverse_zh.md#deep-dive-on-ascend-910b-implementations>
- 性能实现：<https://github.com/huawei-csl/pto-kernels/blob/v0.1.2/csrc/kernel/kernel_tri_inv_rec_unroll.cpp>

## 核心结论

当前 `fulllocal` 可以参考 fast inverse 的优化方向，但不能直接视为已经实现了同类优化。

fast inverse 真正带来收益的关键不是单纯把 Neumann/tri inverse 改成 Cube 乘法，而是：

1. 输入矩阵进入 L1 后，求逆中间态尽量在 L1/L0 内循环消费。
2. 只在入口 `TLOAD` 和出口 `TSTORE` 访问 GM。
3. 用结构化 tile/fractal 搬运把对角块、奇偶块快速送到 L0A/L0B。
4. 用双缓冲让 L1/L0 搬运和 `TMATMUL` 重叠。

当前 `fulllocal` 只覆盖了其中第一点的一小段：16x16 Neumann block 内部的 `power` 与 `INV += INV@P` 在 half L1/TSCM + L0A/L0B/L0C 中完成。其边界之外仍大量依赖 GM，因此性能未形成净收益。

## fast inverse 可借鉴点

### 1. 入口/出口 GM，中间 L1/L0 resident

fast inverse 在博客中明确强调中间结果留在 L1，避免写回 global memory；性能代码中 `InvertSingleTile()` 的主体流程也围绕 L1 tile、L0 tile 和 `TMATMUL/TMATMUL_ACC` 展开。

当前 `fulllocal` 的差距：

- AIV 仍先把 `L_powWsGm_ = L` 和 `INVWsGm_ = I-L` 写到 GM。
- AIC 再通过 raw MMAD 把 GM FP32 矩阵搬到 half L1/TSCM。
- 16x16 inverse 完成后又写回 `INVWsGm_`。
- final RHS `INV @ k_cumdecay` / `INV @ v_beta` 仍从 GM 读 `INV`。

因此现在不是“少一次 GM 往返”，而是“GM -> half L1 -> GM”的额外转换链。

### 2. 结构化 diagonal / odd-even block 搬运

fast inverse 对 16x16 fractal 设计了 `CopyDiagonalFractalsL1ToL0()` 和 `CopyOddOrEvenBlocksL1ToL0()`。这点对应我们当前 C64 compose 的痛点：32/64 compose lower-left block 仍通过 GM raw MMAD 和 scalar full-L1 backfill 维持正确性。

可借鉴方向：

- 为 `C64_3232` 的 `Compose32From16()` / `Compose64From32()` 设计 half L1/TSCM resident 的结构化 block copy。
- 不再把 lower-left block 写回 GM 后再 `CopyGmFloatMatrixToHalfL1()` backfill。
- 先做 16x16/32x32 compact NZ offset 的独立数值探针，再接入主路径。

### 3. L0C CMatrix 累加继续保留

fast inverse 使用 `TMATMUL_ACC` 将结果累加在 L0C。我们当前 halflocal/raw helper 已经有类似语义：`INV@I` 初始化 L0C，`INV@P` 用 `cmatrixInitVal=false` 累加到同一 CMatrix。

可借鉴方向不是回退到标量 add，而是继续保留 CMatrix 累加，并把写回目标从 GM 改成 half/bfloat16 L1/TSCM，作为下一轮 A/B 的输入。

### 4. 双缓冲和异步重叠

fast inverse 为 L0A/L0B/L0C 准备双 buffer，重叠搬运和 `TMATMUL`。当前 `fulllocal` 有 `aicHalfL0AAlt_` / `aicHalfL0BAlt_`，但主要只用于 `HalfL1InvAddBlock()` 中 `INV@I` 与 `INV@P` 两次 MMAD 的输入隔离，没有形成跨 round 或跨 tile 的系统性 pipeline。

可借鉴方向：

- `power@power` 与 `INV@(I+power)` 之间做 L0A/L0B ping-pong。
- 对多个 16x16 block 或多个 `nId/chunk` 批量处理时尝试 preload next tile。
- 只有当 GM 往返减少后，双缓冲才可能显著收益；否则会被外层同步和 GM flush 掩盖。

## 对当前问题的优化判断

### 可以优化的问题

1. `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 的固定开销。
   - fast inverse 的 resident dataflow 正是针对这个问题。
   - 当前要真正收益，必须减少 GM 中转，不只是把 16x16 内部换成 half L1。

2. C64 compose 的 GM/backfill 开销。
   - 这是 `C64_3232 fulllocal` 慢于 halflocal 的重要来源之一。
   - 结构化 odd/even block copy 可以作为 compose resident 的参考。

3. `INV += INV@P` 的额外写回/再读。
   - halflocal 已把 add 放进 L0C，但下一轮仍从 GM 读。
   - 下一步应把 updated `INV` 通过 `Fixpipe<half/bfloat16_t, float, CFG_NZ>` 写入 L1/TSCM，再用 LoadData 进入下一轮。

### 暂时不能直接优化的问题

1. FP32 resident。
   - 当前 DAV2201 上 `float L0C -> float L1/TSCM` 不可编译，已验证只能走 half/bfloat16 等低精度 L1 resident 路线。

2. 直接照搬 PTO kernel。
   - fast inverse 是独立 tri-inverse kernel，一次 AIC 可处理多个 tile，入口/出口固定。
   - 我们的 Stage1 与 AIV 的 `gBK/vBeta/qPrime` 生成交织，并且 final RHS 仍要消费 `INV`，需要重构数据流后才能借鉴。

3. 只优化 16x16 inverse 内部。
   - 当前 C16 fulllocal 已证明这种局部优化和 halflocal 基本持平。
   - 不处理 final RHS resident 时，写回 GM 后再读 GM 的成本仍在。

## 当前 fulllocal 状态对照

### C64_3232 fulllocal

- 精度：已通过全量 compare。
- 性能：`S=16/64/125/2048` Neumann/normal ratio 为 `1.307x / 1.957x / 2.286x / 3.011x`。
- 相比最佳 halflocal 更慢，主要来自 AIC scalar、额外搬运、Fixpipe/MTE 和 compose backfill。

### C16 fulllocal

- 当前 active selector：`fulllocal/C16_FP16/chunkSize=16`。
- 精度：`20 passed`。
- 性能：`S=16/64/125/2048` Neumann/normal ratio 为 `1.084x / 2.140x / 2.503x / 4.076x`。
- 相比 `halflocal/C16_FP16` 基本持平：`1.000x / 1.007x / 1.011x / 0.996x`。
- 结论：可作为 resident inverse PoC，但不是性能突破方案。

## 建议实施顺序

1. C16 final RHS resident PoC。
   - 目标：在 16x16 inverse 结果仍在 half L1/TSCM 时，直接完成 `INV @ gBK` 和 `INV @ vBeta`，或至少减少一次 `INVWsGm_` 写回/读回。
   - 原因：C16 没有 compose，是验证 fast inverse dataflow 是否能给当前算子带来净收益的最小闭环。

2. C64_3232 compose resident。
   - 目标：把 `Compose32From16()` / `Compose64From32()` 的 lower-left block 从 GM raw MMAD + scalar backfill 改成 half L1/TSCM resident。
   - 先验收 compact NZ offset 和 A/B LoadData 语义，再接入主路径。

3. 双缓冲和批 tile pipeline。
   - 目标：在 GM 往返减少后，再重叠 L1/L0 搬运与 MMAD。
   - 当前直接做双缓冲收益有限，因为主耗时还被 GM workspace 和同步主导。

4. C64_1616 暂不优先。
   - 16x16 子块数量太多，当前固定调度开销已经被性能数据放大。
   - 除非先解决 resident dataflow 和批 tile pipeline，否则继续深挖收益风险高。

## 总结

fast inverse 对当前问题有参考价值，而且方向与我们此前 profiling 判断一致：瓶颈不是理论 MAC，而是 tiny MMAD 调度、GM workspace、Fixpipe/MTE 和同步。

但当前 `fulllocal` 还没有复现 fast inverse 的关键收益结构。它只是把 16x16 inverse 子流程局部 resident，仍在 block 边界、compose 和 final RHS 上回到 GM。后续真正值得做的是把 resident 边界向外扩：先覆盖 C16 final RHS，再覆盖 C64 compose，最后再考虑双缓冲和多 tile pipeline。
