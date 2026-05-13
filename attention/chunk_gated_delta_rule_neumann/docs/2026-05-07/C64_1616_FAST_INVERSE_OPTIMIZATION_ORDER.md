# 2026-05-07 C64_1616 Fast Inverse 优化顺序

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

当前 `baseline/C64_1616_FP16` 已将新版 BF16 主链路作为基线，求逆局部使用 FP16 half L1/TSCM。当前实现仍不是 full-local resident：

- AIV 生成 FP32 `L_powWsGm_` / `INVWsGm_`。
- AIC 再将 FP32 GM block 通过 raw MMAD + Fixpipe 转成 half L1/TSCM。
- 16x16 Neumann inner loop 局部 resident，但 block 完成后仍写回 GM。
- C64 compose 与最终 BF16 `attnWsGm_` 仍经过 GM。
- 后续 `attn @ gBK` / `attn @ vBeta` 仍是 BF16 GM MatmulImpl。

用户指出第一项应先改为：AIV 从 `kkWsGm_` 做 mask/beta/gamma 后，直接生成 FP16 inverse 初始矩阵。这里的落地口径是：AIV 直接生成 FP16 `L` 与 FP16 `I-L` 初始 scratch，AIC 继续用 FP16 cube 完成 Neumann/compose；不把 AIV 改成最终完整 inverse 计算单元。

## 从简单到困难的实施顺序

## 强制验证 Gate

每个优化项必须单独实现、单独验证，不允许多个优化叠加后一起验收。

执行规则：

1. 每次只打开一个优化开关或只合入一个数据流变化。
2. 该优化必须先完成 build/install。
3. 先跑最小 smoke 精度，覆盖该优化实际触发的边界。
4. smoke 通过后再跑全量 compare。
5. 只有全量 compare 通过后，才能进入下一个优化项。
6. 若任一 smoke 或 full compare 失败，必须停在当前优化点正向定位和修复；不得继续叠加后续优化。只有明确放弃候选或需要重建干净工作面时，才恢复到上一 PASS baseline。
7. P4/full-local 改造期间只跑精度，不跑性能；阶段内部 probe/substep 只做 smoke/full compare 诊断。
8. P4 的结束条件是完整 fulllocal 路径闭环，而不是 diagonal staging 子项通过。
9. 只有整个 fulllocal 改完且 smoke/full compare 通过后，才跑一次典型 shape 性能并暂停。

最低验证命令：

```bash
CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare
```

推荐 smoke 覆盖：

- `S=16`：只覆盖第一个 16x16 block。
- `S=17`：触发第一层 16->32 compose 边界。
- `S=33`：触发第二个 32 block 边界。
- `S=64`：完整 C64。
- `S=125`：多 chunk + tail。

文档记录要求：

- 每个优化项完成后必须在 `WORK_LOG.md` 记录 build/install/smoke/full compare 结果。
- P4/full-local 改造期间不跑性能；已产生的 P1/P2/P3/P4 partial 性能只作为历史诊断记录，不再作为当前执行节奏。
- 只有完整 fulllocal active path 通过 smoke 和 full compare 后，才跑一次典型 shape 性能，然后立即停止。
- 同步更新 `PROGRESS_AND_ISSUES.md` 的已完成、进行中和阻塞项。
- 未通过精度的优化不得作为下一个优化的基线。

### P1：AIV 直接生成 FP16 `L` / `I-L` 初始 scratch

目标：

- 将当前 AIV `InverseCompute()` 的 FP32 `L_powWsGm_` / `INVWsGm_` 入口改成 FP16 scratch。
- AIC 入口改为消费 FP16 scratch，避免当前 `FP32 GM @ identity -> L0C -> Fixpipe half L1` 的绕路。
- 保持 final `attnWsGm_` 为 BF16，后续主 matmul 仍是 BF16。

预期收益：

- 降低 inverse 入口 GM 带宽。
- 删除一部分 FP32 GM -> half L1 的 raw MMAD/Fixpipe 转换。
- 改动主要集中在 Stage1 求逆相关 workspace、AIV 初始矩阵生成、AIC load helper。

风险：

- 需要同步 host workspace size、tiling data、kernel workspace 切分。
- 当前代码里没有已验证的 `half GM -> half L1/TSCM NZ` 直接搬运闭环，不能默认 `DataCopy` 一步可用。
- 必须先做入口消费路径探针：
  - AIV 写 FP16 `L/I-L` 到 GM scratch。
  - AIC 将 FP16 GM scratch 搬入 half L1/TSCM，并用 `block @ I` 或一轮 `power @ power` 与当前 FP32 入口路径对齐。
  - 若直接搬运无法表达所需 NZ layout，再考虑 half GM MatmulImpl 或其它已验证路径；不能回退到 `FP32 GM @ identity` 绕路。
- workspace 不能只改 kernel 指针类型；host `stageWorkspaceSz` 当前按 `L_pow/tmp/INV` 三块 FP32 `c*c` 计算，P1 若改为 `L/INV` FP16、`tmp` 保持 FP32或删除，必须同步精确重算。

验收：

- 先做 `half GM -> half L1/TSCM -> block@I -> GM dump` 探针，与当前 FP32 入口路径对齐。
- `S=16/17/33/64` smoke。
- `CGDR_GOLDEN_CHUNK_SIZE=64 ... test_run.sh compare` 全量通过。

### P2：缓存 half identity，减少重复 GM identity 准备

目标：

- 避免每个 `Neumann16InverseBlock()` / compose block 都重复 `WriteIdentityBlockGm()` 和 `LoadGmFloatBlockToHalfL1ByMmad(identity)`。
- 每个 C64 chunk 或每个 AIC task 只准备一次 16/32 可复用 half L1 identity。

预期收益：

- 降低 AIC scalar、MTE/Fixpipe、同步开销。
- 风险相对低，可独立验证。

风险：

- compact 16x16 与 32x32 identity 在 L1/NZ layout 中的复用 offset 要确认。

验收：

- 与 P1 相同，另重点观察 `S=16/S=64` task time 和 AIC scalar 指标。

### P3：inverse 内部 FP16 GM 化，减少 FP32 `INVWsGm_` 中间态

目标：

- 在 P1 入口 FP16 的基础上，把 16x16 Neumann 输出和 `Compose32From16()` 的输入/输出也尽量保持 FP16 scratch。
- FP32 `INVWsGm_` 只作为必要调试或回退路径，不作为 inverse 内部主中间态。

预期收益：

- 进一步减少 FP32 GM 写回/读回。
- 减少 compose 前后 FP32->half L1 转换。

风险：

- 当前 helper 大量以 `GlobalTensor<float>` 为输入输出，需要系统性改成 half 分支。
- final BF16 `attnWsGm_` 仍要保留，不能影响后续 BF16 MatmulImpl。

验收：

- smoke 覆盖 `S=17` 和 `S=33`，确保 compose 边界正确。
- 全量 compare 通过后再看性能。

### P4：C64 full half L1/TSCM staging

目标：

- 让 `aicHalfL1Inv_` / `aicHalfL1Power_` 真正保存 C64 full matrix resident 状态。
- 入口从 FP16 GM 只搬一次到 full half L1。
- 4 个 16x16 diagonal inverse 和 C64 compose 都从 full half L1 取数、回写 full half L1。
- 最后一次性写出 final inverse 或 BF16 `attnWsGm_`。

预期收益：

- 把当前多次 `GM -> half L1 -> GM -> half L1 -> GM` 收敛为更接近 `GM -> full L1 -> GM once`。
- 这是 fast inverse 思路真正可能产生收益的关键步骤。

风险：

- full/compact L1 offset、A/B LoadData transpose 语义、tail validLen 都要重新验证。
- 这是从局部 resident 走向 C64 matrix resident 的分水岭。

验收：

- 先做 full L1 block copy 探针，再接入主路径。
- `S=16/17/33/64/125` smoke，全量 compare。

### P5：朴素 Compose32/Compose64 full-L1 resident（可被 P7/P8 取代）

目标：

- `Compose32From16()` 不再从 GM 读 lower/top/bottom block，也不再写 `tmpWsGm_`。
- `Compose64From32()` 不再走 FP32 GM raw MMAD。
- compose 的中间结果保留在 compact/full half L1/TSCM。
- 这是直接替换现有 GM compose 的朴素 full-L1 版本，可作为 correctness baseline；不一定要作为性能主线实现。

预期收益：

- 消除 C64 compose 的主要 GM/backfill 成本。
- 对 `S>=64` 更关键。
- 但该路径仍可能保留多个小 block MMAD 和同步，未利用 fast inverse 的 packed diagonal / odd-even packing。

与 P7/P8 的关系：

- 可以直接用 P7/P8 代替 P5，作为 fast inverse 主线。
- P5 的价值是降低风险：先用最直观的 full-L1 compose 验证 full/compact offset、数学顺序和 tail，再替换成 packed 版本。
- 如果开发时间有限，应优先做 P7/P8；P5 只保留为调试 fallback 或中间对照。

风险：

- 需要保证 16->32、32->64 的 lower-left 数学顺序完全对齐现有 GM baseline。
- `S=17/S=33/S=64` 是必要边界 case。

验收：

- `S=17` 验证第一层 compose。
- `S=33` 验证第二个 32 block。
- `S=64` 验证完整 64 compose。

### P6：AIC 直接写 BF16 `attnWsGm_`，去掉 AIV FP32->BF16 cast

目标：

- 在 final inverse 已位于 half L1/TSCM 后，由 AIC 直接写 BF16 `attnWsGm_`。
- 删除当前 `AIC 写 FP32 INVWsGm_ -> AIV 读 INVWsGm_ -> AIV cast -> 写 attnWsGm_` 的后段 GM 往返。

预期收益：

- 减少 FP32 `INVWsGm_` final materialization 和 AIV cast。
- 保持后续 BF16 `attn@gBK/vBeta` 不变。

风险：

- half L1 -> BF16 GM 的写回仍可能需要 L0C/Fixpipe，中间同步要处理好。
- 建议在 P4 后做；若 P5 被 P7/P8 取代，则应在 P7/P8 compose 主线稳定后做。否则当前 compose 仍依赖 FP32 GM，收益有限。

验收：

- compare 全量通过。
- 观察 `attnWsGm_` 后续两个 final matmul 是否有 cache/order 问题。

### P7：packed diagonal 16x16 Neumann

目标：

- 参考 fast inverse `CopyDiagonalFractalsL1ToL0()`，把 C64 内 4 个 16x16 对角块打包成 block-diagonal tile。
- 用更大的 packed MMAD 批量完成 `power@power` 与 `inv + inv@power`，减少 tiny MMAD 调用次数。

预期收益：

- 减少四个独立 16x16 Neumann block 的调度、LoadData、同步开销。
- 这是 fast inverse 的核心结构化搬运之一。

风险：

- 需要构造 block diagonal L0A/L0B，off-diagonal 必须为 0。
- L0C 输出要正确散回 full half L1 对角位置。
- 难度高，不应在 full L1 staging 稳定前切入默认路径。

验收：

- 先做 `block@I` packed 探针，与逐块结果对齐。
- 再做 packed Neumann，最后接入主路径。

### P8：odd/even packed MBH compose

目标：

- 参考 fast inverse `CopyOddOrEvenBlocksL1ToL0()`，把 16->32、32->64 compose 的奇偶 block 结构化搬运到 L0。
- 用 packed compose 降低多个小 MMAD 调用与同步成本。

预期收益：

- 进一步降低 C64 compose 固定开销。
- 配合 P7 才更接近 fast inverse 的完整 dataflow。

风险：

- 比 P7 更依赖布局和数学顺序。
- 一旦 offset 或 transpose 语义错误，`S=17/33/64` 会出现大误差。

验收：

- odd/even copy 独立探针。
- `S=17/33/64/125` smoke，全量 compare。

### P9：真正的 L0A/L0B ping-pong 与 pipeline

目标：

- 将当前顺序式 `LoadData -> Mmad -> Fixpipe` 拆成可重叠的 preload/current compute。
- 对多个 16x16 block、compose block 或多个 chunk 做搬运和 MMAD overlap。

预期收益：

- 在 GM 往返已经减少后，进一步压低 MTE1/MMAD 等待。

风险：

- 当前依赖链很强，事件和 buffer 生命周期复杂。
- 过早做会被 GM materialization 成本掩盖，且容易引入隐性同步 bug。

验收：

- 只在 P4-P8 稳定后做。
- 必须用 msprof 验证 MTE/MMAD overlap 真实存在。

### P10：LHS resident KVMM，RHS 仍按 BF16 GM 消费

目标：

- 让后续 `attn @ gBK` / `attn @ vBeta` 不再从 GM 读 `attnWsGm_`。
- `gBK/vBeta` 仍保持当前 BF16 GM 形态，由 final matmul 正常从 GM 搬运。
- 真正要省的是 LHS：避免 final inverse 先 materialize 到 BF16 `attnWsGm_`，再被两个 final matmul 分别读一次。

当前判断：

- 这是较难项，不作为近期第一阶段，但不应再表述为“RHS 从 GM 搬运导致额外成本”。
- 当前新版 baseline 中 `gBKWsGm_` / `vBetaWsGm_` 本来就是 BF16 GM，不能再按旧 C16 PoC 的“FP32 RHS GM 转 half L1/TSCM”口径描述。
- 当前 BF16 GM x GM `MatmulImpl` 本来就会搬运 RHS；resident KVMM 若实现为 `A: resident/L1, B: BF16 GM, C: BF16 GM`，RHS 搬运不是新增成本。
- 真正难点是实现形态：
  - 需要 final matmul 支持 LHS 从 half/BF16 L1/TSCM resident tile 进入，RHS 仍从 BF16 GM 进入。
  - 如果为了 raw MMAD 把 BF16 RHS 额外转成 half L1/NZ，那才会引入额外 RHS 转换成本；这不是首选方案。
  - resident LHS 必须跨 `attn@gBK` 与 `attn@vBeta` 两个 matmul 保持有效，期间还要等待 AIV 生成 `gBK/vBeta` 的同步 flag。
- 因此近期仍保持两个 BF16 final matmul：`AICProcess(attnWsGm_, gBKWsGm_, ...)` 与 `AICProcess(attnWsGm_, vBetaWsGm_, ...)`。

验收：

- 只有在 P4-P8 已明显减少 inverse/compose 成本后再重新评估。

## 当前建议顺序

近期建议只推进前 3 步：

1. P1：AIV 直接生成 FP16 `L/I-L` 初始 scratch。
2. P2：identity cache。
3. P3：inverse 内部尽量 FP16 GM 化，减少 FP32 `INVWsGm_` 中间态。

这三步仍保持后续 BF16 final matmul 不变，且只改 Stage1 求逆相关路径。跑通后进入 P4 full L1 staging；compose 主线优先走 P7/P8 的 packed fast inverse 方法，P5 仅作为可选 correctness baseline。
