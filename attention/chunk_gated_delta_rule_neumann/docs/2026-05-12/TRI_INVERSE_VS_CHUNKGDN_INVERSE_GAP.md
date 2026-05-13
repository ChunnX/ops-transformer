# tri_inverse vs chunkGDN 求逆差距记录

Date: 2026-05-12

本文只记录 `tri_inverse` 纯求逆方案和 chunkGDN 当前 `p3l1_p8a / C64_1616_FP16`
求逆路径的差距。RHS 只作为完整算子额外消费项记录，不作为纯求逆性能差距的解释。

## 对比口径

`tri_inverse` 是独立 lower-triangular inverse 算子：

```text
BF16 ND [tile_num, 64, 64] L -> inverse -> BF16 ND output
```

chunkGDN 当前 Stage1 求逆路径在完整算子里执行：

```text
Load L/power
-> P7 diagonal inverse
-> P8 lower compose
-> BF16 GM inverse materialization
-> resident L1 inverse exposure
-> later RHS consumes resident inverse
```

因此纯求逆对比只看：

```text
Load L/power -> P7 -> P8 -> inverse output/materialization
```

不把 `FusedRhsAICProcess()` 计入纯求逆差距。`resident L1` 也不是数学求逆本身，
但它是 chunkGDN 后续 RHS 的输入暴露方式，当前实现会影响完整算子耗时。

## 当前 profile 事实

chunkGDN A44 profile：

```text
tests/pytest/prof_out/packed64_phaseA44_identity_epoch_after_preload_20260512/performance_compare.csv
neumann/normal: S=64/512/2048/10240 = 1.315 / 1.971 / 2.824 / 2.860
```

`S=10240` 的主要增量不是单一 MTE2：

```text
AIC scalar: 44.160 ms vs normal 9.649 ms
AIV scalar: 20.540 ms vs normal 9.736 ms
AIC fixpipe: 10.998 ms vs normal 7.056 ms
AIC MTE2: 5.099 ms vs normal 4.441 ms
```

这说明当前差距包含循环/同步/copy-out/materialization/resident 暴露等开销。
若只比较纯求逆，需要额外做 inverse-only 计时或拆分 profile，否则会混入 RHS 和完整算子路径。

`tri_inverse` Phase13 profile：

```text
experimental/attention/tri_inverse/docs/2026-05-11/MIX_INVERSE_NEXT_WINDOW_HANDOFF.md
shape=10240: Duration(us)=71.52
aic_mte2_time(us)=15.691
aic_fixpipe_time(us)=11.657
aic_scalar_time(us)=13.956
```

Phase13 仍慢于它自己的 v0 reference `66.26 us`，但已经把若干 movement 字段降下来了。

## tri_inverse 已做但 chunkGDN 没完全做到

### 1. lowerWork 一次搬入和复用

`tri_inverse` Phase11/Phase13 已保存：

```text
loaded 32x32 compose lowerWork once
aic_mte2_time(us) at 10240: Phase12 19.025 -> Phase13 15.691
```

chunkGDN 当前 strict full64 path 仍在 `PreloadP8LowerLeftBlocksToLowerWork64NoWaitFull64()` 中分三次搬：

```text
(16,0) 16x16
(48,32) 16x16
(32,0) 32x32
```

其中两个 first-layer lower blocks 具有固定形状和固定 packed lane 关系，应作为独立 phase
合并或 paired preload；`(32,0) 32x32` 是 second-layer 输入，应单独处理。

计划映射：

```text
A46: P8 lower-left first-layer paired preload
A47: P8 second-layer packed-MMAD load/tmp lifetime
```

### 2. P8 first-layer output placement

chunkGDN 当前 `P8ComposeFirstLayerPacked32Direct()` 仍有两段冗余：

```text
tmp = lower @ topInv
-> CopyL0CMatrixToHalfL1(tmp32, 32, 32)
-> 下一步只读取两个 diagonal lane

out = bottomInv @ tmp
-> 先写完整 packed32 到 aicHalfL1TilePower_
-> 再 CopyHalfL1PackedBlockToCompact() 到 result64 的两个 16x16 lane
```

这应改成更接近最终消费者的落点：

```text
first-layer final (16,0), (48,32)
直接落 result64 对应 packed offset
后续再扩展为同时或直接落 resident RHS 消费位置
```

计划映射：

```text
A45: P8 first-layer packed32 output placement
```

### 3. P7 L0C copy-out 仍是 full64

chunkGDN 当前 `P7Packed64DiagonalNeumannOneShotToResult()` 每一轮都：

```text
FixpipePacked64FullToHalfL1(aicHalfL1P3Result_, cInv, ...)
```

但 P7 diagonal-only 后续主要消费四个 diagonal fractal。理论目标是减少 full64
L0C->L1 copy-out。

已知失败边界：

```text
A42 exact four-diagonal L0C->L1 Fixpipe candidate: S=64 timeout
```

原因不是目标错误，而是 L0C->L1 `Fixpipe<..., CFG_NZ>` 不等同于 GM->L1 `DataCopy`，
不能直接假设支持四个 diagonal packed offsets 的 multi-ND scatter。后续要使用硬件支持的
矩形/分组 copy-out 形态，并确认 L0C source stride/offset，不再套用 L1 packed NZ offset。

计划映射：

```text
A42-deferred: P7 L0C copy-out
```

### 4. diagonal inverse 粒度：继续四个 16x16 packed64

`tri_inverse` Phase7 已做：

```text
two direct 32x32 diagonal cube inverses
```

这不是 chunkGDN 的主线。chunkGDN 当前应继续保持：

```text
four 16x16 diagonal fractals packed in one 64x64 tile
one packed64 P7 path handles the four diagonal subproblems together
```

原因：

- 当前 P7 已围绕四个 diagonal fractal 的 packed64 one-shot 组织。
- P8 first-layer/second-layer 也已经围绕 packed64 result 和 lowerWork64 组织。
- 改成两个 direct `32x32` diagonal inverse 会重排 P7/P8 主数据流，可能引入新的
  lower block 搬运和落点问题。
- 用户确认当前更优方向是四个 `16x16` 一起打包，而不是改 diagonal solve 粒度。

可借鉴的只有原则，不是粒度：

```text
减少中间 roundtrip
让最终 inverse block 更早落到消费者位置
避免 full64 copy-out 只服务 diagonal 消费
```

因此后续不得把 `tri_inverse` Phase7 的 two direct `32x32` diagonal inverse
作为 chunkGDN P7 主线。Packed-MMAD 仍是目标，冗余是搬运、copy-out、layout-copy
和 identity 生成。

### 5. identity64 生命周期：独立常驻 L1 buffer

chunkGDN A44 已把 identity64 ensure 调整到 inverse epoch 内：

```text
PreloadP8LowerLeftBlocksToLowerWork64NoWaitFull64()
-> EnsurePacked64IdentityForInverseEpoch()
-> P7Packed64DiagonalNeumannOneShotToResult()
```

但这仍不是 operator-lifetime once。当前还受这些边界约束：

```text
Init()
InitRawMmadLocalBuffers()
RestoreStage1MatmulLocalBuffers()
ParaChunkAIC() inverse-window boundary
```

已知失败：

```text
A30: 只延长 packed64IdentityPrepared_ 生命周期，S=1024 FAIL
A31: same-GM-region full64 identity reload，S=64 FAIL
A38 exact scalar direct NZ writer，S=64 FAIL
```

重要更正：

```text
identity64 本身是常量。
从工程设计上，应申请一块不参与 scratch/reuse 的 L1 buffer，一直存放 identity64。
```

当前难点不是 identity 值，而是当前 `aicHalfL1Identity64_` 仍挂在
`aicHalfL1Buff_` / raw-MMAD local buffer 管理下。`InitRawMmadLocalBuffers()` 和
`RestoreStage1MatmulLocalBuffers()` 都会 `pipe_->Reset()`，并重新初始化 local buffer；
`ParaChunkAIC()` 还会显式清 `packed64IdentityPrepared_`。因此现有实现不是
“独立常驻 L1 identity buffer”，而是“raw-MMAD local buffer epoch 里的一个 tile”。

下一次 identity64 收口不应再做 A30 式“延长 prepared flag”。正确方向是：

```text
申请独立常驻 identity64 L1 buffer
该 buffer 不被 P7/P8 scratch、resident、BF16 materialization 复用
该 buffer 不被 pipe_->Reset() 后的 InitBuffer() 覆盖，或 reset 后明确重新绑定并保持内容有效
每个 AIC core 各自生成一份 identity64，因为 L1 是 core-local
首次生成后，后续 P7/P8 只读该 buffer
```

如果当前 TPipe/LocalTensor 机制无法保证某个 L1 buffer 跨 `pipe_->Reset()` 保持内容，
则需要把 identity64 buffer 从 raw-MMAD reset 域中拆出来，或者把 `pipe_->Reset()`
的使用范围重构到不会覆盖该常驻区。这个方向和 `tri_inverse` 的求逆过程是一致的；
差异只在 chunkGDN 当前 buffer 生命周期还没有按常驻 identity 设计。

### 6. resident L1 落点没有和 P8 final output 对齐

当前 RHS 消费位置：

```text
resident = aicHalfL1ResidentInv_[paraIdx * AIC_LOW_L1_TILE_ELEMS]
RawMmadPackedHalfL1AccumulateRect(resident, rhsPacked, ...)
```

当前流程是：

```text
P7/P8 -> aicHalfL1P3Result_ packed64
StorePackedL1InverseToResidentL1()
  -> ZeroHalfL1(resident, 64*64)
  -> lower 10 个 16x16 block 逐个 CopyHalfL1PackedBlockToCompact()
```

这一步是冗余的过渡实现。更合理的方向是：

```text
P8 每个最终 inverse block 产生时，直接落到 resident 的 RHS 消费位置
```

可分两步：

1. 保守版本：

```text
P7/P8 中间仍用 aicHalfL1P3Result_ 保持 packed64 计算布局
P8 final output 同时或直接写 resident 对应 offset
最后逐块 StorePackedL1InverseToResidentL1() 删除或收缩
```

2. 激进版本：

```text
P7 diagonal inverse 一出来就落 resident 的 diagonal offset
P8 后续从 resident 读取 left-top/right-bottom 组成矩阵参与 compose
P8 lower-left final 也落 resident
```

激进版本需要把 P8 读源从 `aicHalfL1P3Result_` 改到 `resident`，必须重新验证 packed offset、
MTE1 可见性和 RHS 消费布局。推荐先做保守版本。

### 7. BF16 GM inverse materialization 仍是 per-16x16 layout-copy

当前 `StorePackedL1InverseToBf16Attn()` 遍历 4x4 个 `16x16` block：

```text
lower 10 个 block: CopyPackedHalfL1BlockToGmBf16()
upper 6 个 block: CopyHalfL1MatrixToGmBf16ByMmad(zero)
```

这些 helper 本质上是 `X @ I16` layout-copy/convert，再 Fixpipe 到 GM。它们不是数学求逆，
但当前 BF16 GM writeback 仍携带同步/可见性边界，不能直接删除。

后续方向：

```text
row prefix:
row0  -> 16x16
row16 -> 16x32
row32 -> 16x48
row48 -> 16x64
```

BF16 转换仍需 Fixpipe，不能直接 same-dtype DataCopy 替代。先拆同步/可见性边界，再合并 row-prefix。

### 8. AIV baseline/scratch 路径没有完全剥离

`tri_inverse` Phase5 已删除 old AIV baseline inverse compute path，Phase12 又通过重排 AIV 写入删除 top diag reload。

chunkGDN 仍保留 AIV 侧生成和 GM scratch：

```text
L_powHalfWsGm_
INVInitHalfWsGm_
```

这些在完整 chunkGDN 中不一定都能直接删除，因为上游数据来自原算子流水。但能做的方向是：

```text
减少 AIC 从 GM 重载已经在 L1 可得到的 diagonal/lowerWork
减少为了 copy/layout 生成的临时 I16
把只服务求逆的 scratch 写入变成 L1-resident 或 packed direct load
```

## 不应混入纯求逆对比的项

### RHS 计算

`FusedRhsAICProcess()` 做的是：

```text
resident inverse @ gBKHalfWsGm_
resident inverse @ vBetaHalfWsGm_
```

它是求逆后的消费者，不是纯 inverse 核心。完整 chunkGDN latency 会包含它；
和 `tri_inverse` 的纯求逆 latency 对比时必须排除。

### resident L1 copy

`StorePackedL1InverseToResidentL1()` 不是数学求逆，但它是当前 chunkGDN 给 RHS 暴露 inverse 的实现。
若只比较纯 inverse 核心，可以单独列为“输出/暴露开销”；若比较完整算子，它必须计入。

### normal op 对比

`neumann/normal` profile 是完整 op 对比，不等于纯 inverse vs pure inverse。
它能提示瓶颈字段，但不能直接说明 P7/P8 纯求逆慢多少。

## 为什么 A4X 失败而 tri_inverse 能推进

主要不是目标不可行，而是边界不同：

1. `tri_inverse` 边界干净，只需要输出 inverse。
2. chunkGDN 还要维护 BF16 GM materialization、resident L1、RHS 消费和 AIV/AIC 同步。
3. A4X 中有几项切到了最硬边界：
   - A30/A31: identity lifecycle 未证明就延长复用范围。
   - A42: 把 L0C->L1 Fixpipe 误当成 GM->L1 DataCopy 式 scatter。
   - A26/A27: resident/BF16 双写没有拆清 Fixpipe 可见性和 RHS 消费时序。
4. 后续应优先迁移 `tri_inverse` 已证明有效的 movement 优化，再回到 L0C copy-out 和 lifecycle。

## 建议后续顺序

所有步骤仍遵守：

```text
one optimization per phase
build/install
S=64 -> S=256 -> S=1024 smoke
full compare
PASS 后保存 baseline
precision FAIL 时在当前目标正向定位修复，不把回滚当完成
```

推荐顺序：

1. 建立 inverse-only 计时或 profile 口径，剥离 RHS，至少区分：
   - P7/P8 core
   - BF16 GM inverse output
   - resident L1 exposure
2. A45：P8 first-layer output placement。
3. A46：P8 first-layer lower-left paired preload。
4. A47：P8 second-layer `32x32` lowerWork/tmp lifetime。
5. A48：P8 final output direct-to-resident 的保守版本。
6. A49：BF16 GM materialization row-prefix 合并。
7. A44-next：identity64 独立常驻 L1 buffer。
8. A42-deferred：P7 L0C copy-out，用合法矩形/分组形态解决。
9. 完成上述 baseline 后，再用 msprof 或 simulator 查新优化点。

## Phase23-27 增量对比和继承表

重新核对 `experimental/attention/tri_inverse` 后，当前不应只看 Phase23。
该目录已经保存了 Phase25/26/27 baseline，且 active `tri_inverse_mix.h`
又在 Phase27 之后有未保存候选。继承到 chunkGDN 时必须区分：

```text
已保存 full30 PASS baseline -> 可以纳入 chunkGDN 计划
active 未保存候选 -> 只能记录为待验证思路，不能当作已验证事实
```

当前 `tri_inverse` 已核对的 SHA：

```text
Phase23 baseline: 7267d88fbd5dfa317488499b0b0a3c9f87b23ea5c5f27de8f9e484ffe5e55388
Phase25 baseline: ad6cff95037497637706c044f9203468df22b233429a97afb7ba146be5471b5e
Phase26 baseline: f082fad1a791458d8d5ffd1729a6b2630df5a838c68208236543012de268151e
Phase27 baseline: 51e3fec242721ed34ae5479b0c461c680dbb82f05bda2f43bb77159598e3e50c
active candidate:  bbdc2de9dd6b17577f288932c5c9a692cb0795788a735ad48d0b25e9195136eb
```

### Phase25: diagonal32 只清 top-right

`tri_inverse` Phase25 把 `LoadDiag32PowerScratch()` 里的整块 `32x32`
清零改成只清 `32x32` power block 的 top-right `16x16`：

```text
旧：ZeroHalfL1(dstPower, 32x32)
新：只 ZeroHalfL1(dstPower[(0,16)], 16x16)
```

profile 记录：

```text
Phase23 10240 Duration 62.401 us, aic_mte2 10.788 us
Phase25 10240 Duration 63.081 us, aic_mte2 10.628 us
```

总耗时没有全局变好，但 `aic_mte2` 有下降，所以这是可继承的单点：
只要它减少了具体硬件字段，就应保存为分支点，而不是因为总 Duration
波动就丢掉。

chunkGDN 对应点不是直接照搬 `32x32`：

```text
P7 LoadPacked64DiagonalInputsToL1()
当前仍 ZeroHalfL1(aicHalfL1P3Result_, 64x64)
然后只加载四个 diagonal fractal
```

继承方向：

```text
只清 P7 packed64 中后续会被读取但不会被 diagonal load 覆盖的必要 off-diagonal 区域
避免每个 tile 整块 64x64 result/input scratch zero
```

风险边界：

- P7 当前是 four `16x16` diagonal fractals packed in one `64x64` tile，
  不是 two `32x32` diagonal solve。
- 不能破坏 `LoadPacked64DiagOnlyToL0()` 读取四个 diagonal fractal 的假设。
- 如果某个 off-diagonal 仍被 full64 Fixpipe 或后续 BF16/resident materialization
  间接读取，就必须保留对应清零。

### Phase26: result64 只清 top-right

`tri_inverse` Phase26 把 `MixInverseCubeDiag16()` 里的：

```text
ZeroHalfL1(aicHalfL1Result64_, 64x64)
```

替换为只清 result64 的 top-right 两列 `16x16` fractal。因为 lower-triangular
输出最终只需要：

```text
top-left 32x32
bottom-right 32x32
lower-left 32x32
top-right 32x32 = zero
```

chunkGDN 当前对应点：

```text
LoadPacked64DiagonalInputsToL1(): ZeroHalfL1(aicHalfL1P3Result_, 64x64)
P7/P8 result64: aicHalfL1P3Result_
BF16 writeback: lower 10 real blocks + upper 6 zero blocks
resident: zero resident + lower 10 block copy
```

继承方向：

```text
A60: P7/P8 result64 zero 范围收缩
先只从 aicHalfL1P3Result_ 的 full64 zero 入手
保留 BF16 upper-zero row-suffix 和 resident zero
```

不能一次混做：

- 不和 BF16 upper-zero 删除混做。
- 不和 resident zero 删除混做。
- 不和 P7 L0C diagonal-only copy-out 混做。

原因是 chunkGDN 的 `aicHalfL1P3Result_` 同时服务 P7/P8 packed-MMAD、BF16 GM
materialization 和 resident exposure；Phase26 的清零收缩要先证明这些消费者
不会读取未初始化区域。

### Phase27: identity32 core-lifetime

`tri_inverse` Phase27 把 identity32 从每个 `ProcessParaTiles` batch 准备一次，
提升为每个 AIC core 的 `Process()` 生命周期内只准备一次：

```text
Process() 入口 AIC InitRawMmadLocalBuffers()
firstBatch 写一次 identity scratch / AIC load once
后续 batch 复用 aicHalfL1Identity32_
Process() 结束 RestoreRawMmadState(); pipe_->Reset()
```

这是对 chunkGDN `identity64` 最有价值的新增优化。chunkGDN A44 目前只是
inverse-window epoch：

```text
PreloadP8LowerLeftBlocksToLowerWork64NoWaitFull64()
-> EnsurePacked64IdentityForInverseEpoch()
-> P7
```

继承方向应升级为：

```text
A62 / identity64 lifecycle:
每个 AIC core 在 raw-MMAD local buffer 初始化后生成或加载一次 identity64
同一 core 后续 para/batch 复用 aicHalfL1Identity64_
Process/ParaChunk 结束或 pipe_->Reset() 前后明确失效边界
```

注意这仍不是“全算子全局唯一一份”。L1 是 core-local，所以正确描述是：

```text
每个 AIC core 一份 identity64
在该 core 的有效 raw-MMAD/L1 生命周期内复用
```

最终目标仍是用户要求的独立常驻 L1 buffer：

```text
identity64 不参与 P7/P8 scratch/reuse
不被 resident/BF16/result64 覆盖
不被 pipe_->Reset() 后的 InitBuffer() 覆盖，或 reset 后明确重新生成/重绑
```

### active candidate: global identity scratch，未保存

当前 `tri_inverse` active `tri_inverse_mix.h` 在 Phase27 后又尝试：

```text
core0/subBlock0 写 GM offset 0 identity scratch
AIV/AIC 用 CrossCore flag 同步
每个 AIC core 从 offset 0 加载 identity32
后续所有 tile 复用
```

这比 Phase27 更接近“全局 scratch 生成一次”，但当前没有对应 saved baseline。
因此在 chunkGDN 里只能作为待验证思路：

```text
可以借鉴跨 AIV/AIC flag 顺序
不能直接宣称 PASS
不能直接替代 A44/A62 的 identity64 设计
```

chunkGDN 若尝试类似方向，必须先解决：

- `tmpWsGm_` / identity GM scratch 是否被其它 workspace 复用。
- 多 AIC core 从同一 GM identity 读入各自 L1 是否需要 cache clean/flush。
- `pipe_->Reset()` 是否覆盖 `aicHalfL1Identity64_`。
- 和 `ParaChunkAIC()` 的 `packed64IdentityPrepared_` 失效边界是否一致。

### Phase17/18 与 chunkGDN 输出落点

`tri_inverse` Phase17/18 的核心原则仍要继承：

```text
一个 L0C result 同时喂 L1 result64 和 standalone BF16 GM
减少第二次 INV/compose @ I replay
```

chunkGDN 当前还没有做到：

```text
P7: full64 L0C->L1，后续再 BF16/resident copy
P8 first-layer: full packed32 -> result64 two lane copy
P8 second-layer: full tmp32 -> final result64
BF16: lower 10 block X @ I16 / upper zero write
resident: lower 10 block X @ I16 copy
```

继承时优先落在两个方向：

1. 保守方向：

```text
P7/P8 仍产出 aicHalfL1P3Result_
但在 legal Fixpipe/DataCopy 形态内，让同一个 L0C result 少 replay、少 layout-copy
```

2. resident 方向：

```text
P8 final inverse block 产生时更早落到 resident RHS 消费位置
逐步收缩 StorePackedL1InverseToResidentL1()
```

已失败边界仍有效：

- A42/A45 的 partial L0C->L1 `CFG_NZ` copy-out 多次 timeout。
- A26/A27 的 resident/BF16 双写或 direct resident copy 不可照搬。

所以 Phase17/18 只能继承“一个结果服务多个消费者”的数据流原则，不能直接照搬
`tri_inverse` 的 `32x32` L0C->L1+GM Fixpipe 顺序。

### Phase19 与 AIV scratch

`tri_inverse` Phase19 让 subBlock1 直接写 bottom diagonal power scratch，
subBlock0 不再 reload/cast bottom half。

chunkGDN 当前 `InverseCompute()` 里每个 subBlock 按自己的 `subOffset_`
写 `L_powHalfWsGm_` 和 `INVInitHalfWsGm_`：

```text
subBlock0 -> top half
subBlock1 -> bottom half
```

这点看起来已经具备 Phase19 的主要性质：没有明显的 subBlock0 重新处理 bottom half。
但仍有继承检查项：

```text
A63: audit AIV scratch path
确认 P7/P8 full64 path 是否仍有 AIC 重新从 GM load/cast 已在 L1/UB 可得的数据
确认 L_powHalfWsGm_ / INVInitHalfWsGm_ 是否只是 AIV->AIC 交接，还是还有可删除 roundtrip
```

### 继承后的 chunkGDN 优先级

更新后的建议顺序：

1. A60：继承 Phase25/26，收缩 P7/P8 `result64/input64` 清零范围。
2. A61：继续清理 packed32/P8 first-layer tmp/result 的必要清零范围。
3. A62：继承 Phase27，设计 per-AIC-core `identity64` lifecycle，向独立常驻 L1 buffer 收口。
4. A63：Phase19 analogue，审计 AIV scratch 是否仍有 bottom/top 半块 reload/cast 或 GM roundtrip。
5. A64：resident exposure 从 late `StorePackedL1InverseToResidentL1()` 向 P8 final output 落点迁移。
6. A65：BF16 materialization 只保留必要 standalone/visibility 边界，继续 row-prefix 或 replay 合并。
7. A66：P7 L0A no-transpose diagonal load 继续合并。
8. A42/A45 deferred：重新设计 legal L0C copy-out 形态。
9. 完成一个新的 PASS baseline 后，用 msprof/simulator 继续找下一批优化点。
