# QkvRmsNormRopeCacheWithKScale 流水分析

## 1. 分析范围

本文以 `ROPE_FP8_QK` 场景为主，分析 arch35 上 `QkvRmsNormRopeCacheWithKScale` 的 AIV/AIC 流水。

从 Q/K 的数据路径看，它是典型的 VCV：

```text
Vector preprocess -> Cube matmul -> Vector postprocess
```

但整个算子不能简单等价为一条串行 VCV，原因有三点：

1. 一个核组由 `1 AIC + 2 AIV` 组成；两个 AIV 分担同一个 token tile。
2. V 分支不经过 Cube，只在 AIV 上完成缩放、FP8 转换和 Cache 写回。
3. AIV 对当前 tile 做前处理时，还会对前一个 tile 做后处理，形成跨 tile 的一拍 rolling pipeline。

## 2. 核组和 token 切分

执行模型固定为：

```text
KERNEL_TYPE_MIX_AIC_1_2
1 AIC + 2 AIV per core group
```

同一核组内：

```text
AIC cubeIndex = GetBlockIdx()
AIV cubeIndex = GetBlockIdx() / 2
AIV local id  = GetSubBlockIdx()  // 0: AIV0, 1: AIV1
```

对一个 token tile：

```text
actualToken      = tile.tokenSize
cubeToken        = AlignUp(actualToken, 2)
cubeHalfToken    = cubeToken / 2

AIV0 validToken  = CeilDiv(actualToken, 2)
AIV1 validToken  = actualToken - AIV0 validToken
```

如果 tile 中有奇数个 token，AIV1 可能少处理一个 token；用于 Cube/FIXP 对齐的 padding 行不会被量化或写回。

## 3. 第一段 Vector：当前 tile 前处理

每个 AIV 只处理自己的 token slice。`Vec::ComputeTile` 的主要顺序是：

```text
1. 等待 input UB 和目标 buffer 可复用
2. MTE2 搬入 Q/K/V、cos_sin、slot_mapping
3. 根据 slot_mapping 计算 K/V/K-scale Cache 地址
4. 先计算 V × v_scale，并转换为 FP8
5. 对 Q/K 计算 RMSNorm + RoPE
6. 把 Q/K 转成 BF16/NZ，准备作为 Cube 的 A 矩阵
7. MTE3 将 V 散写到 v_cache
8. MTE3 将 Q/K BF16/NZ 写入共享 AL1
9. 发送 A_READY
```

其中 V 分支与 Q/K 分支没有数值依赖，采用 V-first 是为了让 V 的 Vector 计算和 MTE3 写回尽量隐藏在 Q/K 前处理及后续 Cube 计算中。

## 4. Cube：Q/K 右乘 rotation

AIC 在循环前将 `rotation[D,D]` 从 GM 搬入 L1、转换成 NZ，并加载到 L0B 常驻复用。

对于 tile `i`，AIC 需要先满足两个条件：

```text
1. 收到 AIV0 和 AIV1 的 A_READY(i)
2. 收到 output DB 可复用的 FIX_OUTPUT_CONSUMED credit
```

随后分别计算：

```text
Q_rope(i) @ rotation
K_rope(i) @ rotation
```

Q/K 在 L0C 中保持两个 segment。FIXP 使用 dual-destination M 轴切分，把前半 token 对应的结果送到 AIV0 本地 UB，把后半送到 AIV1 本地 UB。

Cube 完成相应消费和输出后：

```text
set A_CONSUMED(i)        // AL1 槽可以再次写入
set FIX_OUTPUT_READY(i)  // AIV 可以读取 Cube 输出
```

## 5. 第二段 Vector：前一个 tile 后处理

AIV 的控制迭代 `i` 先完成当前 tile 的前处理，然后在 `i > 0` 时处理 `lastTile=i-1`：

```text
Preprocess(tile i)
set A_READY(tile i)

wait FIX_OUTPUT_READY(tile i-1)
Postprocess(tile i-1)
set FIX_OUTPUT_CONSUMED(tile i-1)
```

普通 RoPE 场景的 `PostprocessRopeQk` 包含：

```text
Q: max_abs -> scale=max_abs/448 -> FP8 -> q_out/q_scale
K: max_abs -> scale=max_abs/448 -> FP8 -> k_cache/k_scale_cache
```

最后一个 tile 没有下一次控制迭代负责回收，因此在 `EndAiv()` 中单独完成后处理，这就是 pipeline drain。

## 6. 跨 tile rolling

以三个 tile 为例，逻辑顺序是：

```text
Warmup:
  AIC: rotation 常驻、发放 AL1 credit
  AIV: 构造常驻索引、发放 output DB credit

Iteration 0:
  AIV: Preprocess T0 -> A_READY0
  AIC: wait A_READY0 -> Cube T0 -> FIX_OUTPUT_READY0

Iteration 1:
  AIV: Preprocess T1 -> A_READY1
       Postprocess T0
  AIC: Cube T1，可与 AIV Postprocess T0 重叠

Iteration 2:
  AIV: Preprocess T2 -> A_READY2
       Postprocess T1
  AIC: Cube T2，可与 AIV Postprocess T1 重叠

Drain:
  AIV: Postprocess T2
```

这说明 VCV 是数据依赖主干，但硬件执行采用跨 tile 的错拍：

```text
AIV 正在处理 tile i+1 的 Vector 工作时，
AIC 可以处理 tile i 的 Cube 工作；
AIV 随后处理 tile i 的第二段 Vector 工作。
```

## 7. 四条关键 CrossCore 同步边

### A_READY：AIV → AIC

Q/K BF16/NZ 已由 AIV MTE3 写入 AL1。AIC 必须分别等待 AIV0 和 AIV1，才能读取完整 tile。

### A_CONSUMED：AIC → AIV

AIC 已经消费 AL1 中的 Q/K A 矩阵，该 AL1 槽可以被后续 tile 复用。循环开始前由 AIC发放初始 credit。

### FIX_OUTPUT_READY：AIC → AIV

FIXP 已把 Cube FP32 结果写入两个 AIV 的本地 output DB，AIV 可以执行动态量化和输出写回。

### FIX_OUTPUT_CONSUMED：AIV → AIC

AIV 已完成 Q/K 量化及 MTE3 写回，output DB 槽可以由后续 FIXP 复用。循环开始前由 AIV 发放初始 credit。

这四条边组成两个 producer/consumer 闭环：

```text
AL1 闭环:
  A_CONSUMED -> AIV写AL1 -> A_READY -> AIC读AL1 -> A_CONSUMED

Output DB 闭环:
  FIX_OUTPUT_CONSUMED -> AIC FIXP写UB
  -> FIX_OUTPUT_READY -> AIV后处理/写GM -> FIX_OUTPUT_CONSUMED
```

## 8. 双缓冲的作用

主要数据区采用独立计数器管理的双缓冲或多槽缓冲：

- input UB：隐藏 GM 搬入与 Vector 消费；
- V output UB：隐藏 V 量化与 V Cache 写回；
- Q/K preprocess/output UB：在 Q/K 前处理、Cube 输出、FP8 staging 之间复用；
- A_rot AL1：让 AIV 写下一个 tile 与 AIC 消费当前 tile 错开；
- L0A/L0C：让 Cube 内部 MTE1、MMAD、FIXP 流水。

每类资源都有自己的 use counter。`TileParam` 的两槽 ring 只保存 current/previous tile 参数，不是数据 buffer id。

## 9. 性能观察

理想稳态下，希望：

```text
下一 tile 的 Vector 前处理
当前 tile 的 Cube
上一 tile 的 Vector 后处理
```

尽可能在不同计算资源上重叠。实际吞吐由最长阶段以及同步等待决定：

- 若 RMSNorm/RoPE/V quant 较重，AIC 会等待 `A_READY`；
- 若 Cube 较重，AIV 会等待 `FIX_OUTPUT_READY`；
- 若 Q/K 量化和 Cache scatter 较重，会延迟 `FIX_OUTPUT_CONSUMED`，阻塞 FIXP 复用；
- GM 带宽不足时，输入搬运和三路 Cache 写回可能成为瓶颈。

因此该算子的优化目标不是单纯减少某一段计算量，而是平衡前 Vector、Cube、后 Vector以及 MTE 搬运，使等待时间尽可能被其他 tile 的工作覆盖。

## 10. M-RoPE 场景差异

M-RoPE 保留相同的 `A_READY/A_CONSUMED` 主握手和一拍 rolling，但输出路径不同：

- Q 为 BF16 NoQuant，由 AIC Fixpipe 直接写 `q_out` GM；
- K 为 INT8，在 AIV 中量化并写 K Cache/K Scale Cache；
- `FIX_OUTPUT_READY` 发布 K rotation；
- 反向 credit 使用 `MROPE_K_CONSUMED`，而不是普通 RoPE 的 `FIX_OUTPUT_CONSUMED`。

## 11. Q/K/V 三路流水图

下图纵向分别展示 Q、K、V 三条数学数据流，横向表示时间和依赖顺序：

- Q：Vector RMSNorm/RoPE → Cube rotation matmul → Vector FP8 动态量化 → `q_out/q_scale`；
- K：Vector RMSNorm/RoPE → Cube rotation matmul → Vector FP8 动态量化 → K Cache/K Scale Cache；
- V：Vector scale/FP8 → V Cache，不进入 Cube。

图中的横向宽度用于表达先后和并行机会，不表示真实性能耗时比例。真实实现会把 Q/K 组织在共享的 Vec/Cube 流程和 buffer 中，并跨 tile 执行 rolling。

![QkvRmsNormRopeCacheWithKScale 流水分析](assets/qkv_rms_norm_rope_cache_with_k_scale_pipeline_analysis.svg)

## 12. VEC/CUBE 前后计算顺序图

下图使用三条逻辑泳道：`VEC-PRE`、`CUBE` 和 `VEC-POST`，横轴为时间。它重点表示 `Pre(tile i+1)`、`Cube(tile i)`、`Post(tile i-1)` 的跨 tile rolling，以及每个 tile 的 `A_READY/FIX_OUTPUT_READY` 数据依赖。`VEC-PRE` 与 `VEC-POST` 仍使用同一 AIV Vector 资源，实际按 `Pre(tile i) → Post(tile i-1)` 串行 issue。图中矩形宽度不代表真实性能耗时。

![QKV 在 VEC/CUBE 上的前后计算顺序](assets/qkv_compute_order_vec_cube_timeline.svg)

## 13. L1 内存划分和 double buffer

当前 kernel 在 `TPosition::TSCM` 中静态使用 `0–320 KiB`：前 `64 KiB` 为 rotation 预留区，其中当前只填充一份 `32 KiB` 的 `[128,128]` BF16 NZ 矩阵；其后的 `256 KiB` 为 A_rot pool，划分成两个 `128 KiB` logical buffer。

每个 A_rot logical buffer 保存当前 tile 的 Q/K BF16 NZ segment，AIV0 和 AIV1 分别写入前、后半 token rows。两个 logical buffer 通过 `aRotL1BufferUseId_ % 2` 交替使用，并以 `A_READY` 发布可读状态、以 `A_CONSUMED` 返还可写 credit。

![L1 内存划分与 A_rot 双缓冲](assets/qkv_l1_memory_double_buffer.svg)

## 14. UB 内存划分和多缓冲

普通 RoPE 使用 `248 KiB` UB：`80 KiB` input pool、`128 KiB` output DB pool和 `40 KiB` reserve。input、output、cos/sin、slot mapping 和 V-out 均为双缓冲；其中 output DB 的两个 slot 会在 Q/K 前处理、AIC Fixpipe 输出和 Q/K 后处理之间多角色复用，因此以 logical use counter 选择，而不是永久绑定某一种角色。

M-RoPE 使用独立的 `226 KiB` compact layout：input 和 raw cos/sin 为四缓冲；Q/K preprocess、K rotation、V-out、K quant 和 slot mapping 为双缓冲；`12 KiB` position window 是持久化单窗口缓存。

![UB 内存划分与多缓冲分析](assets/qkv_ub_memory_double_buffer.svg)
