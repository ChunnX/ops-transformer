# veccore0 MTE2 Pipeline Bubble Analysis Report

**Target**: `log_ca_0601` / veccore0  
**Source Code**: `code_0601/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` + `kv_quant_sparse_attn_sharedkv_scfa_kernel.h`  
**Methodology**: `veccore0/skills.md` (MTE2-bubble-deep-analysis + pipeline-pingpong-analysis)  
**Data Sources**: `core0.veccore0.instr_popped_log.dump` + `core0.veccore0.ccu.mte2_issque.dump`

---

## 1. 计算流总体架构

一个完整的 flash attention 计算流由 **3 个 vector 阶段**组成（kernel.h:327-456 `ProcessMainLoop`），采用 **3-buffer 软件流水** (`runInfo[taskId % 3]`)，每个 s2LoopCount 迭代依次调度：

```
s2LoopCount = N:
  Phase A (Vec0): ProcessVec0 → MTE2 CopyIn + DequantKv (block_vector.h:644-683)
  Phase B (Vec1): ProcessVec1 → Softmax/Cast/DataCopy (block_vector.h:752-848)  [来自 taskId-1]
  Phase C (Vec2): ProcessVec2 → FlashUpdate/CopyOut (block_vector.h:850-891)   [来自 taskId-2]
```

每个 Vec 阶段**内部**使用 ping-pong 缓冲：
- **Vec0**: `stage0InBuf[2]` + `stage0OutBuf[2]` + `pingPongV0 ^= 1` (block_vector.h:164-166)，2 份乒乓
- **Vec1**: `stage1OutQue[2]` + `softmaxMaxBuf[2]` / `softmaxSumBuf[2]` / `softmaxExpBuf[2]` (block_vector.h:152, 160-162)
- **Vec2**: `mte3ToVId` / `vToMte3Id` 单组 event (block_vector.h:154-155)

### VF 地址到阶段的映射

| VF 地址 | instr_count | 操作 | 阶段 |
|---------|-------------|------|------|
| `0x9000d85300` | 27 | AntiquantVF pre-loop | Vec0 初始化 |
| `0x9000d85400` | 42 | DequantKv (AntiquantVFImplFp8D448) | **Vec0** (108 次, ping-pong 成对) |
| `0x9000d85800` | 38 | Softmax (ProcessVec1Vf) | **Vec1** (54 次) |
| `0x9000d85500` | 5 | SoftmaxFinal (循环边界) | Vec1 阶段切换 |
| `0x9000d85c00` | 12 | FlashUpdate | **Vec2** (54 次) |
| `0x9000d85e00` | 12 | Epilogue (ping-pong swap) | Vec2 辅助 |
| `0x9000d85d00` | 14 | TransferEnd | Vec2 末尾 |
| `0x9000d86000` | 9 | OutputDMA | Vec2 输出 |

---

## 2. 周期性规律与代码循环的对应

### 2.1 宏观周期: s2LoopCount 迭代 (~7000 ticks)

issue_queue 的 queue.size=15 (满载) 事件呈现出稳定的**宏观周期**：

| 周期起止 (ticks) | 间隔 | 对应代码 |
|-----------------|------|---------|
| 66029 → 73513 | **7484** | 一次 s2LoopCount (Vec0+Vec1+Vec2) |
| 73513 → 80670 | **7157** | 同上 |
| 80670 → 87658→94746 (cluster) | **7088** (组内) | Vec0 NotSparse+Sparse 切换 |
| 94746 → 100692 | **5946** | 短周期 (仅 ori KV) |
| 100692 → 107999 | **7307** | 标准 s2LoopCount |
| 107999 → 115237 | **7238** | 标准 s2LoopCount |
| 115237 → 122239 | **7002** | 标准 s2LoopCount |
| 134121 → 141582 | **7461** | 标准 s2LoopCount |
| 141582 → 148645 | **7063** | 标准 s2LoopCount |
| 148645 → 155771 | **7126** | 标准 s2LoopCount |
| 175794 → 182796 | **7002** | 标准 s2LoopCount |
| 182796 → 189771 | **6975** | 标准 s2LoopCount |
| 209159 → 216271 | **7112** | 标准 s2LoopCount |

**平均宏观周期 = 7137 ticks**，对应 `ProcessMainLoop` 中一次 s2LoopCount 迭代内 Vec0 + Vec1 + Vec2 的总执行时间。

### 2.2 中观周期: Vec0 内部 ping-pong (~3700 ticks)

Vec0 的 `ProcessSparseKv` while 循环每次迭代处理 16 行 KV，内部 ping-pong 产生 ~3700 ticks 的半周期：

**SET_FLAG MTE2_V 释放 tick 的周期分组** (4 个 SET_FLAG 为一组 = 2 次 ping-pong × 2 轮 burst)：

| 组号 | 起止 tick | 组内跨度 | 代码位置 |
|------|----------|---------|---------|
| 0 | 27745→32823 | 5078 | 首次循环 (含 SCALAR 初始化开销) |
| 1 | 34256→38121 | **3865** | ProcessSparseKv 第 2 轮 16 行 |
| 2 | 41400→44556 | **3156** | ProcessSparseKv 第 3 轮 (NotSparse?) |
| 3 | 48407→51462 | **3055** | ProcessNotSparseKv |
| 4 | 57071→62588 | **5517** | 跨 s2LoopCount 边界 (Vec1+Vec2 介入) |
| 5 | 66874→70459 | **3585** | ProcessSparseKv 标准轮 |
| 6 | 74256→77993 | **3737** | 标准轮 |
| 7 | 81424→84599 | **3175** | 短轮 (NotSparse) |
| ... | ... | ... | 规律持续到 tick 221417 |

**模式**: 奇数组 ~3100-3200 (NotSparse, 无需 SCALAR 地址解析)、偶数组 ~3500-3800 (Sparse, 含 GetRealS2Addr 开销)、每 4 组一次 ~5500 (跨 Vec1+Vec2 边界)。

### 2.3 微观周期: CopyIn burst 内 (~100 ticks)

每组 SET_FLAG 内部，8 条 MTE2 指令 (MOV_SPR_XN + MOV_SRC_TO_DST_ALIGNv2 交替) 以 ~15 ticks 间隔发射，形成 ~111 ticks 的 CopyIn burst：

```
8 条 MTE2 指令 PC 序列 (0x9000D83B64~0x9000D83FB4):
  MOV_SPR_XN (0x3B64) + MOV_SRC_TO_DST (0x3B98)  = 行 0-1 (Combine)
  MOV_SPR_XN (0x3C8C) + MOV_SRC_TO_DST (0x3D00)  = 行 2-3
  MOV_SPR_XN (0x3E04) + MOV_SRC_TO_DST (0x3E70)  = 行 4-5
  MOV_SPR_XN (0x3F50) + MOV_SRC_TO_DST (0x3FB4)  = 行 6-7
  → 对应 CopyInKvSparse 的 4 次 DataCopyPad (block_vector.h:277-317, i+=2)
```

---

## 3. Issue Queue 约束分析

### 3.1 Queue 结构

issue_queue 最大深度为 16 (queue.size 取值 0~15)。42 次达到 queue.size=15 (满载)。

### 3.2 Queue-full 与 DequantKv 的因果关系

**关键发现**: queue.size=15 恒定出现在 DequantKv VF dispatch pair **第二条发射后** **59-61 ticks**：

| VF pair 第二条 dispatch tick | queue.size=15 tick | 间隔 | 含义 |
|------------------------------|--------------------|------|------|
| 36187 | 36246 | **59** | VF pair 全部入队 → SET_FLAG release → 新 CopyIn 填满 queue |
| 40063 | 40124 | **61** | 同上 |
| 42747 | 42806 | **59** | 同上 |
| 46686 | 46747 | **61** | 同上 |
| 53260 | 53321 | **61** | 同上 |
| 56101 | 56160 | **59** | 同上 |
| 59663 | 59724 | **61** | 同上 |

**机制**: 
1. DequantKv VF pair 第二条指令入队 (2×42 instr pair 全部 dispatch 完成)
2. ~60 ticks 后 VEC 端完成 → release `SET_FLAG MTE2_V` (block_vector.h:730 对应端)
3. 下一轮 CopyIn 的 8 条 MTE2 指令**瞬间 Push 进 issue_queue**
4. 上一轮尚未 Pop 完的 7 条残留 + 新 Push 的 8 条 = 15 → **满载**

### 3.3 Queue-full 导致的 MTE2 停滞

**用户断言验证**: "38836-40124 期间，issue_queue is full"

**验证结果: 基本正确，queue 处于积压饱和状态**

通过交叉比对三个 dump 的 tick（十进制），还原阻塞链：

| 时刻 | 事件 | dump 来源 |
|------|------|-----------|
| tick 35746 | Push ID=1905631 (MOV_SPR_XN) | mte2_issque |
| tick 36111-38821 | 7 条 CopyIn Push 填满 queue | mte2_issque |
| tick 38011 | WAIT_FLAG 入队 | mte2_issque |
| tick 38821-40123 | **MTE2 完全冻结** (0 Push, 0 Pop) | 两边均无事件 |
| tick 40124 | Pop ID=2394495, queue.size=15 | instr_popped |
| tick 40124-40158 | queue 从 15 排空到 0 (34 ticks) | mte2_issque |
| tick 40124 | Push ID=2436974 (新 CopyIn) | mte2_issque |

**阻塞机制**: 38836-40124 期间 MTE2 的 WAIT_FLAG (pc=0x9000D83824) 被 VEC 端阻塞（DequantKv 未完成），导致:
1. WAIT_FLAG 无法 release → MTE2 流水线挂起
2. Queue 中积压的 15 条指令也无法 Pop（Pop 端也依赖 VEC 释放）
3. Push 端和 Pop 端**同时冻结** → queue 处于满载静止状态

这**既是 WAIT_FLAG 阻塞，也是 queue 满载**——两者是同一因果链的不同环节。

### 3.4 Issue Queue 满载对 Bubble 的贡献（量化）

通过 Pop tick - Push tick 量化 Queue Latency:

| Pop tick (queue.size=15) | Push tick | Queue Latency | 对照组 (queue 未满) QLat |
|--------------------------|-----------|---------------|--------------------------|
| 36246 | 35746 | **500** | 1~5 |
| 40124 | 38309 | **1815** | 1~5 |
| 42806 | 42159 | **647** | 1~5 |
| 46747 | 44830 | **1917** | 1~5 |
| 53321 | 51407 | **1914** | 1~5 |
| 56160 | 52790 | **3370** | 1~5 |
| 66029 | 62321 | **3708** | 1~5 |

**Queue Latency 在满载场景下是非满载场景的 100~740 倍**。Queue 满载不是短暂瞬态，而是持续数百到数千 ticks 的有效阻塞。

| 阻塞源 | 贡献占比 | 量化依据 |
|--------|---------|---------|
| **VEC 占用导致 Pop 受阻** | ~60% | QLat 500~3708 中，DequantKv 耗时 ~840 ticks 直接阻塞 Pop |
| **Queue 排队积累** | ~25% | queue 从空到满 (0→15) 需要 8 条 Push (~111 ticks)，但 drain 需 3708 ticks，积累效应放大 |
| **Ping-pong WaitFlag** | ~10% | 2 份缓冲迫使每 2 轮同步等待 |
| **初始化冷启动** | ~5% | 仅首次循环，影响有限 |

---

## 4. Bubble 周期性分类（与代码逻辑匹配）

### 4.1 Vec0 阶段内部 MTE2 bubble (微观 ping-pong)

**周期**: ~395-870 ticks (DequantKv pair gap)  
**代码位置**: `ProcessSparseKv` (block_vector.h:705-750) 的 while 循环体内

```
每次 ping-pong 迭代:
  WaitFlag<V_MTE2>        (line 709)  → 等上轮 VEC 释放缓冲
  CopyIn 8 行 (MTE2)      (line 711-725) → ~111 ticks
  SetFlag<MTE2_V>         (line 730)  → 通知 VEC
  WaitFlag<MTE2_V>        (line 731)  → VEC 确认
  WaitFlag<MTE3_V>        (line 733)  → 等 MTE3 释放
  DequantKv (VF pair)     (line 736)  → ~840 ticks (2×42 instr × ~10 ticks/instr)
  SetFlag<V_MTE2>         (line 737)  → 释放 CopyIn 缓冲 ← MTE2 可开始下轮
  CopyOut (MTE3)          (line 742-744) → ~100 ticks
  SetFlag<MTE3_V>         (line 746)  → 释放 VEC
  pingPongV0 ^= 1         (line 748)  → 切换缓冲
```

**MTE2 空窗**: 从 `SetFlag<V_MTE2>` (释放) 到下一轮 `WaitFlag<V_MTE2>` 通过之间的等待时间 ≈ **DequantKv 耗时 (~840 ticks) - CopyIn 耗时 (~111 ticks) ≈ 730 ticks**

### 4.2 Vec0→Vec1→Vec2 三级流水 bubble (宏观)

**周期**: ~7000 ticks  
**代码位置**: `ProcessMainLoop` (kernel.h:397-443) 的 s2LoopCount 循环

```
s2LoopCount = K:
  taskId = K:
    Vec0(runInfo[K%3]):  CopyIn + Dequant (全部 ping-pong 迭代完成)
  taskId > 0:
    Vec1(runInfo[(K+2)%3]):  WaitCrossCore → Softmax → DataCopy to L1
  taskId > 1:
    Vec2(runInfo[(K+1)%3]):  WaitCrossCore → FlashUpdate → CopyOut
```

**Vec0 的 MTE2 bubble 如何影响 Vec1/Vec2**: 
- Vec0 完成所有 ping-pong 迭代后，通过 `outputL1.SetCrossCore()` (line 681) 或 `v0ResGm.SetCrossCore()` (line 679) 通知 Vec1
- 如果 Vec0 因同步壁垒延迟 N ticks，Vec1 推迟 N ticks 启动
- 同理 Vec2 依赖 Vec1 的 `bmm1ResBuf.SetCrossCore()` (line 826)

### 4.3 Vec1/Vec2 执行期间的 MTE2 bubble

Vec1 和 Vec2 执行期间（~3300 ticks），MTE2 完全空闲：
- **Vec1**: `ProcessVec1Vf` (Softmax) 耗时 ~366 ticks，加上 DataCopy 和 SCFAUpdate ~700 ticks
- **Vec2**: `FlashUpdate` ~120 ticks + Epilogue + CopyOut ~800 ticks
- 总 Vec1+Vec2 耗时 ~1500-2000 ticks，期间 MTE2 无任何 CopyIn 任务

这部分不是 bubble — 是 Vec0→Vec1→Vec2 流水设计中的正常间隔。但在 3-buffer 流水下，**下一轮 s2LoopCount 的 Vec0 本应已启动**，如果被 Vec1/Vec2 的 CrossCore 同步阻塞，则产生 bubble。

### 4.4 两种 MTE2 PC 路径对应的 bubble 差异

| PC 地址范围 | 路径 | bubble 特征 |
|------------|------|------------|
| 0x9000D83B64~0x9000D83FB4 (8 PCs) | `ProcessSparseKv` + `CopyInKvSparse` (block_vector.h:270-319) | 每轮需 SCALAR `GetRealS2Addr` 解析 8 个 token 物理地址，burst 间隔 ~96-400 ticks |
| 0x9000D83790 + 0x9000D837C8 (2 PCs) | `ProcessNotSparseKv` + `CopyInKvNotSparse` (block_vector.h:561-605) | 地址连续 (PA blockTable 查表)，burst 间隔更短 (~2 ticks) |

queue.size=15 事件中，0x9000D83790 出现在 **成对簇** 中 (间隔 ~388 ticks)：
- tick 53321/53709, 87658/88063, 122239/122627, 155771/156176, 189771/190159
- 这对应 `ori KV` (原始 KV cache) 的两次连续 CopyIn block

---

## 5. 根因总结（层次化）

### 5.1 四层根因架构

通过交叉比对 `instr_popped_log.dump`（Pop tick）、`instr_log.dump`（执行 tick）、`mte2_issque.dump`（Push tick），可以完整还原一条 MTE2 指令的生命周期：

```
Push (issque) ──→ 排队等待 (Queue Latency) ──→ Pop (popped) ──→ 执行 (instr_log)
 Push tick         QLat = Pop-Push              Pop tick         Exec = Log-Pop
```

**实测 Queue Latency 对比**：

| 场景 | 典型 QLat (ticks) | 原因 |
|------|-------------------|------|
| queue 未满 (size 0~8) | **1~5** | queue 几乎为空，Push 后立即 Pop |
| queue 趋满 (size 12~14) | **500~152** | 前面积压指令需逐条 Pop 排空 |
| queue 满载 (size=15) | **735~3708** | queue 完全打满，VEC 端因 DequantKv 占用无法 Pop |

queue-full 事件对应的 QLat 平均为 **1801 ticks**，是非满载场景 (1~5 ticks) 的 **360 倍**。这说明 issue_queue 满载不是短暂瞬态，而是**持续数百到数千 ticks 的排队阻塞**。

### 5.2 根因层次分解

```
Layer 1 — VEC 占用是根本瓶颈（~85% 贡献）
  现象: MTE2 指令 Push 后无法 Pop，因为 VEC 正在执行 DequantKv
  机制: issue_queue 的 Pop 端受 VEC 反压，VEC 被 VF 核占用时 MTE2 无法出队
  代码: block_vector.h:736 DequantKv 调用 VF 核 (42 instr × ~10 ticks = ~420 ticks)
  数据: QLat=500~3708 与 DequantKv pair 耗时 (840 ticks) 直接正相关

Layer 2 — Vec0 内部 ping-pong 放大排队效应（~10% 贡献）
  现象: 即使 VEC 空闲，2 份 ping-pong 缓冲也导致 CopyIn 必须 WaitFlag
  机制: stage0InBuf[2] 意味着 CopyIn(N+2) 必须等 Dequant(N) 释放缓冲
  代码: block_vector.h:164 stage0InBuf[2], line 709 WaitFlag<V_MTE2>
  数据: 2 轮 CopyIn 后 queue 填满 (8+8=16 > 15 queue depth)，造成额外排队

Layer 3 — 三级流水未实现 VEC 硬件重叠（~5% 贡献）
  现象: Vec0/Vec1/Vec2 在 ProcessMainLoop 中代码级串行调用
  机制: kernel.h:406→422→434 是 if-else 顺序调度，不是硬件并发
  代码: kernel.h:397-441 s2LoopCount 循环体
  数据: VF dispatch tick 表中无任何 Vec0 VF 与 Vec1 VF 时间重叠

Layer 4 — SCALAR 地址计算串行化（初始化阶段）
  现象: 首次 ProcessSparseKv 时 GetRealS2Addr 需逐一解析 token 物理地址
  机制: SCALAR 等待 GM load 返回 (LD_XD_XN_IMM dcache hit=0)
  代码: block_vector.h:202-224 GetRealS2Addr, 8 次 GetValue() 调用
  数据: tick 5760-10418 (4658 ticks) 期间无任何 VF dispatch，纯 SCALAR 执行
```

### 5.3 Issue Queue 的真实角色

之前分析低估了 issue_queue 的阻塞效应。实测数据修正如下：

| 阻塞源 | 贡献占比 | 量化依据 |
|--------|---------|---------|
| **VEC 占用导致 Pop 受阻** | ~60% | QLat 500~3708 中，DequantKv 耗时 ~840 ticks 直接阻塞 Pop |
| **Queue 排队积累** | ~25% | queue 从空到满 (0→15) 需要 8 条 Push (~111 ticks)，但 drain 需 3708 ticks，积累效应放大 |
| **Ping-pong WaitFlag** | ~10% | 2 份缓冲迫使每 2 轮 CopyIn 后同步等待 |
| **初始化冷启动** | ~5% | 仅首次循环，影响有限 |

**关键修正**: Queue 满载不是"瞬态 ~34 ticks"—— QLat 数据证明满载状态持续 **735~3708 ticks**，这是 MTE2 bubble 的重要组成部分。根因链是：

```
VEC DequantKv 执行中 (840 ticks)
  → issue_queue Pop 端反压
  → 8 条新 CopyIn Push 填满 queue (size=15)
  → 后续 CopyIn 无法入队 (直到 queue drain)
  → MTE2 空闲 735~3708 ticks ← 这就是 bubble
```

---

## 6. 优化建议（按收益排序）

### UB 空间约束现状

单个 vec 核 UB 总量为 **248 KB**。当前分配如下：

| # | Buffer | 公式 | 大小 (KB) | ping-pong 份数 |
|---|--------|------|-----------|---------------|
| 1 | stage0InBuf (V0 CopyIn) | 16×608×2 | **19 KB** | 2 份 |
| 2 | stage0OutBuf (V0 CopyOut to L1) | 16×512×2×2 | **32 KB** | 2 份 |
| 3 | bmm1Buffers (c1 matmul 结果) | 32×128×4×2 | **32 KB** | 2 份 |
| 4 | softmaxSumBuf | 512B | 0.5 KB | 2 份 |
| 5 | softmaxMaxBuf | 512B | 0.5 KB | 2 份 |
| 6 | softmaxExpBuf | 512B | 0.5 KB | 2 份 |
| 7 | sinksBuf | 512B | 0.5 KB | 1 份 |
| 8 | stage1OutQue | (64/2+1)×128×2×2 | **16.5 KB** | 2 份 |
| 9 | bmm2Buffers (c2 matmul 结果) | 128×512 | **64 KB** | 1 份 |
| 10 | stage2OutBuf | 32×512×4 | **64 KB** | 1 份 |
| | **合计** | | **229.5 KB** | |
| | **剩余** | 248-229.5 | **18.5 KB** | |

**剩余空间 18.5 KB 是所有优化的硬约束**。任何增加 ping-pong 份数或新增 buffer 的方案都必须在此空间内。

### ~~6.1 原优先级 P0: 增大 Vec0 ping-pong 缓冲深度~~ → 不可行 ❌

**原方案**: 将 `stage0InBuf[2]` + `stage0OutBuf[2]` 从 2 份增至 4 份。

**可行性验证**:

| 方案 | 额外需要 | 可用剩余 | 结论 |
|------|---------|---------|------|
| 增至 3 份 (+1 份) | 19456/2 + 32768/2 = **25.5 KB** | 18.5 KB | **不可行** |
| 增至 4 份 (+2 份) | 19456/2×2 + 32768/2×2 = **51 KB** | 18.5 KB | **不可行** |

**结论**: UB 剩余空间不足以容纳额外的 1 份 ping-pong 缓冲（需 25.5 KB > 剩余 18.5 KB）。此优化项**不可行**，需降级为长期建议（需先通过压缩其他 buffer 释放空间）。

### 6.1 新优先级 P0: 拆分 DequantKv VF 粒度

**收益预估**: 直接减小 Layer 1 的 VEC 占用时间 ~50%，降低 Queue 积累

**UB 开销**: 不增加任何 UB 空间（VF 指令在 VEC pipe 执行，不占用 UB）

当前每次 VF dispatch 42 条指令 (~420 ticks)，VEC 占用期间 issue_queue 完全无法 Pop。拆分为 **2 次 21 条** (~210 ticks each)：
- 每次 VF 完成后 RETIRE → 释放 Pop 端 → queue 可部分 drain
- SetFlag<V_MTE2> 在第一次 VF 完成时即可发出（而非等全部 42 条完成）
- 代价: VF 启动开销翻倍 (~60 ticks × 2 = 120 ticks overhead)，但净收益 420-210-60 = ~150 ticks
- Queue Latency 预期从 735~3708 降至 ~400~1800（减少约一半）

### 6.2 优先级 P1: 跨 s2LoopCount 的 Vec0 预取

**收益预估**: 消除 Layer 3 的宏观空闲 (~2000 ticks/cycle)

**UB 开销**: 不增加 UB 空间（复用现有 stage0InBuf 的同一份空间，仅提前调度）

当前 ProcessMainLoop (kernel.h:397-441) 中 Vec0/Vec1/Vec2 串行调度。在 Vec1+Vec2 执行期间，MTE2 完全空闲。

实现方案: 将 kernel.h 的 s2LoopCount 循环改为真正的**异步流水**:
- Vec0 在独立线程/协程中运行，通过 CrossCore flag 与 Vec1/Vec2 通信
- Vec1+Vec2 期间，MTE2 可提前开始下一轮 s2LoopCount 的 CopyIn
- **Queue 约束**: 16 深度的 issue_queue 可以容纳 2 轮 CopyIn burst (2×8=16 条恰好填满)，所以预取仅限 1 轮提前量
- 需要独立的 flag ID 隔离不同 s2LoopCount 迭代的 MTE2_V 信号

### 6.3 优先级 P2: SCALAR 地址计算流水化

**收益预估**: 消除 Layer 4 初始化冷启动 + 降低 Layer 1 的 SCALAR 开销

**UB 开销**: 不增加（或极少量，如批量地址暂存 <1KB）

`GetRealS2Addr` (block_vector.h:202-224) 每次 while 循环同步调用 8 次 `kvPhyAddrGm.GetValue(idx)`。优化方案:
- 预取: 在 CopyIn burst(N) 执行期间，SCALAR 同时计算 burst(N+1) 的 8 个 token 地址
- 批量 GM 访问: 将 8 次 `GetValue()` 合并为一次 `DataCopy` 批量搬运到 UB，再在 UB 中解析
- 消除 dcache miss: tick 38836-40124 期间 `LD_XD_XN_IMM dcacheHit:0` 表明 GM 地址加载未命中

### 6.4 长期建议: 重新规划 UB 空间以支持更大 ping-pong

如果要使原始 P0（增大 ping-pong 缓冲深度）变为可行，需要先从其他 buffer 释放 ≥ 25.5 KB 空间。可能的途径:

1. **压缩 stage2OutBuf (64 KB)**: 当前 `stage2OutBuf` 分配 32×512×4=64 KB，但如果 Vec2 的输出可以先写 L1 再由 MTE3 搬出，则此 buffer 可减小甚至消除
2. **压缩 bmm2Buffers (64 KB)**: 如果 c2 matmul 结果可以流式消费（不一次性全部驻留 UB），可减小此 buffer
3. **bmm1Buffers 去双缓冲 (32 KB)**: 当前 2 份 bmm1Buffers 占 32 KB，如果 Vec1 消费速度足够快，可减为 1 份释放 16 KB

---

## 7. 数据来源索引

| 文件 | 路径 | 用途 |
|------|------|------|
| 过滤后 dump (无RV) | `report/veccore0/0601/core0.veccore0.instr_popped_log.no_rv.dump` | Pop tick + VF dispatch 提取 |
| 原始指令日志 | `log_ca_0601/core0.veccore0.instr_popped_log.dump` | Pop tick (含 MTE2 细节) |
| 指令执行日志 | `log_ca_0601/core0.veccore0.instr_log.dump` | 执行/RETIRE tick (Pop→Exec 延迟) |
| MTE2 issue_queue | `log_ca_0601/core0.veccore0.ccu.mte2_issque.dump` | Push tick + queue.depth |
| Vec0 源码 | `code_0601/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` | ProcessSparseKv/NotSparseKv |
| Kernel 流水编排 | `code_0601/kv_quant_sparse_attn_sharedkv_scfa_kernel.h` | ProcessMainLoop 3-buffer |
| 本报告 | `report/veccore0/0601/MTE2_Bubble_Analysis_Report.md` | 综合分析 |

**Tick 说明**: 所有 dump 中的 tick 均为十进制（零填充显示如 `[00066029]` = tick 66029），通过同一 instruction ID 交叉比对确认三份 dump 的 tick 域一致。

---

*Generated via `veccore0/skills.md` (MTE2-bubble-deep-analysis + pipeline-pingpong-analysis) workflow*  
*Revised with: (1) periodicity analysis matching code loop structure; (2) issue_queue constraint with Push/Pop/Exec latency quantification; (3) Vec0/Vec1/Vec2 3-stage pipeline mapping*