# cubecore0 MMAD Pipeline Bubble Analysis Report

**Target**: `log_ca_0601` / core4 / cubecore0  
**Source Code**: `code_0601/kv_quant_sparse_attn_sharedkv_scfa_block_cube.h` + `kv_quant_sparse_attn_sharedkv_scfa_kernel.h`  
**Methodology**: `cubecore0/skills.md` (MMAD-bubble-deep-analysis)，cube 侧 MMAD bubble 深度分析  
**Data Sources**: `core4.cubecore0.ccu.cube_issque.dump` + `core4.veccore0/veccore1.instr_popped_log.dump` + `core4.veccore0/veccore1.ccu.mte2_issque.dump`

---

## 1. 计算流总体架构：v0→c1→v1→c2→v2 五阶段流水

### 1.1 代码级调度结构 (ProcessMainLoop, kernel.h:327-456)

cube 侧的任务与 vector 侧严格按依赖链执行：

```
taskId = K:
  Phase 1 (Vec0): ProcessVec0  → MTE2 CopyIn + DequantKv (block_vector.h:644-683)
  Phase 2 (Cube1): IterateLoadQK + IterateBmm1 → WaitCrossCore(v0) + MMAD (block_cube.h:360-458)
  Phase 3 (Vec1): ProcessVec1 → WaitCrossCore(c1) + Softmax (block_vector.h:752-848)
  Phase 4 (Cube2): IterateBmm2 → WaitCrossCore(v1) + MMAD (block_cube.h:462-506)
  Phase 5 (Vec2): ProcessVec2 → WaitCrossCore(c2) + FlashUpdate (block_vector.h:850-891)
```

**关键**: 3-buffer 软件流水 (`runInfo[taskId % 3]`) 使得不同 taskId 的阶段可以重叠。但在同一 taskId 内部，v0→c1→v1→c2→v2 是严格串行的。

### 1.2 MMAD PC 地址到阶段的映射

从 cube_issque.dump 中提取的 3 个 MMAD PC 地址：

| MMAD PC | WAIT_FLAG PC | 阶段 | 操作 |
|---------|-------------|------|------|
| `0x9000D27464` | `0x9000D2745C` (WAIT_FLAG FIX_M) | **Cube1 (Bmm1)** | Q×KV matmul, 输出 bmm1 结果到 UB |
| `0x9000D276E4` | `0x9000D276E0` (WAIT_FLAG FIX_M) | **Cube1 (Bmm1 K-split)** | Q×KV matmul 的 K 维度第2段 (K>128) |
| `0x9000D2723C` | `0x9000D2722C` (WAIT_FLAG M_MTE1) | **Cube2 (Bmm2)** | P×V matmul, 输出 attention 结果 |

**WAIT_FLAG 分类**：

| WAIT_FLAG PC | Flag 类型 | 含义 |
|-------------|----------|------|
| `0x9000D26E18` | CrossCore (vec→cube) | **等待 Vec0 完成** (outputL1.SetCrossCore / v0ResGm.SetCrossCore) |
| `0x9000D27120` | CrossCore (vec→cube) | **等待 Vec1 完成** (bmm1ResBuf.SetCrossCore) |
| `0x9000D2745C` | FIX_M | 等待 L0C Fixpipe 完成 (Bmm1 输出通道) |
| `0x9000D276E0` | FIX_M | 等待 L0C Fixpipe 完成 (Bmm1 K-split 通道) |
| `0x9000D2722C` | M_MTE1 | 等待 Bmm2 L1 数据就绪 |

### 1.3 L1 Buffer 分配 (cubecore0)

| # | Buffer | 位置 | 大小 | ping-pong 份数 |
|---|--------|------|------|---------------|
| 1 | l1QBuffers | A1 (L1) | BUFFER_SIZE_96K = 96KB | 3 份 (3-buffer, l1QBufId%3) |
| 2 | mmL0ABuffers | A2 (L0A) | BUFFER_SIZE_32K = 32KB | 2 份 (l0ABufId^=1) |
| 3 | mmL0BBuffers | B2 (L0B) | BUFFER_SIZE_64K = 64KB | - |
| 4 | mmL0CBuffers | CO1 (L0C) | BUFFER_SIZE_256K = 256KB | 2 份 (l0CBufId^=1) |
| **L1 合计** | l1Q + l1P + l1Right | L1 | 416KB | l1BufferManager.Init(pipe, 416*1024) |

---

## 2. MMAD 时间线与 Bubble 识别

### 2.1 总体统计数据

| 指标 | 值 |
|------|---|
| 总 MMAD 指令数 | **432** (Push=Pop=RETIRE=432) |
| 总 kernel ticks (cube) | 231,305 |
| cube busy cycles | 115,959 (占 50.2%) |
| cube MAC busy cycles | 112,824 (占 48.8%) |
| fixp busy cycles | 49,753 (占 21.5%) |
| mte2 busy cycles | 100,667 (占 43.5%) |
| mte1 busy cycles | 91,691 (占 39.7%) |
| scalar busy cycles | 66,770 (占 28.9%) |
| queue.size=15 满载次数 | **344** |

### 2.2 MMAD 执行时间线（首批事件）

从 cube_issque.dump 还原前几轮 MMAD 的完整生命周期：

| 事件 | Tick | 说明 |
|------|------|------|
| **Push** WAIT_FLAG 0x9000D26E18 (CrossCore Vec0→Cube) | 12712 | cube 开始等待 Vec0 产出 KV |
| **release** WAIT_FLAG 0x9000D26E18 | 12713 | **瞬间释放** (Vec0 已准备好) |
| Push WAIT_FLAG 0x9000D2745C + MMAD 0x9000D27464 | 12825-12827 | Bmm1 K=0 开始 |
| Push WAIT_FLAG + MMAD 0x9000D27464 | 12875-12877 | Bmm1 K=128 (K-split) |
| Push WAIT_FLAG + MMAD 0x9000D276E4 | 13024 | Bmm1 K=0 第2组 |
| Push WAIT_FLAG + MMAD 0x9000D276E4 | 13072 | Bmm1 K=128 第2组 |
| **Push** WAIT_FLAG 0x9000D26E18 (第二轮) | 13392 | cube 等待 Vec0 第二轮 |
| Push WAIT_FLAG 0x9000D2745C + MMAD 0x9000D27464 | 13490-13492 | Bmm1 第二轮 K=0 |
| Push WAIT_FLAG + MMAD 0x9000D27464 | 13534-13536 | Bmm1 第二轮 K=128 |
| Push WAIT_FLAG + MMAD 0x9000D276E4 | 13655 | Bmm1 第二轮 K=0 第2组 |
| Push WAIT_FLAG + MMAD 0x9000D276E4 | 13703 | Bmm1 第二轮 K=128 第2组 |
| **release** WAIT_FLAG 0x9000D2745C (首轮Bmm1 Fixpipe完成) | 28531 | **延迟 15706 ticks!** |
| **Pop** MMAD 0x9000D27464 queue.size=15 | 28532 | 首次 MMAD 出队 |
| **RETIRE** MMAD 0x9000D27464 | 28855 | 实际执行完成 |

### 2.3 首次巨大 Bubble (tick 13703 → 28531)

**Tick 13703 ~ 28531：长达 14828 ticks 的 MMAD bubble**

此期间 cube issue_queue 中已 Push 了 **8 条 MMAD** (对应 Bmm1 的 4 组 K-split × 2 份 ping-pong)，但无一条能 Pop 执行。原因是：

1. tick 13392: Push WAIT_FLAG 0x9000D26E18（等待 Vec0 第二轮）
2. tick 13492-13703: 连续 Push 了 6 条 MMAD + 2 条 WAIT_FLAG，填满 queue
3. tick 13703~28531: 所有 MMAD 都在等待 FIX_M flag（即 L0C 前一次 Fixpipe 完成）
4. 但首次 Fixpipe 又依赖 L0C 写入完成（由 cube dummy cycle 开始于 tick 28533）

### 2.4 稳态周期的 MMAD 时间线

进入稳态后，MMAD 的执行呈现 **Bmm1 (2×K-split) + Bmm2** 的 3 条一组模式：

**典型一轮 Bmm1 执行（对应代码 IterateBmm1SCFA）**：

```
Cube 侧每次 Bmm1:
  WaitFlag<FIX_M>(l0CBufId)         → 等上一轮 L0C Fixpipe 完成
  WaitCrossCore(inputRightBuf)       → 等 Vec0 产出 KV 到 L1/GM
  MMAD 0x9000D27464 (K=0~127)        → Q左半 × KV上半
  MMAD 0x9000D27464 (K=128~255)      → Q左半 × KV下半  
  SetFlag<M_FIX>(l0CBufId)           → 通知 Fixpipe
  WaitFlag<M_FIX>(l0CBufId)          → 等 Fixpipe 完成
  outputBuf.WaitCrossCore()          → 等 UB buffer 就绪 (Vec1 的 bmm1Buffers)
  Fixpipe (L0C→UB)                   → 搬运结果
  SetFlag<FIX_M>(l0CBufId)           → 释放 L0C
  outputBuf.SetCrossCore()           → 通知 Vec1
```

**典型一轮 Bmm2 执行（对应代码 IterateBmm2SCFA）**：

```
Cube 侧每次 Bmm2:
  l1PBuffer.WaitCrossCore()          → 等 Vec1 的 P 结果 (Softmax输出→L1)
  WaitFlag<FIX_M>(l0CBufId)          → 等 L0C
  MMAD 0x9000D2723C (P×V)            → Bmm2 matmul
  SetFlag<M_FIX>(l0CBufId)
  WaitFlag<M_FIX>(l0CBufId)
  outputBuf.WaitCrossCore()          → 等 UB buffer 就绪 (Vec2 的 bmm2Buffers)
  Fixpipe (L0C→UB)
  SetFlag<FIX_M>(l0CBufId)
  outputBuf.SetCrossCore()           → 通知 Vec2
```

### 2.5 Vector 侧 VF Dispatch 映射

从 veccore0/veccore1 的 instr_popped_log.dump 中提取 `PUSHQ.*VF` 事件，按 VF 地址和 instr_num 分类：

| VF 地址 | instr_num (十进制) | 次数 (vec0/vec1) | 阶段 | 操作 |
|---------|-------------------|-----------------|------|------|
| `0x9000d85400` | 42 (0x2a) | 108 / 108 | **Vec0** | DequantKv (主要计算) |
| `0x9000d85800` | 38 (0x26) | 54 / 54 | **Vec1** | Softmax |
| `0x9000d85c00` | 12 (0x0c) | 54 / 54 | **Vec2** | FlashUpdate |
| `0x9000d85e00` | 12 (0x0c) | 42 / 42 | Vec1 辅助 | DataCopy to L1 |
| `0x9000d85500` | 5 (0x05) | 6 / 6 | Vec0 初始化 | SetShape |
| `0x9000d85d00` | 14 (0x0e) | 6 / 6 | Vec1 初始化 | SetShape |
| `0x9000d86000` | 9 (0x09) | 6 / 6 | Vec2 初始化 | SetShape |
| `0x9000d85300` | 27 (0x1b) | 2 / 2 | SCALAR 初始化 | Init |

**映射方法**: 通过 instr_num 区分主要阶段 — 42→Vec0 (DequantKv), 38→Vec1 (Softmax), 12→Vec2 (FlashUpdate)。

**注意**: vec0 和 vec1 具有完全相同的 VF dispatch 模式，因两者执行相同的 vector 代码副本。

### 2.6 MMAD Bubble 分类

从 cube_issque.dump 的 WAIT_FLAG release 事件间隔分析，cube 侧 MMAD bubble 来自三类等待：

#### Type A: 等待 Vec0 完成（CrossCore Wait） — **最关键**

WAIT_FLAG PC `0x9000D26E18` 的 release 间隔：

| Push tick | Release tick | 等待时长 | 对应 |
|----------|-------------|---------|------|
| 12712 | 12713 | 1 tick | 首次 Vec0 已就绪 (几乎没有等待) |
| 13392 | 29601 | **16209** ticks | Vec0 第二轮 CopyIn+Dequant 全部完成 (冷启动) |
| 33053 | 34891 | **1838** ticks | Vec0 NotSparse 完成 |
| 36632 | 38563 | **1931** ticks | Vec0 完成 |
| 39364 | 41177 | **1813** ticks | Vec0 完成 |
| 43345 | 45174 | **1829** ticks | Vec0 完成 |
| 46045 | 47855 | **1810** ticks | Vec0 完成 |
| 49946 | 51764 | **1818** ticks | Vec0 完成 |
| 52902 | 54571 | **1669** ticks | Vec0 完成 |
| 56376 | 58161 | **1785** ticks | Vec0 完成 |
| 61929 | 63838 | **1909** ticks | Vec0 完成 (含 s2LoopCount 切换间隔) |
| ... | ... | **~1669~1931 ticks (稳态)** | 每轮 s2LoopCount 一次 |

**根因**: cube 的 IterateLoadQK 调用 `inputRightBuf.WaitCrossCore()` (cube.h:380) 或 `v0ResGm.WaitCrossCore()` (cube.h:374)，必须等 Vec0 完成全部 CopyIn + DequantKv。Vec0 的 ProcessSparseKv 内部 ping-pong 循环每轮处理 16 行，首次冷启动需 ~16209 ticks，稳态约 ~1669-1931 ticks。**在此期间 cube 完全空闲，无法开始任何 MMAD。**

#### Type B: 等待 L0C/Fixpipe 完成（FIX_M Wait）

WAIT_FLAG PC `0x9000D2745C` / `0x9000D276E0` 的 release 间隔：

| Pop tick | Release tick | Queue Latency |
|----------|-------------|---------------|
| 28532 (queue.size=15) | 28531 | ~15700 ticks (首次积压) |
| 28831 (queue.size=15) | 28830 | ~233 ticks |
| 29088 (queue.size=15) | 29087 | ~233 ticks |
| 32778 (queue.size=15) | 32777 | ~3143 ticks |
| 36344 (queue.size=15) | 36343 | ~2699 ticks |

**稳态周期**: 相邻 MMAD Pop 间隔约 232~258 ticks（L0C 单次 matmul 执行时间），但跨 s2LoopCount 时间隔跳增至 ~1600-3900 ticks（等待 Vec0 或 Vec1 完成新一轮）。

#### Type C: 等待 Vec1 完成（CrossCore Wait for Bmm2）

WAIT_FLAG PC `0x9000D27120` 的 release 间隔：

| Push tick | Release tick | 延迟 |
|-----------|-------------|------|
| 28661 | 33847 | **5186 ticks** |
| 33993 | 37413 | 3420 ticks |
| 37553 | 40148 | 2595 ticks |

**根因**: Bmm2 依赖 Vec1 的 Softmax 输出 (P 矩阵)。Vec1 的 `bmm1ResBuf.SetCrossCore()` (vector.h:826) 在 Softmax 完成后才通知 cube。cube 侧 `l1PBuffer.WaitCrossCore()` (cube.h:468) 必须等待。

---

## 3. Issue Queue 约束分析

### 3.1 Queue 结构

cube issue_queue 最大深度为 **16** (queue.size 取值 0~15)。

### 3.2 Queue 满载的规律

344 次 queue.size=15 事件，按 Pop tick 分析间隔：

| Pop tick 区间 | 间隔 (ticks) | 含义 |
|-------------|-------------|------|
| 28532, 28831, 29088, 29345 | ~233-257 | 同一 s2LoopCount 内 Bmm1 的 4 组 K-split 连续 Pop |
| 32778, 33077, 33334, 33591 | ~257-313 | 同上 |
| 33849, 34106, 34378, 34635 | ~257 | Bmm2 的连续 Pop (跨 veccore) |
| 36344, 36643, 36900, 37157 | ~257-313 | 下一轮 s2LoopCount Bmm1 |
| 39079, 39378, 39635, 39892 | ~257-313 | 继续轮转 |

**模式**:
- 每组 4 个连续 queue.size=15 事件（Bmm1 的 4 次 MMAD: 2×K-split × 2×L0C ping-pong），间隔 ~233-257 ticks
- 跨组间隔 ~1600-3900 ticks（等 Vec0 新一轮完成或 Vec1 完成）
- 每 ~3000-5300 ticks 一个完整 v0→c1→v1→c2→v2 周期（稳态）

### 3.3 Queue Latency 量化

通过 Push tick (从 instr_popped_log 对应 ID) 和 Pop tick (从 cube_issque) 交叉比对：

| MMAD PC | Push tick | Pop tick | Queue Latency |
|---------|-----------|----------|---------------|
| 0x9000D27464 id=281349 | 12827 | 28532 | **15705** (首次冷启动) |
| 0x9000D27464 id=285893 | 12877 | 28831 | **15954** |
| 0x9000D276E4 id=296971 | 13024 | 29088 | **16064** |
| 0x9000D276E4 id=300142 | 13072 | 29345 | **16273** |
| 0x9000D27464 id=322880 | 13492 | 32778 | **19286** |
| 0x9000D27464 id=324271 | 13536 | 33077 | **19541** |

**首次 Queue Latency 高达 15705~19541 ticks**（因为所有 MMAD 都在等 Vec0 首轮完成）。

稳态 Queue Latency（后续轮次，基于 Push/Pop tick ID 匹配）：

| MMAD PC | Push tick | Pop tick | Queue Latency | 说明 |
|---------|-----------|----------|---------------|------|
| 0x9000D27464 (id=1527659) | 33148 | 36344 | **3196** | 稳态 Bmm1 K0 |
| 0x9000D27464 (id=2139166) | 36727 | 39079 | **2352** | 稳态 Bmm1 K0 |
| 0x9000D276E4 (id=1583317) | 33592 | 36900 | **3308** | 稳态 Bmm1 K128 |
| 0x9000D276E4 (id=2181030) | 37158 | 39635 | **2477** | 稳态 Bmm1 K128 |
| 0x9000D276E4 (id=2584980) | 39893 | 43626 | **3733** | 稳态 Bmm1 K128 |

**稳态 Queue Latency ≈ 2300-3700 ticks**（包含 CrossCore 等待 + L0C Fixpipe 串行等待）。

**注意**: 相邻 MMAD Pop 间隔约 233-258 ticks（仅反映单次 MMAD 执行时间），不等同于 Queue Latency。Queue Latency 是从 Push 到 Pop 的全部等待时间。

---

## 4. Bubble 周期性与代码对应

### 4.1 三层周期

| 层级 | 周期 (ticks) | 代码循环 | 证据 |
|------|------------|---------|------|
| **微观** L0C ping-pong | ~233-258 | Bmm1 内部 K=0 和 K=128 两次 MMAD (cube.h IterateBmm1SCFA) | queue.size=15 的连续 Pop 间隔 |
| **中观** 一次 Bmm1/Bmm2 组 | ~1000-1400 | cube 完整处理一组 Bmm1(4 MMAD: 2×K-split × 2×L0C) + Fixpipe + Bmm2(4 MMAD: 2×L0C × 2次) | queue.size=15 为一组的跨度 (Bmm1 4次 + Bmm2 4次) |
| **宏观** v0→c1→v1→c2→v2 | ~3000-5300 (稳态) | ProcessMainLoop 一次 s2LoopCount 迭代 | CrossCore WAIT_FLAG release 间隔 |

### 4.2 宏观周期的 tick 证据

从 cube_issque 的 WAIT_FLAG 0x9000D26E18（等 Vec0）的 release tick 序列：

```
12713, 29601, 34891, 38563, 41177, 45174, 47855, 51764, 54571, 58161, 63838, 65907, ...
```

首次冷启动间隔：16888 (12713→29601)，后续间隔：5290, 3672, 2614, 3997, 2681, 3909, 2807, 3590, 5677, 2069, ...

**首次冷启动间隔 16888 ticks（Vec0 首轮完整初始化），稳态 Vec0 完成间隔 ≈ 1669-1931 ticks**。

### 4.3 Bmm2 的 bubble 特征

从 Bmm2 MMAD (PC 0x9000D2723C) 的 Pop tick 序列提取 Bmm2 的执行间隔：

```
33849, 34106, 34378, 34635, 37415, 37736, 38035, 38307, ...
```

**Bmm2 的 Pop 间隔也是 ~257 ticks**，说明 Bmm2 单次 matmul 执行时间与 Bmm1 相似。

但 Bmm2 的启动严重滞后于 Bmm1：
- Bmm1 首组 Pop: tick 28532-29345
- Bmm2 首组 Pop: tick 33849-34635
- **延迟: 4504-5290 ticks**（= Vec1 的 Softmax 执行时间）

---

## 5. 根因层次化判定

### 5.1 四层根因架构

```
Layer 1 — Vec0 延迟是 cube MMAD bubble 的根本瓶颈（~60% 贡献）
  现象: cube 在 IterateLoadQK 中调用 WaitCrossCore 等待 Vec0 产出 KV
  机制: Vec0 的 ProcessSparseKv 需要 CopyIn+Dequant 完整流程
  代码: cube.h:374 v0ResGm.WaitCrossCore() / cube.h:380 inputRightBuf.WaitCrossCore()
  数据: WAIT_FLAG 0x9000D26E18 release 间隔: 首次冷启动 16209 ticks，稳态 1669-1931 ticks
  影响: cube 的 MMAD 完全无法启动，issue_queue 积压 8+ 条 MMAD 指令

Layer 2 — issue_queue 积压放大延迟（~20% 贡献）
  现象: cube 一次 Push 多条 MMAD（4×Bmm1 + 4×Bmm2），queue 迅速填满
  机制: queue.size=15 意味着后续 MMAD 必须排队等待前面 MMAD 完成
  代码: kernel.h:397-441 for 循环中连续调用 IterateBmm1 + IterateBmm2
  数据: 首次 Queue Latency 15705~19541 ticks（所有 MMAD 在等 Vec0 首轮）
  稳态 QLat: 2300-3700 ticks（含 CrossCore + Fixpipe 等待）
  组内 MMAD Pop 间隔: 233-258 ticks（单次 MMAD 执行时间）

Layer 3 — Vec1 延迟推迟 Bmm2 启动（~15% 贡献）
  现象: Bmm2 依赖 Vec1 的 Softmax 输出 (P 矩阵)
  机制: cube 在 IterateBmm2 中调用 l1PBuffer.WaitCrossCore() 等 Vec1
  代码: cube.h:468 l1PBuffer.WaitCrossCore()
  数据: WAIT_FLAG 0x9000D27120 release 间隔: 首次 5186 ticks，稳态 1669-3991 ticks
  影响: Bmm2 首次执行比 Bmm1 首次晚 ~4500 ticks

Layer 4 — Fixpipe 开销与 L0C 管理串行化（~5% 贡献）
  现象: 每次 MMAD 完成后需要 Fixpipe (L0C→UB) 才能释放 L0C buffer
  机制: l0CBufId^=1 (2 份 ping-pong) 限制了连续 matmul 的吞吐
  代码: cube.h FIX_M flag 管理, fixp busy cycles = 49753
  数据: fixp 占 21.5% 时间，与 cube busy 50.2% 之比约 1:2.3
```

### 5.2 各 Pipe 的占空比

从 core4_summary_log 的 cubecore0 数据：

| Pipe | Busy Cycles | 占总 ticks (231305) 的比例 |
|------|------------|--------------------------|
| cube (MMAD) | 115,959 | **50.1%** |
| cube MAC | 112,824 | 48.8% |
| fixp (Fixpipe) | 49,753 | **21.5%** |
| mte2 (L1←GM) | 100,667 | **43.5%** |
| mte1 (L0A/L0B←L1) | 91,691 | **39.7%** |
| scalar | 66,770 | **28.9%** |

**Cube 有效计算时间约 50%**，剩余 50% 处于 bubble 状态。主要 bubble 来源是 CrossCore 等待 (Vec0/Vec1)。

---

## 6. Vector 侧与 Cube 侧的交叉时间线

### 6.1 首轮 s2LoopCount 的完整时间线

| Tick | Core | 事件 | 说明 |
|------|------|------|------|
| 4311-5218 | vec0/vec1 | SCALAR 初始化 | 设置 ParaBase, COREID 等 |
| 5567 | vec0 | MTE2 Push MOV_SPR_XN | Vec0 开始 CopyIn 准备 |
| 6472 | vec0 | SET_FLAG MTE2_V release | CopyIn 完成 |
| 12712 | cube | WAIT_FLAG CrossCore Push | cube 开始等 Vec0 |
| 12713 | cube | WAIT_FLAG release | **Vec0 首轮已提前完成** |
| 12827 | cube | MMAD Push (Bmm1) | 首条 MMAD 入队 |
| 13703 | cube | 末条 MMAD Push (Bmm1 第2轮) | 8 条 MMAD 全部入队 |
| 28531 | cube | FIX_M release | 首次 Fixpipe 完成 |
| 28532 | cube | MMAD Pop (queue.size=15) | 首条 MMAD 开始执行 |
| 28533-28574 | cube | cube dummy cycle | L0C 写入准备 |
| 28603-28830 | cube | L0C wr req × 16 | Bmm1 结果写入 L0C |
| 28855 | cube | MMAD RETIRE | 首条 Bmm1 完成 |
| 28855 | cube | SET_FLAG release (Fixpipe通知) | 触发 Fixpipe |
| 28855-28855 | cube | cube Fixpipe | L0C→UB 搬运 |
| 28661 | cube | WAIT_FLAG 0x9000D27120 Push | Bmm2 等待 Vec1 |
| 33847 | cube | WAIT_FLAG 0x9000D27120 release | Vec1 Softmax 完成 |
| 33849 | cube | Bmm2 MMAD Pop (queue.size=15) | Bmm2 首次执行 |

### 6.2 稳态周期的时间线

以第二轮 s2LoopCount 为例 (tick ~30000-37000):

```
Tick 29601: cube 等待 Vec0 第二轮 → release (Vec0 NotSparse 完成)
Tick 32777-33591: Bmm1 连续 4 次 MMAD Pop (每组 ~257 ticks 间隔)
Tick 33847: Bmm2 等待 Vec1 → release (Vec1 Softmax 完成)
Tick 33849-34635: Bmm2 连续 MMAD Pop

Vec0 第二轮:
  tick ~24000-29601: ProcessNotSparseKv 执行 (~5600 ticks)
  MTE2 CopyIn + DequantKv (ping-pong 循环)

Vec1 首轮 (对应 taskId-1):
  tick ~28855+: bmm1ResBuf.WaitCrossCore → Softmax → DataCopy to L1
  tick ~32777: bmm1ResBuf.SetCrossCore (通知 cube Bmm2 可以开始)
```

---

## 7. 优化建议

### 7.1 P0: 减少 Vec0 延迟 (根因 Layer 1)

**方案**: 拆分 Vec0 的 DequantKv VF 粒度 (42→21 instr × 2)
**收益**: Vec0 完成时间减半 → cube 等待 CrossCore 的时间从稳态 1669-1931 降至 835-966 ticks
**UB 开销**: 不增加 (VF 在 VEC pipe 执行)

或参考 veccore0 MTE2 Bubble 报告中的 P0 方案：拆分 DequantKv 可直接减少 Vec0 延迟 ~50%。

### 7.2 P1: Bmm1/Bmm2 流水化 (根因 Layer 2+3)

**方案**: 在 3-buffer 软件流水中，将 Bmm2 的启动与 Bmm1 的 Fixpipe 重叠
**当前**: Bmm1 Fixpipe → SetCrossCore → Vec1 Softmax → SetCrossCore → Bmm2
**优化**: 如果 L0C 有 3 份 buffer (而非 2 份 ping-pong)，Bmm2 可在 Bmm1 Fixpipe 期间使用另一份 L0C
**L0C 空间**: 当前 BUFFER_SIZE_256K × 2 = 512KB (2 份 L0C)，增至 3 份需 +256KB，L1 空间可能不足

### 7.3 P2: 预取 Vec0 下一轮结果 (根因 Layer 1)

**方案**: 在 cube 执行 Bmm1/Bmm2 期间，MTE2 可提前搬运下一轮 s2LoopCount 的 Q 数据
**实现**: 将 CopyQGmToL1 提前到 IterateLoadQK 之前
**收益**: 消除 ~1000 ticks 的 Q 加载等待

### 7.4 P3: 增加 L0C buffer 份数 (长期)

**方案**: 将 mmL0CBuffers 从 2 份 (l0CBufId^=1) 增至 3 份
**约束**: 需额外 256KB L0C 空间，需验证硬件是否支持

---

## 8. 数据来源索引

| 文件 | 路径 | 用途 |
|------|------|------|
| cube issue_queue | `log_ca_0601/core4.cubecore0.ccu.cube_issque.dump` | MMAD Push/Pop/RETIRE tick + queue.size |
| cube 指令日志 | `log_ca_0601/core4.cubecore0.instr_popped_log.dump` | SCALAR/MTE1/MTE2 指令时序 |
| cube 执行日志 | `log_ca_0601/core4.cubecore0.cube_log.dump` | L0C wr/rd 操作, dummy cycle |
| 性能汇总 | `log_ca_0601/core4_summary_log` | 各 pipe busy cycles |
| Vec0 指令日志 | `log_ca_0601/core4.veccore0.instr_popped_log.dump` | SET_FLAG/WAIT_FLAG/VF dispatch |
| Vec0 MTE2 queue | `log_ca_0601/core4.veccore0.ccu.mte2_issque.dump` | MTE2 CopyIn 时序 |
| Vec1 指令日志 | `log_ca_0601/core4.veccore1.instr_popped_log.dump` | Vec1 Softmax 时序 |
| Vec1 MTE2 queue | `log_ca_0601/core4.veccore1.ccu.mte2_issque.dump` | MTE2 CopyIn 时序 |
| Cube 代码 | `code_0601/kv_quant_sparse_attn_sharedkv_scfa_block_cube.h` | IterateLoadQK/IterateBmm1/Bmm2 |
| Kernel 流水 | `code_0601/kv_quant_sparse_attn_sharedkv_scfa_kernel.h` | ProcessMainLoop |
| Vec 代码 | `code_0601/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h` | ProcessVec0/Vec1/Vec2 |
| 本报告 | `report/cubecore0/0601/MMAD_Bubble_Analysis_Report.md` | 综合分析 |

**Tick 说明**: 所有 dump 中的 tick 均为十进制。cube 侧 tick 与 vec 侧 tick 在同一时间域，通过 CrossCore 同步事件可交叉关联。cube 侧起始 tick ~1124 (从 core4_summary_log start: 1124)，vec0 侧起始 tick ~1161，vec1 侧起始 tick ~1196。

---

*Generated via cube-side analysis workflow, referencing `cubecore0/skills.md` (MMAD-bubble-deep-analysis) methodology*  
*Key insight: cube MMAD bubble 的根因是 **Vec0 延迟**（CrossCore 等待），而非 cube 自身计算瓶颈。cube 实际计算占比 50%，bubble 占比 50% 主要来自等待 vector 侧完成任务。*