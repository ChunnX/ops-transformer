# PoolKeyIndexer Kernel 全量性能优化空间检视报告

> 检视日期：2026-09-02
> 检视范围：`op_kernel`（common / pool_key_indexer.cpp / arch22 / arch35 / arch35-vf）全文件
> 检视维度：访存（MTE2/MTE3/L1 L2）、算力（Cube/Vector/标量）、TopK 实现、流水/sync、多核切分/负载均衡
> 关联报告：`docs/code_review_scalar_loop_report.md`（标量循环专项，本报告交叉引用其结论）

---

## 总览

| 类别 | 数量 | 概述 |
|------|------|------|
| 必须优化 | 1 | 量化 weight×scale 标量预乘（= scalar 报告 M-1，此处从热路径视角重申） |
| 建议优化 | 5 | arch22 TopK 全排序 vs radix、mm1 跨核 GM 回传、密集 PipeBarrier、SplitCore 尾核均衡、S2 块全排序截断 |
| 已优化 / 无需优化 | 6 | Query L1 跨 s2 复用、EXPAND_CHUNK 分段写、LD 乒乓缓冲、小 tile PipeBarrier 启发式、尾块标量写等 |

> 核心结论：算子整体流水结构（Cube mm1 → Vector TopK → ExpandAndAppendIndices）合理，关键路径已做 double-buffer 与 L1 复用；主要优化空间集中在 **arch22 的 TopK 实现**（全排序 + LD 跨核归并，与 arch35 的 radix/histogram TopK 存在代际差距）以及一处的量化预乘标量循环。均无阻塞性正确性问题。

---

## 一、必须优化

### [M-1] 量化 weight×qScale 逐元素标量预乘（arch35 quant 热路径）

- **位置**：`arch35/pool_key_indexer_service_vector.h:638-649`（`ProcessVec1`，`if constexpr (LIT::isFp8PerToken)` 分支）
- **佐证**：该函数处于分数计算热路径（每 gS1 块 × 每 s2 块迭代一次），循环对每 s1 行做 `gSize_`（n1，≤64）次「标量 GetValue × 2 + 浮点乘 + SetValue」。
- **改法/收益**：见 `code_review_scalar_loop_report.md` 的 [M-1]——`Cast(half→float) + Mul<float>` 向量化，附 gSize 非 8 倍数尾段处理与回归要求。

---

## 二、建议优化

### [S-1] arch22 TopK 走「全排序 + LD 跨核归并」，与 arch35 radix TopK 差距最大

- **位置**：
  - 全排序：`arch22/pool_key_indexer_vector.h:190-266`（`SortAll`：Sort32 → MrgSort4 逐级归并，O(S2·log S2)）
  - 分块就地合并：`arch22/pool_key_indexer_service_vector.h:612-647`（每 s2 块 `SortAll` 全排序后 `MergeSort` 截断到 virTopK）
  - 跨核归并：`arch22/pool_key_indexer_service_vector.h:820+`（`ProcessLD`，S2 被切多核时经 vec1Res/vec1Param 三段 GM workspace 做头尾规约）
- **对照**：`arch35/vf/pool_key_indexer_topk.h:67-114` 走 `LiTopKVF`（直方图 radix 选择 TopK，O(S2)），无全排序、无 LD 跨核段（scoreGm 单段）。
- **差距**：arch22 对每个 512 元素 s2 块做全排序，再逐块与 running topK 归并；当 `sparseCount`（真实需要的池数）远小于 512（即 poolSize 大）时，全排序 512 是显著浪费。且 isSparseCountOver2K 时还要经 LD 跨核规约，多一程 GM round-trip。
- **建议**：
  1. 低优先：arch22 上评估「threshold 选择 TopK」或「histogram/radix」类向量指令（若 dav_c220 有等价 VF 指令）替换全排序；
  2. 中优先：sparseCount 较小时（poolSize 大）改「分块全排序」为「每块只保留 top sparseCount 再归并」，减少 sort 规模与 UB 往返。

### [S-2] mm1 结果跨核 GM 回传（AIC→AIV，每 s2 块一次 round-trip + 跨核 sync）

- **位置**：写侧 `arch22/pool_key_indexer_service_cube.h:392-396`（`Fixp` 把 L0C 累加器 `DataCopy` 到 `mm1ResGm_`）；读侧 `arch22/pool_key_indexer_service_vector.h:577`（`CopyIn` 从 `mm1ResGm` 搬回 UB）；配合 `CrossCoreSetFlag/WaitFlag`（`kernel.h:709-715`）。
- **性质**：Cube/Vector 分核架构固有。已做 ws double-buffer（`WS_DOUBLE=2`），但每 s2 块仍有一次 GM 读写。
- **建议**：若后续平台支持 Cube→Vector 的 L1 直通（同级 L1 常驻 / `SetFixpipe` 直写对端 L1）可消除该 GM 往返；当前 arch22/arch35 分核模型下优先级低。

### [S-3] TopK/DoScale/DoReduce 链路 PipeBarrier 密集

- **位置**：`arch22/pool_key_indexer_vector.h:82-114`（`DoScale` 每步 `PipeBarrier<PIPE_V>`）、`:129-153`（`DoReduce` 每次 Add 后 barrier）、`:190-250`（`SortAll` 每次 Sort/Mrg 后 barrier）、`service_vector.h:552-653`（块内 barrier 十余处）。
- **性质**：多数 barrier 为依赖正确性所需（V pipe 内退序写读），但存在「上一条向量指令本就在同 pipe、无需 barrier」的可删项；逐条核对可削减同步开销。
- **建议**：以 msprof 的 aiv_sync/barrier 计数为据，逐条审查后删除冗余 `PipeBarrier<PIPE_V>`（参照 `pipebarrier-checker` 的依赖链规则）。低风险、收益中等。

### [S-4] SplitCore 按「基本块静态切分」在 causal 变长下尾核不均衡

- **位置**：`arch22/pool_key_indexer_kernel.h:300-379`（`SplitCore`，`minBlockPerCore + 余数前核多一块`），块数统计 `GetTotalBaseBlockNum:276-296`、causal 逐 gS1 块数 `GetS2BaseBlockNumOnMask:258-274`。
- **性质**：causal 下每个 gS1 块的可见 s2 块数不同（三角掩码），按总块数静态均摊后，尾核可能只分到很薄的工作量；小 shape（如 S1=1、batch 小）下多核大量空转。
- **建议**：
  1. S2 单核能装下（actS2 ≤ s2BaseSize）时不切 s2、优先在 gS1/bN2 轴铺满核，减少三角形带来的切分粒度损失；
  2. 对「块数 < 核数」的小 shape，server 端收敛 usedCoreNum（tiling 已传 usedCoreNum，目前是 max 核数），避免空转核的 InitBuffers/同步开销。

### [S-5] 小 tile 场景的 Cube 指令利用率（M/N 非 16 对齐尾块）

- **位置**：`arch22/pool_key_indexer_service_cube.h:361-375`（`ComuteL0c`，`m=CeilAlign(s1gL0RealSize,16)`、`n=s2L0RealSize`）；`:207-241`（`KeyNd2Nz` 尾块按 `S2_BASIC_BLOCK_L0=128` 对齐分段）。
- **性质**：尾部基本块 `m`（M 轴，≤ s1BaseSize×gSize）与 `n`（s2 块尾）非 16 对齐时，Mmad 按对齐量计算并配合 `Fixp` 写有效区，浪费的 lane 在 UB 计算量里占比随 shape 收窄升高。
- **建议**：登记观察项；仅在「小 S1/小池数」成为真实业务形状时再针对尾块做专门 tiling。

---

## 三、已优化 / 无需优化（肯定项与接受项）

| # | 位置 | 说明 | 判定 |
|---|------|------|------|
| A1 | `arch22/arch35 service_cube.h: ComputeMm1` | Query 每 s1g 块仅首 s2 块搬一次，跨 s2 块复用（`isFirstS2InnerLoop && s2GmOffset==0` 才 `QueryNd2Nz`），避免重复搬运 | 已优化 ✓ |
| A2 | `service_cube.h:38-40` | Key L1 三重缓冲（`KEY_BUF_NUM=3`）、Query/L0 双缓冲，MTE1/MTE2/M 事件流水 | 已优化 ✓ |
| A3 | `arch22 service_vector.h:375-445 / 455-470` | ExpandAndAppendIndices/CleanInvalidOutput 按 `EXPAND_CHUNK=1024` 分段，不随 topk 线性撑爆 192KB UB（缺陷 D/E 修复） | 已优化 ✓ |
| A4 | `service_vector.h:562-587` | mm1/weights 搬运用 `tmpUb_` pingpong 双缓冲，搬运与计算重叠 | 已优化 ✓ |
| A5 | `service_cube.h:373-375` | 小 mmad（m/16 · n/16 < 10）加 `PipeBarrier<PIPE_M>` 防 stall | 已优化 ✓ |
| A6 | scalar 报告 N-1~N-4 | 尾块标量写、PA 逐块 GetValue、元数据 GetValue、常量 SetValue | 接受（见 scalar 报告） |

---

## 附：arch22 vs arch35 关键实现差异（性能角度）

| 维度 | arch22 (A2/910B3) | arch35 (A5/950) |
|------|------|------|
| TopK 实现 | Sort32+MrgSort 全排序 + LD 跨核归并（GM workspace 三段） | VF 直方图/radix `LiTopKVF`（O(n)，scoreGm 单段） |
| TopK 跨核 | `ProcessLD` 头尾规约（GM round-trip） | 无 LD 段 |
| UB 空间 | 192KB 硬约束（固定缓冲 ~171KB） | 248KB |
| Cube 固定缓冲 | sortOut 64KB + tmp 68KB + outQueue 33KB | 无大型 sort 缓冲（radix 复用共享 tmp） |
| 量化反量化 | 不支持（tiling fail-fast 拒绝） | MMad fp8 + LoadData-Mx / vector scale 融合 |

> 上表说明：arch22 的 TopK 全排序 + LD 归并是相对 arch35 的最主要结构性性能差距（S-1），也是 build_fix_record 中缺陷 D/E 崩陷的根因面（LD 区域 sizing），建议将其作为 arch22 侧后续优化的第一优先级。
