# MoeInitRoutingV3 A5（950/arch35）Tiling UB 切分梳理

> 代码位置：`op_host/moe_init_routing_v3_tiling_arch35.cpp`（行号均指该文件）
> kernel 侧：`op_kernel/arch35/`
> 配套文档：`A3_tiling_strategy.md`（A3 对照）、`A3_vs_A5_diff.md`

## 0. UB 总预算：先扣 SIMT DCache

A5 与 A3 的第一个差异在 UB 入口（`CheckSetPlatformInfo` :464）：

```
availUbSize_ = totalUbSize_ − SIMT_DCACHE_SIZE(64KB)   // :31/:476
```

SIMT 指令需要 64KB DCache 空间（kernel 侧配合 `SetLocalMemSize()`），**后续所有阶段的 UB 推算都用 `availUbSize_`**，不再直接用物理 UB。无硬编码物理 UB 值，`totalUbSize_` 来自平台查询。

## 1. 两条 UB 使用路线总览

```
                         DoOpTiling (:503)
                               │
        sortLoopMaxElement_ = min(availUb/32, 8160)  ← :523，8160 = Sort API 上限(32×255)
                               │
        Tiling4VBS / VMSMiddle / SortOut / ExpertTokensCount   ← 分阶段 tiling 无条件先算
                               │
                     IsFullLoad() (:2131) ── 仅 dropless 可能为 true
                ┌──────────────┴──────────────┐
        isFullload_ = true                isFullload_ = false
        单核 UB 全载模板                    分阶段处理模板
        tilingKey = 200000                tilingKey = 10000000
          + (quant+1)×10000                 + sortMode×10⁶ + (quant+1)×10⁴
                                            + rowIdxType×10³ + dropPad×10²
                │                               │
        kernel: moe_v3_full_load_*.h      kernel: sort_multi_core / sort_one_core
                                            + expert_tokens_count + row_idx_gather
                                            + gather_out*.h
```

**注意**：A5 没有 A3 的 CountingSort 快路径和 GatherSort 性能模板，快路径只有全载一条。

## 2. 单核 UB 全载（FullLoad）

### 2.1 触发条件（`IsFullLoad` :2131）

三个**必要条件**，全部满足才进入 UB 预算核算：

| 条件 | 代码 | 含义 |
|---|---|---|
| dropless | :2133 `dropPadMode_ == DROPPAD → false` | 容量裁剪模式永远走分阶段 |
| 单核可排序 | :2150 `totalLength_ ≤ sortLoopMaxElement_` | 排序数据必须能一次放进 UB（且 ≤8160，AscendC::Sort 全排序模式上限） |
| UB 预算有剩余 | :2190 `remainUb > 0` | 见下式 |

UB 预算逐项扣除（:2154-2188）：

```
remainUb = availUbSize_
         − sortSpace    = sortNum × 4B × 6      （6 个排序 buffer，sortNum = totalLength 向上 32 对齐）
         − rowIdxSpace  = sortNum × 4B × 3      （3 个索引 buffer）
         − expertSpace  = align32(expertNum × 8B) × 3
         − 1 KB                                  （杂项预留）
         − gather 空间（按量化模式）：
             非量化:  align32(cols × dtypeSize) × perCoreTokens + 32B
             静态量化: gatherSpace + align(cols) × 5 × perCoreTokens
             动态量化: align32(cols) × 9 + 64B      （DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER=9）
             INT4动态: inputX 双缓冲 + smooth + calc + int4 输出 + scaleOut 逐项精算 (:2170-2181)
```

其中 `perCoreTokens`（:2136-2147）：非 EP 且非动态量化时 = `n/aivCoreNum + 1~2`（余数 >1 时 +2，因为左右边界行可能跨核），EP 场景 = 1。

### 2.2 全载的 UB 布局特征：所有数据**同时驻留** UB

kernel 侧 `moe_v3_full_load_base.h:225-232` 的 buffer 分配证明了这一点：

| buffer | 大小 | 说明 |
|---|---|---|
| sortDataCopyInQueue_ | buffSize × kvFactor | 全部 expertIdx 一次性搬入 |
| sortedExpertIdxQueue_ / sortedRowIdxQueue_ / expandedRowIdxQueue_ | 各 buffSize | 排序结果全程在 UB |
| expertTokensCopyOutQueue_ | align32(expertNum × 8B) | 统计结果 |
| tempBuffer_ / sortedBuffer_（TBuf） | buffSize × kvFactor | 排序临时空间 |
| expertCountBuf_（TBuf） | align32(expertNum) | 专家直方图 |

`buffSize` 按 **totalLength 全量** 计算——排序输入、排序输出、索引、统计、以及每核要 gather 的 token 行**同时**占着 UB。这正是 tiling 侧 `IsFullLoad` 要做逐项预算核算的原因：kernel 不会分轮释放 UB，所有空间需求是**叠加**关系。

### 2.3 执行特征

- 排序：每核独立对 `totalLength` 全量做单核排序（sortMode 隐含为单核，不走路径编码里的 sortMode=1）。
- 无多核排序的 workspace 中转、无 VMS/SortOut 归并阶段、**无 SyncAll 阶段屏障**。
- gather：核内直接按排序结果搬 `perCoreTokens` 行 x，流水在核内闭环。
- tilingKey = `200000 + (quantMode+1) × 10000`（:563-566），INT4 动态量化复用 INT8 动态量化 tilingKey（:564）。

## 3. 分阶段处理（非全载）

与 A3 相同的阶段框架（VBS → VMS Middle → SortOut → ExpertTokensCount → GatherOut），**每个阶段独立占用整块 `availUbSize_`**，阶段间 UB 复用（互不相干地重新规划），GM workspace 做数据中转，SyncAll 做阶段屏障。各阶段 UB 切分：

### 3.1 VBS（排序）— :523, :1597

```
sortLoopMaxElement_ = availUbSize_ / (4×2×4) / 32 × 32     // 每元素 32B：fp32 排序值 × 双缓冲 × 4 组
                    = min(上式, 8160)                       // Sort API 全排序上限 32×255
```

- `totalLength ≤ sortLoopMaxElement_` → sortMode=0 单核排序（注意：若 IsFullLoad 预算没通过，即使单核可排也走分阶段模板）。
- 否则多核：核数取 4 的幂（适配 4 路归并树），perCoreElements 32 对齐均分 + do-while 尾核保护（:1621-1640，逻辑同 A3）。
- UB 占用模型：单轮排序数据 = 元素数 × 32B，**整块 UB 全部用于排序 buffer**。

### 3.2 VMS Middle / SortOut — :1669, :1685

- VBS 核数 > 4 时 VMS Middle 核数 = ceil(vbsCores/4)。
- SortOut `oneLoopMaxElements = MRG_SORT_API_MAX_ELEM = 1024`（:33, :1690）——A5 直接取归并 API 能力上限（A3 是经验值 1504），单核最终归并。

### 3.3 ExpertTokensCount — :1695

```
maxElementsPerLoop = (availUbSize_ − align32(expertNumElement)×(4B×2+8B) − 32B) / 4B
```

UB 里先扣专家直方图缓冲（int32×2 + int64，KEY_VALUE 模式 expertNumElement×2），剩余全部换算成每轮可扫描的 int32 元素数。结构同 A3，仅 UB 基数换成 availUbSize_。

### 3.4 GatherOut（UB 切分最复杂的阶段）— :1887

UB 被切成 **rowIdx 队列 + x 队列（多缓冲）+ scale 队列** 三块，且按量化模式分出五种切法：

**通用流程**：核间均分 indices（`perCoreIndicesElements = ceil(totalLength/aivCoreNum)`），核内：
1. 由 `GetMultipleParams` (:1734) 确定 `colMultiple`（每列字节代价）/ `rowMultiple`（每行索引代价）：
   - 非量化：`2×dtypeSize` / 2
   - 静态量化：`2×dtypeSize + 8` / 2
   - 动态量化（含 INT4）：`21` / 4
   - HIF8 pertoken/pertensor：`5` / 4；HIF8 cast bf16：`2×(dtypeSize+2×dtypeSize)`
2. `GetPerLoopParams` (:1760)：以整列 `perLoopCols = cols` 起步，算 `perLoopMaxIndicesElements = (availUb − cols×colMultiple − …) / rowMultiple / 4B`；**≤0 则 perLoopCols 不断折半**（:1769/:1783/:1844）直到为正——宽 cols 的兜底。
3. **非量化 dropless 专属**（`SetPerLoopParams4NoQuantDropLess` :1810）：UB 先对半（`availUb/2`），算出每轮 indices 后，**剩余空间全部换成额外的 x 队列缓冲份数**：
   ```
   xCopyInQueueBufferNum = min(2 + remaining/xQueueSize, MAX_QUEUE_BUFFER_NUM=6)   // :1865
   ```
   即 GatherOut 的 x CopyIn 队列深度从 2 起步、UB 富余时加深到最多 6 级，用多缓冲吃满搬运流水。kernel 侧对应 `moe_v3_gather_out.h:151-153`（按 tiling 的 `xCopyInQueueBufferNum` 实际 InitBuffer）。
4. **MXFP8/MXFP4 量化**（:2027）与 **FP8 perblock/group 量化**（:2082）：UB 预算按 `availUbSize_/2` 切（:2003/:2079，输入/输出各半），列块分别按 32（MX_QUANT_BLOCK_SIZE）/ 128（FP8_PERBLOCK_BLOCK_SIZE）对齐折半；MXFP4 场景 kernel 还会按 expertTotalCount 在 device 侧重算核间切分（:2055 注释）。

### 3.5 DropPad 两个变体 — :2215, :2326

仅 `dropPadMode=1 且非全载` 时计算（:542-544）：
- SrcToDstDropPad：行均分 + `maxPerLoopRows = (availUb − 32B)/4B/2` 按 8 行（32B/4B）对齐，列方向不切（colLoops=1）。
- GatherOutDropPad：另有 `useCompactGatherOutDropPad` 分支，workspace 增加 `expertNum × expertCapacity × 4B` 的 outputToSrcRow 映射（:620-623）。

## 4. 两条路线的本质区别

| 维度 | 单核 UB 全载 | 分阶段处理 |
|---|---|---|
| 触发 | dropless + totalLength ≤ min(availUb/32, 8160) + UB 预算 remainUb>0 | 其余所有场景（dropPad 必然走这里） |
| UB 使用模型 | **叠加**：排序+索引+统计+gather 全部 buffer 同时驻留，tiling 需逐项预算 | **复用**：每阶段独占整块 availUb，阶段结束即释放给下一阶段 |
| 排序 | 单核一次 AscendC::Sort，≤8160 元素 | 多核 4 路归并树（VBS→VMS→SortOut），规模无上限 |
| 数据中转 | 无 GM 排序中转，UB→输出直通 | sortWorkspace（totalLength×24B）等 GM workspace 中转 |
| 阶段屏障 | 无 SyncAll，核内流水闭环 | 阶段间 SyncAll（coreSyncWorkspace） |
| gather 流水 | 每核 perCoreTokens 行，固定缓冲 | x 队列多缓冲 2~6 级（UB 富余动态加深），perLoopCols 折半兜底宽列 |
| tilingKey | `200000 + (quant+1)×10⁴` | `10000000 + sortMode×10⁶ + …` |
| 适用场景 | 小 totalLength（典型小 BS×k），省掉 workspace 读写和同步开销 | 大 totalLength / dropPad / 宽列 / 复杂量化 |

## 5. 与 A3 的 UB 切分差异速查

| 项 | A3 | A5 |
|---|---|---|
| UB 基数 | `ubSize − aivNum×32B`（仅 VBS 公式扣） | 全局统一 `totalUb − 64KB SIMT DCache` |
| 单轮排序上限 | 纯 UB 推导 | 额外 clamp 到 Sort API 上限 8160 |
| SortOut 窗口 | 1504（经验值） | 1024（MRG_SORT_API_MAX_ELEM） |
| 全载 expertSpace | ×2 buffer | ×3 buffer |
| 全载量化列缓冲 | 静态 11 / 动态 13 | 静态 5 / 动态 9（INT4 单独精算） |
| GatherOut 多缓冲 | 固定双缓冲 | xCopyInQueueBufferNum 动态 2~6 |
| 快路径 | 全载 + CountingSort + GatherSort 性能模板 | 仅全载 |
