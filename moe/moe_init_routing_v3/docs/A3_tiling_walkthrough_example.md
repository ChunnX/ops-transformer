# MoeInitRoutingV3 A3 Tiling 推导样例（64 核 / UB 248KB）

> 配套文档：`A3_tiling_strategy.md`
> 代码位置：`op_host/moe_init_routing_v3_tiling.cpp`（行号均指该文件）、`op_kernel/moe_init_routing_v3.cpp`、`op_kernel/moe_v3_sort_multi_core.h`

## 1. 样例输入

### 1.1 平台参数

| 参数 | 值 |
|---|---|
| `aivNum`（AIV 核数） | 64 |
| `ubSize` | 248 KB = 253,952 B |

### 1.2 算子输入

| 参数 | 值 | 说明 |
|---|---|---|
| n | 16384 | token 数 |
| k | 8 | topK |
| **totalLength = n × k** | **131072** | 排序/搬运总元素数 |
| expertNum | 256 | 专家总数 |
| expertStart / expertEnd | 0 / 256 | 全域（非 EP）→ `actualExpertNum = 256`，`ep_ = 0` |
| cols | 7168 | x 的列数（hidden size） |
| x dtype | bfloat16 | `inuptXDtypeSize_ = 2` |
| quantMode | -1（UN_QUANT） | 非量化 |
| dropPadMode | 0（DROP_LESS） | dropless |
| rowIdxType | 1（SCATTER） | expandedRowIdx 为 scatter 索引 |
| expertTokensNumType | COUNT | 输出每个专家的 token 数 |

选参说明：n=16384 > 8192（`PERFORMANCE_MODE_BS_MAX`）且 actualExpertNum=256 > 32（`PERFORMANCE_MODE_RANGE_MAX`），**不命中 GatherSort 性能模板**；dropless 且 `actualExpertNum×2 = 512 > 256 = expertNum`，`ShouldEnableCountingSortByShape`（:814）返回 false，**不启用 CountingSort**；totalLength 远超单核排序容量，**不命中全载**。因此走**通用多核排序路径**，正好覆盖全部六个阶段的切分。

## 2. 全局量推导

`sortLoopMaxElement`（:1144）——单核单轮排序最大元素数：

```
sortLoopMaxElement = (ubSize − aivNum × 32B) / (4 × 2 × 4) / 32 × 32
                   = (253952 − 64 × 32) / 32 / 32 × 32
                   = 251904 / 32 / 32 × 32
                   = 7872
```

（`4×2×4`：排序每元素 float32 × 双缓冲 × 4 组 buffer；`/32×32`：向下 32 对齐）

## 3. 六阶段切分推导

### 3.1 VBS（排序）— `Tiling4VBSCompute` :1317

`totalLength = 131072 > 7872` → **sortMode = 1（多核排序）**。

`Tinlig4VBSMultiCoreCompute` :1265：

```
needCoreNum = ceil(131072 / 7872) = 17
            → 4^ceil(log4(17)) = 4^3 = 64     （取 4 的幂，适配 4 路归并树）
            → min(64, aivNum=64) = 64
perCoreElements      = 131072 / 64 = 2048
alineFloor           = 2048 − 2048 % 32 = 2048
lastCoreElement      = 131072 − 63 × 2048 = 2048
alineCeil            = 2048 + 32 = 2080
lastCoreElement(2048) ≤ alineCeil(2080) → perCoreElements = alineFloor = 2048
```

do-while 保护（:1287）：第一轮 `lastCoreLastLoopElements = 2048 > 0`，直接通过。

| 字段 | 值 |
|---|---|
| needCoreNum | **64** |
| perCoreElements / perCoreLoops / perLoop / lastLoop | 2048 / **1** / 2048 / 2048 |
| lastCoreElements / loops / perLoop / lastLoop | 2048 / 1 / 2048 / 2048 |

→ **64 个核各排 2048 个元素，单轮完成**，产生 64 条长度 2048 的有序段。

### 3.2 VMS Middle（中间归并）— `Tiling4VMSMiddleCompute` :1334

`vbsNeedCoreNum = 64 > MRG_LIST_NUM(4)` → 需要中间归并：

```
needCoreNum = ceil(64 / 4) = 16
```

→ **16 个核各做 1 次 4 路归并**（4 × 2048 → 8192），产生 16 条长度 8192 的有序段。

### 3.3 SortOut（最终归并）— `Tiling4SortOutCompute` :1346

```
oneLoopMaxElements = mrgSortListMaxElement = 1504   （:231 常量）
```

→ **单核**将 16 条有序段（16 × 8192 = 131072）做最终归并输出，每轮最多处理 1504 元素的归并窗口。

### 3.4 ExpertTokensCount（专家 token 统计）— `Tiling4ExpertTokensCountCompute` :1352

```
totalElements   = n × k = 131072
perCoreElements = ceil(131072 / 64) = 2048
needCoreNum     = 64
lastCoreElements= 131072 − 63 × 2048 = 2048

expertNumElement = 256                     （COUNT 模式，取 actualExpertNum）
专家统计缓冲     = CeilAlign(256, 32) × (int32×2 + int64) = 256 × 16 = 4096 B
maxElementsPerLoop = (253952 − 4096 − 32) / 4 = 62456
```

| 字段 | perCore | lastCore |
|---|---|---|
| elements | 2048 | 2048 |
| loops | 1 | 1 |
| perLoop / lastLoop | 2048 / 2048 | 2048 / 2048 |

→ 64 核各统计 2048 个有序元素，单轮完成（62456 ≫ 2048，UB 绰绰有余）。

### 3.5 SrcToDst（dropless 索引搬运）— `Tiling4SrcToDstCompute` :1527

```
remainUb = 253952 − ASSIST_NUM(256)×4 − 32×(256+1) = 253952 − 1024 − 8224 = 244704 B
perLoopMaxElements = 244704 / (32B + 4B) = 6797
perCoreElements = ceil(131072 / 64) = 2048 → needCoreNum = 64，lastCore = 2048
```

perCore / lastCore 均为：loops = 1，perLoop = 2048，lastLoop = 2048。

→ 64 核各搬运 2048 个索引，单轮完成。

### 3.6 SrcToDstDropPad（计算但本场景不使用）— :1399

```
perCoreRows = 2048，needCoreNum = 64，lastCoreRows = 2048
rowSize = align32(2048 × 4 × 2)  = 16384 B
colSize = align32(7168 × 2)      = 14336 B
rowSize + colSize = 30720 < 253952 → 命中「行全载」分支（SetGatherTilingData，单 loop）
```

dropPadMode=0 时 kernel 不执行该路径，tiling 数据仅作占位。

### 3.7 GatherOut（按专家聚簇搬 x 行）— `Tiling4GatherOutCompute` :1562

```
perCoreIndicesElements = 2048 → needCoreNum = 64，lastCore = 2048
perLoopCols  = 7168
colMultiple  = 2 × 2 = 4        （非量化：双缓冲 × dtypeSize）
rowMultiple  = 2
perLoopMaxIndicesElements = (253952 − 7168×4 − 64) / 2 / 4 = 225216 / 8 = 28152 > 0
colsLoops = ceil(7168 / 7168) = 1，lastLoopCols = 7168
perCorePerLoopIndices = min(28152, 2048) = 2048 → loops = 1，lastLoop = 2048
```

→ 64 核各 gather 2048 行 × 7168 列（bf16），列方向整载、单轮完成。

## 4. 快路径判定（均不命中）

| 快路径 | 判定 | 结果 |
|---|---|---|
| IsFullLoad :760 | `totalLength(131072) > sortLoopMaxElement(7872)` → :777 直接返回 false | ✗ |
| CountingSort :821 | n=16384 ∈ [256, 1M]，但 `ShouldEnableCountingSortByShape`：dropless 且 256×2 > 256 → false | ✗（`countingSortFullLoad_ = -1`） |
| GatherSort 性能模板 :1022 | n=16384 ∉ [384, 8192]，且 actualExpertNum=256 > 32 | ✗ |

## 5. TilingKey、Workspace、BlockDim

### 5.1 TilingKey（:1199 通用分支）

```
tilingKey = 1000000 + sortMode(1)×100000 + (quantMode+1)(0)×10000
          + rowIdxType(1)×1000 + dropPadMode(0)×100
          = 1101000
```

对应 `MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER_NODROP`（多核排序、非量化、SCATTER、非 DropPad，`moe_init_routing_v3.cpp:58`）。

### 5.2 Workspace（`GetWorkspaceSize` :1206）

| 项 | 计算 | 字节数 |
|---|---|---|
| sortWorkspace | 131072 × 2 × 3 × 4 | 3,145,728 |
| coreSyncWorkspace | 64 × 32 × 2 | 4,096 |
| scatterWorkspace | 131072 × 4 | 524,288 |
| expertTokensCount | 256 × 4 | 1,024 |
| expertTokenTotalCount | align32(4) | 32 |
| 基础空间 | 16 MB | 16,777,216 |
| **合计** | | **20,452,384 B ≈ 19.51 MB** |

（非动态量化、非 dropPad、countingSortWorkspace=0，无附加项）

### 5.3 BlockDim（`PostTiling` :1238）

```
SetBlockDim(aivNum) = 64
```

## 6. Kernel 执行过程（tilingKey = 1101000）

`moe_init_routing_v3.cpp` 中该 tilingKey 的执行序列（**阶段严格串行，阶段内多核并行**）：

```
┌─ 阶段 1：MoeSortMultiCore（sortPipe，:345）──────────────────────────┐
│                                                                      │
│  ① VBS：64 核并行，core i 排序 expertIdx[2048i, 2048(i+1))           │
│         （key=expertId, value=原始行号），结果写入 sortWorkspace      │
│         ── SyncAll（moe_v3_sort_multi_core.h:245）──                 │
│  ② VMS Middle：16 核并行，core j 对第 4j~4j+3 条有序段做 4 路归并    │
│         → 16 条长度 8192 的有序段                                     │
│         ── SyncAll（:282）──                                         │
│  ③ SortOut：单核（core 0）对 16 条有序段做最终归并，                 │
│         每轮窗口 ≤ 1504 元素，输出全局有序的 (expertId, rowIdx)       │
│         ── SyncAll（:298）──                                         │
└──────────────────────────────────────────────────────────────────────┘
                ↓ 排序结果（workspace，全局有序）
┌─ 阶段 2：ExpertTokensCount<COUNT_MODE>（:382）───────────────────────┐
│  64 核并行，各扫描 2048 个有序元素做专家直方图统计，                  │
│  汇总写出 expertTokensCountOrCumsum[256]                             │
└──────────────────────────────────────────────────────────────────────┘
                ↓
┌─ 阶段 3：RowIdxGather（SrcToDst，:422）──────────────────────────────┐
│  64 核并行，各处理 2048 个排序位置，生成 expandedRowIdx               │
│  （SCATTER 语义：expandedRowIdx[排序后位置] = 原始 token 行号）       │
└──────────────────────────────────────────────────────────────────────┘
                ↓ expandedRowIdx
┌─ 阶段 4：MoeGatherOut<bf16, ep=0>（:442）────────────────────────────┐
│  64 核并行，各 gather 2048 行：expandedX[i, :] = x[rowIdx[i], :]     │
│  每行 7168 列 bf16 整载（14336 B ≪ UB），单轮完成                    │
└──────────────────────────────────────────────────────────────────────┘
```

要点：

1. **阶段间严格串行**：后一阶段消费前一阶段的 GM 输出，多核排序内部有三处显式 `SyncAll`（:245/:282/:298，同步屏障使用 coreSyncWorkspace = 4 KB）；后续阶段之间由同一 kernel 内的程序顺序 + 跨核 GM 依赖处的同步隔离。
2. **阶段内并行度各不相同**：VBS 用满 64 核 → VMS Middle 16 核 → SortOut 收缩到 1 核 → 统计/搬运重新用满 64 核。这正是每个阶段必须独立切分的原因。
3. **本样例所有阶段核内 loops 均为 1**：每核 2048 元素远小于各阶段的 perLoop 上限（7872 / 62456 / 6797 / 28152），UB 利用率瓶颈不在此处；若 n 继续增大（如 n=65536，perCoreElements=8192），VBS 阶段将首先出现 `perCoreLoops = ceil(8192/7872) = 2` 的多轮切分。
