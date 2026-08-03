# MoeInitRoutingV3 A3 Tiling UB 切分梳理：分阶段处理 vs 其他路径

> 代码位置：`op_host/moe_init_routing_v3_tiling.cpp`（行号均指该文件）、`op_kernel/`
> 配套文档：`A3_tiling_strategy.md`（阶段切分细节）、`A5_tiling_ub_strategy.md`（A5 对照）

## 0. UB 总入口

A3 直接使用 `PlatformAscendC` 查询的 `ubSize`，**没有 A5 的 64KB SIMT DCache 预留**。唯一的全局性扣除发生在 VBS 公式里（`− aivNum × 32B`），其余各阶段都按完整 `ubSize` 逐项预算。

## 1. 路径总览：一条主路 + 三条快路径

```
DoOpTiling (:1113)
    │
    ├─ X_1_7168 性能模板判定 (:996) ── 命中 → aivNum = totalLength = 8，tilingKey 2000000
    │
    ├─ 七个 Tiling4Xxx 分阶段切分（无条件先算，:1147-1153）
    │
    ├─ IsFullLoad (:760) ─────────── 命中 → 全载模板，tilingKey 21/22/23xxxxx
    │
    └─ 非全载 → ComputeCountingSortTiling (:821)
            ├─ countingSortFullLoad_ = 1 → 全载计数排序，tilingKey sortMode=2
            ├─ countingSortFullLoad_ = 0 → cutOrigin 计数排序，tilingKey sortMode=3
            └─ countingSortFullLoad_ = -1 → 回退通用分阶段路径
                  （此时若命中 GatherSort 性能模式 :1022，tilingKey sortMode=2/3）
```

**优先级**：X_1_7168 模板 > 全载 > CountingSort（全载/cutOrigin）> GatherSort 性能模式 > 通用分阶段。
所有路径的六/七个阶段 tiling 都会被计算，但 kernel 由 tilingKey 决定实际执行哪套。

## 2. 分阶段处理（通用路径）

UB 使用模型：**复用**——每个阶段独占整块 ubSize，阶段间 UB 重新规划、GM workspace 中转、SyncAll 屏障。各阶段 UB 切分公式：

### 2.1 VBS（排序）— :1144, :1265

```
sortLoopMaxElement = (ubSize − aivNum×32B) / (4×2×4) / 32 × 32
```

- 每元素代价 32B = fp32 排序值 × 双缓冲 × 4 组 buffer；`aivNum×32B` 是多核同步等辅助预留（A3 特有，A5 改为统一扣 64KB）。
- **A3 无 Sort API 元素上限 clamp**（A5 clamp 到 8160）。
- 多核时核数取 4 的幂，perCoreElements 32 对齐均分 + do-while 尾核保护。

### 2.2 SortOut — :1346

`oneLoopMaxElements = mrgSortListMaxElement = 1504`（经验值，A5 是 API 上限 1024），单核最终归并。

### 2.3 ExpertTokensCount — :1352

```
maxElementsPerLoop = (ubSize − align32(expertNumElement)×(4B×2+8B) − 32B) / 4B
```

UB 先扣专家直方图缓冲（int32×2+int64，KEY_VALUE 模式 expertNumElement = (actualExpertNum+1)×2），余量换算成每轮 int32 元素数。

### 2.4 SrcToDst（dropless 索引搬运）— :1527

```
remainUb = ubSize − ASSIST_NUM(256)×4B − 32B×(256+1)     // 辅助区 + 对齐预留
perLoopMaxElements = remainUb / (32B + 4B)                // 每元素 36B
```

### 2.5 SrcToDstDropPad — :1399（非动态量化）/ :1491（动态量化）

- 行均分；`rowSize + colSize < ubSize` → 行全载单 loop。
- 否则行列二维切分：列上限 `MAX_COLS_ONE_LOOP = 16376`；行上限 = `(ubSize − 列缓冲 − 32B)/4B/2`；cols 小于上限时优先压行、反向放大列块。
- 动态量化变体：按 `DYNAMIC_QUANT_SRC_TO_DST_BUFFER = 15` 个 int8 buffer + 64B scale 估算，列上限 `MAX_COLS_DYNAMIC_QUANT = 6144`。

### 2.6 GatherOut — :1562

```
perLoopMaxIndicesElements = (ubSize − Align(perLoopCols)×colMultiple − 64B) / rowMultiple / 4B
```

| 量化模式 | colMultiple | rowMultiple |
|---|---|---|
| 非量化 | 2×dtypeSize | 2 |
| 动态量化 | 21 | 4 |
| 静态量化 | int8×2 + fp32 + int16 + dtypeSize×2 | 2 |

结果 ≤0 时 `perLoopCols` 不断折半（:1588-1595）兜底宽列。**A3 的 x 队列固定双缓冲**，没有 A5 的 2~6 级动态加深。

## 3. 其他路径一：单核 UB 全载（FullLoad）

### 3.1 触发条件（`IsFullLoad` :760）

- `dropPadMode == 1` → 直接排除（:777）
- `totalLength > sortLoopMaxElement` → 排除（单核排不下）
- UB 逐项预算 `remainUb > 0`：

```
remainUb = ubSize
         − sortSpace   = sortNum × 4B × 6        （ONE_CORE_SORT_BUFFER=6，:786）
         − rowIdxSpace = sortNum × 4B × 3
         − expertSpace = align32(expertNum × 8B) × 2
         − 1 KB
         − gather 空间（按量化模式）：
             非量化:  gatherSpace + 32B           （gatherSpace = align32(cols×dtype) × perCoreTokens）
             静态量化: gatherSpace + align(cols) × 11 × perCoreTokens   （STATIC_QUANT_FULLLOAD_COLS_BUFFER=11）
             动态量化: align32(cols) × 13 + 64B                         （DYNAMIC_QUANT_FULLLOAD_COLS_BUFFER=13）
```

- EP 场景（expertStart/End 非全域）还支持 **gatherFirstFullload**（:749，先 gather 剔除无效专家再排序）。

### 3.2 UB 使用模型：叠加

kernel 侧 `moe_v3_full_load_base.h:148-154`：5 个 queue（sortDataCopyIn、expandedExpertIdx、expandDstToSrcRow、expandedRowIdx、expertTokens CopyOut）+ 2 个 TBuf（temp、sorted）全部按 totalLength 全量分配——**排序输入/输出/索引/统计/gather 行同时驻留 UB**，所以 tiling 必须逐项叠加预算。无 workspace 排序中转、无 SyncAll 阶段屏障。

### 3.3 TilingKey

非量化 2100000 / 静态 2200000 / 动态 `2300000 + ep×10000 + smoothType×1000`。

## 4. 其他路径二：CountingSort（计数排序）

前置条件（`ShouldEnableCountingSortByShape` :809 + :824）：`n ∈ [256, 1M]`、`expertNum ≤ 1024`、非动态量化、dropless 时要求 `actualExpertNum×2 ≤ expertNum`（即实际专家数不超过一半，计数排序才有收益）、dropless+静态量化时 n>2560、dropPad 时 n≥512。三态标志 `countingSortFullLoad_`：

### 4.1 全载计数排序（=1）

额外条件：`ep == 1`（EP 场景）+ 非量化 + k=8 + dropless + COUNT + `actualExpertNum ≤ 128`，且估算 UB（`EstimateCountingSortFullLoadUB` :872）≤ `ubSize − 1KB`。UB 估算逐项：

```
x 行全载     = coreTokenNum × align32(cols×dtype)        // 每核 token 的 x 行整块驻留
+ entries 索引 align32(coreEntries × 4B)
+ expertCount 直方图（核数 × expertStride × 4B）
+ 比较区 entriesAligned(64 对齐) × fp32 + mask×3 + int32×3
```

UB 模型同样是**叠加**（filter + 计数 + 搬运数据同时驻留）。workspace = `filterNeedCoreNum × expertCountStride × 4B`。tilingKey sortMode=2。

### 4.2 cutOrigin 分块计数排序（=0，`ComputeCountingSortCutOriginTiling` :902）

UB 模型是第三种形态：**persistent + max(phase)**——跨阶段常驻区 + 各阶段峰值取最大值（阶段间 UB 复用）：

```
totalUB = persistentSize(专家计数常驻 align32(expertStride×4B))
        + max( Phase A, Phase B, Phase C )
```

| Phase | 功能 | UB 构成 |
|---|---|---|
| A | 4096 chunk 过滤（`COUNTING_SORT_FILTER_CHUNK_SIZE`） | chunk×int32 + chunk×fp32 + mask×3 + int32×3 + pairs 缓冲 |
| B | 多核计数归约 | 核数×expertStride×4B + expertStride + actualExpertNum×8B |
| C | 搬运 | 固定开销（pairsBuf 2048×2×4B + idxBuf + 3×stack 数组）+ **perLoopCols 由剩余 UB 动态推导**（:945-951，上限 `COUNTING_SORT_MAX_PER_LOOP_COLS_BYTES = 64KB`） |

注意：Phase C 若 `colsLoops > 1` 且量化模式非 UN_QUANT → 直接禁用（置 -1，:960-963）。
workspace = pairs（核数 × align8(maxCoreEntries)×2 × 4B）+ expertCount。tilingKey sortMode=3。

## 5. 其他路径三：性能特化模板

### 5.1 X_1_7168 单 token 模板（:996，tilingKey 2000000）

shape 精确匹配 `x(1,7168) / expertIdx(1,8) / scale(256,7168)`、bf16、动态量化、KEY_VALUE、无 offset 时命中。`aivNum = totalLength = 8`（:1141）——**每核处理恰好 1 个 topK 结果**，UB 需求极小且固定，跳过所有通用公式。kernel 为 `MoeSortActualExpert`。

### 5.2 GatherSort 性能模式（`GetPerformanceMode` :1022）

条件：`expertNum = 256`、expertRange ≤ 32、`BS ∈ [384, 8192]`、k=8：

| totalLength | 模式 | tilingKey sortMode | kernel |
|---|---|---|---|
| < 21845 | ONE_CORE_GATHER_SORT | 2 | 单核 gather+排序融合 |
| ≤ 65536（=BS_MAX×k） | MULTI_CORE_GATHER_SORT | 3 | `moe_v3_gather_sort_multi_core.h` / `moe_v3_sort_multi_core_performance.h` |

仅在非量化 + SCATTER + COUNT + CountingSort 未启用时替换 tilingKey（:1184-1197）。UB 切分由特化 kernel 内部固化，不走通用阶段公式。

## 6. 各路径 UB 使用模型对比

| 维度 | 分阶段（通用） | 全载 | CountingSort 全载 | CountingSort cutOrigin | 性能模板 |
|---|---|---|---|---|---|
| UB 模型 | **复用**：每阶段独占整块 UB | **叠加**：全部 buffer 同时驻留 | **叠加** | **persistent + max(phase)** | 固化 |
| 排序方式 | 多核 4 路归并树，规模无上限 | 单核 Sort，≤ sortLoopMaxElement | 计数排序 O(n) | 计数排序，4096 chunk | 特化 |
| GM 中转 | sortWorkspace 等 ~24B/元素 | 无排序中转 | 核间计数归约表 | pairs + expertCount | 视 kernel |
| 阶段屏障 | SyncAll | 无 | 少 | Phase 间复用 UB | 无/少 |
| 触发 | 兜底 | dropless + 小 totalLength + UB 预算通过 | EP + 小专家数 + UB 估算通过 | 满足前置但 UB 不足全载 | shape/规格精确匹配 |
| tilingKey | 1xxxxxx | 21/22/23xxxxx | sortMode=2 | sortMode=3 | 2000000 / sortMode=2,3 |

## 7. 设计小结

1. **UB 切分的一切起点是 ubSize**（PlatformAscendC 动态查询，无硬编码）；VBS 公式额外扣 `aivNum×32B` 是 A3 唯一的多核辅助预留。
2. **分阶段路径是兜底**：七段 tiling 无条件计算，UB 复用模型让它对任意规模/列宽/量化模式都能工作（perLoopCols 折半、do-while 尾核保护、行列二维切分三重兜底）。
3. **快路径本质是 UB 使用模型的切换**：全载（叠加）→ CountingSort 全载（叠加）→ cutOrigin（persistent+max）→ 性能模板（固化），触发条件逐级放宽，性能逐级让步，最终回落到复用模型的分阶段路径。
4. 快路径全部通过 tilingKey 在编译期选择 kernel 模板，运行时零开销。
