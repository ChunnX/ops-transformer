# MoeInitRoutingV3 A3 Tiling 切分策略

> 代码位置：`op_host/moe_init_routing_v3_tiling.cpp` / `moe_init_routing_v3_tiling.h` / `moe_init_routing_v3_tiling_base.cpp`
> 适用平台：A3/A2（非 950 平台，fallback 模板，优先级 10000）；950（A5）走 `moe_init_routing_v3_tiling_arch35.cpp`

## 1. 整体框架

### 1.1 入口与模板注册

- `moe_init_routing_v3_tiling_base.cpp:22`：`IMPL_OP_OPTILING(MoeInitRoutingV3)` 注册 tiling 入口，`TilingRegistry::DoTilingImpl` 按平台分发。
- `moe_init_routing_v3_tiling.cpp:1633`：`REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV3, MoeInitRountingV3TilingBase, 10000)` —— A3 实现是**兜底模板**。
- 平台参数在 `TilingPrepareForMoeInitRountingV3`（`tiling_base.cpp:27`）中通过 `PlatformAscendC` **动态获取** `aivNum` 与 `ubSize`，无硬编码。

### 1.2 执行流程（TilingBaseClass 七步）

```
GetPlatformInfo          → aivNum / ubSize，写入 coreNum
GetShapeAttrsInfo        → 4 输入 / 4 输出 shape + 9 个 attr 指针
DoOpTiling               → CheckAttr → CheckInputShape → CheckOutShape → CheckDtype
                         → 7 个 Tiling4Xxx 阶段切分
                         → IsFullLoad / ComputeCountingSortTiling 快路径判断
DoLibApiTiling           → 空（未使用高阶 API）
GetTilingKey             → 按场景编码 tilingKey
GetWorkspaceSize         → workspace 计算
PostTiling               → SetBlockDim(aivNum)，SaveToBuffer
```

### 1.3 核心切分量

| 量 | 含义 |
|---|---|
| `totalLength_ = n * k` | token 数 × topK，排序/搬运的总元素数 |
| `sortLoopMaxElement` | 单核单轮排序最大元素数，由 UB 推导：`(ubSize - aivNum*32B) / (4*2*4) / 32 * 32`，32 对齐（`tiling.cpp:1144`） |
| `aivNum` | AIV 核数，全量并发的上限 |

## 2. 六个计算阶段的切分

每个阶段对应 kernel 侧一个类，均遵循「**核间均分 + 核内按 UB 余量分 loop，尾核/尾轮单独记录**」的通用模式，tiling 字段统一为 `perCoreXxx / perCoreLoops / perCorePerLoopXxx / perCoreLastLoopXxx / lastCoreXxx / lastCoreLoops / lastCorePerLoopXxx / lastCoreLastLoopXxx`。

### 2.1 VBS（Vector Bitonic Sort，排序）— `Tiling4VBSCompute` :1317

| 场景 | 条件 | 策略 |
|---|---|---|
| 单核排序 sortMode=0 | `totalLength ≤ sortLoopMaxElement` | 1 个核排全部元素 |
| 多核排序 sortMode=1 | 否则 | `needCoreNum = 4^ceil(log4(ceil(totalLength / sortLoopMaxElement)))`，再 `min(aivNum)` |

要点：

- **核数取 4 的幂**（`Tinlig4VBSMultiCoreCompute` :1265）：与后续 4 路归并（`MRG_LIST_NUM=4`）树形结构对齐。
- 每核元素数先按 32 对齐**向下**取整，尾核拿走剩余；若尾核元素超过向上对齐值，则改为向上对齐并重算核数（:1276-1284）。
- `do-while` 保护（:1287-1309）：逐次将 `perCoreElements` 减 32 重算，直到 `lastCoreLastLoopElements > 0`，保证尾核尾轮不为空。

### 2.2 VMS Middle（中间归并）— `Tiling4VMSMiddleCompute` :1334

- VBS 核数 ≤ 4：无需中间归并，`needCoreNum = 0`。
- VBS 核数 > 4：`needCoreNum = ceil(vbsNeedCoreNum / 4)`，每核归并 4 个有序段。

### 2.3 SortOut（最终归并输出）— `Tiling4SortOutCompute` :1346

- 仅设置 `oneLoopMaxElements = mrgSortListMaxElement = 1504`，单核完成最终归并。

### 2.4 ExpertTokensCount（专家 token 统计）— `Tiling4ExpertTokensCountCompute` :1352

- `n*k` 均分到 `aivNum` 个核。
- UB 内先扣除专家统计缓冲区（`expertNumElement × (int32×2 + int64)`，KEY_VALUE 模式专家数 ×2）+ 32B，剩余空间换算出 `maxElementsPerLoop`，再拆 perCore / lastCore 的 loops。

### 2.5 SrcToDst（dropless 按排序结果搬运）— `Tiling4SrcToDstCompute` :1527

- UB 预留 `ASSIST_NUM(256)` 个 int32 及对齐空间后，每元素按 `32B + 4B` 估算 → `perLoopMaxElements`。
- 标准的均分核 + 核内分 loop 套路。

### 2.6 DropPad 搬运（容量裁剪模式）— 两个变体

**非动态量化** `Tiling4SrcToDstDropPadCompute` :1399：

- 行（totalLength）均分到核；`rowSize + colSize < ubSize` 时一行全载（`SetGatherTilingData`，单 loop）。
- 否则行列二维切分：列上限 `MAX_COLS_ONE_LOOP = 16376`；行上限由 UB 扣除列缓冲后推算；cols < 上限时优先压行，perCoreRows 小于行上限时反向放大列块。

**动态量化** `Tiling4SrcToDstDropPadDynamicCompute` :1491：

- 结构相同，按 `DYNAMIC_QUANT_SRC_TO_DST_BUFFER = 15` 个 int8 buffer + 64B scale 估算，列上限 `MAX_COLS_DYNAMIC_QUANT = 6144`。

### 2.7 GatherOut（按专家聚簇搬 x 行 + 量化）— `Tiling4GatherOutCompute` :1562

- 核间：indices（totalLength）均分。
- 列方向：UB 扣除 `Align(perLoopCols) × colMultiple` 后算每 loop 可处理的 indices 数。`colMultiple` 按量化模式区分：
  - 非量化：`2 × dtypeSize`（双缓冲）
  - 动态量化：`DYNAMIC_QUANT_COLS_BUFFER = 21`，行方向 rowMultiple=4
  - 静态量化：`int8×2 + fp32 + int16 + dtypeSize×2`
- **若每 loop 可处理 indices ≤ 0，则 `perLoopCols` 不断折半**（:1588-1595 while 循环）直到为正 —— 宽 cols 场景的兜底。

## 3. 三条特殊路径

### 3.1 全载（FullLoad）— `IsFullLoad` :760

UB 预算逐项扣除：

```
remainUb = ubSize - sortSpace(6缓冲) - rowIdxSpace(3) - expertSpace(2) - 1KB
         - gatherSpace（按 perCoreTokens 行 × cols）
         - 量化空间（非量化:0 / 静态:11 buffer / 动态:13 buffer + 64B）
```

- `remainUb > 0` → 全载。`dropPadMode=1` 或 `totalLength > sortLoopMaxElement` 直接排除。
- EP 场景（expertStart/End 非全域）额外支持 **gatherFirstFullload**（先 gather 剔除无效专家再排序，:749）。
- tilingKey：非量化 2100000 / 静态 2200000 / 动态 `2300000 + ep×10000 + smoothType×1000`。

### 3.2 CountingSort（计数排序）— `ComputeCountingSortTiling` :821

非全载时尝试启用，前置条件：`n ∈ [256, 1M]`、`expertNum ≤ 1024`、非动态量化、dropPad 或 actualExpertNum×2 ≤ expertNum 等（`ShouldEnableCountingSortByShape` :809）。三态标志 `countingSortFullLoad_`：

| 值 | 含义 | 切分 |
|---|---|---|
| 1 | 全载计数排序 | n 均分到核，UB 逐项预算（`EstimateCountingSortFullLoadUB` :872）；workspace = 核数 × expertStride × 4B |
| 0 | cutOrigin 分块（:902） | Phase A：4096 chunk 过滤；Phase B：多核计数归约；Phase C：搬运，**perLoopCols 由剩余 UB 动态推导**（上限 64KB）。量化场景列循环 >1 时禁用（置 -1） |
| -1 | 不启用 | — |

### 3.3 性能特化模板

- **X_1_7168 单 token 模板**（:996）：shape 精确匹配 `(1,7168)/(1,8)/(256,7168)`、bf16、动态量化、KEY_VALUE 时，`aivNum = totalLength = 8`，tilingKey = 2000000。
- **GatherSort 性能模式**（`GetPerformanceMode` :1022）：`BS ∈ [384, 8192]`、k=8、expertNum=256 时：
  - `totalLength < 21845` → ONE_CORE_GATHER_SORT（tilingKey sortMode=2）
  - `totalLength ≤ 65536` → MULTI_CORE_GATHER_SORT（tilingKey sortMode=3）

## 4. TilingKey 编码

| 场景 | TilingKey |
|---|---|
| 通用 | `1000000 + sortMode×100000 + (quantMode+1)×10000 + rowIdxType×1000 + dropPadMode×100` |
| 全载非量化 / 静态 / 动态 | 2100000 / 2200000 / `2300000 + ep×10000 + smoothType×1000` |
| X_1_7168 性能模板 | 2000000 |
| 空 tensor（n=0） | 3000000（DoOpTiling 提前返回，blockDim=1） |

## 5. Workspace 与 BlockDim

`GetWorkspaceSize` :1206：

```
sortWorkspace     = totalLength × 2 × 3 × 4B     （排序临时空间）
coreSyncWorkspace = coreNum × 32 × 2             （多核同步）
scatterWorkspace  = totalLength × 4B
expertTokensCount = actualExpertNum × 4B
+ 16MB 基础空间
+ 动态量化: aivNum × cols × 4B
+ dropPad : aivNum × 2 × 4B
与 countingSortWorkspaceSize 取 max
```

`PostTiling` :1238：`SetBlockDim(aivNum)`（空 tensor 为 1），tiling 数据 `SaveToBuffer`。

## 6. 设计思想总结

1. **排序段（VBS）**：按 UB 决定的单轮容量切核，核数取 4 的幂以适配 4 路归并树；尾核合法性用 do-while 迭代保证。
2. **统计/搬运段**：一律「核间均分 + 核内按 UB 余量分 loop」，UB 预算逐项扣除，余量换算为每 loop 元素数。
3. **GatherOut 列方向**：放不下时对折降列宽，保证任意 cols 都能运行。
4. **快路径**：全载 / 计数排序（全载 + cutOrigin 两级）/ 性能模板，全部由 TilingKey 区分 kernel 模板，运行时零开销选择。
5. **无硬编码平台参数**：aivNum、ubSize 均来自 PlatformAscendC，常量仅为算法相关（对齐 32B、归并路数 4、列上限等）。
