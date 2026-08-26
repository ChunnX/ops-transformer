# moe_init_routing_v3 计数排序非全载（CutOrigin）阶段拆分设计

> 目标：将 arch35 计数排序非全载模板 `MoeV3CutOriginT`（单类内部封装 Phase A/B/C）
> 重构为「A/B 独立类 + 复用既有 Stage3/Stage5」，对齐 `moe_init_routing_v3_apt.cpp` 中原分阶段路径的组织方式。
> 拆分范围：**仅非全载 CutOrigin（tilingKey 500000 族：500000/500100/501000/510000/511000）**；
> **FullLoad（400000/401000）保持单类不动，不在本次范围内。**

---

## 1. 现状梳理

### 1.1 单类结构

`op_kernel/arch35/moe_v3_cut_origin_arch35_t.h` 的 `MoeV3CutOriginT<T>` 在 `Process()` 内顺序执行三阶段：

| 阶段 | 方法 | 职责 | 产出位置 |
|------|------|------|---------|
| A | `FilterAndCountChunked()` + `WriteExpertCountToWorkspace()` | 分块 filter expertIdx → pairs；scalar 计数 | pairs→GM workspace；expertCount→GM workspace |
| B | `ComputeGlobalOffset()` + `WriteExpertTokens()` | 跨核归约 + 前缀和 → 全局 offset；写 expertTokens | offset→UB persistent；expertTokens→GM |
| C | `GatherAndWriteChunked()` / `GatherAndWriteDroppad()` | 按 pairs 逐条 gather + 搬 x + 写 rowIdx | expandedX/expandedRowIdx/expandedScale→GM |

阶段间同步：A→B 一个 `SyncAll()`（`Process()` 内第 372 行）。

### 1.2 workspace 布局（当前 CutOrigin）

`op_host/moe_init_routing_v3_tiling_arch35.cpp::GetWorkspaceSize()` 第 247-269 行，
kernel `Init()` 第 278-282 行按序解析：

```
offset0: pairs       = filterNeedCoreNum * pairsPerCore     * sizeof(int32_t)
offset1: expertCount = filterNeedCoreNum * expertCountStride * sizeof(int32_t)
（另 + 16M 框架开销）
```

其中 `pairsPerCore = CeilAlign(max(filterPerCoreTokens, lastCoreTokens) * k, 8) * 2`，
`expertCountStride = CeilAlign(actualExpertNum, 8)`。

### 1.3 数据流依赖（拆分的关键约束）

```
        A ──pairs(GM)──────────────┐
        A ──expertCount(GM)──┐      │
                             ▼      ▼
        B ──offset(UB)──► [读 expertCount 算 offset]
                             │
                             ▼
        C ──读 pairs(GM) + offset(UB) ──► 搬 x
```

---

## 2. 拆分方案总览（v2：C 复用既有 Stage3/Stage5）

拆成**两个**独立模板类，entry 内串联后，Phase C 由既有 Stage3/Stage5 承接：

| 新类 | 承接原方法 | 职责 |
|------|-----------|------|
| `MoeV3CutOriginFilter<T>` | `FilterAndCountChunked` + `WriteExpertCountToWorkspace` + `InitGlobalMemory` 预填 rowIdx | Phase A |
| `MoeV3CutOriginOffset<T>` | `ComputeGlobalOffset` + **离散搬出（新）** + `WriteExpertTokens` | Phase B |

Phase C **不再新建类**，改为复用 `apt.cpp` 里现成的：

- **Stage3 `RowIdxGather`**（apt 461-494）→ 生成 `expandedRowIdx`
- **Stage5 全套搬出/量化类**（apt 536-739）→ 搬 x / 量化

> 新 A/B 两类的核数划分一致（都用 `filterNeedCoreNum_`，按 n 分核）；Stage3/Stage5 沿用其自身
> 按 `expertTotalCount` 的分核口径（`expertTokensCountTilingDataOp`/`gatherOutComputeParamsOp`）。
> 两者 blockDim 不同，靠各 stage 内 `if (blockIdx_ < ownNeedCoreNum)` 过滤 + entry 层 `SyncAll()` 对齐。

---

## 3. 关键改造点

### 3.1 离散搬出桥接（核心改动：Stage5 前把有序 id 输出到 GM）

组合 Stage5 的前提：Stage5 是**完全脱离的**，只从 GM 读「有序 id 列表」，逐条搬 x/量化
（`MoeGatherOut` 在 `Init` 第 166 行把 `expandedRowIdxGm_` 指向 `workspace + Align(n*k)`，
`ProcessGatherLoop` 第 240-251 行按「输出位置 i → 读 list[i] 得 flatIdx → 搬 x」顺序写）。

CutOrigin 的 offset（scatter 写起点）不能直接喂给 Stage5，必须由 B 阶段在 Stage5 之前多做一趟
**离散搬出（discrete scatter）**，把 pairs 转成有序 id 列表：

```
B 在 ComputeGlobalOffset() 之后，再遍历一遍本核 pairs：
    for each (flatIdx, expertIdx) in pairs:
        pos = offsetLocal[expertIdx]++          // 与 C 里算 newPos 完全同一段逻辑
        sortedRowIdxGm[pos] = flatIdx           // 写到 workspace + Align(n*k) 区
```

**为什么叫「离散搬出」**：每个 flatIdx 被写到不连续的 GM 位置 `pos = offset[expertIdx]++`，
与 Stage5 的「顺序 gather」方向相反——本步离散 scatter 写索引列表，Stage5 顺序读列表、顺序搬 x。

**解耦边界（关键）**：pairs 与 offset 是 **A/B 内部临时量**，不跨越 A/B → Stage5 边界。
Stage5（以及 Stage3）只消费 GM 上的有序 id 列表（`sortedRowIdx`）+ `expertTotalCount`；
**pairs 与 offset 不需要再被 Stage5 使用**。因此：

- 跨阶段产物：
  - **非 dropPad**：两个 —— `sortedRowIdx`（有序源 flatIdx 列表，`workspace + Align(n*k)`）与
    `expertTotalCount`（`workspace + 2*Align(n*k) + Align(actualExpertNum)`）。
  - **dropPad（500100，非量化 GATHER）**：`sortedRowIdx` 仍然产出（即下游 `expandDstToSrcRow`），
    `expertTotalCount` **不写**，改为补 `expandedExpertIdx`（`workspace` 起始区）+ `expertIdxValue`
    （同一 `workspace + 2*Align(n*k) + Align(actualExpertNum)` 槽位，`coreNum_*2`），详见 §6.4。
- pairs（offset0）与 expertCount（offset1）在 B 结束即失活，workspace 可考虑后续复用（非必需；
  dropPad 下 `workspace` 起始区被复用为 `expandedExpertIdx`）。

> 命名说明：这里写的是 `sortedRowIdx`（**源 flatIdx 有序列表**，即 Stage5 读的「id」），
> 不是 `sortedExpertIdx`（expert id 列表，offset 0，只被 Stage2 ExpertTokensCount 消费，CutOrigin 不复用 Stage2）。
> 该区正是回退路径 sort 阶段 `expandedRowIdxGm`（GATHER 模式）与 Stage5 `CopyExpertIn` 读的同一块。

`offsetLocal`（即 `expertCountLocal`）已含全局前缀，天然保证稳定有序。

**代价**：B 多一趟 pairs 遍历。这等价于把 C 里「算 newPos」那部分前移，只是 scatter 目标
从「expandedX 搬 x」改成「sortedRowIdx 写 flatIdx」，不是纯新增遍历。

**收益**：offset 不再需要落盘 GM（原 v1 方案的 offset 区取消），offset 全程留在 B 的 UB 内。

### 3.2 shortPath 优化取舍（拆分代价，需明确接受）

单类里 `shortPath_ = (coreEntries_<=2048 && ep_==1)` 时，gathered pairs 留在 UB persistent 区从 A 直接带到 C，
省一次 workspace 往返（`Init()` 第 287 行、`Process()` 相关分支）。

拆分后 B 必须从 GM 读 pairs，**shortPath 退化为普通 workspace 路径**，small 场景多一次 pairs 写读。
这是结构对齐的必然代价，功能不受影响。

### 3.3 UB 校验逻辑变化

当前 tiling（`ComputeArch35CountingSortCutOriginTiling()` 第 2248 行）用
`persistentSize + max(phaseA, phaseB, phaseC) <= availUbSize_` 校验单类总 UB。

拆分后 Filter / Offset 各自独立 `InitBuffer`，各自的 UB 上限均为 `availUbSize_`（196608）：

| 类 | UB 构成 |
|----|---------|
| Filter | `expertCountAlign`（expertCountLocal，scalar 计数用）+ `phaseASize`（filter 临时区，**去掉 shortPath 条件**，gatheredExpert/gatheredIdx 恒计入） |
| Offset | `expertCountAlign`（offset 数组）+ `phaseBSize`（totalCount/prefixSum/batchBuf）+ 离散搬出临时区 |

tiling 侧改为分别校验 `expertCountAlign + phaseX <= availUbSize_`（X∈{A,B}），
比当前 `persistent + max(...)` 更宽松也更直观。现有 `phaseASize/phaseBSize` 推导可复用。

### 3.4 无效核 guard

单类 `Process()` 开头的 `if (blockIdx_ >= filterNeedCoreNum_) { SyncAll(); return; }`（第 363 行）
拆到两个类的 `Process()` 开头各保留一份（每个含 `SyncAll()` 的阶段，无效核都必须参与同步，避免死锁）。

---

## 4. 两个新类详细设计

### 4.1 `MoeV3CutOriginFilter<T>`（Phase A）

```cpp
__aicore__ inline void Init(GM_ADDR expertIdx, GM_ADDR expandedRowIdx,
                            GM_ADDR workspace,
                            const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
__aicore__ inline void Process();
```

- 解析 tiling：`n_/k_/expertStart_/expertEnd_/rowIdxType_/ep_/filterNeedCoreNum_/filterPerCoreTokens_/filterChunkSize_` 等（同单类 `Init` 的 filter 相关部分）。
- workspace 解析：`pairsWorkspaceGm_ = workspace`，`expertCountWorkspaceGm_ = workspace + filterNeedCoreNum_*pairsPerCore_`。
- `InitGlobalMemory` 预填 `expandedRowIdx=-1`（GATHER 模式，含 `SetWaitFlag<MTE3_MTE2>` + `SyncAll`）逻辑移入此处。
- `Process()`：`FilterAndCountChunked()` + `WriteExpertCountToWorkspace()`，末尾 `SyncAll()`。
- UB 布局：`expertCountLocal`（expertCountAlign）+ Phase A 临时区（expertIdx/expertIdxFp32/3 mask/flatIdx/gatheredExpert/gatheredIdx）。

### 4.2 `MoeV3CutOriginOffset<T>`（Phase B：offset + 离散搬出）

```cpp
__aicore__ inline void Init(GM_ADDR expertTokens,
                            GM_ADDR workspace,
                            const MoeInitRoutingV3Arch35TilingData *tiling, TPipe *pipe);
__aicore__ inline void Process();
```

- workspace 解析：
  - `expertCountWorkspaceGm_`（读，offset0 + filterNeedCoreNum_*pairsPerCore_）
  - `pairsWorkspaceGm_`（读，offset0）
  - `sortedRowIdxGm_ = workspace + Align(n*k)`（**写**，供 Stage3/Stage5 复用；dropPad 下即下游 `expandDstToSrcRow`）
  - `expertTotalCountGm_ = workspace + Align(n*k)*2 + Align(actualExpertNum)`（**写**，**仅非 dropPad**，Stage3/Stage5 分核用）
  - dropPad 下同一槽位为 `expertIdxValueGm_`（`coreNum_*2`，每核 `[lastExpertId, lastTokenCount]`，供 `RowIdxGatherDropPad`）
- `Process()`：
  1. `ComputeGlobalOffset()`（读全核 expertCount → 算 offset → 写 `expertCountLocal`）
  2. **离散搬出 `ScatterToSortedRowIdx()`**（dropPad 也必须走）：遍历本核 pairs，`pos = expertCountLocal[expertIdx]++`，离散 scatter 写 `sortedRowIdxGm_[pos] = flatIdx`（供 Stage3/Stage5 从 GM 读；pairs/offset 不再向下传递）
  3. 写跨阶段产物：
     - 非 dropPad：写 `expertTotalCountGm_[0]`（有效 token 总数，Stage3/Stage5 分核用）
     - dropPad：写 `expandedExpertIdx` + `expertIdxValueGm_`（详见 §6.4）
  4. `WriteExpertTokens()`（若 flag）
  5. 末尾 `SyncAll()`
- **offset 统一为有效序列偏移**：`ComputeGlobalOffset` 中 dropPad 与非 dropPad 一致，
  `offset[e] = 全局前缀 + 本核前缀`（`[0, expertTotalCount)` 连续）。原单类 dropPad 分支的 padded 槽位偏移
  `(expertStart_+e)*expertCapacity_ + clampedPrefix` 已删除——padded 槽位改由 Stage3 `RowIdxGatherDropPad`
  依据 `expandedExpertIdx` + 跨核 `tokenCount` 在 CopyOut 中逐位推算。
- UB 布局：`expertCountLocal`（offset）+ totalCount/prefixSum/batchBuf（沿用单类 Phase B 布局公式）。
- 分核对齐：写 `sortedRowIdx`/`expandedExpertIdx` 时 pos 为全局输出位置（连续），下游 dropPad 按自身 `perCoreRows` 分块读，见 §6.4。

---

## 5. entry 函数改动

替换 `moe_init_routing_v3_apt.cpp` 第 361-383 行的单类调用。A/B 完成后**不再 `return`**，
复用既有 Stage3/Stage5 的 dispatch：

```cpp
} else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_CS_CUTORIGIN_UNQUANTIZED) ||
           TILING_KEY_IS(MOE_INIT_ROUTING_V3_CS_CUTORIGIN_UNQUANTIZED_S) ||
           TILING_KEY_IS(MOE_INIT_ROUTING_V3_CS_CUTORIGIN_UNQUANTIZED_D) ||
           TILING_KEY_IS(MOE_INIT_ROUTING_V3_CS_CUTORIGIN_STATIC_QUANT) ||
           TILING_KEY_IS(MOE_INIT_ROUTING_V3_CS_CUTORIGIN_STATIC_QUANT_S)) {
    constexpr bool isUnquantOrStatic = IsSameType<DTYPE_X, bfloat16_t>::value ||
                                       IsSameType<DTYPE_X, half>::value ||
                                       IsSameType<DTYPE_X, float>::value;
    if constexpr (isUnquantOrStatic) {
        // Phase A: Filter
        TPipe filterPipe;
        MoeV3CutOriginFilter<DTYPE_X> filterOp;
        filterOp.Init(expertIdx, expandedRowIdx, userWS, t, &filterPipe);
        filterOp.Process();
        filterPipe.Destroy();
        SyncAll();

        // Phase B: Offset + 离散搬出 → sortedRowIdx + expertTotalCount
        TPipe offsetPipe;
        MoeV3CutOriginOffset<DTYPE_X> offsetOp;
        offsetOp.Init(expertTokensCountOrCumsum, userWS, t, &offsetPipe);
        offsetOp.Process();
        offsetPipe.Destroy();
        SyncAll();
    }
    // 之后不再 return，落入下方既有 Stage3(RowIdxGather) + Stage5(搬出/量化) 的 dispatch。
}
```

**关键接线**：Stage3/Stage5 的 dispatch（第 461-739 行）目前以 `TILING_KEY_IS(SORTONECORE_*/SORTMULTICORE_*)`
为条件，需要让其条件**同时命中 CutOrigin 的 key**（或把 Stage3/Stage5 抽成公共函数由两处调用），
否则 CutOrigin 路径会从 537 行的 `if` 链一路落到末尾不执行。

---

## 6. 桥接：C 阶段替换为 Stage3+Stage5 的契约对齐

### 6.1 复用关系

| CutOrigin key | 桥接产出 sortedRowIdx 去向 | Stage3 | Stage5 |
|---------------|---------------------------|--------|--------|
| 500000（非量化 GATHER） | `workspace + Align(n*k)` | `RowIdxGather` 生成 expandedRowIdx | `MoeGatherOut` |
| 501000（非量化 SCATTER） | 直接写 expandedRowIdx 输出 | 跳过（scatter 时 Stage1 已写） | `MoeGatherOut`（SCATTER 分支读 expandedRowIdx） |
| 510000/511000（静态量化） | `workspace + Align(n*k)` / expandedRowIdx | 同非量化 | `MoeV3GatherStaticQuant` |
| 500100（dropPad） | `workspace + Align(n*k)`（即 `expandDstToSrcRow`，dropPad 也走离散搬出） | `RowIdxGatherDropPad` | `MoeV3GatherOutDropPad`（非量化） |

### 6.2 为什么 Stage5 不能直接吃 A/B 的 offset

Stage5 是**顺序写**模型：输出位置 i = 迭代序，源 = `sortedRowIdx[i]`。
A/B 产出的是**源序 pairs + scatter offset**，方向相反。离散搬出（§3.1）就是两者的桥梁，
本质是计数排序的「第二趟稳定放置」，把「scatter 写 x」改写成「scatter 写 flatIdx」。
Stage5 完全脱离 pairs/offset，只消费 GM 上的有序 id 列表。

### 6.3 收益

完全复用 Stage5 全部搬出/量化变体：非量化（`MoeGatherOut`）、`MoeGatherOutMxfp8`/`MoeV3GatherOutMxfp8`、
`MoeGatherOutMxFp4`、静态量化（`MoeV3GatherStaticQuant`）、动态量化（`MoeGatherOutDynamicQuant` int8/int4）、
MXFP8Quant、FP8GroupQuant、FP8PerBlockQuant、Hif8（cast/pertensor/pertoken）、Mxfp4Quant、FP8GroupAmaxQuant 等。
**零新增搬出/量化代码。**

### 6.4 dropPad（500100）专用桥接：离散搬出仍然需要，另补两个 GM 产物

**契约约束**（`apt.cpp` L176-179）：`ValidateQuantMode/ValidateRowIdxType` 强制 **dropPad ⇒ quant_mode=-1 且 GATHER**，
因此 CutOrigin 里只有 **非量化 GATHER 的 dropPad key（500100）可达**，不存在 dropPad+SCATTER 组合。

由此可得：

1. **离散搬出不能跳过**：500100 也是 GATHER，`MoeV3RowIdxGatherDropPad`（Stage3）与 `MoeV3GatherOutDropPad`（Stage5）
   都读 `workspace + Align(n*k)` 区的 `expandDstToSrcRow`（row_idx_gather_droppad L278 / gather_out_droppad L173），
   该区正是 Phase B 离散搬出写 `sortedRowIdx` 的位置 —— **dropPad 的 dst→src 有序 id 就由离散搬出承载**。
   `expertIdxValue` 是 dropPad 的**补充**产物，不是替代离散搬出。

2. **dropPad 下 Phase B 需多写两个 GM 产物**（非 dropPad 不消费，因此只在 `dropPadMode_ == DROP_PAD_MODE` 分支写）：

   | 产物 | 位置 | 内容 | 消费者 |
   |------|------|------|--------|
   | `expandedExpertIdx` | `workspace` 起始区（pairs/expertCount 在 B 结束后原地覆写） | 每个输出位置 token 的 expert id | `RowIdxGatherDropPad` CopyIn（L275） |
   | `expertIdxValue` | `workspace + 2*Align(n*k) + Align(actualExpertNum)`，`coreNum_*2` | 每核 `[lastExpertId, lastTokenCount]` | `RowIdxGatherDropPad.AssistInit`（L84-95）跨核推算 token 偏移 |

   - **`expandedExpertIdx` 不从 pairs 重读，而是由 `totalCountLocal`（全量 expert 计数）推导**：
     每个专家占全局位置 `[globalPrefix[e], globalPrefix[e]+totalCountLocal[e])`，每 Phase B 核按本核全局输出区间
     `[globalStart, globalStart+coreTotalPairs)` 内的每专家连续 run 做 `Duplicate+DataCopyPad` 写入。
     **不能「离散搬出时顺带写」**：`expandedExpertIdx` 位于 `workspace` 起始区（= pairs 区），
     若在遍历 pairs 时同步写 `expandedExpertIdx[pos]`，对 `blockIdx==0`（pairs 区与全局位置区重叠）会覆盖未读 pairs，
     造成数据竞争；推导法完全不读 pairs，天然规避该冲突。无效尾区 `[expertTotalCount, n*k)` 由 `blockIdx 0` 填 `-1`
     （与所有核的有效写 `[0, expertTotalCount)` 无重叠，无需跨核同步）。
   - **`expertIdxValue` 由 `blockIdx 0` 一次性计算**：依据 `totalCountLocal` 与 `srcToDstDropPadParamsOp`
     （`perCoreRows`/`lastCoreRows`/`needCoreNum`）对每个下游核段推「段末有效 token 所属专家及其段内计数」，
     写 `expertIdxValueGm_[i*2..]`，无需跨核 reduce、不依赖 pairs/scatter。
   - **expertTotalCount 在 dropPad 下不写**：同一 GM 槽位（`workspace + 2*Align(n*k) + Align(actualExpertNum)`）
     在 dropPad 下是 `expertIdxValue`（Stage2 注释「DropPad模式不保留expertTotalCountGm空间」）。

3. **分核对齐约束**：dropPad 下游按 `srcToDstDropPadParamsOp.perCoreRows` / `gatherOutComputeParamsOp.perCoreIndicesElements`
   分核读 `expandedExpertIdx`/`expandDstToSrcRow`（`blockIdx_*perCoreRows`），Phase B 按 `filterNeedCoreNum` 分核写。
   Phase B 写这两个区时需保证**全局连续**（pos 即全局输出位置），下游按自身分块读；跨阶段需 `SyncAll()` 对齐，
   与 §6 开头「blockDim 不同，靠各 stage 内 `if (blockIdx_ < ownNeedCoreNum)` 过滤 + entry 层 `SyncAll()`」一致。
   **B 内部还需一次栅栏**：`expandedExpertIdx` 写在 `workspace` 起始区（= pairs 区），而各核 `ScatterToSortedRowIdx`
   仍在并行读自己的 pairs；某核提前进入 `WriteExpandedExpertIdx` 时，其写的全局位置可能落在**其他核未读的 pairs** 上
   （例如 `maxCoreEntries=64, total_0=50,total_1=60,total_2=20` 时，核 2 写 `[110,130)` 命中核 1 的 flatIdx 读区 `[128,188)`）。
   因此 `Process()` 在「离散搬出」与「写跨阶段产物」之间插一次 `SyncAll()`（所有核调用次数一致，避免栅栏错配）。

---

## 7. 同步与 cache 一致性

| 边界 | 同步手段 | 说明 |
|------|---------|------|
| A→B | `SyncAll()` | B 读所有核的 expertCount + pairs，必须等 A 全部写完 |
| B 内（离散搬出→写 expandedExpertIdx） | `SyncAll()` | 各核 `ScatterToSortedRowIdx` 读毕 pairs 后，才能原地覆写 `workspace` 起始区（expandedExpertIdx），避免覆盖他核未读 pairs，见 §6.4 ③ |
| B→Stage3/Stage5 | `SyncAll()` | Stage3/Stage5 读 B 写出的 sortedRowIdx/expertTotalCount（dropPad 另含 expandedExpertIdx/expertIdxValue） |
| sortedRowIdx 落盘后 | `DataCacheCleanAndInvalid`（Stage3/Stage5 读前） | B 用 MTE3 写 GM，读者需先失效 cache（对齐 Stage3/Stage5 读 pairs/sortedExpertIdx 的既有做法） |

> `SyncAll()` 放在 entry 层（每类 `Process()` 之外），因为每类的 `TPipe` 在 `Destroy()` 后才轮到下一类。

---

## 8. tiling 侧改动

1. `GetWorkspaceSize()`（第 247-269 行）：CutOrigin 分支追加 `sortedRowIdx` 区与 `expertTotalCount` 区，
   与 fall-through 布局对齐（`Align(n*k)` 偏移的 n*k int32 + `2*Align(n*k)+Align(actualExpertNum)` 的 1 int32），
   并确保与既有 pairs/expertCount 两块不重叠。`mode==1`（FullLoad）不分配，保持两块布局不动。
   注意 dropPad（500100）下同一尾区是 `expertIdxValue`（`coreNum_*2`），需按 `max(1, coreNum_*2)` 分配；

2. `ComputeArch35CountingSortCutOriginTiling()`（第 2146 行）：
   - `persistentSize` 中 shortPath 的 gathered 双区（第 2170-2172 行）删除；
   - 末段校验由 `persistentSize + max(phaseA,phaseB,phaseC) <= availUbSize_`
     改为分别校验 `expertCountAlign + phaseX <= availUbSize_`（X∈{A,B}）。

3. **新增 tiling 字段填充**：CutOrigin 路径需填 `expertTokensCountTilingDataOp`（needCoreNum/perCoreElements，
   Stage3 分核用）与 `gatherOutComputeParamsOp`（perCorePerLoopIndicesElements/colsLoops/perLoopCols/...，
   Stage5 用）。这两个 struct 目前只有 fall-through 路径填。

---

## 9. 风险与验证计划

### 风险

| 风险 | 缓解 |
|------|------|
| shortPath 优化丢失（small 场景多一次 workspace 往返） | 功能等价；如需保留可在 B 类内部重建 shortPath |
| sortedRowIdx 落盘引入 cache 一致性问题 | Stage3/Stage5 读前 `DataCacheCleanAndInvalid` |
| UB 校验口径变化导致某些 shape 回退（countingSortMode=0） | 分别校验 A/B 两阶段，比单类更宽松；用现有 CSV 全量回归确认可达性不变 |
| Stage3/Stage5 dispatch 条件不命中 CutOrigin key，导致 C 阶段空跑 | entry 接线时确保 dispatch 条件覆盖 CutOrigin key（或抽公共函数） |
| expandedRowIdx 语义翻转（GATHER 预填 -1 vs RowIdxGather 的 scatter 方向） | 见下 |
| FullLoad 路径被误伤 | 拆分仅动 CutOrigin（mode==2）；FullLoad 类与 workspace 两块布局保持不动 |

### expandedRowIdx 语义决策点

- CutOrigin GATHER 现在：`Init` 预填 `-1`，C 写 `expandedRowIdx[newPos] = srcRow`（输出位置 → 源行）。
- fall-through Stage3 `RowIdxGather`（`ComputeSimt` 第 53-54 行）：写 `expandedRowIdx[srcFlatIdx] = outPos`（源 token → 输出位置）。

方向相反。需在实现前拍板：**(a) 接受翻转**（统一到 fall-through 约定，Stage3+Stage5 本来就是配套的），
还是 **(b) 只复用 Stage5 搬 x/量化，expandedRowIdx 仍在 A/B 里按 CutOrigin 原约定自己写**。

### 验证

1. **编译**：`moe_v3_cut_origin_arch35_t.h` 拆出两个新头文件后，`apt.cpp` 改 include 与调用点，全量编译通过。
2. **可达性回归**：用 `moe_init_routing_v3_smallT_cs_cases.csv`、`moe_init_routing_v3_smallCols_largeT_cs.csv`
   的 57 个 CutOrigin 用例，确认拆分前后 tilingKey 均落到 500000/510000（不因 UB 校验改动而回退）。
3. **精度**：拆分是纯结构重构（无计算逻辑变化），精度应严格一致；跑 allclose（atol=1e-5, rtol=1e-3）确认。
4. **同步正确性**：重点回归多核场景（filterNeedCoreNum>1）无死锁、无数据竞争。

---

## 10. 文件清单（预计改动）

| 文件 | 改动 |
|------|------|
| `op_kernel/arch35/moe_v3_cut_origin_filter_arch35.h`（新增） | Phase A 类 |
| `op_kernel/arch35/moe_v3_cut_origin_offset_arch35.h`（新增） | Phase B 类（offset + 离散搬出） |
| `op_kernel/arch35/moe_v3_cut_origin_arch35_t.h` | 拆空/删除（迁移到两个新类） |
| `op_kernel/moe_init_routing_v3_apt.cpp` | include + entry 调用点改为 A/B 串联 + 复用 Stage3/Stage5 dispatch |
| `op_host/moe_init_routing_v3_tiling_arch35.cpp` | GetWorkspaceSize 加 sortedRowIdx/expertTotalCount 区；CutOrigin UB 校验改分别校验；填 expertTokensCountTilingDataOp/gatherOutComputeParamsOp |

（可选：`moe_init_routing_v3_arch35_tiling_def.h` 无需新增字段。）
