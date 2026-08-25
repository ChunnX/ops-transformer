# CountingSort 模板 Phase A/B 与阶段5（搬运量化）解耦方案分析

## 1. 背景

MoE InitRoutingV3 算子有两套执行路径，由 host 侧 tiling 根据 shape 与量化模式选择：

| 路径 | 形态 | tilingKey 前缀 | 说明 |
|------|------|----------------|------|
| 原分阶段流程 | 5 个独立 stage 串行 | `10000000` / `11000000` | 排序 → 计数 → rowIdx → 搬运量化 |
| CountingSort 性能模板 | 单 kernel 内 Phase A/B/C 融合 | `400000`（FullLoad）/ `500000`（CutOrigin） | 用计数排序替代显式排序 |

`apt` 入口 `moe_init_routing_v3_apt.cpp` 中，CountingSort 模板在分阶段流水**之前**提前 `return`（`moe_init_routing_v3_apt.cpp:346-383`），走融合 `Process()`；未命中时落入后面的 5 阶段流水（`:385-739`）。

## 2. 两套路径的流程对照

### 2.1 原分阶段流程（5 stage）

```
阶段1 排序         MoeSortOneCore / MoeSortMultiCore        → sortedExpertIdx + sortedRowIdx
阶段2 TokensCount  ExpertTokensCount                       → expertTokensCountOrCumsum
阶段3 RowIdxGather RowIdxGather                            → expandedRowIdx
阶段4 TopkWeightOut MoeV3TopkWeightOut                     → (V3 恒不执行, isInputTopkWeight=0)
阶段5 搬运量化      MoeGatherOut / *Quant 变体              → expandedX (+ expandedScale)
```

### 2.2 CountingSort 模板（Phase A/B/C 融合）

两个子模板骨架一致（`moe_v3_cut_origin_arch35_t.h:361-392`、`moe_v3_counting_sort_full_load_unquantized.h:184-210`）：

```
Phase A  过滤 + 计数                    → 写本核 expertCount 到 workspace
SyncAll  (跨核归约前置同步)
Phase B  跨核归约 + 偏移计算             → expertCountLocal[e] = 本核在专家 e 的全局写起始偏移(seed)
Phase C  聚合搬出 / 逐 pair 搬出 + 量化  → expandedX + expandedRowIdx
```

| 阶段 | FullLoad（模板4） | CutOrigin（模板5） |
|------|------------------|-------------------|
| Phase A | `LoadExpertIdx` + `LoadXBackground`(x 后台全载) + `FilterAndCount` | `FilterAndCountChunked`(chunk 分块过滤) |
| Phase B | `ComputeGlobalOffset`(allCoreExpertCount 一次全读) | `ComputeGlobalOffset`(batchBuf 分批加载) |
| Phase C | `BucketByExpert` + `GatherAndWriteByExpert`(专家分桶聚合搬出) | `GatherAndWriteChunked` / `GatherAndWriteDroppad`(双缓冲逐 pair) |

## 3. Phase A/B 与阶段5 合并的依赖分析

阶段5 搬运量化本质是三段流水：

```
① 源加载 MTE2: x[srcRow] → UB         依赖 srcRow = flatIdx/k     ← Phase A 已知
② 量化 V:      UB → quant             依赖源数据 + scale          ← 不依赖偏移
③ 写回 MTE3:   UB → expandedX[newPos]  依赖 newPos = seed+slot     ← Phase B 已知
```

关键结论：

- **① 源加载、② 量化不依赖 Phase B 偏移**，可提前与 Phase A/B overlap。
- **③ 写回依赖偏移 seed**，必须等 Phase B 完成，无法提前。
- **Phase A→B 的 `SyncAll` 是跨核归约硬边界**，无法靠合并消除（除非换跨核原子归约 / ReduceScatter，复杂度大增）。

因此「继续合并」的物理上限是：**源加载(+量化)与归约 overlap，写回必须留在偏移之后**。

FullLoad 已通过 `LoadXBackground`（`moe_v3_counting_sort_full_load_base.h:367-380`）在 Phase A 发起 x 异步 DMA、Phase C 才 `WaitXLoadCommon`（`:663-667`），把源加载与 Phase B 的 `SyncAll`+归约 overlap 掉了，已接近上限。CutOrigin 因非全载、UB 紧张，源加载仍在 Phase C 内逐 pair 进行。

## 4. 提出的解耦方案

> 把 Phase A 中的 x 搬入 UB 去除；Phase B 结束后先把 expandedRowIdx 单独输出；之后 SyncAll 接入第五阶段搬运量化。

### 4.1 方案流程

```
现在:  Phase A(含 x 全载) → SyncAll → Phase B → Phase C(聚合搬出 + 量化)
改为:  Phase A(无 x 加载) → SyncAll → Phase B → 写 expandedRowIdx → SyncAll → 阶段5 MoeGatherOut
```

### 4.2 方案本质

将 CountingSort 从「融合 kernel（含 Phase C 搬运）」退化为「**计数排序替代 + 索引输出**」，搬运交还给阶段5：

```
等价于:  用 CountingSort 替换原分阶段的 阶段1(排序) + 阶段2(计数) + 阶段3(rowIdx)，保留 阶段5(搬运)
```

核心动机：阶段5 有全套量化变体（`MoeGatherOutStaticQuant/DynamicQuant/Mxfp8Quant/Hif8Quant/...`），而 CountingSort Phase C 当前仅支持非量化 / 静态量化。解耦后「省排序」收益可覆盖所有量化模式。

## 5. 可行性分析

### 5.1 关键发现：MoeGatherOut 本身也聚合搬出

`MoeGatherOut::ProcessGatherLoop`（`moe_v3_gather_out_arch35.h:232-277`）在 `colsLoops_==1` 时，将一批 `rowIdx` 对应源行 gather 进连续 `xLocal`，再 `CopyXOut`（`:298`）一次 `DataCopyPad` 搬出多行（`blockCount=rows`）。

| 聚合维度 | CountingSort Phase C | MoeGatherOut（阶段5） |
|----------|---------------------|----------------------|
| 聚合键 | 按 **专家** 分桶 | 按 **输出 flat 位置** 分段 |
| 源加载 | FullLoad 全载 / CutOrigin 双缓冲 | 逐行串行 gather（`:250`，无双缓冲） |

### 5.2 三个必须对齐的点

**① rowIdx 语义对齐（最易踩坑）**

`MoeGatherOut` 读的是 **dst→src** 索引（遍历输出位置 `dst`，`rowIdx/k` 找源行，`:241-242`）。而现有 Phase C 写入：

- SCATTER：`expandedRowIdx[newPos]=origFlatIdx`（dst→src）→ 与阶段5 **天然兼容**。
- GATHER：`expandedRowIdx[origFlatIdx]=newPos`（src→dst）→ **方向相反**，需额外写一份 dst→src 到 workspace（替代原阶段1 的 sortedRowIdx）。

CountingSort 无需全序排序即可产出 dst→src：遍历 pairs，用偏移 seed 算 `dst=seed+slot`、`src=flatIdx` 直接写。

**② `expertTotalCount` 必须落 workspace 固定位置**

`MoeGatherOut::InitBaseData`（`:107-112`）从 workspace 固定偏移读 `expertTotalCount_` 来算 `perCoreIndicesElements_` / `needCoreNum_`。Phase B 归约后须把「全核有效 pair 总数」写回该位置，否则阶段5 分核错误。

**③ workspace 布局对齐**

`MoeGatherOut.Init`（`:158、:165-168`）按约定偏移读 `sortedExpertIdx` / GATHER 模式 workspace 索引，CountingSort 的输出需与此布局一致。

## 6. 收益与代价

### 6.1 收益

1. **覆盖所有量化模式**：复用阶段5 全套量化算子，计数排序「省排序」收益覆盖动态量化 / MXFP8 / HIF8 / MXFP4 等全部场景。
2. **UB 门槛骤降**：去掉 `LoadXBackground` 后，`EstimateArch35CountingSortFullLoadUB` 中最大的 x 全载项（`:2045`）消失，CountingSort 可覆盖更多 shape（更易命中全载而非降级 CutOrigin）。

### 6.2 代价

1. **搬运端源加载退化**：FullLoad 的 x 全载、CutOrigin 的双缓冲 MTE2/MTE3 overlap，退化为 MoeGatherOut 的逐行串行 gather（源行按 dst 乱序散列，无 overlap）。
2. **多一次边界**：跨 kernel 多一次启动 + GM 往返；同 kernel 需 `SyncAll` + `DataCacheCleanAndInvalid` 刷回再读，并涉及 tiling 数据合并，复杂度高。

## 7. 结论与落地建议

- **量化场景（动态 / MXFP8 / HIF8 / MXFP4...）**：值得。这些场景本就走阶段5，将阶段1/2/3 换成 CountingSort 省掉显式排序，是净收益。
- **非量化 / 静态量化场景**：谨慎。当前 Phase C 的「x 全载 + 专家分桶聚合搬出」或「双缓冲」是更优路径，拆给阶段5 会退化源加载。

**落地建议**：优先采用 **跨 kernel 拆分**——CountingSort 输出 rowIdx + workspace 后结束，阶段5 `MoeGatherOut` 独立执行，复用其现有 tiling 与算子，仅在 CountingSort 侧补齐「dst→src 索引写入 + expertTotalCount 落位 + workspace 布局对齐」三件事。同 kernel 内 SyncAll 后直接调 MoeGatherOut 逻辑，会引入 cache 一致性与 tiling 合并复杂度，收益却不比跨 kernel 多。

## 附录：关键代码位置索引

| 模块 | 文件 | 关键行 |
|------|------|--------|
| CS 模板 key 定义 | `op_kernel/moe_init_routing_v3_apt.cpp` | 168-179 |
| CS 模板分派（提前 return） | `op_kernel/moe_init_routing_v3_apt.cpp` | 346-383 |
| 阶段5 搬运量化 | `op_kernel/moe_init_routing_v3_apt.cpp` | 536 |
| CutOrigin Process（Phase A/B/C） | `op_kernel/arch35/moe_v3_cut_origin_arch35_t.h` | 361-392 |
| FullLoad Process（Phase A/B/C） | `op_kernel/arch35/moe_v3_counting_sort_full_load_unquantized.h` | 184-210 |
| LoadXBackground（x 后台全载） | `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` | 367-380 |
| FilterAndCount | `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` | 383-395 |
| ComputeGlobalOffset（FullLoad） | `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` | 562 |
| WaitXLoadCommon | `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` | 663-667 |
| BucketByExpert（分桶） | `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` | 675 |
| GatherAndWriteByExpert（聚合搬出） | `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` | 728-802 |
| FilterAndCountChunked（CutOrigin Phase A） | `op_kernel/arch35/moe_v3_cut_origin_arch35_t.h` | 396-589 |
| ComputeGlobalOffset（CutOrigin Phase B） | `op_kernel/arch35/moe_v3_cut_origin_arch35_t.h` | 605-686 |
| GatherAndWriteChunked（CutOrigin Phase C） | `op_kernel/arch35/moe_v3_cut_origin_arch35_t.h` | 784 |
| MoeGatherOut Process | `op_kernel/arch35/moe_v3_gather_out_arch35.h` | 280-302 |
| MoeGatherOut ProcessGatherLoop | `op_kernel/arch35/moe_v3_gather_out_arch35.h` | 232-277 |
| MoeGatherOut InitBaseData（读 expertTotalCount） | `op_kernel/arch35/moe_v3_gather_out_arch35.h` | 89-129 |
| MoeGatherOut Init（workspace 布局） | `op_kernel/arch35/moe_v3_gather_out_arch35.h` | 131-169 |
| CountingSort tiling（模板1/2 选择） | `op_host/moe_init_routing_v3_tiling_arch35.cpp` | 127-160 |
| EstimateArch35CountingSortFullLoadUB | `op_host/moe_init_routing_v3_tiling_arch35.cpp` | 2031-2097 |
