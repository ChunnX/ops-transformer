# A5（arch35/950）性能优化模板实施方案

> 适用范围：`moe/moe_init_routing_v3/`，目标芯片 Ascend 950（arch35）
> 配套文档：`A5_tiling_ub_strategy.md`、`A3_tiling_ub_strategy.md`、`A3_vs_A5_diff.md`
> 参考实现：A3 `MoeV3FullLoadCutOriginT` / `MoeV3CutOriginT`（`op_kernel/moe_v3_full_load_cut_origin_t.h` 等）

---

## 一、背景与目标

### 1.1 性能问题

A5 当前在 **`T ∈ [256, 1024×1024]` 且 `LE<256 || LE=E=256`** 场景下性能不佳，其中：
- **`T = n`**（token 数，即输入 x 的第 0 维；注意 **不是** `totalLength = n×k`）
- `LE = actualExpertNum = expertEnd - expertStart`
- `E = expertNum`

> A3 CountingSort 的规模判定正是基于 `n_`（`moe_init_routing_v3_tiling.cpp:824`：`if (n_ < COUNTING_SORT_MIN_N || n_ > COUNTING_SORT_MAX_N) return;`），与本方案的 T 定义一致。`COUNTING_SORT_MIN_N=256`、`COUNTING_SORT_MAX_N=1024*1024`。

根因：
- `IsFullLoad()`（`op_host/moe_init_routing_v3_tiling_arch35.cpp:2131-2191`）在 `totalLength_ > sortLoopMaxElement_` 时直接返回 false。A5 的 `sortLoopMaxElement_ = min(availUb/32, 8160)`（`:523`），所以 `n×k > 8160` 时（即绝大多数非微型 batch）一定走分阶段路径。
- 分阶段路径用 **多核 4 路归并树**（VBS → VMS Middle → SortOut）+ GM workspace 中转 + SyncAll 屏障，复杂度 O(N log N)，且需要 ~24B/元素的排序 workspace 读写（N = totalLength = n×k）。
- 当 LE 较小时（专家值域窄），用 **计数排序（O(N)）** 替代归并树收益显著——这正是 A3 已验证过的优化方向。

### 1.2 目标

在 A5 上补两条性能模板，对齐 A3 已有的 CountingSort 双路径，并扩展到 A5 支持的全部 dropPadMode × quantMode 组合（后续通过结果穿刺收敛触发条件）。

| 模板 | 思路 | 借鉴 A3 | 触发 T(=n) 规模（典型） |
|------|------|---------|---------------------|
| **模板1：多核切T后UB全载** | 多核均分 T(=n)，每核把自己那一份 tokens 对应的 entries（= filterPerCoreTokens × k）全部装入 UB 做计数排序 | `countingSortFullLoad_=1` | 中小 T（每核 entries 能整块装进 UB） |
| **模板2：多核切T+核内切T** | 多核均分 T(=n)，核内按 4096 chunk 流式过滤计数（chunk 以 entries 为单位），再核间归约、按列块搬运 | `countingSortFullLoad_=0`（cutOrigin） | 大 T（单核装不下整块 entries） |

### 1.3 与 A3 的关键差异（必须在模板里处理）

| 维度 | A3 | A5（本方案） |
|------|----|--------------|
| UB 基数 | `ubSize - aivNum×32B`（仅 VBS 公式扣） | `totalUb - 64KB SIMT_DCACHE_SIZE` 全局扣（`moe_init_routing_v3_tiling_arch35.cpp:476`） |
| 计数实现 | 纯 Vector + 标量循环 | 使用 SIMD VF `AscendC::Reg::Histograms` API（源 `uint8_t`、目的 `uint16_t`、BIN0[0-127]/BIN1[128-255] 双 bin，FREQUENCY 模式），见 §3.4.1 |
| TilingData 结构 | `MoeInitRoutingV3TilingData`（BEGIN_TILING_DATA_DEF 反射） | `MoeInitRoutingV3Arch35TilingData`（plain struct，`op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h`） |
| 全载 expertSpace | ×2 buffer | ×3 buffer（`:2158`） |
| 全载量化列缓冲 | 静态 11 / 动态 13 | 静态 5 / 动态 9（`:2167, 2184`） |
| SortOut 窗口 | 1504（经验） | 1024（API 上限） |
| kernel 入口 | `op_kernel/moe_init_routing_v3.cpp` | `op_kernel/moe_init_routing_v3_apt.cpp` |

### 1.4 两个模板的「分阶段」属性辨析

「分阶段」在本仓代码里有**广义**和**狭义**两层含义，必须先厘清才能正确理解模板1/2 的本质，避免 UB 模型与执行模型混淆。

**狭义「分阶段」**（文档里「分阶段处理（通用路径）」）：特指 VBS → VMS Middle → SortOut → ExpertTokensCount → GatherOut 这一套多核 4 路归并树路径（A5 tilingKey `1xxxxxxxx`），UB 模型为**复用**（每阶段独占整块 UB，阶段间释放重规划）。

**广义「分阶段」**：任何具有「阶段间通过 GM workspace 中转 + SyncAll 屏障」的执行模型。按此定义，A3 的 cutOrigin 计数排序（Phase A/B/C）也属于分阶段。

| 模板 | 广义分阶段？ | UB 模型 | 阶段屏障 | 排序算法 | 与现有路径类比 |
|------|--------------|---------|----------|----------|----------------|
| **模板1**（CountingSort FullLoad） | **否** | **叠加**（所有 buffer 同时驻留 UB） | 无 SyncAll（核间无依赖） | 计数排序 O(N) | 本质是 A5 全载路径（`2xxxxx`）的多核 + 计数排序变体 |
| **模板2**（CountingSort CutOrigin） | **是** | **persistent + max(Phase A,B,C)** | Phase A→B 之间 SyncAll | 计数排序 O(N)，4096 chunk 流式 | 本质是 A5 分阶段路径（`1xxxxxxxx`）的特化变体（阶段更少、排序更快、UB 模型更省） |

**模板2 的阶段划分**（参考 `moe_v3_full_load_cut_origin_t.h:34-43`）：
```
Phase A: LoadExpertIdxAndScale → FilterAndCount → WriteExpertCountToWorkspace
            （4096 chunk 流式过滤计数，结果写到 GM workspace）
            ↓ SyncAll（跨核计数可见）
Phase B: ComputeGlobalOffset → WriteExpertTokens
            （读全核 workspace 做归约 + prefix sum，写 expertTokensCountOrCumsum）
Phase C: GatherAndWrite
            （按列块搬运 x 行 + 量化 + 写 expandedX/expandedRowIdx/expandedScale）
```

**与通用分阶段路径的对比**：

| 维度 | 通用分阶段（A5 `1xxxxxxxx`） | 模板2（cutOrigin 计数排序） |
|------|------------------------------|------------------------------|
| 阶段数 | 5~7 段（VBS/VMS/SortOut/ExpertTokensCount/GatherOut） | 3 段（Phase A/B/C） |
| 排序算法 | 多核 4 路归并树 O(N log N) | 计数排序 O(N) |
| 阶段间 UB | **复用**（每阶段独占整块 availUb） | **persistent + max**（常驻区 + 各阶段峰值取最大） |
| GM workspace | sortWorkspace（~24B/元素）排序中转 + coreSyncWorkspace | expertCount 归约表（核数×expertCountStride×4B）+ pairs |
| 收益来源 | 通用兜底 | 阶段更少 + 排序复杂度更低 + UB 利用率更高 |

### 1.4.1 对 A5 方案设计的影响

1. **模板1 与 A5 现有全载路径同源**：UB 模型相同（叠加），只是排序换成计数、单核变多核。tilingKey 若沿用 A5 的 `FULLLOAD_TILINGKEY_BASE(200000)` 段会语义混乱（现有全载是单核），所以**应该用独立 base**。
2. **模板2 与 A5 现有分阶段路径同源**：UB 模型不同（persistent+max vs 复用），但执行模型都是分阶段。tilingKey 可以选两种风格：
   - **方案A（独立 base）**：`500000 + quantMode×10⁴ + ...`——编码简单，但与 A5 现有 `1xxxxxxxx` 分阶段段落在不同区段，语义割裂。
   - **方案B（沿用 sortMode）**：在 `TILINGKEY_BASE(10000000)` 上扩 sortMode=2/3（A5 的 `SORT_CORE_TILINGKEY_BASE=1000000`），即 `12000000 / 13000000 + ...`——与 A5 现有分阶段编码连续，与 A3 的 sortMode=2/3 命名规律对齐。
3. **UB 模型必须严格区分**：模板1 的 tiling UB 估算按「叠加」逐项相加；模板2 按 `persistent + max(Phase A, Phase B, Phase C)`，**不能混用**——这是 §3.3 和 §4.2 公式的核心区别。

> 本方案 §2.1 当前采用「方案A」（两个独立 base 400000/500000）。若维护方偏好编码连续性，可改为「方案B」（sortMode=2/3 沿用），只需同步调整 `GetTilingKey()`（`moe_init_routing_v3_tiling_arch35.cpp:584-586`）和 kernel 入口（`moe_init_routing_v3_apt.cpp`）的宏定义。两种方案的 tiling/kernel 逻辑完全等价。

---

## 二、总体架构设计

### 2.1 tilingKey 编码扩展

A5 现状（`moe_init_routing_v3_apt.cpp:156-160`、`moe_init_routing_v3_tiling_arch35.cpp:553-587`）：
- 全载：`FULLLOAD_TILINGKEY_BASE(200000) + quantModeFactor×10000`，范围 200000~230000+
- 分阶段：`TILINGKEY_BASE(10000000) + sortMode×10⁶ + ...`
- EMPTY_TENSOR = 3000000

**新增两个 base，落在空闲区段**：

```cpp
// op_kernel/moe_init_routing_v3_apt.cpp 新增宏
#define MOE_INIT_ROUTING_V3_COUNTING_SORT_FULLLOAD_UNQUANTIZED   400000
#define MOE_INIT_ROUTING_V3_COUNTING_SORT_FULLLOAD_STATIC_QUANT  410000
#define MOE_INIT_ROUTING_V3_COUNTING_SORT_FULLLOAD_DYNAMIC_QUANT 420000

#define MOE_INIT_ROUTING_V3_COUNTING_SORT_CUTORIGIN_UNQUANTIZED   500000
#define MOE_INIT_ROUTING_V3_COUNTING_SORT_CUTORIGIN_STATIC_QUANT  510000
#define MOE_INIT_ROUTING_V3_COUNTING_SORT_CUTORIGIN_DYNAMIC_QUANT 520000
```

> INT4 动态量化复用 INT8 动态量化 key（与现有全载一致，kernel 内通过 `quantMode` 区分），见 `moe_init_routing_v3_tiling_arch35.cpp:564-565`。
> MXFP8/FP8 PerBlock/FP8 Group/HIF8 等量化模式暂不纳入首版（后述），若结果穿刺需要再扩展对应 key。

**dropPadMode 编码**：沿用在十位和百位编码（与分阶段路径一致，见 `:586` `dropPadMode_ * DROP_MODE_TILINGKEY_BASE`，`DROP_MODE_TILINGKEY_BASE=100`）。即：
- 模板1 非量化 dropless：`400000`
- 模板1 非量化 droppad：`400000 + 1×100 = 400100`
- 模板2 静态量化 dropless：`510000`
- 模板2 静态量化 droppad：`510100`

### 2.2 新增 Tiling 数据结构

在 `op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h` 新增子结构，并挂到 `MoeInitRoutingV3Arch35TilingData`：

```cpp
// CountingSort 专用 TilingData（模板1/模板2共用，通过 countingSortMode 区分）
struct MoeV3Arch35CountingSortTilingData {
    int64_t countingSortMode{0};      // 1=FullLoad(模板1), 2=CutOrigin(模板2)
    int64_t filterNeedCoreNum{0};     // 多核切T的实际核数 = ceil(n, filterPerCoreTokens)
    int64_t filterPerCoreTokens{0};   // 每核分到的 token 数 = ceil(n, aivCoreNum)
    int64_t lastCoreTokens{0};        // 尾核 token 数
    int64_t coreEntries{0};           // 每核 flat entries = filterPerCoreTokens × k
    int64_t expertCountStride{0};     // align(actualExpertNum, ONE_BLOCK_ELEMENT)
    int64_t filterChunkSize{0};       // 模板1=0；模板2=4096
    int64_t csPerLoopCols{0};         // 模板2 Phase C 每轮列数
    int64_t csColsLoops{0};           // 模板2 Phase C 列方向轮数
    int64_t csLastLoopCols{0};        // 模板2 Phase C 尾轮列数
    int64_t maxPerLoopEntries{0};     // 模板2 Phase A 每轮最大 entries（由 UB 推导）
};

struct MoeInitRoutingV3Arch35TilingData {
    // ... 现有字段 ...
    MoeV3Arch35CountingSortTilingData countingSortParamsOp; // 新增
};
```

> 由于 arch35 tiling struct 是 plain struct（非反射），新增字段不影响二进制兼容性，但**必须保证 host 侧写入顺序与 device 侧 `REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT` 读取顺序一致**——二者都引用同一 header，天然一致。

### 2.3 kernel 路由扩展

在 `op_kernel/moe_init_routing_v3_apt.cpp` 的 `moe_init_routing_v3` 入口函数中，紧跟现有三个全载分支（`:251-310`）之后插入两个新分支：

```cpp
// 模板1：CountingSort FullLoad
if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_COUNTING_SORT_FULLLOAD_UNQUANTIZED) || ...) {
    TPipe pipe;
    MoeV3CountingSortFullLoadArch32<DTYPE_X, QuantTag> op;
    op.Init(...); op.Process(); pipe.Destroy();
    return;
}

// 模板2：CountingSort CutOrigin
if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_COUNTING_SORT_CUTORIGIN_UNQUANTIZED) || ...) {
    TPipe pipe;
    MoeV3CountingSortCutOriginArch35<DTYPE_X, QuantTag> op;
    op.Init(...); op.Process(); pipe.Destroy();
    return;
}
```

### 2.4 文件清单

**新增 kernel 文件**（`op_kernel/arch35/`）：
```
moe_v3_counting_sort_full_load_base.h           # 模板1 基类（多核切T + UB全载）
moe_v3_counting_sort_full_load_unquantized.h    # 模板1 非量化
moe_v3_counting_sort_full_load_static_quant.h   # 模板1 静态量化
moe_v3_counting_sort_full_load_dynamic_quant.h  # 模板1 动态量化（INT8/INT4）

moe_v3_counting_sort_cut_origin_base.h          # 模板2 基类（多核切T + 核内chunk）
moe_v3_counting_sort_cut_origin_unquantized.h   # 模板2 非量化
moe_v3_counting_sort_cut_origin_static_quant.h  # 模板2 静态量化
moe_v3_counting_sort_cut_origin_dynamic_quant.h # 模板2 动态量化
```

**修改文件**：
- `op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h`：新增 `MoeV3Arch35CountingSortTilingData` 子结构
- `op_host/moe_init_routing_v3_tiling_arch35.cpp`：
  - 新增 `IsCountingSortApplicable()`
  - 新增 `EstimateArch35CountingSortFullLoadUB()`
  - 新增 `ComputeArch35CountingSortFullLoadTiling()`
  - 新增 `ComputeArch35CountingSortCutOriginTiling()`
  - 修改 `DoOpTiling()`（`:503`）：在 IsFullLoad 之后插入 CountingSort 判定与切分
  - 修改 `GetTilingKey()`（`:553`）：增加两个新 base 分支
  - 修改 `GetWorkspaceSize()`（`:589`）：增加 CountingSort workspace
- `op_kernel/moe_init_routing_v3_apt.cpp`：增加两个新 tilingKey 路由分支
- `build.sh` / `CMakeLists.txt`：新 header 自动随目录扫描，一般无需改；若有显式列表需补

---

## 三、模板1：多核切T后UB全载（CountingSort FullLoad）

### 3.1 触发条件（首版宽松，后续穿刺收敛）

```cpp
bool IsCountingSortApplicable() {
    // T(=n) 规模：覆盖目标场景 [256, 1M]（注意判据是 n，不是 totalLength=n*k）
    if (n_ < COUNTING_SORT_MIN_N || n_ > COUNTING_SORT_MAX_N) return false;
    // LE 范围：用户场景 LE<256 或 LE=E=256
    int64_t le = expertEnd_ - expertStart_;
    if (!(le < 256 || le == expertNum_)) return false;
    // expertNum 上限（与 A3 一致）
    if (expertNum_ > 1024) return false;
    // 其余条件（量化、dropPad）首版全放行，由 tilingKey 编码区分
    return true;
}
```

> `COUNTING_SORT_MIN_N=256`、`COUNTING_SORT_MAX_N=1024*1024` 复用 A3 常量（`moe_init_routing_v3_tiling.cpp:49-50`）。A5 需在 arch35 tiling 文件顶部 redeclare 或抽到公共头。
> 注意：A3 `ComputeCountingSortTiling`（`:824`）用的就是 `n_`，不是 `totalLength_`。本方案保持一致。

**FullLoad vs CutOrigin 选择**（在 `IsCountingSortApplicable` 通过后）：
```cpp
int64_t fullLoadUB = EstimateArch35CountingSortFullLoadUB(...);
countingSortMode_ = (fullLoadUB <= availUbSize_ - LENGTH_1024) ? 1 : 2;
```

### 3.2 Tiling 策略

```cpp
void ComputeArch35CountingSortFullLoadTiling() {
    int64_t filterPerCoreTokens = Ops::Base::CeilDiv(n_, aivCoreNum_);
    int64_t filterNeedCoreNum   = Ops::Base::CeilDiv(n_, filterPerCoreTokens);
    int64_t lastCoreTokens      = n_ - (filterNeedCoreNum - 1) * filterPerCoreTokens;
    int64_t coreEntries         = filterPerCoreTokens * k_;
    int64_t expertCountStride   = Ops::Base::CeilAlign(actualExpertNum_, ONE_BLOCK_ELEMENT);

    // 列方向搬运参数：复用 gatherOutComputeParamsOp 的 perLoopCols/colsLoops
    // 推导方式同 SetPerLoopParams4NoQuantDropLess，但基数是 availUbSize_（已扣 SIMT DCache）
    ...
}
```

### 3.3 UB 预算公式（模板1）

对齐 A3 `EstimateCountingSortFullLoadUB`（`moe_init_routing_v3_tiling.cpp:872-900`），但 **UB 基数换成 `availUbSize_`**（已扣 64KB SIMT DCache）：

```
fullLoadUB = align(coreTokenNum × cols × dtypeSize)            // x 行全载（每核）
           + align(coreEntries × 4B)                            // entries 索引（expertIdx+rowIdx）
           + align(coreTokenNum × scaleSlot × 4B)               // scale（isInputScale 时）
           + align(expertCountStride × 4B)                      // 本核专家计数（int32 累加结果）
           + align(filterNeedCoreNum × expertCountStride × 4B)  // 全核专家计数归约表
           + align(actualExpertNum × 8B)                        // expertTokens 输出
           + align(expertCountStride × 4B)                      // prefix sum
           + align(coreEntries × 2 × 4B)                        // filtered pairs (expertId, srcRow)
           + align(entriesAligned × 1B)                         // Histogram 源缓冲（uint8 归一化后的 expertId）
           + align(histoBinStride × 2B) × 2                     // Histogram 目的缓冲（uint16，BIN0 + BIN1 双 bin）
           + align(entriesAligned × 4B)                         // Cast uint16→int32 中转
           + 128B                                                // SIMD VF 寄存器/对齐杂项预留
```

其中 `entriesAligned = ceilAlign(coreEntries, 64)`（SIMD VF 每拍 64 元素，64 对齐），`histoBinStride = 128`（每 bin 最多 128 个不同值，对应 BIN0[0-127]/BIN1[128-255]）。

**关键变化**：
- 旧 SIMT 方案需要的「fp32 比较区 + 3 个 mask + 3 个 int32 辅助」（≈ `entriesAligned × 4B × 7`）被替换为「uint8 源 + uint16 双 bin 目的 + int32 中转」（≈ `entriesAligned × 1B + 256 × 2B + entriesAligned × 4B`）。
- 由于 LE ≤ 256，**uint8 完整覆盖专家值域**，无需溢出处理；uint16 单 bin 计数上限 65535，远大于单核 worst case（coreEntries ≈ T/k × k = T，T ≤ 1M 时分核后单核 entries ≈ 21K，若 LE=16 则 worst case ≈ 21K < 65535）。

**触发判定**：`fullLoadUB ≤ availUbSize_ - 1024` → 模板1，否则落模板2。

### 3.4 Kernel 流程（模板1）

基类 `MoeV3CountingSortFullLoadBase<T>` 设计，计数部分采用 SIMD VF `AscendC::Reg::Histograms` API（详见 §3.4.1），整体流程参考 A3 `MoeV3FullLoadCutOriginT` 的多核计数流程：

```
Init:
  - 解析 tiling（filterPerCoreTokens / coreEntries / expertCountStride / csPerLoopCols ...）
  - 计算 coreTokenStart/CoreEnd/CoreTokenNum = 本核 token 范围
  - 设置 GM buffer：expertIdxGm / xGm / scaleGm / expandedXGm / expandedRowIdxGm /
                    expertTokensGm / workspaceGm（含 cross-core count 表）
  - InitBuffer：按 3.3 的 offset 表切分单个 TBuf（与 A3 cut_origin_t.h 一致，用 offset 而非多 TQue）

Process:
  1. CopyIn expertIdx + scale（本核 coreEntries 范围）
  2. FilterAndCount (SIMD VF Histograms，详见 §3.4.1)：
     - 过滤掉 expertIdx < expertStart 或 >= expertEnd 的 entry（Vector Compare + Mask）
     - 对每个有效 entry 的 (expertId - expertStart) Cast 到 uint8
     - LoadAlign + Histograms<uint8_t, uint16_t, BIN, FREQUENCY> 累加得到每专家计数
     - 同时把 (expertId, srcRow) 顺序写入 filteredPairs（Vector API 配合双指针）
  3. WriteExpertCountToWorkspace：本核 expertCount（uint16→int32 后）写到 workspace[coreIdx × stride ...]
  4. SyncAll（跨核屏障）
  5. ComputeGlobalOffset：
     - 读所有核的 expertCount，做按专家的 prefix sum
     - 得到每个专家在本核 expandedRowIdx 输出的起始位置
  6. WriteExpertTokens：按 expertTokensNumType 写 expertTokensCountOrCumsum
     - COUNT_MODE：直写每专家计数
     - CUMSUM_MODE：prefix sum
     - KEY_VALUE_MODE：(expertId, count) 对
  7. GatherAndWrite：
     - 按 filteredPairs 顺序，对每个 (expertId, srcRow)：
       - dropPadMode==DROP_LESS：dstIndex = globalOffset[expertId]++ （容量无上限）
       - dropPadMode==DROP_PAD ：dstIndex = expertId × expertCapacity + localCount，
                                 localCount < expertCapacity 才写（参考 full_load_base.h:411 ComputeGatherIdxWithCapacity）
     - 按 csPerLoopCols 列块搬 x 行（含量化分支）
     - SCATTER 模式：expandedRowIdx = srcRow；GATHER 模式：expandedRowIdx[dstIndex] = srcRow（参考 full_load_base.h:434-450）
```

### 3.4.1 计数核心：SIMD VF Histograms API 设计

**API 签名**（[CANN 社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0425.html)，Atlas 350 = 950/arch35 已支持）：

```cpp
template <typename T = uint8_t, typename U = uint16_t,
          HistogramsBinType mode, HistogramsType type>
__simd_callee__ inline void Histograms(V& dstReg, S& srcReg, MaskReg& mask);
```

- `T = uint8_t`：源元素类型，输入范围 `[0, 255]`
- `U = uint16_t`：目的统计类型，最大 65535
- `mode = BIN0 / BIN1`：BIN0 统计 `[0, 127]`，BIN1 统计 `[128, 255]`
- `type = FREQUENCY`（每值计数）/ `ACCUMULATE`（累计前缀和）
- **关键特性**：结果**累加在 dstReg 现有内容之上**——天然支持多次 LoadAlign + Histograms 的滚动累加，无需额外加法。

**计数流程（伪代码）**：

```cpp
// 输入：expertIdxGm[core_flat_start : core_flat_end]（int32，值域 [expertStart, expertEnd)，LE ≤ 256）
// 输出：expertCountLocal[0 : actualExpertNum]（int32 计数）

// Step 1: 过滤 + 归一化（一次性 Cast 到 uint8）
//   过滤：保留 expertStart ≤ idx < expertEnd 的 entry
//   归一化：normalized = (uint8_t)(idx - expertStart)，值域 [0, LE)
LocalTensor<int32_t> expertIdx = ...;       // CopyIn 后
LocalTensor<uint8_t>  normalized = ...;     // 同 offset 区
//   Vector: Compare(idx, expertStart) >= 0 && < expertEnd → mask
//   Vector: Sub(idx, expertStart) → Cast int32→uint8 → 写 normalized
//   同时按 mask 把 (expertIdx, srcRow) 写 filteredPairs

// Step 2: Histograms 计数（分 BIN 处理 LE 上限）
//   情况 A：LE ≤ 128 → 只用 BIN0
//   情况 B：128 < LE ≤ 256 → BIN0 + BIN1 都跑，最后拼接
constexpr int64_t ELEMS_PER_VF_OP = 64;     // SIMD VF 每拍 64 元素
int64_t entriesAligned = CeilAlign(coreEntries, ELEMS_PER_VF_OP);

// BIN0
Reg<uint16_t> bin0Count;  bin0Count.SetValue(0);  // 初始化清零（128 lanes）
Reg<uint8_t>  srcReg;
MaskReg       mask = ...;                        // 全 1 mask（或过滤 mask 的子集）
for (int64_t off = 0; off < entriesAligned; off += ELEMS_PER_VF_OP) {
    LoadAlign(srcReg, normalizedAddr + off);
    Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0,
               HistogramsType::FREQUENCY>(bin0Count, srcReg, mask);
    // bin0Count 内部累加，无需手动 Add
}
StoreAlign(bin0CountDstAddr, bin0Count);          // 128 × uint16 → UB

// BIN1（仅 LE > 128 时）
if (actualExpertNum_ > 128) {
    Reg<uint16_t> bin1Count; bin1Count.SetValue(0);
    for (int64_t off = 0; off < entriesAligned; off += ELEMS_PER_VF_OP) {
        LoadAlign(srcReg, normalizedAddr + off);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1,
                   HistogramsType::FREQUENCY>(bin1Count, srcReg, mask);
    }
    StoreAlign(bin1CountDstAddr, bin1Count);
}

// Step 3: uint16 → int32 转换 + 拼接 BIN0/BIN1
//   Vector Cast: expertCountLocal[0..127]   ← bin0Count (uint16→int32)
//                expertCountLocal[128..255] ← bin1Count (uint16→int32)
```

**为什么比 SIMT 性能更好**：
- Histograms 是硬件直建的统计指令，**1 拍处理 64 元素**、单核单拍完成 128 直方图 bin 的累加，吞吐 ~64 elem/cycle。
- SIMT 方案每 entry 需要一次 `asc_atomic_add`（多线程争用同一 bin），当 LE 小、热点 bin 集中时存在 bank 冲突；Histograms 内部硬件解决冲突。
- Histograms 直接输出 `uint16[128]` 直方图，省去 SIMT 方案的标量聚合 + 跨线程归约。

**UB 占用对比（每核 coreEntries 假设 8K，entriesAligned=8K）**：

| 项 | 旧 SIMT 方案 | 新 Histogram 方案 |
|----|--------------|---------------------|
| 比较区 fp32 | 32 KB | — |
| mask × 3 | ~3 KB | — |
| int32 辅助 × 3 | 96 KB | — |
| uint8 源 | — | 8 KB |
| uint16 目的 × 2 bin | — | 512 B |
| int32 中转（Cast 输出） | — | 32 KB |
| **合计** | **~131 KB** | **~40 KB** |

Histogram 方案 UB 占用降低约 70%，**直接扩大 FullLoad 模板1 的触发窗口**（同样 UB 下能容纳更大 coreEntries）。

**边界处理**：
- `actualExpertNum = 0`：跳过 Histograms，expertCountLocal 全零，filteredPairs 空。
- `LE = 1`：所有有效 entry 落 bin0[0]，单次 Histograms 即可，性能最佳。
- `coreEntries` 非 64 对齐：`entriesAligned` 向上对齐，尾部用 mask 屏蔽越界元素（mask 由 Compare 生成）。

### 3.5 量化分支（模板1）

参考现有 `arch35/moe_v3_full_load_unquantized.h / *_dynamic_quant.h / *_static_quant.h` 的分工：

| 量化模式 | 子类 | GatherAndWrite 中的差异 |
|----------|------|-------------------------|
| UN_QUANT | `MoeV3CountingSortFullLoadUnquantized<T>` | 直接 DataCopy x 行，可选 scale 直传 |
| STATIC_QUANT | `MoeV3CountingSortFullLoadStaticQuant<T>` | 搬 x + 计算 quant (×scale+offset → int8) |
| DYNAMIC_QUANT (INT8) | `MoeV3CountingSortFullLoadDynamicQuant<T>` | 搬 x + per-token absmax + 量化 + 写 expandedScale |
| DYNAMIC_QUANT (INT4) | 复用 INT8 子类，模板参数 `int4b_t` | 同上，输出 int4 |
| MXFP8/FP8/HIF8 | 首版不纳入 | 若穿刺需要再扩展子类 + tilingKey |

### 3.6 dropPadMode 处理（模板1）

- **DROP_LESS**：`outputRows = activeNum`，dstIndex 单调递增，无需容量判定。
- **DROP_PAD**：`outputRows = expertNum × expertCapacity`，在 GatherAndWrite 步骤 7 加入容量判定（`tokenCount < expertCapacity`），参考 `full_load_base.h:399-417`。workspace 需额外 `expertNum × expertCapacity × 4B` 的 outputToSrcRow 映射（参考 `moe_init_routing_v3_tiling_arch35.cpp:619-623` `useCompactGatherOutDropPad`）。

---

## 四、模板2：多核切T+核内切T（CountingSort CutOrigin）

### 4.1 触发条件

`IsCountingSortApplicable()` 通过 **且** `fullLoadUB > availUbSize_ - 1024`（模板1装不下）时进入模板2。

### 4.2 Tiling 策略 + UB 公式（persistent + max(phase)）

对齐 A3 `ComputeCountingSortCutOriginTiling`（`moe_init_routing_v3_tiling.cpp:902-980`），UB 模型是 **persistent + max(Phase A, Phase B, Phase C)**：

```
totalUB = persistentSize + max(phaseASize, phaseBSize, phaseCSize)

persistentSize = align(expertCountStride × 4B)   // 跨阶段常驻专家计数表
```

| Phase | 功能 | UB 构成 |
|-------|------|---------|
| **A** | 4096 chunk 过滤计数 | `align(chunk × 4B int32)`（expertIdx 输入）+ `align(chunk × 1B uint8)`（归一化源）+ `align(histoBinStride × 2B) × 2`（Histograms uint16 目的 BIN0+BIN1）+ `align(chunk × 4B int32)`（uint16→int32 中转）+ `align(chunk × 2 × 4B)`（filtered pairs）+ `maskBytes`（过滤 mask） |
| **B** | 多核计数归约 + prefix sum | `align(filterNeedCoreNum × expertCountStride × 4B) + align(expertCountStride × 4B) + align(actualExpertNum × 8B)` |
| **C** | 按列块搬运 x 行 + 量化 | 固定开销（pairsBuf 2048×2×4B + idxBuf + 3×stack 数组）+ `csPerLoopCols` 由剩余 UB 动态推导 |

**Phase C `csPerLoopCols` 推导**（关键，对齐 A3 `:944-951`）：
```
phaseCBudget   = availUbSize_ - persistentSize - 32B
phaseCFixedOverhead = pairsBuf(2048×2×4B) + idxBuf(32B)
                    + (dropPadMode!=DROP_PAD ? 3×2048×4B : 0)   // 3 个 stack 数组
remainingForCols   = phaseCBudget - phaseCFixedOverhead - 1024
colBytesPerElement = 2 × dtypeSize + (STATIC_QUANT ? fp32+int8×2 : 0)
dynamicMaxPerLoopCols = remainingForCols / colBytesPerElement
                     （按 ONE_BLOCK_BYTE 向下对齐）
csPerLoopCols    = min(dynamicMaxPerLoopCols, COUNTING_SORT_MAX_PER_LOOP_COLS_BYTES / dtypeSize)
csColsLoops      = ceilDiv(cols, csPerLoopCols)
```

> **A3 限制**：`csColsLoops>1 && quantMode_!=UN_QUANT → countingSortFullLoad_=-1`（`:960-963`）禁用 cutOrigin。
> **本方案**：首版保留该限制作为安全网（量化 + 宽列场景退回分阶段路径），后续若结果穿刺表明可放宽再解禁。

### 4.3 Kernel 流程（模板2）

基类 `MoeV3CountingSortCutOriginBase<T>`，参考 A3 `MoeV3FullLoadCutOriginT`（`moe_v3_full_load_cut_origin_t.h`）的三阶段流程，计数部分用 SIMD VF `Histograms` API（§3.4.1）替代 SIMT：

```
Init:
  - 解析 tiling（filterChunkSize=4096 / maxPerLoopEntries / csPerLoopCols ...）
  - 单 TBuf 按 persistentSize + max(phase) 切分 offset（同 A3 cut_origin_t.h:92-110 风格）

Process:
  Phase A（chunked filter and count）:
    // Histogram 累加跨 chunk 持续，bin0Count/bin1Count 寄存器在循环外初始化清零
    Reg<uint16_t> bin0Count; bin0Count.SetValue(0);
    Reg<uint16_t> bin1Count; bin1Count.SetValue(0);   // 仅 LE>128 用
    for (offset = coreFlat_start; offset < core_flat_end; offset += chunk) {
      chunkSize = min(filterChunkSize, remain);
      CopyIn expertIdx[offset : offset+chunkSize]
      // 过滤 + 归一化（Cast int32 → uint8）
      //   mask = Compare(idx, expertStart) >= 0 && < expertEnd
      //   normalized = (uint8)(idx - expertStart)
      //   同时把 (expertId, srcRow) 压入 filteredPairs
      // Histograms 计数（累加到 bin0Count/bin1Count，无需手动 Add）
      for (inner_off = 0; inner_off < chunkEntriesAligned; inner_off += 64) {
          LoadAlign(srcReg, normalizedAddr + inner_off);
          Histograms<uint8_t, uint16_t, BIN0, FREQUENCY>(bin0Count, srcReg, mask);
          if (actualExpertNum_ > 128) {
              Histograms<uint8_t, uint16_t, BIN1, FREQUENCY>(bin1Count, srcReg, mask);
          }
      }
    }
    StoreAlign(bin0Dst, bin0Count);                   // 128 × uint16
    if (actualExpertNum_ > 128) StoreAlign(bin1Dst, bin1Count);
    // uint16 → int32 转换，拼接成 expertCountLocal[0..actualExpertNum]
    WriteExpertCountToWorkspace（本核累计的 expertCount → workspace）

  SyncAll

  Phase B（cross-core reduce and prefix sum）:
    CopyIn 所有核的 expertCount（filterNeedCoreNum × expertCountStride）
    按专家求和 → 全局 expertCount
    按 expertTokensNumType 计算：
      COUNT_MODE ：全局 count
      CUMSUM_MODE：prefix sum
      KEY_VALUE_MODE：(expertId, count) 对（参考 full_load_base.h:490-506 CopyExpertCountToOutput）
    WriteExpertTokens

  Phase C（gather and write，按 csColsLoops × csPerLoopCols 列块）:
    for (colLoop = 0; colLoop < csColsLoops; colLoop++) {
      for (each (expertId, srcRow) in filteredPairs) {
        // 容量判定（DROP_PAD）
        if (dropPadMode == DROP_PAD && localCount[expertId] >= expertCapacity_) continue;
        CopyIn x[srcRow, colLoop×csPerLoopCols : +csPerLoopCols]
        （量化分支：static 加 scale+offset 计算；dynamic 加 absmax 量化并写 expandedScale）
        CopyOut expandedX[dstIndex, ...] + expandedRowIdx[dstIndex] = srcRow
      }
    }
```

**关键点**：Phase A 的 chunk 循环里，`bin0Count`/`bin1Count` SIMD 寄存器在循环外初始化、跨 chunk 持续累加——充分利用 Histograms API「结果累加在 dstReg 现有内容之上」的特性，**chunk 越多越省 UB**（不用每个 chunk 都 StoreAlign 再归约），仅循环结束后 StoreAlign 一次。

### 4.4 量化分支与 dropPadMode（模板2）

同模板1的 3.5 / 3.6，但 **Phase C 按列块搬运** 时：
- STATIC_QUANT：`colBytesPerElement = 2×dtypeSize + fp32 + int8×2`，每列块需加载 quantFloat + quantInt8
- DYNAMIC_QUANT：每行 absmax 计算 + 量化 + scale 输出
- DROP_PAD：在 Phase C 的内层循环加容量判定，并对 expandedRowIdx 初值填 -1（参考 `full_load_base.h:394`）

---

## 五、A5 专属适配点

1. **UB 基数**：所有公式用 `availUbSize_`（已扣 64KB SIMT DCache），**不要用 `totalUbSize_`**。
2. **SIMD VF Histograms 计数**：
   - 模板1 FilterAndCount、模板2 Phase A 的计数逻辑用 `AscendC::Reg::Histograms<uint8_t, uint16_t, BIN, FREQUENCY>`（见 §3.4.1）。
   - **uint8 源约束**：必须先把 `int32` expertIdx Cast 成 `uint8`（归一化 `(idx - expertStart)` 后值域 `[0, LE) ⊆ [0, 256)`），由用户场景 `LE ≤ 256` 天然保证。
   - **BIN 切分**：`LE ≤ 128` 只用 BIN0；`128 < LE ≤ 256` BIN0+BIN1 各跑一次，最后拼接为 `expertCountLocal[0..actualExpertNum]`。
   - **跨 chunk 累加**：模板2 Phase A 的 `Reg<uint16_t> binCount` 在循环外初始化、循环内反复 `Histograms` 累加，循环结束统一 `StoreAlign`。
   - **uint16 → int32 转换**：输出 `expertCount` 需 `int32`（用于跨核归约与 prefix sum），用 Vector `Cast` 完成；worst case 单核 entries ≈ 21K（T=1M, k=8, 48 核）远小于 uint16 上限 65535，无溢出风险。
   - **头文件**：`#include "kernel_utils_vec_binary_op.h"` 或 arch35 对应的 SIMD VF 寄存器头（`AscendC::Reg` namespace），与 `Histograms` 同源。
3. **Sort API 上限**：模板1/2 不用 AscendC::Sort（用计数），所以 **不受 8160 上限约束**——这正是性能优化的核心收益点。
4. **overflow mode**：kernel 入口需保留 `GetCtrlSpr<OVERFLOW_MODE_CTRL>` 的 save/restore（参考 `moe_init_routing_v3_apt.cpp:185-187, 262-264`）。
5. **同步**：
   - 核间计数归约：`SyncAll`（Phase A → Phase B 屏障）
   - workspace 读写依赖：`DataCacheCleanAndInvalid<int32_t, SINGLE_CACHE_LINE, CACHELINE_OUT>`（参考 `full_load_base.h:531`）
6. **arch35 tiling struct**：plain struct，不能用 `set_xxx()` 反射接口，直接 `tilingDataPtr_->countingSortParamsOp.xxx = ...` 赋值。

---

## 六、Workspace 计算

在 `GetWorkspaceSize()`（`:589`）新增分支：

```cpp
if (countingSortMode_ == 1) {
    // 模板1：仅 cross-core expert count 表
    workspaceSize_ += filterNeedCoreNum_ * expertCountStride_ * sizeof(int32_t);
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        workspaceSize_ += expertNum_ * expertCapacity_ * sizeof(int32_t); // outputToSrcRow
    }
} else if (countingSortMode_ == 2) {
    // 模板2：cross-core count 表 + filtered pairs
    workspaceSize_ += filterNeedCoreNum_ * expertCountStride_ * sizeof(int32_t);
    int64_t maxCoreEntries = filterPerCoreTokens_ * k_;
    workspaceSize_ += filterNeedCoreNum_ * Ops::Base::CeilAlign(maxCoreEntries, 8) * 2 * sizeof(int32_t);
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        workspaceSize_ += expertNum_ * expertCapacity_ * sizeof(int32_t);
    }
}
// 框架开销 16M 仍保留
```

---

## 七、实施步骤（建议顺序）

| 步骤 | 内容 | 验证 |
|------|------|------|
| 1 | 新增 `MoeV3Arch35CountingSortTilingData` 子结构 + tilingKey 宏 | 编译 host 通过 |
| 2 | 实现 `IsCountingSortApplicable` + `EstimateArch35CountingSortFullLoadUB`，在 `DoOpTiling` 接入（暂只算 tiling，kernel 路由先空实现 return） | 编译 + 跑 UT，看 tiling 字段日志 |
| 3 | 实现模板1 kernel（先 UN_QUANT + DROP_LESS 单路径） | Simple 模式精度验证 |
| 4 | 模板1 扩展 STATIC_QUANT / DYNAMIC_QUANT / DROP_PAD | 全组合精度验证 |
| 5 | 实现 `ComputeArch35CountingSortCutOriginTiling` + 模板2 kernel（先 UN_QUANT） | 大 T 场景精度验证 |
| 6 | 模板2 扩展全量化 / dropPad 组合 | 全组合精度验证 |
| 7 | 性能对比（profiling）：vs 分阶段路径 | 收集收益数据 |
| 8 | 结果穿刺：根据性能/精度数据收敛 `IsCountingSortApplicable` 条件 | 出最终触发阈值表 |

---

## 八、验证计划

### 8.1 精度验证（DumpTensor 7 步法）

每个模板 × 每个量化模式 × 每个 dropPadMode 至少 1 个用例：
- 用 `ascendc-dumptensor` skill，按 CopyIn → FilterAndCount → Phase B 归约 → Phase C 搬运 逐步对比 NPU vs CPU golden
- 容差：`atol=1e-5, rtol=1e-3`（量化场景适当放宽）

### 8.2 性能验证（profiling）

用 `ascendc-profiling` skill，对比：
- Baseline：分阶段路径（tilingKey=10xxxxxx）
- 模板1 / 模板2
- 关注指标：kernel 总耗时、pipeline 利用率、GM 带宽利用率

典型测试点矩阵（T = n，LE = actualExpertNum，totalLength = n×k）：
```
T(=n) ∈ {256, 4K, 32K, 256K, 1M}
LE    ∈ {16, 64, 128, 256 (=E)}
k     ∈ {8}（默认 topK）
cols  ∈ {7168, 32768}
quant ∈ {UN_QUANT, STATIC, DYNAMIC}
dropPad ∈ {DROP_LESS, DROP_PAD}
```

### 8.3 边界用例

- LE=1（极端窄专家域）
- T(=n)=256 / T(=n)=1M（规模边界，注意 totalLength=n×k 此时分别为 256×k 和 1M×k）
- actualExpertNum=0（空过滤，需 return 不崩）
- 尾核 tokens=1（`filterNeedCoreNum` 边界）
- 列方向 csPerLoopCols=cols（colsLoops=1）和 csPerLoopCols 折半（colsLoops>1）

---

## 九、风险与回退

| 风险 | 缓解 |
|------|------|
| Histograms 计数在 LE 接近 256 时性能不及归并树 | `IsCountingSortApplicable` 设 LE 上界阈值，穿刺后调参；Histograms API 硬件吞吐固定 ~64 elem/cycle，理论无热点退化 |
| STATIC_QUANT + 宽列场景模板2 Phase C 装不下 | 保留 A3 的 `csColsLoops>1 && 非UN_QUANT → 退回分阶段` 安全网 |
| 模板2 Phase B 多核归约 SyncAll 卡死 | 用 `ascendc-sync` skill 核对 SetFlag/WaitFlag/eventID 使用 |
| arch35 tiling struct 字段顺序错位 | host/device 共用同一 header，CI 编译期保证 |
| INT4/MXFP8 等量化首版未覆盖 | tilingKey 不命中 → 自动走原路径，无功能回退 |

**整体回退**：任何模板未命中（tilingKey 不匹配新增的 6 个 base），kernel 入口 fall through 到现有分阶段路径，功能与现状完全一致。

---

## 十、交付物

1. 本文档（实施方案）
2. 8 个新 kernel header（模板1/2 × base + 3 量化变体，INT4 复用）
3. arch35 tiling_def 修改
4. arch35 tiling.cpp 修改（IsApplicable / Estimate / Compute×2 / DoOpTiling / GetTilingKey / GetWorkspaceSize）
5. moe_init_routing_v3_apt.cpp 修改（6 个新 tilingKey 路由分支）
6. UT 用例（`tests/ut/op_host/test_moe_init_routing_v3_arch35_tiling.cpp` 扩展）
7. 精度 + 性能报告（穿刺数据 + 最终触发阈值表）
