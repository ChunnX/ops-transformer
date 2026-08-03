# CountingSort 计数段 Histograms API 整改方案

> 目标：在 2 个新 CountingSort 模板（FullLoad / CutOrigin）的计数段引入 `AscendC::MicroAPI::Histograms` 直方图 API，替换现有的标量逐元素计数累加，对比整改前后性能是否有优化。
>
> 芯片：Ascend 950DT (arch35 / `__NPU_ARCH__ == 3510`)，device 2。

---

## 1. Histograms API 能力与约束（来自知识库 910D_knowledge/Histograms.md）

| 项 | 说明 |
|----|------|
| 命名空间 | `AscendC::MicroAPI::Histograms` |
| 函数原型 | `template <typename T, typename U, HistogramsBinType mode, HistogramsType type, typename S, typename V> __simd_callee__ void Histograms(V& dstReg, S& srcReg, MaskReg& mask)` |
| 源类型 T | **仅 `uint8_t`**（950PR/950DT） |
| 目的类型 U | **仅 `uint16_t`** |
| bin 数量 | 固定 **256 个**（值域 [0,255]）。dst 为 uint16_t，一个 VL(256B) 存 128 个元素，故需 **2 个 dstReg**：`BIN0` 存 [0,127]，`BIN1` 存 [128,255] |
| mode (HistogramsBinType) | `BIN0`（低位）/ `BIN1`（高位），分别统计 src 前/后半区间 |
| type (HistogramsType) | `FREQUENCY`（每个值的频次）/ `ACCUMULATE`（前缀累计） |
| 累加语义 | 在 dstReg 已有数据基础上**累加**统计结果（可跨 repeat 累积） |
| 约束说明 | 文档标注"无"额外约束 |

**调用范式**（来自文档示例，`__VEC_SCOPE__` 内）：
```cpp
RegTensor<uint8_t> srcReg;
RegTensor<uint16_t> dstReg0, dstReg1;      // BIN0 + BIN1
MaskReg mask = CreateMask<uint8_t>();
for (uint16_t i = 0; i < repeatTimes; ++i) {
    LoadAlign(srcReg, srcAddr + oneRepeatSize * i);
    Histograms<uint8_t, uint16_t, BIN0, FREQUENCY>(dstReg0, srcReg, mask);
    Histograms<uint8_t, uint16_t, BIN1, FREQUENCY>(dstReg1, srcReg, mask);
}
// dstReg0/dstReg1 store 回 UB，即 256 个专家的 count
```

---

## 2. 现状分析：两个模板计数段瓶颈定位

### 2.1 FullLoad — `FilterAndCountVector()`（base.h:309-372）

流程：
1. **向量过滤**：`Cast(int32→fp32)` → `CompareScalar(GE expertStart)` + `CompareScalar(LT expertEnd)` → `And` → `GatherMask` 压缩出 `gatheredExpertLocal` / `gatheredIdxLocal`（已向量化，快）。
2. **标量计数 + pairs 构建**（360-371 行 for 循环，**瓶颈**）：
   ```cpp
   for (j = 0; j < filteredInBatch; j++) {
       expertId = gatheredExpertLocal.GetValue(j);        // 标量读
       localFlatIdx = gatheredIdxLocal.GetValue(j);       // 标量读
       expertOffset = expertId - expertStart_;
       curCount = expertCountLocal.GetValue(expertOffset);// 标量读
       expertCountLocal.SetValue(expertOffset, curCount+1);// 标量写（计数累加）
       filteredPairsLocal.SetValue(filteredCount_*2, localFlatIdx);   // pairs
       filteredPairsLocal.SetValue(filteredCount_*2+1, expertOffset); // pairs
       filteredCount_++;
   }
   ```
   该循环同时做两件事：**(a) expert count 累加**、**(b) filteredPairs 构建**（Phase C 搬运用）。

### 2.2 CutOrigin — `FilterAndCount()`（base.h:281-292）

**纯标量**，比 FullLoad 更慢（连过滤都是标量）：
```cpp
for (j = 0; j < coreEntries_; j++) {
    expertId = expertIdxLocal.GetValue(j);
    if (expertId < expertStart_ || expertId >= expertEnd_) continue;  // 标量过滤
    expertOffset = expertId - expertStart_;
    curCount = expertCountLocal.GetValue(expertOffset);
    expertCountLocal.SetValue(expertOffset, curCount+1);              // 计数累加
    filteredPairsLocal.SetValue(filteredCount_*2, j);                 // pairs
    filteredPairsLocal.SetValue(filteredCount_*2+1, expertOffset);    // pairs
    filteredCount_++;
}
```

### 2.3 关键洞察：Histograms 只能替换"计数累加(a)"，替换不了"pairs 构建(b)"

- Histograms 输出的是 **每个专家的总 count**（等价于步骤 a 的结果 `expertCountLocal`），一次向量操作完成。
- 但 **filteredPairs（每个 filtered entry 的 [flatIdx, expertOffset]）无法由 Histograms 产出** —— 它是 Phase C 搬运/写回 rowIdx 的必需输入，仍需标量或 GatherMask 构建。
- 因此本整改**只优化计数累加子步骤**，pairs 构建保留（或用 GatherMask 已压缩的结果继续标量填充）。

---

## 3. 适配性评估（重要）

| 维度 | expert_idx 现状 | Histograms 要求 | 是否满足 |
|------|----------------|----------------|---------|
| 源 dtype | int32 | uint8_t | ❌ 需 Cast/窄化 int32→uint8 |
| 值域 | expertOffset ∈ [0, actualExpertNum_) | [0,255] | ✅ `actualExpertNum_ ≤ COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM = 256` |
| bin 数 | 需 actualExpertNum_ 个 | 固定 256 | ✅ 256 上限恰好对齐 |
| 输出 dtype | int32 count | uint16_t | ⚠️ 单核单专家 count 可能 > 65535？见下方风险 |

### 3.1 dtype 窄化方案
- 计数的 key 是 **expertOffset = expertId - expertStart_**，天然落在 [0, actualExpertNum_) ⊆ [0,255]，可安全窄化为 uint8_t。
- 过滤后（GatherMask 已剔除越界专家）的 `gatheredExpertLocal`(int32) → 减 expertStart_ → Cast/位截断到 uint8_t 后喂给 Histograms。
- **越界专家处理**：FullLoad 已用 GatherMask 压缩，只剩合法专家；CutOrigin 若走 Histograms 需先做同样的向量过滤（把纯标量过滤也向量化），否则越界 expertId 窄化后会污染 bin。

### 3.2 风险点
1. **uint16_t 溢出**：单核内单个专家的 token 数若 > 65535，uint16_t count 溢出。需评估 `filterPerCoreTokens_ * k_` 上限；`COUNTING_SORT_MAX_N=1M`、多核分摊后单核 entries 一般 << 65535，但**极端单专家倾斜**需加保护（分段累加到 int32，或断言 tiling 保证）。
2. **CutOrigin 需先向量化过滤**：当前 CutOrigin 是纯标量，引入 Histograms 前必须先补齐"Cast+CompareScalar+And+GatherMask"这套向量过滤（可复用 FullLoad 的实现），否则收益无从谈起。
3. **pairs 仍是标量**：若 pairs 构建本身才是主瓶颈，则 Histograms 优化计数的收益会被 pairs 循环掩盖 → 需 profiling 确认瓶颈占比。
4. **256B 对齐 / repeat 尾块**：uint8_t 源一个 repeat 处理 256 元素，`coreEntries_` 尾块需 mask 处理。

---

## 4. 整改设计

### 4.1 新增公共方法（放入各自 base.h，属公共计数逻辑）

`FilterAndCountHistogram()` —— 用 Histograms 产出 `expertCountLocal`：

```cpp
// 伪代码，__VEC_SCOPE__ 内
// 输入：gatheredExpertLocal(int32, 已过滤压缩) + filteredInBatch
// 步骤：
//   1. expertOffset = gatheredExpert - expertStart_  (向量 Subs)
//   2. Cast int32 -> uint8_t（窄化，值域已保证 <256）
//   3. 分 repeat 载入 uint8 源，Histograms<BIN0/BIN1, FREQUENCY> 累加到 dstReg0/dstReg1
//   4. dstReg0/dstReg1(uint16) store 到 UB 临时区
//   5. Cast uint16 count -> int32 写入 expertCountLocal（对齐现有下游 int32 契约）
```

### 4.2 UB 布局新增
- `expertU8LocalOffset_`：过滤后 expertOffset 窄化为 uint8 的缓冲（entriesAligned_ 字节）。
- `histBin0/Bin1Offset_`：uint16 直方图输出（各 128×2=256B）。
- 追加在 `commonBufSize_` 之前的公共区（因是公共计数逻辑）；注意与量化子类区不冲突。

### 4.3 pairs 构建保留
- FullLoad：pairs 循环保持（已在 GatherMask 压缩结果上标量填充，长度=filteredInBatch）。
- CutOrigin：需先向量过滤 → 再在压缩结果上标量填 pairs。

### 4.4 开关控制（便于 A/B 对比）
- 用编译宏 `MOE_CS_USE_HISTOGRAM`（或 tiling 字段）在 `FilterAndCount()` 内切换旧标量计数 vs `FilterAndCountHistogram()`，保证可回退、可对比。
- **计数结果必须与旧实现逐核一致**（expertCountLocal 完全相同），仅性能不同。

---

## 5. 实施步骤（分阶段，可回退）

| 阶段 | 内容 | 验证 |
|------|------|------|
| S1 | 在 FullLoad base 加 `FilterAndCountHistogram()`，宏开关切换；UB 布局补齐 | 编译通过 |
| S2 | 单测：Histogram count == 旧标量 count（同输入逐核比对 expertCountLocal） | 数值一致 |
| S3 | TTK 回归 ti=17/19/20（FullLoad 非/静/动） | PRECISION PASS，DYN_GOLD 无退化 |
| S4 | Profiling 对比 S1 前后计数段耗时（`ascendc-profiling`） | 量化 DYN_PERF 变化 |
| S5 | 若 FullLoad 有收益，CutOrigin 先补向量过滤，再套 Histogram（ti=18/21/22） | PASS + Profiling |
| S6 | 溢出保护（单专家 >65535 分段累加）、尾块 mask 校验 | 边界用例 |

---

## 6. 预期与判定

- **乐观**：计数累加从 O(filteredInBatch) 标量串行 → O(repeat) 向量并行，计数段耗时显著下降；CutOrigin 因原本纯标量，收益应大于 FullLoad。
- **悲观/需警惕**：
  - 若 pairs 标量构建占主导，Histogram 只优化计数累加，整体 DYN_PERF 提升有限。
  - int32→uint8 窄化 + uint16→int32 回写的额外 Cast 开销可能吃掉部分收益。
  - CutOrigin 补向量过滤的引入成本需一并计入。
- **判定标准**：以 S4/S5 profiling 的 **DYN_PERF（越大越好）** 为准，且 S2/S3 计数一致性 + 精度 PASS 为前置门槛。**若无正收益则回退（保留宏开关）**。

---

## 7. 涉及文件

| 文件 | 改动 |
|------|------|
| `arch35/moe_v3_counting_sort_full_load_base.h` | 加 `FilterAndCountHistogram()` + UB 布局 + 宏开关 |
| `arch35/moe_v3_counting_sort_cut_origin_base.h` | 补向量过滤 + `FilterAndCountHistogram()` + UB 布局 + 宏开关 |
| `arch35/moe_v3_common.h` | 可选：新增开关常量/宏 |

## 8. 编译探针结论 & CutOrigin 优先落地设计（2026-08-05 更新）

### 8.1 编译探针结果（S1 前置）
在 FullLoad base 临时插入 `MoeV3HistogramProbeVF`（`#define MOE_CS_HISTOGRAM_PROBE` 保护），编包验证：
- `LoadAlign` / `StoreAlign` **存在**（toolkit `npu_arch_3510/.../reg_load/asc_loadalign_impl.h`、`reg_store/asc_storealign_impl.h`）。
- `Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0/BIN1, HistogramsType::FREQUENCY>(dstReg, srcReg, mask)` **编译通过**，枚举全名正确。
- `MaskReg mask = CreateMask<uint8_t, MaskPattern::ALL>()`、`__simd_vf__` + `__ubuf__ uint8_t*/uint16_t*` 签名 **OK**。
- 编包 100% 通过无 `error:`。探针已删除，base 已还原。

### 8.2 关键结论：FullLoad 不宜先改（用户决策 → 先做 CutOrigin）
FullLoad 计数段的标量循环（`FilterAndCountVector` 392-403）**必须保留**——它在 GatherMask 压缩结果上构建 `filteredPairs[flatIdx,expertOffset]`（Phase C 搬运必需），Histograms 产不出。Histograms 只能替换循环内 2 个计数 op（Get/SetValue），却要额外加一整个向量 pass（int32→uint8 窄化 + LoadAlign + 2×Histograms + uint16→int32 回写），**对 FullLoad 很可能不减反增**。

CutOrigin 计数段（`FilterAndCount` 281-292）当前是**纯标量扫描整个 coreEntries_**（连过滤都是标量），Histograms 收益更明确 → **优先落地 CutOrigin**。

### 8.3 CutOrigin S5 落地设计（A/B 宏开关 `MOE_CS_USE_HISTOGRAM`）

复用 FullLoad 的向量过滤思路，在 CutOrigin base 内实现：

**旧路径**（默认，保留可回退）：现有纯标量单扫描（filter + count + pairs 一体）。

**新路径**（`MOE_CS_USE_HISTOGRAM` 开）：
1. **向量过滤**（新增，复用 FullLoad 实现）：Cast int32→fp32 → CompareScalar(GE expertStart)+CompareScalar(LT expertEnd) → And → GatherMask 压出 `gatheredExpertLocal`（合法专家 int32）+ `gatheredIdxLocal`（原始 j），得 `filteredInBatch`。
2. **窄化**：`gatheredExpertLocal - expertStart_` → 值域 [0, actualExpertNum_)⊆[0,256) → Cast int32→uint8（VF 内）。因源已 GatherMask 剔除越界专家，**无越界污染问题，无需 sentinel bin**（回避 actualExpertNum_==256 无空闲 bin 的难点）。
3. **Histograms**：分 repeat 载入 uint8 源，`Histograms<...BIN0/BIN1, FREQUENCY>` 累加到 dstReg0/dstReg1(uint16) → store 回 UB → Cast uint16→int32 写入 `expertCountLocal`。
4. **pairs 构建**：在压缩结果 `gatheredIdxLocal` 上标量填 `filteredPairsLocal`（长度=filteredInBatch，比旧路径 coreEntries_ 短），**不再做 count 累加**。

### 8.4 CutOrigin UB 布局新增（追加在 commonBufSize_ 之前公共区）
- `expertIdxFp32LocalOffset_`：entriesAligned fp32（Cast 目的）
- `compareMask0/1Offset_`、`gatherMaskOffset_`：过滤掩码
- `gatheredExpertLocalOffset_`、`gatheredIdxLocalOffset_`、`flatIdxBufferLocalOffset_`：GatherMask 输出/输入
- `expertU8LocalOffset_`：窄化 uint8 源（coreEntries 对齐）
- `histBinOffset_`：uint16 直方图输出（256×2=512B）

> 注：`entriesAligned_` 当前按 `filterChunkSize_` 算（166 行），但 FilterAndCount 扫描的是整个 `coreEntries_`。向量过滤需按 `coreEntries_` 对齐重算，与 FullLoad 一致。

### 8.5 溢出与尾块
- uint16 count 溢出：单核单专家 >65535 时溢出。CutOrigin 单核 entries=`coreTokenNum_*k_`，多核分摊后一般 <<65535；极端倾斜需断言或分段。S6 处理。
- uint8 源 repeat 尾块：`coreEntries_` 非 256 整数倍时末 repeat 用 mask。

### 8.6 A/B 实测结论（2026-08-05，device 2）——**负收益，回退**

> DYN_PERF = **耗时（越小越好）**。每路径 3 次重复，分布无交叠。

| 用例 | TilingKey | 旧标量路径 (3次) | Histograms路径 (3次) | 结论 |
|------|-----------|-----------------|---------------------|------|
| ti=21 静态 | 510000 | 185.0 / 186.1 / 186.7 | 188.4 / 189.2 / 188.8 | **慢 ~2.4 (+1.3%)** |
| ti=22 动态 | 520000 | 761.1 / 760.9 / 760.8 | 763.9 / 764.7 / 763.3 | **慢 ~3.0 (+0.4%)** |

- ti=18(TilingKey 200000) **非 CutOrigin 模板**，不纳入对比。
- 两路径区间完全不重叠（ti=21 旧 max 186.7 < Hist min 188.4；ti=22 旧 max 761.1 < Hist min 763.3）→ 确凿负收益。
- 精度全 PASS，计数逐核 bit 一致（功能正确），但性能变慢。

**根因**：`filteredPairs` 标量循环是硬约束无法消除；Histograms 只替换计数累加，却额外引入「向量过滤(Cast+2×CompareScalar+And+2×GatherMask) + int32→int16→uint8 两步窄化 + VF + uint16→int32 回写」一整套开销，**净增 > 净减**。这两个用例 filteredInBatch 占 coreEntries_ 比例不高，计数累加本非瓶颈。

**判定（依 §6 标准，无正收益则回退）**：Histograms 整改**不予采纳**。处理方式：
- `MOE_CS_USE_HISTOGRAM` 宏**默认关闭**，回退旧标量路径（已重新编包安装验证 baseline）。
- 新路径代码（`FilterAndCountHistogram` + `MoeV3CutOriginHistogramVF` + UB 布局）**保留在 base.h**，供后续超大 N/单专家极端倾斜用例复验（此类场景计数累加占比可能升高，或有正收益）。
- FullLoad 未改（§8.2 分析预判负收益更明显，未落地）。

### 8.7 A/B 两条用例具体信息（CutOrigin，均来自 `moe_v3_test.csv`）

| 项 | ti=21 `moe_v3_cs_cutorigin_static_quant` | ti=22 `moe_v3_cs_cutorigin_dynamic_quant` |
|----|------------------------------------------|-------------------------------------------|
| TilingKey | 510000 | 520000 |
| quant_mode | 0（静态 per-tensor） | 1（动态 per-token absmax） |
| x shape / dtype | (9000, 7168) / float16 | (9000, 7168) / float16 |
| expert_idx shape / dtype | (9000, 1) / int32 | (9000, 1) / int32 |
| scale 输入 | (1,) float32（标量 scale） | (256, 7168) float32（per-expert smooth） |
| offset 输入 | (1,) float32 | None |
| 输出 dtype | int8 / int32 / int64 / float32 | int8 / int32 / int64 / float32 |
| 输出 shape | (9000,7168)、(9000,)、(256,)、(9000,) | 同左 |
| N / k / cols | 9000 / 1 / 7168 | 9000 / 1 / 7168 |
| expert_num / active_expert_range | 256 / [0,256] | 256 / [0,256] |
| drop_pad_mode | 0（dropless） | 0（dropless） |
| expert_idx 取值范围 | [0, 255] | [0, 255] |
| 精度容忍 (atol,rtol) | (0.02, 0.02) | (0.02, 0.02) |

> **关键洞察（解释无收益）**：两用例 `active_expert_range=[0,256]` 且 expert_idx∈[0,255] → **全部 9000 个 entry 都在范围内，过滤几乎不淘汰任何元素**，`filteredInBatch ≈ coreEntries_`。此时计数累加相对总耗时占比低，而 Histograms 路径新增的「向量过滤(Cast+2×CompareScalar+And+2×GatherMask)」在此场景近乎**纯开销**（过滤没剔掉东西），叠加窄化/回写，故净负收益。若 `active_expert_range` 收窄（大量 entry 被过滤淘汰），旧标量 `continue` 分支变多、pairs 循环变短，Histograms 相对优势才可能显现——这是后续复验方向。

### 8.8 代码前后修改对比（`moe_v3_counting_sort_cut_origin_base.h`）

改动全集中在 base 公共计数段，`MOE_CS_USE_HISTOGRAM` 宏 A/B 隔离；量化子类（static/dynamic/unquant）零改动。

**(1) `FilterAndCount()` —— 改前（纯标量单扫描，扫描整个 coreEntries_）**
```cpp
filteredCount_ = 0;
for (int64_t j = 0; j < coreEntries_; j++) {                          // 标量扫描全部 entry
    int32_t expertId = expertIdxLocal.GetValue(j);
    if (expertId < expertStart_ || expertId >= expertEnd_) continue;  // 标量过滤
    int32_t expertOffset = expertId - expertStart_;
    int32_t curCount = expertCountLocal.GetValue(expertOffset);       // 计数累加(a)
    expertCountLocal.SetValue(expertOffset, curCount + 1);            // 计数累加(a)
    filteredPairsLocal.SetValue(filteredCount_ * 2, j);              // pairs(b)
    filteredPairsLocal.SetValue(filteredCount_ * 2 + 1, expertOffset);// pairs(b)
    filteredCount_++;
}
```

**(1') `FilterAndCount()` —— 改后（宏分派，旧路径逐字保留在 `#else`）**
```cpp
__aicore__ inline void MoeV3CountingSortCutOriginBase<T>::FilterAndCount() {
#ifdef MOE_CS_USE_HISTOGRAM
    FilterAndCountHistogram();      // 新路径
#else
    ... 原纯标量单扫描，逐字保留 ...   // 旧路径（默认）
#endif
}
```

**(2) 新增 `FilterAndCountHistogram()`（仅宏开启）—— 向量过滤 + 直方图计数 +【pairs 与 count 分离】**
```cpp
// 1) 向量过滤（复用 FullLoad 思路）：Cast int32->fp32 -> CompareScalar(GE/LT) -> And
//    -> GatherMask 压出 gatheredExpertLocal(合法专家) + gatheredIdxLocal(原始 j)，得 filteredInBatch
// 2) 窄化：gatheredExpert -= expertStart_ (值域[0,actualExpertNum_)) -> int32->int16->uint8 两步 Cast
// 3) Histograms 计数：MoeV3CutOriginHistogramVF -> uint16 bins -> 标量回写 int32 expertCountLocal
// 4) pairs：仅在压缩结果上标量填 filteredPairs（长度 filteredInBatch < coreEntries_），【不再做 count 累加】
for (int64_t f = 0; f < filteredInBatch; f++) {
    filteredPairsLocal.SetValue(filteredCount_ * 2, gatheredIdxLocal.GetValue(f));
    filteredPairsLocal.SetValue(filteredCount_ * 2 + 1, gatheredExpertLocal.GetValue(f));
    filteredCount_++;
}
```

**(3) 新增 `__simd_vf__ MoeV3CutOriginHistogramVF`（计数核心，仅宏开启）**
```cpp
__simd_vf__ inline void MoeV3CutOriginHistogramVF(__ubuf__ uint8_t *srcAddr, __ubuf__ uint16_t *dstAddr,
                                                  uint32_t validCount) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint8_t> srcReg; RegTensor<uint16_t> dstReg0, dstReg1;
    Duplicate(dstReg0, (uint16_t)0); Duplicate(dstReg1, (uint16_t)0);
    uint16_t repeatTimes = (validCount + 255) / 256; uint32_t remain = validCount;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        MaskReg mask = UpdateMask<uint8_t>(remain);       // 尾块 mask，padding 不计入 bin
        LoadAlign(srcReg, srcAddr + 256 * i);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(dstReg0, srcReg, mask);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(dstReg1, srcReg, mask);
    }
    MaskReg maskAll = CreateMask<uint16_t, MaskPattern::ALL>();
    StoreAlign(dstAddr, dstReg0, maskAll); StoreAlign(dstAddr + 128, dstReg1, maskAll); // BIN0[0,127]+BIN1[128,255]
}
```

**(4) UB 布局新增（`ComputeCommonUbLayout()`，`#ifdef MOE_CS_USE_HISTOGRAM` 内）**
| 缓冲 | 用途 | 大小 |
|------|------|------|
| `expertIdxFp32LocalOffset_` | Cast int32→fp32（CompareScalar 用） | histEntriesAligned fp32 |
| `compareMask0/1Offset_`、`gatherMaskOffset_` | 过滤掩码 | 各 histMaskBytes |
| `gatheredExpertLocalOffset_`、`gatheredIdxLocalOffset_`、`flatIdxBufferLocalOffset_` | GatherMask 输出/输入 | 各 coreEntries int32 |
| `expertU8LocalOffset_` | 窄化 uint8 源 | Ceil(coreEntries,256)*256 uint8 |
| `histBinOffset_` | uint16 直方图输出 | 256 uint16 (=512B) |

> 向量过滤按整个 `coreEntries_` 对齐（`histEntriesAligned = Ceil(coreEntries_, 64)*64`），非 `filterChunkSize_`。宏关闭时这些成员与缓冲**完全不参与编译**，旧路径 UB 占用不变。


---

> 注：本方案仅为**性能验证**（A/B 对比），默认保留旧标量路径可回退；量化子类（static/dynamic/unquant）不改动，计数段完全在 base 公共逻辑内。

