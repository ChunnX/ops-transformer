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

> 注：本方案仅为**性能验证**（A/B 对比），默认保留旧标量路径可回退；量化子类（static/dynamic/unquant）不改动，计数段完全在 base 公共逻辑内。
