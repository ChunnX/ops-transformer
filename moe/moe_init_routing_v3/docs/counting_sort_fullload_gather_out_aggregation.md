# moe_init_routing_v3 CountingSort FullLoad —— 按专家外循环 + k 行切批聚合搬出方案

## 1. 方案核心结构（按你的设计）

### 1.1 主循环骨架
```
for (expertOffset = 0; expertOffset < actualExpertNum_; expertOffset++) {
    // 1) 从 filteredPairsLocal 中抽出所有属于当前 expertOffset 的行（tokenRow + slot）
    // 2) 对该专家的 slot 段按 k 切批，每批一次 DataCopyPad 搬出
    while (该专家剩余 > 0) {
        batchRows = min(k, 该专家剩余);
        // UB→UB 拷贝 batchRows 行到 gatherOutBuf
        DataCopyPad(expandedXGm[batchStartNewPos*cols], gatherOutBuf, {blockCount:batchRows, ...});
    }
}
```

- **外层**：按 `expertOffset` 0..actualExpertNum-1 遍历每个专家。
- **内层**：对当前专家的 slot 段按 k 切批，每批 `batchRows = min(k, 剩余)` 行，一次 DataCopyPad 写连续 GM 段。

### 1.2 与前面方案的关系
这是本系列分析的第三版。前两版围绕"凑连续段"的难点，本版你直接给出结构：**外层专家循环 + 内层 k 行切批**。下文按此结构细化为可实施分析。

---

## 2. 关键前提：为什么按专家外循环天然满足 GM 连续

### 2.1 dropPad 模式 newPos 公式
```
newPos = (expertOffset + expertStart_) * expertCapacity_ + slot
```
- 同一 expertOffset 内，slot 从 0 递增到 count-1，**newPos 连续递增**（差 1）。
- 因此同一专家内的任意 slot 段 `[slot_start, slot_start+batchRows)` 对应的 GM 地址连续，一次 DataCopyPad 可写。

### 2.2 跨专家不连续但不需要跨
外层切到下一个专家时 `batchStartNewPos` 重置为新专家的起始，本就重新开新批次，**不存在跨专家凑齐的问题**——这正是按专家外循环结构的好处：天然把"可连续聚合的行"圈在同一专家内。

### 2.3 dropless 模式 newPos
```
newPos = slot   (全局 prefix sum 累积位置)
```
- 同一专家内 slot 递增但 newPos = 跨专家 prefix + slot，**同专家内仍连续**（slot 差 1 → newPos 差 1）。
- 所以 dropless 下按专家外循环也成立——同专家内 slot 段依然 GM 连续。
- 修正前面结论：**dropless 也可用此结构**，连续性来自"同专家内 slot 递增"，与 dropPad 一样。

→ 本方案**对 dropPad 和 dropless 都成立**（区别仅在 newPos 公式与边界）。

---

## 3. 难点：如何找到当前专家的 (tokenRow, slot) 行

你选了"从 filteredPairs 向量抽取"。当前 `filteredPairsLocal` 布局是 token-major，存 `(localFlatIdx, expertOffset)` 对，slot 没有显式存储（原实现是动态累加）。要按专家抽取，需要：

### 3.1 方案 A：CompareScalar + GatherMask 抽取（向量）
对每个 expertOffset 做一次：
```
// mask = (filteredPairsLocal[2i+1] == expertOffset)
CompareScalar(mask, filteredPairsLocal[奇数列], expertOffset, EQ, filteredCount_);
// 用 mask 做 GatherMask 把匹配行的 localFlatIdx 压缩到 curExpertRows[]
```
- **开销**：每个专家一次 CompareScalar + 一次 GatherMask，共 actualExpertNum_ 次。
- **优势**：纯向量，无标量循环，适合 actualExpertNum_ 较多时。
- **风险**：actualExpertNum_ 可能很大（如 64+），向量指令次数线性增长；filteredPairsLocal 是 interleaved 布局（localFlatIdx/expertOffset 交替），需用 stride 访问奇数列。

### 3.2 方案 B：单次扫描 + 按专家分桶（标量）
扫描 filteredPairsLocal 一次，把每个 expertOffset 的 localFlatIdx 追加到按专家分桶的小数组：
```
for (i = 0; i < filteredCount_; i++) {
    eo = filteredPairsLocal[2i+1];
    lidx = filteredPairsLocal[2i];
    bucket[eo].push(lidx);          // bucket[eo] 是 UB 内连续区
}
```
- **开销**：filteredCount_ 次标量 SetValue（S 流水），一次扫描。
- **优势**：开销与 actualExpertNum_ 无关，只随 filteredCount_ 增长；结构简单。
- **风险**：标量循环慢；需在 UB 内开 actualExpertNum_ 个桶，每个桶长度不定，需用偏移表管理。

### 3.3 slot 生成
两种方案都需补 slot。在分桶/抽取时，slot = 该专家内行的出现序号（0,1,2,...）。扫描时维护一个 `curSlot[expertOffset]` 数组，每次追加时 slot=curSlot[eo]++。这等价于 stable counting。

### 3.4 推荐选择
- actualExpertNum_ 小（≤16）、filteredCount_ 大：方案 A（向量）更快。
- actualExpertNum_ 大、filteredCount_ 中等：方案 B（标量扫描）更稳。
- 实测对比；建议默认方案 B，结构简单易调试，性能不够再换 A。

---

## 4. UB 布局

### 4.1 新增区域（从 xLocal 末尾切 k 行）
```
|<--- xLocal (搬入，减 k*colsAligned_*sizeof(T)) --->|<--- gatherOutBuf (k 行) --->|
                                                       ^gatherOutBufOffset_
```
- `k = floor(切分字节数 / (colsAligned_*sizeof(T)))`。
- 校验：xLocal 减少后仍装得下 coreTokenNum_ 行搬入。

### 4.2 分桶区（方案 B 需要）
- 在 commonBuf 区追加 `bucketBaseOffset_` 和 `bucketOffset_[actualExpertNum_]`（偏移表）。
- 总容量 = filteredCount_ 个 int32（与 filteredPairsLocal 同量，可复用其空间）。

### 4.3 当前专家行索引区
- 方案 A：`curExpertRowsOffset_`，存 GatherMask 压缩后的当前专家行 localFlatIdx，上限 = 该专家 count。
- 方案 B：直接遍历桶，无需额外压缩区。

---

## 5. 单批搬出详细流程（内层 while）

对当前 expertOffset，已知其 count_e 和分桶后的行索引序列：

```text
slot_start = 0
remaining = count_e
while (remaining > 0):
    batchRows = min(k, remaining)
    # 1) UB→UB 拷贝 batchRows 行到 gatherOutBuf
    for (r = 0; r < batchRows; r++):
        tokenRow = curExpertRows[slot_start + r] / k_   # localFlatIdx→tokenRow
        DataCopy(gatherOutBuf[r*colsAligned_], xLocal[tokenRow*colsAligned_], cols)  # UB→UB
    PipeBarrier<PIPE_V>()
    # 2) 一次 DataCopyPad 写连续 GM 段
    batchStartNewPos = newPosOf(expertOffset, slot_start)   # dropPad: (eo+es)*cap+slot_start; dropless: prefix[eo]+slot_start
    DataCopyPad(expandedXGm[batchStartNewPos*cols], gatherOutBuf,
                {blockCount: batchRows, blockLen: cols*sizeof(T), srcStride:0, dstStride:0})
    slot_start += batchRows
    remaining -= batchRows
```

### 5.1 UB→UB 拷贝的方式
- 逐行 `DataCopy`（UB→UB，V 流水）：batchRows 次。
- 或向量化 `Gather`/`Brcb`：若 xLocal 行内连续可一次搬多行（需校验地址对齐）。
- batchRows 小（≤k，k 通常 4~16）时逐行 DataCopy 足够。

### 5.2 newPos 计算
- dropPad：`batchStartNewPos = (expertOffset + expertStart_) * expertCapacity_ + slot_start`
- dropless：`batchStartNewPos = prefixSumLocal[expertOffset] + slot_start`（prefix 来自 ComputeGlobalOffset）

### 5.3 同步
- xLocal 读必须在 `WaitXLoadCommon`（MTE2→S）之后。
- UB→UB 拷贝后需 `SetWaitFlag<HardEvent::V_MTE3>` 再 DataCopyPad。
- 批次间 `PipeBarrier<PIPE_V>`。

---

## 6. 旁路处理（不可聚合的部分）

按专家外循环结构，旁路在何处穿插：

### 6.1 drop 行（dropPad，slot >= expertCapacity_）
- 这些行不写 expandedX，只写 expandedRowIdxGm=-1（GATHER 模式）。
- 在分桶/抽取阶段就应排除（slot < capacity 才进桶），或在循环内判断。
- 保持逐行写 rowIdx=-1，无法聚合。

### 6.2 expandedRowIdxGm 写（每行一个 int32）
- 原 GatherOneRow 中：GATHER 模式写 `expandedRowIdxGm[origFlatIdx]=newPos`，SCATTER 模式写 `expandedRowIdxGm[newPos]=origFlatIdx`。
- 这是单元素写，无法与 expandedX 行聚合复用。
- **建议**：在按专家循环中，对当前专家的每行仍逐行写 rowIdx（穿插在批次内），保持正确性。
- 若 rowIdx 也想聚合：SCATTER 模式下 `expandedRowIdxGm[newPos]` 同专家内 newPos 连续，可另开 int32 聚合区按 k 切批；GATHER 模式写 `[origFlatIdx]` 不连续，不可聚合。复杂度高，**不建议首版做**。

### 6.3 expandedScaleGm 写（isInputScale_，单 float）
- 同 rowIdx，单元素写，无法与 expandedX 聚合。
- 保持逐行穿插写。

### 6.4 旁路稀释收益
若 isInputScale_ 为真或 GATHER 模式，每行还有 rowIdx/scale 写，MTE3 指令数 = `filteredCount_（rowIdx）+ filteredCount_（scale）+ Σceil(count_e/k)（expandedX）`。仅 expandedX 部分被优化，rowIdx/scale 仍逐行。需评估这两部分占比。

---

## 7. 收益量化

### 7.1 MTE3 指令数对比（仅 expandedX）
- 现状：`filteredCount_` 次。
- 方案后：`Σ_e ceil(count_e / k)` 次。

### 7.2 典型估算
dropPad、expertCapacity=64、filteredCount=256、8 专家均匀（count_e=32）、k=8：
- 现状 256 次 → 方案后 `8 * ceil(32/8) = 32` 次，**减少 87.5%**。

count_e 普遍 < k（专家多每专家少）、k=8、count_e=4、8 专家：
- 方案后 `8 * ceil(4/8) = 8` 次（每批 4 行凑不满 k），仍减少 96.9%，但 UB 利用率低。

### 7.3 新增开销
- 分桶/抽取：filteredCount_ 次标量 SetValue（方案 B）或 actualExpertNum_ 次向量（方案 A）。
- UB→UB 拷贝：filteredCount_ 次 V 流水（逐行 DataCopy）。
- 同步：每批 V→MTE3 + 批间 PipeBarrier。

### 7.4 净收益判定
若 expandedX MTE3 减少量 × 单次 MTE3 延迟 > 新增分桶+UB拷贝+同步开销，则正收益。需结合单次 MTE3 延迟（与 cols_ 强相关）评估。

---

## 8. 适用判定与风险

### 8.1 适用条件（建议 tiling 加 `csAggrEnable` 开关）
1. `colsAligned_*sizeof(T) <= 切分字节/2`（k>=2，否则退化）。
2. `filteredCount_ > 阈值`（如 16，否则排序/分桶开销占比高）。
3. 典型 case 下 `count_e` 中位数 >= k/2（凑齐有效）。
4. dropPad 和 dropless 都可启用（同专家内 slot 连续性两种模式都成立）。

### 8.2 主要风险
- **分桶/抽取开销**：actualExpertNum_ 大时方案 A 向量指令多，filteredCount_ 大时方案 B 标量慢。需选型 + 实测。
- **UB→UB 拷贝**：cols_ 大时 V 流水压力，可能抵消 MTE3 节省。
- **xLocal 容量减少**：可能迫使 tiling 减小 filterPerCoreTokens_。
- **旁路未优化**：rowIdx/scale 逐行，isInputScale_ 或 GATHER 模式下收益被稀释。
- **UB 内桶管理**：方案 B 需偏移表，actualExpertNum_ 大时偏移表本身占 UB。

### 8.3 前置 profiling 建议
实施前离线统计典型 case：
- 每专家 count_e 直方图、filteredCount_ 总量、cols_、actualExpertNum_。
- 若 `Σ ceil(count_e/k) / filteredCount_ < 0.5` 且新增开销可接受 → 推进；否则收益有限。

---

## 9. 实现步骤概要（不含代码）

1. **tiling**：加 `csAggrEnable`、`gatherOutRows`(=k)、`gatherOutBufBytes` 字段；校验 xLocal 容量。
2. **UB 布局**：`ComputeCommonUbLayout` 在 xLocalOffset_ 段末尾切 `gatherOutBufOffset_` 预留 k 行；方案 B 追加分桶区与偏移表。
3. **分桶/抽取方法**：
   - 方案 B（推荐首版）：`BucketByExpert()` —— 扫描 filteredPairsLocal，按 expertOffset 分桶，每桶存 localFlatIdx，同时累计 slot。
4. **聚合搬出方法**：`GatherAndWriteByExpert()` —— 外层 expertOffset 循环，内层按 k 切批，UB→UB 拷贝 + DataCopyPad。
5. **旁路穿插**：drop 行、rowIdx、scale 在循环内按原逐行逻辑处理。
6. **GatherAndWrite 分流**：`csAggrEnable` 为真走聚合路径，否则走原逐行。
7. **同步**：聚合路径加 V→MTE3、批间 PipeBarrier<PIPE_V>。
8. **验证**：单 case 精度对齐 baseline → coverage → 性能对比。

---

## 10. 结论

### 方案评价
**结构清晰、可行性高**，按专家外循环 + 内层 k 行切批，天然利用"同专家内 slot 连续 → GM 连续"特性，避免了跨专家凑齐的难题（前两版的核心痛点）。**对 dropPad 和 dropless 都成立**。

### 核心要点
1. **按专家外循环**把可连续聚合的行圈在同一专家内，天然满足 GM 连续性。
2. **分桶/抽取**是必要前置（filteredPairs 是 token-major），推荐首版用方案 B 标量扫描，简单易调试。
3. **k 行切批**：每批 `min(k, 剩余)`，一次 DataCopyPad 写连续 GM 段。
4. **旁路（drop/rowIdx/scale）不可聚合**，保持逐行穿插，会稀释收益。

### 收益
expandedX 的 MTE3 指令数从 `filteredCount_` 降到 `Σ_e ceil(count_e/k)`，典型场景减少 80%+。净收益取决于新增的分桶+UB 拷贝开销是否能被 MTE3 节省覆盖，需实测。

### 实施门槛
先做离线 profiling 统计 count_e 分布，若 MTE3 减少率 ≥ 50% 且新增开销可接受 → 推进。否则当前算子语义下收益有限，应考虑更 upstream 的方向（核间按专家切分）。

### 与前两版差异
- **第一版**：按 newPos 升序紧凑凑满 k → 错误，跨跳跃段 GM 不连续。
- **第二版**：按 GM 连续性凑 k（newPos 差 1 才同批）→ 正确但结构散，未给出主循环。
- **第三版（本版）**：按专家外循环 + 内层 k 切批 → 结构清晰、天然满足连续性、可实施。
