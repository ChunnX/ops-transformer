# moe_init_routing_v3 CountingSort FullLoad —— 按专家外循环 + k 行切批聚合搬出 概要设计

## 1. 概述

### 1.1 目标
针对 `MoeV3CountingSortFullLoadUnquantized`（A5 CountingSort FullLoad 非量化子类，模板1），在 Phase C 搬出阶段将"逐行 DataCopyPad 写 expandedXGm"优化为"按 expertOffset 外循环 + 内层按 k 行切批，每批一次 DataCopyPad 写连续 GM 段"。

### 1.2 核心原理
同一 expertOffset 内 slot 从 0 递增到 count-1，newPos 连续递增（差 1），对应 GM 地址连续。按专家外循环天然把"可连续聚合的行"圈在同一专家内，每批 min(k, 剩余) 行可一次 DataCopyPad 写出。

### 1.3 适用范围
- 仅非量化子类（`MoeV3CountingSortFullLoadUnquantized`）。
- dropPad 和 dropless 都适用（同专家内 slot 连续性两种模式都成立）。
- 通过 tiling 标志位 `csAggrEnable` 开关，默认按收益判定启用。

---

## 2. 总体修改清单

| 序号 | 文件 | 修改类型 | 说明 |
|------|------|----------|------|
| M1 | `op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h` | 新增字段 | `MoeV3Arch35CountingSortTilingData` 增 3 个聚合相关字段 |
| M2 | `op_host/moe_init_routing_v3_tiling.cpp` | 新增逻辑 | `ComputeCountingSortTiling()` 中计算聚合参数；`EstimateCountingSortFullLoadUB()` 增聚合区容量 |
| M3 | `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h` | 新增成员与方法 | UB 布局增 `gatherOutBufOffset_`；新增 `BucketByExpert()`、`GatherAndWriteByExpert()`；`GatherAndWrite` 分流 |
| M4 | `op_kernel/arch35/moe_v3_counting_sort_full_load_unquantized.h` | 调用分流 | `GatherAndWrite()` 根据 `csAggrEnable_` 调原逐行或新聚合路径 |

---

## 3. 详细修改点

### M1. tiling_def.h（新增字段）

**文件**：`op_kernel/arch35/moe_init_routing_v3_arch35_tiling_def.h`

**修改位置**：`struct MoeV3Arch35CountingSortTilingData`（第 93-105 行）

**新增字段**：
```cpp
struct MoeV3Arch35CountingSortTilingData {
    // ... 现有字段不变 ...
    int64_t csAggrEnable{0};          // 聚合搬出开关：0=逐行(现状), 1=按专家k行切批
    int64_t csAggrOutRows{0};         // 搬出聚合 UB 容纳行数 k
    int64_t csAggrOutBufBytes{0};     // 搬出聚合区字节数 = k * colsAligned * sizeof(T)
};
```

**说明**：3 个字段全部由 host 侧计算下发，kernel 侧只读取。

---

### M2. tiling.cpp（参数计算与 UB 估算）

**文件**：`op_host/moe_init_routing_v3_tiling.cpp`

#### M2.1 修改 `EstimateCountingSortFullLoadUB()`（第 872-900 行）

在函数末尾 `return ub;` 前增加聚合区容量计算逻辑：

- 读取候选聚合区字节数（建议从 xLocal 区末尾切，初始 10KB 或 `min(10*1024, xLocalCap/4)`）。
- 计算 `colsAligned`（已有）、`aggrOutRows = csAggrOutBufBytes / (colsAligned * dtypeSize)`。
- 判定启用条件（见 M2.2），决定 `csAggrEnable` 为 0 或 1。
- 若启用：`ub` 不增加（聚合区从 xLocal 区内部切分，不额外占 UB），但需校验 `xLocalCap - csAggrOutBufBytes >= coreTokenNum * cols_ * dtypeSize`（xLocal 减少后仍装得下搬入）。若校验失败则 `csAggrEnable=0` 回退。

**关键点**：聚合区**不增加总 UB 占用**，是从 xLocal 区末尾切出的，因此 `EstimateCountingSortFullLoadUB` 的返回值不变，但需新增校验逻辑确保切分后 xLocal 仍够用。

#### M2.2 修改 `ComputeCountingSortTiling()`（第 821-870 行）

在 `countingSortFullLoad_ == 1` 分支内（第 855-859 行），追加聚合参数计算：

```cpp
if (countingSortFullLoad_ == 1) {
    // ... 现有 workspace/gatherOut 设置不变 ...

    // 新增：聚合搬出参数计算
    int64_t colsAligned = Ops::Base::CeilDiv(cols_ * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE / inuptXDtypeSize_;
    int64_t xLocalCap = Ops::Base::CeilAlign(coreTokenNum * colsAligned * inuptXDtypeSize_, ONE_BLOCK_BYTE);
    int64_t aggrBufBytes = std::min(static_cast<int64_t>(10 * 1024), xLocalCap / 4); // 候选
    int64_t aggrOutRows = aggrBufBytes / (colsAligned * inuptXDtypeSize_);

    // 启用判定
    bool aggrEnable = (aggrOutRows >= 2) &&                              // k>=2，否则退化
                      (xLocalCap - aggrBufBytes >= coreTokenNum * cols_ * inuptXDtypeSize_) && // xLocal够装
                      (actualExpertNum <= COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM);                 // 桶数受限
    // 可选：基于 filteredCount 估算收益（需统计 count_e 分布，首版可不做）

    moeInitRoutingV3TilingData.countingSortParamsOp.set_csAggrEnable(aggrEnable ? 1 : 0);
    moeInitRoutingV3TilingData.countingSortParamsOp.set_csAggrOutRows(aggrEnable ? aggrOutRows : 0);
    moeInitRoutingV3TilingData.countingSortParamsOp.set_csAggrOutBufBytes(aggrEnable ? aggrBufBytes : 0);
}
```

#### M2.3 日志补充
第 864-869 行的 `OP_LOGI` 增补 `csAggrEnable / csAggrOutRows` 字段，便于调试。

---

### M3. full_load_base.h（UB 布局 + 公共方法）

**文件**：`op_kernel/arch35/moe_v3_counting_sort_full_load_base.h`

#### M3.1 新增成员变量（第 90-159 行 protected 区）

```cpp
// 聚合搬出参数（从 tiling 读取）
int64_t csAggrEnable_;
int64_t aggrOutRows_;           // k
int64_t aggrOutBufBytes_;
int64_t gatherOutBufOffset_;    // 搬出聚合区在 buf_ 中的字节偏移
// 分桶区（方案B 标量扫描用）
int64_t bucketBaseOffset_;      // 分桶基址（存 localFlatIdx，按 expert 分段）
int64_t bucketOffsetTblOffset_; // 偏移表基址（actualExpertNum_ 个 int32，每桶起始偏移）
int64_t bucketCountTblOffset_; // 每桶元素数（actualExpertNum_ 个 int32）
```

#### M3.2 修改 `InitCommon()`（第 163-226 行）

在第 189 行 `quantMode_ = tiling->quantMode;` 之后，新增聚合参数解析：

```cpp
csAggrEnable_ = tiling->countingSortParamsOp.csAggrEnable;
aggrOutRows_ = tiling->countingSortParamsOp.csAggrOutRows;
aggrOutBufBytes_ = tiling->countingSortParamsOp.csAggrOutBufBytes;
```

#### M3.3 修改 `ComputeCommonUbLayout()`（第 259-318 行）

在 xLocal 区计算（第 262-263 行）处调整：

- **xLocal 容量减少**：若 `csAggrEnable_==1`，xLocal 区大小从 `coreTokenNum_*colsAligned_` 减为 `coreTokenNum_*colsAligned_ - aggrOutBufBytes_/sizeof(T)`（元素数）。
- **新增 gatherOutBufOffset_**：紧跟 xLocal 之后（即 xLocal 减少后的末尾），`gatherOutBufOffset_ = xLocalOffset_ + xLocalReducedSize`。
- **新增分桶区**：在 `commonBufSize_` 累计之后追加：
  - `bucketBaseOffset_`：`filteredCount_` 个 int32（上限 = coreEntries_）。
  - `bucketOffsetTblOffset_`：`actualExpertNum_` 个 int32。
  - `bucketCountTblOffset_`：`actualExpertNum_` 个 int32。
  - 这三块计入 `commonBufSize_`，**会增加总 UB 占用**——需回 M2 的 UB 估算中补上，否则可能超 UB。

**注意**：分桶区是新增 UB 占用，需在 `EstimateCountingSortFullLoadUB` 中加上 `coreEntries*4 + actualExpertNum*8`，并重新校验 `fullLoadUB <= ubSize - 1024`。

#### M3.4 新增方法 `BucketByExpert()`

按方案 B（标量扫描）实现，在 `FilterAndCountVector()` 之后、`GatherAndWrite` 之前调用：

```cpp
__aicore__ inline void BucketByExpert()
{
    LocalTensor<int32_t> filteredPairsLocal = buf_.Get<int32_t>()[filteredPairsLocalOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketBase = buf_.Get<int32_t>()[bucketBaseOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketOffsetTbl = buf_.Get<int32_t>()[bucketOffsetTblOffset_ / sizeof(int32_t)];
    LocalTensor<int32_t> bucketCountTbl = buf_.Get<int32_t>()[bucketCountTblOffset_ / sizeof(int32_t)];

    // 初始化每桶 count=0，偏移表累加起始位置
    // 扫描 filteredPairsLocal：每行 (localFlatIdx, expertOffset)
    //   slot = bucketCountTbl[expertOffset]++
    //   bucketBase[bucketOffsetTbl[expertOffset] + slot] = localFlatIdx
}
```

**关键**：偏移表 `bucketOffsetTbl[e]` 需先计算每桶起始偏移（= Σ_{i<e} count_i），这要两遍扫描或一遍扫描 + 后缀累加。建议两遍：第一遍统计 count，第二遍按偏移填入。

#### M3.5 新增方法 `GatherAndWriteByExpert()`

聚合搬出主循环，替代逐行 `GatherAndWrite` 的 expandedX 部分：

```cpp
__aicore__ inline void GatherAndWriteByExpert()
{
    for (int64_t eo = 0; eo < actualExpertNum_; eo++) {
        int64_t count_e = bucketCountTbl.GetValue(eo);
        if (count_e == 0) continue;

        int64_t slotStart = 0;
        int64_t remaining = count_e;
        while (remaining > 0) {
            int64_t batchRows = Min(aggrOutRows_, remaining);

            // 1) UB→UB 拷贝 batchRows 行到 gatherOutBuf
            for (int64_t r = 0; r < batchRows; r++) {
                int32_t localFlatIdx = bucketBase.GetValue(bucketOffsetTbl.GetValue(eo) + slotStart + r);
                int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
                DataCopy(gatherOutBuf[r*colsAligned_], xLocal[tokenRow*colsAligned_], cols_);
            }
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

            // 2) 一次 DataCopyPad 写连续 GM 段
            int64_t batchStartNewPos = ComputeBatchStartNewPos(eo, slotStart); // 见 M3.6
            DataCopyExtParams xCopyParams{
                static_cast<uint16_t>(batchRows),
                static_cast<uint32_t>(cols_ * sizeof(T)), 0, 0, 0};
            DataCopyPad(expandedXGm_[batchStartNewPos * cols_], gatherOutBuf, xCopyParams, padParams);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

            // 3) 旁路穿插：rowIdx / scale / drop 逐行写（见 M3.7）
            WriteBypassForRowIdxScale(eo, slotStart, batchRows);

            slotStart += batchRows;
            remaining -= batchRows;
        }
    }
}
```

#### M3.6 新增辅助 `ComputeBatchStartNewPos()`

```cpp
__aicore__ inline int64_t ComputeBatchStartNewPos(int64_t expertOffset, int64_t slotStart)
{
    if (dropPadMode_ == DROP_PAD_MODE) {
        return (expertOffset + expertStart_) * expertCapacity_ + slotStart;
    } else {
        // dropless: prefixSumLocal[expertOffset] + slotStart
        return prefixSumLocal.GetValue(expertOffset) + slotStart;
    }
}
```

**注意**：dropless 下 prefix 来自 `ComputeGlobalOffset()` 中已计算的 `expertCountLocal[e]`（dropless 时存的是 cumulativeSum + prefixForExpert），需确认字段语义。

#### M3.7 新增旁路方法 `WriteBypassForRowIdxScale()`

聚合路径中，rowIdx / scale / drop 不能与 expandedX 聚合复用，保持逐行：

```cpp
__aicore__ inline void WriteBypassForRowIdxScale(int64_t eo, int64_t slotStart, int64_t batchRows)
{
    LocalTensor<int32_t> idxBuf = buf_.Get<int32_t>()[prefixSumLocalOffset_ / sizeof(int32_t)];
    DataCopyExtParams idxCopyParams{1, sizeof(int32_t), 0, 0, 0};

    for (int64_t r = 0; r < batchRows; r++) {
        int32_t localFlatIdx = bucketBase.GetValue(bucketOffsetTbl.GetValue(eo) + slotStart + r);
        int64_t tokenRow = static_cast<int64_t>(localFlatIdx) / k_;
        int64_t origFlatIdx = coreFlatStart_ + static_cast<int64_t>(localFlatIdx);
        int64_t newPos = ComputeBatchStartNewPos(eo, slotStart + r);

        // expandedRowIdx 写（GATHER/SCATTER 不同地址，逐行）
        if (rowIdxType_ == GATHER) {
            idxBuf.SetValue(0, static_cast<int32_t>(newPos));
        } else {
            idxBuf.SetValue(0, static_cast<int32_t>(origFlatIdx));
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(expandedRowIdxGm_[rowIdxType_==GATHER ? origFlatIdx : newPos], idxBuf, idxCopyParams);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);

        // scale 写（isInputScale_ 时）
        if (isInputScale_) {
            DataCopyExtParams scaleCopyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPad(expandedScaleGm_[newPos], scaleLocal[tokenRow*scaleSlotSize_], scaleCopyParams);
        }
    }
}
```

**drop 行处理**（dropPad slot>=capacity）：在 `BucketByExpert()` 阶段就排除（只桶装 slot<capacity 的行），drop 行的 rowIdx=-1 写仍按原 `GatherAndWrite` 逐行逻辑在聚合路径之前/之后单独处理。

---

### M4. full_load_unquantized.h（分流调用）

**文件**：`op_kernel/arch35/moe_v3_counting_sort_full_load_unquantized.h`

#### M4.1 修改 `GatherAndWrite()`（第 132-171 行）

在函数入口加分流：

```cpp
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadUnquantized<T>::GatherAndWrite()
{
    if (this->csAggrEnable_) {
        this->BucketByExpert();
        this->GatherAndWriteByExpert();
        // drop 行的 rowIdx=-1 写（dropPad 模式）：单独遍历处理
        this->WriteDroppedRowIdx(); // 新增方法，见下
        return;
    }
    // ... 原 filteredPairsLocal 逐行循环逻辑不变 ...
}
```

#### M4.2 新增 `WriteDroppedRowIdx()`（dropPad 专用）

dropPad 模式下 slot>=capacity 的行不进桶（不写 expandedX），但 GATHER 模式需写 `expandedRowIdxGm[origFlatIdx]=-1`。需在 `BucketByExpert` 时记录被 drop 的 origFlatIdx 列表，或重新扫描 filteredPairsLocal 判定 drop：

```cpp
__aicore__ inline void WriteDroppedRowIdx()
{
    if (dropPadMode_ != DROP_PAD_MODE || rowIdxType_ != GATHER) return;
    // 扫描 filteredPairsLocal，对 slot>=expertCapacity_ 的行写 rowIdx=-1
    // （slot 需动态累计，与 BucketByExpert 共用逻辑）
}
```

**简化建议**：首版可在 `BucketByExpert()` 中同时处理 drop 行（进桶前判断，drop 的直接写 rowIdx=-1），避免二次扫描。

#### M4.3 Process() 不变
`Process()` 第 174-199 行流程不变，`GatherAndWrite()` 内部分流即可。

---

## 4. UB 布局总览（修改后）

```
|<- xLocal (搬入，容量减少 aggrOutBufBytes_) ->|<- gatherOutBuf (k行) ->|
                                                  ^gatherOutBufOffset_
... 之后是原有公共区 ...
... 末尾新增 ...
|<- bucketBase (filteredCount 个 int32) ->|<- bucketOffsetTbl (actualExpertNum 个 int32) ->|
|<- bucketCountTbl (actualExpertNum 个 int32) ->|
```

**总 UB 占用变化**：
- xLocal 区减少 `aggrOutBufBytes_`。
- 新增 gatherOutBuf `aggrOutBufBytes_`（与 xLocal 减少抵消，净 0）。
- 新增分桶区 `coreEntries*4 + actualExpertNum*8` 字节（净增加）。

→ `EstimateCountingSortFullLoadUB` 需补上分桶区容量并重新校验 `<= ubSize - 1024`。

---

## 5. 同步与流水依赖

### 5.1 关键同步点
| 位置 | 同步 | 原因 |
|------|------|------|
| xLocal 读之前 | `WaitXLoadCommon()`（MTE2→S/MTE3）已有 | 搬入完成才能读 |
| UB→UB 拷贝后、DataCopyPad 前 | `SetWaitFlag<V_MTE3>` 新增 | V 写完 UB 才能 MTE3 读 |
| 每批 DataCopyPad 后 | `SetWaitFlag<MTE3_S>` | MTE3 写完才能下一轮 S 操作 |
| 批次间 | `PipeBarrier<PIPE_V>` | UB→UB 拷贝顺序 |
| rowIdx/scale 逐行写前后 | `S_MTE3` / `MTE3_S` | 与原 GatherOneRow 一致 |

### 5.2 流水并行影响
- 原方案：Phase C 是 MTE3 密集（filteredCount_ 次）。
- 新方案：Phase C 变 V（UB→UB 拷贝）+ MTE3（批次写出）+ S（rowIdx/scale 标量）混合，流水更复杂，需注意 V 与 MTE3 的串行依赖（V→MTE3 同步）。

---

## 6. 验证计划

### 6.1 精度验证
- 单 case 对齐 baseline（逐行路径）：dropPad + dropless、有/无 scale、GATHER/SCATTER。
- 边界：count_e=0、count_e<k、count_e=k 的整数倍、filteredCount_=1、actualExpertNum=1。
- 全量 coverage：cs_coverage.csv 用例集。

### 6.2 性能验证
- 对比指标：Phase C MTE3 指令数、总耗时、UB 占用。
- 典型 case：dropPad/DROP_LESS × cols={128,1024,4096} × n={512,4096,16384} × actualExpertNum={4,16,64}。
- 关注负优化场景：cols 大（k=1 退化）、filteredCount 小。

### 6.3 开关回退
- `csAggrEnable=0` 必须等价于现状（逐行路径），确保开关可安全回退。

---

## 7. 风险与待确认项

| 项 | 风险 | 待确认 |
|----|------|--------|
| 分桶区 UB 占用 | actualExpertNum 大时偏移表占 UB 多 | 在 EstimateUB 中校验，超限则 csAggrEnable=0 |
| dropless prefix 语义 | `ComputeGlobalOffset` 中 dropless 的 expertCountLocal[e] 是否 = prefix+slot 起始 | 需读 ComputeGlobalOffset 确认（M3.6 依赖） |
| UB→UB DataCopy 性能 | cols 大时 V 流水压力 | 实测对比 |
| BucketByExpert 两遍扫描 | 标量循环开销 | filteredCount_ 大时可能慢，备选方案 A 向量抽取 |
| drop 行二次扫描 | 若不合并到 BucketByExpert，需额外遍历 | 首版合并处理避免二次扫描 |
| xLocal 容量减少导致 tiling 调整 | filterPerCoreTokens 可能被迫减小 | 影响 filterNeedCoreNum，需 tiling 层联动 |

---

## 8. 实施顺序建议

1. **M1**：tiling_def.h 加 3 字段（最小改动，先打通链路）。
2. **M2**：tiling.cpp 计算参数 + UB 估算校验（host 侧可单独验证日志）。
3. **M3.1-M3.3**：base.h 成员 + InitCommon 解析 + UB 布局调整（先跑通编译）。
4. **M3.4**：BucketByExpert（分桶，先方案 B 标量）。
5. **M3.6**：ComputeBatchStartNewPos（确认 dropless prefix 语义）。
6. **M3.7**：WriteBypassForRowIdxScale（旁路逐行）。
7. **M3.5**：GatherAndWriteByExpert（主循环）。
8. **M4**：unquantized.h 分流 + WriteDroppedRowIdx。
9. **验证**：单 case 精度 → coverage → 性能对比。
10. **调优**：若标量分桶慢，换方案 A 向量抽取；若 k 退化，动态调整 aggrBufBytes。

---

## 9. 预期收益

- expandedX 的 MTE3 指令数：`filteredCount_` → `Σ_e ceil(count_e / k)`，典型场景减少 80%+。
- 代价：分桶扫描 + UB→UB 拷贝 + 旁路逐行（rowIdx/scale/drop 不被优化）。
- 净收益：取决于 cols_/filteredCount_/actualExpertNum_ 分布，需实测。若旁路占比高（isInputScale_ 或 GATHER），整体收益被稀释，可能仅 expandedX 部分有收益。

---

## 10. 与前序分析文档的关系

- `counting_sort_fullload_in_out_asymmetry.md`：搬入搬出不对称的现象分析（问题定义）。
- `counting_sort_fullload_gather_out_aggregation.md`：方案演进（第一版错误→第二版修正→第三版定结构）。
- 本文档（`counting_sort_fullload_aggr_gather_out_design.md`）：基于第三版结构的可实施概要设计，明确所有修改点。
