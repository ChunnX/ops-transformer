# PR #9784 检视报告

## PR 概述

**文件**: `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h`

**目标**: 优化 `MoeDistributeDispatchV2FullMesh` 类的 `WaitAndFormatOutput` 路径，通过双缓冲（ping/pong）和细粒度事件标志同步替换全量流水线屏障，提升数据搬运与输出的流水并行度。同时简化 `CheckDataArriveWithFlag` 的标志检测逻辑。

## 主要变更

| 变更 | 说明 |
|------|------|
| `CheckDataArriveWithFlag` 简化 | 用 `DataCopy + ReduceSum`（高维接口，mask=1, srcRepStride=1）替换 `GatherMask + CompareScalar + ScalarGetSFFValue` 循环 |
| `CopyInAndOut` 双缓冲 | 新增 `dataBufferId` / `isDataBufferReused` 参数，用 `SetFlag/WaitFlag` 替换 `SyncFunc` 实现 MTE2↔MTE3 细粒度同步 |
| `WaitAndFormatOutput` 重构 | 新增 ping/pong 缓冲布局计算；移除每轮 `PipeBarrier<PIPE_ALL>()`；标志清除延迟到所有输出完成后批量执行 |
| `ClearLocalWindowDataFlags` | 新增方法，批量清除窗口数据标志，支持分块清除（受 `DATA_COPY_MAX_BLOCK_COUNT=4095` 限制） |
| `GetLocalWindowSrcDataBlockIdx` | 从内联代码提取为独立方法 |
| 移除成员变量 | 删除 `flagCompResultU8_`、`flagCompResultLtU64_`、`flagRecvGatherMask_`、`cleanUpTensor_`、`xTmpTensor_`、`flagGatherOutTensor_` 及其初始化代码 |

## 问题清单

### 1. [低] `maxCopyTokenCnt` 降为 0 时缺少保护，可能导致 UB 溢出和死循环

**位置**: `WaitAndFormatOutput` 第 1946-1957 行

```cpp
uint32_t maxCopyTokenCnt = static_cast<uint32_t>(tBufRealSize_ / minUbSizePerToken);
// ...
while (maxCopyTokenCnt > 0) {
    dataBufferSize = static_cast<uint32_t>(packedTokenSize * maxCopyTokenCnt);
    reduceWorkSize = Ceil(blockCntPerToken_ * maxCopyTokenCnt * sizeof(float), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    flagRecvSize = blockCntPerToken_ * maxCopyTokenCnt * UB_ALIGN;
    uint64_t totalBufferSize = ...;
    if (totalBufferSize <= tBufRealSize_) { break; }
    --maxCopyTokenCnt;
}
```

**问题**: 初始 `maxCopyTokenCnt` 使用未对齐的 `minUbSizePerToken` 估算（其中 reduce work 部分按 `sizeof(float)=4` 字节计算），但 while 循环内 `reduceWorkSize` 使用 `Ceil(..., 256) * 256` 对齐。对齐引入最多 255 字节填充，可能导致 `totalBufferSize > tBufRealSize_` 即使 `maxCopyTokenCnt = 1`。

当循环降至 `maxCopyTokenCnt = 0` 时退出，但 `dataBufferSize`、`reduceWorkSize`、`flagRecvSize` 仍保留 `maxCopyTokenCnt = 1` 时的值（大于实际可用空间）。后续 `tBuf.GetWithOffset` 请求的空间可能超过 `tBufRealSize_`，造成 UB 缓冲区溢出。同时主循环会因 `copyCnt = 0` / `arriveCount = 0` 永远无法推进 `finishNum`，导致死循环。

**触发条件**: `tBufRealSize_` 刚好略大于 `minUbSizePerToken`（差距 < 256 字节），且 `blockCntPerToken_ * sizeof(float)` 不是 256 的倍数。实际场景中 `tBufRealSize_` 通常远大于 `minUbSizePerToken`（~180KB vs ~1-2KB），触发概率极低。

**建议**: 循环后添加 `maxCopyTokenCnt == 0` 的保护检查，或在 `minUbSizePerToken` 计算中使用对齐后的 `reduceWorkSize`。

### 2. [信息] 标志清除时机变更——从即时清除改为延迟批量清除

**位置**: `WaitAndFormatOutput` 第 2004-2008 行

旧代码在每个 expert 的所有 token 处理完成后立即清除窗口标志：
```cpp
if (expertLeftNumTensor_(index) == 0) {
    for (uint32_t i = 0; i < expertFinishNumTensor_(index); i++) {
        DataCopy(cleanGlobal[flagIndex], cleanUpTensor_, cleanUpParams);
    }
}
```

新代码延迟到所有 expert 处理完成后统一清除：
```cpp
ClearLocalWindowDataFlags(tBuf, validNum, localExpertNum);
```

**影响**: 窗口标志在所有数据消费完毕后才被清零，发送方在此期间看到旧标志（1.0f），不会重新发送。这在单核处理范围内是正确的——接收方已消费完所有数据后才清除，发送方写入新数据时会覆盖旧标志。但如果未来有多个核共享同一窗口区域或引入超时重发机制，此变更需要重新评估。

### 3. [信息] 移除 `PipeBarrier<PIPE_ALL>()` 改变同步行为

**位置**: `WaitAndFormatOutput` 旧代码第 1942 行（已删除）

旧代码每轮迭代后执行 `PipeBarrier<PIPE_ALL>()`，确保所有流水线操作完成后再进入下一轮。新代码移除此屏障，改用事件标志（`SetFlag/WaitFlag<MTE3_V>`）实现缓冲区级别的同步。

**分析**: 新的同步方案对 `flagRecvTensor_`（V 读 → MTE2 写）和 ping/pong 缓冲区（MTE3 写 → MTE2 写）的数据依赖覆盖是正确的：
- `flagRecvTensor_` 的 V→MTE2 依赖由 `CheckDataArriveWithFlag` 中的 `SyncFunc<V_S>()` 保证（S 控制后续 MTE2 指令发射）
- ping/pong 缓冲区的 MTE3→MTE2 依赖由 `WaitFlag<MTE3_V>(dataBufferId)` 保证

移除全量屏障是合理的性能优化，但依赖发送方标志初始化协议（整个 32B 标志区域填充 1.0f）保持不变。

### 4. [信息] arch35 变体未同步更新

`arch35/moe_distribute_dispatch_v2_host_kfc.h` 仍使用旧的 `GatherMask + CompareScalar` 方式（第 1948-1961 行），未应用本 PR 的优化。如果两者是独立代码路径则无影响，但意味着优化仅覆盖非 KFC 路径。

## 正确性验证

### `CheckDataArriveWithFlag` 的 `ReduceSum` 用法

```cpp
DataCopyParams expFlagCopyParams{static_cast<uint16_t>(flagNum), 1U,
    static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE / UB_ALIGN), 0U};
DataCopy(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams);
// ...
ReduceSum(flagSumOutTensor, flagRecvTensor_, flagReduceWorkTensor_, 1U, flagNum, 1U);
```

- `DataCopyParams`: `blockLen=1`（32 字节/DataBlock），`srcGap=15`（480 字节 = SPLIT_BLOCK_DATA_SIZE），`dstGap=0`（目的连续）
- 每个源块从标志位置（SPLIT_BLOCK_DATA_SIZE 偏移）拷贝 32 字节（8 个 float），标志在首 4 字节
- `ReduceSum` 高维接口: `mask=1`（每次迭代取 1 个元素），`repeatTime=flagNum`（迭代 flagNum 次），`srcRepStride=1`（每次迭代跳 1 个 DataBlock = 8 个 float）
- 效果: 精确求和每个 32B 块的首个 float（即标志值），等价于旧代码的逐标志检查

**结论**: 正确。`ReduceSum` 的 `mask=1, srcRepStride=1` 参数精确匹配 `DataCopy` 的 32B 分块布局，逐个求和标志值。

### 双缓冲事件标志管理

追踪 `copyBatchId` 与 `dataBufferId` 的关系：

| copyBatchId | dataBufferId | isDataBufferReused | WaitFlag | SetFlag |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | false | - | MTE3_V(0) |
| 1 | 1 | false | - | MTE3_V(1) |
| 2 | 0 | true | MTE3_V(0) | MTE3_V(0) |
| 3 | 1 | true | MTE3_V(1) | MTE3_V(1) |

最终等待:
```cpp
uint32_t usedDataBufferNum = copyBatchId > LOCAL_COPY_BUFFER_NUM ? LOCAL_COPY_BUFFER_NUM : copyBatchId;
for (uint32_t dataBufferId = 0; dataBufferId < usedDataBufferNum; ++dataBufferId) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dataBufferId);
}
```

**结论**: 正确。首次使用不等待（缓冲区空闲），后续使用等待上次 MTE3 完成。最终等待所有活跃缓冲区的 MTE3 完成后才执行清除。

### `ClearLocalWindowDataFlags` 清除位置

```cpp
uint32_t flagIndex = recordOffset * SPLIT_BLOCK_COUNT + SPLIT_BLOCK_DATA_COUNT;
DataCopy(cleanGlobal[flagIndex], cleanUpTensor, cleanUpParams);
```

与旧代码 `i * SPLIT_BLOCK_COUNT * blockCntPerToken_ + SPLIT_BLOCK_DATA_COUNT` 等价（`recordOffset` 遍历所有 block 记录，总记录数 = `expertFinishNumTensor_(index) * blockCntPerToken_`）。清除范围覆盖每个 SPLIT_BLOCK 的标志区域（字节 480-511）。

**结论**: 正确。

### `CopyInAndOut` 内部同步链

```
MTE2(DataCopyPad) → SetFlag(MTE2_MTE3) → WaitFlag(MTE2_MTE3) → MTE3(CopyScalesToOut) → MTE3(DataCopyPad×2)
  → [if hasExpertScalesFlag_]:
     SetFlag(MTE3_MTE2) → WaitFlag(MTE3_MTE2) → MTE2(DataCopyPad) → SetFlag(MTE2_MTE3) → WaitFlag(MTE2_MTE3) → MTE3(DataCopyPad)
  → SetFlag(MTE3_V)
```

`CopyScalesToOut` 内部调用 `DataCopyPad`（MTE3 操作），与 `WaitFlag<MTE2_MTE3>` 的同步语义一致。

**结论**: 正确。

## 总结

| 维度 | 评价 |
|------|------|
| 正确性 | 核心逻辑正确。双缓冲同步、ReduceSum 标志检测、批量清除均经验证正确 |
| 性能 | 移除 `PipeBarrier<PIPE_ALL>`、引入双缓冲和细粒度事件标志，预期显著提升 MTE2/MTE3 流水并行度 |
| 健壮性 | 缺少 `maxCopyTokenCnt == 0` 的防御性检查（低概率触发） |
| 可维护性 | `GetLocalWindowSrcDataBlockIdx` 提取为独立方法是良好重构；移除 6 个成员变量减少了状态复杂度 |
| 一致性 | arch35 变体未同步更新，两条代码路径存在实现差异 |

**建议合入**，建议补充 `maxCopyTokenCnt == 0` 的保护。
