# attention_to_ffn_v2 / ffn_to_attention_v2 增加 URMA 通信通路设计说明书

## 1. 背景与目标

### 1.1 现状

`mega_moe` 算子已支持两种通信通路：

| 通路 | 模板文件 | TilingKey | 通信机制 |
|------|---------|-----------|---------|
| UB-MEM (MTE) | `mega_moe.h` | `TILINGKEY_TPL_MTE (0)` | Peermem 窗口直接 DataCopyPad 读写 |
| URMA | `mega_moe_layered.h` | `TILINGKEY_TPL_URMA (1)` | `Hcomm<COMM_PROTOCOL_UBC_CTP>` URMA 通道 + Server 分层中转 |

`attention_to_ffn_v2` 和 `ffn_to_attention_v2` 当前仅支持 UB-MEM 通路（`TILINGKEY_TPL_A5`），通过 V1 模板 `attention_to_ffn.h` / `ffn_to_attention_v2.h` 实现，使用 `GetWindowAddrByRankId` + `DataCopyPad` 直接写远端卡 win 区。

### 1.2 目标

为 `attention_to_ffn_v2` 和 `ffn_to_attention_v2` 增加 URMA 通信通路，将 token 传输方式从 Peermem 直接 `DataCopyPad` 改为 URMA 通道 `hcomm_.WriteWithNotifyNbi` 发送。

**关键约束：这两个算子不跨 server，不需要分层中转/relay 架构。** 通信模型保持 "源卡直接发送到目标卡"，仅替换底层传输方式。

### 1.3 设计原则

- **不引入 server 分层**：不使用 `mega_moe_layered.h` 的 relay/mask/dispatch L1-L2 中转机制，保持扁平 rank 通信模型。
- **最小化改动**：新建 URMA 模板文件，现有 UB-MEM 模板不动。URMA 模板复用现有类结构，仅替换通信原语。
- **win 区布局不变**：infoTable + tokenData 两段布局保持不变，URMA 只是换一种方式写远端 win。
- **保持 V1/V2 tiling data 布局兼容**：URMA 模式标识通过 tiling key 区分，不扩展 tiling data 结构。

---

## 2. UB-MEM vs URMA 通信原语对照

### 2.1 核心差异（仅传输方式，不含分层）

| 维度 | UB-MEM（当前） | URMA（目标） |
|------|---------------|-------------|
| 跨卡写数据 | `DataCopyPad(remoteGlobalTensor, localTensor, ...)` | `hcomm_.WriteNbi(channel, remoteAddr, localAddr, bytes)` |
| 跨卡写数据+通知 | `DataCopyPad` 数据 + `DataCopyPad` flag（两步） | `hcomm_.WriteWithNotifyNbi(channel, dst, src, bytes, flagAddr, flagVal)`（一步原子） |
| 通道获取 | 无 | `GetUrmaCommHandle(mc2Context_, rankId, epRankId)` |
| 通道冲刷 | 无 | `hcomm_.Drain(channel)` |
| UB 占用 | 无额外 | 512B hcomm 初始化区 |
| 数据源地址 | 必须 GlobalTensor | GM 地址即可（本地 win 或 workspace） |

### 2.2 不需要的 mega_moe URMA 机制

以下 `mega_moe_layered.h` 中的机制**本设计不涉及**：

| 机制 | 原因 |
|------|------|
| Server 拓扑（`rankPerServer_`/`serverNum_`/`serverId_`/`rankIdInServer_`） | 不跨 server |
| relay 中转（`QuantTokenToLocalRelay`/`CopyTokensFromLocalRelay`/`CopyTokensFromRemoteRelay`） | 不跨 server，直接发送 |
| mask slot + count 机制（`maskSlotPtr`/`maskSlotSize_`/`SendMaskCal` 分轮次） | attention_to_ffn 已知目标 rank，不需 mask 广播 |
| dispatch L1/L2 通信区（`dispatchL1CommPtr`/`dispatchL2CommPtr`/`dispatchCursorPtr`/`dispatchDonePtr`） | 不分层中转 |
| token 去重（`sendDedupMaskTensor_`） | 不分层，无重复发送 |
| `CrossRankSyncInWorldSize` URMA 全卡同步 | 用 `SyncAll` 即可 |
| compute/sender 核拆分（`DispatchTokenToRmtServer`/`DispatchComputeKernel`/`DispatchSendKernel`） | 不分层 |

---

## 3. attention_to_ffn_v2 URMA 适配设计

### 3.1 当前通信流程（UB-MEM）

`attention_to_ffn.h` 的 `SendTokenToFFNByTokenIdx` 核心流程（`attention_to_ffn.h:613-667`）：

```
1. xInTensor_ = xQueue_.AllocTensor()
2. DataCopyPad(xInTensor_, xGMTensor_[tokenId * axisH_], ...)   // MTE2: 从 GM 读 token
3. SetExpertAndRank(tokenIdx, tokenId, topkId)                   // Scalar: 查表得 toRankId_, localExpId_
4. QuantProcess(dstExpertId_)                                    // Vector: 量化（如需）
5. CheckFlagAndSetTableGM(toRankId_, toRankAddr, ...)            // MTE2: 轮询目标卡 win flag → MTE3
6. DataCopyPad(tokenDataGMTensor, xOutTensor_, ...)              // MTE3: 直接写远端 win tokenData
7. DataCopyPad(tokenInfoTableGMTensor, statusTensor_, ...)       // MTE3: 直接写远端 win infoTable flag
```

关键点：
- `GetWindowAddrByRankId(rankId)` 获取目标卡 win 基址（`attention_to_ffn.h:581-592`）
- `DataCopyPad` 直接写远端 win 的 `winOffset_[1]` (tokenData) 和 `winInfoTableOffset_` (infoTable)
- `CheckFlagAndSetTableGM` 通过 `DataCopy` 读远端 win flag 位实现同步（`attention_to_ffn.h:595-610`）

### 3.2 URMA 改造方案

**核心思路**：保持现有流程骨架不变，将写远端 win 的 `DataCopyPad` 替换为 URMA 通道写。

#### 3.2.1 改造对照表

| 步骤 | UB-MEM 实现 | URMA 实现 | 改动类型 |
|------|------------|-----------|---------|
| 读 token | `DataCopyPad(xInTensor_, xGMTensor_[...], ...)` | 不变 | 无 |
| 量化 | `QuantProcess(dstExpertId_)` | 不变 | 无 |
| 轮询目标卡 flag | `DataCopy(ffnFlagTensor_, tokenInfoTableGMTensor, ...)` 读远端 win | `ReadGmByPassDCache(remoteFlagPtr)` 直接读远端 GM | **改造** |
| 写 token 数据 | `DataCopyPad(tokenDataGMTensor, xOutTensor_, hCommuCopyOutParams)` | `hcomm_.WriteNbi(channel, remoteDataAddr, localDataAddr, hCommuSize_)` | **替换** |
| 写 infoTable flag | `DataCopyPad(tokenInfoTableGMTensor, statusTensor_, statusCopyParams)` | 合并到上面用 `WriteWithNotifyNbi` 一步完成 | **合并** |

#### 3.2.2 具体适配点

##### 3.2.2.1 Tiling Key 扩展

**文件**: `op_kernel/attention_to_ffn_v2_tiling_key.h`

当前 tiling key 末尾维度为 `ARCH_TAG`（值 `TILINGKEY_TPL_A5`）。在其后增加 `COMM_MODE` 维度：

```cpp
#define TILINGKEY_COMM_MTE 0
#define TILINGKEY_COMM_URMA 1

ASCENDC_TPL_ARGS_DECL(AttentionToFfnV2,
    ASCENDC_TPL_UINT_DECL(ATTN_FFN_QUANT_MODE, ...),
    ASCENDC_TPL_UINT_DECL(ATTN_FFN_OUT_DTYPE, ...),
    ASCENDC_TPL_BOOL_DECL(ATTN_FFN_SCALE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_SYNC, 0, 1),
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_ACTIVE_MASK, 0, 1),
    ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
    ASCENDC_TPL_UINT_DECL(COMM_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                          TILINGKEY_COMM_MTE, TILINGKEY_COMM_URMA), );

ASCENDC_TPL_SEL(
    // ... 现有 MTE 选择 ...
    ASCENDC_TPL_ARGS_SEL(..., ASCENDC_TPL_UINT_SEL(COMM_MODE, ..., TILINGKEY_COMM_URMA),
                         ASCENDC_TPL_TILING_STRUCT_SEL(AttentionToFfnV2TilingData)), );
```

##### 3.2.2.2 Kernel Entry 扩展

**文件**: `op_kernel/arch35/attention_to_ffn_v2.cpp`

增加 `CommModeType` 模板参数，在 `if constexpr (ArchTag == TILINGKEY_TPL_A5)` 内分支：

```cpp
template <uint8_t QuantMode, uint8_t OutDtype, bool ScaleMode, bool isSync,
          bool IsActiveMask, uint8_t ArchTag, uint8_t CommModeType>
__global__ __aicore__ void attention_to_ffn_v2(...)
{
    REGISTER_TILING_DEFAULT(AttentionToFfnV2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5 && CommMode == TILINGKEY_COMM_URMA",
                                  AttentionToFfnV2TilingData);
    TPipe pipe;

    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(AttentionToFfnV2TilingData, tilingData, tilingGM);
        const auto *v1TilingData = reinterpret_cast<const AttentionToFFNTilingData *>(&tilingData);
        if constexpr (CommModeType == TILINGKEY_COMM_MTE) {
            // 现有路径：复用 V1 AttentionToFFN 模板
            AttentionToFFN<DTYPE_X, ..., AttentionToFFNContextType::MC2_CONTEXT> op;
            op.Init(...);
            op.Process();
        } else if constexpr (CommModeType == TILINGKEY_COMM_URMA) {
            // 新增路径：URMA 模板
            AttentionToFFN<DTYPE_X, ..., AttentionToFFNContextType::MC2_CONTEXT, true> op;
            op.Init(...);
            op.Process();
        }
    }
}
```

**设计选择**：不新建独立模板类，而是在现有 `AttentionToFFN` 类中通过 `bool IsUrma` 模板参数做编译期分支。这样最大化复用现有代码，仅差异部分走 `if constexpr`。

##### 3.2.2.3 AttentionToFFN 类模板扩展

**文件**: `attention_to_ffn/op_kernel/attention_to_ffn.h`

在模板参数中增加 `bool IsUrma = false`：

```cpp
#define TemplateAttentionToFFNTypeClass \
    typename XType, typename XOutType, uint32_t QuantMode, bool isSync, bool isActiveMask, \
        AttentionToFFNContextType contextType, bool IsUrma
```

新增成员：

| 新增成员 | 类型 | 用途 | 参照 |
|---------|------|------|------|
| `hcomm_` | `Hcomm<COMM_PROTOCOL_UBC_CTP>` | URMA 通道对象 | `mega_moe_layered.h:161` |

新增 `#include`（仅 IsUrma=true 时需要）：
```cpp
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif
```

##### 3.2.2.4 Init() 适配

**文件**: `attention_to_ffn.h` — `Init()` 函数

改动点：

1. **Hcomm 初始化**（仅 IsUrma）：
   ```cpp
   if constexpr (IsUrma) {
       // 从 context 获取 hcommHandle（需 context 支持该字段）
       // hcomm_ 的 Init 需要一个 512B UB 区，在 QuantInit/普通 buffer 初始化中分配
   }
   ```

2. **win 区偏移计算**：不变。URMA 模式下仍然写远端 win 的相同偏移，只是用 URMA 写而非 DataCopyPad。

3. **Context 字段**：需从 `mc2Context_` 获取 `hcommHandle` 数组。当前 `AttentionFFNContext` 结构需扩展（见第 5.1 节）。

##### 3.2.2.5 CheckFlagAndSetTableGM() 改造

**文件**: `attention_to_ffn.h:595-610`

当前实现（UB-MEM）：
```cpp
// 通过 DataCopy 读远端 win flag → 轮询
DataCopy(ffnFlagTensor_, tokenInfoTableGMTensor, STATUS_REP_STRIDE);
SyncFunc<AscendC::HardEvent::MTE2_S>();
ffnFlage = ffnFlagTensor_.GetValue(0);
```

URMA 改造：
```cpp
if constexpr (IsUrma) {
    // URMA 模式：直接读远端 GM flag（通过 ReadGmByPassDCache 轮询）
    __gm__ int32_t *remoteFlagPtr = reinterpret_cast<__gm__ int32_t *>(tokenInfoTable);
    while (ReadGmByPassDCache(remoteFlagPtr) == 1) {
        // 等待目标卡清除上一轮 flag
    }
} else {
    // 现有 UB-MEM 路径
    DataCopy(ffnFlagTensor_, tokenInfoTableGMTensor, STATUS_REP_STRIDE);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    ffnFlage = ffnFlagTensor_.GetValue(0);
}
```

注意：UB-MEM 模式下 flag 同步是 "先轮询目标卡旧 flag 清零 → 再写新数据+flag"。URMA 的 `WriteWithNotifyNbi` 保证数据先于 flag 写入，因此 flag 写入语义不变，只是传输方式不同。

##### 3.2.2.6 SendTokenToFFNByTokenIdx() 改造

**文件**: `attention_to_ffn.h:613-667`

这是核心改造点。当前写远端 win 的两步 `DataCopyPad` 改为 URMA 一步发送。

**量化路径**（`QuantMode != 0`）当前：
```cpp
// 步骤 5: CheckFlagAndSetTableGM(...)
// 步骤 6: DataCopyPad(tokenDataGMTensor, xOutBytesTensor, hCommuCopyOutParams)  // 写 token 数据
// 步骤 7: DataCopyPad(tokenInfoTableGMTensor[...], statusTensor_[...], statusCopyParams)  // 写 flag
```

URMA 改造：
```cpp
if constexpr (IsUrma) {
    // 量化后 xOutTensor_ 已在 UB，需先 DataCopyPad 到本地 GM 临时区（或本地 win 区）
    // 因为 hcomm_.WriteNbi 的源地址必须是 GM 地址，不能是 UB
    GlobalTensor<uint8_t> localTmpGm;
    localTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
        workspaceGM_ + aivId_ * hCommuSize_));  // 本地 workspace 临时区
    xOutQueue_.EnQue(xOutTensor_);
    xOutTensor_ = xOutQueue_.DeQue<StorageXOutType>();
    auto xOutBytesTensor = xOutTensor_.template ReinterpretCast<uint8_t>();
    DataCopyPad(localTmpGm, xOutBytesTensor, hCommuCopyOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();  // 等 MTE3 写完本地 GM

    // 通过 URMA 发送到远端 win（token 数据 + flag 一步完成）
    uint64_t channel = GetUrmaCommHandle(mc2Context_, toRankId_, rankId_);
    GM_ADDR remoteDataAddr = toRankAddr + winOffset_[1] + ...;  // 远端 win tokenData 地址
    GM_ADDR remoteFlagAddr = toRankAddr + winInfoTableOffset_ + ...;  // 远端 win infoTable 地址
    GM_ADDR localDataAddr = workspaceGM_ + aivId_ * hCommuSize_;

    // 方式 1: 数据和 flag 合并发送（WriteWithNotifyNbi）
    hcomm_.WriteWithNotifyNbi(channel, remoteDataAddr, localDataAddr, hCommuSize_,
                              remoteFlagAddr, uint64_t(1));
    xOutQueue_.FreeTensor<StorageXOutType>(xOutTensor_);
} else {
    // 现有 UB-MEM 路径（不变）
    ...
}
```

**非量化路径**（`QuantMode == 0`）当前：
```cpp
// DataCopyPad(tokenDataGMTensor, xInTensor_, xCopyParams)  // 直接写远端 win
// DataCopyPad(tokenInfoTableGMTensor[...], statusTensor_[...], statusCopyParams)  // 写 flag
```

URMA 改造：
```cpp
if constexpr (IsUrma) {
    // token 数据在 UB（xInTensor_），需先搬到本地 GM
    xQueue_.EnQue(xInTensor_);
    xInTensor_ = xQueue_.DeQue<XType>();
    GlobalTensor<XType> localTmpGm;
    localTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(
        workspaceGM_ + aivId_ * hSize_));
    DataCopyPad(localTmpGm, xInTensor_, xCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    uint64_t channel = GetUrmaCommHandle(mc2Context_, toRankId_, rankId_);
    GM_ADDR remoteDataAddr = toRankAddr + winOffset_[1] + ...;
    GM_ADDR remoteFlagAddr = toRankAddr + winInfoTableOffset_ + ...;
    GM_ADDR localDataAddr = workspaceGM_ + aivId_ * hSize_;

    hcomm_.WriteWithNotifyNbi(channel, remoteDataAddr, localDataAddr, hSize_,
                              remoteFlagAddr, uint64_t(1));
    xQueue_.FreeTensor<XType>(xInTensor_);
} else {
    // 现有 UB-MEM 路径（不变）
    ...
}
```

##### 3.2.2.7 SetFlagToFFN() 改造

**文件**: `attention_to_ffn.h:713-765`

当前通过 `DataCopyPad(tableFlagGMTensor, statusTensor_, statusParams)` 写远端 win 的 table flag。

URMA 改造：用 `hcomm_.WriteNbi` 发送 status 到远端 win。因为 `SetFlagToFFN` 只写 2 个 int32（flag=1 + layerId），用 `WriteNbi` 即可（无需 notify，因为后续有 `SyncAll` 保证）。

```cpp
if constexpr (IsUrma) {
    // status 数据先写到本地 GM
    GlobalTensor<int32_t> localStatusGm;
    localStatusGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        workspaceGM_ + aivId_ * TOKEN_INFO_TABLE_COPY_BLOCK_CNT * sizeof(int32_t)));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(localStatusGm, statusTensor_, statusParams);

    uint64_t channel = GetUrmaCommHandle(mc2Context_, ffnIdx, rankId_);
    GM_ADDR remoteTableFlagAddr = toRankAddr + winInfoTableOffset_;
    hcomm_.WriteNbi(channel, remoteTableFlagAddr,
                    workspaceGM_ + aivId_ * TOKEN_INFO_TABLE_COPY_BLOCK_CNT * sizeof(int32_t),
                    static_cast<int64_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT));
} else {
    // 现有路径
    DataCopyPad(tableFlagGMTensor, statusTensor_, statusParams);
}
```

##### 3.2.2.8 SetFlagInAttn() 改造

**文件**: `attention_to_ffn.h:768-795`

当前通过 `DataCopyPad(attnTableGMTensor[startId], tempTensor, dataCopyParams)` 写本卡 win 区（`GetWindowAddrByRankId(rankId_)`）。

**注意**：这里写的是**本卡** win 区，不是远端。因此 URMA 模式下此处**不需要改造**，保持 `DataCopyPad` 即可。

##### 3.2.2.9 Process() 适配

```cpp
void Process() {
    if ASCEND_IS_AIV {
        if constexpr (isActiveMask) { ActiveMaskCalCnt(); }

        // URMA: 初始化 hcomm_（512B UB 区）
        if constexpr (IsUrma) {
            LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512);
            hcomm_.Init(hcommTensor, ALIGN_512);
        }

        SendTokenToFFN();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();

        // URMA: 冲刷所有使用过的通道
        if constexpr (IsUrma) {
            DrainUsedChannels();
        }

        SetFlagToFFN();
        if constexpr (isActiveMask) { SetFlagInAttn(); }
    }
}
```

##### 3.2.2.10 新增 DrainUsedChannels()

```cpp
__aicore__ inline void DrainUsedChannels() {
    // 冲刷本核发送过的所有目标 rank 的 URMA 通道
    // SendTokenToFFN 中已记录使用过的 rank 列表（或遍历 ffnNum_）
    for (uint32_t ffnIdx = ffnStartRankId_; ffnIdx < ffnStartRankId_ + ffnNum_; ++ffnIdx) {
        if (ffnIdx != rankId_) {
            hcomm_.Drain(GetUrmaCommHandle(mc2Context_, ffnIdx, rankId_));
        }
    }
}
```

参照 `mega_moe_layered.h:2011-2020` 的 `DrainCombineChannels`。

##### 3.2.2.11 workspace 需求

URMA 模式需要额外 workspace 用于本地 GM 中转（因为 `hcomm_.WriteNbi` 源地址必须是 GM）：

| workspace 区域 | 大小 | 用途 |
|---------------|------|------|
| 本地 token 临时区 | `aivNum_ * max(hCommuSize_, hSize_)` | UB→GM 中转后通过 URMA 发送 |
| 本地 status 临时区 | `aivNum_ * TOKEN_INFO_TABLE_COPY_BLOCK_CNT * sizeof(int32_t)` | flag 数据中转 |

---

## 4. ffn_to_attention_v2 URMA 适配设计

### 4.1 当前通信流程（UB-MEM）

`ffn_to_attention_v2.h` 的 `Process()` 核心流程（`ffn_to_attention_v2.h:179-240`）：

```
对每个 token:
  1. xTmpTensor_ = xQueue_.AllocTensor()
  2. DataCopyPad(xTmpTensor_, xGMTensor_[YOffset * axisH_], ...)   // MTE2: 从 GM 读 token
  3. ReadTokenMetaData(metaDataStruct, YOffset)                      // Scalar: 读 sessionId/microBatchId/tokenId/expertOffset
  4. curRankWinAddr = GetWindAddrByRankId(curAttenWorkRank)          // 获取目标 attn 卡 win 地址
  5. DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams)        // MTE3: 直接写远端 win tokenData
  6. DataCopyPad(tokenInfoTableGMTensor, statusTensor_, dataCopyParams)  // MTE3: 直接写远端 win infoTable flag
```

### 4.2 URMA 改造方案

#### 4.2.1 改造对照表

| 步骤 | UB-MEM 实现 | URMA 实现 | 改动类型 |
|------|------------|-----------|---------|
| 读 token | `DataCopyPad(xTmpTensor_, xGMTensor_[...], ...)` | 不变 | 无 |
| 读元数据 | `ReadTokenMetaData(...)` | 不变 | 无 |
| 获取目标地址 | `GetWindAddrByRankId(curAttenWorkRank)` | 不变（仍需远端 win 基址计算偏移） | 无 |
| 写 token 数据 | `DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams)` | UB→GM 中转 + `hcomm_.WriteNbi` | **替换** |
| 写 flag | `DataCopyPad(tokenInfoTableGMTensor, statusTensor_, dataCopyParams)` | 合并到 `WriteWithNotifyNbi` | **合并** |

#### 4.2.2 具体适配点

##### 4.2.2.1 Tiling Key 扩展

**文件**: `op_kernel/ffn_to_attention_v2_tilling_key.h`

```cpp
#define TILINGKEY_COMM_MTE 0
#define TILINGKEY_COMM_URMA 1

ASCENDC_TPL_ARGS_DECL(FFNToAttentionV2,
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_RANK_TABLE_MODE, 0, 1),
    ASCENDC_TPL_UINT_DECL(ARCH_TAG, ..., TILINGKEY_TPL_A5),
    ASCENDC_TPL_UINT_DECL(COMM_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                          TILINGKEY_COMM_MTE, TILINGKEY_COMM_URMA), );
```

##### 4.2.2.2 Kernel Entry 扩展

**文件**: `op_kernel/arch35/ffn_to_attention_v2.cpp`

```cpp
template <bool RankTableMode, uint8_t ArchTag, uint8_t CommModeType>
__global__ __aicore__ void ffn_to_attention_v2(...)
{
    REGISTER_TILING_DEFAULT(FFNToAttentionV2TilingData);
    TPipe pipe;

    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(FFNToAttentionV2TilingData, tilingData, tilingGM);
        const auto *v1TilingData = reinterpret_cast<const FFNToAttentionTilingData *>(&tilingData);
        if constexpr (CommModeType == TILINGKEY_COMM_MTE) {
            // 现有路径
            FFNToAttention<DTYPE_X, RankTableMode, FFNToAttentionContextType::MC2_CONTEXT> op;
            op.Init(...);
            op.Process();
        } else if constexpr (CommModeType == TILINGKEY_COMM_URMA) {
            // 新增 URMA 路径
            FFNToAttention<DTYPE_X, RankTableMode, FFNToAttentionContextType::MC2_CONTEXT, true> op;
            op.Init(...);
            op.Process();
        }
    }
}
```

##### 4.2.2.3 FFNToAttention 类模板扩展

**文件**: `ffn_to_attention/op_kernel/ffn_to_attention.h` 和 `ffn_to_attention_v2/op_kernel/ffn_to_attention_v2.h`

在模板参数中增加 `bool IsUrma = false`。

新增成员：
```cpp
Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
```

新增 include：
```cpp
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif
#include "../../common/op_kernel/mc2_moe_context.h"  // GetUrmaCommHandle
```

##### 4.2.2.4 Init() 适配

**改动点**：
1. 保存 `workspaceGM` 供后续 URMA 中转使用
2. 无需计算 server 拓扑（不跨 server）

```cpp
// 新增成员
GM_ADDR workspaceGM_;  // 保存 workspace 供 URMA 中转

// Init 中
workspaceGM_ = workspaceGM;
```

##### 4.2.2.5 Process() 改造

**文件**: `ffn_to_attention_v2.h:179-240`

当前写远端 win：
```cpp
// 步骤 5: 写 token 数据到远端 win
xQueue_.EnQue(xTmpTensor_);
xTmpTensor_ = xQueue_.DeQue<xType>();
DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams);
xQueue_.FreeTensor<xType>(xTmpTensor_);
PipeBarrier<PIPE_MTE3>();
// 步骤 6: 写 flag 到远端 win
DataCopyPad(tokenInfoTableGMTensor, statusTensor_, dataCopyParams);
```

URMA 改造：
```cpp
if constexpr (IsUrma) {
    // 1. token 从 UB 搬到本地 GM 临时区（WriteNbi 源必须是 GM）
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<xType>();
    GlobalTensor<xType> localTmpGm;
    // 每个 aiv 使用独立 workspace 槽位，避免并发覆盖
    localTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(
        workspaceGM_ + aivId_ * hSize_));
    DataCopyPad(localTmpGm, xTmpTensor_, xCopyParams);
    xQueue_.FreeTensor<xType>(xTmpTensor_);
    SyncFunc<AscendC::HardEvent::MTE3_S>();  // 等本地 GM 写完

    // 2. 通过 URMA 发送 token + flag 到远端 win
    uint64_t channel = GetUrmaCommHandle(mc2Context_, metaDataStruct.curAttenWorkRank, rankId_);
    GM_ADDR remoteDataAddr = curRankWinAddr + winTokenInfoTableSize_ + tokenDataOffset;
    GM_ADDR remoteFlagAddr = curRankWinAddr + (tokenInfoTableOffset + ...);
    GM_ADDR localDataAddr = workspaceGM_ + aivId_ * hSize_;

    hcomm_.WriteWithNotifyNbi(channel, remoteDataAddr, localDataAddr,
                              static_cast<int64_t>(axisH_ * sizeof(xType)),
                              remoteFlagAddr, uint64_t(1));
} else {
    // 现有 UB-MEM 路径
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<xType>();
    DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams);
    xQueue_.FreeTensor<xType>(xTmpTensor_);
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(tokenInfoTableGMTensor, statusTensor_, dataCopyParams);
}
```

##### 4.2.2.6 Process() 增加通道冲刷

```cpp
void Process() {
    if ASCEND_IS_AIV {
        // URMA: 初始化 hcomm_
        if constexpr (IsUrma) {
            LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512);
            hcomm_.Init(hcommTensor, ALIGN_512);
        }

        // ... 现有 token 发送循环 ...

        // URMA: 冲刷所有使用过的通道
        if constexpr (IsUrma) {
            DrainUsedChannels();
        }
    }
}
```

##### 4.2.2.7 新增 DrainUsedChannels()

ffn_to_attention 的目标 rank 来自 `metaDataStruct.curAttenWorkRank`，是动态的。需要在循环中记录使用过的 rank，或循环结束后遍历所有可能的 attn rank 做 Drain：

```cpp
__aicore__ inline void DrainUsedChannels() {
    // 遍历所有可能的 attn work rank 做 Drain
    // worldSize_ 个 rank，排除本卡
    for (uint32_t rankIdx = 0; rankIdx < worldSize_; ++rankIdx) {
        if (rankIdx != rankId_) {
            hcomm_.Drain(GetUrmaCommHandle(mc2Context_, rankIdx, rankId_));
        }
    }
}
```

##### 4.2.2.8 workspace 需求

| workspace 区域 | 大小 | 用途 |
|---------------|------|------|
| 本地 token 临时区 | `aivNum_ * hSize_` | UB→GM 中转后通过 URMA 发送 |

---

## 5. 通用适配项

### 5.1 Context 结构扩展

两个算子共享 `Mc2Aclnn::AttentionFFNContext`。需增加 `hcommHandle` 字段供 `GetUrmaCommHandle` 使用。

**文件**: `common/attention_ffn_context.h`

```cpp
struct AttentionFFNContext {
    // ... 现有字段 ...
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE] = {};  // 新增：URMA 通道句柄
};
```

参照 `Mc2MoeContext`（`mega_moe_base.h:25-31`）中的 `hcommHandle` 字段。

`GetUrmaCommHandle` 函数（`mega_moe_base.h:235-239`）依赖 context 中的 `hcommHandle` 数组：
```cpp
__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2MoeContext *mc2Context_, uint32_t rankId, uint32_t epRankId) {
    uint32_t index = rankId > epRankId ? rankId - 1 : rankId;
    return mc2Context_->hcommHandle[index];
}
```

需为 `AttentionFFNContext` 提供类似的通道获取函数，或将 `GetUrmaCommHandle` 泛化为接受 `hcommHandle` 指针。

**影响文件**:
- `common/attention_ffn_context.h` — 结构定义
- `op_api/aclnn_attention_to_ffn_v2.cpp` — API 层填充 `hcommHandle`
- `op_api/aclnn_ffn_to_attention_v2.cpp` — API 层填充 `hcommHandle`

### 5.2 GetUrmaCommHandle 适配

当前 `GetUrmaCommHandle` 接受 `Mc2MoeContext*`。需增加一个接受 `AttentionFFNContext*` 的重载，或改为模板/通用接口：

```cpp
// 方案 A: 重载
__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2Aclnn::AttentionFFNContext *ctx,
                                              uint32_t rankId, uint32_t epRankId) {
    uint32_t index = rankId > epRankId ? rankId - 1 : rankId;
    return ctx->hcommHandle[index];
}

// 方案 B: 通用化（接受 hcommHandle 指针）
__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ uint64_t *hcommHandle,
                                              uint32_t rankId, uint32_t epRankId) {
    uint32_t index = rankId > epRankId ? rankId - 1 : rankId;
    return hcommHandle[index];
}
```

### 5.3 Win 区布局

**不变。** URMA 模式下 win 区仍使用 `winOffset_[0]` (infoTable) + `winOffset_[1]` (tokenData) 两段布局。URMA 只是替换了写入远端 win 的传输方式，目标地址计算逻辑不变。

### 5.4 Workspace 扩展

URMA 模式需要额外 workspace 用于 UB→GM 中转（因为 `hcomm_.WriteNbi` 的源地址必须是 GM）：

| 算子 | workspace 区域 | 大小计算 |
|------|---------------|---------|
| attention_to_ffn_v2 | 本地 token 中转区 | `aivNum_ * max(hCommuSize_, hSize_)` |
| attention_to_ffn_v2 | 本地 status 中转区 | `aivNum_ * TOKEN_INFO_TABLE_COPY_BLOCK_CNT * sizeof(int32_t)` |
| ffn_to_attention_v2 | 本地 token 中转区 | `aivNum_ * hSize_` |

Host tiling 中根据 `COMM_MODE` 判断是否追加此 workspace。

### 5.5 Host Tiling 适配

**文件**: `attention_to_ffn_v2_tiling.cpp` / `ffn_to_attention_v2_tilling.cpp`

1. **判断通信模式**：检查 context 中 `hcommHandle` 是否有效 → 设置 `COMM_MODE` tiling key 维度
2. **计算 workspace**：URMA 模式追加本地中转区 workspace
3. **无需新增 tiling data 字段**：URMA 标识完全由 tiling key 承载，不扩展 tiling data 结构，保持 V1/V2 `static_assert` 大小一致

### 5.6 CMakeLists 适配

新增的 `hcomm.h` include 需确保编译系统可找到。如果 `adv_api/hcomm/hcomm.h` 已在 mega_moe 编译路径中可用，attention_to_ffn / ffn_to_attention 的 CMakeLists 需添加相同的 include 路径。

### 5.7 V1/V2 兼容性

当前 V2 kernel entry 中的 `static_assert` 约束：
```cpp
static_assert(sizeof(AttentionToFfnV2TilingData) == sizeof(AttentionToFFNTilingData), ...);
```

**本设计不扩展 tiling data 结构**，URMA 模式标识通过 tiling key 的 `COMM_MODE` 维度区分，workspace 扩展通过现有 `workspaceGM` 传递。因此 V1/V2 大小一致约束**不受影响**。

V1 模板 `AttentionToFFN` / `FFNToAttention` 增加 `IsUrma` 模板参数（默认 `false`），V2 kernel entry 中以 `true` 实例化 URMA 路径。V1 算子（arch22）不受影响。

---

## 6. 适配清单总览

### 6.1 attention_to_ffn_v2

| 序号 | 文件 | 改造内容 | 优先级 |
|------|------|---------|--------|
| 1 | `common/attention_ffn_context.h` | 增加 `hcommHandle[HCCL_MAX_RANK_SIZE]` 字段 | P0 |
| 2 | `common/mc2_kernel_utils.h` 或 `mega_moe_base.h` | 增加 `GetUrmaCommHandle` 对 `AttentionFFNContext` 的重载 | P0 |
| 3 | `op_api/aclnn_attention_to_ffn_v2.cpp` | API 层填充 `hcommHandle` | P0 |
| 4 | `op_kernel/attention_to_ffn_v2_tiling_key.h` | 增加 `COMM_MODE` tiling key 维度 | P0 |
| 5 | `op_kernel/arch35/attention_to_ffn_v2.cpp` | 增加 `CommModeType` 模板参数和 URMA 分支 | P0 |
| 6 | `attention_to_ffn/op_kernel/attention_to_ffn.h` | 类模板增加 `IsUrma` 参数；`SendTokenToFFNByTokenIdx`/`CheckFlagAndSetTableGM`/`SetFlagToFFN`/`Process` 增加 `if constexpr (IsUrma)` 分支；新增 `DrainUsedChannels` | P0 |
| 7 | `op_host/op_tiling/attention_to_ffn_v2_tiling.cpp` | URMA 模式 workspace 计算、`COMM_MODE` tiling key 设置 | P0 |
| 8 | `op_host/attention_to_ffn_v2_def.cpp` | 注册 URMA tiling key（如需） | P1 |
| 9 | `tests/ut/op_host/arch35/test_attention_to_ffn_v2_tiling.cpp` | 增加 URMA tiling 测试 | P1 |
| 10 | `tests/ut/op_api/test_aclnn_attention_to_ffn_v2.cpp` | 增加 URMA 通信测试 | P1 |

### 6.2 ffn_to_attention_v2

| 序号 | 文件 | 改造内容 | 优先级 |
|------|------|---------|--------|
| 1 | `common/attention_ffn_context.h` | 同上（与 attention_to_ffn_v2 共享） | P0 |
| 2 | `common/mc2_kernel_utils.h` 或 `mega_moe_base.h` | 同上 | P0 |
| 3 | `op_api/aclnn_ffn_to_attention_v2.cpp` | API 层填充 `hcommHandle` | P0 |
| 4 | `op_kernel/ffn_to_attention_v2_tilling_key.h` | 增加 `COMM_MODE` tiling key 维度 | P0 |
| 5 | `op_kernel/arch35/ffn_to_attention_v2.cpp` | 增加 `CommModeType` 模板参数和 URMA 分支 | P0 |
| 6 | `ffn_to_attention_v2/op_kernel/ffn_to_attention_v2.h` | 类模板增加 `IsUrma` 参数；`Process` 增加 `if constexpr (IsUrma)` 分支；新增 `DrainUsedChannels` | P0 |
| 7 | `op_host/op_tiling/ffn_to_attention_v2_tilling.cpp` | URMA 模式 workspace 计算、`COMM_MODE` tiling key 设置 | P0 |
| 8 | `op_host/ffn_to_attention_v2_def.cpp` | 注册 URMA tiling key（如需） | P1 |
| 9 | `tests/ut/op_host/arch35/test_ffn_to_attention_v2_tiling.cpp` | 增加 URMA tiling 测试 | P1 |
| 10 | `tests/ut/op_api/test_aclnn_ffn_to_attention_v2.cpp` | 增加 URMA 通信测试 | P1 |

### 6.3 函数级改造清单

#### AttentionToFFN（IsUrma=true 分支）

| 函数 | 文件:行 | 改造内容 |
|------|--------|---------|
| `Init` | `attention_to_ffn.h:295` | 保存 `workspaceGM_`；增加 hcomm include |
| `CheckFlagAndSetTableGM` | `attention_to_ffn.h:595` | UB-MEM `DataCopy` 轮询 → `ReadGmByPassDCache` 直接轮询远端 GM flag |
| `SendTokenToFFNByTokenIdx` | `attention_to_ffn.h:613` | 量化/非量化两条路径的 `DataCopyPad` 写远端 → UB→GM 中转 + `hcomm_.WriteWithNotifyNbi` |
| `SetFlagToFFN` | `attention_to_ffn.h:713` | `DataCopyPad` 写远端 table flag → `hcomm_.WriteNbi` |
| `SetFlagInAttn` | `attention_to_ffn.h:768` | **不改造**（写本卡 win，非远端） |
| `Process` | `attention_to_ffn.h:798` | 增加 hcomm 初始化 + `DrainUsedChannels` 调用 |
| `DrainUsedChannels` | **新增** | 冲刷所有目标 rank 的 URMA 通道 |

#### FFNToAttention（IsUrma=true 分支）

| 函数 | 文件:行 | 改造内容 |
|------|--------|---------|
| `Init` | `ffn_to_attention_v2.h:97` | 保存 `workspaceGM_`；增加 hcomm include |
| `Process` | `ffn_to_attention_v2.h:179` | `DataCopyPad` 写远端 → UB→GM 中转 + `hcomm_.WriteWithNotifyNbi`；增加 hcomm 初始化 + Drain |
| `DrainUsedChannels` | **新增** | 冲刷所有目标 rank 的 URMA 通道 |

---

## 7. 风险与注意事项

### 7.1 UB→GM 中转的必要性

`hcomm_.WriteNbi` / `WriteWithNotifyNbi` 的源地址必须是 GM 地址，不能直接用 UB 上的 `LocalTensor`。因此 URMA 模式下需要先把 token 数据从 UB 搬到本地 GM 临时区，再通过 URMA 发送。这引入一次额外的 `DataCopyPad(UB→GM)` 开销。

**优化建议**：如果 token 数据本身在 GM（如非量化路径的 `xGMTensor_`），可以直接用 `xGMTensor_` 的地址作为 URMA 源地址，跳过 UB 中转。但需注意 `ReadTokenMetaData` 中 `DataCacheCleanAndInvalid` 对源 GM 的 cache 操作可能与 URMA 读冲突，需加 `PipeBarrier<PIPE_ALL>` 确保有序。

### 7.2 flag 同步语义

UB-MEM 模式下数据写和 flag 写是两步 `DataCopyPad`，依赖 MTE3 流水线顺序保证数据先于 flag 到达。

URMA 的 `WriteWithNotifyNbi` 保证数据写完成后才在远端置 flag，语义更严格，不会出现 flag 先到数据未到的问题。

但 `CheckFlagAndSetTableGM` 中的 flag 轮询逻辑（等待目标卡清除上一轮旧 flag）需要改为 `ReadGmByPassDCache` 直接读远端 GM，不再通过 `DataCopy` 搬到 UB 再读。

### 7.3 通道冲刷时机

`hcomm_.Drain` 必须在所有发送完成后调用。attention_to_ffn 中有两个发送阶段：
1. `SendTokenToFFN`：逐 token 发送数据
2. `SetFlagToFFN`：发送 table flag

Drain 应在 `SetFlagToFFN` 之前调用（`SendTokenToFFN` 之后），确保所有 token 数据已到达远端。`SetFlagToFFN` 本身也用 URMA 发送，其 Drain 可放在 `Process` 末尾。

### 7.4 workspace 并发安全

UB→GM 中转区按 `aivId_ * slotSize` 分配独立槽位，避免多核并发写同一 GM 区域。需确保 workspace 大小 = `aivNum_ * maxSlotSize`。

### 7.5 hcomm UB 区占用

`hcomm_.Init` 需要 512B UB 区（`ALIGN_512`）。需在现有 UB 布局中预留此区域，不影响其他 buffer。参照 `mega_moe_layered.h:413-414`：
```cpp
LocalTensor<uint8_t> hcommTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512);
hcomm_.Init(hcommTensor_, ALIGN_512);
```
后续 buffer 从 `ALIGN_512` 偏移开始分配。

### 7.6 attention_to_ffn 量化路径的 GM 中转大小

量化路径下 `hCommuSize_` 包含数据 + scale padding，与非量化路径的 `hSize_` 不同。workspace 中转区大小需取 `max(hCommuSize_, hSize_)` 以覆盖所有路径。

---

## 8. 测试验证计划

### 8.1 单元测试

1. **Tiling 测试**：验证 URMA tiling key 正确生成、workspace 大小正确计算
2. **Context 测试**：验证 `hcommHandle` 字段正确传递

### 8.2 集成测试

1. **URMA 通信正确性**：attention_to_ffn_v2 → ffn_to_attention_v2 端到端数据一致性
2. **量化模式覆盖**：NO_QUANT / MX(E5M2/E4M3/E2M1) / MX_CLIP 各模式
3. **与 mega_moe URMA 路径联调**：确保 attention_to_ffn_v2 + mega_moe(URMA) + ffn_to_attention_v2 端到端正确
4. **多 rank 场景**：验证 Drain 通道冲刷完整性

### 8.3 性能验证

1. URMA vs UB-MEM 延迟对比
2. UB→GM 中转开销评估
3. 大 BS 场景下 URMA 通道冲刷是否成为瓶颈
