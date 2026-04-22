# ffn_to_attention 算子全面分析报告

## 1. 算子概述

`ffn_to_attention` 是 **FFN→Attention 流水线并行** 的反向数据传输算子。与 `attention_to_ffn` 配对使用，在 Pipeline Parallel 模型中将 FFN 阶段处理完成的 Token 结果发送回对应的 Attention Worker。通过 HCCL Window 共享内存机制实现跨 Rank 数据传输。

```
FFN Worker                       Attention Worker
┌──────────────────┐             ┌──────────────────┐
│ Token Data (GM)  │             │ Window Memory    │
│ Token Metadata   │──Window───→ │  ├ Token Info    │
│ (session, batch, │    IPC      │  ├ Token Data    │
│  expert, offset) │             │  └ Status Flag   │
└──────────────────┘             └──────────────────┘
```

### 1.1 模板参数

```cpp
// ffn_to_attention_tilling_key.h
ASCENDC_TPL_ARGS_DECL(FFNToAttention,
    TILINGKEY_RANK_TABLE_MODE,  // bool: 是否提供 attnRankTable
    ARCH_TAG);                  // uint8: 架构标记 (仅 A3)
// 2 种有效组合
//   key=2: 无 RankTable (TILINGKEY_RANK_TABLE_MODE=false, A3)
//   key=3: 有 RankTable (TILINGKEY_RANK_TABLE_MODE=true, A3)
```

---

## 2. 目录结构与文件职责

```
ffn_to_attention/
├── op_kernel/
│   ├── ffn_to_attention.cpp              (2,026字节) 内核入口
│   ├── ffn_to_attention.h                (11,031字节) 主类实现
│   ├── ffn_to_attention_tiling.h         (1,339字节) Tiling 结构
│   └── ffn_to_attention_tilling_key.h    (1,745字节) 模板参数
├── op_host/
│   ├── ffn_to_attention_def.cpp          (3,163字节) 算子定义
│   ├── ffn_to_attention_infershape.cpp   (1,184字节) Shape 推导
│   └── op_tiling/
│       └── ffn_to_attention_tilling.cpp  (27,439字节) Tiling 策略
├── op_api/
│   ├── aclnn_ffn_to_attention.cpp        (4,969字节) ACLNN 实现
│   └── aclnn_ffn_to_attention.h          (3,635字节) ACLNN 声明
├── op_graph/
│   ├── ffn_to_attention_gen_task.cpp     (3,060字节) 图模式任务
│   └── ffn_to_attention_proto.h          (1,828字节) 图模式注册
├── config/ascend910_93/
│   ├── ffn_to_attention_binary.json      (5,291字节) 二进制配置
│   └── ffn_to_attention_simplified_key.ini (1,238字节)
├── examples/
│   └── test_aclnn_ffn_to_attention.cpp   (12,757字节) 多卡示例
└── tests/ut/
    ├── op_api/test_aclnn_ffn_to_attention.cpp (4,174字节)
    ├── op_host/test_ffn_to_attention_infershape.cpp (2,912字节)
    └── op_host/arch32/test_ffn_to_attention_tiling.cpp (16,629字节)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `ffn_to_attention_def.cpp`（第19行定义 `FFNToAttention` 类；第21-74行构造函数；第77行 `OP_ADD` 注册）

**输入张量**（第22-56行声明）:

| 序号 | 名称 | 类型 | 维度 | 说明 |
|------|------|------|------|------|
| 0 | x | FP16/BF16 | [BS×K, HS] | FFN 输出 Token 数据 |
| 1 | session_ids | INT32 | [BS×K] | 会话 ID |
| 2 | micro_batch_ids | INT32 | [BS×K] | 微批次 ID |
| 3 | token_ids | INT32 | [BS×K] | Token 在原始序列中的位置 |
| 4 | expert_offsets | INT32 | [BS×K] | Expert 偏移量 |
| 5 | actual_token_num | INT64 | [1] | 实际 Token 数量 |
| 6 | attn_rank_table | INT32 (可选) | [A] | Attention Worker→Rank 映射 |

**关键属性**（第57-60行声明）:

| 属性名 | 类型 | 说明 |
|--------|------|------|
| group | String | HCCL 通信域 |
| world_size | Int | 全局卡数 |
| token_info_table_shape | ListInt | Token 信息表形状 (3D) |
| token_data_shape | ListInt | Token 数据形状 (4D) |

**平台**: 仅支持 `ascend910_93`（第62-72行配置）

### 3.2 Tiling 策略

**文件**: `ffn_to_attention_tilling.cpp`（459行。第102-155行 `CheckAndSetAttrs()`，第157-197行 `CheckInputDim0Dim1()`，第357-365行 `SetWorkSpace()`，第367-382行 `CalWinSize()`，第396-448行主 Tiling 入口 `FFNToAttentionTilingFunc()`）

#### 参数校验

```
CheckAndSetAttrs():
├── group: 非空字符串, 长度 < 128
├── worldSize: [2, 768]
├── tokenInfoTableShape: 3D
├── tokenDataShape: 4D
├── 跨 Shape 维度一致性检查
├── microBatchNum == 1
├── BS ∈ (0, 512]
└── HS ∈ [1024, 8320]

CheckInputDim():
├── x: 2D [BS×K, HS]
├── 所有 metadata: 1D, dim0 == x.dim0
├── actualTokenNum: 1D, dim0 == 1
└── attnRankTable: 1D, dim0 ≤ worldSize-1

CheckInputDim0Dim1():
├── H = x.dim1 → 必须 ∈ [1024, 8192]
└── 所有 1D 输入 dim0 == x.dim0
```

#### Window 与 Workspace 计算

```
CalWinSize():
├── tokenInfoTableSize = microBatchNum * BS * expertNumPerToken * sizeof(int32)
├── tokenDataSize = microBatchNum * BS * expertNumPerToken * HS * sizeof(half)
├── totalWinSize = (tokenInfoTableSize + tokenDataSize) / MB + 1
└── 比较 totalWinSize 与平台最大 Window 大小

SetWorkSpace():
└── libApiWorkSpaceSize (平台库 API 需求, 至少 16MB)
```

#### TilingKey 计算

```
CalTilingKey():
  rankTableMode = (attnRankTable != nullptr) ? true : false
  key = GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A3)
  → key=2: 无 RankTable
  → key=3: 有 RankTable
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `ffn_to_attention.cpp`（第40-43行内核入口签名；第45行 `REGISTER_TILING_DEFAULT`；第49行 `GET_TILING_DATA_WITH_STRUCT`；第50行实例化 `FFNToAttention`；第51-53行 `Init()` + `Process()`）

```cpp
template<bool RankTableMode, uint8_t ArchTag>
__global__ __aicore__ void ffn_to_attention(...)
{
    REGISTER_TILING_DEFAULT(tilingGM);
    GET_TILING_DATA_WITH_STRUCT(FFNToAttentionTilingData, tilingData, tilingGM);
    
    FFNToAttention<DTYPE_X, RankTableMode> op;
    op.Init(x, sessionIds, microBatchIds, tokenIds, expertOffsets,
            actualTokenNum, attnRankTable, workspace, &tilingData);
    op.Process();
}
```

### 4.2 主类 FFNToAttention

**文件**: `ffn_to_attention.h`（第46行类定义，224行总计。`Init()` 第93-133行，`GetWindowAddr()` 第136-144行，`ReadTokenMetaData()` 第147-166行，`Process()` 第169-222行）

```cpp
namespace FFNToAttentionImpl {
template <typename xType, bool isInputRankTable>
class FFNToAttention;
}
```

#### 常量定义

```cpp
static constexpr uint32_t BUFFER_NUM = 2;     // 双缓冲队列深度
static constexpr uint32_t UB_ALIGN = 32;       // UB 对齐字节
static constexpr uint32_t WIN_ADDR_ALIGN = 512; // Window 地址对齐
```

#### Init() — 初始化

```
Init(x, sessionIds, microBatchIds, tokenIds, expertOffsets,
     actualTokenNum, attnRankTable, workspace, tiling):
├── 设置 7 个输入的 GlobalTensor 地址
├── 读取 Tiling:
│   ├── H, A, microBatchNum, BS
│   ├── expertNumPerToken, HS
│   ├── aivNum, worldSize
│   ├── isInputRankTable, windowType
│   └── totalUbSize, totalWinSize
├── 初始化 Pipe:
│   └── TQue<QuePosition::VECIN, BUFFER_NUM>
├── 分配 UB buffer (双缓冲):
│   └── xLocal_[BUFFER_NUM], 每个 H * sizeof(xType) + UB_ALIGN
└── 计算偏移量:
    ├── batchSizeSendCnt_ = BS * expertNumPerToken * HS
    ├── perMicroBatchOffset_ = BS * expertNumPerToken
    ├── perExpertOffset_ = expertNumPerToken
    └── winTokenInfoTableSize_ = BS * expertNumPerToken * 4  // 4 = info table entry size
```

#### Process() — 主流程

```
Process():
├── 仅 AIV 核执行
├── 实际 tokenNum = actualTokenNumGM.GetValue(0)
├── SplitToCore(tokenNum): 将 Token 分配到各 AIV
│   └── tokenIdx = aivId_ + tokenCnt * aivNum_ (轮询分配)
└── for each assigned token:
    ├── 从队列获取空闲 xLocal (双缓冲)
    ├── DataCopy: xGM[tokenIdx * H] → xLocal (H 元素)
    ├── ReadTokenMetaData(tokenIdx):
    │   ├── curAttenWorkIds = sessionIdsGM[tokenIdx]
    │   ├── curMicroBatchIds = microBatchIdsGM[tokenIdx]
    │   ├── curTokenBatchOffset = tokenIdsGM[tokenIdx]
    │   ├── curTokenTopkOffset = expertOffsetsGM[tokenIdx]
    │   └── [isInputRankTable]:
    │       ├── attenWorkRank_raw = attnRankTableGM[curAttenWorkIds]
    │       └── curAttenWorkRank = attenWorkRank_raw
    │       (否则: curAttenWorkRank = curAttenWorkIds)
    ├── 计算 Window 目标地址:
    │   ├── targetRankWindow = GetWindowAddr(curAttenWorkRank)
    │   ├── infoTableOffset = winTokenInfoTableSize_ * curMicroBatchIds
    │   │                    + curTokenTopkOffset * 4
    │   ├── tokenDataOffset = tokenDataStart + curMicroBatchIds * batchSizeSendCnt_
    │   │                    + curTokenTopkOffset * HS
    │   └── 对齐到 WIN_ADDR_ALIGN
    ├── DataCopyPad: 写 Token 数据到 Window
    │   └── xLocal → targetRankWindow + tokenDataOffset, H 元素
    ├── DataCopyPad: 写状态标志到 Window info table
    │   └── 写 flag=1 到 targetRankWindow + infoTableOffset
    └── 释放 xLocal 到队列 (双缓冲回收)
```

#### GetWindowAddr() — Window 地址解析

```
GetWindowAddr(rankId):
├── mc2KernelUtils::GetHcclContext(contextGM, rankId)
├── if (userMemType == 0):
│   └── return remoteWindowBaseAddr  ← 设备远端内存
└── if (userMemType == 1):
    └── return userDefinedAddr       ← 用户定义内存
```

#### ReadTokenMetaData() — Token 元数据读取

```
ReadTokenMetaData(tokenIdx):
├── DataCacheCleanAndInvalid: 刷新 metadata 区域缓存
├── curAttenWorkIds = sessionIdsGM.GetValue(tokenIdx)
├── curMicroBatchIds = microBatchIdsGM.GetValue(tokenIdx)
├── curTokenBatchOffset = tokenIdsGM.GetValue(tokenIdx)
├── curTokenTopkOffset = expertOffsetsGM.GetValue(tokenIdx)
└── [isInputRankTable]:
    └── curAttenWorkRank = attnRankTableGM.GetValue(curAttenWorkIds)
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

Tiling 结构定义在 `ffn_to_attention_tiling.h`（第21-34行 `FFNToAttentionInfo`，第36-40行 `FFNToAttentionTilingData`）：

```
FFNToAttentionInfo:
├── H                    : uint32  ← 隐藏层维度
├── A                    : uint32  ← attnRankTable dim0 (有 RankTable 时)
├── microBatchNum        : uint32  ← 微批次数量 (固定为 1)
├── BS                   : uint32  ← 批次大小
├── expertNumPerToken    : uint32  ← 每 Token 的 Expert 数 (K + shared)
├── HS                   : uint32  ← Hidden + Scale 维度
├── aivNum               : uint32  ← AIV 核数
├── worldSize            : uint32  ← 全局卡数
├── isInputRankTable     : bool    ← 是否有 RankTable
├── windowType           : uint8   ← Window 类型
├── totalUbSize          : uint64  ← UB 总大小
└── totalWinSize         : uint64  ← Window 总大小

FFNToAttentionTilingData:
├── Mc2InitTiling        : Mc2InitTiling   ← HCCL 初始化
├── mc2CcTiling1         : Mc2CcTiling     ← 通信参数
└── FFNToAttentionInfo                    ← 算子参数
```

### 5.2 Workspace 布局

Workspace 大小由 `SetWorkSpace()`（第357-365行）计算，至少 16MB。HCCL Window 内存由 HCCL 框架分配，大小通过 `CalWinSize()`（第367-382行）计算。

```
Workspace + Window 布局:
┌─────────────────────────────────────────────────────────┐
│ Workspace (≥16MB)                                       │
│ └── libApiWorkSpaceSize (平台库 API 需求)                │
├─────────────────────────────────────────────────────────┤
│ HCCL Window Memory (由 HCCL 框架分配)                    │
│ └── 每个 Attention Worker 的 Window 包含:                │
│     ├── Token Info Table (BS × expertNum × 4B)          │
│     └── Token Data Region (BS × expertNum × HS × 2B)    │
└─────────────────────────────────────────────────────────┘
```
Attention Worker 的 Window Memory:
┌─────────────────────────────────────────────────────────┐
│ Token Info Table (per microBatch)                       │
│ ┌──────────────────────────────────────────────────┐   │
│ │ [status, sessionId, microBatchId, tokenId,       │   │
│ │  expertOffset, ...] × BS × expertNumPerToken     │   │
│ │ 每个条目: 4 字节 (int32)                          │   │
│ └──────────────────────────────────────────────────┘   │
│                                                         │
│ Token Data Region                                       │
│ ┌──────────────────────────────────────────────────┐   │
│ │ Token 0: [HS elements × sizeof(xType)]           │   │
│ │ Token 1: [HS elements × sizeof(xType)]           │   │
│ │ ...                                              │   │
│ │ Token N: [HS elements × sizeof(xType)]           │   │
│ └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

偏移计算:
  infoTableOffset = winTokenInfoTableSize_ * curMicroBatchIds
                  + curTokenTopkOffset * 4
  tokenDataOffset = tokenDataRegionStart
                  + curMicroBatchIds * batchSizeSendCnt_
                  + curTokenTopkOffset * HS
```

---

## 6. 完整数据流

```
Host 侧                              Kernel 侧
────────                              ────────
aclnnFFNToAttentionGetWorkspaceSize()  ffn_to_attention.cpp
  │  校验参数 (第74-85行)              第40行: 内核入口
  │  计算 Tiling                       第45行: 注册 TilingData
  │  计算 Workspace + Window           第49行: 解析 TilingData
  └→ 返回 workspaceSize, executor      ↓
                                      FFNToAttention::Init() (第93-133行)
                                      ├── 设置 7 个输入 GM 地址
                                      ├── 读取 Tiling (H, BS, expertNum, ...)
                                      ├── 初始化双缓冲队列
                                      └── 计算偏移量
                                      ↓
aclnnFFNToAttention()                 FFNToAttention::Process() (第169-222行)
  └→ 执行内核                          ├── SplitToCore: 轮询分配 Token 到 AIV
                                       └── for each assigned token:
                                           ├── DataCopy: xGM → xLocal (H 元素)
                                           ├── ReadTokenMetaData (第147-166行)
                                           │   ├── sessionIds → 目标 Attention Worker
                                           │   └── [RankTable] attnRankTable 查表
                                           ├── GetWindowAddr (第136-144行)
                                           │   └── HCCL Context → Window 基地址
                                           ├── DataCopyPad: xLocal → Window Data Region
                                           └── DataCopyPad: flag=1 → Window Info Table
```

**Window 写入完成后**：Attention Worker 侧的 `attention_to_ffn` 算子通过轮询 Info Table 中的 status 标志来检测 Token 数据就绪，然后读取 Token Data 进行后续处理。

---

## 7. 通信模式

### 7.1 数据传输流程

```
FFN Worker → Attention Worker 数据传输:

FFN Worker (本地 GM)                  Attention Worker (Window)
┌─────────────────┐                  ┌──────────────────────┐
│ x[tokenIdx, :]  │                  │ Window Base Addr     │
│ [H 个元素]      │                  │                      │
│                 │   DataCopyPad    │ ┌─Info Table──────┐  │
│ sessionIds[]    │──────────────→  │ │ [flag=1, ...]   │  │
│ microBatchIds[] │   (写 Token)    │ └─────────────────┘  │
│ tokenIds[]      │                  │                      │
│ expertOffsets[] │   DataCopyPad   │ ┌─Token Data──────┐  │
│                 │──────────────→  │ │ [H elem data]   │  │
│ attnRankTable[] │   (写标志)      │ └─────────────────┘  │
└─────────────────┘                  └──────────────────────┘
```

### 7.2 Token 路由机制

```
Token 路由决策:
  Token[0] → sessionIds[0] = 3 → attnRankTable[3] = Rank 7 → 发到 Rank 7 的 Window
  Token[1] → sessionIds[1] = 1 → attnRankTable[1] = Rank 2 → 发到 Rank 2 的 Window
  ...

无 RankTable 模式 (isInputRankTable=false):
  sessionIds 直接作为目标 Rank ID
  Token[0] → sessionIds[0] = 3 → 直接发到 Rank 3

有 RankTable 模式 (isInputRankTable=true):
  sessionIds 作为查表索引
  Token[0] → sessionIds[0] = 3 → attnRankTable[3] = Rank 7 → 发到 Rank 7
```

---

## 8. 性能相关设计

### 8.1 纯 AIV 架构

本算子仅使用 `KERNEL_TYPE_AIV_ONLY`（无 AIC 参与）。所有数据传输通过 `DataCopyPad` 指令完成，Token 元数据通过 `GetValue` 逐元素读取。

### 8.2 双缓冲流水线

`BUFFER_NUM=2`（第36行）的双缓冲队列深度允许一个 Token 的 DataCopy 写入 Window 的同时，下一个 Token 的 x 数据从 GM 加载到 UB，实现传输-加载重叠。

### 8.3 多核并行

`SplitToCore(tokenNum)`（第169-222行）以轮询方式将 Token 分配到多个 AIV 核，每个核独立处理自己负责的 Token 子集，互不干扰。各核写入不同 Rank 的 Window 区域，无需跨核同步。

---

## 9. 与 attention_to_ffn 的核心差异

| 特性 | attention_to_ffn | ffn_to_attention |
|------|-----------------|-----------------|
| 方向 | Attention → FFN | FFN → Attention |
| Token 来源 | Attention 输出 | FFN 输出 |
| Expert 路由 | expertIds → expertRankTable 查表 | sessionIds → attnRankTable 查表 |
| 量化 | 支持 (动态量化 FP16→INT8) | 不支持 |
| 同步模式 | sync/async 可选 | 默认同步 |
| Active Mask | 支持 | 不支持 |
| TilingKey 数 | 8 种 | 2 种 |
| Token 元数据 | sessionId, microBatchId, layerId, expertIds | sessionId, microBatchId, tokenId, expertOffsets |
| 主类大小 | 621 行 | ~300 行 (更简洁) |
| Window 目标 | FFN Worker 的 Window | Attention Worker 的 Window |

---

## 10. 函数调用关系

```
ffn_to_attention (内核入口)
└── FFNToAttention::Init()
    ├── 设置 7 个输入 GM 地址
    ├── 读取 Tiling (H, BS, expertNumPerToken, HS, aivNum, worldSize)
    ├── 初始化 TQue<VECIN, 2> (双缓冲)
    ├── 分配 UB: xLocal_[2] (每个 H * sizeof(xType))
    └── 计算偏移量 (batchSizeSendCnt_, perMicroBatchOffset_, ...)

└── FFNToAttention::Process()
    ├── 读取 actualTokenNum
    ├── SplitToCore(tokenNum): 轮询分配到 AIV
    └── for each assigned token:
        ├── TQue::Alloc() → 获取 xLocal
        ├── DataCopy(xGM → xLocal): 复制 H 元素
        ├── ReadTokenMetaData(tokenIdx):
        │   ├── sessionIds → curAttenWorkIds
        │   ├── microBatchIds → curMicroBatchIds
        │   ├── tokenIds → curTokenBatchOffset
        │   ├── expertOffsets → curTokenTopkOffset
        │   └── [isInputRankTable] attnRankTable → curAttenWorkRank
        ├── GetWindowAddr(curAttenWorkRank) → targetWindow
        ├── 计算 infoTableOffset + tokenDataOffset
        ├── DataCopyPad(xLocal → Window + tokenDataOffset)
        ├── DataCopyPad(flag → Window + infoTableOffset)
        └── TQue::Free() → 释放 xLocal
```

---

*分析完成时间: 2026-04-17，修订时间: 2026-04-21（补充行号引用、Host-Kernel接口、完整数据流、性能设计章节）*
