# attention_to_ffn 算子全面分析报告

## 1. 算子概述

`attention_to_ffn` 是 **Attention→FFN 流水线并行** 的数据传输算子。在 Pipeline Parallel 模型中，Attention 阶段完成的 Token 数据需要发送给 FFN 阶段，本算子负责将 Token 通过 HCCL Window 共享内存机制传输到目标 FFN Worker 的 Window 内存区域。

```
Attention Worker                    FFN Worker
┌──────────────────┐               ┌──────────────────┐
│ Token Data (GM)  │               │ Window Memory    │
│ Expert ID Table  │               │  ├ Token Info    │
│ Expert Rank Table│──Window IPC──→│  ├ Token Data    │
│ Active Mask      │               │  └ Status Flag   │
└──────────────────┘               └──────────────────┘
```

### 1.1 模板参数

```cpp
// attention_to_ffn_tiling_key.h
ASCENDC_TPL_ARGS_DECL(AttentionToFFN,       // 第21-26行: 模板参数声明
    TILINGKEY_QUANT,       // bool: 是否启用量化
    TILINGKEY_SYNC,        // bool: 是否同步模式
    TILINGKEY_ACTIVE_MASK, // bool: 是否有 active mask
    ARCH_TAG);             // uint8: 架构标记
ASCENDC_TPL_SEL(                            // 第27-85行: 8种有效组合 (2^3 × A3架构)
    ...
);
```

---

## 2. 目录结构与文件职责

```
attention_to_ffn/
├── op_kernel/
│   ├── attention_to_ffn.cpp              (45行)   内核入口
│   ├── attention_to_ffn.h                (621行)  主类实现
│   ├── attention_to_ffn_tiling.h         (53行)   Tiling 结构
│   └── attention_to_ffn_tiling_key.h     (87行)   模板参数
├── op_host/
│   ├── attention_to_ffn_def.cpp          (89行)   算子定义
│   ├── attention_to_ffn_infershape.cpp   (33行)   Shape 推导
│   └── op_tiling/
│       └── attention_to_ffn_tiling.cpp   (655行)  Tiling 策略
├── op_api/
│   ├── aclnn_attention_to_ffn.cpp        (103行)  ACLNN 实现
│   └── aclnn_attention_to_ffn.h         (72行)   ACLNN 声明
├── op_graph/
│   ├── attention_to_ffn_gen_task.cpp     (82行)   图模式任务生成
│   └── attention_to_ffn_proto.h          (66行)   图模式算子注册
├── config/ascend910_93/
│   ├── attention_to_ffn_binary.json      (287行)  二进制配置
│   └── attention_to_ffn_simplified_key.ini (13行)
├── examples/
│   └── test_aclnn_attention_to_ffn.cpp   (327行)  多卡示例
└── tests/ut/
    ├── op_api/test_aclnn_attention_to_ffn.cpp (107行)
    ├── op_host/test_attention_to_ffn_infershape.cpp (75行)
    └── op_host/arch32/test_attention_to_ffn_tiling.cpp (1446行)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `attention_to_ffn_def.cpp` (89行)

类定义 `class AttentionToFFN : public OpDef`（第19行），构造函数 `AttentionToFFN(const char *name)`（第21-84行）。

**输入张量**（第22-61行）:

| 序号 | 名称 | 代码行 | 类型 | 维度 | 说明 |
|------|------|--------|------|------|------|
| 0 | x | 第22-26行 | FP16/BF16 | [X, BS, H] | Token 数据 |
| 1 | session_id | 第27-31行 | INT32 | [X, BS] | 会话 ID |
| 2 | micro_batch_id | 第32-36行 | INT32 | [X, BS] | 微批次 ID |
| 3 | layer_id | 第37-41行 | INT32 | [X, BS] | 层 ID |
| 4 | expert_ids | 第42-46行 | INT32 | [X, BS, K] | 每个Token的Expert ID |
| 5 | expert_rank_table | 第47-51行 | INT32 | [moeExpertNum, ...] | Expert→Rank 映射表 |
| 6 | scales | 第52-56行 | FP32 (可选) | - | 量化平滑缩放 |
| 7 | active_mask | 第57-61行 | BOOL (可选) | - | 活跃Token掩码 |

**关键属性**（第62-70行）:

| 属性名 | 代码行 | 类型 | 说明 |
|--------|--------|------|------|
| group | 第62行 | String | HCCL 通信域 |
| world_size | 第63行 | Int | 全局卡数 |
| ffn_token_info_table_shape | 第64行 | ListInt | FFN Token 信息表形状 |
| ffn_token_data_shape | 第65行 | ListInt | FFN Token 数据形状 |
| attn_token_info_table_shape | 第66行 | ListInt | Attention Token 信息表形状 |
| moe_expert_num | 第67行 | Int | MoE 专家数 |
| quant_mode | 第68行 | Int | 量化模式 (0=不量化, 1/2=量化) |
| sync_flag | 第69行 | Int | 同步标志 (0/1) |
| ffn_start_rank_id | 第70行 | Int | FFN 起始 Rank ID |

**平台配置**（第72-84行）: AICore 配置声明于第72-81行，仅支持 `ascend910_93`（第82行）；MC2 HCCL Group 配置于第83-84行。

**算子注册**（第87行）: `OP_ADD(AttentionToFFN)`

### 3.2 Tiling 策略

**文件**: `attention_to_ffn_tiling.cpp` (655行)

#### 3.2.1 TilingKey 编码

```
CalTilingKey():
  key = GET_TPL_TILING_KEY(syncFlag, quantMode>0, activeMask, TILINGKEY_TPL_A3)
  → 8 种组合 (sync × quant × activeMask)
```

#### 3.2.2 参数校验

```
CheckAndSetAttrs():
├── group: 非空字符串
├── worldSize: [2, 768]
├── moeExpertNum: [0, 1024]
├── quantMode: {0, 1, 2}
├── syncFlag: {0, 1}
└── ffnStartRankId: [0, worldSize)

CheckTensorShapeAndSetTinglingData():
├── X == 1
├── BS ∈ (0, 512]
├── H ∈ [1024, 8192]
├── K ∈ (0, min(16, moeExpertNum)]
├── L == 1
└── expertRankTable dim1 ∈ [moeExpertNum, moeExpertNum+4]
```

#### 3.2.3 Workspace 与 Window 计算

```
SetWorkSpace():
├── libApiWorkSpaceSize (平台库 API 需求)
└── ffnNumAlignSize * aivNum (AIV 对齐需求)

CalWinSize():
├── [非量化]: TokenInfoTable + TokenData(HS * sizeof(half))
└── [量化]:   TokenInfoTable + TokenData(HS * sizeof(int8) + dequantScale)
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `attention_to_ffn.cpp` (45行)

入口函数签名（第29-32行）: `template<bool isQuant, bool isSync, bool isActiveMask, uint8_t ArchTag> __global__ __aicore__ void attention_to_ffn(...)`。

Tiling 注册（第34-35行）: `REGISTER_TILING_DEFAULT` 与 `REGISTER_TILING_FOR_TILINGKEY` 声明模板实例化策略。

Tiling 数据获取（第39行）: `GET_TILING_DATA_WITH_STRUCT(AttentionToFFNTilingData, tilingData, tilingGM)` 解析 Host 下发的 Tiling 二进制。

```cpp
template<bool isQuant, bool isSync, bool isActiveMask, uint8_t ArchTag>
__global__ __aicore__ void attention_to_ffn(...)
{
    REGISTER_TILING_DEFAULT(AttentionToFFNTilingData);                      // 第34行
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A3", ...);     // 第35行

    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        GET_TILING_DATA_WITH_STRUCT(AttentionToFFNTilingData, tilingData, tilingGM);  // 第39行
        AttentionToFFN<DTYPE_X, isQuant, isSync, isActiveMask> op;
        op.Init(...);
        op.Process();
    }
}
```

### 4.2 主类 AttentionToFFN

**文件**: `attention_to_ffn.h` (621行)

类模板声明（第57-58行）: `template <typename XType, bool isQuant, bool isSync, bool isActiveMask> class AttentionToFFN`。

#### 模板参数

```cpp
template <typename XType, bool isQuant, bool isSync, bool isActiveMask>
class AttentionToFFN;   // 第57-58行
```

#### Init() — 初始化（第217-269行）

```
Init(x, sessionId, microBatchId, layerId, expertIds, expertRankTable, scales, activeMask,
     workspace, tiling):   // 签名见第217-218行
├── 读取 Tiling:
│   ├── BS, H, K, L, expertNum, moeExpertNum, sharedExpertNum
│   ├── HS (Hidden + Scale), microBatchNum, attentionWorkerNum
│   ├── infoTableLastDimNum, aivNum, worldSize
│   ├── syncFlag, quantMode, ffnStartRankId
│   └── isActiveMask, windowType
├── 从 GM 读取 sessionId/microBatchId/layerId
├── SplitToCore(BS * K): 每个 AIV 分配 (BS*K / aivNum) 个 Token
├── [isQuant] 初始化量化相关 UB buffer
│   ├── xLocal_ (FP32, H 元素)
│   ├── smoothScaleLocal_ (FP32, H 元素)
│   └── quantTempLocal_ (INT32, H 元素)
├── [isActiveMask] 初始化 active mask UB buffer
└── 计算 Window 偏移量:
    ├── perTokenDataSize = HS * sizeof(元素)
    ├── winTokenDataTableOffset = BS * K * infoTableLastDimNum * sizeof(int32)
    └── 每个 Token 的数据偏移 = winTokenDataTableOffset + tokenIdx * perTokenDataSize
```

#### Process() — 主流程（第605-619行）

```
Process():
├── [isActiveMask] ComputeActiveMaskCount()
├── SendTokenToFFN()           ← AIV: 发送所有 Token
├── SyncAll()
├── SetFlagToFFN()             ← 设置完成标志
├── SyncAll()
└── [isActiveMask] SetFlagInAttn()  ← 在 Attention 侧设置标志
```

#### SendTokenToFFN() — Token 分发

```
SendTokenToFFN():
├── SplitToCore(BS * K): 将 Token 分配到各 AIV
├── 复制 expert IDs 到本地 UB
└── for tokenIdx in assignedTokens:
    └── SendTokenToFFNByTokenIdx(tokenIdx)
```

#### SendTokenToFFNByTokenIdx() — 单 Token 发送

```
SendTokenToFFNByTokenIdx(tokenIdx):
├── DataCopy: 从 GM 复制 H 个元素到 xLocal_ (UB)
├── SetExpertAndRank(tokenIdx):
│   ├── expertId = expertIds[tokenIdx]
│   ├── FindExpertRank(expertId):
│   │   └── 查表 expertRankTable[expertId] → (rankId, localExpertId)
│   └── 计算 target rank 和 local expert index
├── CheckFlagAndSetTableGM(expertId, rankId):
│   └── 自旋等待 FFN 侧 info table flag 清零
├── [isQuant] QuantProcess():
│   ├── Cast<half→float>(xLocal_)
│   ├── [isScales] Muls(smoothScaleLocal_) × xLocal_
│   ├── ReduceMax(xLocal_) → absMax
│   ├── Muls(127.0 / absMax) → dynamicScale
│   ├── Cast<float→int32>(xLocal_)
│   ├── Cast<int32→half>(xLocal_)  ← 精度中间转换
│   ├── Cast<half→int8>(xLocal_)
│   └── 追加 dequantScale 到数据末尾
├── 计算 Window 目标地址:
│   ├── tableAddr = windowBase + expertOffset * tableEntrySize * sizeof(int32)
│   └── dataAddr = windowBase + winTokenDataTableOffset + tokenOffset * perTokenDataSize
├── DataCopyPad: 写 Token 数据到 Window
└── DataCopyPad: 写 Token 信息 (flag=1) 到 Window info table
```

#### QuantProcess() — 动态量化流水线

```
量化流水线 (类型转换):
  FP16/BF16 ──Cast──→ FP32
                     │
            [isScales] × smoothScale (FP32)
                     │
  ReduceMax → absMax (FP32)
  Muls(127.0/absMax) → scale (FP32)
                     │
  FP32 ──Cast──→ INT32
  INT32 ──Cast──→ FP16 (中间精度)
  FP16 ──Cast──→ INT8
                     │
  追加 dequantScale = absMax / 127.0 → FP32
```

#### SetFlagToFFN() — 完成标志设置

```
SetFlagToFFN():
├── [isSync 同步模式]:
│   ├── 每个 AIV 将本核发送状态写入特定 slot
│   ├── ReduceSum 汇总所有 AIV 的状态
│   └── 确认所有 AIV 发送完毕后，设置 flag 到 Window
└── [异步模式]:
    └── 每个 AIV 独立设置 flag

Flag 格式:
  flag_value = (magicValue << 16) | 1
  → 高 16 位: magic 值 (用于跨次调用隔离)
  → 低 16 位: 状态 (1=数据就绪)
```

#### SetFlagInAttn() — Attention 侧标志

```
SetFlagInAttn():
├── 仅 isActiveMask=true 时执行
├── 对于未被 mask 的 Token 位置:
│   └── 在本地 attention token info table 中写 flag=1
└── 标记该 Token 已发送
```

#### FindExpertRank() — Expert→Rank 查表

```
FindExpertRank(expertId):
├── rankInfo = expertRankTable[expertId]
├── rankId = rankInfo & 0xFFFF           ← 低 16 位: 目标 Rank
├── localExpertId = rankInfo >> 16       ← 高 16 位: 目标 Rank 内 Expert 索引
└── return (rankId, localExpertId)
```

#### CheckFlagAndSetTableGM() — 自旋等待

```
CheckFlagAndSetTableGM(expertId, rankId):
├── tableGM = GetWindowAddr(rankId) + expertOffset
├── do:
│   ├── DataCacheCleanAndInvalid(tableGM)  ← 刷新 Cache
│   └── 读取 flag = tableGM[flagOffset]
├── while (flag == magicValue)            ← 等待上一次处理的 flag 清除
└── 设置 tableGM 为本次写入地址
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

**文件**: `attention_to_ffn_tiling.h` (53行)

Host 侧 Tiling 计算完成后，将参数打包为 `AttentionToFFNTilingData` 结构体，通过 `tilingGM` 指针传递到 Kernel。

```
AttentionToFFNInfo (第21-45行):
├── X, BS, H, K, L               ← 维度参数
├── expertNum                     ← 总专家数
├── moeExpertNum                  ← MoE 专家数
├── sharedExpertNum               ← 共享专家数
├── HS                            ← Hidden + Scale 大小
├── expRankTableM                 ← Expert Rank Table 行数
├── microBatchNum                 ← 微批次数量
├── attentionWorkerNum            ← Attention Worker 数
├── infoTableLastDimNum           ← Info Table 最后一维大小
├── aivNum                        ← AIV 核数
├── worldSize                     ← 全局卡数
├── syncFlag                      ← 同步标志
├── quantMode                     ← 量化模式
├── ffnStartRankId                ← FFN 起始 Rank
├── isScales                      ← 是否有平滑缩放
├── isActiveMask                  ← 是否有 Active Mask
├── windowType                    ← Window 类型
├── totalUbSize                   ← UB 总大小
└── totalWinSize                  ← Window 总大小

AttentionToFFNTilingData (第47-51行):
├── Mc2InitTiling                 ← HCCL 初始化
├── Mc2CcTiling                   ← 通信参数
└── AttentionToFFNInfo            ← 算子参数
```

### 5.2 Workspace 布局

Host 侧 `SetWorkSpace()` 分配的 Workspace 由两部分组成:

```
Workspace 布局:
┌─────────────────────────────────────────────────────────┐
│ libApiWorkSpaceSize                                     │
│ (HCCL 库 API 内部使用的空间)                             │
├─────────────────────────────────────────────────────────┤
│ ffnNumAlignSize * aivNum                                │
│ (每个 AIV 一段，用于同步模式下的 ffnStatus 缓存)          │
│ ┌──────────┬──────────┬─────┬──────────┐               │
│ │ AIV 0    │ AIV 1    │ ... │ AIV N    │               │
│ │[ffnNum × │[ffnNum × │     │[ffnNum × │               │
│ │ int32]   │ int32]   │     │ int32]   │               │
│ └──────────┴──────────┴─────┴──────────┘               │
└─────────────────────────────────────────────────────────┘

Kernel 侧访问 (Init 第253行):
  syncStatusWorkspaceGM_ = workspaceGM;
  每个 AIV 偏移 = aivWorkspaceOffset_ * get_block_idx()
```

### 5.3 Window 内存布局

```
FFN Worker 的 Window Memory:
┌─────────────────────────────────────────────────────────┐
│ Token Info Table                                        │
│ ┌──────────┬──────────┬─────┬──────────┐               │
│ │Expert 0  │Expert 1  │ ... │Expert E  │               │
│ │[status,  │[status,  │     │[status,  │               │
│ │ sessionId│ sessionId│     │ sessionId│               │
│ │ ...]     │ ...]     │     │ ...]     │               │
│ └──────────┴──────────┴─────┴──────────┘               │
│                                                         │
│ Token Data Region                                       │
│ ┌──────────┬──────────┬─────┬──────────┐               │
│ │Token 0   │Token 1   │ ... │Token N   │               │
│ │[H elem]  │[H elem]  │     │[H elem]  │               │
│ │[quantScale]          │     │[quantScale]              │
│ └──────────┴──────────┴─────┴──────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 完整数据流

本节描述从 Host 侧调用到 Kernel 侧执行完成的端到端数据流。

```
Host 侧                              Device 侧 (Ascend NPU)
─────────                            ──────────────────────

┌─────────────────────┐
│ 1. 用户调用         │
│    aclnnAttention   │
│    ToFFn(...)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 2. GetWorkspaceSize │
│    ├ Tiling 计算    │  attention_to_ffn_tiling.cpp
│    │  CalTilingKey()│  → 确定 isQuant/isSync/isActiveMask 组合
│    │  SetWorkSpace()│  → 计算 Workspace 大小
│    │  CalWinSize()  │  → 计算 Window 大小
│    └ 序列化         │  → AttentionToFFNTilingData 二进制
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌─────────────────────────────────────┐
│ 3. aclnnAttention   │     │ 4. Kernel 入口                      │
│    ToFFn 执行       │────→│    attention_to_ffn.cpp 第29-45行   │
│    (下发到Device)   │     │    ├ REGISTER_TILING_DEFAULT        │
└─────────────────────┘     │    ├ GET_TILING_DATA_WITH_STRUCT    │
                            │    └ 实例化 AttentionToFFN<>        │
                            └──────────────┬──────────────────────┘
                                           │
                            ┌──────────────▼──────────────────────┐
                            │ 5. Init() 第217-269行               │
                            │    ├ InitByTinglingData()           │
                            │    │  解析 Tiling 参数              │
                            │    ├ SetGlobalBuffer()              │
                            │    │  绑定 GM 输入张量              │
                            │    ├ 从 GM 读取 session/microBatch  │
                            │    │  /layerId (DataCacheClean)     │
                            │    ├ InitBuffer() 分配 UB           │
                            │    │  ├ expertIdsBuf_               │
                            │    │  ├ ffnFlagBuf_                 │
                            │    │  ├ [isQuant] QuantInit         │
                            │    │  ├ [isSync] ffnStatusBuf_      │
                            │    │  └ [isActiveMask] activeMaskBuf│
                            │    └ 计算 Window 偏移量             │
                            └──────────────┬──────────────────────┘
                                           │
                            ┌──────────────▼──────────────────────┐
                            │ 6. Process() 第605-619行            │
                            │    ├ [isActiveMask]                 │
                            │    │  ActiveMaskCalCnt()             │
                            │    ├ SendTokenToFFN()               │
                            │    │  └ for each token:             │
                            │    │    └ SendTokenToFFNByTokenIdx()│
                            │    │      ├ DataCopy: GM→UB         │
                            │    │      ├ FindExpertRank()        │
                            │    │      ├ CheckFlagAndSetTableGM()│
                            │    │      ├ [isQuant] QuantProcess()│
                            │    │      ├ DataCopyPad: UB→Window  │
                            │    │      └ 写 flag=1 到 info table│
                            │    ├ PipeBarrier + SyncAll          │
                            │    ├ SetFlagToFFN()                 │
                            │    └ [isActiveMask] SetFlagInAttn() │
                            └─────────────────────────────────────┘
```

### 6.1 Host→Kernel Tiling 传递流程

```
Host Tiling 计算结果 (attention_to_ffn_tiling.cpp)
        │
        │ 序列化为二进制
        ▼
┌─────────────────────────────────────┐
│ AttentionToFFNTilingData 二进制布局 │
│ ┌─────────────────────────────────┐ │
│ │ Mc2InitTiling                   │ │ HCCL 初始化参数
│ │   (mc2InitTiling)               │ │
│ ├─────────────────────────────────┤ │
│ │ Mc2CcTiling                     │ │ 通信参数
│ │   (mc2CcTiling1)                │ │
│ ├─────────────────────────────────┤ │
│ │ AttentionToFFNInfo              │ │ 算子业务参数
│ │   X, BS, H, K, L, expertNum... │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
        │
        │ tilingGM 指针传入 Kernel
        ▼
Kernel 第39行:
  GET_TILING_DATA_WITH_STRUCT(AttentionToFFNTilingData, tilingData, tilingGM);
        │
        ▼
Init 第221行: InitByTinglingData(tilingData)
  → 逐字段提取到成员变量
```

---

## 7. 通信模式

### 7.1 HCCL Window IPC

```
通信架构:
┌──────────────┐   Window IPC   ┌──────────────┐
│ Attention    │ ────WRITE────→ │ FFN Worker   │
│ Worker Rank  │   (共享内存)    │ Window Mem   │
│ AIV Core     │                │              │
└──────────────┘                └──────────────┘

地址解析:
  GetWindowAddr(rankId):
  ├── mc2KernelUtils::GetHcclContext(contextGM, rankId)
  ├── userMemType == 0 → remote device memory
  └── userMemType == 1 → user-defined memory
```

### 7.2 Flag 同步机制

```
Flag 同步时序:
  [Attention Worker]                    [FFN Worker]
       │                                     │
       │  CheckFlag (flag==old_magic?)       │
       │  ←─── 自旋等待 ────→               │  处理上一次数据
       │                                     │  清除 flag
       │  检测到 flag 清除                    │
       │                                     │
       │  DataCopy (Token Data)              │
       │  ────────WRITE──────→               │
       │                                     │
       │  SetFlag (flag = new_magic|1)       │
       │  ────────WRITE──────→               │
       │                                     │  检测到 flag
       │                                     │  读取数据
       │                                     │
```

---

## 8. 与 ffn_to_attention 的关系

`attention_to_ffn` 和 `ffn_to_attention` 是 Pipeline Parallel 中的配对算子:

| 特性 | attention_to_ffn | ffn_to_attention |
|------|-----------------|-----------------|
| 方向 | Attention → FFN | FFN → Attention |
| 数据流 | 发送 Token 到 FFN Worker | 发送结果回 Attention Worker |
| Expert 路由 | 有 (Expert→Rank 查表) | 有 (通过 Token 元数据) |
| 量化 | 支持动态量化 | 无 |
| 同步模式 | sync/async | sync |
| Active Mask | 支持 | 不支持 |

---

## 9. 性能相关设计

### 9.1 纯 AIV 架构（无 Cube 计算）

本算子采用纯 AIV（AI Vector）核心设计，不使用 Cube（矩阵乘）单元。所有操作均为向量级：

- **DataCopy / DataCopyPad**: GM 与 UB 之间的数据搬运
- **Cast**: 类型转换（FP16/BF16 → FP32 → INT32 → INT8）
- **Muls / ReduceMax**: 标量乘法与最大值归约
- **Duplicate**: 常量填充
- **DataCacheCleanAndInvalid**: 缓存一致性维护

这一设计使得算子不与 Attention/FFN 计算中的 Cube 操作竞争硬件资源，可在流水线中与计算重叠执行。

```
硬件资源利用:
┌─────────────────────────────────────┐
│           Ascend A3 AICore          │
│  ┌─────────┐    ┌──────────────┐   │
│  │  Cube   │    │  AIV (使用)  │   │
│  │ (空闲)  │    │  ├ DataCopy  │   │
│  │         │    │  ├ Cast      │   │
│  │         │    │  ├ Muls      │   │
│  │         │    │  └ ReduceMax │   │
│  └─────────┘    └──────────────┘   │
│  ┌─────────────────────────────┐   │
│  │  UB (Unified Buffer)        │   │
│  │  ├ expertIdsBuf_            │   │
│  │  ├ ffnFlagBuf_              │   │
│  │  ├ [isQuant] xLocal_ 等     │   │
│  │  └ [isActiveMask] maskBuf_  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 9.2 多核并行 Token 路由

Token 分发采用多 AIV 核心并行策略。`SplitToCore(BS * K)` 将 BS*K 个 Token 均匀分配到 `aivNum` 个 AIV 核心:

```
Token 分配策略:
  totalTokens = BS * K
  tokensPerAIV = totalTokens / aivNum
  startTokenId = blockIdx * tokensPerAIV
  endTokenId   = startTokenId + tokensPerAIV

示例 (BS=128, K=4, aivNum=4):
  AIV 0: Token 0~127   (128 tokens)
  AIV 1: Token 128~255 (128 tokens)
  AIV 2: Token 256~383 (128 tokens)
  AIV 3: Token 384~511 (128 tokens)

每个 AIV 独立执行:
  for tokenIdx in [start, end):
    SendTokenToFFNByTokenIdx(tokenIdx)
    → 不同 Token 可能路由到不同 FFN Rank
    → 各 AIV 天然实现目标 Rank 的并行写入
```

**关键设计点**:

- 每个 AIV 拥有独立的 expertIds UB 缓冲区，无锁读取
- Window 写入通过 `CheckFlagAndSetTableGM()` 自旋等待保证同一 Expert 位置的顺序性
- `SyncAll()` 在所有 Token 发送完毕后同步，确保全局一致性
- `[isSync]` 模式通过 Workspace 中的 per-AIV status slot + `ReduceSum` 汇总，避免 Flag 竞争

### 9.3 HCCL Window IPC 机制

算子使用 HCCL Window 实现跨 Rank 的零拷贝数据传输，避免了传统通信原语（如 AllToAll/AllGather）的额外拷贝开销:

```
Window IPC 数据通路:
┌──────────────────────┐                   ┌──────────────────────┐
│ Attention Worker     │                   │ FFN Worker           │
│ Rank 0               │                   │ Rank 4               │
│                      │   Window Memory   │                      │
│  AIV Core            │   (共享映射)      │                      │
│  ┌────────────┐      │                   │  ┌────────────┐      │
│  │ Token Data │──────┼───WRITE──────────→│  │ Token Data │      │
│  │ (GM)       │      │  绕过 HCCL 协议栈 │  │ (Window)   │      │
│  └────────────┘      │                   │  └────────────┘      │
│                      │                   │                      │
│  地址获取:           │                   │                      │
│  GetWindowAddr(4)    │                   │                      │
│  → mc2KernelUtils::  │                   │                      │
│    GetHcclContext()  │                   │                      │
│  → 直接返回远端      │                   │                      │
│    Window 映射地址   │                   │                      │
└──────────────────────┘                   └──────────────────────┘

性能优势:
  1. 零拷贝: DataCopyPad 直接写入远端 Window 映射地址
  2. 绕过协议栈: 无 HCCL 集合通信的序列化/反序列化开销
  3. 细粒度同步: 基于 Flag 的自旋等待，而非全局 Barrier
  4. 流水线友好: Token 级别的粒度允许 FFN 侧逐 Token 消费
```

Window 地址解析通过 `mc2KernelUtils::GetHcclContext()` 获取 HCCL 上下文中的远端 Rank 映射地址。`windowType` 字段区分设备内存映射（type=0）与用户自定义内存映射（type=1）。

---

## 10. 函数调用关系

```
attention_to_ffn (内核入口)
└── AttentionToFFN::Init()
    ├── 读取 Tiling
    ├── SplitToCore()
    ├── [isQuant] 初始化量化 buffer
    └── 计算 Window 偏移

└── AttentionToFFN::Process()
    ├── [isActiveMask] ComputeActiveMaskCount()
    ├── SendTokenToFFN()
    │   ├── SplitToCore(BS * K)
    │   └── for each token:
    │       └── SendTokenToFFNByTokenIdx()
    │           ├── DataCopy(xGM → UB)
    │           ├── SetExpertAndRank() → FindExpertRank()
    │           ├── CheckFlagAndSetTableGM() (自旋等待)
    │           ├── [isQuant] QuantProcess()
    │           ├── DataCopyPad(UB → Window data)
    │           └── DataCopyPad(UB → Window info table)
    ├── SyncAll()
    ├── SetFlagToFFN()
    │   ├── [isSync] ReduceSum → 等待所有 AIV
    │   └── 设置 flag = magicValue|1
    ├── SyncAll()
    └── [isActiveMask] SetFlagInAttn()
```

---

*分析完成时间: 2026-04-21*
