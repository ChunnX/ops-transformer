# Moe EP Win区异常Dump实施指导

## 1. 实施约束

- 适配范围为 `MoeEpDispatch`、`MoeEpDispatchEpilogue`、`MoeEpCombine`，以及已经实现的 `MoeEpCombineEpilogue`。
- 本文假设 `MoeEpCombineEpilogue` 已按业务设计实现。如果当前代码中不存在该算子，异常Dump改造不得包含拆分或新增CombineEpilogue的动作。
- 9974仅作为方案设计参考，不复用其中的代码、类型或接口。
- 仅复用原有 `mc2_exception_dump` 中适用的异常参数获取、Dump路径、文件命名和文件落盘逻辑；不适用的Context解析和结构化Dump逻辑重新实现。
- 不考虑图模式和图依赖适配。
- Win区固定新增64 KiB Metadata区和 `100 * 512 B` Per-core维测区，总固定前缀为114 KiB，即116736 B。
- Metadata不包含 `magic`、`valid` 和动态region id。Region通过固定数组下标表达语义，每项仅包含 `offset` 和 `size`。
- Metadata只描述控制类Region，不描述、也不Dump任何通信数据区。
- Metadata由当前执行到的算子写入。四个算子使用公共布局函数生成相同内容，并在各自Init阶段由core 0写入。
- 禁止在GM上执行大量Scalar写。Metadata和Per-core维测信息先在UB准备，再通过DataCopy写入Win区。
- 不新增Per-core槽清零操作，未写区域由HCCL buffer初始化保证。实际维测字段更新仍使用UB到GM的搬运，不逐字段写GM。
- Per-core槽允许后续整槽或按对齐范围覆盖，本方案不保留跨算子的历史记录。
- 暂不增加 `devArgsLen`、`epRankId`、`aivNum` 上界检查，暂不增加总Dump大小上限校验。
- 异常时只Dump本卡 `epRankId` 对应的本地Win。
- Metadata固定完整D2H 64 KiB；每个Region分别输出一个文件。
- 任意D2H失败时记录Dump错误并退出，不回退到raw dump。

## 2. 代码目录结构

```text
mc2/common/
├─ op_kernel/
│  ├─ moe_ep_exception_dump_defs.h
│  │    固定前缀、Region下标、Metadata和Per-core公共定义
│  └─ moe_ep_exception_dump_writer.h
│       Metadata及Per-core信息的UB构造和DataCopy写入
│
├─ op_host/op_tiling/
│  ├─ moe_ep_window_layout.h
│  └─ moe_ep_window_layout.cpp
│       direct/hybrid统一Win布局计算及Metadata生成
│
└─ utils/
   ├─ mc2_exception_dump_common.h
   │    从原mc2_exception_dump抽取的通用异常参数、路径和文件操作
   └─ moe_ep_exception_dump.h
        MoeCommContext解析、Metadata读取、Region D2H及文件输出
```

算子目录适配：

```text
mc2/moe_ep_dispatch/
mc2/moe_ep_dispatch_epilogue/
mc2/moe_ep_combine/
mc2/moe_ep_combine_epilogue/       # 仅在该算子已经存在时适配
torch_extension/cann_ops_transformer/ops/elastic_buffer.py
```

## 3. 公共Win Layout计算

### 3.1 提取范围

Dispatch、DispatchEpilogue、Combine必须共用同一个Win layout纯计算函数。公共函数负责：

- 64 KiB Metadata区和 `100 * 512 B` Per-core区；
- Count notify、Expert count、Dispatch slot state和Combine state布局；
- Dispatch、Scaleout和Combine的固定预留slot大小；
- direct/hybrid数据区offset和size；
- `dispatchNotifyCount`；
- HCCL buffer最小字节数。

公共layout不得依赖 `gert::TilingContext` 或具体算子的tiling结构，不直接读取attr，不设置workspace、blockDim、tiling key和schedule mode。

以下内容保留在算子侧：

- 输入输出shape、dtype和format校验；
- 从各算子不同的attr下标读取参数；
- Dispatch基于真实dtype、scales计算的实际通信slot大小；
- Dispatch的send entry、workspace和通信AIV分配；
- DispatchEpilogue的hit count workspace；
- Combine的workspace总大小；
- 公共结果向各算子tiling字段的赋值。

### 3.2 公共定义

```cpp
constexpr uint64_t MOE_EP_UB_ALIGN = 32;
constexpr uint64_t MOE_EP_WIN_ALIGN = 512;
constexpr uint64_t MOE_EP_NOTIFY_TOKEN_LIMIT = 15000;
constexpr uint64_t MOE_EP_MAX_OUT_DTYPE_SIZE = 2;
constexpr uint64_t MOE_EP_METADATA_DTYPE_SIZE = 4;

uint64_t AlignMoeEpUb(uint64_t value);
uint64_t AlignMoeEpWin(uint64_t value);
```

公共layout输入：

```cpp
struct MoeEpWindowLayoutParams {
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t nmt;
    uint32_t topK;
    uint32_t hidden;
    uint32_t networkMode;
    uint32_t rankNumPerServer;
    uint32_t serverNum;
};
```

`epRankId`、实际dtype和实际 `perSlotBytes` 不参与固定Win物理布局计算。

### 3.3 公共layout结果

```cpp
struct MoeEpWindowLayout {
    uint32_t dispatchNotifyCount;

    uint64_t metadataOffset;
    uint64_t metadataSize;
    uint64_t perCoreDiagOffset;
    uint64_t perCoreDiagSize;

    uint64_t cntWinStateOffset;
    uint64_t cntWinStateSize;
    uint64_t slotWinStateOffset;
    uint64_t dispatchSlotStateSize;
    uint64_t combineStateWinOffset;
    uint64_t combineStateWinSize;

    uint64_t scaleoutRecvDataOffset;
    uint64_t scaleoutRecvDataSize;
    uint64_t winDataOffset;
    uint64_t dispatchRecvDataSize;
    uint64_t scaleoutRecvStatusOffset;
    uint64_t scaleoutRecvStatusSize;
    uint64_t combineDataWinOffset;
    uint64_t combineDataSize;
    uint64_t payloadStashWinOffset;
    uint64_t payloadStashWinSize;

    uint64_t dispatchReservedPerSlotBytes;
    uint64_t scaleoutReservedPerSlotBytes;
    uint64_t combineReservedPerSlotBytes;
    uint64_t requiredBytes;
};

ge::graphStatus CalcMoeEpWindowLayout(
    const MoeEpWindowLayoutParams &params,
    MoeEpWindowLayout &layout);
```

`CalcMoeEpWindowLayout`只计算布局。容量检查单独处理：

```cpp
ge::graphStatus CheckMoeEpWindowCapacity(
    uint64_t requiredBytes,
    uint64_t cclBufferSize,
    const char *nodeName);
```

### 3.4 拓扑归一化

各算子仍从自己的attr位置获取参数，但以下数值校验和归一化逻辑共用：

```text
serverNum = epWorldSize / rankNumPerServer

networkMode =
    requestedNetworkMode == HYBRID && serverNum > 1
        ? HYBRID
        : DIRECT
```

建议接口：

```cpp
struct MoeEpTopology {
    uint32_t networkMode;
    uint32_t rankNumPerServer;
    uint32_t serverNum;
};

ge::graphStatus ResolveMoeEpTopology(
    uint32_t epWorldSize,
    int64_t requestedNetworkMode,
    int64_t rankNumPerServer,
    MoeEpTopology &topology);
```

### 3.5 算子结果映射

| 公共结果 | Dispatch | DispatchEpilogue | Combine |
|---|---|---|---|
| `dispatchNotifyCount` | 写入tiling | 写入tiling | 不需要 |
| `cntWinStateOffset` | 写入tiling | 不需要 | 不需要 |
| `slotWinStateOffset` | 写入tiling | 写入tiling | 不需要 |
| `winDataOffset` | 写入tiling | 写入tiling | 不需要 |
| `scaleoutRecvDataOffset` | 写入tiling | 不需要 | 不需要 |
| `scaleoutRecvStatusOffset` | 写入tiling | Metadata使用 | Metadata使用 |
| `payloadStashWinOffset` | 写入tiling | 不需要 | 不需要 |
| `combineStateWinOffset` | Metadata使用 | Metadata使用 | 写入tiling |
| `combineDataWinOffset` | 不写Metadata | 不需要 | 写入tiling |
| `combineDataSize` | 不写Metadata | 不需要 | 用于workspace计算 |
| `requiredBytes` | CCL容量校验 | CCL容量校验 | CCL容量校验 |

三个算子都根据完整公共layout结果生成相同Metadata，不允许仅根据各自tiling中保留的字段反推Metadata。

如果已经存在CombineEpilogue，则它复用同一个layout结果，映射方式与Combine接收阶段需要的字段一致；本改造不新增或拆分CombineEpilogue。

### 3.6 调用顺序

公共物理布局不依赖 `aivNum`，但Metadata包含 `aivNum`。各算子统一采用以下顺序：

```text
读取并校验公共attr
-> ResolveMoeEpTopology
-> 检查算子输入输出
-> 获取平台aivNum
-> CalcMoeEpWindowLayout
-> CheckMoeEpWindowCapacity
-> FillMoeEpDumpMetadata(params, layout, aivNum)
-> 将公共结果映射到算子tiling
-> 计算算子专有workspace和tiling key
```

Metadata生成与layout纯计算分离，保证不同算子的公共物理布局不受各自AIV调度逻辑影响。

## 4. Metadata定义

Metadata区固定为64 KiB，有效结构位于区域开头。有效结构按512 B对齐并满足 `sizeof(MoeEpDumpMetadata) <= 512 B`，剩余空间保留。

```cpp
struct MoeEpDumpRegion {
    uint64_t offset;
    uint64_t size;
};

struct MoeEpDumpMetadata {
    uint32_t layoutVersion;
    uint32_t nmt;
    uint32_t topK;
    uint32_t hidden;
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t aivNum;
    uint32_t networkMode;
    uint32_t serverNum;
    uint32_t rankNumPerServer;
    MoeEpDumpRegion regions[MOE_EP_DUMP_REGION_COUNT];
};
```

固定Region下标：

```text
0  PER_CORE_DIAG
1  COUNT_NOTIFY
2  EXPERT_COUNT
3  DISPATCH_SLOT_STATE
4  COMBINE_TOKEN_STATE
5  COMBINE_STATE_TAIL
6  HYBRID_SCALEOUT_STATUS
```

Metadata自身固定从Win首地址读取，不需要在Region数组中重复描述。direct模板的 `HYBRID_SCALEOUT_STATUS` 填 `{0, 0}`。

## 5. 适配点

### 5.1 公共Host布局

新增统一的 `BuildMoeEpWindowLayout`，由Dispatch、DispatchEpilogue、Combine以及已经存在的CombineEpilogue调用。该函数同时生成：

- 各算子tiling需要的Win offset；
- CCL buffer最小字节数；
- 内容一致的 `MoeEpDumpMetadata`。

不得继续由各算子分别维护同一套Win布局公式。

### 5.2 Kernel写入

- core 0在每个算子Init阶段将Host生成的Metadata从tiling搬入UB，再写入本地Win的Metadata区。
- 不对完整64 KiB Metadata区执行清零，只覆盖有效Metadata结构。
- 不新增Per-core槽清零。后续写入维测字段时，先在UB准备有效内容，再搬运到：

```text
winBase + 64 KiB + coreId * 512 B
```

- `MoeEpCombineEpilogue`仅在已经存在时增加相同的Metadata和Per-core写入适配，不在本改造中创建或拆分该算子。

### 5.3 Host异常回调

为四个算子分别注册薄封装回调，统一进入：

```cpp
MoeEpExceptionImpl(args, userdata, opName);
```

原 `mc2_exception_dump` 仅复用以下适用逻辑：

- `aclrtGetArgsFromExceptionInfo` 动态获取；
- Dump路径获取；
- stream、task、时间戳文件命名；
- 文件落盘。

MoeEp的Context解析、Win首地址解析、Metadata解析和分Region D2H使用专用实现。

### 5.4 Torch extension

修改 `ElasticBuffer.get_moe_ep_ccl_buffer_size`，在字节级布局计算中增加一次116736 B固定前缀，再执行现有MiB粒度取整。不得为四个算子分别重复增加前缀。

## 6. 异常Dump流程

```text
算子异常
  -> 对应算子异常回调
  -> 获取异常kernel参数地址和deviceId
  -> 从kernel首个参数D2H MoeCommContext
  -> winBase = epHcclBuffer[epRankId]
  -> D2H [winBase, winBase + 64 KiB) 到metadata.bin
  -> 解析固定Metadata结构
  -> 按固定Region下标逐个处理size非0的Region
       -> D2H winBase + offset, size
       -> 每个Region单独写文件
  -> 完成
```

任意Context、Metadata或Region D2H失败时：

```text
记录错误 -> 停止当前Dump -> 返回
```

不读取其他rank的Win，不回退到固定1 MiB raw dump，不Dump数据区。

建议文件名：

```text
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.metadata.bin
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.per_core_diag.bin
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.count_notify.bin
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.expert_count.bin
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.dispatch_slot_state.bin
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.combine_token_state.bin
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.combine_state_tail.bin
mc2_exception_info_<op>.<stream>.<task>.<timestamp>.<rank>.hybrid_scaleout_status.bin
```

## 7. Region Offset和Size

定义：

```text
A = 512
M = 64 KiB = 65536
C = 100 * 512 = 51200
P = M + C = 116736

N = nmt
K = topK
H = hidden
E = epWorldSize
L = localExpertNum
S = serverNum
Q = ceil(N / 15000)

Align32(x)  = ceil(x / 32) * 32
Align512(x) = ceil(x / 512) * 512
```

`Q`为每个源rank预留的Dispatch notify批次数。Dispatch将实际发送token拆成Q组，每组对应一个512 B notify槽；DispatchEpilogue等待 `E * Q` 个notify。

固定区域：

| 区域 | Offset | Size |
|---|---:|---:|
| Metadata | `0` | `65536` |
| Per-core诊断区 | `65536` | `51200` |

控制类Region：

```text
CountNotifyOffset = P
CountNotifySize   = E * A

ExpertCountOffset = CountNotifyOffset + CountNotifySize
ExpertCountSize   = E * Align512(L * sizeof(int32_t))

DispatchSlotStateOffset = ExpertCountOffset + ExpertCountSize
DispatchSlotStateSize   = Q * E * A

CombineTokenStateOffset = DispatchSlotStateOffset + DispatchSlotStateSize
CombineTokenStateSize   = N * K * A

CombineStateTailOffset = CombineTokenStateOffset + CombineTokenStateSize
CombineStateTailSize   = E * A

DataBase = CombineStateTailOffset + CombineStateTailSize
```

对应Metadata：

| Region | Offset | Size |
|---|---:|---:|
| `PER_CORE_DIAG` | `M` | `C` |
| `COUNT_NOTIFY` | `P` | `E * A` |
| `EXPERT_COUNT` | `P + E*A` | `E * Align512(L*4)` |
| `DISPATCH_SLOT_STATE` | `ExpertCountOffset + ExpertCountSize` | `Q * E * A` |
| `COMBINE_TOKEN_STATE` | `DispatchSlotStateOffset + DispatchSlotStateSize` | `N * K * A` |
| `COMBINE_STATE_TAIL` | `CombineTokenStateOffset + CombineTokenStateSize` | `E * A` |

hybrid模板还需要描述状态区：

```text
HiddenAlign = Align32(H * 2)
TopKAlign = Align32(K * sizeof(int32_t))

DispatchReservedPerSlot =
    Align512(HiddenAlign + 2 * TopKAlign + 32)

ScaleoutReservedPerSlot =
    Align512(DispatchReservedPerSlot + K * sizeof(int32_t))

ScaleoutRecvDataSize = S * N * ScaleoutReservedPerSlot
ScaleupFinalRecvDataSize = E * N * DispatchReservedPerSlot

HybridScaleoutStatusOffset =
    DataBase + ScaleoutRecvDataSize + ScaleupFinalRecvDataSize

HybridScaleoutStatusSize = S * N * A
```

因此：

| 模板 | `HYBRID_SCALEOUT_STATUS` |
|---|---|
| direct | `{0, 0}` |
| hybrid | `{HybridScaleoutStatusOffset, S*N*A}` |

虽然计算hybrid状态区offset需要经过两个数据区的size，但这两个数据区不写入Metadata，也不执行异常Dump。

## 8. HCCL Buffer Size

HCCL buffer仍需为功能数据区分配空间，即使数据区不参与Dump。

公共定义：

```text
CombineReservedPerSlot = Align512(HiddenAlign + 32)

StateAndDumpBytes = DataBase
```

direct模板：

```text
DispatchRecvDataSize = E * N * DispatchReservedPerSlot
CombineDataSize = N * K * CombineReservedPerSlot
PayloadStashSize = E * N * DispatchReservedPerSlot

DirectRequiredBytes =
    StateAndDumpBytes
    + DispatchRecvDataSize
    + CombineDataSize
    + PayloadStashSize
```

hybrid模板：

```text
ScaleoutRecvDataSize = S * N * ScaleoutReservedPerSlot
ScaleupFinalRecvDataSize = E * N * DispatchReservedPerSlot
ScaleoutRecvStatusSize = S * N * A
CombineDataSize = N * K * CombineReservedPerSlot
PayloadStashSize = N * ScaleoutReservedPerSlot

HybridRequiredBytes =
    StateAndDumpBytes
    + ScaleoutRecvDataSize
    + ScaleupFinalRecvDataSize
    + ScaleoutRecvStatusSize
    + CombineDataSize
    + PayloadStashSize
```

算子Host tiling已知network mode时，使用对应模板的RequiredBytes检查 `ccl_buffer_size`。

Torch extension无法取得实际拓扑时，沿用现有保守策略：枚举 `epWorldSize` 的有效 `rankNumPerServer`，取所有hybrid布局和direct布局中的最大值，然后执行现有MiB粒度取整：

```text
MinimumWindowBytes =
    max(DirectRequiredBytes, max(HybridRequiredBytes))
```

固定114 KiB前缀已经包含在 `StateAndDumpBytes/DataBase` 中，不得再次叠加。
