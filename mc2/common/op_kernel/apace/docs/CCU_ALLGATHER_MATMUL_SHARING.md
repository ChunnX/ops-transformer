# CCU AllGather + Matmul 通算融合算子——技术分享

> 平台：Ascend 950（`--npu-arch=dav-3510`）
> 通信调度：CCU（`HCCL_OP_EXPANSION_MODE=CCU_SCHED`）
> 仓库：`ops-transformer/mc2/common/op_kernel/apace/examples/ccu_all_gather_matmul/`

---

## 1. 背景：`<<<>>>` 直调 CCU kernel

### 1.1 过去的能力与局限

Ascend C 传统的集合通信（HCCL）通过 Host 侧 `HcclAllGather` 等 API 发起，通信任务由 CANN 运行时内部调度，开发者无法直接控制 CCU（Communication Compute Unit）的执行逻辑。通信和计算之间只能通过 stream event 做粗粒度同步，无法实现细粒度的通算重叠。

### 1.2 新能力：`<<<>>>` 直调 CCU kernel

Ascend 950（dav-3510）平台新增了 **`<<<>>>` 直调 CCU kernel** 的能力——开发者可以像 launch AIC kernel 一样，用 `<<<>>>` 语法直接拉起 CCU kernel：

```cpp
// 直接 launch CCU kernel（与 AIC kernel launch 语法一致）
CcuAllGatherMesh1DMem2MemKernel<<<schd, insHandle, streamCcu>>>(kernelArg);

// 同时 launch AIC kernel（另一条 stream）
all_gather_matmul_kernel<<<usedCoreNum, 0, streamAivAic>>>(
    sendBuf, recvBuf, weightBuf, outputBuf, ...);
```

这意味着：
- **CCU kernel 是用户自定义代码**，开发者完全掌控通信逻辑（轮次、参数、同步）
- **CCU 和 AIC 在不同 stream 上并行**，可以通过 CKE Event 做细粒度跨流同步
- **通算流水成为可能**——CCU 通信 round d+1 与 AIC 计算 tile d 真正并行

### 1.3 本算子做什么

基于 `<<<>>>` 直调 CCU kernel 的能力，实现 **AllGather + MXFP8 Matmul 通算融合**：

- scale 一次性全量 AllGather（所有 M 行 scale），data 按 `tileM` 分 `commTurn` 轮 AllGather
- CCU kernel 内部循环 `commTurn+1` 次（1 次 scale + commTurn 次 data），每次完成一轮 AllGather
- AIC 先轮询 scale done，再循环 `commTurn` 次：每轮 data 通信完成后立即启动对应 tile 的 MXFP8 Matmul
- **round d 的 Matmul 与 round d+1 的 data AllGather 并行执行**（通算重叠）

---

## 2. 硬件角色

| 单元 | 角色 | 本算子中的职责 |
|---|---|---|
| **CCU**（Communication Compute Unit） | 专用通信微核 | AllGather 数据传输（卡间） |
| **AIC**（AI Core） | 矩阵计算核 | Blaze Matmul（卡内） + 轮询 CKE Event 的 GM 后背地址（同步） |
| **Host CPU** | 任务编排 | 资源分配 + 双流 launch |

### 双流架构

```
streamCcu      ──→  1 次 CCU kernel launch（kernel 内循环 commTurn+1 次：1 次 scale AG + commTurn 次 data AG）
streamAivAic   ──→  1 次 AIC kernel launch（轮询 CKE Event：scale done + commTurn 次 data done + MXFP8 Matmul）
```

两条 stream 并行执行，通过 CKE Event 的 GM 后背地址实现跨流同步。

> **纯 AIC**：GemmUniversal（`kernel_qbmm_mx.h`）的 `Run` 内置 `if ASCEND_IS_AIV { return; }`，天然纯 AIC，无需声明 `KERNEL_TYPE_MIX_AIC_1_1`，不再浪费 AIV 核资源。

---

## 3. 核心接口：CCU 通信体系

CCU 通信体系分为两层：**Host 侧资源管理**和 **kernel 侧通信原语**。Host 侧负责分配通信资源（channel、instance、variable、event），kernel 侧在 `__ccu_host__` 函数中使用这些资源执行实际通信。

### 四个核心接口

kernel 侧最核心的 4 个接口，一轮 AllGather 的完整闭环仅靠它们即可描述：

| 接口 | 作用 | 代码位置 |
|---|---|---|
| `ccu::WriteVariableWithNotify` | 将本 rank 的 Variable（output 地址、token）写入 channel 并通知 peer，建立通信参数桥 | CCU kernel 循环体，L641/643 |
| `ccu::NotifyWait` | 等待 peer rank 的 Notify 到达（参数桥就绪 / 传输完成确认） | L649（参数桥就绪）/ L658（传输完成） |
| `ccu::Write` | 卡间数据传输（本 rank src → peer rank dst），AllGather 的核心传输原语 | `ExecuteAllGatherTransfer` 内，L573 |
| `ccu::EventRecord` / `ccu::EventWait` | Event 信号量——CCU→AIC 通知通信完成、CCU 内部等所有 rank 传输完成 | L624 等 wake / L581 记录完成 / L584 等所有 rank / L660 通知 AIC done |

```
__ccu_host__ void AllGatherKernel(arg)
{
    // 初始化：从 channel 获取 peer 的 output/token Variable（通信目标地址）
    for (peer = 0; peer < rankSize; peer++)
        if (peer != rankId)
            output[peer] = GetResByChannel<Variable>(channel, OUTPUT_XN_ID);

    // round 0：等 AIC 唤醒
    ccu::EventWait(events[0]);                          // ④ 等 wake

    for (d = 0; d < commTurn; d++) {
        // ① 建立参数桥：把 output 地址 + token 写入 channel，通知 peer
        for (i = 0; i < channelCount; i++) {
            ccu::WriteVariableWithNotify(channels[i], output[rankId], OUTPUT_XN_ID, ...);
            ccu::WriteVariableWithNotify(channels[i], token[rankId],  TOKEN_XN_ID,  ...);
            // ② 确认参数桥就绪：等 peer 的 output + token 也写好
            ccu::NotifyWait(channels[i], allBit);
        }

        // ③ 执行卡间数据传输：向每个 peer 发送数据
        for (peer = 0; peer < rankSize; peer++) {
            if (peer != rankId)
                ccu::Write(channel, dst[peer], src, sliceSize, event, mask);
        }
        // 本地拷贝（省略 GroupCopy 细节）

        // ④ 传输完成同步：本 rank 完成 → 等所有 rank 完成
        ccu::EventRecord(event, 1 << rankId);
        ccu::EventWait(event, totalMask);

        // ④ 通知 AIC：round d 通信 done，可以 matmul
        ccu::EventRecord(events[d + 1]);
    }
}
```

**闭环**：`WriteVariableWithNotify` 建桥 → `NotifyWait` 确认就绪 → `Write` 传数据 → `EventRecord`/`EventWait` 通知完成。每轮 AllGather 都走一遍这个闭环。

### 3.1 Host 侧资源管理

通信执行前，Host 侧需要完成以下资源编排：

#### 通信域与通道

| 接口 | 作用 | 本算子用法 |
|---|---|---|
| `HcclCommInitRootInfo` | 初始化 HCCL 通信域（多 rank 协商） | 每个线程（rank）调用，传入 rootInfo |
| `HcclGetRankId` / `HcclGetRankSize` | 查询本 rank 编号 / 总 rank 数 | 确定通信拓扑 |
| `HcclRankGraphGetLinks` | 查询本 rank 与 peer rank 的链路信息 | 获取物理链路（协议、地址） |
| `HcclChannelDescInit` | 初始化通道描述符 | 设置 notifyNum、channelProtocol 等 |
| `HcclChannelAcquire` | 获取 CCU 通信通道 | 每 peer rank 一个 channel，后续所有通信走 channel |

> **channel** 是 CCU 通信的基础——每个 channel 对应一个 peer rank，承载双向数据传输和 Variable/Notify 同步。

#### CCU 实例

| 接口 | 作用 | 本算子用法 |
|---|---|---|
| `HcommCcuInsResDescCreate` | 创建资源描述符（指定 dieId） | 为每个可用 die 创建描述符 |
| `HcommCcuInsCreate` | 创建 CCU 实例（分配硬件资源） | 传入所有 die 的描述符 |
| `HcclCommAssignCcuIns` | 将 CCU 实例绑定到通信域 | 所有 rank 共享同一 CCU 实例配置 |
| `HcommCcuInsDestroy` | 销毁 CCU 实例 | 程序结束时释放 |

> **CCU 实例**是通信硬件资源的抽象——包含 die、channel buffer、variable pool、event pool 等。`<<<>>>` launch 时需要传入 `insHandle`。

#### Variable（参数传递"信箱"）

| 接口 | 作用 | 本算子用法 |
|---|---|---|
| `HcommCcuVariableAlloc` | 分配 N 个 Variable 资源 | 分配 `commTurn × 10` 个（每 round 10 个参数） |
| `HcommCcuVariableGetAddr` | 获取 Variable 的 GM 后背地址 | AIC 通过此地址写入参数值，CCU 读取 |

> **Variable** 是 AIC → CCU 的参数传递机制。每个 Variable 有一块 GM 后背地址：AIC 用 `WriteGmByPassDCache` 写入参数值，CCU kernel 内用 `vars[i]` 读取。本算子每 round 需要 10 个 Variable（sendBuf 地址、recvBuf 地址、token、inputOffset、outputOffset、sliceSize、goSize×4）。

#### Event（跨流同步信号量）

| 接口 | 作用 | 本算子用法 |
|---|---|---|
| `HcommCcuEventAlloc` | 分配 N 个 Event 资源 | 分配 `commTurn + 1` 个（1 wake + commTurn done） |
| `HcommCcuEventGetAddr` | 获取 Event 的 GM 后背地址 | AIC 轮询 done，CCU EventRecord done |

> **Event** 是 CCU ↔ AIC 跨流同步的核心。每个 Event 有一块 GM 后背地址：CCU 用 `EventRecord` 写入信号，AIC 用 `ReadGmByPassDCache` 轮询读取。

#### 内存 Token

| 接口 | 作用 | 本算子用法 |
|---|---|---|
| `HcommCcuGetMemToken` | 获取 GM 内存的访问 token | CCU 传输 sendBuf 数据时需要 token 鉴权 |

### 3.2 CCU kernel 侧通信原语（`__ccu_host__`）

CCU kernel 用 `__ccu_host__` 修饰，通过 `<<<>>>` 拉起。其中最核心的 4 个接口见上文 [四个核心接口](#四个核心接口) 表，以下按功能分组展开使用细节：

#### Variable 同步（参数桥就绪通知）

每轮通信开始前，本 rank 需要通知 peer rank "我的参数已就绪"，同时等待 peer rank 的参数就绪：

| 接口 | 作用 |
|---|---|
| `ccu::WriteVariableWithNotify` | 将本 rank 的 Variable（output 地址、token）写入 channel，并通知 peer |
| `ccu::NotifyWait` | 等待 peer rank 的 Variable 写入完成（阻塞直到 Notify 到达） |

```cpp
// 单 channel 用法示例（实际对每个 channel 循环调用）
ccu::WriteVariableWithNotify(channel, output, OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
ccu::WriteVariableWithNotify(channel, token,  TOKEN_XN_ID,  CKE_IDX_0, 1 << TOKEN_XN_ID);

// 等 peer 的 output/token 都就绪
uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
ccu::NotifyWait(channel, CKE_IDX_0, allBit);
```

#### 数据传输

| 接口 | 作用 |
|---|---|
| `ccu::Write` | 卡间数据传输（本 rank src → peer rank dst），带 Event 同步 |
| `ccu::LocalCopy` | 本地拷贝（sendBuf → recvBuf 本 rank 位置），走 CCU buffer 中转 |
| `ccu::GetResByChannel<ccu::Variable>` | 从 channel 获取 peer 的 Variable（output 地址、token） |

> AllGather = N-1 次 `ccu::Write`（发给每个 peer）+ 1 次 `GroupCopy`（本地拷贝）。`GroupCopy` 内部用 `ccu::LocalCopy` + `ccu::Loop` 实现分块拷贝。

#### Event 同步（跨流信号）

| 接口 | 作用 |
|---|---|
| `ccu::EventRecord` | 记录 Event（写入 GM 后背地址，AIC 可轮询感知） |
| `ccu::EventWait` | 等待 Event（阻塞直到所有 mask 的 rank 都 Record） |

```cpp
// CCU round 0 开始前：等 AIC 唤醒
ccu::EventWait(events[0]);           // 阻塞直到 AIC 写 eventVa[0]=1

// CCU round d 完成后：通知 AIC
ccu::EventRecord(events[d + 1]);     // 写入 GM 后背地址，AIC 轮询 eventVa[d+1]==1

// ExecuteAllGatherTransfer 内部：等所有 rank 传输完成
ccu::EventRecord(ctx.event, 1 << rankId);    // 本 rank 传输完成
ccu::EventWait(ctx.event, totalMask);        // 等所有 rank 传输完成
```

#### Notify 同步（传输完成确认）

| 接口 | 作用 |
|---|---|
| `ccu::NotifyRecord` | 记录传输完成信号（POST_SYNC_ID） |
| `ccu::NotifyWait` | 等待所有 rank 的传输完成信号 |

```cpp
// 传输完成后：对每个 channel 记录 + 等待所有 rank 的传输完成信号
for (i = 0; i < channelCount; i++) {
    ccu::NotifyRecord(channel[i], CKE_IDX_0, 1 << POST_SYNC_ID);
}
for (i = 0; i < channelCount; i++) {
    ccu::NotifyWait(channel[i],  CKE_IDX_0, 1 << POST_SYNC_ID);
}
```

### 3.3 `<<<>>>` launch 语法

CCU kernel 的 launch 语法与 AIC kernel 一致，但参数含义不同：

```cpp
CcuAllGatherMesh1DMem2MemKernel<<<schd, insHandle, streamCcu>>>(kernelArg);
//                                    ↑     ↑          ↑          ↑
//                               调度参数  CCU实例    stream     参数
```

| 参数 | 说明 |
|---|---|
| `schd` | CCU 调度参数 `HCommCcuSchd {1, 0, 0x01}` |
| `insHandle` | CCU 实例句柄（Host 侧 `CreateCcuInsWithEnabledDies` 返回） |
| `streamCcu` | CCU stream（与 AIC stream 并行） |
| `kernelArg` | `CcuKernelArg` 结构体指针（channel、varHandle、eventHandle、roundId 等） |

---

## 4. CCU ↔ AIC 同步机制：CKE Event

CCU 和 AIC 在不同 stream 上并行执行，需要跨流同步。核心机制是 **CKE Event**——CCU 的 Event 资源有一块 GM 后背地址，AIC 可以直接读写。

> **对应关系**：CCU kernel 内 `events[i]` 和 AIC 内 `eventVa[i]` 指向同一个 Event 资源——CCU 侧通过 `ccu::Array<ccu::Event>(eventHandle, eventNum)` 构造，AIC 侧通过 `HcommCcuEventGetAddr` 获取同一资源的 GM 后背地址。CCU 用 `EventRecord`/`EventWait` 操作，AIC 用 `WriteGmByPassDCache`/`ReadGmByPassDCache` 操作同一块 GM 内存。

### 4.1 Event 分配

```
commTurn + 1 个 Event：
  eventVa[0]      = wake   （AIC → CCU，唤醒通信启动）
  eventVa[1]      = done 0 （CCU → AIC，round 0 通信完成）
  eventVa[2]      = done 1 （CCU → AIC，round 1 通信完成）
  ...
  eventVa[commTurn] = done (commTurn-1)
```

### 4.2 同步流程

```
AIC (streamAivAic)                        CCU (streamCcu)
─────────────────                         ──────────────
block 0 预写参数桥
  WriteGmByPassDCache → startVa[d][0..9]
写 wake
  eventVa[0] = 1   ──────────────────→    EventWait(events[0])  ← round 0 启动
                                          AllGather round 0
                                          EventRecord(events[1]) ──┐
                                                                      │
所有核轮询 done 0                         ←──────────────────────────┘
  ReadGmByPassDCache(eventVa[1]) == 1?
  YES → Matmul tile 0                     AllGather round 1（与 tile0 计算并行）
                                          EventRecord(events[2]) ──┐
                                                                      │
所有核轮询 done 1                         ←──────────────────────────┘
  ReadGmByPassDCache(eventVa[2]) == 1?
  YES → Matmul tile 1
```

**关键设计**：
- CCU kernel 内部串行循环，round d 完成后自动进入 round d+1，**不需要 AIC 再唤醒**（仅 round 0 需 wake）
- 每轮 done 用独立的 Event 地址，AIC 不会读到上一轮的残留值，**无需重置**
- AIC 用 `ReadGmByPassDCache` 直接读 GM 后背地址，不经过 CCU，**不占用 CCU 资源**

### 4.3 Variable 参数桥

CCU kernel 每轮需要读取 10 个参数（sendBuf 地址、recvBuf 地址、token、inputOffset、outputOffset、sliceSize、goSize×4）。这些参数由 AIC block 0 通过 `WriteGmByPassDCache` 写入 Variable 的 GM 后背地址，CCU 通过 `vars[base + i]` 读取。

```
Variable 布局（(commTurn+1) × 10 个，1 个 varHandle 管理）：
  组 0 (scale round): vars[0..9]      → startVa[0][0..9]    // 全量 scale 通信参数
  组 1 (data round 0): vars[10..19]   → startVa[1][0..9]    // tile0 data 通信参数
  组 2 (data round 1): vars[20..29]   → startVa[2][0..9]    // tile1 data 通信参数
  ...
```

> **两阶段通信**：组 0 是 scale AllGather（一次性全量 M 行 scale），组 1..commTurn 是 data AllGather（每轮 tileM 行 FP8 data）。scale 和 data 分别获取 token 和 goSize。

AIC block 0 **一次性预写所有 round 的参数**（在 CCU 启动前全部写好），循环中不再写 Variable——这样所有 AIC 核（含 block 0）预写完成后立即进入轮询 + matmul 循环，全力参与 Blaze 多核切分计算，不会因为每轮写参数而打断计算流水。

### 4.4 AllGather 前后数据排布（基线）

本节记录基线代码中 AllGather 通信前后各 buffer 的数据布局，是理解通信参数和 matmul 偏移计算的基础。

#### 符号约定

| 符号 | 含义 | 示例值（基线） |
|---|---|---|
| `rankSize` | rank 数（卡数） | 2 |
| `M` | 每 rank 矩阵 A 的行数 | 512 |
| `tileM` | 每 round 通信的行数 | 256 |
| `commTurn` | data 通信轮数 = M / tileM | 2 |
| `K` | 矩阵 A 的列数 / B 的行数 | 256 |
| `N` | 矩阵 B 的列数 / C 的列数 | 256 |
| `scaleKLen` | 每行 scale 元素数 = ceil(K/64) × 2 | 8 |
| `dataSliceSize` | 每 round 每 rank data 通信字节 = tileM × K × 1(FP8) | 65536 |
| `scaleSliceSize` | 全量 scale 通信字节 = M × scaleKLen × 1(E8M0) | 4096 |

#### sendBuf（AllGather 前，每 rank 独立）

**data（FP8 E4M3FN，1 byte/element）**：M 行 × K 列，按 round 切分：

```
sendBuf (M=512, K=256, FP8 1B)
├── round 0: rows [0, tileM)       = rows [0, 256)     inputOffset[0] = 0
├── round 1: rows [tileM, 2*tileM) = rows [256, 512)   inputOffset[1] = 65536
└── round r: rows [r*tileM, (r+1)*tileM)               inputOffset[r] = r * dataSliceSize

每 round 通信量 = dataSliceSize = tileM * K * 1B = 256 * 256 * 1 = 65536 B
```

**scaleA（E8M0，1 byte/element）**：M 行 × scaleKLen 列，全量通信（不分 round）：

```
sendScaleBuf (M=512, scaleKLen=8, E8M0 1B)
├── 全量: rows [0, M) = rows [0, 512)                   inputOffset = 0

通信量 = scaleSliceSize = M * scaleKLen * 1B = 512 * 8 * 1 = 4096 B
```

#### recvBuf（AllGather 后，所有 rank 汇聚）

**data（FP8 E4M3FN）**：rankSize × M 行 × K 列，各 rank 数据按 rankId 拼接，每 round 汇聚一组：

```
recvBuf (rankSize=2, M=512, K=256, FP8 1B)
├── round 0 汇聚区: [rank0 rows 0-255][rank1 rows 0-255]        outputOffset[0] = 0
├── round 1 汇聚区: [rank0 rows 256-511][rank1 rows 256-511]    outputOffset[1] = 131072
└── round r 汇聚区: [rank0 rows r*tileM..(r+1)*tileM]
                   [rank1 rows r*tileM..(r+1)*tileM]
                   ...
                   [rankR rows r*tileM..(r+1)*tileM]

outputOffset[r] = r * rankSize * dataSliceSize + rankId * dataSliceSize

每 round 汇聚后总行数 = rankSize * tileM（所有 rank 的 tileM 行拼接）
每 round 的 A 字节数 = tileABytes = rankSize * tileM * K * 1B
```

**scaleA（E8M0）**：rankSize × M 行 × scaleKLen 列，全量汇聚（1 轮完成）：

```
recvScaleBuf (rankSize=2, M=512, scaleKLen=8, E8M0 1B)
├── [rank0 scale rows 0-511][rank1 scale rows 0-511]

outputOffset = rankId * scaleSliceSize
```

#### outputBuf（matmul 输出，BF16，2 bytes/element）

每 round 的 matmul 输出 rankSize × tileM 行 × N 列：

```
outputBuf (rankSize=2, M=512, N=256, BF16 2B)
├── round 0: C[0 : 512*256*2B]        = C[0 : 262144 B]       cOffset = 0
├── round 1: C[262144 : 524288 B]                             cOffset = 262144
└── round r: C[r * tileCBytes : (r+1) * tileCBytes]

tileCBytes = rankSize * tileM * N * 2B
```

#### matmul 读取地址（round d）

```
A      = recvBuf      + d * tileABytes      (= d * rankSize * tileM * K * 1)
scaleA = recvScaleBuf + d * tileScaleABytes (= d * rankSize * tileM * scaleKLen * 1)
B      = weightBuf                          （本 rank 权重，不通信）
scaleB = weightScaleBuf                     （本 rank 权重 scale，不通信）
C      = outputBuf    + d * tileCBytes      (= d * rankSize * tileM * N * 2)
```

#### 完整示例（基线：2 rank, M=512, tileM=256, K=256, N=256, commTurn=2）

```
sendBuf (rank 0, 131072 B):
  round 0: rows 0-255   (offset 0)
  round 1: rows 256-511 (offset 65536)

sendScaleBuf (rank 0, 4096 B):
  全量: rows 0-511 (offset 0)

recvBuf (rank 0, 262144 B):
  round 0: [r0 rows 0-255][r1 rows 0-255]      (offset 0, 512 行)
  round 1: [r0 rows 256-511][r1 rows 256-511]  (offset 131072, 512 行)

recvScaleBuf (rank 0, 8192 B):
  [r0 scale rows 0-511][r1 scale rows 0-511]
  round 0 取前 512 行 scale, round 1 取后 512 行 scale

outputBuf (rank 0, 524288 B):
  round 0: C[0 : 262144 B]       (512 行 × 256 × 2B)
  round 1: C[262144 : 524288 B]

matmul round 0: A=recvBuf[0:131072], scaleA=recvScaleBuf[0:4096], C=outputBuf[0:262144]
matmul round 1: A=recvBuf[131072:262144], scaleA=recvScaleBuf[4096:8192], C=outputBuf[262144:524288]
```

> **关键设计点**：
> - AllGather 后各 rank 数据按 rankId 顺序拼接（rank 0 在前，rank 1 在后），不是交织排列
> - scale 全量 1 轮通信，但 matmul 按 round 切分读取（每 round 取 rankSize × tileM 行 scale，与 data 的 tile 行数对齐）
> - B / scaleB 是本 rank 权重，不参与通信，所有 round 共用同一地址

---

## 5. 流水编排

### 时序伪代码（commTurn=2 为例，MXFP8 两阶段通信）

```
streamAivAic (AIC)                        streamCcu (CCU)
─────────────────                       ─────────────────

// block 0: 一次性预写 + 唤醒
write startVa[0][0..9]                   // 预写 scale round 参数桥（组0）
for d in [0, 1]:
    write startVa[d+1][0..9]             // 预写 data round 参数桥（组1..commTurn）
write eventVa[0] = 1   ──────────────→   // wake 信号

                                        EventWait(events[0])          // scale round: 等 wake
                                        AllGather scale（全量）
                                        EventRecord(events[1]) ──→   // scale done

poll eventVa[1] == 1  ←─────────────────   // 等 scale done（所有 scale 数据就绪）

                                        // data round 0: 自动进入（无需再唤醒）
                                        AllGather data round 0
                                        EventRecord(events[2]) ──→

// 所有核循环 commTurn 次
// round 0:
poll eventVa[2] == 1  ←─────────────────
MXFP8 Matmul tile 0                     // data round 1: 自动进入
                                        AllGather data round 1
                                        EventRecord(events[3]) ──→   // ← tile0 计算与 round1 通信并行 overlap

// round 1:
poll eventVa[3] == 1  ←─────────────────
MXFP8 Matmul tile 1
```

**关键时序约束**：
1. AIC block 0 先预写所有 round 参数桥（scale 组 + data 组）+ 写 wake，CCU 才能启动
2. CCU 仅 scale round 前 `EventWait(events[0])` 等 wake，之后串行自动进入下一 round
3. CCU 先完成 scale AllGather（全量），再串行执行 commTurn 轮 data AllGather
4. AIC 先轮询 scale done（eventVa[1]），再循环 commTurn 轮：轮询 data done（eventVa[d+2]）→ matmul
5. round d 的 matmul 与 round d+1 的 data 通信并行 overlap（最后一轮 matmul 无通信可 overlap）

---

## 6. 关键代码

### 6.1 Host 侧资源编排（`host_launcher.h`）

```cpp
HcclResult PrepareAllGatherMatmul(...)
{
    // 1. 通信域
    HcclGetRankId(comm, &ctx.rankId);
    HcclGetRankSize(comm, &ctx.rankSize);

    // 2. CCU 通道与实例
    AcquireCcuChannels(comm, rankId, rankSize, kernelChannels);    // 每 peer rank 一个 channel
    CreateCcuInsWithEnabledDies(insHandle);                         // 创建 CCU 实例
    HcclCommAssignCcuIns(comm, insHandle);                          // 绑定到通信域

    // 3. Variable：(commTurn+1) × 10 个（组0=scale, 组1..commTurn=data）
    HcommCcuVariableAlloc(insHandle, DIE_ID, (commTurn + 1) * 10, &varHandle);
    for (d = 0; d < commTurn + 1; d++)
        for (i = 0; i < 10; i++)
            HcommCcuVariableGetAddr(varHandle, d * 10 + i, &mappedAddrs.startVa[d][i]);

    // 4. Event：commTurn + 2 个（1 wake + 1 scale done + commTurn data done）
    HcommCcuEventAlloc(insHandle, DIE_ID, commTurn + 2, &eventHandle);
    for (i = 0; i < commTurn + 2; i++)
        HcommCcuEventGetAddr(eventHandle, i, &mappedAddrs.eventVa[i]);

    // 5. token + goSize + offsets → device GM（scale 和 data 分别获取）
    HcommCcuGetMemToken(sendBuf, inputSize, &token);
    HcommCcuGetMemToken(sendScaleBuf, scaleInputSize, &scaleToken);
    // ...
}
```

### 6.2 `<<<>>>` launch（`.asc`）

```cpp
// Host 侧：1 次 CCU launch + 1 次 AIC launch，双流并行
CcuAllGatherMesh1DMem2MemKernel<<<ctx.schd, ctx.insHandle, streamCcu>>>(
    ctx.kernelArg.get());

all_gather_matmul_kernel<<<tilingData.usedCoreNum, 0, streamAivAic>>>(
    sendBuf, recvBuf, weightBuf, outputBuf, ctx.mappedAddrs, ...);
```

### 6.3 CCU kernel（`.asc`，`__ccu_host__`）

```cpp
__ccu_host__ void CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);

    // 初始化：从 channel 获取 peer 的 output/token Variable（通信目标地址）
    for (peerId = 0; peerId < rankSize; peerId++) {
        if (peerId != rankId) {
            output[peerId] = GetResByChannel<Variable>(channel, OUTPUT_XN_ID);
            token[peerId]  = GetResByChannel<Variable>(channel, TOKEN_XN_ID);
        }
    }

    ccu::Array<ccu::Event> events(kernelArg->eventHandle, kernelArg->eventNum);
    ccu::Array<ccu::Variable> vars(kernelArg->varHandle, kernelArg->varNum);

    // 两阶段循环：d=0 是 scale round，d=1..commTurn 是 data round
    uint32_t totalRounds = kernelArg->commTurn + 1;
    for (uint32_t d = 0; d < totalRounds; d++) {
        if (d == 0) {
            ccu::EventWait(events[0]);  // 仅 scale round 前 等 AIC 唤醒
        }

        // 读 round d 的 10 个 Variable（组0=scale参数, 组1..commTurn=data参数）
        uint32_t base = d * HCOMM_HOST_TASK_ARG_NUM;
        ctx.input                       = vars[base + 0];  // sendBuf/sendScaleBuf 地址
        ctx.output[rankId]              = vars[base + 1];  // recvBuf/recvScaleBuf 地址
        ctx.token[rankId]               = vars[base + 2];  // token
        ctx.currentRankSliceInputOffset = vars[base + 3];  // inputOffset
        ctx.currentRankSliceOutputOffset= vars[base + 4];  // outputOffset
        ctx.sliceSize                   = vars[base + 5];  // sliceSize
        ctx.goSize.addrOffset           = vars[base + 6];  // goSize[0..3]
        ctx.goSize.loopParam            = vars[base + 7];
        ctx.goSize.parallelParam        = vars[base + 8];
        ctx.goSize.residual             = vars[base + 9];

        // 参数桥就绪通知 + 等待 peer
        for (i = 0; i < channelCount; i++) {
            ccu::WriteVariableWithNotify(channels[i], output[rankId], OUTPUT_XN_ID, ...);
            ccu::WriteVariableWithNotify(channels[i], token[rankId],  TOKEN_XN_ID,  ...);
            ccu::NotifyWait(channels[i], allBit);  // 等 peer 的 Variable 就绪
        }

        // 执行 AllGather 传输
        ExecuteAllGatherTransfer(ctx);     // ccu::Write（远程）+ GroupCopy（本地）

        // 传输完成确认
        for (i = 0; i < channelCount; i++) {
            ccu::NotifyRecord(channels[i], POST_SYNC_ID);
        }
        for (i = 0; i < channelCount; i++) {
            ccu::NotifyWait(channels[i], POST_SYNC_ID);
        }

        // 通知 AIC：round d done（d=0→events[1]=scale done, d=1→events[2]=data round0 done, ...）
        ccu::EventRecord(events[d + 1]);
    }
}
```

#### `ExecuteAllGatherTransfer` 内部

```cpp
static CcuResult ExecuteAllGatherTransfer(ctx)
{
    // 远程传输：向每个 peer rank 发送数据
    for (rankIdx = 0; rankIdx < rankSize; rankIdx++) {
        if (rankIdx != rankId) {
            ccu::Write(channel, dst[rankIdx], src, sliceSize, event, mask);
        }
    }

    // 本地拷贝：sendBuf → recvBuf 本 rank 位置
    GroupCopy(ctx, localDst, src, goSize);

    // 等所有 rank 传输完成
    ccu::EventRecord(ctx.event, 1 << rankId);   // 本 rank 完成
    ccu::EventWait(ctx.event, totalMask);       // 等所有 rank
}
```

> **注意**：`GroupCopy` 内部的 Loop Entity lambda 按值捕获地址参数，循环中每轮必须重建，否则 round 1 复用 round 0 地址导致本地拷贝写错位置。

### 6.4 AIC 侧同步代码（`matmul_fusion_kernel.h`）

```cpp
__aicore__ inline void AicDirectPollProcess(...)
{
    // block 0 一次性预写 (commTurn+1) 组参数桥 + 写 wake
    if (GetBlockIdx() == 0) {
        // 组0: scale round 参数桥（sendScaleBuf, recvScaleBuf, scaleToken, ...）
        WriteGmByPassDCache(startVa[0][0..9], scaleArgs);
        // 组1..commTurn: data round 参数桥（sendBuf, recvBuf, token, ...）
        for (d = 0; d < commTurn; d++) {
            WriteGmByPassDCache(startVa[d+1][0..9], dataArgs);
        }
        WriteGmByPassDCache(eventVa[0], 1);                   // 写 wake，唤醒 CCU
    }

    // step 1: 所有核轮询 scale done（eventVa[1]）
    while (ReadGmByPassDCache(eventVa[1]) != 1) {
        Nop<800>();
    }

    // step 2: 所有核循环 commTurn 次：轮询 data done → MXFP8 matmul
    for (d = 0; d < commTurn; d++) {
        while (ReadGmByPassDCache(eventVa[d+2]) != 1) {       // data round d done
            Nop<800>();  // 降频轮询
        }
        // MXFP8 matmul：A=recvBuf+偏移, scaleA=recvScaleBuf+偏移, B=weightBuf, scaleB=weightScaleBuf
        MatmulProcess(recvBuf + d*tileABytes, recvScaleBuf + d*tileScaleABytes,
                      weightBuf, weightScaleBuf,
                      outputBuf + d*tileCBytes, tilingData);
    }
}
```

---

## 7. 验证结果

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段一 | CCU AllGather 通信验证 | ✅ PASS |
| 阶段二 | AllGather + Matmul 串联 | ✅ PASS |
| 阶段三 | 通算并行流水（commTurn round overlap） | ✅ PASS |
| 阶段四 | MXFP8 matmul + 纯 AIC + scale/data 两阶段通信 | ✅ PASS |

**精度**：tile0 全 256.0，tile1 全 512.0（2 卡，M=512, tileM=256, commTurn=2, K=N=256, FP8 E4M3FN + E8M0 scale → BF16）

> **推导**：输入 A 前 256 行 FP8=1.0(0x38)、后 256 行 FP8=2.0(0x40)，B 全 1.0，scaleA/scaleB 全 E8M0=127(2^0=1.0)，K=256。C = dequant(A)×dequant(B) = 256×1.0=256.0（tile0）、256×2.0=512.0（tile1）。两 rank 数据相同，AllGather 后结果一致。

---

## 8. 后续展望

- **阶段五**：验证是否可测试性能？和性能测试及优化
