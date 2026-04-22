# MoeDistributeCombineA2Layered::Process() 核心逻辑深度解析

> 源文件: `moe_distribute_combine_a2_layered.h`
> 函数: `MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Process()`
> 运行核心: **AIV** (向量计算核心)
> 示例参数: **worldSize=64, serverNum=8, aivNum=40**

---

## 一、全局背景：MoE Combine 阶段的定位

在 MoE（Mixture of Experts）分布式推理中，**Combine 阶段** 负责将各专家卡计算完成的结果进行加权聚合，输出最终的 Token 表示。本算子实现了 A2（AlltoAll v2）层级通信的 Combine 逻辑。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MoE 推理整体流水线                                        │
│                                                                             │
│   [Dispatch阶段] ──► [Expert计算] ──► [Combine阶段] ◄── 本算子位置            │
│        │                  │                   │                              │
│   Token分发到        各Expert并行          Token结果                         │
│   各专家卡           计算FFN               加权聚合                           │
│                                                                             │
│   通信类型:                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│   │ 机内 IPC 通信    │    │ 机间 RDMA 通信   │    │ 输出到GM        │         │
│   │ (同一服务器8卡)  │    │ (跨服务器通信)   │    │ (最终结果)      │         │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、关键参数约定

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `worldSize_` | EP总卡数（全局） | 64 (8台服务器 × 8卡) |
| `serverNum_` | 服务器数量 | 8 |
| `SERVER_RANK_SIZE` | 每服务器卡数 | 8 |
| `aivNum_` | AIV核心数量 | 40 |
| `axisBS_` | 本地Batch Size（Token数） | 256 |
| `axisH_` | Hidden Size（隐层维度） | 4096 |
| `axisK_` | TopK（每个Token选K个Expert） | 8 |
| `localMoeExpertNum_` | 本卡Expert数量 | 8 |
| `moeExpertNum_` | 总Expert数量 | 512 |
| `maxBs_` | 单卡最大Token数 | globalBs / worldSize |
| `IPC_REDUCE_USED_CORE_NUM` | IPC Reduce核数 | 32 |

### 服务器拓扑示意 (worldSize=64, serverNum=8)

```
  Server 0           Server 1           Server 2           ...    Server 7
┌────────────┐   ┌────────────┐   ┌────────────┐              ┌────────────┐
│ Rank 0-7   │   │ Rank 8-15  │   │ Rank 16-23 │    ...       │ Rank 56-63 │
│ (8张NPU卡) │   │ (8张NPU卡) │   │ (8张NPU卡) │              │ (8张NPU卡) │
│            │   │            │   │            │              │            │
│ 机内IPC    │   │ 机内IPC    │   │ 机内IPC    │              │ 机内IPC    │
│ 机间RDMA   │   │ 机间RDMA   │   │ 机间RDMA   │              │ 机间RDMA   │
└────────────┘   └────────────┘   └────────────┘              └────────────┘
        │                │                │                          │
        └────────────────┴────────────────┴──────────────────────────┘
                              RDMA 互联网络
```

---

## 三、Process() 整体流程

### 3.1 核心分工策略

```
总核数 = aivNum_ (例如 40)
├── [0, 32): IPC_REDUCE_USED_CORE_NUM=32核 → 执行 SumToWindow()
├── [32, 40): serverNum_=8核 → 执行 AlltoAllServerDispatch() (每Server一个核)
└── [40, aivNum_): 其余核 → 仅执行 SyncAll() 等待同步

核分配示意:
┌────────────────────────────────────────────────────────────────────────┐
│ Core 0-31   │ Core 32    │ Core 33    │ ... │ Core 39    │ Core 40+  │
│ SumToWindow │ Server 0   │ Server 1   │     │ Server 7   │ SyncAll   │
│ (机内聚合)   │ RDMA发送   │ RDMA发送   │     │ RDMA发送   │ (等待)    │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 四阶段流水线

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Process() 四阶段流水线                              │
│                                                                             │
│   阶段1: GM2IPC + WaitIPC                                                   │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 将本地Token数据复制到IPC共享内存，等待同步                    │    │
│   │ 通信: 机内IPC（同服务器8卡之间）                                    │    │
│   │ 输入: expandXGlobal_, expandScalesGlobal_                         │    │
│   │ 输出: IPC共享内存中的Token数据                                      │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│   阶段2: SumToWindow                                                        │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 对机内8卡的Token进行加权聚合，写入WindowOut                   │    │
│   │ 通信: 机内IPC读取 + 本地计算                                       │    │
│   │ 输入: IPC共享内存                                                  │    │
│   │ 输出: localOutWindow_ (RDMA发送缓冲区)                             │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│   阶段3: AlltoAllServerDispatch                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 通过RDMA将聚合后的数据发送到目标服务器                        │    │
│   │ 通信: 机间RDMA（跨服务器通信）                                      │    │
│   │ 输入: localOutWindow_                                              │    │
│   │ 输出: 远端 localInWindow_                                          │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│   阶段4: WaitDispatch + SumToServer                                         │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 功能: 等待远端数据到达，完成最终聚合输出                            │    │
│   │ 通信: 机内IPC + 输出到GM                                           │    │
│   │ 输入: localInWindow_ (来自各Server的RDMA数据)                      │    │
│   │ 输出: expandOutGlobal_ (最终结果)                                  │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、逐阶段详细分析

### 阶段 1: GM2IPC() - 数据搬运到IPC共享区

#### 功能描述
将本卡的Token数据（`expandXGlobal_`）和对应的权重（`expandScalesGlobal_`）按照专家分组，搬运到IPC共享内存区域。

#### 数据流向

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ expandXGlobal_  │      │    UB (inUb)    │      │ dstshareMemGlobal_│
│ (GM 输入数据)    │ ───► │  拼接data+scale │ ───► │ (IPC共享内存)    │
│ [tokenNum × H]  │      │ [H + 16 元素]   │      │ [tokenNum × (H+16)]│
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │
        │ DataCopy               │ DataCopy               │ 写入IPC区
        ▼                        ▼                        ▼
```

#### Token数据格式

```
单个Token在IPC共享区的存储格式:
┌────────────────────────────────────────┬────────────────────────┐
│            data[H]                     │      scale[16]         │
│       H × sizeof(ExpandXType)          │   32B (16个fp16/bf16)  │
│       Token特征数据                     │   权重值+对齐填充       │
└────────────────────────────────────────┴────────────────────────┘
       ←───────────── axisH_ ─────────────→←── WEIGHT_VALUE_NUM ──→

总大小 = (axisH_ + WEIGHT_VALUE_NUM) × sizeof(ExpandXType)
```

#### 详细流程

```
GM2IPC() 执行流程:
│
├── 1. 申请UB缓冲区
│   ├── sendCountLocal: 存储发送计数
│   ├── expandScalesLocal: 存储权重值
│   └── inUb: 拼接 data + scale 的临时缓冲区
│
├── 2. 加载sendCount到UB (DataCopy)
│
├── 3. 双层循环遍历
│   │
│   │  for expertId = 0 to localMoeExpertNum_-1:
│   │      for dstRankId = startRankId to endRankId_-1:
│   │          │
│   │          ├── 计算preCount和tokenNum
│   │          ├── 加载expandScales到UB
│   │          │
│   │          └── for tokenId = 0 to tokenNum-1:
│   │              ├── 读取scaleVal
│   │              ├── 拼接: inUb = [data[H], scaleVal]
│   │              └── 写入IPC共享区: dstshareMemGlobal_
│
├── 4. SyncAll() - 等待所有核完成
│
├── 5. 写入同步Flag (8个核并行)
│   └── 前8个核分别向远端写入 GM2IPC_SYNC_FLAG + magicValue_
│
└── 6. SyncAll() - 确保Flag可见
```

#### 同步Flag机制

```
同步Flag布局 (每Server 8个Flag位置):

Server 0的IPC Flag区:
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│Flag[0] │Flag[1] │Flag[2] │Flag[3] │Flag[4] │Flag[5] │Flag[6] │Flag[7] │
│Rank 0  │Rank 1  │Rank 2  │Rank 3  │Rank 4  │Rank 5  │Rank 6  │Rank 7  │
│写入    │写入    │写入    │写入    │写入    │写入    │写入    │写入    │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Flag值 = GM2IPC_SYNC_FLAG(12345) + magicValue_
magicValue_: 动态生成的唯一标识，防止读到过期数据
```

---

### 阶段 1b: WaitIPC() - 等待IPC数据就绪

#### 功能描述
等待来自同服务器其他7张卡的IPC数据就绪。

#### 详细流程

```
WaitIPC() 执行流程:
│
├── 1. 核选择: 仅 coreIdx < 8 的核参与
│
├── 2. 计算srcRankId
│   └── srcRankId = (rankId / 8) × 8 + coreIdx
│
├── 3. 轮询等待Flag
│   │
│   └── while (true):
│       ├── DataCopy: 读取Flag到UB
│       ├── if Flag >= (GM2IPC_SYNC_FLAG + magicValue_):
│       │       break  ← 数据就绪
│       └── 继续轮询
│
├── 4. 清除Flag (写入0)
│
├── 5. 记录性能信息 (可选)
│
└── 6. SyncAll() - 确保所有核看到数据
```

---

### 阶段 2: SumToWindow() - 机内聚合写入Window

#### 功能描述
将机内8卡的Token数据按专家路由信息进行加权聚合，结果写入`localOutWindow_`（RDMA发送缓冲区）。

#### 核心算法

```
加权聚合公式:
output[token] = Σ scale[i] × data[i]   (i ∈ 该Token的所有源卡)

展开:
┌─────────────────────────────────────────────────────────────┐
│  Token聚合示例 (假设Token来自Rank 0,2,5)                     │
│                                                             │
│  output = scale[0] × data[0] + scale[2] × data[2]          │
│         + scale[5] × data[5]                               │
│                                                             │
│  实现方式:                                                   │
│  - 第一个源: Muls(sumFloat, data[0], scale[0], H)           │
│  - 后续源:   Axpy(sumFloat, data[i], scale[i], H)           │
│    (Axpy = y = a × x + y, 累加形式)                         │
└─────────────────────────────────────────────────────────────┘
```

#### 详细流程

```
SumToWindow() 执行流程:
│
├── 1. 确定目标Server
│   ├── targetServerId = coreIdx / coreNumPerServer
│   └── targetRankId = rankId % 8 + targetServerId × 8
│
├── 2. 加载偏移表到UB
│   ├── countInnerGlobal_: 每个Token的源数量
│   └── offsetInnerGlobal_: 每个Token的源偏移索引
│
├── 3. 筛选有效Token
│   └── 只处理 copyNum > 0 的Token
│
├── 4. 核间负载均衡
│   └── 每个核处理 (i % coreNumPerServer) == (coreIdx % coreNumPerServer) 的Token
│
├── 5. 聚合循环
│   │
│   │  for each token:
│   │      │
│   │      ├── 初始化 sumFloatLocal = 0
│   │      │
│   │      └── for each source (copyNum次):
│   │          ├── 计算IPC源地址: shareMemGlobal_
│   │          ├── DataCopy: 读取 data + scale
│   │          ├── Cast: data → fp32
│   │          ├── 加权累加:
│   │          │   - 第一个: Muls(sumFloat, data, scale, H)
│   │          │   - 后续:   Axpy(sumFloat, data, scale, H)
│   │          └── offsetIndex++
│   │
│   │      └── 量化/非量化处理 (见下文)
│
├── 6. 写入结果 + Flag
│   ├── DataCopy: 写入 localOutWindow_
│   └── DataCopyPad: 写入 Token Flag (RDMA_TOKEN_ARRIVED_FLAG)
│
└── 7. 写入结束Flag (RDMA_TOKEN_END_FLAG)
```

#### 量化处理详解

##### FP16 动态量化流程

```
FP16 量化流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  sumFloatLocal (fp32)                                                       │
│        │                                                                    │
│        ▼ Cast (CAST_RINT)                                                   │
│  sumHalfLocal (fp16)                                                        │
│        │                                                                    │
│        ▼ Abs                                                                │
│  absScaleTensor (绝对值)                                                     │
│        │                                                                    │
│        ▼ BlockReduceMax (g16, 每16元素取最大)                                │
│  reduceMaxOutTensor [scaleNum个scale值]                                     │
│        │                                                                    │
│        ▼ Muls (× 1/127)                                                     │
│  scaleData [scale = max/127]                                                │
│        │                                                                    │
│        ▼ Brcb (广播到H维)                                                    │
│  absScaleTensor [H个scale值]                                                │
│        │                                                                    │
│        ▼ Div                                                                │
│  sumHalfLocal (量化后的值)                                                   │
│        │                                                                    │
│        ▼ Cast (CAST_RINT)                                                   │
│  castDataInt8 (int8)                                                        │
│        │                                                                    │
│        ▼ DataCopy                                                           │
│  localOutWindow_ [int8数据(H/2字节) + scale值]                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

量化后Token格式 (FP16):
┌──────────────────────────────────┬──────────────────────────┐
│      int8_data[H]                │    scale[scaleNum]       │
│      H × 1 byte                  │    scaleNum × 2 bytes    │
│      (H/2 个 int16)              │    (fp16)                │
└──────────────────────────────────┴──────────────────────────┘
  scaleNum = H / 16 (每16个元素共享1个scale)
```

##### BF16 动态量化流程

```
BF16 量化流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  sumFloatLocal (fp32)                                                       │
│        │                                                                    │
│        ▼ Abs                                                                │
│  castInFloatLocal (绝对值)                                                   │
│        │                                                                    │
│        ▼ BlockReduceMax (g8, 每8元素取最大)                                  │
│  reduceMaxTensorFloat [scaleNum个scale值]                                   │
│        │                                                                    │
│        ▼ Muls (× 1/127)                                                     │
│  reduceMaxTensorFloat [scale = max/127, fp32]                               │
│        │                                                                    │
│        ▼ Brcb (广播到H维)                                                    │
│  castInFloatLocal [H个scale值, fp32]                                        │
│        │                                                                    │
│        ▼ Div                                                                │
│  sumFloatLocal (量化后的值, fp32)                                            │
│        │                                                                    │
│        ├──────────────────────┬─────────────────────────┐                  │
│        ▼ Cast (fp32→bf16)     ▼ Cast (fp32→bf16)                            │
│  scaleData (bf16)        halfLocal (fp16中间)                               │
│        │                      │                                             │
│        │                      ▼ Cast (fp16→int8)                            │
│        │                castDataInt8 (int8)                                 │
│        │                      │                                             │
│        └──────────────────────┴─────────────────────────┘                  │
│                               │                                             │
│                               ▼ DataCopy                                    │
│  localOutWindow_ [int8数据(H/2字节) + bf16 scale值]                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

量化后Token格式 (BF16):
┌──────────────────────────────────┬──────────────────────────┐
│      int8_data[H]                │    scale[scaleNum]       │
│      H × 1 byte                  │    scaleNum × 2 bytes    │
│      (H/2 个 int16)              │    (bf16)                │
└──────────────────────────────────┴──────────────────────────┘
  scaleNum = H / 8 (每8个元素共享1个scale)
```

#### 非量化处理流程

```
非量化流程 (直接输出FP16/BF16):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  sumFloatLocal (fp32)                                                       │
│        │                                                                    │
│        ▼ Cast (CAST_RINT)                                                   │
│  dataInLocal (fp16/bf16)                                                    │
│        │                                                                    │
│        ▼ DataCopy                                                           │
│  localOutWindow_ [H × sizeof(ExpandXType)]                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 阶段 3: AlltoAllServerDispatch() - 跨服务器RDMA发送

#### 功能描述
通过RDMA将聚合后的Token数据发送到目标服务器。

#### 核分配策略

```
每个核负责一个目标Server:
┌────────────────────────────────────────────────────────────────────────────┐
│  Core 32  │  Core 33  │  Core 34  │  Core 35  │  Core 36  │  Core 37-39   │
│ Server 0  │  Server 1 │  Server 2 │  Server 3 │  Server 4 │  Server 5-7   │
│ RDMA发送  │  RDMA发送 │  RDMA发送 │  RDMA发送 │  RDMA发送 │  RDMA发送     │
└────────────────────────────────────────────────────────────────────────────┘

checkServer = coreIdx - stepCoreNum (32)
dstRankId = rankId % 8 + 8 × checkServer
```

#### 详细流程

```
AlltoAllServerDispatch() 执行流程:
│
├── 1. 初始化
│   ├── checkServer = coreIdx - 32
│   ├── dstRankId = rankId % 8 + 8 × checkServer
│   └── 计算copyLen_ (量化: H + scaleNum; 非量化: H)
│
├── 2. 主循环 (轮询Token Flag)
│   │
│   └── while (!stopFlag):
│       │
│       ├── 轮询检查 Token Flag
│       │   ├── if Flag == RDMA_TOKEN_ARRIVED_FLAG:
│       │   │       copySum++  ← Token就绪
│       │   ├── elif Flag == RDMA_TOKEN_END_FLAG:
│       │   │       stopFlag = true  ← 所有Token完成
│       │   └── 继续轮询
│       │
│       └── if (累积足够Token):
│           │
│           ├── if (dstRankId != rankId):  ← 跨卡发送
│           │       AIVRDMAPostSend(srcAddr, dstAddr, dstRankId, len, qp_info)
│           │
│           └── else:  ← 本地复制
│                   DataCopy(tmpLowUb_, srcGlobal)
│                   DataCopy(dstGlobal, tmpLowUb_)
│
├── 3. 发送完成Flag
│   └── AIVRDMAPostSend(readStateGlobal_, remoteFlagAddr, dstRankId, 32)
│
└── 4. SyncAll() - 确保所有核完成
```

#### RDMA发送机制 (AIVRDMAPostSend)

```
RDMA发送流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. 检查SQ (Send Queue) 空间                                                │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ while (curHead - tail >= QP_DEPTH - 1):                         │    │
│     │     轮询等待空间可用                                              │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. 构造WQE (Work Queue Entry)                                              │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Offset 0-3:   控制字                                            │    │
│     │               - opcode = 3 (RDMA_WRITE)                         │    │
│     │               - owner_bit (翻转位)                              │    │
│     │               - signaled = 1                                    │    │
│     │ Offset 4-7:   message_len (消息长度)                            │    │
│     │ Offset 8-11:  imm_data = 0                                      │    │
│     │ Offset 12-15: num_sge = 1                                       │    │
│     │ Offset 16-19: start_sge_idx = 0                                 │    │
│     │ Offset 20-23: R_Key (远端内存键)                                │    │
│     │ Offset 24-31: Remote VA (远端虚拟地址)                          │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  3. 构造SGE (Scatter Gather Element)                                        │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Offset 0-3:   data_len (数据长度)                               │    │
│     │ Offset 4-7:   L_Key (本地内存键)                                │    │
│     │ Offset 8-15:  Local VA (本地虚拟地址)                           │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  4. 刷新Cache                                                               │
│     cacheWriteThrough(wqeAddr, sizeof(WQE) + sizeof(SGE))                  │
│                                                                             │
│  5. 写DoorBell触发发送                                                       │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ doorBellInfo:                                                    │    │
│     │   [0:23]  wqn (QP number)                                       │    │
│     │   [24:27] cmd = 0 (SQ_DB)                                       │    │
│     │   [32:47] pi = curHead                                          │    │
│     │   [48:50] sl (Service Level)                                    │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  6. 更新Head指针                                                            │
│     DataCopyPad(HeadGlobalTensor, ubLocalHead)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 阶段 4a: Preload() - 预加载偏移表

#### 功能描述
预加载`offsetOuterGlobal_`和`countOuterGlobal_`到UB，为`SumToServer()`做准备。

#### 核分配

```
仅8个核 (coreIdx < 8) 参与:
┌────────────────────────────────────────────────────────────────────────────┐
│  Core 0    │  Core 1    │  Core 2    │  ...  │  Core 7    │  Core 8+      │
│ Token 0-n  │ Token n-m  │ Token m-k  │       │ Token x-y  │  跳过         │
│ processNum │ processNum │ processNum │       │ processNum │               │
└────────────────────────────────────────────────────────────────────────────┘

processNum = axisBS / 8
每个核处理 processNum 或 processNum+1 个Token
```

---

### 阶段 4b: WaitDispatch() - 等待RDMA数据到达

#### 功能描述
等待来自其他服务器的RDMA数据到达。

#### 详细流程

```
WaitDispatch() 执行流程:
│
├── 1. 核选择
│   └── 条件: coreIdx < serverNum && coreIdx != (rankId / 8)
│       (等待其他Server的数据，不等待自己)
│
├── 2. 计算srcRankId
│   └── srcRankId = rankId % 8 + coreIdx × 8
│
├── 3. 轮询等待
│   │
│   └── while (true):
│       ├── DataCopy: 读取状态Flag
│       ├── if Flag == sumTarget_:
│       │       break  ← 数据到达
│       └── 继续轮询
│
├── 4. 记录性能信息 (可选)
│
└── 5. SyncAll() - 确保所有核看到数据
```

---

### 阶段 4c: SumToServer() - 最终聚合输出

#### 功能描述
从各服务器接收的数据中，按路由表进行最终聚合，输出到`expandOutGlobal_`。

#### 详细流程

```
SumToServer() 执行流程:
│
├── 1. 核选择: 仅 coreIdx < 8 参与
│
├── 2. 计算处理范围
│   ├── startBs = coreIdx × processNum (+ resNum调整)
│   └── endBs = startBs + processNum
│
├── 3. 申请UB缓冲区
│   ├── sumFloatLocal: 累加缓冲区
│   ├── dataIn: 输入数据缓冲区
│   ├── castFp32: 类型转换缓冲区
│   └── 量化相关缓冲区 (如需要)
│
├── 4. 主循环 (遍历每个Token)
│   │
│   └── for i = startBs to endBs-1:
│       │
│       ├── 读取 copyNum = countReduceLocal_[i]
│       ├── 初始化 offsetIndex = i × serverNum_
│       │
│       ├── if (量化模式):
│       │       │
│       │       └── for j = 0 to copyNum-1:
│       │               ├── 计算源地址: localInWindow_[srcServer]
│       │               ├── 读取 int8数据 + scale
│       │               ├── 反量化: int8 → fp16/bf16 → fp32
│       │               ├── 乘scale: MulAddDst
│       │               └── offsetIndex++
│       │
│       ├── else (非量化模式):
│       │       │
│       │       └── for j = 0 to copyNum-1:
│       │               ├── 计算源地址: localInWindow_[srcServer]
│       │               ├── 读取 fp16/bf16 数据
│       │               ├── Cast → fp32
│       │               ├── 累加: Add
│       │               └── offsetIndex++
│       │
│       ├── Cast: sumFloat → ExpandXType (fp16/bf16)
│       └── DataCopy: 写入 expandOutGlobal_[i × H]
│
└── 5. SyncAll() - 确保所有核完成
```

#### 反量化流程 (INT8 → FP16/BF16)

##### FP16 反量化

```
FP16 反量化流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  localInWindow_ (int8数据 + fp16 scale)                                     │
│        │                                                                    │
│        ├─────────────────────────┬───────────────────────────┐             │
│        ▼ DataCopy                ▼ DataCopy                               │
│  dataInt8 [H个int8]         scaleData [scaleNum个fp16]                     │
│        │                         │                                         │
│        ▼ Cast (int8→fp16)        ▼ Brcb (广播到H维)                         │
│  castFp16 [H个fp16]          scaleDup [H个fp16]                            │
│        │                         │                                         │
│        └─────────────────────────┴───────────────────────────┐             │
│                                  ▼ MulAddDst                               │
│                           sumFp16Local [H个fp16]                            │
│                                  │                                         │
│                                  ▼ DataCopy                                │
│                           expandOutGlobal_                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### BF16 反量化

```
BF16 反量化流程:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  localInWindow_ (int8数据 + bf16 scale)                                     │
│        │                                                                    │
│        ├─────────────────────────┬───────────────────────────┐             │
│        ▼ DataCopy                ▼ DataCopy                               │
│  dataInUbInt8 [H个int8]     scaleDataBf16 [scaleNum个bf16]                 │
│        │                         │                                         │
│        ▼ Cast (int8→fp16)        ▼ Cast (bf16→fp32)                         │
│  castDataHalf [H个fp16]     castFp32scale [scaleNum个fp32]                 │
│        │                         │                                         │
│        ▼ Cast (fp16→fp32)        ▼ Brcb (广播到H维)                         │
│  castDataFp32 [H个fp32]     castFp32ScaleBrcb [H个fp32]                    │
│        │                         │                                         │
│        └─────────────────────────┴───────────────────────────┐             │
│                                  ▼ MulAddDst                               │
│                           sumFloatLocal1 [H个fp32]                          │
│                                  │                                         │
│                                  ▼ Cast (fp32→bf16)                        │
│                           sumBf16Local [H个bf16]                           │
│                                  │                                         │
│                                  ▼ DataCopy                                │
│                           expandOutGlobal_                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、数据结构

### 5.1 核心类成员变量

| 变量名 | 类型 | 用途 |
|--------|------|------|
| `expandXGlobal_` | `GlobalTensor<ExpandXType>` | 输入Token数据 |
| `sendCountGlobal_` | `GlobalTensor<ExpandIdxType>` | 发送计数 |
| `expandScalesGlobal_` | `GlobalTensor<float>` | Token权重 |
| `expandOutGlobal_` | `GlobalTensor<ExpandXType>` | 输出聚合结果 |
| `localOutWindow_` | `GlobalTensor<ExpandXType>` | 本地发送窗口（RDMA源） |
| `localInWindow_` | `GlobalTensor<ExpandXTransType>` | 本地接收窗口（RDMA目的） |
| `shareMemGlobal_` | `GlobalTensor<ExpandXType>` | IPC共享内存（读） |
| `dstshareMemGlobal_` | `GlobalTensor<ExpandXType>` | IPC共享内存（写） |
| `shareFlagGlobal_` | `GlobalTensor<uint64_t>` | IPC同步Flag |
| `readStateGlobal_` | `GlobalTensor<uint64_t>` | RDMA到达状态 |

### 5.2 UB缓冲区布局

```
TBuf Layout (TBUF_SIZE = 185KB):
┌────────────────────────────────────────────────────────────────────────────┐
│ Offset 0: TBUF_TEMP_OFFSET                                                 │
├────────────────────────────────────────────────────────────────────────────┤
│ sendCountLocal:     RoundUp(moeExpertNum_, 8) × sizeof(int32_t)            │
│ expandScalesLocal:  RoundUp(maxBs_, 32) × sizeof(float)                    │
│ inUb:               (axisH + 16) × sizeof(ExpandXType)                     │
│ countReduceLocal:   RoundUp(realBS, 16) × sizeof(int16_t)                  │
│ offsetReduceLocal:  RoundUp(realBS × K, 8) × sizeof(int32_t)               │
│ tmpTokenIdxLocal:   RoundUp(realBS, 8) × sizeof(uint32_t)                  │
│ offsetIndexs:       listLen × sizeof(uint32_t)                             │
│ copyNums:           listLen × sizeof(uint32_t)                             │
│ dataOffsets:        listLen × sizeof(uint32_t)                             │
│ dataInLocal:        (axisH + 16) × sizeof(ExpandXType)                     │
│ castInFloatLocal:   RoundUp(axisH × 4, 256) bytes                          │
│ sumFloatLocal:      axisH × sizeof(float)                                  │
│ ... (量化相关缓冲区)                                                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Token数据格式对比

| 模式 | 数据格式 | Token大小 | Scale粒度 |
|------|----------|-----------|-----------|
| 非量化 | FP16/BF16 | H × 2 bytes | N/A |
| FP16量化 | INT8 + FP16 scale | H/2 + scaleNum × 2 bytes | 16元素/scale |
| BF16量化 | INT8 + BF16 scale | H/2 + scaleNum × 2 bytes | 8元素/scale |

---

## 六、内存排布

### 6.1 HCCL WindowIn 布局

```
WindowIn (Ping半, Combine阶段):
┌────────────────────────────────────────────────────────────────────────────┐
│ Offset 0: RDMA Arrived Flag                                                │
│ ┌────────────────────────────────────────────────────────────────────────┐│
│ │ Server 0 Flag (512B) │ Server 1 Flag (512B) │ ... │ Server 7 Flag (512B)││
│ │ ←────── A1 = serverNum × 512B ──────────────────────────────────────→ ││
│ └────────────────────────────────────────────────────────────────────────┘│
├────────────────────────────────────────────────────────────────────────────┤
│ Offset A1: RDMA Inner Flag (A1 bytes)                                      │
├────────────────────────────────────────────────────────────────────────────┤
│ Offset 2×A1: RDMA Inner Data (1MB - 2×A1 bytes)                            │
├────────────────────────────────────────────────────────────────────────────┤
│ Offset 1MB: RDMA Token Data                                                │
│ ┌────────────────────────────────────────────────────────────────────────┐│
│ │ Server 0 Data         │ Server 1 Data         │ ... │ Server 7 Data    ││
│ │ RDMA_DATA_SIZE/8      │ RDMA_DATA_SIZE/8      │     │ RDMA_DATA_SIZE/8 ││
│ └────────────────────────────────────────────────────────────────────────┘│
├────────────────────────────────────────────────────────────────────────────┤
│ IPC Data区                                                                 │
│ ┌────────────────────────────────────────────────────────────────────────┐│
│ │ localExp 0 (srcRank 0~n/2-1) │ localExp x │ ... │ localExp n           ││
│ └────────────────────────────────────────────────────────────────────────┘│
├────────────────────────────────────────────────────────────────────────────┤
│ Offset W/2 - 2MB: IPC Flag区                                               │
│ ┌────────────────────────────────────────────────────────────────────────┐│
│ │ GM2IPC Sync Flag (256B) │ Token Flag │ Magic Value                     ││
│ └────────────────────────────────────────────────────────────────────────┘│
├────────────────────────────────────────────────────────────────────────────┤
│ Offset W/2: Pong半 (结构相同)                                               │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 HCCL WindowOut 布局

```
WindowOut (Combine阶段):
┌────────────────────────────────────────────────────────────────────────────┐
│ Offset 0: Ping RDMA Flag (32B)                                             │
├────────────────────────────────────────────────────────────────────────────┤
│ Offset 2×A1: Ping RDMA Inner Data                                          │
├────────────────────────────────────────────────────────────────────────────┤
│ Offset 1MB: Ping RDMA Token Data                                           │
│ ┌────────────────────────────────────────────────────────────────────────┐│
│ │ DstServer 0 Data       │ DstServer x Data │ ... │ DstServer 7 Data     ││
│ │ RDMA_DATA_SIZE/8       │                  │     │                       ││
│ └────────────────────────────────────────────────────────────────────────┘│
├────────────────────────────────────────────────────────────────────────────┤
│ Offset W/2: Pong半 (结构相同)                                               │
├────────────────────────────────────────────────────────────────────────────┤
│ Offset W - 32B: BufferId (32B) - Ping/Pong切换标识                          │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 IPC数据地址计算

```
IPC数据地址计算公式:
┌────────────────────────────────────────────────────────────────────────────┐
│ addr = ipcDataBase_                                                        │
│        + halfOffset (Ping=0 或 Pong=W/2)                                   │
│        + expertOffset (localExpertId × rankSizeOnIpcData_)                 │
│        + rankOffset ((srcIpcRank % halfWorldSize_) × rankSizeOnIpcData_    │
│                      / halfWorldSize_)                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 七、时序流水同步信息

### 7.1 整体时序图

```
时间轴 ════════════════════════════════════════════════════════════════════════►

Core 0-31 (SumToWindow):
├───────────────┼───────────────┼─────────────────────────────┼───────────────┤
│   GM2IPC      │   WaitIPC     │       SumToWindow           │   Preload     │
│  (数据搬运)    │  (等待同步)   │     (机内聚合+量化)          │  (预加载)      │
├───────────────┴───────────────┴─────────────────────────────┼───────────────┤
│                           SyncAll                            │  WaitDispatch │
├─────────────────────────────────────────────────────────────┴───────────────┤
│                          SumToServer (最终聚合输出)                          │
└─────────────────────────────────────────────────────────────────────────────┘

Core 32-39 (AlltoAllServerDispatch):
├───────────────┼───────────────┼─────────────────────────────────────────────┤
│   GM2IPC      │   WaitIPC     │         AlltoAllServerDispatch              │
│  (数据搬运)    │  (等待同步)   │            (RDMA发送)                        │
├───────────────┴───────────────┴─────────────────────────────────────────────┤
│                              SyncAll                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│   Preload    │  WaitDispatch │           SumToServer                        │
└─────────────────────────────────────────────────────────────────────────────┘

Core 40+ (其他核):
├───────────────┼───────────────┼─────────────────────────────────────────────┤
│   GM2IPC      │   WaitIPC     │              SyncAll                        │
│  (数据搬运)    │  (等待同步)   │              (等待)                          │
├───────────────┴───────────────┴─────────────────────────────────────────────┤
│   Preload    │  WaitDispatch │           SumToServer                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 同步点总结

| 同步点 | 类型 | 参与核 | 目的 |
|--------|------|--------|------|
| `SyncAll` after GM2IPC | 全核同步 | 所有核 | 确保数据搬运完成 |
| Flag write/read | 8核并行 | 0-7核 | IPC数据就绪通知 |
| `SyncAll` after WaitIPC | 全核同步 | 所有核 | 确保IPC数据可见 |
| Token Flag | 32核并行 | 0-31核 | SumToWindow完成通知 |
| `SyncAll` after AlltoAllServerDispatch | 全核同步 | 所有核 | 确保RDMA发送完成 |
| `SyncAll` after SumToServer | 全核同步 | 所有核 | 确保输出完成 |

### 7.3 Flag同步机制

```
Flag类型及用途:
┌────────────────────────────────────────────────────────────────────────────┐
│ Flag名称                    │ 值              │ 用途                       │
├────────────────────────────────────────────────────────────────────────────┤
│ GM2IPC_SYNC_FLAG            │ 12345           │ GM2IPC完成通知             │
│ RDMA_TOKEN_ARRIVED_FLAG     │ 123             │ Token数据就绪              │
│ RDMA_TOKEN_END_FLAG         │ 321             │ 所有Token处理完毕          │
│ magicValue_                 │ 动态生成         │ 防止读到过期数据           │
└────────────────────────────────────────────────────────────────────────────┘

Flag检查公式:
expectedValue = FLAG_BASE + magicValue_
```

---

## 八、开发注意事项

### 8.1 内存对齐要求

| 类型 | 对齐粒度 | 原因 |
|------|----------|------|
| UB缓冲区 | 32B | 硬件DMA要求 |
| RDMA数据 | 4KB | RDMA NIC页对齐 |
| IPC数据 | 512B | HCCL内部对齐 |
| Token Flag | 64B | Cache line对齐 |

### 8.2 同步屏障使用

```cpp
// 正确: 使用PipeBarrier确保流水线阶段间同步
PipeBarrier<PIPE_ALL>();

// 正确: 使用SyncFunc确保Buffer间数据一致性
SyncFunc<AscendC::HardEvent::MTE2_S>();   // MTE2 → S
SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // MTE3 → MTE2
SyncFunc<AscendC::HardEvent::V_MTE3>();   // V → MTE3

// 正确: 使用SyncAll确保多核同步
SyncAll<true>();  // true = 需要等待所有核
```

### 8.3 量化精度注意事项

1. **FP16量化**:
   - 每16个元素共享一个scale
   - `scale = max(|data|) / 127`
   - 量化误差: ~0.5%

2. **BF16量化**:
   - 每8个元素共享一个scale
   - 中间计算用FP32累加
   - 量化误差: ~0.3%

3. **累加顺序**:
```cpp
// 错误: 直接累加可能导致精度丢失
sum += scale * data;

// 正确: 使用高精度累加
Muls(sumFloatLocal, castInFloatLocal, scaleVal, axisH_);  // 第一个
Axpy(sumFloatLocal, castInFloatLocal, scaleVal, axisH_);  // 后续
```

### 8.4 性能优化建议

1. **减少同步次数**:
   - 合并相邻的`PipeBarrier<PIPE_ALL>()`
   - 使用`SyncFunc`替代`PipeBarrier`（更轻量）

2. **流水线重叠**:
   - `Preload()`与`AlltoAllServerDispatch()`并行执行
   - 数据搬运与计算重叠

3. **内存访问优化**:
   - 连续地址访问（利于DMA）
   - 避免跨cache line访问
   - 使用`cacheWriteThrough()`刷新关键数据

### 8.5 常见问题排查

| 问题 | 可能原因 | 排查方法 |
|------|----------|----------|
| 死锁 | Flag未正确设置 | 检查magicValue是否匹配 |
| 数据错误 | 量化溢出 | 检查scale值是否为0 |
| 性能下降 | 同步过多 | 分析SyncAll调用次数 |
| RDMA失败 | WQE格式错误 | 检查R_Key/L_Key是否正确 |

### 8.6 调试技巧

```cpp
// 开启性能信息收集
needPerformanceInfo_ = true;

// 检查关键变量
printf("coreIdx=%d, stepCoreNum=%d, serverNum=%d\n",
       coreIdx_, stepCoreNum, serverNum_);

// 验证Flag值
uint64_t flag = shareFlagGlobal_(0);
printf("Flag value: %lu, expected: %lu\n",
       flag, GM2IPC_SYNC_FLAG + magicValue_);
```

---

## 九、关键常量定义

```cpp
constexpr uint32_t UB_ALIGN = 32U;                    // UB对齐粒度
constexpr uint32_t BLOCK_SIZE = 32U;                  // 块大小
constexpr uint32_t B16_PER_BLOCK = 16U;               // 每32B块的int16数量
constexpr uint32_t B32_PER_BLOCK = 8U;                // 每32B块的int32数量
constexpr uint32_t SERVER_RANK_SIZE = 8U;             // 每服务器卡数
constexpr uint32_t VEC_LEN = 256U;                    // 向量长度
constexpr uint32_t TBUF_SIZE = 185U * 1024U;          // UB总大小
constexpr uint32_t IPC_REDUCE_USED_CORE_NUM = 32U;    // IPC Reduce核数
constexpr uint32_t WEIGHT_VALUE_NUM = 16U;            // Token尾部权重区大小
constexpr uint64_t GM2IPC_SYNC_FLAG = 12345ULL;       // GM2IPC同步标志
constexpr uint64_t RDMA_TOKEN_ARRIVED_FLAG = 123ULL;  // RDMA到达标志
constexpr uint64_t RDMA_TOKEN_END_FLAG = 321ULL;      // RDMA结束标志
```

---

## 十、数据流端到端示意图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Combine 阶段数据流                                    │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │   输入Token      │  expandXGlobal_ [axisBS × H]                          │
│  │   (各Expert结果) │  expandScalesGlobal_ [axisBS × K]                     │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   GM2IPC        │  按Expert分组，写入IPC共享区                            │
│  │   (数据搬运)     │  格式: [data(H) + scale(32B)]                          │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   IPC共享内存    │  同服务器8卡共享                                        │
│  │   (机内通信)     │  GetIpcDataAddr(srcRank, expertId, dstRank)            │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   SumToWindow   │  加权聚合: output = Σ scale × data                     │
│  │   (机内聚合)     │  可选量化: INT8 + scale                                 │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   WindowOut     │  RDMA发送缓冲区                                         │
│  │   (本地发送窗口) │  按Server划分                                           │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │ AlltoAllDispatch│ ───► │   RDMA网络      │                              │
│  │ (跨Server发送)   │      │   (机间通信)     │                              │
│  └────────┬────────┘      └────────┬────────┘                              │
│           │                        │                                        │
│           ▼                        ▼                                        │
│  ┌─────────────────┐      ┌─────────────────┐                              │
│  │   WindowIn      │ ◄─── │   远端Server    │                              │
│  │   (本地接收窗口) │      │   发送的数据    │                              │
│  └────────┬────────┘      └─────────────────┘                              │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   SumToServer   │  最终聚合: output = Σ data (来自各Server)              │
│  │   (输出聚合)     │  反量化 (如需要)                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │   最终输出       │  expandOutGlobal_ [axisBS × H]                        │
│  │   (加权结果)     │  FP16/BF16                                             │
│  └─────────────────┘                                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

*文档生成时间: 2026-03-14*
*基于源文件: moe_distribute_combine_a2_layered.h*
