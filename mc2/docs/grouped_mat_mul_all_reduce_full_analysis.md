# grouped_mat_mul_all_reduce 算子全面分析报告

## 1. 算子概述

`grouped_mat_mul_all_reduce` 实现 **GroupedMatMul + AllReduce** 融合，对多组独立矩阵乘法计算后执行跨卡 AllReduce 归约。采用 AIC-AICPU 协作架构：AIC 负责 MatMul 计算，AICPU 负责通信，两者通过 RAC Server 消息传递机制实现计算-通信重叠。

```
x[0] × weight[0] → partial_y[0] ─┐
x[1] × weight[1] → partial_y[1] ─┤ AllReduce → y[0..G-1]
...                               │
x[G-1] × weight[G-1] → partial_y[G-1] ─┘
```

**注意**: 此接口已标记为 `ACL_DEPRECATED_MESSAGE`。

**平台**: ascend910b, ascend910_93 | **HCCL组**: `"group"` | **数据类型**: BF16/FP16

### 1.1 架构特点

```
┌─────────────────────────────────────────────────────────┐
│ AIC Core                                                │
│ ├── GMMProcess: 逐组调度 GMM                            │
│ ├── GMMCompute:  单组 MM 计算 (MatmulImpl)               │
│ └── HcclServer:  TurnNotifyRun → 通知 AICPU 通信         │
│                                                          │
│ AICPU                                                    │
│ ├── 接收 AIC 的 Turn 通知                                │
│ └── 执行 HCCL AllReduce (与后续 AIC 计算重叠)            │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
grouped_mat_mul_all_reduce/
├── op_kernel/
│   ├── grouped_mat_mul_all_reduce.cpp          (3,664字节)  内核入口
│   ├── grouped_mat_mul_all_reduce.h            (7,679字节)  GMM 调度与计算
│   ├── grouped_mat_mul_all_reduce_utils.h      (3,938字节)  工具函数
│   └── rac_server.h                            (11,026字节) RAC 通信服务
├── op_host/
│   ├── grouped_mat_mul_all_reduce_def.cpp      (3,082字节)  算子定义
│   ├── grouped_mat_mul_all_reduce_infershape.cpp (11,836字节) Shape推导
│   └── op_tiling/
│       ├── grouped_mat_mul_all_reduce_tiling.cpp (32,152字节) Tiling 策略
│       └── grouped_mat_mul_all_reduce_tiling.h   (2,767字节)  Tiling 结构
├── op_api/
│   ├── aclnn_grouped_mat_mul_all_reduce.cpp    (24,013字节) ACLNN 实现
│   └── aclnn_grouped_mat_mul_all_reduce.h      (3,924字节)  ACLNN 声明
├── op_graph/
│   ├── grouped_mat_mul_all_reduce_proto.h      (3,455字节)  图模式注册
│   └── fallback_grouped_mat_mul_all_reduce.cpp (5,765字节)  回退路径
└── tests/ut/ (省略)
```

---

## 3. Host 侧分析

### 3.1 算子定义

源文件：`grouped_mat_mul_all_reduce_def.cpp` — 类定义 第19行，构造函数 第22-68行

**输入张量** (动态输入，第24-44行):

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0..N | x | FP16/BF16 | 各组输入矩阵 |
| 0..N | weight | FP16/BF16 | 各组权重矩阵 |
| 0..N | bias | FP16/FP32 (可选) | 各组偏置 |
| opt  | group_list | INT64 | 分组偏移列表 |

**输出张量** (动态输出，第45-49行):

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0..N | y | FP16/BF16 | 各组输出 |

**关键属性** (第50-53行):

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| splitItem | Int | 0 | 分组模式 (0/1/2/3) |
| group | String | - | HCCL 通信域 (必填) |
| reduceOp | String | "sum" | 归约操作 |
| commTurn | Int | 0 | 通信轮次 |

**平台**: `ascend910b`, `ascend910_93` — OP_ADD 注册 第71行

### 3.2 Tiling 策略

`grouped_mat_mul_all_reduce_tiling.cpp` (32,152字节)

#### splitItem 模式

```
splitItem=0: 无分组分割
  x = [x0, x1, ..., xG-1], weight = [w0, w1, ..., wG-1]
  y[i] = AllReduce(x[i] × weight[i])

splitItem=1: x 轴分割 (单个 x, 多个 weight)
  x = 单个 [M_total, K], group_list = [m0, m1, ...]
  x[i] = x[m_i : m_{i+1}, :]

splitItem=2: y 轴分割 (多个 x, 单个 y)
  y = 拼接所有组的输出

splitItem=3: 双轴分割 (单个 x, 单个 y)
```

#### MatMul Tiling 参数

```
CalMMTiling():
├── baseN: 目标 256, 调整以适配 maxN
├── baseK: 从 L0B 容量计算, 上限 128
├── baseM: 从 L0A/L0C 容量计算
├── 所有参数对齐到 16 字节
└── 双缓冲考虑: L0A = BEST_L1_PARTA, L0B = BEST_L1_PARTB

CalculateMMTiling():
├── lambdaN = Ceil(N, baseN * aicCoreNum)
├── singleN = baseN * lambdaN
├── splitM = Ceil(M, baseM)
├── lambdaM = Ceil(splitM, aicCoreNum)
├── singleM = baseM * lambdaM
├── tileTurnNum = Ceil(splitM * lambdaN, aicCoreNum)  ← 总 tile 轮次
└── tailTurnNum = 余数处理
```

#### AllReduce Tiling

```
DoAllReduceTiling():
├── for each group:
│   ├── Mc2Msg:
│   │   ├── sendOff, recvOff (workspace 偏移)
│   │   ├── counts (元素数量)
│   │   ├── turnNum (通信轮次)
│   │   ├── HCCL data type
│   │   └── notify 偏移
│   └── workspace 分配:
│       ├── sendBuf[group]: 每组独立发送缓冲区
│       └── recvBuf[group]: 每组独立接收缓冲区
└── workspaceSize = 16MB (系统需求)
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

源文件：`grouped_mat_mul_all_reduce.cpp` — 入口函数签名 第57-58行

```
入口函数签名 (第57-58行):
  __global__ __aicore__ void grouped_mat_mul_all_reduce(
      GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR group_list,
      GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)

REGISTER_TILING_DEFAULT (第61行)

关键流程:
  1. 解析tiling: GET_TILING_DATA_MEMBER(GMMAllReduceTilingData, aicoreTiling, ...) (第63行)
  2. 获取HCCL上下文: context = GetHcclContext<0>()  ← HcclCombinOpParam
  3. OOMInit(context) ← 地址范围检查 (workspace + windows)
  4. 初始化HcclServer: Init(workspaceMsg, notifyOff)
  5. TilingKey check (第78行)
  6. 仅AIC核运行: AIV核直接return
  7. A3特有: PRELOAD(4) ← 防止阻塞
  8. 创建MatmulImpl: MMImplType<xType, weightType, yType, biasType, CFG_MDL>
     └── CFG_MDL = MatmulConfig中MDL(Maddnn)模式
  9. GMMCompute → GMMProcess → Process(hcclServer)

HcclCombinOpParam结构 (第29-41行):
  ├── WorkSpace / WorkSpaceSize   ← AICPU通信工作区
  ├── rankId / rankDim            ← Rank信息
  ├── windowsIn[8] / windowsOut[8] ← Window IPC地址
  ├── hcomId[128]                 ← HCCL通信域ID
  ├── streamInfo[8]               ← HCCL Stream信息
  └── signalInfo                  ← 跨核信号参数
```

### 4.2 GMMProcess — 分组调度

源文件：`grouped_mat_mul_all_reduce.h:49-143`

```
MNConfig结构 (第31-45行):
  m, k, n             ← 当前组的矩阵维度
  baseM, baseN        ← Tiling分块基准
  mIdx, nIdx          ← 当前分块索引
  numBlocksM/N        ← M/N轴分块数
  singleM/N           ← 单次计算的M/N
  splitM              ← M轴核分配数
  tileCnt             ← tile计数

Init() (第77-90行):
  ├── blockIdx = GetBlockIdx()
  ├── coreIdx = blockIdx            ← AIC核索引
  ├── groupNum = tilingData->groupNum
  ├── baseM/baseN = mmTiling参数
  └── coreNum = tilingData->coreNum

Process() (第93-143行):
  for groupIdx = 0..groupNum:
    ├── N轴并行计算:
    │   ├── lambdaN = Ceil(N, coreNum * baseN)
    │   ├── singleN = lambdaN * baseN
    │   ├── numBlocksN = Ceil(N, singleN)
    │   └── splitM = (coreNum % numBlocksN == 0) ? coreNum/numBlocksN : 1
    ├── M轴并行计算:
    │   ├── lambdaM = Ceil(M, MAX_TURN_NUM(16) * splitM * baseM)
    │   ├── singleM = lambdaM * baseM
    │   ├── numBlocksM = Ceil(M, singleM)
    │   └── tileCnt = Max(1, M/singleM/splitM)
    ├── totalRun = Ceil(M, splitM * singleM)
    └── for i = 0..totalRun:
        ├── mIdx = i*splitM + coreIdx/numBlocksN
        ├── nIdx = coreIdx % numBlocksN
        ├── if coreIdx < splitM*numBlocksN && mIdx < numBlocksM:
        │   └── computeOp.MMCompute(groupIdx, mnConfig)
        └── hcclServer.TurnNotifyRun(coreIdx, splitM*numBlocksN, i+1)
            ← 通知AICPU: 该tile计算完成
    最后: [coreIdx==0] hcclServer.TurnWait(totalRun) ← 等待AllReduce完成
```

```cpp
template <typename ComputeType>
class GMMProcess;
```

#### Init()

```
Init(tiling, tPipe):
├── blockIdx_ = GetBlockIdx()
├── coreIdx_ = blockIdx_ (AIC 核索引)
├── groupNum = tiling.groupNum
├── baseM, baseN = tiling.baseM, baseN
└── coreNum_ = tiling.coreNum
```

#### Process()

```
Process(hcclServer):
├── for groupIdx = 0..groupNum:
│   ├── 计算 N 轴并行:
│   │   ├── lambdaN = Ceil(N, baseN * coreNum)
│   │   ├── singleN = baseN * lambdaN
│   │   └── numBlocksN = Ceil(N, singleN)
│   ├── 计算 M 轴并行:
│   │   ├── splitM = Ceil(M, baseM)
│   │   ├── singleM = baseM * Ceil(splitM, coreNum)
│   │   └── numBlocksM = Ceil(M, singleM)
│   ├── for turn = 0..tileTurnNum:
│   │   ├── MNConfig mnConfig(m, k, n, baseM, baseN, mIdx, nIdx, ...)
│   │   └── computeOp_.MMCompute(groupIdx, mnConfig)
│   └── [coreIdx_==0] hcclServer.TurnNotifyRun()
│       ← 通知 AICPU: 该组的一轮计算完成
│
├── [coreIdx_==0] hcclServer.TurnWait()
│   ← 等待 AICPU 完成所有 AllReduce
└── 结束
```

### 4.3 GMMCompute — 单组 MM 计算

源文件：`grouped_mat_mul_all_reduce.h:147-224`

```
模板参数: mmType (MatmulImpl类型), sync=false
类型别名:
  AT = mmType::AT::T  (输入类型: BF16/FP16)
  BT = mmType::BT::T  (权重类型: BF16/FP16)
  CT = mmType::CT::T  (输出类型: BF16/FP16)
  BiasT = mmType::BiasT::T  (偏置类型: FP32/FP16)

MMImplType<AT_, BT_, CT_, BiasT_, CFG_MDL>:
  └── MT = matmul::MatmulImpl<AT, BT, CT, BiasT, CFG_MDL>
      └── CFG_MDL: MDL (Maddnn) 配置, 优化Cube利用率

Init() (第179-190行):
  ├── 存储TensorList地址指针 (x, weight, bias, y)
  ├── hasBias = (bias != nullptr && GetTensorAddr<BiasT>(0, bias) != nullptr)
  └── workspaceGm设置

MMCompute() (第193-222行):
  ├── 尾块处理:
  │   ├── curSingleN = (nIdx == numBlocksN-1) ? N - nIdx*singleN : singleN
  │   └── curSingleM = (mIdx >= numBlocksM-1) ? M - mIdx*singleM : singleM
  ├── 地址偏移计算:
  │   ├── xOffset = mIdx * singleM * K
  │   ├── wOffset = nIdx * singleN
  │   └── outOffset = mIdx * singleM * N + nIdx * singleN
  ├── 设置GlobalTensor: xGm, weightGm, yGm (按groupIdx从TensorList获取)
  ├── MatmulImpl配置:
  │   ├── SetOrgShape(M, N, K)
  │   ├── SetSingleShape(curSingleM, curSingleN, K)
  │   ├── SetTensorA(xGm[xOffset])
  │   ├── SetTensorB(weightGm[wOffset])
  │   ├── [hasBias] SetBias(biasGm[nIdx * singleN])
  │   └── IterateAll<sync>(yGm[outOffset], false)
  └── 完成该tile的MM计算
```

### 4.4 RAC Server — AIC↔AICPU 通信

源文件：`rac_server.h` — 起始 第24行 (11,026字节)

#### 架构双版本 (A3 vs A2)

```
A3 架构 (__CCE_AICORE__ == 220):
  消息传递: __gm__ 指针直接访问
  同步机制: CrossCoreSetFlag / CrossCoreWaitFlag
  Cache: DataCacheCleanAndInvalid

A2 架构 (__CCE_AICORE__ < 220):
  消息传递: __ubuf__ + DataCopy (GM↔UB)
  同步机制: SyncAll (软同步)
```

#### 关键数据结构

```
AivAicpuOpParam (64字节对齐消息):
├── commType: AicpuComType       ← 通信类型 (ALLREDUCE等)
├── sendBuf, recvBuf: GM_ADDR    ← 发送/接收缓冲区
├── counts: uint32_t             ← 元素计数
├── turnIdx, tileTurnNum         ← 轮次信息
├── groupIdx                     ← 当前分组
├── mc2BufferType                ← 缓冲区类型
└── dataType                     ← 数据类型

HcclConfig:
├── rankId, rankDim              ← Rank 信息
├── workspace                    ← 工作区地址
├── windows[8]                   ← 各 Rank 的 Window 地址
└── hcomId                       ← HCCL 通信域 ID
```

#### TurnNotifyRun() — 通知 AICPU

```
A3 版本:
  TurnNotifyRun():
  ├── 填充 AivAicpuOpParam 消息 (commType=ALLREDUCE, counts, buffers)
  ├── DataCacheCleanAndInvalid: 刷新消息到 GM
  └── CrossCoreSetFlag<AC_MSG_CNT + HARD_SYNC_EVENT_ID>: 唤醒 AICPU
```

#### TurnWait() — 等待 AICPU

```
A3 版本:
  TurnWait():
  ├── CrossCoreWaitFlag<HARD_SYNC_EVENT_ID>: 等待 AICPU 完成标志
  └── DataCacheCleanAndInvalid: 刷新完成状态
```

---

## 5. Host-Kernel 接口

**注意**: 本算子没有独立的 `tiling_key.h` 或 `tiling_struct.h` 文件，Tiling 数据结构直接内联在 Kernel 侧代码中。

### 5.1 Tiling 数据结构

Host 侧通过 `GMMAllReduceTilingData` 将计算参数和通信参数传递给 Kernel 侧。Host 侧 tiling 定义在 `grouped_mat_mul_all_reduce_tiling.h`，Kernel 侧通过 `GET_TILING_DATA_MEMBER` 宏（第63行）解析。

```
GMMAllReduceBaseParams:
├── groupNum, coreNum             ← 组数/核数
├── activeType                    ← 激活类型
├── ubBaseM, ubBaseN              ← UB 基准分块
├── ubCalSize, ubRestSize         ← UB 计算/剩余大小
├── workspaceSize                 ← 工作区大小
├── mList[64], kList[64], nList[64] ← 每组的 M/K/N 维度
└── (最多 64 组)

GMMAllReduceCoreTiling:
├── GMMAllReduceBaseParams        ← 基础参数
├── TCubeTiling mmTiling          ← MM Tiling
├── notifyOff                     ← 通知偏移
└── debugMode                     ← 调试模式

GMMAllReduceTilingData:
├── KFCGroupedTiling aicpuTiling  ← AICPU 侧 Tiling (含 Mc2Msg)
│   ├── msg[64*128]               ← 消息缓冲区
│   ├── groupNum                  ← 组数
│   └── groupTilingMagicNum       ← 魔数验证
└── GMMAllReduceCoreTiling aicoreTiling ← AIC 侧 Tiling
```

### 5.2 Workspace 布局

Workspace 由 Host 侧 `GetWorkspaceSize()` 分配，传递给 Kernel 侧使用。布局如下：

```
Workspace 布局:
├── [0, workspaceSize)                    ← 整体工作区
│   ├── sendBuf[group][offset]            ← 各组发送缓冲区 (Mc2Msg.sendOff)
│   ├── recvBuf[group][offset]            ← 各组接收缓冲区 (Mc2Msg.recvOff)
│   ├── notify区域                        ← 通知偏移 (GMMAllReduceCoreTiling.notifyOff)
│   └── 系统保留 (16MB)                   ← HCCL 通信基础开销
│
├── Host 侧分配 (aclnnGetWorkspaceSize):
│   ├── DoAllReduceTiling() 计算各组 sendOff/recvOff
│   ├── workspaceSize = max(计算值, 16MB)
│   └── Mc2Msg 结构体填充通信参数
│
└── Kernel 侧使用:
    ├── HcclCombinOpParam.WorkSpace       ← AICPU 通信工作区基地址
    ├── HcclCombinOpParam.WorkSpaceSize   ← 工作区大小
    └── HcclServer.Init(workspaceMsg, notifyOff) ← 初始化通信服务
```

---

## 6. 计算与通信重叠

### 6.1 流水线时序

```
时序图 (以 2 组为例):

AIC:  [GMM_0 Turn0] [GMM_0 Turn1] [GMM_1 Turn0] [GMM_1 Turn1]
         ↓ Notify       ↓ Notify       ↓ Notify       ↓ Notify
AICPU:    [AllReduce_0 Turn0] [AllReduce_0 Turn1] [AllReduce_1 Turn0] ...
                              ↑ 结果可用        ↑ 结果可用

重叠效果:
  Turn N 的 AllReduce 与 Turn N+1 的 GMM 计算并行
  AIC 和 AICPU 通过 CrossCoreFlag 同步, 无需等待全部完成
```

### 6.2 核分配策略

```
AIC 核 (仅 AIC 运行):
  N 轴并行: numBlocksN = Ceil(N, baseN * aicCoreNum)
  M 轴并行: splitM = Ceil(M, baseM)
  Tile 轮次: tileTurnNum = Ceil(splitM * numBlocksN, aicCoreNum)

  每个 AIC 核负责:
    blockIdx → mIdx, nIdx 映射
    → 计算该分片的 MM 结果
    → 通知 AICPU 进行通信

AICPU:
  等待 AIC 的 TurnNotifyRun 信号
  → 执行 HCCL AllReduce
  → 发送 TurnWait 完成信号
```

---

## 7. 依赖关系

```
grouped_mat_mul_all_reduce/
├── op_kernel/
│   ├── rac_server.h             ← AIC-AICPU通信服务
│   │   ├── HcclServer类         ← CrossCoreFlag同步
│   │   ├── AivAicpuOpParam      ← 64B对齐消息结构
│   │   └── HcclCombinOpParam    ← HCCL上下文 (Windows, Rank信息)
│   ├── grouped_mat_mul_all_reduce_utils.h  ← 工具函数
│   │   ├── MMImplType           ← MatmulImpl类型包装 (CFG_MDL配置)
│   │   ├── GetTensorAddr<T>()   ← TensorList地址解引用
│   │   └── AlignUp/AlignDown    ← 对齐工具
│   └── grouped_mat_mul_all_reduce.h
│       ├── GMMProcess           ← 分组调度
│       ├── GMMCompute           ← 单组MM计算
│       └── MNConfig             ← 分块配置
├── op_host/
│   ├── grouped_mat_mul_all_reduce_def.cpp       ← 算子定义
│   ├── grouped_mat_mul_all_reduce_infershape.cpp ← Shape推导 (4种splitItem模式)
│   └── op_tiling/
│       └── grouped_mat_mul_all_reduce_tiling.cpp ← Tiling (CalMMTiling + DoAllReduceTiling)
├── op_api/
│   ├── aclnn_grouped_mat_mul_all_reduce.h       ← ACLNN声明 (已DEPRECATED)
│   └── aclnn_grouped_mat_mul_all_reduce.cpp     ← ACLNN实现
└── op_graph/
    └── fallback_grouped_mat_mul_all_reduce.cpp  ← 图模式回退路径

外部依赖:
  ├── lib/matmul_intf.h           ← AscendC MatmulImpl (CFG_MDL配置)
  ├── kernel_tiling/kernel_tiling.h ← Mc2InitTiling/Mc2CcTiling
  └── 3rd/grouped_matmul/         ← GroupedMatmul Tiling工具
```

---

*分析完成时间: 2026-04-21*
