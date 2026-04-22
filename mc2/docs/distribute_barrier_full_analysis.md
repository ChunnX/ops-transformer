# distribute_barrier 算子全面分析报告

## 1. 算子概述

`distribute_barrier` 是 MOE 分布式计算中的 **全局同步屏障算子**。在多卡 MOE Dispatch/Combine 场景中，确保所有 Rank 在进入下一阶段前完成当前阶段的工作。通过 HCCL Window 共享内存实现高效的跨 Rank 通知-等待机制，无需传统集合通信原语。

```
Rank 0 ──SetStatus──→ Window[Rank 1..N] 的状态槽位
Rank 1 ──SetStatus──→ Window[Rank 0..N] 的状态槽位 (跳过自己)
...
Rank N ──SetStatus──→ Window[Rank 0..N-1] 的状态槽位

每个 Rank 自旋等待本地 Window 被所有其他 Rank 写入 → 屏障完成
```

### 1.1 版本

- **V1**: 基本版本，无超时和弹性信息
- **V2**: 增加 `timeOut` 和 `elasticInfo` 可选输入

---

## 2. 目录结构与文件职责

```
distribute_barrier/
├── op_kernel/
│   ├── distribute_barrier.cpp              (1,451字节) 内核入口
│   ├── distribute_barrier.h                (15,222字节) 主类实现
│   └── distribute_barrier_tiling.h         (1,273字节) Tiling 结构
├── op_host/
│   ├── distribute_barrier_def.cpp          (4,515字节) 算子定义
│   ├── distribute_barrier_infershape.cpp   (2,457字节) Shape 推导
│   └── op_tiling/
│       └── distribute_barrier_tiling.cpp   (11,944字节) Tiling 策略
├── op_api/
│   ├── aclnn_distribute_barrier.cpp        (1,929字节) V1 ACLNN
│   ├── aclnn_distribute_barrier.h          (2,585字节) V1 声明
│   ├── aclnn_distribute_barrier_v2.cpp     (2,006字节) V2 ACLNN
│   ├── aclnn_distribute_barrier_v2.h       (2,714字节) V2 声明
│   ├── distribute_barrier_base.cpp         (3,710字节) 共享逻辑
│   └── distribute_barrier_base.h           (1,938字节) 共享声明
├── op_graph/
│   ├── distribute_barrier_gen_task.cpp     (2,748字节) 图模式任务
│   └── distribute_barrier_proto.h          (2,264字节) 图模式注册
├── config/ascend910_93/
│   ├── distribute_barrier_binary.json      (17,957字节) 12 种 dtype 二进制
│   └── distribute_barrier_simplified_key.ini (1,241字节)
└── tests/ut/
    ├── op_api/test_aclnn_distribute_barrier.cpp   (6,060字节)
    ├── op_host/test_distribute_barrier_infershape.cpp (1,885字节)
    ├── op_host/arch32/test_distribute_barrier_tiling.cpp (7,901字节)
    └── op_kernel/test_distribute_barrier.cpp      (2,986字节)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `distribute_barrier_def.cpp` (4,515字节)

**类定义**: `class DistributeBarrier` — 第19行

**构造函数**: 第21-64行

**输入张量**:

| 序号 | 名称 | 类型 | 维度 | 说明 | 源码位置 |
|------|------|------|------|------|----------|
| 0 | x_ref | 多类型(12种) | [-2] | 引用张量 (仅用于 Shape 传递) | 第22-27行 |
| 1 | time_out | INT32 (可选) | [1] | 超时时间 (微秒) | 第28-33行 |
| 2 | elastic_info | INT32 (可选) | [4+2*worldSize] | 弹性训练信息 | 第34-39行 |

**输出张量**:

| 序号 | 名称 | 类型 | 说明 | 源码位置 |
|------|------|------|------|----------|
| 0 | x_ref | 同输入 | 透传输出 | 第41-45行 |

**关键属性**:

| 属性名 | 类型 | 说明 | 源码位置 |
|--------|------|------|------|----------|
| group | String | HCCL 通信域 | 第47-48行 |
| world_size | Int | 全局卡数 | 第47-48行 |

**OP_ADD 注册**: 第67行

**支持数据类型**: BF16, FP16, FP32, BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64

**平台**: `ascend910_93`, `ascend950`

### 3.2 Tiling 策略

**文件**: `distribute_barrier_tiling.cpp` (11,944字节)

#### 参数校验

```
CheckAndSetAttrs():
├── group: 非空字符串, 长度 < HCCL_GROUP_NAME_MAX
└── world_size: [2, 384]

CheckTimeOut():
├── 1D, dim0 == 1
├── dtype == INT32
└── format != FRACTAL_NZ

CheckElasticInfo():
├── 1D, dim0 == 4 + 2 * worldSize
├── dtype == INT32
└── format != FRACTAL_NZ
```

#### Tiling 配置

```
DistributeBarrierTilingFunc():
├── 检测可选输入 (timeOut, elasticInfo)
├── CheckAndSetAttrs()
├── [hasTimeOut] CheckTimeOut()
├── [hasElasticInfo] CheckElasticInfo()
├── SetWorkSpace(): 16 MB 系统 workspace
├── SetHcommCfg():
│   ├── engine: AIV_ENGINE
│   └── algorithm: "AlltoAll=level0:fullmesh;level1:pairwise"
├── blockDim = aivNum (所有 AIV 核)
├── scheduleMode = batch (所有核同时启动)
└── 记录 aivNum, ubSize 到 tiling
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `distribute_barrier.cpp` (1,451字节)

**入口函数签名**: 第27-28行

**`REGISTER_TILING_DEFAULT`**: 第29行

**`GET_TILING_DATA_WITH_STRUCT`**: 第32行

```cpp
__global__ __aicore__ void distribute_barrier(...)
{
    REGISTER_TILING_DEFAULT(tilingGM);
    GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingGM);
    
    DistributeBarrier<DTYPE_X_REF> op;
    op.Init(xRef, timeOut, elasticInfo, xRefOut, workspaceGM, &tilingData);
    op.Process();
}
```

### 4.2 主类 DistributeBarrier

**文件**: `distribute_barrier.h` (15,222字节)

```cpp
namespace DistributeBarrierImpl {
template <typename XType>
class DistributeBarrier;
}
```

#### 常量定义

```cpp
static constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;  // Window 状态区偏移 512KB
static constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;  // 状态 Window 偏移 900KB
static constexpr uint64_t STATE_OFFSET = 512;              // 状态槽位间距 512B
static constexpr uint32_t UB_ALIGN = 32;                   // UB 对齐
static constexpr uint32_t CYCLES_PER_US = 50;              // 每微秒周期数
```

#### Init() — 初始化

```
Init(xRef, timeOut, elasticInfo, xRefOut, workspace, tiling):
├── 读取 Tiling:
│   ├── rankId, worldSize, aivNum
│   ├── isInputTimeOut, isInputElasticInfo
│   └── totalUbSize, totalWinSize
├── [isInputTimeOut] 从 GM 读取 timeOut 值 (微秒)
├── [isInputElasticInfo] 从 GM 读取 elasticInfo
├── 获取 HCCL Window Context:
│   └── mc2KernelUtils::GetHcclContext(contextGM, rankId)
├── InitStatus(): 读取/切换数据状态 flag
├── SplitToCore(): 分配 Rank 到 AIV
└── 分配 UB buffer:
    ├── statusLocal_ (XType, 32 元素) — 状态读取
    ├── gatherMask_ (XType, 32 元素) — 汇聚掩码
    ├── scalarLocal_ (XType, 32 元素) — 标量
    ├── timeOutInfo_ (XType, 32 元素) — 超时信息
    └── waitStatusLocal_ (XType, 32 元素) — 等待状态
```

#### InitStatus() — 状态初始化

```
InitStatus():
├── 从 Window 状态区读取当前 flag
├── flag 翻转策略:
│   ├── dataState_ = (lastFlag == 0.0) ? 1.0 : 0.0
│   └── 每次调用交替 0↔1, 实现跨次调用隔离
└── 将新 flag 写回 Window
```

#### SplitToCore() — 负载分配

```
SplitToCore():
├── workerNum = aivNum - 1  (AIV 0 保留给超时监控)
├── baseRanks = worldSize / workerNum
├── remainder = worldSize % workerNum
├── 每个 Worker 分配:
│   ├── 前 remainder 个: baseRanks + 1 个 Rank
│   └── 其余: baseRanks 个 Rank
└── sendRankNum_ = 本 AIV 负责的 Rank 数
```

#### Process() — 主流程

```
Process():
├── if (aivId_ == 0):            ← AIV 0: 超时监控
│   └── TimeOutTest()
├── else:                         ← AIV 1..N: 屏障执行
│   └── BarrierProcess()
└── SyncAll<true>()
```

#### SetStatus() — 通知其他 Rank

```
SetStatus():
├── for each assignedRank:
│   ├── targetWindowAddr = GetWindowAddr(assignedRank)
│   ├── offset = WIN_STATE_OFFSET + rankId * STATE_OFFSET
│   ├── DataCacheCleanAndInvalid(targetWindowAddr + offset)
│   └── 写入 statusLocal_ = 1.0 到 targetWindowAddr + offset
└── 每个 AIV 通知自己负责的 Rank 子集
```

#### WaitStatus() — 等待所有 Rank

```
WaitStatus():
├── expectedCount = sendRankNum_ * 8   ← 每个 Rank 写 8 字节 (FP16 half = 2B × 4)
├── localWindowAddr = selfWindowAddr
├── offset = WIN_STATE_OFFSET + STATE_OFFSET  ← 各 Rank 的状态槽位
├── do:
│   ├── DataCacheCleanAndInvalid(localWindowAddr + offset)
│   ├── sum = 0
│   ├── for each rank (跳过自己):
│   │   ├── 读取 localWindowAddr + rank * STATE_OFFSET
│   │   └── sum += value
│   └── ReduceSum(statusLocal_) → sum
├── while (sum != expectedCount)
└── 自旋等待直到所有 Rank 的状态都为 1.0
```

#### CleanStatus() — 清除状态

```
CleanStatus():
├── 写入完成标志 (finishFlag) 到 Window
├── 清零本地接收状态区:
│   ├── for rank 0..worldSize:
│   │   └── 写 0.0 到 selfWindowAddr + rank * STATE_OFFSET
└── 为下一次屏障调用做准备
```

#### TimeOutTest() — 超时监控 (AIV 0)

```
TimeOutTest() [仅 AIV 0]:
├── do:
│   ├── 读取所有 Worker AIV 的 finishFlag
│   ├── ReduceSum(gatherMask_) → sum
│   └── if (sum == workerNum): break  ← 所有 AIV 完成
│
│   [isInputTimeOut]:
│   ├── elapsedCycles = GetCycleCount() - startCycles
│   ├── if (elapsedCycles > timeOut * CYCLES_PER_US * 1000):
│   └── assert(false)  ← 超时报错
├── while (true)
└── 超时阈值: timeOut (μs) × 50 (cycles/μs) × 1000 (ms→μs)
```

---

## 5. 数据结构

### 5.1 Tiling 数据结构

**文件**: `distribute_barrier_tiling.h`

**`DistributeBarrierInfo` 结构体**: 第20-28行

**`DistributeBarrierTilingData` 结构体**: 第30-34行

```
DistributeBarrierInfo:
├── worldSize       : uint32  ← 全局卡数
├── rankId          : uint32  ← 本 Rank ID
├── aivNum          : uint32  ← AIV 核数
├── totalUbSize     : uint64  ← UB 总大小
├── totalWinSize    : uint64  ← Window 总大小
├── isInputTimeOut  : bool    ← 是否有超时输入
└── isInputElasticInfo: bool  ← 是否有弹性信息输入

DistributeBarrierTilingData:
├── Mc2InitTiling   : Mc2InitTiling   ← HCCL 初始化
├── mc2CcTiling1    : Mc2CcTiling     ← 通信参数
└── DistributeBarrierInfo             ← 算子参数
```

### 5.2 Window 内存布局

```
Window Memory 布局 (每个 Rank):
┌─────────────────────────────────────────────────────────┐
│ 0x00000                                                 │
│   ... (其他用途) ...                                     │
│                                                         │
│ WIN_STATE_OFFSET (512KB)                                │
│ ┌─────────┬─────────┬─────────┬─────┬─────────┐       │
│ │ Rank 0  │ Rank 1  │ Rank 2  │ ... │ Rank N  │       │
│ │ 512B    │ 512B    │ 512B    │     │ 512B    │       │
│ │[status] │[status] │[status] │     │[status] │       │
│ └─────────┴─────────┴─────────┴─────┴─────────┘       │
│                                                         │
│ STATE_WIN_OFFSET (900KB)                                │
│ ┌─────────────────────────────────────────┐             │
│ │ 本地状态/完成标志区                       │             │
│ └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘

每个 Rank 的状态槽位:
  offset = WIN_STATE_OFFSET + srcRank * STATE_OFFSET
  容量: 512B (对齐，实际使用 ~8B)
```

---

## 6. 通信模式

### 6.1 屏障同步机制

```
屏障同步流程 (以 4 Rank, 3 Worker AIV 为例):

Step 1: SetStatus — 每个 Rank 通知其他 Rank
┌──────┐     写 1.0     ┌──────┐     写 1.0     ┌──────┐     写 1.0     ┌──────┐
│Rank 0│──────────────→ │Rank 1│──────────────→ │Rank 2│──────────────→ │Rank 3│
│Rank 0│←────────────── │Rank 1│←────────────── │Rank 2│←────────────── │Rank 3│
│Rank 0│──────────────→ │Rank 1│←────────────── │Rank 2│──────────────→ │Rank 3│
│Rank 0│←────────────── │Rank 1│──────────────→ │Rank 2│←────────────── │Rank 3│
└──────┘                └──────┘                └──────┘                └──────┘

Step 2: WaitStatus — 每个 Rank 自旋等待本地 Window
  Rank 0: 等待 Rank 1/2/3 的槽位都变为 1.0
  Rank 1: 等待 Rank 0/2/3 的槽位都变为 1.0
  ...

Step 3: CleanStatus — 清除所有槽位为 0.0
```

### 6.2 多核并行策略

```
AIV 核分配:
┌─────────┬──────────────────────────────────────┐
│ AIV 0   │ 超时监控 (TimeOutTest)               │
├─────────┼──────────────────────────────────────┤
│ AIV 1   │ 负责向 Rank 0..(N/(aivNum-1)-1) 写   │
│ AIV 2   │ 负责向下一批 Rank 写                  │
│ ...     │ ...                                  │
│ AIV M   │ 负责向最后一批 Rank 写                │
└─────────┴──────────────────────────────────────┘

优势:
  - 多个 AIV 并行写不同 Rank 的 Window → 减少总通知延迟
  - AIV 0 独立监控超时 → 不影响屏障执行效率
```

---

## 7. ACLNN API

### V1 API

```cpp
// 获取 workspace 大小
aclnnDistributeBarrierGetWorkspaceSize(
    aclTensor* xRef,
    const char* group,
    int64_t worldSize,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

// 执行
aclnnDistributeBarrier(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

### V2 API (增加超时和弹性信息)

```cpp
aclnnDistributeBarrierV2GetWorkspaceSize(
    const aclTensor* xRef,
    const aclTensor* timeOut,        // 可选
    const aclTensor* elasticInfo,    // 可选
    const char* group,
    int64_t worldSize,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

**HCCL Server Type**: MTE (非 AICPU) — 更优的退出性能

---

## 8. 函数调用关系

```
distribute_barrier (内核入口)
└── DistributeBarrier::Init()
    ├── 读取 Tiling (rankId, worldSize, aivNum)
    ├── [isInputTimeOut] 读取超时值
    ├── [isInputElasticInfo] 读取弹性信息
    ├── GetHcclContext() → Window 地址
    ├── InitStatus() → 状态 flag 翻转
    ├── SplitToCore() → Rank 分配到 AIV
    └── 分配 UB buffer

└── DistributeBarrier::Process()
    ├── [aivId==0] TimeOutTest()
    │   ├── 自旋等待所有 AIV finishFlag
    │   └── [isInputTimeOut] 超时检测
    └── [aivId>0] BarrierProcess()
        ├── SetStatus()
        │   └── for each assignedRank:
        │       └── DataCacheCleanAndInvalid + 写 1.0
        ├── WaitStatus()
        │   └── 自旋: ReduceSum(所有 Rank 状态) == expected
        └── CleanStatus()
            ├── 写 finishFlag
            └── 清零所有状态槽位
    └── SyncAll<true>()
```

---

## 9. 关键设计点

### 9.1 为什么不用 HCCL AllReduce 做屏障

```
传统方式: AllReduce(1.0) → 等待 result == worldSize
  优点: 标准化
  缺点: 通信开销大 (需要跨卡同步数据)

Window 方式: 直接写 Window 内存
  优点: 低延迟 (点到点直接写), 无需集合通信 API
  缺点: 需要预先注册 Window 内存
```

### 9.2 状态翻转 (Ping/Pong)

```
第 1 次调用: dataState_ = 1.0, 等待 1.0
第 2 次调用: dataState_ = 0.0, 等待 0.0
第 3 次调用: dataState_ = 1.0, 等待 1.0
...

目的: 防止上一次屏障的残留状态干扰下一次
```

### 9.3 超时监控

```
AIV 0 独立运行 TimeOutTest():
  - 不参与屏障通知, 仅监控
  - 计算经过的 cycle 数
  - 超过 timeOut * CYCLES_PER_US * 1000 则 assert
  - 用于检测死锁或慢 Rank
```

---

*分析完成时间: 2026-04-21*
