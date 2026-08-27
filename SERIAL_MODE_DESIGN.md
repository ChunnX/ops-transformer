# A5 FullMeshV2 小 bs 串行模式优化报告

## 背景

FullMeshV2 在 bs=1, h=4096, k=6 场景下耗时 11.7us，而 FullMeshV1 仅 9.3us，差距 2.4us，目标 10us 以内。

## 根因分析

通过打点数据（fullmesh_core_debug 分支）定位到瓶颈：

```
send_token 核 (0-67):
  WaitCumSumFlag 轮询次数: 36-53 次
  WaitCumSumFlag 结束: ~9900-10800 cycles

cumsum 核 (68-71):
  WaitDispatch 等待: 4170-4856 cycles
  GetCumSum 计算: 774-1912 cycles
```

V2 比 V1 慢的三个根因：

| 差异点 | V1 | V2 Parallel |
|--------|----|-------------|
| cumsum 计算 | LocalWindowCopy 内联 GatherMask+ReduceSum | 专门核并行计算，需 WaitCumSumFlag 软同步（37-53次轮询） |
| 数据/count 发送顺序 | 先数据后count（保序） | count 先于数据（需 CheckDataArriveWithFlag 轮询） |
| 同步链 | 1次 WaitDispatch | 2次：WaitDispatch + WaitCumSumFlag |

## 优化方案：Serial 模式

核心思路：**小 bs（bs < 32）时，让所有核串行走完整流程，用"每个核独立读完整 window status 算全局前缀和"替代 parallel 模式的"核分组 + workspace 跨核聚合"**。

### Process 分流

```cpp
void Process() {
    if (useSerialMode_) {
        ProcessSerial();
    } else {
        ProcessParallel();
    }
}

void ProcessSerial() {
    AllToAllDispatchSerial();       // 先数据后count
    PipeBarrier<PIPE_ALL>();
    WaitDispatchAndCalcPrefix();   // 等count + 独立算全局前缀和
    LocalWindowCopySerial();        // 直接拷贝，无CheckDataArrive
}
```

### 三个核心改动

#### 1. AllToAllDispatchSerial：先发数据后发 count

```cpp
void AllToAllDispatchSerial() {
    InitDispatchBuffers(inQueue, outBuf);
    if (activeMaskBsCnt_ > 0) {
        AllToAllDispatchA5(inQueue, outBuf);  // 先发数据
    }
    CalAndSendCnt();                           // 后发count（MTE3保序）
}
```

同核 MTE3 流水保序：count 在数据之后发送，count 到达则数据必然已到达，无需 CheckDataArriveWithFlag。

#### 2. WaitDispatchAndCalcPrefix：独立算全局前缀和

```cpp
void WaitDispatchAndCalcPrefix() {
    tpipe_->Reset();
    SplitToCore(rscvStatusNum_, aivNum_, ...);
    BufferInit();

    // 等本核负责区间的count到达
    while (sumOfFlag != compareTarget) { ... }

    // 每个核独立读完整window status到UB
    DataCopy(fullStatusTensor, windowInstatusFp32Tensor_, fullCopyParams);

    // 清状态（只清本核区间）
    WaitDispatchClearStatus();

    // 从头累加算全局前缀和，只写本核负责区间的结果
    for (index = 0; index < rscvStatusNum_; index++) {
        curCnt += count[index];
        if (index ∈ [startStatusIndex_, endStatusIndex_)) {
            outCountLocal.SetValue(index - startStatusIndex_, curCnt);
        }
        if (index == expertEndStatusId && index ∈ 本核区间) {
            expertTokenPrefixTensor.SetValue(...);
        }
    }

    // 写workspace（供SetValidExpertInfo读取）
    DataCopyPad(workspaceGlobal[startStatusIndex_], outCountLocal, ...);

    SyncFunc<MTE3_S>();
    SyncAll<true>();  // 保证所有核写完后LocalWindowCopySerial才读
}
```

关键设计决策：**不用 workspace 做跨核前缀和聚合**，而是像 V1 一样每个核读全量 status 各自算。彻底消除 workspace 地址踩踏风险，代价是每个核多读一份 GM status，但小 bs 时 status 数量不大（如 448），可接受。

#### 3. LocalWindowCopySerial：直接拷贝

跳过 WaitCumSumFlag 和 CheckDataArriveWithFlag，直接从 window 拷贝数据到输出。

### 代码简化

提取 `InitDispatchBuffers` 公共方法，消除 `AllToAllDispatchSerial` 与 `AllToAllDispatch` 的约 47 行重复代码。

### 共享专家支持

Serial 模式保留 `sharedUsedAivNum_` 的原始计算值，`AllToAllDispatchA5` 中 `aivId_ >= moeUsedAivNum_` 的核走 `SendToSharedExpert`，其余走 `SendToMoeExpert`。

## 修复的问题

### 阻塞问题

| 问题 | 修复 |
|------|------|
| workspace 并发覆盖（临时和区与前缀和区重叠） | 改为每个核独立读全量 status 算前缀和，不经过 workspace 跨核聚合 |
| expertTokenPrefix 核间写写冲突（全量循环导致多核写同一 GM 区域） | expertEndStatusId 匹配时加 `[startStatusIndex_, endStatusIndex_)` 区间限制 |
| BufferInit 中 expertTokenPrefixNum 估算低估 | 改为 `isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_` |
| recStatusNumPerCore_ == 0 时 DataCopy blockCount=0 | 加 `if (recStatusNumPerCore_ > 0)` 保护，但仍参与 SyncAll |

### 建议问题

| 问题 | 修复 |
|------|------|
| SyncAll 前缺 MTE3 等待 | SyncAll 前加 `SyncFunc<MTE3_S>()` |
| UB 空间不足（AllToAllDispatchSerial 的 buffer 未清理） | WaitDispatchAndCalcPrefix 开头加 `tpipe_->Reset()` |

## 预期收益

| 消除项 | 节省 |
|--------|------|
| WaitCumSumFlag 37-53 次轮询 | 1.0-1.5us |
| CheckDataArriveWithFlag 轮询 | 0.3-0.6us |
| cumsum 核冗余 ExpIdsCopyAndMaskCal | 0.2-0.3us |
| GatherSumRecvCnt + GetCumSum 跨核同步 | 0.3-0.5us |
| **合计** | **1.8-2.9us** |

## 已知限制

- Serial 模式仅适用于 `axisBS_ < 32` 且无共享专家卡（`sharedExpertRankNum_ == 0`）的场景
- 每个核独立读全量 status 增加 GM 带宽压力，大 bs 时不适用
- 逐元素 SetValue 计算前缀和未用 Vector/CumSum API，小 bs 可接受

## 文件改动

- `mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h`
  - 新增：`SERIAL_MODE_BS_THRESHOLD`、`useSerialMode_`、`ProcessSerial`、`ProcessParallel`、`AllToAllDispatchSerial`、`InitDispatchBuffers`、`WaitDispatchAndCalcPrefix`、`LocalWindowCopySerial`
  - 修改：`Process` 分流、`SetTilingDataAndCal` 设置 serial 模式、`BufferInit` 修正 expertTokenPrefixNum
