# 多核并行通信方案演进史

> 日期: 2026-08-21 (最终版)
> 平台: Ascend 950PR (dav-3510), CANN 9.2.0
> 代码版本: 方案B — 后R核通信 + 前(28-R)核计算 (严格分离)

## 一、最终方案 (方案B, 已验证)

### 核心设计

| 组件 | totalJobs | 作用 |
|------|-----------|------|
| **teamBarrier** | **1** | block 0 (jobIndex=0) 的 CrossDevice step=1，正确轮询所有 remote rank |
| **AllToAll** | **rankSize** | 后 R 个核各负责 1 个 target rank，并行 PUT |

### 关键参数

```cpp
constexpr uint64_t WIN_DATA_OFFSET = 128;  // PUT 数据跳过 win 区前 128B barrier counter
uint32_t jobIndex = GetBlockNum() - 1 - GetBlockIdx();  // block 27: jobIndex=0, block 26: jobIndex=1
bool isCommBlock = (jobIndex < rankSize);               // 后 R 核通信
uint32_t rsCoreNum = usedCoreNum - rankSize;             // 28-R 核计算
bool isComputeBlock = (blockIdx < rsCoreNum);            // 前 28-R 核计算
```

### 同步流程

```
通信核: Commit<BARRIER_NONE> → Wait<BARRIER_NONE> (Drain only, 保证本 rank PUT 到达远程)
所有核: SyncAll (本地 28 核同步, 通信核 Drain 完成)
block 0 (jobIndex=0): teamBarrier_.CrossDevice() (step=1, 轮询所有 remote localFlag)
所有核: SyncAll (确认跨设备数据可见)
计算核: reduceSum (仅前 28-R 核, rsCoreNum 传参)
```

### 验证结果

- 精度: R=2 ALL PASS, R=4 ALL PASS (4/4)
- 性能 (msprof Task Duration 中位数, 跨 rank max):
  - vs 旧基线 (totalJobs=1, winOffset=0): 平均 -11.1% (R=2 -7.4%, R=4 -14.7%)
  - 最佳 case: 4096x3584x4096 R=4 = 307us vs 基线 415us (-26.0%)

## 二、历史失败尝试与教训

### 尝试1: totalJobs=rankSize + winOffset=0

**结果**: Rank 0 PASS, Rank 1 FAIL (NaN, 20784/4194304)

**根因**: winOffset=0 导致 PUT 数据从 win 区 offset 0 开始写入，覆盖 barrier 的 localFlag (offset 0) 和 teamSyncCounter (offset 32)。CrossDeviceExecute 读写被覆盖的 flag，读到 PUT 的 BF16 数据（值≈10）而非 count 值，同步逻辑失效。

**修复**: winOffset=128，PUT 数据从 offset 128 开始写入，跳过前 128B 的 barrier counter 区域。

### 尝试2: totalJobs=rankSize + winOffset=128 + Wait\<BARRIER_DEVICE\>

**结果**: Rank 0 PASS, Rank 1 FAIL (80% 元素错误)

**根因**: CrossDeviceExecute 的 step 逻辑在 totalJobs=rankSize 时不轮询 remote rank：

```cpp
int32_t step = totalJobs_ < nranks ? (int32_t)totalJobs_ : (int32_t)nranks;
for (uint32_t i = jobIndex_; i < nranks; i += (uint32_t)step) {
    if (i == myRank) continue;  // step=rankSize 时只遇到自己, 直接退出
}
```

Wait\<BARRIER_DEVICE\> 中的 CrossDevice 不轮询 remote，SyncAll 只同步本地 28 核。接收方在远程 PUT 数据到达前开始 reduceSum，读到未初始化数据。

### 尝试3: totalJobs=rankSize + 额外 SyncAll

**结果**: Rank 0 PASS, Rank 1 FAIL (80% 错误，与尝试2相同)

**根因**: SyncAll 是本地同步（同步本设备 28 核），不具备跨设备同步能力。加多少次 SyncAll 都无法保证远程 rank 的 PUT 数据已到达本 rank 的 HBM。

### 尝试4: totalJobs=rankSize + ManualCrossDeviceSync (手动轮询 remote localFlag)

**结果**: 死循环超时 (do-while(true) 无法退出)

**根因**: MTE3 跨设备 GM 写入可见性无保证。手动轮询 remote localFlag 使用的 MTE2 读取无法看到 remote 的 MTE3 写入（跨设备 GM 可见性需要框架的双向 fence 机制保证）。此外，手动 UB buffer 和 flag 操作与框架的 CrossDevice 冲突。

### 尝试5: totalJobs=1 + winOffset=128 (单核通信 + 真同步)

**结果**: R=2 ALL PASS, **R=4 死锁挂起**

**根因**: winOffset=128 把 CrossDevice 同步从"假通过"（winOffset=0 时 flag 被覆盖，读到的 BF16 值恒 >= count）变成真的同步。R=4 单核通信需要串行 3 次 PUT + 3 次 Drain，在真实同步下暴露死锁竞态。

## 三、方案B 成功的关键突破

### 突破点: teamBarrier 和 AllToAll 使用不同的 totalJobs

之前所有尝试都用**同一个 totalJobs** 传给 teamBarrier 和 AllToAll。方案B 的突破在于分离：

- **teamBarrier totalJobs=1**: CrossDevice 用 step=1 正确轮询所有 remote rank 的 localFlag
- **AllToAll totalJobs=rankSize**: 后 R 个核各负责 1 个 target，并行 PUT

这样 CrossDevice 的跨设备 fence 正确工作（step=1），同时 AllToAll 的 work partition 实现多核并行通信。

### 方案A → 方案B 的改进

| 维度 | 方案A (通信核也计算) | 方案B (严格分离) |
|------|---------------------|-----------------|
| 通信核职责 | 通信 + reduceSum (Phase 2 补做) | 仅通信 |
| 计算核职责 | reduceSum (Phase 1 部分) | 仅 reduceSum (全部) |
| reduceSum 核数 | 28 (所有核) | 28-R (仅计算核) |
| Phase 2 | 有 (通信核补做 reduceSum) | 无 |
| 关键路径 | max(AT, RS_phase1) + RS_phase2 | max(AT, RS_strict) |
| 性能 vs 方案A | — | 平均 -7.5% |

方案B 消除了 Phase 2 的串行追加，所有 reduceSum 在 Phase 1 与通信完全并行。虽然 reduceSum 核数减少（28 → 28-R），但消除串行的收益 > 减少核数的代价。

## 四、框架限制总结

### 框架 CrossDeviceExecute 的 step 逻辑

```cpp
int32_t step = totalJobs_ < nranks ? (int32_t)totalJobs_ : (int32_t)nranks;
```

- totalJobs=1: step=1，job0 轮询所有 remote ✓
- totalJobs=rankSize: step=rankSize，每 job j 仅一轮迭代 i=j（j≠myRank 时轮询 remote j）——
  历史分析曾误判为"不轮询任何 remote"，实际是**逐 job 分摊**而非不轮询

### 方案B 的绕过方式

不修改框架，而是分离 teamBarrier（totalJobs=1，正确 CrossDevice）和 AllToAll（totalJobs=rankSize，并行 work partition）。手动调用 `teamBarrier_.CrossDevice()` 而非依赖 `Wait<BARRIER_DEVICE>` 内部的 CrossDevice。

### 仍需框架修改才能实现的

- **框架新增 ReduceScatter PUT 算子**: 全新通信算子，绕过 AllToAll 语义

## 五、v4 复审修订（三轮评审）

### totalJobs=rankSize 在收尾单次调用下验证通过

三轮评审意见 2 要求消除单核串行 CrossDevice 的性能瓶颈。复核尝试 2 的失败条件后发现：
其失败场景是 `Wait<BARRIER_DEVICE>` **每 turn 内嵌调用**且 self-target job 因 DoWait 早退不参与；
而**收尾段一次性调用**（F4→F1'→F2→F3）下，全部 R 个通信 job 无条件进入 CrossDevice，
rendezvous 配对完整（本 rank job j 轮询远端 j 的本 rank 槽位 ← 由远端 j 的 job-myRank 写入）。

- 配置：teamBarrier totalJobs=rankSize + 收尾单次 CrossDevice + win 数据区起点
  alignUp(2*R*32, 128)（barrier 区 = localFlag R*32 + teamSyncCounter R*32；
  旧 winOffset=128 在 R=4 时与 counter 槽位 [128,160) 重叠——旧 R=4 fence 实际处于
  "计数被 PUT 数据覆写→比较值恒通过"的假通过状态，本次一并修复）
- 验证：R=2 / R=4 全量用例 PASS（含 m=4032 tail、T=1、大 M 系列）

### v2 存量 bug：段级 tiling M 不匹配（与本 fence 分析无关，同期修复）

v2 三段式把每 tile 拆成 REMOTE((R-1) 个 fragment)/LOCAL(1 个) 两次调用，但 sub/tail tiling 仍按
旧语义计算（m=headMSize / rankSize*tailMSize），调度器 tail 参数（mTailMain/mBaseTailSplitCnt）
与段实际逻辑 M 不匹配 → tile 形状非法 → 设备侧崩溃（errcode 264 标量 GM 非法 / 5 L0A 越界）。
触发条件：sub/tail tiling 存在非平凡 M-tail 结构（m=4032 系列必触发）。v4 按段实际逻辑 M 计算
tiling（REMOTE (R-1)*tileM / LOCAL tileM）修复。

### v2 存量 bug 2：reduceSum N 分段 slot 布局随段收缩重叠（k/n 族失败根因）

**现象**: 全量 101 case 中 22 例失败，全部集中在 n=16384（任意 k）与 n=10240×R=2（特定 k/m）；
v2/v3 代码均失败，v1（ed7d74662, 97 用例基线）全部通过。错误窗口精确锁定：n=16384 时
坏列恒为 **[7856, 8528) = [alignedN−maxSegLen, maxSegLen)**，且坏值 ≈ 单 rank 输入数据幅值。

**根因**: reduceSum（405f8e6f7 重写版）的 6-slot UB 布局按**当前段 segLen** 重算。n 分段为
(8528, 7856) 时尾段收缩，其 inIO slot 起始字节 4×7856=31424B 落入前段 sum slot
[0, 4×8528=34112B) 内，重叠 672 元素；且下段输入 DataCopy（MTE2 写）与上段输出 DataCopy
（MTE3 读）之间无管道排序 → 竞态以前段输入数据覆写前段输出尾部 672 列。
v1 不触发纯属几何巧合：其 180KB 预算 → maxSegLen=7680，n=16384 分段 (7680,7680,1024)，
等长段间无重叠；v2 预算 200KB → maxSegLen=8528，分段 (8528,7856) 出现收缩尾段。
失败面随 (n,k,m) 漂移是竞态本质（MTE3/MTE2 完成时序随规模变化），非确定性边界。

**修复**: slot 布局按 maxSegLen 固定于分段循环外计算，全部段共用同一布局（互不重叠）；
固定布局下 inIO/inSum/out slot 与 sum slot 天然不相交，无需引入额外 MTE3→MTE2 排序。

当前方案B 通过分离 totalJobs 绕过了框架限制，是当前框架下的最优解。
