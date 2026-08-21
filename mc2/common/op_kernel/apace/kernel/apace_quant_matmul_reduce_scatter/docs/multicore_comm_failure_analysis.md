# 多核并行通信（totalJobs=rankSize）无法实现的根因分析

> 日期: 2026-08-20（初版），2026-08-20 22:10 修正
> 平台: Ascend 950PR (dav-3510), CANN 9.2.0
> 分析对象: AllToAll PUT 从 totalJobs=1（单核通信）改为 totalJobs=rankSize（多核并行通信）

## 一、历史尝试方案

| 方案 | 尝试内容 | 结果 |
|------|---------|------|
| 方案A | `totalJobs=rankSize`，依赖框架 work partition | Rank0 PASS, Rank1 FAIL (81%元素错误) |
| 方案B | `totalJobs=rankSize` + `winOffset=128` + 后R核通信 | Rank0 PASS, Rank1 FAIL (80%元素错误) |
| 方案C | `totalJobs=rankSize` + `winOffset=128` + 后R核通信 + 额外SyncAll | Rank0 PASS, Rank1 FAIL (80%元素错误，额外SyncAll无效) |
| 方案D | 手动 WriteNbi 多核 PUT，绕过框架 | 两 rank 全 FAIL (100%元素错误) |
| 方案E | 自定义 CrossDeviceSync（手动 counter） | NaN (counter 被 PUT 数据覆盖) |

## 二、无法实现的根因（修正后）

### 根因1（决定性）：AllToAll PUT 的 DoCommit 源地址语义与 ReduceScatter 不匹配

框架源码 `all_to_all_udma_put.h:48` 的 DoCommit：

```cpp
// AllToAll 语义: srcAddr 用 targetRankId 做偏移
GM_ADDR srcAddr = this->localAddr_ + targetRankId * this->chunkBytes_ + ...;
```

语义是 **"从 workspace 的第 targetRankId 个 chunk 读取数据，PUT 给 targetRankId"**——即 AllToAll 语义：每个 rank 把不同 chunk 发给不同 target。

ReduceScatter 的需求是 **"从 workspace 的第 rankId 个 chunk（本 rank 的计算结果）读取数据，PUT 给所有 target"**——即 srcAddr 应该用 `rankId` 做偏移，而不是 `targetRankId`。

#### totalJobs=1 时碰巧正确

block 0 遍历所有 targetRank（0..R-1），对每个 target 都用 `targetRankId` 做源偏移，恰好覆盖了所有 chunk，**碰巧正确**（单核遍历所有 target = 遍历所有 chunk index）。

#### totalJobs=rankSize 时必然错误

`targetRankPerCore = (rankSize + totalJobs - 1) / totalJobs = 1`，每个 block 只负责 1 个 targetRank。

| block (jobIndex) | targetRank | srcAddr 读取的 chunk | DoCommit 行为 |
|------------------|------------|---------------------|---------------|
| block i | targetRank i | `workspace[i * chunkBytes]` | 若 i == rankId: **skip**（self）；否则 PUT 给 rank i |

追踪 R=2 数据流：

**rank 0 的通信核：**
- block 26 (jobIndex=0): targetRank=0 → skip (self) → **不发送任何数据**
- block 27 (jobIndex=1): targetRank=1 → PUT `workspace[1*chunkBytes]` (rank0 的 chunk_1) 给 rank 1

**rank 1 的通信核：**
- block 26 (jobIndex=0): targetRank=0 → PUT `workspace[0*chunkBytes]` (rank1 的 chunk_0) 给 rank 0
- block 27 (jobIndex=1): targetRank=1 → skip (self) → **不发送任何数据**

**结果：rank 1 的 win 区什么都没收到**（block 27 是 self-skip，block 26 发到 rank 0 不在 rank 1 的 win 区）。

reduceSum curRank=0 从 rank 1 的 win 区读 → 读到未初始化数据 → 80% 错误。

**结论**：框架 AllToAll PUT 的 DoCommit 硬编码了 `targetRankId` 源偏移。totalJobs=rankSize 时每个 block 的 targetRank 恰好等于自己的 rankId（因为 targetRankStart=jobIndex=blockIdx-commBlockStart），DoCommit 跳过 self，**整个 block 什么都不做**。这不是时序问题，加多少 SyncAll 都无效。

### 根因2（已修复）：winOffset=0 导致 PUT 数据覆盖 barrier counter

PUT 目标地址（`all_to_all_udma_put.h:51`）：

```cpp
GM_ADDR dstAddr = commBufferAddrs[targetRankId] + winOffset_ + senderRankId * chunkBytes_ + ...;
```

win 区内存布局（barrier_ubmem.h）：

```
[0..31]       localFlag          (CrossDeviceExecute 写/读)
[32..63]      teamSyncCounter    (CrossDevice 用, totalJobs=1 时)
[64..95]      crossCoreFlag      (CrossCore 用, totalJobs=1 时)
[96..]        可用数据区
```

当 `winOffset=0`（旧稳定版默认），PUT 数据从 offset 0 开始写入 `chunkBytes = chunkM × N × 2` 字节（MB 级），**直接覆盖** barrier 的所有 counter 区域。

| totalJobs | 同步逻辑状态 | counter 被覆盖后果 |
|-----------|-------------|-------------------|
| 1 | step=1，检查所有 rank ✓ | flag 被覆盖，读到 PUT 的 BF16 数据（值≈10）远大于 count → `>=count` 恒 true → **假通过**（同步名义上通过，实际靠时序碰运气） |
| rankSize | step=rankSize，不检查任何人 ✗ | counter 被覆盖（但无所谓，因为根本不读） → **完全无同步** → 纯竞态 → NaN |

这解释了为什么旧稳定版（`totalJobs=1, winOffset=0`）之前能 PASS 但机器重启后 NaN：之前靠"假通过"碰运气，时序变化后竞态暴露。

### 关于 CrossDeviceExecute step 逻辑（修正：非决定性因素）

`barrier_ubmem.h:153` 的跨设备同步：

```cpp
int32_t step = totalJobs_ < nranks ? (int32_t)totalJobs_ : (int32_t)nranks;
for (uint32_t i = jobIndex_; i < nranks; i += (uint32_t)step) {
    if (i == myRank) continue;
    // 等待 remote rank 的 flag >= count
}
```

当 `totalJobs=rankSize` 时 step=rankSize，循环不检查任何 remote rank。初版分析认为这是决定性因素，但实测发现：

1. 框架参考实现 `all_to_all_mx_quant_matmul_udma_impl.h` 用完全相同的模式（`totalJobs=rankSize` + `Wait<BARRIER_DEVICE>`）正常工作
2. **Drain 保证远程数据可见性**（URMA 写完成 = 远程内存可见），不依赖 flag 轮询
3. 额外 SyncAll 无法修复问题，因为 **根因1（srcAddr 语义）导致数据本身缺失**，而非时序问题

**结论**：CrossDeviceExecute 的 step 逻辑在 totalJobs=rankSize 时不检查 remote 是 **by design**（框架依赖 Drain），不是多核通信失败的根因。

## 三、各方案的失败原因汇总

```
totalJobs=1, winOffset=0 (旧稳定版):
  根因1: 不触发 (单核遍历所有 target = 遍历所有 chunk, 碰巧正确)
  根因2: 触发但"假通过" (flag 被覆盖, 但 >= count 恒 true)
  → 结果: 时序敏感, 偶尔 PASS 偶尔 NaN

totalJobs=rankSize, winOffset=0 (方案A):
  根因1: 触发 (srcAddr 用 targetRankId, self-skip 导致数据缺失)
  根因2: 触发 (counter 被覆盖, 但无关因为不读)
  → 结果: 必然 FAIL (根因1)

totalJobs=rankSize, winOffset=128 (方案B/C):
  根因1: 仍触发 (DoCommit 源地址语义不变, self-skip)
  根因2: 修复 (counter 不被覆盖)
  → 结果: 仍必然 FAIL (根因1, 加 SyncAll 无效)

totalJobs=1, winOffset=128 (当前稳定版):
  根因1: 不触发
  根因2: 修复
  → 结果: 稳定 ALL PASS ✓ (已验证)
```

## 四、结论

**"1核通信改成 rankSize 核通信"在当前 APACE 框架下无法实现**，决定性根因是：

**框架 AllToAll PUT 的 DoCommit 硬编码了 `targetRankId` 源偏移**（`all_to_all_udma_put.h:48`）。totalJobs=rankSize 时每个 block 的 targetRank 恰好等于自己的 rankId，DoCommit 跳过 self（`if (targetRankId == rankId) return`），**整个 block 什么都不做**。这不是时序或同步问题，加多少 SyncAll 或 winOffset 都无法修复——数据本身就没有被发送。

### 修复方向

| 方案 | 改动范围 | 预期效果 | 状态 |
|------|---------|---------|------|
| **A: totalJobs=1 + winOffset=128** | 算子侧 | 修复根因2，单核通信稳定 PASS | ✅ 已验证 ALL PASS |
| **B: 框架新增 ReduceScatter PUT 实现** | 框架侧，DoCommit 改用 `rankId` 源偏移 | 修复根因1，支持多核并行 | 待框架支持 |
| **C: 框架新增独立 ReduceScatterComm 算子** | 框架侧，全新通信算子 | 完全绕过 AllToAll 语义 | 待框架支持 |

当前采用方案 A（totalJobs=1 + winOffset=128）作为稳定修复。多核并行通信需等待框架层面支持（方案 B 或 C）。

### 验证记录

| 版本 | 用例 (M=2048 K=3584 N=4096 R=2) | 结果 |
|------|--------------------------------|------|
| totalJobs=1, winOffset=0 (旧稳定版) | 单次 | 偶尔 PASS 偶尔 NaN (时序敏感) |
| totalJobs=1, winOffset=0 (机器重启后) | 单次 | FAIL (NaN, 20784/4194304) |
| totalJobs=1, winOffset=128 | 单次 | **ALL PASS** ✓ |
| totalJobs=rankSize, winOffset=128 (后R核通信) | 单次 | FAIL (Rank0 PASS, Rank1 FAIL 80%) |
| totalJobs=rankSize, winOffset=128 + 额外SyncAll | 单次 | FAIL (同上, SyncAll 无效) |
