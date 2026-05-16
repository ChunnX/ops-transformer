# MegaMoe Refine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 `bs=1/4/8` 精度通过的前提下，尽可能精炼 `mega_moe` 的流程代码，删掉冗余调试/重复逻辑，并把 dispatch 路由扫描从“重复全表扫描”优化成“预构建紧凑索引后直接消费”。

**Architecture:** 这次优化分三层推进。第一层先冻结验收基线和现有精度，避免“越改越不知道坏在哪”。第二层做代码瘦身：删除大批临时 debug 打印、压缩脚本维测、清理备份残留，让主流程回到必要最少。第三层做算法级优化：把 `BuildRoutesFromSharedTopkIds()` 从“只统计计数”提升成“同时构建 per-(srcRank, localExpert) 的 flatIdx 索引”，让 `DispatchTokens()` 直接消费 route list，而不是每个 expert 再扫一遍完整 `topk_ids`。

**Tech Stack:** AscendC / C++ kernel, Python test harness, `build.sh`, `mega_moe.py`, `graph.py`, NPU runtime

---

### Task 1: 冻结基线与验收入口

**Files:**
- Modify: `/home/0506/moe_test/mega_moe/mega_moe.py`
- Create: `/home/0506/dispatch_combine/ops-transformer/docs/knowledge_base/mega_moe_refine_checklist.md`

- [ ] **Step 1: 写入最小验收清单文档**

```md
# MegaMoe Refine Checklist

## Required pass cases
- bs=1
- bs=4
- bs=8

## Required signals
- `[INFO] tensor x output 精度结果为：PASS`
- `tensor expert_token_nums output测试的精度结果为：('PASS', 100.0)`
```

- [ ] **Step 2: 让脚本支持一次性切换 batch，而不是手改全局常量**

```python
import os

bs = int(os.environ.get("MEGA_MOE_BS", "8"))
```

- [ ] **Step 3: 运行单个用例验证入口可控**

Run:
```bash
MEGA_MOE_BS=8 python /home/0506/moe_test/mega_moe/mega_moe.py
```
Expected: 脚本打印 `bs=8`，并正常跑完。

- [ ] **Step 4: 记录 refine 前的基线结果**

Run:
```bash
for bs in 1 4 8; do
  MEGA_MOE_BS=$bs python /home/0506/moe_test/mega_moe/mega_moe.py | tee /tmp/mega_moe_bs${bs}_baseline.log
done
```
Expected: 三个 batch 都保留原始 PASS/FAIL 日志。

- [ ] **Step 5: Commit**

```bash
git add /home/0506/moe_test/mega_moe/mega_moe.py /home/0506/dispatch_combine/ops-transformer/docs/knowledge_base/mega_moe_refine_checklist.md
git commit -m "chore: add mega moe refine validation baseline"
```

### Task 2: 删掉脚本层冗余维测，保留可控开关

**Files:**
- Modify: `/home/0506/moe_test/mega_moe/mega_moe.py`
- Modify: `/home/0506/moe_test/mega_moe/graph.py`

- [ ] **Step 1: 为脚本调试打印引入统一开关**

```python
DEBUG_VERBOSE = os.environ.get("MEGA_MOE_DEBUG_VERBOSE", "0") == "1"

if DEBUG_VERBOSE:
    print_tensor_diff_summary(...)
```

- [ ] **Step 2: 将 `graph.py` 中临时 raw-byte 打印收敛到 debug 开关下**

```python
if DEBUG_VERBOSE and rank < 2:
    print(f"[trace][rank {rank}] dispatch_expand_x_raw_head={expand_x_cpu.view(torch.uint8)}")
```

- [ ] **Step 3: 将 `mega_moe.py` 中大段 trace/debug 输出折叠为必要最少输出**

```python
if DEBUG_VERBOSE:
    print(f"[trace][rank {rank}] cascade forward start")
```

- [ ] **Step 4: 跑一次 debug-off 用例确认日志明显收缩且结果不变**

Run:
```bash
MEGA_MOE_DEBUG_VERBOSE=0 MEGA_MOE_BS=8 python /home/0506/moe_test/mega_moe/mega_moe.py
```
Expected: 只保留核心 compare 结果与必要 summary。

- [ ] **Step 5: 跑一次 debug-on 用例确认仍能调试**

Run:
```bash
MEGA_MOE_DEBUG_VERBOSE=1 MEGA_MOE_BS=8 python /home/0506/moe_test/mega_moe/mega_moe.py
```
Expected: 仍能看到维测与 raw-byte 输出。

- [ ] **Step 6: Commit**

```bash
git add /home/0506/moe_test/mega_moe/mega_moe.py /home/0506/moe_test/mega_moe/graph.py
git commit -m "refactor: gate mega moe debug traces"
```

### Task 3: 清理 kernel 中的大批临时 `PRINTF`，只保留最小诊断点

**Files:**
- Modify: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h`
- Modify: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_combine_send.h`

- [ ] **Step 1: 在 kernel 侧引入统一 debug 宏**

```cpp
#ifndef MEGA_MOE_DEBUG
#define MEGA_MOE_DEBUG 0
#endif

#if MEGA_MOE_DEBUG
#define MEGA_MOE_PRINTF(...) PRINTF(__VA_ARGS__)
#else
#define MEGA_MOE_PRINTF(...)
#endif
```

- [ ] **Step 2: 把高频打印切换成 `MEGA_MOE_PRINTF`**

```cpp
MEGA_MOE_PRINTF("[AIV DispatchTripletWrite] ...\n", ...);
```

- [ ] **Step 3: 删除明显只为本轮排障临时存在的收尾打印**

```cpp
// remove: [BeforeGMM2Loop] / [AfterGMM2Loop] / [AfterGMM2SyncAll] / [MegaMoe Process End]
```

- [ ] **Step 4: 保留真正有价值的最小诊断点**

```cpp
// keep only
// 1. QuantPacketHead
// 2. RecvTokenPerExpert
// 3. DispatchTripletWrite (debug only)
// 4. CombineTriplet / CombineWrite (debug only)
```

- [ ] **Step 5: 编译确认没有后端因打印触发的新问题**

Run:
```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe
```
Expected: 编译通过。

- [ ] **Step 6: Commit**

```bash
git add /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_combine_send.h
git commit -m "refactor: trim mega moe kernel debug prints"
```

### Task 4: 把 dispatch 路由从“重复全表扫描”优化成“预构建紧凑索引”

**Files:**
- Modify: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h`
- Modify: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_base.h`
- Possibly Modify: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_workspace_info.h`

- [ ] **Step 1: 为 route list 定义清晰的物理表示**

```cpp
struct RouteEntry {
    int32_t flatIdx;   // tokenIdx * topK + topkIdx
};
```

如果不新增 struct 文件，也至少明确两段区：

```cpp
// routeOffsets[srcRank][expert]
// routeFlatIdxList[packed_rows]
```

- [ ] **Step 2: 在 workspace 或 peermem 中预留 route index 存储区**

```cpp
// pseudo layout
ptrRouteOffsets = ...;   // size = worldSize * expertPerRank + 1
ptrRouteFlatIdx = ...;   // size = m * topK * worldSize
```

- [ ] **Step 3: 改造 `BuildRoutesFromSharedTopkIds()`，从“只计数”升级为“计数 + 写 routeFlatIdx”**

```cpp
for (tokenIdx ...)
  for (topkIdx ...)
    if (remoteTopkIdsGm.GetValue(flatIdx) == globalExpertIdx) {
        int32_t dst = routeWriteOffset[srcRankIdx][expertIdx]++;
        routeFlatIdxList[dst] = flatIdx;
        tokenPerExpertWin_.SetValue(... + 1)
    }
```

- [ ] **Step 4: 改造 `DispatchTokens()`，不再扫描完整 `topk_ids`，而是直接遍历 route list**

```cpp
int32_t routeBegin = routeOffsets[srcRankIdx][expertIdx];
int32_t routeEnd = routeOffsets[srcRankIdx][expertIdx + 1];
for (int32_t routePos = routeBegin + rowSrcOffsetPerCore;
     routePos < min(routeEnd, routeBegin + rowSrcOffsetPerCore + rowsPerCore);
     ++routePos) {
    int32_t flatIdx = routeFlatIdxList[routePos];
    uint32_t tokenIdx = flatIdx / topK_;
    uint32_t topkIdx = flatIdx % topK_;
    CopyGMToGMPerTokenV2(... gmRemoteA[flatIdx * copyInNum] ...);
    WriteGmByPassDCache(... EncodeTriplet(srcRankIdx, tokenIdx, topkIdx));
}
```

- [ ] **Step 5: 删除 `DispatchTokens()` 中对 `remoteTopkIdsGm` 的重复扫描逻辑**

```cpp
// remove nested loops over tokenIdx/topkIdx from DispatchTokens
```

- [ ] **Step 6: 跑 focused 验证，先看 `expert_token_nums` 是否仍正确**

Run:
```bash
MEGA_MOE_BS=8 python /home/0506/moe_test/mega_moe/mega_moe.py
```
Expected: `tensor expert_token_nums output测试的精度结果为：('PASS', 100.0)`。

- [ ] **Step 7: Commit**

```bash
git add /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_base.h /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_workspace_info.h
git commit -m "perf: precompute mega moe dispatch route lists"
```

### Task 5: 精炼 `Process()` 主流程，把阶段边界变清楚

**Files:**
- Modify: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h`

- [ ] **Step 1: 提取前置阶段 helper，压缩 `Process()` 长度**

```cpp
__aicore__ inline void PrepareDispatchInputs()
{
    PrepareIdentityExpandedRowIdx();
    QuantizeInputTokenMajor();
    ResetTripletReadyState();
    ShareTopkIdsToPeermem();
}
```

- [ ] **Step 2: 提取 dispatch+gmm1 阶段 helper**

```cpp
__aicore__ inline void RunDispatchAndGmm1(GMMAddrInfo &gmmAddrInfo)
{
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        ...
    }
}
```

- [ ] **Step 3: 提取 gmm2+combine 阶段 helper**

```cpp
__aicore__ inline void RunGmm2AndCombine(GMMAddrInfo &gmmAddrInfo)
{
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        ...
    }
}
```

- [ ] **Step 4: 把 `Process()` 收缩成阶段调度函数**

```cpp
PrepareDispatchInputs();
InitWorkspaceAndCounters();
CrossRankSyncCntPerExpert();
RunDispatchAndGmm1(gmmAddrInfo);
RunGmm2AndCombine(gmmAddrInfo);
FinalizeOutput();
```

- [ ] **Step 5: 编译并确认没有行为回归**

Run:
```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe
```
Expected: 编译通过。

- [ ] **Step 6: Commit**

```bash
git add /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h
git commit -m "refactor: split mega moe process into clear stages"
```

### Task 6: 清理临时备份和无用残留

**Files:**
- Delete: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h.bak_agent`
- Delete: `/home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_base.h.bak_agent`
- Possibly Delete: `/home/0506/moe_test/mega_moe/mega_moe.py.bak_agent`

- [ ] **Step 1: 先确认这些备份文件不再参与构建与调试**

Run:
```bash
grep -R -n 'bak_agent' /home/0506/dispatch_combine/ops-transformer /home/0506/moe_test/mega_moe
```
Expected: 没有构建链依赖。

- [ ] **Step 2: 删除备份文件**

Run:
```bash
rm -f /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe.h.bak_agent
rm -f /home/0506/dispatch_combine/ops-transformer/mc2/mega_moe/op_kernel/mega_moe_base.h.bak_agent
rm -f /home/0506/moe_test/mega_moe/mega_moe.py.bak_agent
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove mega moe temporary backup files"
```

### Task 7: 端到端回归验证

**Files:**
- No code changes required

- [ ] **Step 1: 重新编译**

Run:
```bash
bash build.sh --pkg --soc=ascend950 --ops=mega_moe
```
Expected: SUCCESS。

- [ ] **Step 2: 重新安装**

Run:
```bash
./build_out/cann-ops-transformer-custom_linux-x86_64.run --force
```
Expected: `SUCCESS`。

- [ ] **Step 3: 清卡**

Run:
```bash
npu-smi info
# kill all conflicting processes if any
npu-smi info
```
Expected: 所有 NPU 无冲突进程。

- [ ] **Step 4: 回归 `bs=1`**

Run:
```bash
MEGA_MOE_BS=1 python /home/0506/moe_test/mega_moe/mega_moe.py
```
Expected: `tensor x output 精度结果为：PASS`

- [ ] **Step 5: 回归 `bs=4`**

Run:
```bash
MEGA_MOE_BS=4 python /home/0506/moe_test/mega_moe/mega_moe.py
```
Expected: `tensor x output 精度结果为：PASS`

- [ ] **Step 6: 回归 `bs=8`**

Run:
```bash
MEGA_MOE_BS=8 python /home/0506/moe_test/mega_moe/mega_moe.py
```
Expected: `tensor x output 精度结果为：PASS`

- [ ] **Step 7: 输出精炼收益总结**

```md
- 删除了哪些调试残留
- `Process()` 缩短了多少
- `DispatchTokens()` 删除了多少重复扫描
- 精度回归结果如何
```

- [ ] **Step 8: Commit**

```bash
git add /home/0506/dispatch_combine/ops-transformer
git commit -m "refactor: streamline mega moe dispatch pipeline"
```

---

## Self-Review

### Spec coverage

已覆盖用户提出的全部目标：

- 删冗余：Task 2 / Task 3 / Task 6
- 算法优化：Task 4
- 精炼主流程：Task 5
- 精度通过：Task 1 / Task 7

### Placeholder scan

本计划没有使用 `TODO`、`之后补上`、`适当处理` 这类空占位语句；每个任务都给出了明确文件、示例代码或具体命令。

### Type consistency

计划中统一使用这些命名：

- `routeOffsets`
- `routeFlatIdxList`
- `flatIdx`
- `QuantizeInputTokenMajor`
- `DispatchTokens`

后续实现时应保持这些命名一致，避免 route index 相关变量在不同任务里漂移。

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-mega-moe-refine.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
