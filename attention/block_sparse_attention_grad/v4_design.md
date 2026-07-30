# 确定性方案 v4：分批归约 + per-task 槽位（设计文档）

## 与 v3 的区别

| | v3 (当前) | v4 (目标) |
|---|---|---|
| 槽位索引 | (coreIdx, qHeadInGroup) | (coreIdx, taskIdxInBatch) |
| atomicAdd | 需要（跨 Q-block 累加） | 不需要（每 task 独立槽位） |
| 归约次数 | 1 次（全部 task 完成后） | ceil(taskLength / K) 次 |
| SyncAll | 1 次 | 每批一次 |
| workspace | aicNum × groupSize × dkvSize | aicNum × K × dkvSize |
| K (槽数/核) | groupSize (4~16) | ceil(40/aicNum) = 2 |

## 数据流

```
VecPre (AIV):
  ├── 清零 shared dQ/dK/dV workspace
  └── 清零 detDkWorkspace + detDvWorkspace (aicNum × K × dkvSize)

AIC (按批次处理):
  for batch = 0 .. numBatches-1:
    for taskInBatch = 0 .. K-1:
      ├── blockMmad1: S, dP
      ├── CUBE2VEC / VEC2CUBE (与 AIV 同步 softmax)
      ├── blockMmad2: dQ → shared workspace (不变)
      ├── blockMmad3: dV → detDvWorkspace[coreIdx × K + taskInBatch] (isAtomicAdd=false)
      └── blockMmad3: dK → detDkWorkspace[coreIdx × K + taskInBatch] (isAtomicAdd=false)
    CrossCoreSetFlag(CUBE2BATCH)  → 通知 AIV 归约本批
    SyncAll                       → 等所有核完成本批
    WaitEvent(BATCH2CUBE)         → 等 AIV 归约完成

AIV:
  VecPre → VecOp (per task, same as v3)
  for batch = 0 .. numBatches-1:
    WaitEvent(CUBE2BATCH) + SyncAll
    VecDeterReduce: 读 aicNum×K 个槽位 → 累加到 shared dK/dV
    ZeroDetWorkspace: 清零 per-task workspace
    CrossCoreSetFlag(BATCH2CUBE) → 通知 AIC 继续
  VecPost
```

## 为什么能消除 atomicAdd

v3: 同一 Q-head 的不同 Q-block 写入同一槽位 → 需 atomicAdd
v4: 每个 task (Q-block + Q-head) 有独立槽位 → 无重叠 → isAtomicAdd=false

每个 task 内, KV block 循环写不同 offset (不同 KV 位置) → 无重叠
不同 task 写不同槽位 → 无冲突
不同核写不同槽位 → 无冲突

## Workspace

```
detDkWorkspace = aicNum × K × dkvSize × sizeof(float)
detDvWorkspace = aicNum × K × dkvSize × sizeof(float)
K = ceil(batchSize / aicNum), batchSize=40, aicNum=20 → K=2
```

| 场景 | v3 workspace | v4 workspace | 缩减 |
|------|-------------|-------------|------|
| g=4, B=2, S=128 | 40 MB | 20 MB | 2× |
| g=8, B=8, S=512 | 2.56 GB | 640 MB | 4× |
| g=8, B=32, S=2048 | 20.5 GB | 5.12 GB | 4× |

## 实现复杂度

主要挑战：AIC 循环的 ping-pong 机制
- 当前: task N 的 dK/dV 在 task N+1 的迭代中计算
- v4: 批末需 drain pipeline (处理最后一个 task 的 dK/dV), 再 SyncAll

## 切换方式

同 v3: `export BSAG_DETERMINISTIC=1`
