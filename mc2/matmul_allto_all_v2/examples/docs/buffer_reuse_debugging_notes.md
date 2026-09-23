# matmul_allto_all_v2 Win 区 Buffer 复用调试记录

> 记录 buffer 复用功能开发与调试过程中遇到的精度/性能问题、根因分析与解决方案，
> 供后续维护与类似算子（通算融合类）开发参考。

## 0. 复用设计概述

Win 区（HCCL 通信窗口）预算 400MB（`WIN_REUSE_BUDGET_CAP`）。当 matmul 结果总量
`m × n × 2 > 400MB` 时启用复用：每轮次（tile）的搬运数据占一个 slot，
`slotNum = min(budgetDepth, totalTileCnt, WIN_REUSE_DEPTH_CAP)`，轮次超过 slotNum 后
地址与 flagId 回绕复用。同步链：

```
AIC(圈k, slot s):  [复用圈] Wait<0x2, PIPE_M>(s) ← AIV 通知  → 写 slot → Set<0x2, PIPE_FIX>(s)
AIV(圈k, slot s):  Wait<0x2, PIPE_S>(s) ← AIC 数据就绪 → GET(Drain) → [被覆盖] CrossDevice+SyncAll → Set<0x2, PIPE_MTE3>(s)
```

配套参数：`slotNum`（host tiling 计算）、`BufferChannel`（kernel 侧环形分配器，
`offset = raw × slotSize`、`flagId = raw % FLAG_ID_MODULO(11)`）。

---

## 1. 精度问题

### 1.1 三处 slot 粒度不一致（数据错位，确定性 fail）

- **现象**：合入复用代码后，原本精度 pass 的用例（m=2048）立即 fail，输出中 tile≥1 段全部错位。
- **根因**：写/读/分配器三处 slot 步进粒度不一致：
  - AIC 写：`slotIdx × bufferSlotBytes`，其中 `bufferSlotBytes = rankSize × tileChunkBytes`
  - GET 读：`bufferOffset = raw × slotSize_`，`Init` 传入的 `slotSize_` 是**单 rank chunk**
    （缺 rankSize 倍）
  - 旧逻辑两侧均为 `raw × bufferSlotBytes`
- **后果**：ws=4 时 tile1 起 AIV 读地址比 AIC 写地址少 3×chunk，读到错位数据；tile0 因偏移
  均为 0 侥幸正确。
- **解决**：统一粒度——`BufferChannel.Init` 的 `slotSize` 传 `rankSize × tileChunkBytes`；
  AIC 写地址改用 `curSlot.offset`（与 GET 的 `bufferOffset` 同源）。
- **经验**：环形缓冲的 slot 粒度必须与"该 slot 的实际占用"（此处为 rankSize 份 chunk）
  一致；`offset`（地址）与 `flagId`（同步）是两个独立回绕域，寻址永远用前者。

### 1.2 复用等待/通知条件恒为 false

- **现象**：修复 1.1 后复用圈仍存在数据竞争风险（精度 flaky）。
- **根因**：等待/通知条件写成 `curSlot.slotIdx >= GetSlotNum()`，而
  `slotIdx = raw % FLAG_ID_MODULO(11)`：slotNum ≤ 10 时 `slotIdx = raw < slotNum`；
  slotNum > 10 时 `slotIdx ≤ 10 < slotNum`——**恒 false**，AIC 覆盖前从不等待、AIV 从不通知。
- **解决**：改用全局 tile 序号判定——AIC 侧 `tid >= GetSlotNum()`（tid ≥ slotNum 即
  GetNextSlot 已进入回绕圈）；AIV 侧 `tid + GetSlotNum() < totalTileCnt`（本 slot 存在
  下一圈使用者才通知）。
- **经验**：`flagId` 是 `%11` 之后的同步编号，**不能**用于"是否回绕"的逻辑判断；回绕
  判定必须基于不回绕的全局计数（tid 或 raw）。

### 1.3 AIC 覆盖等待缺少显式 0x2 mode

- **根因**：`CrossCoreWaitFlag(slotIdx)` 走默认模板 `<modeId=0, PIPE_S>`——mode 0 是
  **单核内 pipe 间 event** 语义，与 AIV 侧 `CrossCoreSetFlag<0x2, PIPE_MTE3>`（跨核 0x2）
  不配对。该缺陷被 1.2 掩盖（条件恒 false 从未执行，表现为精度错而非 hang）。
- **解决**：显式 `CrossCoreWaitFlag<0x2, PIPE_S>(...)`，与 Set 构成 MTE3→S 跨核对。
- **经验**：跨核同步的 Set/Wait 必须成对显式写 mode 与 pipe；两个 bug 叠加时会互相掩盖
  （hang 与精度错的症状切换），排查时要逐个还原条件。

### 1.4 flag 空间约束：slotNum ≤ FLAG_ID_MODULO(11)

- **根因**：硬件 flagId 空间为 0~10（11~14 被 SyncAll 占用），`flagId = raw % 11`。
  slotNum > 11 时 flag 回绕周期（11）与 slot 回绕周期（slotNum）不一致：AIC 覆盖 slot s
  前的 Wait 可能被其他 slot（s±11）的消费通知放行 → 数据竞争（flaky）。
- **解决**：host 侧 `slotNum` 计算后 cap 至 FLAG_ID_MODULO；`WIN_REUSE_DEPTH_CAP=128`
  作为预留的策略上限（当前实际生效上限为 11，扩 flag 方案后再放开）。
- **经验**：参考实现（yf_mte_experiment）DESIGN 文档明确记录硬件约束：
  **flagId 0~10、同一 flagId 最多 Set 15 次**（计数器宽度有限）。Set/Wait 严格配对时
  计数回落、可无限循环；需保证配对完整且流水领先深度不超过上限。

### 1.5 伪复用：深度 cap 位置错误

- **现象**：空间足够（totalWin ≤ 400MB）但 tileCnt > 128 的场景，出现 128 次无谓的
  SyncAll/Wait（性能劣化）且 slotNum=128 > 11（精度风险）。
- **根因**：`slotNum = min(slotNum, WIN_REUSE_DEPTH_CAP)` 写在**两分支汇合处**——不复用
  分支（slotNum = totalTileCnt）也被 cap 压到 128 < totalTileCnt，`slotNum < totalTileCnt`
  被误判为"复用启用"。
- **解决**：cap 移入复用分支内部；不复用分支 `slotNum = totalTileCnt` 保持不变。此后
  `slotNum < totalTileCnt` 与"真复用"严格等价，AIV/AIC 的逐 tile 条件自动精确生效。
- **经验**：派生开关类变量（slotNum < totalTileCnt ⟺ 复用）的分支 cap 必须限定在
  产生该语义的分支内，否则会反向污染判定条件。

### 1.6 跨 rank 消费确认缺失（0x2 硬件语义，最核心）

- **现象**：复用覆盖存在跨 rank 数据竞争，理论推演即可确认（未依赖实测）。
- **根因**：**0x2 硬件 flag 是"单个 AI Core 内 C 核 ↔ V 核"同步**（950 架构 1C+2V），
  不是全 device/rank 聚集。数据流决定覆盖条件：

  ```
  rank r 的 AIC 写 rank r 的 Win 区 slot → 消费者是其他 rank 的 AIV（各拉自己的 chunk）
  → rank r 覆盖 slot 前需要"所有远端 rank 的 AIV 完成 GET"的确认
  ```

  而原通知链 `SyncAll<true>`（rank 内硬同步）+ `SetFlag<0x2>`（core 内）只能证明
  rank r 自己的 AIV 完成了"从远端读"——与"远端读完了 rank r 的 Win"是两个独立事件，
  互不证明。参照工程（yf_mte_experiment）用 GET 后的**第二次跨 rank barrier**
  （`aclshmemx_barrier_all_vec`）补齐该语义。
- **解决**：通知侧两级确认——
  `teamBarrier_.CrossDevice()`（跨 rank：所有 rank 通信核都 Drain 完成）
  → `SyncAll<true>()`（rank 内：把全局完成信息分发给非通信核，它们不参与 CrossDevice）
  → `SetFlag<0x2, PIPE_MTE3>`。**CrossDevice 必须放在 SyncAll 之前**：SyncAll 通过即
  通信核已过 CrossDevice，非通信核的 Set 因此也携带全局完成信息，保证非通信 Core 的
  C 核（blockIdx ≥ rankSize，同样写 Win、同样有复用等待）的覆盖同样受保护。
- **经验**：
  1. 0x2（core 内 C↔V）、TeamBarrier 软件 barrier（V 核间/跨 rank）、SyncAll（rank 内
     全 AIV）三者作用域完全不同，混用时必须画清"信息从谁流到谁"；
  2. 通知的信息量 = 通知者到达通知点前掌握的事实——消费者完成确认必须由
     **GET 完成之后的跨 rank barrier** 产生，GET 发起前的 barrier（Commit 内的
     CrossCore/CrossDevice，"生产者就绪"语义）不能替代；
  3. 排查这类问题时，参照同类实现（yf 工程两次 barrier_all_vec 的结构）是最快路径。

---

## 2. 性能问题

### 2.1 逐 tile Drain 破坏 GET 流水化（实测劣化 15us）

- **现象**：合入复用代码后，未触发复用的场景（m=2048，totalWin 12/21MB ≪ 400MB）
  性能劣化约 15us。
- **根因**：`Wait(true)`（仅最后 tile 排空，读请求跨 tile 积累、传输被后续 tile 隐藏）
  改成了 `Wait()`（每 tile 强制 Drain）——每个 tile 末尾插入硬同步点，暴露 barrier +
  ReadNbi 下发 + DMA 完成通知的尾延迟；4 tile × 3~5us ≈ 15us，与实测吻合。
- **解决**：按 tile 是否会被覆盖区分 Drain 深度——
  ```cpp
  if (tid + GetSlotNum() < totalTileCnt) { allToAll_.Wait(); }      // 会被覆盖：排空保证通知有效
  else { allToAll_.Wait(true); }                                     // 尾段：流水化 + 末 tile 收尾
  ```
  不复用场景与旧代码逐位一致，15us 完全回归；复用场景仅覆盖链路上的 tile 串行化。
- **经验**：为正确性引入的同步必须限定在"需要它的最小 tile 集合"上；流水化通信
  （nbi 发起 + 延迟汇聚等待）的排空点数量直接决定性能。

### 2.2 伪复用引发的 SyncAll 开销

见 1.5——同一个 bug 的性能面。修复后"复用才进 SyncAll/Wait"由
`slotNum < totalTileCnt` 严格保证，不复用场景零额外同步。

### 2.3 真复用场景 SyncAll 的进一步优化（待验证）

SyncAll 的作用是把全局 GET 完成信息分发给非通信核（它们不参与 CrossDevice）。理论上
0x2 若为计数式聚集则非通信核提前 Set 无害、SyncAll 可省——但已确认 0x2 为 core 内
C↔V 语义、SyncAll 是 rank 内硬同步，该优化不成立，保留两级确认结构。后续如需削减，
方向是扩大 CrossDevice 参与集（TeamBarrier totalJobs 扩至全体 AIV + UB flag 区扩容），
而非删除 SyncAll。

---

## 3. 工具与方法论问题

### 3.1 profiler 插桩污染计时基线

- **现象**：性能数字与 ST 直调工程差距大（173us vs 278us），且波动。
- **根因**：profiler 采集脚本（Level1 + AIC metrics + profile_memory）无条件包裹计时
  循环；MC2 算子 AIC↔AIV / rank 间同步链对插桩扰动极其敏感，4 rank 下发节奏不一致
  被同步放大。
- **解决**：profiler 采集与 Event 计时基线分离——性能脚本 `--profile` 显式开启采集
  （默认关闭）；基准数字只认无插桩的 Event 计时；trace 只用于结构归因（占比/流水）。
- **经验**：MC2 类同步耦合算子的 profiler 数据只看结构不看绝对值。

### 3.2 extract_profile 提取 no matching data

- **现象**：`extract_profile.py <dir> MatmulAlltoAllV2 10 30` 报 no matching data。
- **根因**：profiler 的 Step Id 由 `prof.step()` 推进；采集脚本 step() 在循环外只调用
  一次，csv 中 Step Id 全为 0，`[10, 30]` 过滤必然空匹配。（torch 原生接口有自动关联，
  aclnn 直调路径没有，两边行为不同易误判。）
- **解决**：`prof.step()` 移入循环内逐次调用；Step Id 0~loop-1 与调用次数一一对应
  （step 0 为 profiler warmup）。

### 3.3 ST 直调 vs torch extension 的性能口径

ST 工程（C++ 直调 kernel、tiling 算一次、无 aclnn 框架）测得的是 device 纯执行时间；
torch 路径每次调用走完整 aclnn 两段式（GetWorkspaceSize 重跑 tiling + executor + workspace
分配 + python 层通信上下文准备），host 单次开销可能超过 device 执行时间，Event 计时
口径下测得的是 host 节拍。两者对比前必须先统一口径（profiler trace 中看 kernel 间隔
与 duration 即可区分 host bound）。

---

## 4. 复用验证用例（边界值）

| # | 验证目标 | m | n | tileCnt | slotNum | 回绕数 |
|---|---|---|---|---|---|---|
| 1 | 最小回绕（tileCnt = budgetDepth+1） | 5632 | 37376 | 11 | 10 | 1 |
| 2 | 分界对偶·不复用侧 | 5120 | 37376 | 10 | 10 | 0 |
| 3 | totalWin 精确 = CAP（`<=` 分支） | 4096 | 51200 | 8 | 8 | 0 |
| 4 | slotNum = FLAG_ID_MODULO(11) | 6144 | 37236 | 12 | 11 | 1 |
| 5 | depth cap 恰好咬合（tileCnt=128） | 65536 | 3072 | 128 | 128 | 0 |
| 6 | 伪复用回归（空间足够 tileCnt>128） | 66048 | 2048 | 129 | 129 | 0 |
| 7 | 极小 tail tile（m%512=1） | 6145 | 37376 | 13 | 10 | 3 |
| 8 | ⚠️ slotNum=12>11（flag 错配风险，预期 FAIL） | 6656 | 34112 | 13 | 12 | 1 |

全部 k=4096、ws=4。运行方式与 CPU/NPU 内存注意事项见
`scripts/test_golden_matmul_alltoall.py` 头部说明。#8 用于确认问题边界，flag 方案
扩展后回归。

---

## 5. 检查清单（合入复用类逻辑前自查）

1. slot 粒度三方一致：Init 的 slotSize == AIC 写步进 == GET 读步进
2. 回绕判定用不回绕的全局计数（tid/raw），不用 flagId
3. 跨核 Set/Wait 显式成对（mode + pipe），等待方语义与通知携带的信息量匹配
4. flag 空间约束：slotNum ≤ FLAG_ID_MODULO；同一 flagId 的 Set 累计次数 ≤ 15
   （长迭代场景确认 Set/Wait 配对完整）
5. 覆盖放行的通知必须携带"所有 rank 消费完成"——core 内 0x2 通知前需
   跨 rank barrier（CrossDevice）+ rank 内分发（SyncAll），且 CrossDevice 在前
6. Drain/排空的粒度 = "会被覆盖的 tile"集合，其余保持流水化
7. 分支 cap（深度/预算）不得作用于不复用分支，防止伪复用
8. profiler 采集与计时基线分离；多迭代压力验证 flag 长期配对与竞争消除
