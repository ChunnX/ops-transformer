# MegaMoe A8W8 Dispatch/Combine PIPE_ALL 粗筛设计

日期：2026-07-31

## 目标

针对 `mega_moe_layered` 的 A8W8 复现问题，在 Dispatch 和 Combine 的可疑流水边界大面积加入
`PipeBarrier<PIPE_ALL>();`，先判断本地流水排序是否会影响问题，再通过分组移除逐步定位。

本轮是临时诊断，不作为最终同步修复。

## 范围与约束

- 仅新增 `PipeBarrier<PIPE_ALL>();` 及对应的 `SYNC_DIAG` 标签注释。
- 不新增或移动 `hcomm_.Drain()`，不改变通信完成协议。
- 不添加 A5 不需要的纯 V→V 同步。
- 不在轮询循环、`SetFlag/WaitFlag` 握手内部或 AIC 自动同步区域插入 barrier。
- 保留当前 `ResetGmm2CombineSyncCounters` 中已有的 `PipeBarrier<PIPE_ALL>();`，本轮不修改其他逻辑。
- 涉及文件：
  - `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h`
  - `mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h`

## 初始插点

### Dispatch 组

| 标签 | 插入位置 | 诊断边界 |
| --- | --- | --- |
| `SYNC_DIAG[D1]` | `QuantTokenToLocalRelay` 中 `MTE3_S` 后、写 relay READY 前 | relay payload 写回 → READY 发布 |
| `SYNC_DIAG[D2]` | `DispatchComputeToken` 中写 `tokenIdx` 后、写 slot READY 前 | slot metadata → READY 发布 |
| `SYNC_DIAG[D3]` | `DispatchComputeKernel` 中 token 循环结束后、写 `donePtr` 前 | 所有 slot 发布 → round DONE |
| `SYNC_DIAG[D4]` | `LoadTokenFromLocalRelay` 中观察到 READY 后、`DataCopy` 前 | READY 观察 → relay payload 读取 |
| `SYNC_DIAG[D5]` | `MetaInfoCalAndDispatch` 中当前 round 的 token/meta 搬运完成后、发布 GMM1 `AtomicAdd` 前 | dispatch 数据写回 → GMM1 flag 发布 |

### Combine 组

| 标签 | 插入位置 | 诊断边界 |
| --- | --- | --- |
| `SYNC_DIAG[C1]` | `ProcessCombineBatch` 中 triple `DataCopy` 后、调用 `CombineTokenGroup` 前 | triple 搬入 → Combine 消费 |
| `SYNC_DIAG[C2]` | `CombineTokenGroup` 中量化 `V_S` 后、调用 `CombineSendTokenToRemote` 前 | 量化完成 → staging/send |
| `SYNC_DIAG[C3]` | `CombineSendTokenToRemote` 中 `MTE3_S` 后、远端 `WriteNbi` 前 | staging 写回 → Hcomm 读取源地址 |
| `SYNC_DIAG[C4]` | `ProcessCombineBatch` 中 `CombineTokenGroup` 返回后、更新 batch 进度前 | 当前 batch 发送发起 → UB/进度复用 |

## 诊断步骤

1. 一次性启用 D1～D5 和 C1～C4，运行原 A8W8 复现用例。
2. 如果问题仍存在，说明这些本地流水 barrier 未覆盖根因；不得据此排除 Hcomm completion 问题，因为
   `PIPE_ALL` 不等价于 `hcomm_.Drain()`。
3. 如果问题消失，先整体移除 Dispatch 组 D1～D5：
   - 问题仍不出现：继续定位 Combine 组。
   - 问题重新出现：恢复 Dispatch 组并整体移除 Combine 组进行反向确认。
4. Dispatch 组内先二分为 D1～D3（producer 发布）和 D4～D5（consumer/GMM1 发布），然后继续单点缩减。
5. Combine 组内先二分为 C1～C2 和 C3～C4，然后继续单点缩减。
6. 每次只改变一个分组，并记录编译版本、启用标签和复现结果。若原问题为偶发问题，每个组合需重复运行。

## 判定边界

- “有效”仅表示某个 `PIPE_ALL` 改变了复现结果，不代表该位置就是最终应保留的最小同步。
- C3 后续调用的 `WriteNbi` 是否完成不受 `PIPE_ALL` 保证；即使所有 Combine barrier 均无效，也不能排除
  通信 Drain 缺失。
- 定位到最小标签后，应重新按真实 producer/consumer 流水选择精确事件同步，并删除其余诊断 barrier。

## 验证

- 使用 `rg -n "SYNC_DIAG" ...` 确认新增标签恰好为 D1～D5、C1～C4。
- 检查所有新增调用均位于 AIV 可达路径，且没有插入轮询循环体。
- 运行仓库可用的格式/编译检查；最终以用户原 A8W8 复现用例结果作为诊断依据。
