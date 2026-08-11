# hybrid-perf 8×1 多轮 continue 死锁调查报告

> 生成日期：2026-08-08
> 分支：`hybrid-perf`（ops-transformer）
> 用途：给外部 AI 辅助定位。本报告只包含事实、代码与实验证据。

> 阅读建议：请先独立阅读并分析 **第 1~4 节（事实与证据）、第 7 节（代码位置）、第 8 节（日志）**，
> 形成自己的结论后，再参考 **第 5、6 节（标注“仅供参考”的阶段性推断）**。

## 1. 环境与复现配置

- 代码：`mc2/moe_ep_dispatch` 的 **hybrid 模板**（`moe_ep_dispatch_hybrid.h`）+ epilogue（`moe_ep_dispatch_epilogue.h`）。
- 测试入口：`moe_train_test_slot_idx_wt/moe_train_test.py`（continue 模式：dispatch → epilogue → combine 每轮串行）。
- 物理拓扑：8 卡，每 server 1 卡（8 server）。
- 逻辑拓扑（debug 覆盖）：`MOE_EP_DEBUG_RANK_NUM_PER_SERVER=2`
  - `world_size=8`，`rankNumPerServer=2`，`serverNum=4`，`remoteServerCount=3`。
- 规模：`num_tokens=4096`，`num_max_tokens_per_rank=4096`，`hidden=7168`，`topk=8`，`num_experts=256`，BF16，`dispatch_do_cpu_sync=True`。
- 模式：`is_continue=1`、`is_dispatch=0`、`is_combine=0`。
  - `is_performance=0`（单轮）时 **第一轮就卡死**；
  - `is_performance=1`（多轮）同样卡死。
- 卡死无超时：kernel 内轮询无超时，host 侧 `do_cpu_sync` 等 dispatch 的 host counter。

## 2. 现象（症状）

- **rank0 的 epilogue 永久卡在 `WaitDispatch`**：
  `sum=4 cmp=8`，槽位 `s0/s2/s4/s6=1`，`s1/s3/s5/s7=0`（缺的正是 rank1/3/5/7 的 final-ready）。
- **除 rank0 外，7 个 rank 的 fanout 全部卡死**：
  `fanout poll completed=0/3`（rank3 只完成 src0，来自 rank1），三个源状态一直是 `sX=(-1,-1)`。
- rank0 完整走完：`process begin → getRecvCount → transfer begin → scaleout … done → fanout all done → transfer done → process done`，然后 epilogue 等 final-ready 卡住。
- 完整链条：**其它 rank 卡在 fanout（收不到 scaleout 状态/数据）→ 永远不发 final-ready → rank0 epilogue 等满 8 个 final-ready 死等**。

## 3. 关键证据链

### 3.1 count 交换全部通过

- 所有 rank `getRecvCount done sum=8` + `status cleared`。
- count 写在 `cntWinState` 区域（偏移 ~8KB），用 `WriteWithNotifyNbi(..., notifyAddr, notifyValue)`，**所有 rank 都收到**。

### 3.2 scaleout 发送端“全部成功”

- 24 笔（8 rank × 3 proxy）`scaleout proxy=... count=... done` 全部打印；
- 在 kernel 里给全部 `WriteWithNotifyNbi`/`Drain` 加了返回值断言（`ASCENDC_ASSERT(ret==0)`），**没有触发** → 所有调用返回 0。

### 3.3 接收端读到的状态/数据为 0

- fanout 轮询 `ReadGmByPassDCache` 读 scaleout 状态槽，卡死源一直为 0；
- 曾改为 **MTE2 DataCopy 读**（和 count 路径一致），仍然为 0；
- 在 fanout 卡住时读**数据区第一个字**：`d=0`（数据也没到）。d 分布：`268×d=0`，`8×d=1082572900`（到达源的真实 payload 首字），`4×d=1`。
- 结论：**状态和数据都没到达接收端地址**，不是“只差 notify”。

### 3.4 布局/地址一致性

- 各 rank 打印的 win 偏移完全一致：
  `data=16793600`（~16MB）、`status=754991104`（~720MB）、`win=268451840`、`slot=8192`。
- 地址公式：`GetWindowAddrByRankId(rank, offset) = epHcclBuffer[rank] + offset`（`moe_ep_dispatch_base.h`）。
- 发送端行 = 自己 `currentServerIndex_`；接收端行 = `srcScaleoutIndex`，对全部 (sender→receiver) 对逐一验证一致。
- count 写（~8KB 偏移）能到所有接收端，证明 `epHcclBuffer[dst]` 的 base 映射正确；唯独 scaleout 区域（16MB/720MB 大偏移）大部分收不到。

### 3.5 HCOMM 实现要点（asc-comm，URMA 平台实现）

- 文件：`/home/zerozill/.openclaw/projects/asc-comm/src/aicore/hcomm/detail/impl/platform_v310/hcomm_aiv_urma.h`
- buffer 校验：`/home/zerozill/.openclaw/projects/asc-comm/src/aicore/hcomm/detail/common/hcomm_utils.h`

- `platform_v310/hcomm_aiv_urma.h`：
  - `PostSend` 先 `HcommFindBufferIdx(remoteAddr, len)`：要求 `addr+len` 落在某注册 buffer 内，否则返回 `HCOMM_FAILED`，写被静默丢弃（错误只 `KERNEL_LOG(KERNEL_ERROR)`，用户环境打印不出来）。
  - `Drain` = `PollCq(wqeCnt)`：等 CQE，若 CQE status/substatus 非 0 会打日志并返回错误。
  - `WRITE_WITH_NOTIFY` 的 WQE 占 2 个 block；notify 通过 `HcommUrmaNotifyCtx`（携带 buffer 的 tokenId/tokenValue + notifyAddr + notifyVal）写远端。
  - SGE `len = messageLen`（uint32）。
- 关键待办：在 `PostSend` 打印命中 buffer 的 `idx/base/size`，对比 count（len=512）与 scaleout（len=7168）命中的远端 buffer（当前尚未跑，见 §5）。

## 4. 尝试过的方案与结果

| # | 假设/方案 | 做法 | 结果 |
|---|-----------|------|------|
| 1 | 时序竞争（打印偏置） | 只留 `SendCountToRemoteRank` 里 Drain 后的 done 打印 | **能过**（唯一能“治好”的组合） |
| 2 | 同上 | 只留 Write 前 begin 打印 | 卡 |
| 3 | 同上 | Drain 后 200k cycle 忙等 | 卡（说明不是纯延时） |
| 4 | 同上 | Drain 后 `PipeBarrier<PIPE_ALL>` | 卡 |
| 5 | 跨轮次 count 状态复用（ABA） | count 交换加“轮次消费握手”（sender 等 receiver consumedTag，epoch） | 卡；且单轮也卡，直接排除跨轮 |
| 6 | 接收端标量读缓存旧值 | fanout 状态改 MTE2 DataCopy 读 | 卡（仍是 0） |
| 7 | 大长度写导致 notify 失效 | scaleout 拆成 `WriteNbi(数据)` + 独立 `WriteWithNotifyNbi(状态)` | 卡（现象不变） |
| 8 | outstanding WQE 过多丢写 | scaleout 每个 tokenRange 后 Drain 一次 | 卡 |
| 9 | 发送端 counter 读旧值 | 打印 `rangeSendEntryCount` | 全非 0（49~54），排除 |
| 10 | 布局/地址不一致 | 打印各 rank 偏移 | 完全一致，排除 |
| 11 | HCOMM 返回错误 | 全部调用点加 `ASCENDC_ASSERT(ret==0)` | 全部返回 0，排除（API 层面） |

> 结论：所有“kernel 侧能改的”方案都没解决；失败发生在 HCOMM API 之下（URMA 引擎/远端侧静默丢写），或存在尚未发现的地址/注册错位。

## 5. 当前分支状态与下一步（仅供参考）

> ⚠️ 本节与第 6 节是**我们自己的阶段性推断/后续设想**，不是已证实的事实。
> 请先基于第 1~4、7、8 节的客观事实与代码独立分析，再决定是否参考本节。

- 分支已回退到干净基线：
  - `467769213` A1/A2 fix（fanout source state V→S 同步、win 布局对齐）
  - `cdeb1faae` final-ready 布局统一（fanout dummy notify 对齐 `dispatchNotifyCount_`）
  - `00c5109d9` debug 环境变量（MOE_EP_DEBUG_*）
  - `c5452a58d` debug 打印 + hcomm 返回值断言（当前 HEAD，最后一笔）
- 当前 scaleout 写 = **单笔 `WriteWithNotifyNbi` + 每个 proxy 一次 Drain**（和 master 完全一致）。
- **master 对比**：master 的 hybrid count/final-ready/scaleout 写与当前分支一致；唯一结构差异是 **fanout**：master 为“先收齐 + 全量构建 + 两次 SyncAll + 统一发送”，本分支为“边到边按源轮询转发（streaming）”。master 在该场景不卡。

### 下一步（已规划，未执行）

1. 在 `hcomm_aiv_urma.h::PostSend`（`HcommFindBufferIdx` 成功后）加 `PRINTF`，打印 `opcode/remoteIdx/remoteAddr/len/命中buffer的base/size`（限 `len>4096 && addr 2MB 对齐`），对比：
   - count 写（len=512，~8KB 偏移）与 scaleout 写（len=7168，16MB/720MB 偏移）命中**同一个远端 buffer** 吗？
   - scaleout 命中 buffer 的 base/size 是否与接收端自己读的 base 一致？
2. 若命中 buffer 正常 → 检查远端侧是否静默丢弃（URMA token/地址转换、大偏移、WRITE_WITH_NOTIFY 大 len）。
3. 若 buffer 命中异常（如所有发送端命中 rank0 的 buffer，或命中一个 size 很小/过期注册的 buffer）→ 定位 buffer 注册来源（`elastic_buffer.cpp` 的 `HcclGetHcclBuffer`/`HcclChannelGetHcclBuffer`，或 HCCL 通道注册）。

## 6. 待外部 AI 重点分析的问题（仅供参考）

> ⚠️ 同上：以下问题只是我们给出的**候选方向**，可能对也可能错，
> 建议独立分析后自行形成问题清单，再与本节的候选方向对照。

1. **为什么 scaleout 大偏移写（16MB/720MB）在大部分 (sender→receiver) 对上是“API 成功但远端不可见”，而 count 小偏移写全部可见？** 除了“注册 buffer 范围/token 错位”和“远端静默丢弃”，还有什么机制会造成这种确定性（且按接收端分布）的现象？
2. **可工作的 4 对**（rank2/4/6→rank0、rank1→rank3）与其他 20 对的区别是什么？是否与 channel 注册顺序、rank0 的 buffer 特殊、或“哪个 rank 先创建 channel”有关？
3. `WRITE_WITH_NOTIFY` 的 notify 与数据是否可能因为**大 len**（>单 WQE 上限）而在硬件上被拆分/丢弃，但 CQE 仍正常返回？
4. **streaming fanout 与 master 批量 fanout 的时序差异**，是否可能让接收端在“写未完成”时读到 0 并缓存住（即使 ReadGmByPassDCache/MTE2 都不刷新）？

## 7. 关键代码位置

| 位置 | 说明 |
|------|------|
| `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h:1313` | scaleout 单笔写（数据+notify） |
| `...:1330-1345` | `SendScaleoutPayloadToProxy`：72 range 后一次 Drain |
| `...:1498` | `TryGetScaleoutSlotCount`（状态读，当前 ReadGmByPassDCache） |
| `...:1517` | `SendFanoutPayloads`（streaming fanout 状态机） |
| `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_base.h:71` | `GetWindowAddrByRankId = epHcclBuffer[rank] + offset` |
| `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h:330` | `WaitDispatch`（等 final-ready，卡死点） |
| `/home/zerozill/.openclaw/projects/asc-comm/src/aicore/hcomm/detail/impl/platform_v310/hcomm_aiv_urma.h:293` | `PostSend`（含 `HcommFindBufferIdx`） |
| `/home/zerozill/.openclaw/projects/asc-comm/src/aicore/hcomm/detail/common/hcomm_utils.h:103` | `HcommFindBufferIdx`（addr+len 必须落在注册 buffer） |
| `/home/zerozill/.openclaw/projects/asc-comm/src/aicore/hcomm/detail/impl/platform_v310/hcomm_aiv_urma.h:484` | `Drain`（PollCq） |
| `torch_extension/.../ops/csrc/elastic_buffer.cpp` | context 的 `epHcclBuffer[i]` 来源（Hccl*GetHcclBuffer） |

## 8. 关键日志片段

```
[DH] r=0 aiv=0 transfer begin serverNum=4 remote=3 scaleoutAiv=3 scaleupAiv=2
[DH] r=1 aiv=3 fanout poll=0 completed=0/3        # 卡死源：状态一直 (-1,-1)
[DH] r=1 aiv=3 fanout poll=1 completed=0/3
...
[EP] r=0 aiv=71 waitDispatch enter cmp=8 addr=...
[EP] r=0 aiv=71 waitDispatch poll=0 sum=4 cmp=8   # 缺 s1/3/5/7
[EP] r=0 aiv=71 waitDispatch poll=1 sum=4 cmp=8
...
# fanout 卡住时的数据区首字分布：
#   d=0          ×268  （卡死源：数据未到）
#   d=1082572900 ×8    （到达源：真实 payload 首字）
#   d=1          ×4
# 各 rank 偏移一致：
[DH] r=X aiv=0 offsets data=16793600 status=754991104 win=268451840 slot=8192
```
