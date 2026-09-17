# Full-mesh MoE CocTimer 打点

基于 [ops-transformer PR #5122](https://gitcode.com/cann/ops-transformer/pull/5122/diffs)
的 `mc2/tools/coc_timer/README.md` 和工具，参考提交
`c78261ddd060db5d8981f9dfc01e29bc19bdc5c9`。
`scripts/process.py` 原样取自该提交。

这里是面向 `moe_distribute_dispatch_v2_full_mesh.h` 的紧凑适配版，**不是原版
CocTimerDevice 的二进制布局**，不能交给原版 `CocTimerHost::GetTimeResult` 解码。
保留原版的 GetSystemCycle、start/end cycle、50 cycle/us 和 CSV 字段约定；使用
`export_moe_full_mesh.py` 转换后，可直接用 PR 的 `process.py` 生成 Chrome Trace。
不修改算子签名，不增加可选输入。

## 开启与关闭

默认关闭。在原有 CMake 配置参数中追加：

```text
-DMOE_DISPATCH_FULL_MESH_TIMER=ON
```

然后重新编译并安装自定义算子。该选项通过 `add_ops_compile_options` 为
`MoeDistributeDispatchV2` 传递 `-DMOE_DISPATCH_FULL_MESH_TIMER=1`。
直接调用设备编译器时也可直接传递该宏。关闭时设置 CMake 选项为 OFF 并重新编译。
确认实际 tiling 选择 A3 full-mesh；其他算法不会产生这个布局。

使用仓库 build.sh 时，可以直接通过现有编译选项入口开启（保留你自己的其他构建参数）：

```bash
bash build.sh --pkg --soc=ascend910_93 --ops=moe_distribute_dispatch_v2 \
  --ops-compile-options=-DMOE_DISPATCH_FULL_MESH_TIMER=1
```

此方式不依赖外层 CMake 向准备阶段的子构建转发自定义选项。关闭时移除该编译选项并重新编译。

## assist_info_for_combine 布局与容量

这里的 A 使用 `MoeDistributeDispatchV2Info::a`，是输出容量，不是实际接收 token 数。
输出的合法分配大小为 `A * 128 * sizeof(int32_t)`。
本 full-mesh 路径在前 `3*A` 个 int32 中存储连续的 rank/token/top-k 三元组，
对应 CombineV2 的 `expandIdxGM_[startTokenId * EXPAND_IDX_INFO]` 读取。
只针对这一对算法，不能把布局结论推广到 layered/A2/A5 等路径。

```text
字节偏移 0                   12*A               align_up(12*A,128)
         [Combine 三元组区域] [对齐空隙]         [128 B header]
                                               [AIV 0：512 B]
                                               [AIV 1：512 B] ...
```

容量条件：`align_up(12*A,128) + 128 + 512*aivNum <= 512*A`（ABI v5 每核 512 B）。
例如 48 个 AIV 至少需要 A=50。每核记录区为四个连续的独占 cache line（512 B），避免与其他核或
三元组搬运共享 cache line。原始时间戳在核内暂存，Process 结束时统一写 GM 并 DCCI。

header 是 16 个 little-endian uint64：magic、version、status、A、aivNum、
recordWords、eventCount、rankId、sendCoreCount、hCommuSize、hOutSize、offsetBytes、
requiredBytes、cyclesPerUs、sendLayout、reserved。
sendLayout=0 为旧版发送核/计数发送核分工；sendLayout=1 为 PR #12108 的同核先发数量通知、再发 payload。
当前布局为 ABI v5；解码器兼容 ABI v2/v3/v4，v2 旧数据没有 TAIL，v3 只有 4 对区间，v4 只有 9 对区间。
magic=`0x434F434D4F453031`，当前 version=5；status=1 表示启用，2 表示空间不足。
空间不足时不写任何核记录；若连 header 都放不下则完全不写，解码器会明确报错。
不能为此缩小三元组预留区。

每核 64 个 uint64（512 B）：magic、逻辑 AIV ID、5 个保留字段（早期版本写 data-wait 统计，
现恒为 0）；随后是 14 对 start/end cycle，顺序为 STAGE、SEND、COMM_TOTAL、TAIL、
INIT、CNT_SEND、PAYLOAD、CNT_CALC、RECV_COPY、EXPID_COPY、CNT_NOTIFY、WAIT_DISPATCH、
CUMSUM_CAL、RECV_WAIT。
纯 AIV 内核直接用 GetBlockIdx()，不套用原工具的 AIC+AIV 混合核编号。
未执行的事件为 0/0，每次初始化清空核内状态、Flush 覆盖所有槽位。
当前解码器支持 ABI v2/v3/v4/v5，不支持最初的 12 区间 ABI v1 dump。

## 打点含义

| CSV 指标 | 范围 |
| --- | --- |
| AIV_STAGE_0 | 本核 Process 入口 → Process 结束（Flush 之前）；全核记录 |
| AIV_SEND_0 | 发送阶段入口（CNT_SEND 之前）→ PIPE_ALL 后；仅发送核，PR #12108 中先数量通知再 payload |
| AIV_COMM_TOTAL_0 | 与 SEND 共用起点 → 与 STAGE 相邻的终点（Process 末尾）；没有发送则为 0/0 |
| AIV_TAIL_0 | RECV_COPY 结束 → MTE3_S 排空完成；位于 STAGE 内部，全核记录 |
| AIV_INIT_0 | Timer 初始化后（SetTilingDataAndCal 之后）→ Init 结束；覆盖窗口状态初始化等 GM 读写，ctx 初始化与 tiling 计算不在其内 |
| AIV_CNT_SEND_0 | SendCumSum 阶段：ExpIdsCopyAndMaskCal 与专家 tokenCnt 计算、数量通知发送；仅发送核（aivId < sendCoreCount） |
| AIV_PAYLOAD_0 | AllToAllDispatch 阶段：token payload 量化打包与窗口写入；仅发送核，空发送（activeMaskBsCnt==0）也为 0/0 |
| AIV_CNT_CALC_0 | CalCumSum 阶段：等待数量通知（含轮询）、cumsum 计算、expertTokenNums 输出；仅 cumsum 核（aivId >= sendCoreCount） |
| AIV_RECV_COPY_0 | LocalWindowCopy 阶段：等 cumsum flag、接收轮询与连续化输出（含 CopyInAndOut）；全核记录 |
| AIV_EXPID_COPY_0 | ExpIdsCopyAndMaskCal：expertIds GM→UB 搬入与 mask/validExpertIds 构建；仅发送核，CNT_SEND 的子区间 |
| AIV_CNT_NOTIFY_0 | SendExpertCntRange：按 rank 边界合并的数量通知发送；仅主路径（弹性回退的逐专家交织路径不打点），可缺省 |
| AIV_WAIT_DISPATCH_0 | WaitDispatch 轮询循环：等待各 rank 数量通知就绪（含每轮 DataCopy+ReduceSum 开销）；仅 cumsum 核，CNT_CALC 的子区间 |
| AIV_CUMSUM_CAL_0 | CalRecvAndSetFlag：cumsum 计算 + epRecvCnt 写出 + 软同步 flag 发送；仅 cumsum 核，WAIT_DISPATCH 之后 |
| AIV_RECV_WAIT_0 | WaitCumSumFlag 轮询循环：等 cumsum 软同步 flag，接收流水的启动等待；全核，RECV_COPY 的子区间 |
| DATA_WAIT_* | 已废弃：循环内打点会引入逐批读时钟开销，v5 起不再采集，CSV 列恒为 0（仅旧 ABI dump 有值） |

PR #12108 中，前 aivUsedAllToAll 个核执行 SendCumSum 后再执行 AllToAllDispatch；
SEND 从发送阶段入口开始，到 Process 原有 PIPE_ALL 后结束。
其余核只执行 CalCumSum 接收数量通知与累计计算，不发送；它们的 SEND/COMM_TOTAL 为 0/0，
应使用 STAGE 观察其接收处理。
新 dump 的 CSV send_kind 为 payload_count / receive_only；旧 dump 仍导出 payload / count。
旧实验 summarize.py 中固定的 payload/count 分组不能直接用于新 dump，需同步调整分组标签；
新旧 SEND 语义不同，不应直接作为同口径性能对比。

STAGE 在 Process 末尾结束；COMM_TOTAL 终点与其相邻。时间不包含 Init 和打点 Flush，
其他准备、数量通知等待、核间同步、接收整理均可能包含在阶段内。
COMM_TOTAL 的起点是本核发送阶段入口；它不是跨卡 ACK 往返时间。

### 时钟与开销

保留原始 uint64 cycle，Host 端除以 50.0 得到微秒。普通打点不新增 PIPE_ALL；
接收出口后用 MTE3_S 确保输出完成后再读 TAIL 终点。
全部打点均位于循环之外（Tik 幂等保证只在边界读一次时钟），循环体内零打点开销。
正式性能比较应同时运行关闭打点的版本，计时仍有指令与寄存器开销。

记录均为每个逻辑 AIV 的本地时间。确认同卡计数器可比较后，才可取本卡相关核的最早起点
与最晚终点计算全卡跨度；不能用 max(每核耗时) 代替这一跨度。跨卡只比较已计算的持续时间，
不能相减不同 rank 的绝对 cycle。
统计发送包括目标为本卡的通信窗口写入，尚未统计每目标字节数；算卡间有效带宽时，应另行
统计实际远端 token 副本数，不能将输入张量大小或 hCommuSize 直接当成总发送字节数。

## 导出与可视化

在 dispatch 所在 stream 同步成功后、下一次复用这个输出之前，将**完整**的
assist_info_for_combine（连续 int32，A*128 个元素）保存为二进制。建议在 Combine 之前取数。
例如现有 Python 调用端已拿到 `assist_info_for_combine` 时：

```python
import torch_npu
torch_npu.npu.synchronize()
assist_info_for_combine.detach().cpu().contiguous().numpy().astype('<i4', copy=False).tofile('assist_rank_0.bin')
```

aclnn 调用端可在 `aclrtSynchronizeStream(stream)` 成功后，使用 `aclrtMemcpy` 的
`ACL_MEMCPY_DEVICE_TO_HOST` 复制 `A*128*sizeof(int32_t)` 字节并原样保存。
每 rank 单独保存，不要把 int32 数值转换成 int64 数值（这会破坏时间戳的位模式）。

```bash
python3 mc2/tools/coc_timer/scripts/export_moe_full_mesh.py assist_rank_0.bin --output output_timer
# 对其他 rank 重复上一条命令，header 中的 rankId 自动决定文件名。
python3 mc2/tools/coc_timer/scripts/process.py output_timer MoeDispatchFullMesh
```

导出器仅依赖 Python 标准库；PR 的 process.py 需要 pandas。
输出包括每 rank 的 duration CSV、`data/timer_rank_N_timeline.csv`、Chrome Trace JSON。
在 `chrome://tracing` 中加载 JSON。每个逻辑 AIV 使用一个 group 的 AIV_0 轨道；
原版 process.py 还会创建空 AIC/AIV_1 轨道，这不表示本内核使用了这些核。
每 rank 单独归零，不能据图推断跨卡起跑偏差。

解码器校验版本、容量、每核完成记录以及时间段配对；空间不足、关闭打点、dump 不完整
都会失败，不会将零记录伪装成有效测量。每次同步后立即导出；不支持多次并发调用共享同一输出。

## 验证

```bash
python3 -m unittest discover -s mc2/tools/coc_timer/scripts -p 'test_*.py' -v
g++ -std=c++17 -fsanitize=address,undefined mc2/tools/coc_timer/scripts/test_moe_full_mesh_timer.cpp -o /tmp/test_moe_full_mesh_timer
ASAN_OPTIONS=detect_leaks=0 /tmp/test_moe_full_mesh_timer
```

C++ 测试以 Host stub 验证真实 timer 类的状态机、首发时间保持、重复初始化、无任务路径和
缓冲区边界，不替代设备同步语义验证。Python 测试验证 ABI、异常 dump、累计等待和 PR 流水图兼容性。

## 仅观察到达的带宽实验模式

当前 full_mesh.h 默认 `MOE_DISPATCH_FULL_MESH_ARRIVAL_ONLY=1`。
LocalWindowCopy 调用的 WaitAndFormatOutput 保持原批次大小、来源轮询和 flag 检查，
成功后只推进 expertFinishNum/expertLeftNum，不执行 CopyInAndOut，且不等待未发起的输出 MTE3 事件。
仍保留 cumsum/flag 的 GM→UB 读取、必要同步、flag 清理和打点落盘。
因此 expand_x、dynamic_scales、assist 三元组等输出不再有效，不能用于 Combine 或正确性测试。
恢复正常行为需编译时传 `-DMOE_DISPATCH_FULL_MESH_ARRIVAL_ONLY=0`。
打点开关与本实验开关独立；关闭打点不会自动恢复输出搬运。

header 最后一个 uint64 记录 arrival_only（旧 dump 为 0），解码器将其导出到 metadata。
仅到达模式中 STAGE/COMM_TOTAL 的终点为完成到达检查和必要同步后、清 flag 前；
SEND 起止不变。DATA_WAIT 仍只计完整失败扫描后的累计无进展等待。
模式切换后需要重新编译、安装、采集，不应混合比较两种模式的接收阶段耗时。


## ABI v3：尾部打点与整卡跨度

`PollReady(lastBatch=true)` 在最后一批 flag 检查成功后、推进最后一次接收计数之前记录
TAIL.start；没有接收任务的核由 FinishStage 设置起点，不算作“最后到达”。
Process 中 LocalWindowCopy 返回后，通过 MTE3_S 等待 flag 清理和 RunPosRecord 写入完成，
随后 FinishTail 记录 TAIL.end。Flush 在该终点之后，打点落盘不计入尾部。
新增同步只在打点开启时执行，会有测量开销。

仅到达模式下尾部不含 payload 搬运。若 ARRIVAL_ONLY=0，最后一批 CopyInAndOut 及其完成等待
发生在 TAIL 内，不能将两种模式的尾部耗时按同口径比较。
STAGE/COMM_TOTAL 仍在清 flag 前结束；TAIL 与它们存在小段重叠，不可直接相加。
每核记录仍为 128 B，使用原来的两个保留时间戳槽，不增加 assist 容量要求。

导出新增 `AIV_TAIL_0_start_cycle/end_cycle/duration_us`（timeline CSV）和
`AIV_TAIL_0_duration(us)`（duration CSV），metadata 增加 timer_version=3。
原 process.py 可直接生成 TAIL 时间线。

运行 `/root/workspace/Q3/summarize.py TEST_DIR` 后：
- 原 `last_iteration_summary_tprs_N.csv` 增加 tail_core_min/median/max_us。
- 新 `card_span_summary_tprs_N.csv` 每 rank 一行：
  - first_send_cycle：所有非零 SEND.start 的最小值。
  - last_send_cycle：有效发送核 SEND.end 的最大值。
  - last_receive_cycle：receive_batches>0 的核的 TAIL.start 最大值，包含 receive_only 核。
  - last_tail_end_cycle：所有核 TAIL.end 的最大值。
  - card_send_span_us=(last_send_cycle-first_send_cycle)/50。
  - card_first_send_to_all_received_us=(last_receive_cycle-first_send_cycle)/50。
  - card_post_receive_tail_us=(last_tail_end_cycle-last_receive_cycle)/50。

整卡公式要求同卡各核 GetSystemCycle 可比较，输出 clock_basis 明示这一假设；不跨 rank 相减。
没有有效发送或接收任务时对应跨度留空。ABI v2 仍可算发送跨度，但其接收/尾部跨度留空，
receive_span_status=unavailable_old_dump，绝不用 STAGE.end 冒充最后到达确认。
这些是软件观察到的完成时刻，不代表精确链路到达时刻，也不是对端接收 ACK。

## ABI v4：全阶段打点

在 v3 的 STAGE/SEND/COMM_TOTAL/TAIL 之上新增 5 对阶段区间，覆盖从 Init 后半段到
Process 结束的各核各阶段：

- `INIT`：Timer 初始化（位于 SetTilingDataAndCal 之后）到 Init 结束。ctx 初始化、
  tiling 计算发生在 Timer 可用之前，不计入；SetDataStatus/窗口状态初始化计入。
- `CNT_SEND`：发送核的 SendCumSum（expertIds 搬入、mask 计算、专家 tokenCnt 计算与数量通知发送）。
  SendCumSum 内部的 BeginSend 仍只记录首次通信写入时刻，不改变 SEND 语义。
- `PAYLOAD`：发送核的 AllToAllDispatch（含 SendToSharedExpert/SendToMoeExpert/SendToMoeExpertByBS）。
  AllToAllDispatch 因 activeMaskBsCnt==0 提前返回时，PAYLOAD 只含空壳耗时，仍成对记录。
- `CNT_CALC`：cumsum 核的 CalCumSum（WaitDispatch 轮询等待数量通知、CalRecvAndSetFlag、
  SetExpertTokenNums）。该区间天然包含接收数量通知的等待时间，与 DATA_WAIT 互补。
- `RECV_COPY`：全核的 LocalWindowCopy（WaitCumSumFlag、SetValidExpertInfo、WaitAndFormatOutput
  接收轮询与 CopyInAndOut、ClearLocalWindowDataFlags）。STAGE 在清 flag 前结束，
  RECV_COPY 在 LocalWindowCopy 返回后结束，因此 RECV_COPY.end 可以晚于 STAGE.end。

每个阶段用通用 `Tik/Tok` 记录，Tik 不覆盖已有起点、Tok 保留最早终点；打点为寄存器内
条件判断加一次读时钟，不新增同步与 GM 写。阶段间的PipeBarrier<PIPE_ALL>、EndSend、
FinishTail 同步与 v3 一致。解码器（ABI v4）校验：发送核链路
CNT_SEND.end <= PAYLOAD.end <= SEND.end <= RECV_COPY.start，接收核不得记录发送阶段，
全部阶段不得逃出 [STAGE.start, TAIL.end]。

v4 每核记录从 128 B 翻倍到 256 B，容量条件相应变为
`align_up(12*A,128) + 128 + 256*aivNum <= 512*A`；空间不足仍以 status=2 报告，不写核记录。
导出 CSV 为每个事件输出 start_cycle/end_cycle/duration_us 列；PR 的 process.py 按
`*_start_cycle/*_end_cycle` 列名生成时间线，新阶段自动进入 Chrome Trace。
summarize_new.py 只引用原有列名，无需修改即可继续使用。

## ABI v5：纯 Tik/Tok 与循环零打点

在 v4 的 9 对阶段区间之上再细分 5 对子阶段；同时移除全部非 Tik/Tok 接口
（BeginSend/EndSend/PollMiss/PollReady/TokLast/FinishStage/FinishTail）与循环内打点，
避免逐批/逐 token 读时钟导致性能失真：

- `EXPID_COPY`：发送核 ExpIdsCopyAndMaskCal（expertIds 搬入、mask/validExpertIds 构建），
  CNT_SEND 的子区间。
- `CNT_NOTIFY`：SendExpertCntRange 按 rank 边界合并的数量通知发送（主路径）。弹性回退的
  CalAndSendCntByExp 计算与发送逐专家交织，不打该点；无分摊专家的发送核也可缺省。
- `WAIT_DISPATCH`：cumsum 核 WaitDispatch 轮询循环，等全部 rank 数量通知就绪；含每轮
  DataCopy+ReduceSum 的软件观察开销，CNT_CALC 的子区间，CUMSUM_CAL 之前。
- `CUMSUM_CAL`：cumsum 核 CalRecvAndSetFlag（cumsum 计算、epRecvCnt 复制 aivNum_ 份写出、
  软同步 flag 发送）。SetExpertTokenNums 仅 newAivId==0 核执行且耗时小，不单独打点。
- `RECV_WAIT`：WaitCumSumFlag 轮询循环（等 cumsum 软同步 flag），接收流水的启动等待，
  全核记录，RECV_COPY 的子区间。

语义调整：SEND/COMM_TOTAL 起点改为 Process 发送阶段入口（不再记录首次 DataCopy 时刻）；
STAGE 终点改为 Process 末尾；TAIL 改为 RECV_COPY 结束到 MTE3_S 排空完成的段内区间；
`COPY_OUT` 与 DATA_WAIT 统计废弃（需循环内打点，采集开销大于信息价值）。

导出校验（ABI v5）：发送核 EXPID_COPY ⊆ CNT_SEND、CNT_NOTIFY.start >= SEND.start、
CNT_NOTIFY.end <= PAYLOAD.start；cumsum 核 WAIT_DISPATCH ⊆ CNT_CALC 且
WAIT_DISPATCH.end <= CUMSUM_CAL.start <= CNT_CALC.end；全核 RECV_WAIT ⊆ RECV_COPY、
recv_copy[1] <= tail[0] <= tail[1] <= stage[1]。

v5 每核记录从 256 B 翻倍到 512 B（四个独占 cache line），容量条件变为
`align_up(12*A,128) + 128 + 512*aivNum <= 512*A`；48 AIV 需要 A>=50。空间不足仍以
status=2 报告。解码器保持向后兼容 v2/v3/v4。
