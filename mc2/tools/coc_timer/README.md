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

容量条件：`align_up(12*A,128) + 128 + 512*aivNum <= 512*A`（ABI v6 每核 512 B）。
例如 48 个 AIV 至少需要 A=50。每核记录区为四个连续的独占 cache line（512 B），避免与其他核或
三元组搬运共享 cache line。原始时间戳在核内暂存，Process 结束时统一写 GM 并 DCCI。

header 是 16 个 little-endian uint64：magic、version、status、A、aivNum、
recordWords、eventCount、rankId、sendCoreCount、hCommuSize、hOutSize、offsetBytes、
requiredBytes、cyclesPerUs、sendLayout、arrival_only。
sendLayout=1 表示同核先发数量通知、再发 payload（PR #12108 布局）；
header 的 sendCoreCount/hCommuSize/hOutSize 在 Timer 初始化时尚未推导完成，
先写 0，由 Flush 时 core 0 用 SetMeta 存下的值补写。
当前布局为 ABI v6；magic=`0x434F434D4F453031`，version=6；
status=1 表示启用，2 表示空间不足。
空间不足时不写任何核记录；若连 header 都放不下则完全不写，解码器会明确报错。
不能为此缩小三元组预留区。

每核 64 个 uint64（512 B）：magic、逻辑 AIV ID、6 个保留字段（恒为 0）；
随后是 6 对 start/end cycle。
纯 AIV 内核直接用 GetBlockIdx()，不套用原工具的 AIC+AIV 混合核编号。
未执行的事件为 0/0，每次初始化清空核内状态、Flush 覆盖所有槽位。

## 打点含义（ABI v6：纯串行，无嵌套）

每个核只有一条串行区间链，任意两段不重叠（允许管线屏障造成的小间隙），
各段耗时相加即为本核从 Init 入口到输出排空的完整跨度：

| 事件 | 记录核 | 范围 |
| --- | --- | --- |
| AIV_INIT_0 | 全核 | 内核 Init 入口（Timer 初始化之后第一条语句）→ Init 结束；覆盖 tpipe/ctx 初始化、SetTilingDataAndCal、SetDataStatus 等全部初始化工作 |
| AIV_CNT_SEND_0 | 发送核 | SendCumSum：expertIds 搬入、mask/validExpertIds 构建、专家 tokenCnt 计算与按 rank 合并的数量通知发送 |
| AIV_PAYLOAD_0 | 发送核 | AllToAllDispatch：token payload 量化打包与窗口写入；紧跟 CNT_SEND 结束，串行 |
| AIV_CNT_CALC_0 | cumsum 核 | CalCumSum：等待数量通知轮询（WaitDispatch）、cumsum 计算、epRecvCnt 写出、软同步 flag 发送；发送核恒为 0/0 |
| AIV_RECV_COPY_0 | 全核 | LocalWindowCopy：等 cumsum flag、接收轮询与连续化输出（含 CopyInAndOut） |
| AIV_TAIL_0 | 全核 | RECV_COPY 结束 → MTE3_S 排空完成 |

发送核串行链：`INIT → CNT_SEND → PAYLOAD → RECV_COPY → TAIL`；
cumsum 核串行链：`INIT → CNT_CALC → RECV_COPY → TAIL`。
导出校验强制相邻区间不重叠、核类型与记录的阶段集合一致。

PR #12108 布局中，前 aivUsedAllToAll 个核（send_cores）执行 SendCumSum 后再执行
AllToAllDispatch；其余核只执行 CalCumSum，不发送。CSV 的 send_kind 为
payload_count / receive_only。

### 时钟与开销

保留原始 uint64 cycle，Host 端除以 50.0 得到微秒。普通打点不新增 PIPE_ALL；
接收出口后用 MTE3_S 确保输出完成后再读 TAIL 终点。
全部打点均位于循环之外（Tik 幂等保证只在边界读一次时钟），循环体内零打点开销。
正式性能比较应同时运行关闭打点的版本，计时仍有指令与寄存器开销。

记录均为每个逻辑 AIV 的本地时间。确认同卡计数器可比较后，才可取本卡相关核的最早起点
与最晚终点计算全卡跨度；不能用 max(每核耗时) 代替这一跨度。跨卡只比较已计算的持续时间，
不能相减不同 rank 的绝对 cycle。
算卡间有效带宽时，应另行统计实际远端 token 副本数，不能将输入张量大小或 hCommuSize
直接当成总发送字节数。

## 导出与可视化

在 dispatch 所在 stream 同步成功后、下一次复用这个输出之前，将**完整**的
assist_info_for_combine（连续 int32，A*128 个元素）保存为二进制。建议在 Combine 之前取数。
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

解码器校验版本、容量、每核完成记录以及串行区间链；空间不足、关闭打点、dump 不完整
都会失败，不会将零记录伪装成有效测量。每次同步后立即导出；不支持多次并发调用共享同一输出。
只支持当前 ABI v6 dump，不兼容历史 v1-v5 布局。

## 验证

```bash
python3 -m unittest discover -s mc2/tools/coc_timer/scripts -p 'test_*.py' -v
```

Python 测试验证 ABI、异常 dump、串行链校验与导出列。
