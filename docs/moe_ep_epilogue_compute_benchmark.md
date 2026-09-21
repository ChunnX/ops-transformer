# D/C epilogue 处理耗时测试：通信等待前移

## 目的与计时口径

本轮将出站 Drain 和入站数据就绪等待放到 `MoeEpDispatch` / `MoeEpCombine` 返回前。
这是用于拆分通信与 epilogue 处理时间的同步测试版本，不是异步流水性能优化。
所有 rank 必须使用同一版本，且沿用现有调用链的 stream 顺序/事件依赖。

单独增加本 rank 的 Drain 不足以达到目的：本端已经发完时，其他 rank 仍可能尚未发完本 rank 需要的数据。
因此每个 sender 在 Drain 后，还等待本 rank 接收窗口中其负责的全部源 rank 完成标志。
kernel 结束意味着32个 sender都完成该等待，之后 epilogue 不再承担对应的网络依赖等待。

epilogue 仍执行原来的标志检查与清零、索引/前缀和、搬运/重排/归约和核间同步。
所以新计时是**数据就绪后的 epilogue 处理时间**，不是只包含算术指令；检查与清理的固定成本也计入。
发送 kernel 时延预期会增加，不应将该版本的 Dispatch/Combine 单项耗时直接当成异步提交性能回归。
同步边界也改变了跨阶段重叠、缓存和访存竞争，Total 不保证与原异步版严格相等。

## 发送完成边界

当前 Hcomm 的 batch Drain 只等待已产生的 CQE；全部 WQE 为 `cqe=0` 时直接返回，不能证明完成。
核对依据：本地 `D:/code/asc-comm/docs/zh/api/aicore/hcomm/Drain.md`，以及
`src/aicore/hcomm/impl/platform_v310/hcomm_aiv_urma.h` 的 `BatchPostSend`、`BatchCommit`、`PollBatchCq`、`Drain`。

按测试要求，所有远端请求均设置 `cqe=1`：Dispatch 的 count 交换、payload、完成 flag，
以及 Combine 的 payload（包括多 SGE）和完成 flag。保留 payload 的 `odr=5, fence=1` 和
flag 的 `odr=6, fence=1`。本地 GM 搬运不产生 WQE/CQE。每个 WQE 产生一个 CQE，而非每个 SGE 一个。

CQ 只有64项，不能累计所有请求到 kernel 末尾才 Drain。本版本将发送批次上限统一为
`MOE_SHARED_JETTY_CQ_DEPTH / 2 = 32` 个 WQEBB，并在每次非空 `BatchCommit` 后立即 batch Drain。
一个 WQE 至少占一个 WQEBB，因此每批最多32个 CQE；多 SGE 只会进一步减少每批请求数。
这不是新增队列深度断言，而是测试模式的提交/回收策略；刻意不追求前半段通信吞吐。
原 batch UB 分配大小保持不变，Drain 使用已提交 batch 的工作区，不覆盖待提交请求。

- Dispatch count：每次 `WriteWithNotifyNbi<DATA_CFG>` 后 `BatchCommit(batch)` → `Drain(batch)`，
  完成后才销毁局部 handle 或复用工作区。
- Dispatch payload/flag：`ReserveBatchSpace` / `CommitBatch` 按32 WQEBB分批提交并Drain；
  `SendTokenPipeline()` 结束后 `CommitBatch(true)` 提交并Drain尾批，然后等待入站数据。
- Combine payload/flag：`ReservePreparedWrites` / `FlushPreparedWrites` 按同样上限提交并Drain；
  `SendMetadataPipeline()` 结束后 `FlushPreparedWrites()` 提交并Drain尾批，再重置handle、等待入站数据。
- 空批次不提交、不Drain；远端空payload仍发送完成flag。peer遍历顺序和多SGE布局保持不变。
- 继续使用 batch Drain，不与普通 ChannelHandle Drain 混用。保留末尾全数据缓存清理及既有流水依赖，
  没有修改 Hcomm 实现。

## 接收完成边界

每个 sender `owner` 只读本 rank 的 `[srcRank][owner]` 标志，遍历全部 `srcRank`；32个 sender共同覆盖
完整 `P×32` 标志矩阵，没有写死8P，也不把计算核数改成32或64。

- Dispatch 使用 `localSlotStateWinAddr_ + (srcRank * dispatchNotifyCount_ + owner) * 512`。
- Combine 使用原 `combineStateWinOffset + numMaxTokensPerRank * K * 512` 之后的完成标志矩阵。
- 每次用 MTE2 读32 B到已有UB，按 `S_MTE2 → DataCopy → MTE2_S → GetValue(0)` 顺序轮询。
- 等待前所有本核请求均已提交并Drain，不需要其他rank进入epilogue来推动进展。
- sender不清标志；仍由原epilogue消费和清零，保持standalone/cached/continue调用方式。

两个 epilogue 文件、host tiling、Python/API、context/窗口/metadata ABI 均未修改；没有新增UB/workspace。
32 Jetty / SQ4096 / CQ64、BS分核、多SGE、TopK累加顺序和已修复的Dispatch流水同步保持不变。
没有新增设备日志、assert、返回值检查或 `(void)` 包装。

## 验证与限制

- `git diff --check`通过。逐函数比较确认只修改发送完成相关函数；两个epilogue、context与runtime未变。
  新增/修改的生产函数均不超过50个非空非注释行；没有新增设备断言、日志或返回值包装。
  源码检查确认尾批提交在Drain之前、Combine保留活跃handle、入站等待只读且覆盖全部源rank。
- `python -B mc2/common/tests/test_moe_ep_sender_completion.py -v`：9项CPU模型测试通过。
  覆盖P=1/2/3/8/16/32/33/64/65/128/1024的全CQE请求、空payload、尾批提交、CQ多轮跨64复用、
  count逐请求提交/Drain/重建handle、1/2/4 WQEBB混合批次和8192条请求的持续回收，
  33/48/56/64/80/100 AIV下的接收标志覆盖、出站完成但入站延迟、清零仍属于epilogue。
  P1仅为模型边界，当前host实际支持范围仍从2P开始。
- 模型只验证任务顺序/计数/所有权，不模拟DMA可见性、硬件CQE或实际并发时序。
- 在32 Jetty独占发送、SQ4096/CQ64配置及Hcomm正常完成的前提下，每批Drain之后才继续提交，
  未回收CQE和已提交未回收WQEBB均不超过32，不随BS或peer数量累计。CPU模型不证明实际硬件队列配置，
  也不覆盖Hcomm超时/错误；按要求未新增返回值检查。
- 本地没有执行CANN编译/NPU验证，也没有操作远端。独立DeepSeek reviewer调用及一次重试均失败，
  原因为当前ChatGPT账号不支持`deepseek-flash`；未替换模型，独立四维审查未完成。
  当前结论仅来自主代理源码复核和CPU模型，仍需上机验证精度、继续多轮及计时。
- 本轮修改在两个算子kernel中，需要重建/安装包含它们的算子包；没有新增wheel接口。
  对比同一环境、同一输入分布下六项计时，并注明采用该同步测试版本。
