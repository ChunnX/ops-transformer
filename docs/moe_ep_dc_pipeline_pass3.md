# D/C 第三轮：Dispatch 源索引预取与 Combine 首批提交

## 范围与实测信息

基于 `a84181d5cb`。用户反馈上一轮 Dispatch/Combine 实测改善约10–20 us；未提供新的六项时延，
因此不把该反馈换算成确定的总带宽增幅。本轮新修改尚未执行 NPU 验证。

不修改 Hcomm，不修改 context/窗口/metadata 格式、完成协议、TopK 累加顺序。
32 Jetty、SQ4096、CQ64、32发送 owner、其余 Dispatch prepare 核和 epilogue 全部可用计算核均保留。
无新增设备 assert、printf、Trap、返回值检查或 `(void)` 调用包装。

## 一、共享本地逆向索引的用途

Combine 输入 x 按 expert 输出行排列，epilogue 输出按原 token 排列。metadata 给出正向关系，
例如一行说明本 rank 的 token37、TopK位置2 对应 `recvXIdx=891`；逆向索引就是：

`localRecvIndex[37 * K + 2] = 891`。

当前 `BuildLocalRecvIndex()` 让每个有输出的 AIV 扫描整个本 rank metadata 区间，仅留下属于自己 BS
分片的记录。假设56个活跃核、本 rank 约3072条记录，总计约172032次记录检查。
共享版本的目标是把记录分工扫描一次、发布一张索引，再让各核取自己的 BS 切片。
BS4096/K6 的一张 int32 表是98304 B（96 KiB），该大小随 BS/K 变化，不固定典型值。

这是本地索引构建优化，不减少 URMA payload。实现需要初始化无效项、跨核发布、避免同 cache line 的
并发写回覆盖、以及 standalone/cached 路径一致性；额外 workspace 和同步也有成本。
可以放在 epilogue 私有 workspace 内，不必修改 Hcomm 或远端窗口。本轮尚未实现该索引重构。

## 二、按 BS/chunk 就绪归约需要修改什么协议

现有 Combine completion 的语义是 `[source rank][Jetty owner]` 完成该源/owner的全部发送；
epilogue 的0号核等待全部标志、清理后 SyncAll，所有计算核再开始归约。

细化为 BS/chunk 不是简单移动 Wait，至少需要同时修改：

1. **发送分组与依赖**：目前每个 peer 按 metadata 行切给32个 sender，行主要按 expert 分布，
   不是按接收方 BS chunk 排列。同一 BS chunk 可能依赖多个 Jetty，不能看见任意一个 flag 就计算。
   需要按目标 chunk 建 bucket，或准确计算每个 chunk 的预期贡献和各 sender 完成关系。
2. **发布顺序与标志格式**：chunk 的所有 payload WQE 在先，ordered completion 在后；
   仅 BatchCommit 不算远端完成。空贡献要有明确处理，不能等待永远不会发送的标志。
3. **接收调度**：不同计算核等待自己负责区间的完整输入集合，保持原 TopK 浮点累加顺序。
4. **多轮生命周期**：用序号/代际或受控清零，防止上轮 ready 被本轮误用、提前清掉下一轮通知。
5. **流量与资源**：新增标志会增加 WQE、带宽和轮询开销；必须计入共享物理 SQ 的累计 outstanding。

这是我们自己的应用层完成协议，不要求改 Hcomm。Dispatch 也可以分块推进，但还牵涉 expert 输出行
前缀和 rank-grouped metadata 布局，不能只把 payload wait 拆小就声称等价。
本轮不修改这套协议。

## 三、Combine 已有多 SGE，为什么不能直接合并任意多 token

`SendRemoteMetadataSlot()` 的带权重路径已经使用两个 SGE：hidden、weight。
无权重路径仅需一个 SGE；没有必要把一个连续 hidden 人为切成多个 SGE。

本地 Hcomm 的 `WriteNbi(handle, dst, srcDescs, srcNum)` 只有一个远端起点，多个本地 SGE 在远端连续拼接，
不是多目标地址 scatter。当前 Combine 目标为 `(token * K + topk) * perSlotBytes`，通常是不连续 slot。

H7168/BF16时，hidden14336 B；slot按32 B字段空间再512 B对齐，为14848 B。
带权重时 hidden+weight=14340 B，距下一个slot尚有508 B padding；直接追加下一token会错位。
若确认多个目标slot相邻，可以在SGE中包含合法padding来源再合并，但还需要比较：

- metadata 重排/分桶的成本及真正相邻目标的比例；
- 增加 padding 网络字节和 SGE 数量的成本；
- 请求/WQEBB 减少带来的收益；
- 不能跨过别人拥有的有效 slot，也不能仅因源行相邻就合并。

Dispatch 当前压缩 peer window 的 slot 本来连续，每个 slot 的 x+meta 恰好覆盖其完整跨度，
因此已能一条请求发6个slot/12个SGE；Combine不能直接照搬而不处理目的地址与padding。
不再把“改 Hcomm 固定SGE fast path”列为本任务可实施项，也不在算子里复制其私有 WQE 实现。

## 四、本轮已实现

### Dispatch 跨 peer 源 token ID 预取

旧 `SendTokenData()` 每个 peer 都先搬入源 token ID、立即等待，再填SGE。
新流程在处理当前分片前启动下一个分片的源索引搬入，可以跨 peer，仍按原错峰 rank 顺序消费。
`GetTokenRange()` 和原 prepare-ready 依赖不变。

- 原160 KiB `tempBuf_` 分成两块80 KiB，不增加该部分UB。
- 每块最多20478个int32；是6的倍数，32 B padding后不越过80 KiB半区。
- 每路持有一个 `MTE2_S` event，非空块各Set/Wait一次，消费完再释放。
- 复用前保留 `S_MTE2`。预取不切通信peer；消费新peer首块时才调用原 `GetHandleRef`。
- descriptor长度每轮发送只初始化一次，不再每peer重复填；地址仍逐slot更新。
- 本地/空peer仍只有原completion，远端非空peer仅在最后一块发送completion。

它减少的是源索引搬入等待和重复准备，不减少hidden网络字节。
跨peer look-ahead也会把下一peer的边界计算提前到当前peer发送前，是否净提速必须实测，不能假定全被掩盖。

### Combine 小首批、原后续大批

原首批按256 WQEBB积累。现在首批门槛32 WQEBB，之后恢复256；复用Dispatch已有的小首批思路。
带weight的请求占2 WQEBB，不含flag时首批约16条payload后可触发提交，而非等约128条。
实际代码在加入下一条会超过门槛时提交，或在发送结束时提交尾批。

这改变的是开始发出网络请求的时刻，不是等待远端完成。原单/双SGE、odr/fence、cqe=0、
completion顺序、handle/cursor和最后Flush均保留。小首批会改变doorbell数量，32是待实测候选值，非最优结论。

## 五、验证与边界

- `git diff --check` 通过。
- MSVC C++17 `/O2 /W4 /WX` 编译运行：既有Dispatch区间调度测试、新增3072组Dispatch预取模型、
  8192组Combine调度模型及新增首批账目模型均通过。
- Dispatch模型覆盖P=1/2/3/8/33/1024、32个sender、本地/空peer、SQE尾部、20478边界和多chunk。
  测试local peer不更改remote binding，与实际kernel一致。
- 四维独立审查：逻辑、Clean Code、架构、功能完整性；逻辑审查提出的local-peer模型差异已修复并重跑。
- 未执行CANN编译/NPU回归。CPU模型不模拟DMA实际时序，也不证明链路利用率或设备正确性。
- **单batch小于4096不等于累计SQ安全**：BatchCommit不会回收远端完成credit，更快生产仍可能暴露已有的
  大BS/极端偏斜下SQ复用风险。本轮不增加CQE/Drain，也没有解决该既有问题。
- 回归应覆盖单跑/continue、cached、weights开关、FP8/scales、空/小BS、不同P及大分片。
  同一环境下比较六项计时，尤其D/C Total，不能只以提交kernel缩短推断整体带宽提高。
