# Dispatch 尽早退出：恢复异步发送与后续计算方案

> 本文保留异步恢复与 A/B/C 设计过程。A/B/C 已在后续工作树中实施，当前流程、workspace和验证结果见
> [Dispatch tile 流水实现](moe_ep_dispatch_tile_pipeline.md)。下面“尚未实施”等措辞描述当时的计划阶段，
> 不用于判断当前代码；D 接收计数并行化仍未实施。

## 本轮已落地的边界

目标是尽早完成前置计算和发送提交，而不是在 Dispatch 中等网络传输全部完成。
本轮保留未提交的全 AIV 计数/寄存器 Histograms 优化，撤销 `84d0aba660` 的同步测量改动：

- Dispatch count、payload、flag 和 Combine payload、flag 全部恢复 `cqe=0`。
- 删除 sender 中逐批 Drain、末尾入站完成轮询及其辅助函数。
- 恢复首批32 WQEBB提前启动；后续 Dispatch 使用原512 WQEBB批容量，Combine使用原256 WQEBB批容量。
- 保留尾批强制提交、payload/flag 的 `odr=5/6, fence=1` 顺序和本地 DMA 同步。
- 两个 epilogue 原有等待、清标志和消费前屏障保持不动；没有把 Drain 调用机械搬进 epilogue。
  `cqe=0` 的完成判据是原有接收窗口完成标志，而不是 CQE。
- 保留 Dispatch 的 `GetRecvCount()`：它产生后续输出需要的 rank/expert 计数和可选 host `actualA`，
  不是本次要求后移的 payload 完成等待。

32 Jetty / SQ4096 / CQ64 不改，不固定卡数，也不把计算核数固定成32或要求64 AIV。
不修改 Hcomm、host API、context、注册窗口布局或 TopK 累加顺序，不新增设备日志、assert或返回值检查。
撤销测试模式不等于删除正确性同步；`PublishPrepareReady`、UB复用事件和既有精度修复仍保留。

## 当前关键路径及为何仍不能立即退出

```text
全部 AIV：BufferInit/清零 → 全核屏障 → 分片计数/去重 → 全核屏障
                                  ├─32发送核：交换计数 → 等本 BS 片准备 → 构造/提交WQE → 返回
                                  └─剩余计算核：准备索引/metadata → 发布就绪 → 返回
                                                        └─最后一核：接收计数归约 → 返回
下一阶段：epilogue等待远端完成标志 → 重排/归约
```

Dispatch kernel 退出取所有 AIV 完成时间的最大值，不是只看发送核。
近似关系是 `T_init + T_count + max(T_prepare+T_submit_tail, T_count_exchange+T_recv_count)`，
其中准备与提交只在已有就绪区间允许的部分重叠；式子用于辨别依赖，不是性能预测模型。
恢复异步后的 Dispatch 单项下降不等于 Total 等比例下降，部分时间会重新计入 epilogue 等待。

当前计算的三个可见开销来自实际代码：

1. `CalSendCnt()` 已全核分 BS，但为兼容旧 prepare 行会分割交集、原子累计；
   后面的 `SendPhase()` 又加载 dedup rank 并逐 peer 筛选，metadata还会重新读取 TopK。
2. `GetSlotStartNum()` 各 prepare 核重复归约前面的行；发送侧 `GetTokenRange()` 再读前缀，
   并在边界 prepare 的有序 token 列表中二分查找。
3. `PublishPrepareReady()` 到整个 prepare 区间结束才发布。其前面的 `ProcessMetaSlot()` 虽按小批量准备，
   发送端仍不能消费其中已完成的小批次。

这些是源码可见的优化机会，并非已用 NPU 计时证明的瓶颈排名。

## 建议方案与先后顺序（尚未实施）

### A. 将与 slot 无关的 metadata 准备融入全核计数阶段

TopK ID/weight、scale、src rank、src token 都不依赖远端计数和最终 slot。
在每个 token 唯一计数 owner 处理它时，一并生成原格式的本地注册窗口 metadata stash，
使全部 AIV 分担当前集中在剩余计算核上的任务，并复用已载入的 TopK。

后续 prepare 核仍负责依赖 slot 前缀的索引/slot和本地接收窗口metadata，不能直接删掉本地分支。
从 stash 复制本地 metadata 还是保留 UB 数据，需要按 UB 生命周期选实现，避免多搬运抵消收益。
发送前的就绪信号必须同时覆盖索引与 metadata；全核计数后的屏障仍先保留。
这是先减少重复工作，不靠增加同步来追求“所有核都忙”。

涉及 `CalSendCntGroup`、`CopyInMetaInfo`、`ProcessMetaSlot`、`SendPhase` 及其 UB 分配；
不改远端窗口/完成标志协议。验证重点为 fp8 scale、可选 weights、cached slot 和本地 rank 路径。

### B. 用逻辑 BS tile 的计数和共享前缀替代 prepare 交集原子累计

将固定于物理核的 prepare 行改为按 token 顺序排列的逻辑 tile：tile 大小由 K、metadata大小、UB预算、BS决定，
不由8P或固定计算核数决定。计数时全部 AIV 领取不重叠 tile，每个 tile 的 `[peer]` count 只由一个核写。
各核还累积自己的 expert histogram，最后并行归并，避免所有核向同一 global expert count 原子写。

对 `[tile][peer]` count 做沿 tile 轴的独占前缀，得到 `slotBase[tile][peer]`。
前缀阶段可按 peer 列块分工，较大 tile 数再按行块做局部扫描和块间进位；必须计入这一步新增的固定开销。
索引列表仍按 BS 递增，所以远端 peer-contiguous slot 布局与 golden 所要求的顺序不变。

这需要修改本地 workspace 规划和相应读写者，不能复用一块仍有 count 消费者的内存后假定语义不变。
额外空间按 `tileCount * P` 的count/前缀与 `AIV * E` 的expert部分和计算，明确UB和GM预算；
只改本地工作区，不扩展注册通信窗口。小 BS 时tile数和有效核数自然缩减，避免过度细分。

### C. 基于 tile 描述符发布就绪，发送端边消费边提交

角色按阶段明确：A/B的计数、独立metadata、计数归并使用全部AIV；进入发送重叠阶段后，
32发送核各自绑定一个Jetty，按tile owner分摊BS，剩余 `AIV-32` 核生成依赖slot的索引/本地metadata并发布。
这里没有要求发送核同时执行耗时的tile准备，也没有把全部计算永久限制在剩余核；
已通过A将可提前的metadata计算分给全部AIV。每个就绪描述符至少明确：tile序号、目标peer、源索引起点/数量、目标slot起点。
本地 metadata 与源索引全部写出后才发布 ready；零命中tile也必须推进完成状态。

不能只把当前 `PublishPrepareReady()` 放进 metadata 小循环：
当前 `GetTokenBoundary()` 会在整个边界 prepare 列表内二分，如果列表只有前半段写好，就可能读未准备数据。
新发送器必须只访问已经就绪描述符给出的范围，不再搜索未就绪列表；描述符可按tile成组发布以降低控制开销。

实施时必须显式建立以下状态依赖，不能把“当前tile算完”当作所有前缀已经可用：

```text
tile count-ready（含零命中tile）
    → 该peer所有前序tile的count完成 → prefix-ready
    → slot/index与本地metadata完成 + A阶段stash完成
    → descriptor-ready（发布前保证上述GM写完成）
    → sender读取有界描述符并构造/提交WQE
    → 本sender对本peer所有tile均提交 → 最终完成flag
```

全局 per-rank/per-expert totals 必须归并并发布后才能执行原 `Communication()` 的count交换；
不能用某个tile完成代替全卡计数完成。接收侧每expert输出/actualA仍等待完整接收计数。
完成flag的身份是 `[src rank][sender jetty]`，每个sender给每个peer恰好一个，包括没有payload的peer，
不是整张卡给每个peer只写一个flag，也不是每tile写一次。

一次调用优先使用不复用的有界描述符表，每轮在既有初始化边界清理。
如果以后改循环队列，另需代际序号/消费确认，不能用重复的布尔ready避免覆盖问题。
发送UB索引预取、SGE构造和描述符消费也必须保留各自的复用事件。

远端仍在该 sender 对该 peer 的全部 tile 提交后才发一次最终完成flag，保持现有 epilogue 完成协议。
这是本地准备/消费协议调整，不是远端按chunk消费协议；不要求本轮改变 epilogue。
当前批容量/首批启动策略可继续复用，尾tile必须强制提交，不能为了更早退出漏掉最后批次。

预期收益：首批发送由“整个prepare区间完成”提前为“首个相关tile完成”；
剩余tile准备时间与WQE提交重叠。风险是tile太小导致直方图、DMA、ready和门铃固定成本放大。
候选tile大小应按典型及其他BS/K/P实测选择，而不是硬编码对某一8P配置最佳的值。

### D. 接收计数归约并行化放在收益确认之后

`GetRecvCount()` 处理的是 `[P][E/P]` count，不是 `BS*K`；典型8P/256专家仅256个int32有效项。
并行方案应按expert输出块/rank输出块分摊给剩余计算核，保留唯一actualA发布者，
确认所有消费者完成读状态后再由唯一owner清理通知。不能把原函数直接广播给全部计算核。
若主要时间是等最后一个peer通知，更多核不会缩短等待，反而会增加同步。

另可评估让一个计算核优先执行count接收、其他核先准备tile，但必须量化少一个prepare核的代价；
不能仅为Dispatch计时好看把这些输出的生成后移到epilogue，后者还有host同步和API依赖。

## SQ/CQ 与源数据生命周期边界

恢复 `cqe=0` 后，没有全请求CQE堆积的64项上限问题，但**单批WQEBB小于SQ深度不证明累计在途请求安全**。
本轮恢复原异步路径，不新增流控或深度断言。已读本地 `D:/code/asc-comm/src/aicore/hcomm/impl/platform_v310/hcomm_aiv_urma.h`
的 `BatchCommit(UbcBatchHandle&)`：该实现只检查本批 `preSqCnt`，按SQ环形位置拷贝并更新门铃，
没有按 `sqTail` 检查累计在途量。大BS、倾斜路由、大P的SQ覆盖风险属于恢复父版后仍存在的边界，
并非本次测试或源码对照已证明安全；远端实际安装的Hcomm版本也未经核对。
任何后续tile改造必须按每个Jetty累计未完成WQEBB审查，不能把512/256的batch容量误当成4096的在途上限。
本轮CPU提交模型也不声称证明SQ不覆盖或硬件队列深度正确。

描述符表容量需在tiling中按tile数、peer数和描述符大小精确分配；采用不复用整表时不存在环形槽覆盖，
但不能消除发送SQ的容量约束。若后续引入有界环形描述符队列，则生产者必须等待消费确认形成反压，
代际标记只解决新旧轮区分，不能替代槽位容量管理。

异步退出不代表输入可立即释放/覆盖。X、count workspace、metadata stash和completion source必须保持有效，
沿用原有stream/event调用链；调用者在安全的完成边界之前不能复用异步发送源。
跨rank下一轮窗口复用还依赖原协议，不能从本地epilogue返回推导所有rank都已退出。
尤其本rank epilogue只等incoming flags，不能独立证明本rank的所有outgoing源已被读完；
输入/临时count workspace的释放与复用仍需调用链提供完成边界。这是恢复父版的既有生命周期风险，
本轮没有把它包装成已验证安全；后续tile源也必须保持到对应远端完成边界，不能ready后立即覆写。

## 验证与验收

本轮检查入口：

- `python -B mc2/moe_ep_dispatch/tests/ut/op_kernel/test_dispatch_count_histograms.py -v`
- `python -B mc2/common/tests/test_moe_ep_async_completion.py -v`
- `git diff --check`

源码对照范围包含：未提交的全核计数/直方图、缓存分支、掩码尾块与分桶、UB别名/同步、workspace大小注释，
本轮两条发送路径的异步恢复，以及未改动的epilogue等待与清理消费者。
四维独立审查按用户指定使用 `gpt-5.6-sol`，主代理核对实际代码后处理如下：

| 维度 | 发现与处理 |
| --- | --- |
| 代码逻辑 | 未发现静态可确定的生产缺陷；主代理复核UB复用、cached同步、page/tail mask、原子行累计和prepare分片切换。 |
| Clean Code | 修正文档中笼统“Combine未变”：Combine仅恢复父版异步路径；将历史审查失败与当前四维结果分开。新增/修改生产函数未超过50个非空非注释有效行。 |
| 功能完整性 | 补独立cached slot oracle、同rank多有效项/混合slot用例；补actualA唯一/条件发布与cached slot拷贝源码契约。确认host禁止BS0，空任务只指空核片。 |
| 架构设计 | 补tile状态依赖、count totals先于通信、sender-peer最终flag、描述符容量/反压和源生命周期边界；明确累计SQ风险是父版既有边界，未偷偷新增Drain/断言。 |

主代理复核不把审查中的运行风险当作已验证结论：SyncAll/原子完成、Reg代码生成、真正的flag内存序和
Hcomm队列进度仍需设备验证。没有未处理的静态确认生产缺陷；上述SQ/源生命周期限制仍然存在。
补测和文档修订后，功能完整性与架构两个reviewer再次只读复核，确认原意见已处理，无新的确定未处理项。

本轮实执行结果：8项histogram CPU模型、12项异步提交/源码契约测试通过；`git diff --check`通过，
未跟踪文件的尾随空白扫描为0。源码对照确认Combine和两个epilogue与 `84d0aba660^` 一致，
Dispatch相对父版只保留前置计数及prepare分片切换优化；生产改动中没有assert、printf、返回值包装或残留Drain。
原同步测试模型由异步模型替换，历史benchmark文档保留并标明不描述当前行为。

本地没有CANN编译/NPU执行证据，CPU模型不能替代Reg指令兼容性、DMA可见性、实际原子行为或性能证明。
未得到本任务远端执行授权，不操作远端环境。
上机需同环境比较 `84d0aba660^` 与当前异步版，验证standalone/cached/continue精度、全部metadata/offset和六项计时。
覆盖小BS、非8P、AIV33/48/56/64等角色边界、K1/6/32、负数padding、同rank多expert和直方图128/256边界。
小BS空任务指 `BS<AIV` 的空核片；host当前拒绝 `BS=0`（`CheckInputTensorShape` 的 `xDim0<=0`），
不把整体零BS当成已支持能力。cached模型另有独立slot选择oracle，覆盖同rank多有效项和有效/无效slot混合；
这是分支语义验证，不承诺任意构造的cached slot都是上层API合法输入。
先报告Dispatch/Combine单项与Total，再判断是否达到50us目标；目前没有依据承诺具体微秒收益。
