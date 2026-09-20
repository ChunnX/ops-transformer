# Shared-Jetty 的 BS 分组、32/56 Jetty 对照与优化判断

2026-09-20 补充：用户明确32 Jetty只约束发送 owner，epilogue 保持全部可用计算核，
本轮不实施本文后续历史实验建议中的计算核数缩减。最新核内优化见
[metadata 批处理与 Combine 输入预取](moe_ep_pipeline_optimization_pass1.md)。
下文初版的运行时深度检查已在此前清理中删除，32/SQ4096/CQ64配置本身不变；
下文“尚未采集性能”属于历史状态，不覆盖用户已报告的最新性能数据。

日期：2026-09-19。此文补充并修订 [原优化方案](moe_ep_shared_jetty_bandwidth_optimization.md)。当前工作区已实现首版 32-Jetty 调度；尚未在 NPU 上编译、运行或采集性能数据，下面的带宽数字仍是容量估算和实验设计。

### 当前首版实现边界

- context 固定创建 32 个 Shared Jetty；每个 sender AIV 独占同编号 Jetty，运行时校验实际 SQ=4096 WQEBB、CQ=64；
- Jetty 和 AIV 的绑定不依赖 `epWorldSize`。32 个 sender 都遍历全部 peer，`peerIndex` 只用于在 MultiChannelHandle 内选择逻辑通道；
- Dispatch 独立把 BS 均分给 32 个 sender，prepare 则按 `aivNum - 32` 个核独立均分 BS。两种边界无需对齐：prepare 完成后通知所有与其区间相交的 sender；sender 只等待与自己区间相交的非空 prepare，不依赖卡数或固定 64 AIV；
- 每个 peer 的发送 slot 边界由原 `sendCounter[prepare][peer]` 前缀与边界 prepare 内有序 source-token 表的 lower_bound 得到。只读取已 ready 的边界分片；边界恰好落在 prepare 起点时直接用计数前缀，不读取下一个尚未 ready 的分片。没有新增完整 TopK 扫描、window 区域或 count exchange；
- Dispatch 每个 owner 在 payload 阶段只创建一个 MultiBatchHandle，跨 peer 复用；按实际 WQEBB 计量，首批阈值 32 WQEBB（最多 8 个完整 12-SGE 请求），后续阈值为原有 UB 容量 512 WQEBB，尾批强制提交。阈值前预留完整请求空间，空 peer flag 也计入预算；该策略不提供累计 SQ credit 保护，也不代表参数已经性能调优；
- Combine 由 0–31 号核各自绑定一个 Jetty；每个核遍历全部 peer，并把该 peer 的 metadata 连续区间按 32 份切分。payload 后的全核 barrier 已删除，flag source 改为发送前初始化；
- Dispatch/Combine epilogue 仍等待本轮全部 `worldSize × 32` ready 后再消费。本版尚未实现接收侧按 chunk 提前搬运/reduce，因此不能把发送侧重叠等同于端到端流水完成；
- 代码不再限制 8P，但 window/UB 等其他资源上限仍由 tiling 校验；典型 8P 只是测试形状，不是调度常量。

## 1. 结论与最新约束

建议优先验证 **32 个 Shared Jetty、每 Jetty 单一提交 owner、按 BS chunk 分配工作、chunk 内按 peer 聚合请求**。56 Jetty/SQ2K 继续作为必要对照，现有数据不能证明哪个在 D/C 中一定更快。

共享 Jetty 不要求按目标卡分组。当前每个 MultiChannel handle 已具备面向多个 peer 的能力；`peerIndex*lanesPerPeer+channel` 是旧 kernel 的调度选择。新实现由 sender `i` 固定持有 Jetty `i`，再用 `peerIndex` 选择该 Jetty 内的逻辑通道。上版将 `J=(P-1)*C_ctx` 写成主要设计约束过于保守，新方案不再要求 J 能被 P-1 整除，也不从 Jetty 数导出卡数上限。可用 peer 范围仍受实际 endpoint/拓扑、context rank 容量和 Hcomm 能力约束。

| 配置 | 每 SQ 深度 | 每 CQ 深度 | 全部 SQ 槽位合计 | 判断 |
| --- | ---: | ---: | ---: | --- |
| 32 Jetty | 4096 WQEBB | 64 | 131072 | 单队列容量更大，有利于较少提交核和多 peer 聚合 |
| 56 Jetty | 2048 WQEBB | 64 | 114688 | 更多独立队列，可能降低单队列长尾，但需更多 owner 或轮转 |

这些深度来自用户最新配置，P0 应读取实际 context 核对。合计槽位只是容量账本，不能跨 SQ 借用空闲槽位；每个 SQ 都要独立保证 outstanding 不越界。

用户观察：约 28–32 个 AIV 开 double buffer 可以打满 **GM/HBM 搬运带宽**。该结论用于接收侧 copy/reduction 和搬运核数实验；URMA 下发核数由 WQE 生成速率、队列并行、peer 调度、链路和 NIC/UBC 的 DMA 决定，必须单独测。也不能把某个连续 GM copy 测试的饱和核数直接等同于带 metadata 扫描、BF16 cast/add 的 epilogue 饱和点。

## 2. UBC_CTP 数据能支持哪些结论

本轮读取 `D:/数据/UBC_数据分析/UBC_分析.md`、`01_完整明细_MiB.csv`、`02_Jetty同形状对照_MiB.csv`、`03_Jetty峰值与大数据汇总_MiB.csv`、`04_Batch达标点_MiB.csv` 和 `05_对端数与mode4对照_MiB.csv`。共216条记录，汇总范围为 server 0 的 rank 0–7；测试中的每 Jetty 15 个 peer 不等于8P。

| 观测 | 数据 | 能支持的判断 |
| --- | --- | --- |
| J16→J32 最佳平均 | 310.16→318.48，原表标记 GB/s | 收益约2.68%，已进入增大 J 的收益递减区间 |
| J15、peer=1/2/4/8/15，大 shape | 315.87/316.88/318.11/313.75/310.60 | 多 peer 不必导致吞吐崩塌；peer 多也不保证更快 |
| J32 首个达到全组峰值95%的已测点 | BS32/H8192、记录 batch 0.5 MiB、309.94 | 小于数十 MiB 的 batch 也有接近平台的证据，但不是最小 batch 因果结论 |
| J4 大 shape | 约274降至约163 | 少队列可能出现明显回落，不能仅按 J 线性外推 |
| J56 | 没有记录 | 不能用该数据证明 J32 优于 J56 |

**计量口径需要修正。** 用峰值行的 `61440 MiB / 188.3948624 ms` 重算，得到 **318.480 GiB/s = 341.965 GB/s**。CSV 的 GB/s 标签实际是二进制口径。若400 GB/s为十进制理论值，峰值对应约85.49%，不是直接用318.48/400得到的79.62%。此处仅对已有 payload/时间做单位换算，没有证明硬件持续链路利用率达到85.49%。

外推限制：该峰值每 rank 发送60 GiB，D/C 典型 payload 只有约236/308 MB；H=16384也不同于7168。J 扫描随 J 增加总字节，peer 扫描也改变总字节；记录的 Drain 阈值8192和最大 batch2048未经实际 tiling/deployed binary核验。原始耗时已扣2 us，本轮不再扣。不能将这些数据当成32/SQ4K与56/SQ2K的受控对照，或把0.5 MiB直接写成最佳 batch 阈值。

因此数据支持的优先级是：先验证32 Jetty能否维持高位通信吞吐，并减少 owner/准备开销；同时保留56 Jetty，用固定流量实验检验尾部和实际D/C总时长。

也有较接近D/C数据量的证据：J32、15 peer、BS32/H8192时，每rank总发送240 MiB，耗时0.756192215 ms，得到309.941 GiB/s（约332.80 GB/s）。这比60 GiB峰值点更适合说明中等流量下仍可能有较高吞吐；但其H、peer数、路由与WQE组织不同，仍不能作为D/C实测替代。该行的0.5 MiB是记录中的单batch值，不是整个rank的总发送量。

## 3. 分组方案比较

| 分组 | 优点 | 主要代价 | 建议 |
| --- | --- | --- | --- |
| 按 peer，再切 token list | 目标连续，Dispatch容易6 slot/12 SGE聚合；完成状态简单 | peer倾斜时核/队列空闲，J和peer数绑定；32无法均分7个peer | 保留为基线 |
| 按 BS，token遇到哪个peer就马上发 | 每个token路由只处理一次；J独立于P | Dispatch碎片化，一slot双SGE约2 WQEBB；频繁切peer和小请求可能降速 | 不作为首选 |
| **BS chunk × peer bucket** | 按BS分配owner，保留peer内连续slot与多SGE；每Jetty真正多对端 | 需要chunk×peer计数/前缀，完成状态更多 | **Dispatch首选候选** |
| 原metadata行分块 × 多peer Jetty | Combine输入metadata/hidden读取相对连续，较易改现有发送器 | 不能直接表示一段原始BS的K路已齐 | **Combine低风险候选** |
| **目标BS chunk × source-rank bucket** | 对应token范围的K路齐后可提前reduce；SQ owner负载不按peer锁死 | 需要单次metadata分桶，hidden读取可能更散 | **Combine重叠候选，需与上一项对照** |

Dispatch和Combine不必采用相同的物理分组。Dispatch的BS是本卡输入token；Combine发送端的输入是expert输出/recv metadata，原token号属于目标卡，必须用 `(destinationRank, srcTokenIdx)` 标识，不能只按本地 recvX 行号宣称“按原BS分组”。

## 4. 通用 owner/Jetty/逻辑 chunk 映射

定义 `P=worldSize`、`J=实际创建的32或56个Jetty`、`S=发送owner核数`、`Q=逻辑BS chunk数`。J、S、Q独立；Q不是每peer lane数，也不要求等于J。当前公开接口要求 `P>=2`；1P 纯本地模式属于后续独立路径，不在本版实现中。

```text
jetty(q) = q % J
owner(j) = j % S
peerIndex(peer, localRank) = peer < localRank ? peer : peer - 1
```

同一 Jetty 永远由同一个 owner 持有并更新 batch cursor；owner可以串行驱动多个Jetty。一个chunk可以向多个peer发送，在同一外层 MultiBatchHandle 内：

```text
select peer with GetHandleRef(multiBatch, peerIndex, remoteAddress)
append that peer's WQE(s) immediately
select next peer and append WQE(s)
BatchCommit(multiBatch) when byte/WQEBB/latency threshold is reached
```

`GetHandleRef` 返回的是同一个内部可变 handle 引用，切peer会更新其remoteInfo；不能缓存多个peer引用供以后交错使用。每次选peer后立即构造对应WQE。不能为同一Jetty同时创建多个拥有独立sqHead快照的BatchHandle；切peer也不需要重新MakeBatchHandle或强制commit。

当前 Combine 的 `BeginPreparedWrites` 在同handle时提前返回，后续 `PrepareWrite` 会再次 `GetHandleRef`；多peer改造仍须检查每次写使用当前peer和注册窗口，尤其切换peer/window后的remoteInfo，不能只改GetCommHandle索引。

用固定 `Q_cap` 为一个protocol/context预留状态容量，用调用级一致的 `Q<=Q_cap` 控制遍历。接收端状态地址可采用 `(sourceRank*Q_cap+q)*stateStride`，发送端物理句柄只使用 `jetty(q)`，两者不能混为一个索引。Q或J变化必须在所有rank一致，并同步Host/Python布局、context缓存键、注册buffer容量、tiling、dump版本及解析器。拓扑有多个endpoint共享组时，映射还必须限定在各自合法共享组内。

## 5. Dispatch：BS分组但保留连续远端排布

把本卡BS切成Q个有序区间，区间内按目标peer去重；同一个token命中同一卡多个expert仍只发一次hidden。复用 `CalSendCnt` 已生成的去重rank信息，不额外反复扫描完整TopK。

```text
begin(q) = floor(q * BS / Q)
end(q)   = floor((q + 1) * BS / Q)
chunk(token) = ceil((token + 1) * Q / BS) - 1     // BS>0

n[q,p]    = 此BS区间发给peer p的去重token数
base[q,p] = sum(n[r,p], r<q)
```

只在同一个peer的连续目标区间内组合最多6个slot，继续使用hidden+meta的12-SGE WQE。不同peer可混在同一个batch，但不能拼成同一个连续目标multi-SGE写。以BS序排列chunk、chunk内token顺序不变，能够保持现有每source-rank接收窗口slot顺序；最终输出仍按原virtual-consumer切片与prefix恢复，不能改为简单chunk-major输出。

采用两步实施：先用完整token表建立chunk边界并改发送调度；再让prepare完成一个chunk后发布ready、sender开始发送。后者仍需要预先确定 `base[q,p]`，不能在前序计数未知时直接分配远端slot。prepare-ready含epoch、chunk、peer、count和base；每个数据区间只有一个producer。

接收侧也必须知道每个 `(sourceRank,q)` 的slot边界：首版扩展count exchange，发布本接收rank对应的 `n[q,p]`，并由双方一致的前缀规则得到base；边界数据必须先于其ready被消费。只有ready标志不能推导变长chunk的范围。接收核按该范围与原virtual-consumer slice求交处理，不另辟固定chunk保留区，避免改变主线payload排布。首版先交换完整边界再发payload；后续若流式发布边界，须单独验证计数、payload、ready的可见性顺序。

若复用GM中的metadata stash/source buffer，**BatchCommit不能作为可复用证明**。NIC可能尚未读完源数据。首版让源数据持久到远端完成/epilogue结束；若用有限ring复用，必须以对应完成checkpoint或接收完成反馈释放，而不是仅以“sender已经排队”释放。batch描述符UB在BatchCommit完成其搬出后可复用，与payload GM生命周期是两件事。

## 6. Combine：先验证简单多peer，再验证目标BS流水

第一步按现有metadata行分块给owner，peer切换通过同一MultiBatchHandle完成；保持remote slot=`srcTokenIdx*K+topkIdx`。它能验证32队列能否撑起发送速率，但仍采用完整发送完成协议，不宣称已经获得按BS提前reduce的收益。

第二步，针对每个目标rank的实际BS，单次扫描metadata生成 `(destinationRank, q)` bucket。所有source expert rank对同一目标rank使用相同token边界；不同目标rank的BS若不同，则必须通过已有协议/新增元数据显式交换，不能套用发送rank自己的BS。每个bucket可含不同expert的route，保留原slot与topk位置。

receiver在 `q` 上等齐所有source-rank ready后，让搬运/reduction核处理该token区间。空bucket也发ready（或使用已交换count显式跳过），本卡source由本地路径完成。某peer的某chunk若由多个物理SQ发送，则必须等待每个SQ的完成；首版令一个bucket由一个Jetty owner发送以简化证明。

当前 Combine 已在 payload 前由 AIV0 初始化持久 flag 源并做一次全 AIV 同步；每个 owner 随后按 peer 顺序直接提交 payload 和同 SQ 的有序 flag，payload 后没有全核 barrier。若继续做 bucket 级流水，需要让接收侧按 bucket 增量消费。强序 flag 只保证同一 SQ 的前序请求完成，不能代表其他 SQ。多 peer 共享 SQ 还会产生跨 peer 队头阻塞：某个 peer 慢，可能延迟该 SQ 后续 strong-order flag。按小批 peer 轮转、错开各 owner 首个 peer、避免所有 owner 按相同 peer 顺序长时间突发，并对照 2/4/全部 peer 组；选择依赖固定流量实验。

32 Jetty可先取Q=32，56 Jetty可先取Q=56作为owner均衡实验；典型BS4096对应128或73/74-token区间。Q与J的影响要分离：固定Q比较两种J时，映射覆盖的队列数为 `min(Q,J)`，有实际payload的队列还可能更少。Q=32、J=56只能作为“创建56但至多启用32”的资源控制组，不能检验56条活动队列的吞吐。Q=64覆盖两种J，但J56有8条队列各承载2块、48条各承载1块；可另取Q=224（32与56的最小公倍数）均衡块数，代价是块更小、通知更多。这些只是控制实验值，不是代码常量或推荐最优Q；每组都记录实际活动队列、每SQ字节和WQEBB分布。状态由原 `P*8` 增至 `P*Q`；Q=32/56时8P分别256/448个ready槽，必须计入状态空间、strong-order和通知开销。

## 7. 32与56的典型容量估算

假设8P、每卡BS4096、K6、H7168、BF16/FP16、均匀路由；其余形状使用真实bucket计数重新计算。Dispatch远端约15913个slot、236.277 MB，Combine远端21504条route、308.281 MB（带weight另加约0.086 MB）。以下按Q=J、各Jetty面向7个remote peer估算。

| 指标 | 32 Jetty/SQ4K | 56 Jetty/SQ2K |
| --- | ---: | ---: |
| Dispatch平均remote slot/Jetty | 497.3 | 284.2 |
| Dispatch payload/Jetty | 7.38 MB | 4.22 MB |
| Dispatch WQEBB/Jetty（每peer独立6-slot打包） | 约343 | 约203 |
| Combine平均route/Jetty | 672 | 384 |
| Combine payload/Jetty | 9.63 MB | 5.51 MB |
| Combine WQEBB，hidden-only | 约679 | 约391 |
| Combine WQEBB，hidden+weight | 约1351 | 约775 |

WQEBB含每remote peer一个1-WQEBB最终flag，未计count exchange和checkpoint；Dispatch按平均peer bucket向上取整估算，真实值取决于各bucket余数。新56多peer方案约203与旧56 peer-exclusive方案约191不能直接混用，后者每SQ只对一个peer发flag。

BS分组使Dispatch的纯逐token双SGE方案需要约2 WQEBB/slot，而每peer6-slot聚合约2/3 WQEBB/slot，descriptor压力可能相差约3倍。这是保留chunk内peer聚合的主要原因。

多peer共享后的SQ水位按**同一物理SQ上所有peer累计**计算。最坏专家集中到一张卡时，该卡Combine要向其他P-1卡归还数据；Q=J、目标BS chunk分组的保守上界为：

```text
routes_per_jetty <= (P-1) * K * ceil(BS/Q)      // Q=J，一个chunk/Jetty
```

典型32配置可到5376条route，56配置可到3108条（74-token最大chunk）；带weight约再乘2，两者都可超过对应SQ。Q>J时还要累加映射到该Jetty的所有chunk。不能因均匀平均安全就去掉credit控制；按实际每SQ计划的WQEBB，在越界前插入strong-order checkpoint+cqe=1，提交并回收后再发下一credit window。CQ64按实际物理CQ归属计数；若多个SQ共享CQ，必须汇总它们的outstanding CQE，不能每SQ各自假定拥有64个独立槽。

## 8. GM核数与URMA核数怎么配

如果设备有64个AIV，可用32 prepare/搬运核 + 32 URMA owner核作为融合流水候选；S=32时每个owner管理1个Jetty，或在J=56时其中24个owner再各管一个Jetty。28/32/36个GM核、16/24/32/更多sender核都只作为扫描点，最终由GM与WQE阶段实测决定。

当前发送与epilogue是独立kernel，发送侧预处理也不等同于epilogue大hidden搬运。因此第一步可在独立epilogue中扫描28–32个物理copy/reduction核，发送kernel单独扫描S，不必强行让两者核数相加等于64；只有同一融合kernel并行驻留的角色才共享这一预算。

改变物理consumer核数时，用其轮转执行原virtual-consumer slices，保持旧 `[virtualAiv][rank][expert]` 计数/输出顺序。不能直接把原aivNum从64改32，导致recv_x/cached recvXIdx顺序变化。

GM double buffer可以重叠MTE搬入/Vector处理/搬出，但需要UB容纳两份tile。URMA DMA也访问同一HBM，通信与本地搬运并发时会争用HBM控制器；隔离GM带宽打满和隔离URMA带宽打满，不能保证同时各自维持峰值。需要同时测 `T_GM_only`、`T_URMA_only` 和 `T_overlap`。

## 9. 哪些改动更可能有效

1. **验证32/56多peer对照与首批提交调整。** 当前 Dispatch 已使用首批32 WQEBB、后续512 WQEBB的跨peer batch。首批参数来自原方案候选范围，需实测与64 WQEBB等候选对照；提前首批可能降低启动延迟，但不能把发送kernel的75us全部算成可节省时间。
2. **Combine提前flag源初始化，解除payload后的全核flag barrier。** 输出布局无需变化，但只有接收端确实能提前消费时才能兑现大部分收益。
3. **Dispatch采用BS×peer bucket，Combine比较metadata分块与目标BS分桶。** 前者更有机会减少重复路由扫描并保持multi-SGE；后者分桶成本和源hidden离散读可能抵消收益。
4. **按ready chunk消费，同时测试28–32个GM处理核。** 这是降低epilogue总时间的主要候选，但必须证实等待区间和GM/Vector实际开销。

单位统一后，若D/C能达到数据集峰值341.965 GB/s，其典型payload传输预算约为Dispatch691us、Combine902us；400 GB/s理想预算为591/771us。这里只是体积除以带宽，不是预测D/C总时间。总时间还取决于计数/分桶、issue gaps、ready tail、输出copy/reduction及重叠竞争。

净收益判断应记录：减少的暴露等待、减少的重复扫描，与新增分桶/prefix、peer切换、ready通知、SQ checkpoint、GM争用的差额。只有actual-byte goodput和四kernel总时长同时改善，才能接受；sender时间更短但epilogue更长不算优化。旧文档的0.90–1.10/1.30–1.55ms属于探索目标，不能凭本次资料确认。

## 10. 最小实验矩阵与判退条件

| 实验 | 固定项 | 变化项 | 回答的问题 |
| --- | --- | --- | --- |
| E0 peer-exclusive基线（32需适配） | 同一BS/H/K/路由/权重/版本 | 32与56资源配置；32按peer不等额分配，需实现后验证 | 原分组策略下两配置差异，含32映射适配影响 |
| E1 metadata/token表就绪后只测传输 | 相同真实payload、WQE大小、Q、首批字节、总提交字节 | J32/SQ4K vs J56/SQ2K；固定S，再扫描S | Jetty数或owner数是否带来收益 |
| E2 Dispatch调度 | 输出/路由一致 | peer分组 vs 逐token BS vs BS×peer bucket | multi-SGE保留是否必要 |
| E3 Combine调度 | same remote slot与FP32累加顺序 | metadata分块 vs目标BS分桶 | 分桶是否换来可见的ready提前 |
| E4 GM处理 | 旧virtual-consumer排布、相同dtype/tile | 28/32/36/全核；double buffer开关 | 本算子的GM饱和核数 |
| E5 组合流水 | 最优单项配置 | 独立 vs并发；Q固定和Q扫描分开 | HBM争用与真实净收益 |

E1每Jetty-peer数量对照应固定总payload和每commit总bytes/WQEBB，分别报告每peer bytes，不重复mode3“每增加Jetty就复制一份全量BS”的流量增长。32/56配置同时改变SQ深度，结果只能归因于整套配置；若想单独归因J，要在设备允许时补同SQ深度控制组。

E1的主要活动队列对照应使用固定Q≥56，并遵循第6节的负载均衡与活动队列核验；Q32只作为闲置资源控制组。固定Q不等于固定每SQ负载，必须同时报告其分布，不能把Q64下56队列的2:1分块差异归因于硬件。

记录每物理Jetty的payload bytes、WQEBB、commit/checkpoint数、first/last issue、各peer ready spread、GM等待/有效处理、Total与慢rank；单卡cycle不可跨rank直接相减。correctness覆盖非整除BS/Q、BS<Q、BS=0、1P、多P、非均匀BS（若接口支持）、全空peer、极端expert倾斜、权重、cached和连续多轮。BS=0不计算反向chunk公式，但仍完成约定的空bucket通知。

所有外部数据来自本地静态文件；未连接远端NPU。资料未包含56 Jetty实测，且旧benchmark队列参数与当前两配置不同，故最终推荐为“32优先实验，56保留对照”，不宣称已获得加速。

## 11. 源码与方法依据

- `mc2/common/torch_extension/csrc/moe_shared_jetty_context.cpp::GetHcclCommResource`：每个Jetty用全部channelDescs建立MultiChannel，证明可面向多个peer。
- `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h::GetOwnedJettyHandle/SendTokenChunkToPeer`：sender按AIV号固定选择物理Jetty，再遍历peer发送该BS chunk；每个payload WQE保持连续slot的multi-SGE排布。
- `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h::BeginPreparedWrites/PrepareWrite/PrepareMultiSgeWrite`：outer handle生命周期、每次写重新选择peer，以及1/2-WQEBB路径。
- `D:/code/asc-comm/src/aicore/hcomm/impl/platform_v310/hcomm_aiv_urma.h::GetHandleRef/BatchPostSend/BatchCommit`：GetHandleRef更新同一内部remoteInfo，BatchPostSend编码当前peer，BatchCommit提交共享SQ。
- `D:/code/asc-comm/docs/zh/api/aicore/hcomm/GetHandleRef.md`：通过peer引用加入任务，通过外层MultiBatchHandle提交；注册窗口范围必须匹配。
- `D:/数据/UBC_数据分析/01_完整明细_MiB.csv`：峰值行原来源行号108，61440MiB/188.3948624ms的单位复算；其余汇总见第2节。
- `D:/code/cannbot-skills/ops/ascendc-perf-optimize/SKILL.md` 和原方案引用的MoE分核/同步资料：分别测通信、GM/计算、并发膨胀；按阶段划分核角色并保持状态完成语义。
