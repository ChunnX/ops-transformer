# Combine epilogue：共享本地逆向索引

## 作用与范围

基于 `69d4d380c8`，本轮只修改 Combine epilogue 的本地索引构建，以及对应的 host workspace 大小。

Combine 的输入 `x` 是专家输出行，归约输出按原始 token 排列。metadata 中的
`[src_rank, src_token, src_topk, slot, recvXIdx]` 给出两者的关系。例如：

`localRecvIndex[37 * K + 2] = 891`

表示本 rank 的 token 37、第 2 个 TopK 位置对应本地 `x[891]`；没有本地结果的项为 `-1`。
这不是新的网络路由表，也不替代远端 completion。

原实现让每个有输出 token 的 AIV 扫描整个本 rank metadata 区间，只留下属于自己的 token。
新实现将这个区间按行均分给全部实际 AIV，每条记录只被一个 builder 扫描，形成公共索引后，
各输出核将自己的 BS 切片一次搬入 UB。归约循环仍只访问 UB 索引，没有增加逐 TopK 的 GM 标量读取。

## 实现与同步

1. 仍按 BS 均分输出 token。各核将自己的索引切片初始化为 `-1`，MTE3 完成后参加全核屏障。
   初始化覆盖整个表，不能依赖 workspace 首次为零，也不能让初始化覆盖其他核刚写出的有效索引。
2. 独立按本 rank metadata 行数分工，**没有输出 token 的核也可以参与构建**。
   GM metadata 仍是每行 5 个 int32；搬入 UB 时每行补齐到 32 B。
3. 在 UB 中把该行 `recvXIdx` 暂存到已不再需要的 `src_rank` 位置，取得 32 B 对齐的 DMA 源地址。
   使用 `DataCopyPad` 向共享表的 `(token * K + topk)` 位置写入恰好 4 B。
   没有修改原 GM metadata，没有 GM 标量缓存写回，也没有引入原子操作。
4. 每块结束等待 MTE3 完成，再复用 metadata UB；`MTE2_S`、`S_MTE3`、`MTE3_S`、`S_MTE2`
   约束搬入、标量读写、搬出与下一块搬入之间的依赖。
5. 0 号核保留原远端 completion 等待与清理。复用原归约前 `SyncAll<true>()` 发布所有 builder 的结果；
   输出核随后批量读回自己的索引切片，继续原预取、FP32 TopK 顺序累加与输出流程。

因此只新增一次初始化后的全核屏障；不新增 completion、不提前清 flag、不改变网络完成协议。
所有核都经过两处屏障，不能因本核没有输出或 metadata 就提前退出。

有效 Dispatch handle 保证本 rank 区间中 `(src_token, src_topk)` 唯一；同一 token 的不同 TopK 位置是
不同索引项。畸形的重复 key 不属于有效 handle；本实现不为其增加设备断言或原子仲裁。
保留原有负值/越界 token、TopK、recvXIdx 过滤，以及实际行数和本 rank 区间的截断语义。

### 非对齐搬运依据

UB 源行按 32 B 对齐，GM 项按 int32 对齐。官方 `DataCopyPad` 说明：非对齐 UB→GM 搬出只写有效
字节，补齐的 dummy 字节在写入 GM 时丢弃；GM→UB 多块搬入分别补齐，因此这里使用 20 B 的 GM 行、
32 B 的 UB 行和 4 B 的 GM scatter，不使用会带出冗余数据的普通整块 `DataCopy`。
参见[昇腾官方非对齐搬运说明，第 3 节](https://www.hiascend.com/developer/techArticles/20250627-1)。
同一 cache line 的不同项仍可能产生访存竞争；实际多核 DMA 可见性和代价需要 NPU 验证。

## 内存与性能预期

- 私有 GM workspace 新增 `Align512(BS * K * sizeof(int32_t))`，使用 64 位大小计算。
  典型 BS4096/K6 为 **96 KiB**，与卡数、Jetty 数无固定绑定。
- 沿用现有 Dispatch epilogue 的 user-workspace 前缀约定，host 仍额外保留原系统 workspace。
  不改变 tiling 结构、context 长度、外部 metadata、远端窗口或算子参数。
  Ascend C 算子工程传入的 workspace 已是用户区，无需再加系统区偏移，见
  [官方 workspace 冗余操作说明](https://asc.gitcode.com/guide/operator_practice/simd_operator_optimization/overhead_optimization/remove_workspace_redundancy.html)。
  两处无参 `SyncAll<true>()` 沿用纯 AIV 硬同步，不使用这张 GM 索引作为软件同步区。
- 每核 metadata UB 不超过原来的 `Align32(max(1, min(actualA, 128)) * 20)` 字节预算；
  据此反推每批 32 B UB 行数，最多 80 行、2560 B，再按本核实际记录数缩小。
  这避免补齐后膨胀到 4 KiB，挤掉大 BS/K 下最后一组输入 buffer。
  典型均匀 8P、BS4096/K6、56 AIV 时，本 rank 约 3072 行，每 builder 至多 55 行，约 1760 B。
  `CalcBufferNum()` 继续根据实际 UB 占用决定输入队列容量；本次不会因 metadata 增长降低该容量，
  也不固定计算核数。
- 在上述典型分布下，metadata 记录检查从 56×3072≈17.2 万次变为约 3072 次。
  这是总工作量的下降，**不代表时延缩短 56 倍**：原各核并行工作，也可能命中缓存。
- 新成本包括表初始化/回读、4 B scatter 和一次全核屏障。metadata 很少时收益可能为负；
  对热点写入、不同卡数和小 BS 必须实测。该优化不减少 URMA payload，不保证链路瞬时带宽增加。

32 Jetty / SQ4096 / CQ64、32 个发送 owner、全部可用 epilogue 计算核、Hcomm、多 SGE、
完成标志与 TopK 累加顺序均不变。没有新增设备 printf、assert、Trap、返回值检查或 `(void)` 包装。
Dispatch 已修复的流水同步未触碰。

## 验证与部署

- 独立 C++17 CPU 模型覆盖 BS=1/7/33/129/4096、K=1/6/32、AIV=1/2/7/32/33/56/64/100，
  全/稀疏/空本地路由、打乱 metadata、动态 UB 分块尾部、负值/越界、容量截断及同一 workspace 多轮复用。
  对照原逐核扫描，检查完整且唯一的初始化、每行只扫描一次、每个有效 key 只有一个 writer、
  20 B→32 B 布局、4 B scatter 边界和最终索引一致性。
  首版（尚未按旧 UB 预算收紧 chunk）**2880 组运行通过**；修正版的同组模型已更新并编译通过，
  但新生成的 exe 被本机组织 Device Guard 阻止执行，未将其记作再次运行通过。
- 另检验不同 actualA/本核行数下 metadata UB 不超过旧预算，覆盖 248/256 KiB UB、56 AIV、
  K32/H8192、有 weights 的单 buffer 临界形状（每核分别 560/592 个 token）。
  独立 PowerShell 数值检查已通过 131841 组预算组合和这两个临界形状，修正后的预算不大于旧版。
- 模型使用 MSVC C++17 `/O2 /W4 /WX` 编译；它不执行 AscendC kernel，不模拟 DMA 时序，
  即使运行通过也不能替代 CANN 编译、多 rank 正确性和性能回归。
- 既有 Dispatch 分工、3072 组 Dispatch 预取、8192 组 Combine 调度 CPU 模型重新编译运行均通过。
- `git diff --check` 通过；变更的生产函数及新模型函数不超过 50 个非空、非注释行。
  原 completion 等待/清理、TopK 输入预取/消费/累加、队列容量计算函数与 HEAD 逐函数比对未变。
- 已完成代码逻辑、Clean Code、功能完整性、架构四维独立子 agent 审查。
  功能审查发现初版 padded metadata 占用增长会破坏单 buffer 临界形状，已按旧预算限制 chunk 并补模型；
  逻辑、完整性和架构复审确认该问题修复。workspace 入参语义疑问经工程文档和上下游确认，未增加冗余转换。
  没有遗留静态审查阻塞项，设备验证限制仍保留。
- 本地未执行 CANN 编译或 NPU 测试。部署须同时重建/安装此次 host tiling 与 kernel，不能只替换 kernel
  而保留旧 workspace 申请大小；本轮没有 wheel 接口变更。
- NPU 回归应覆盖 standalone/continue/cached、weights 开关、本地为空/全本地/混合路由、小 BS、
  不同 P/K、单/双输入 buffer。比较相同环境下 Combine epilogue 和 Combine Total，而非只比较提交耗时。

测试源：`mc2/moe_ep_combine_epilogue/tests/ut/op_kernel/test_shared_local_recv_index.cpp`。
