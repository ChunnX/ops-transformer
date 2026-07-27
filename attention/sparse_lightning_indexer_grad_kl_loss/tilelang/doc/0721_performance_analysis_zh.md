# 0721 TileLang 性能分析

> 本文保留早期 profiling 和实验过程。当前 4 核基线、官方对齐状态、后端能力
> 边界及最新优化路线请优先阅读
> [0721_ascendc_alignment_summary_zh.md](0721_ascendc_alignment_summary_zh.md)。
> 通用开发方法见
> [tilelang-ascend优化实操指南.md](tilelang-ascend优化实操指南.md)。

## 范围与基线

本文记录 Case 0 下已经通过精度验证的 TileLang 0721.py 实现：

~~~text
B=1, S1=1, K=2048, Query=128x512, QueryIndex=64x128, TopK=128
~~~

官方对比对象：

- ../op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_service_cube.h
- ../op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h

已稳定的精度边界：

~~~text
dQueryIndex: max_diff=9.765625e-04，rel_err 约 7.8e-05
dKeyIndex:   max_diff=1.459122e-04，rel_err 约 1.8e-04
dWeights:    max_diff=2.441406e-04，rel_err 约 9.4e-06
Loss:        本测试用例中完全一致
~~~

重复测试中，TileLang 中位数通常为 0.474-0.487 ms。0.472 ms 是单次最佳
值，尚不能当作新的性能基线。对应测试中官方中位数约 0.287-0.299 ms，故
0721 当前约为官方 0.60-0.63x。若官方为 0.29 ms，达到 80% 需要 TileLang
不高于约 0.366 ms，仍有约 0.11 ms 的差距。

性能结论必须依据重复中位数和命名 kernel 的 profile，不能只看一次 Python
端到端时间。

## 当前 0721 流水

0721 是单个混合 AIC+双 AIV kernel，使用手工同步
(TL_ASCEND_AUTO_SYNC=False)。当前形状 B*S1=1，只有 cid=0 负责主要
C1/V1/C2 工作。官方 profile 同样只使用一个主 core，因此其他空闲 core
不是可以直接分摊的计算资源。

1. **V0 gather**：AIV 将 Key、KeyRope、KeyIndex 写入 GM workspace，并按
   每个 1024-key 外层块发布 P、SY 的就绪标志。
2. **C1-P**：AIC 计算 128x2048 score。每个 128-key tile 将 4 个 Query
   chunk 与 1 个 Rope chunk 累加到 bmm1Res。
3. **C1-SY**：AIC 使用 QueryIndex 与 KeyIndex 计算 64x2048 ReLU 投影，
   通过 FIX 写入 bmm2Res。
4. **V1 P/SY**：AIV0 由 bmm1Res 生成 PReduced，AIV1 由 bmm2Res 生成
   SoftmaxI；随后经过 AIV->AIC->AIV 汇合才进入 backward。
5. **V1 backward**：两个 AIV 分别处理 32 行，计算 reluGrad、dWeight、
   Loss partial。每个 128 列 K tile 使用两个 32x64 物理向量子块，FP32
   计算后用 CAST_RINT 转为 FP16。
6. **C2/V2**：C2 生成 dQuery 累加与 dKey partial；V2 确定性地汇总 dKey、
   转换输出、归约 Loss。

0721.py:481-600 已保留 C1-P 的单槽 Key-L1 预取：MTE1 消费当前 Key tile
到 L0B 后，MTE2 在当前 MMA 期间搬运下一个 Query 维度或 Rope tile。
0721.py:610-661 的 C1-SY 仍以单槽流水为主。

## 最新 V1 等待拆分

当前生成 C++ 已符合预期：

~~~cpp
if (vid == 0) {
  AscendC::CrossCoreWaitFlag(4); // C1-P / bmm1Res 就绪
} else {
  AscendC::CrossCoreWaitFlag(6); // C1-SY / bmm2Res 就绪
}
~~~

后续汇合保持不变：

~~~cpp
CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
CrossCoreWaitFlag(12);
// AIC 等待 10，随后发布 12。
~~~

因此 AIV0 可以在 C1-SY 期间执行 P；backward 仍需等待 P 和 SY 均完成。
该改动已在硬件上验证不死锁、精度保持基线。profile 约减少 5 us，这是其
合理上限，无法消除 C1-SY、P/SY 汇合、C2、V2 的后续关键路径。

## Profile 解读

修改前，core 0 的近似时间线为：

~~~text
0-30 us     V0 独占，AIC 等待 gather 数据
30-68 us    V0 与 C1-P 重叠
68-105 us   C1-SY 独占，两个 AIV 等待
105-142 us  V1 P/SY 前半段
142-172 us  V1 backward，AIC 等待
172-185 us  C2，两个 AIV 等待
185 us 之后  V2，AIC 不再计时
~~~

等待拆分把 AIV0 的 P 移入 68-105 us 的 C1-SY 区间。它是正确的小收益，
但真正的瓶颈仍是 Cube/Vector 的物理流水重叠不足。

## 与官方 AscendC 的差异

### 1. 官方有物理 Cube ping-pong，0721 没有

官方 Cube service 明确分配：

~~~text
L0A: 2 x 32 KiB
L0B: 2 x 32 KiB
L0C: 2 x 64 KiB
~~~

见 sparse_lightning_indexer_grad_kl_loss_service_cube.h:77-80、276-290。
l0abPingPongFlag 和 l0cPingPongFlag 轮换两个精确物理槽，配合多个 L1
buffer 和 event，使未来 tile 的 MTE2、另一 tile 的 MTE1、当前 tile 的
MMA、较早结果的 FIX 可以重叠。官方 profile 中多个短 Cube 段、前几段被
Vector 掩盖，来源正是这套物理流水。

最新 0721 生成 C++ 的总容量并不小：

~~~text
A2  = 64 KiB
B2  = 64 KiB
CO1 = 128 KiB
~~~

但 TileLang 静态地为 C1-P、C1-SY、C2 分配不同形状和偏移。C1-P 需要
128x128，C1-SY 需要 64x128，C2 又有其他形状。不能以大 L0 父分配切片
伪造小 L0 矩阵：生成的 L1->L0 描述符可能继承父分配物理形状。此前尝试已
产生 illegal instruction 或错误结果。这是与官方的最大结构差异。

### 2. C1-SY 与 C2 的搬运/计算重叠较少

C1-P 已有单 L1 槽预取，但没有第二个 L0/L0C 槽。C1-SY 与 C2 仍存在更多
GM->L1、L1->L0、MMA、FIX 的串行依赖，无法像官方的多 L1 buffer 和
ping-pong 物理槽一样持续重叠。

### 3. backward 有必须保留的正确性串行化

生成 C++ 在 UB offset 20992 复用：

~~~cpp
v1_bw_bmm2    // FP32
v1_bw_rg_half // FP16
~~~

下一次 MTE2 填充前必须等待 MTE3 完成 reluGrad store。该 MTE3_MTE2 的
WAR 依赖修复了 reluGrad、dQ、dK 的非确定性 NaN/Inf；当前单槽 backward
不能在 FP16 store 期间自由预取下一 BMM2 tile。该依赖不能为了性能删除。

### 4. V2 更保守

0721 保留确定性的全任务输出汇总和阶段边界。官方 Vector2 更晚、更紧凑。
V2 不是当前 shape 的主 Cube 关键路径，但仍有端到端开销差异。

### 5. core 数不是瓶颈

官方同样只使用一个主 core。将单一任务拆到 24 core 会引入 score、P/SY、
梯度和输出的跨核归约；当前规模下同步和 GM 成本将大于收益。

## 已尝试改动

| 改动 | 结果 | 结论 |
| --- | --- | --- |
| gather 移入 V0 并检查 debug workspace | 初始全零，随后修复 | 根因是 caller workspace ABI/index，不是 gather DMA；必须审计生成 call(...) 参数。 |
| 分阶段 sentinel 与 host 公式 | 成功定位 | 可按 gather -> BMM -> P/SY -> reluGrad -> dQ/dK 定位；仅调试使用。 |
| PReduced、SoftmaxI、Loss 修复 | 成功 | 正确公式与对齐 Loss partial 消除了 Inf Loss；保持当前归约顺序。 |
| 仅缩小 reluGrad 向量 tile | 不充分 | 32x128 改两个 32x64 不能单独修复 NaN，根因是 buffer ownership。 |
| MTE3_MTE2 WAR 保护 | 必须且成功 | 消除 BMM2/FP16 store 覆盖竞争，必须保留。 |
| V1 backward ping-pong | 精度失败 | 重现不安全的别名/生命周期行为，当前 UB 布局下不可重试。 |
| 每任务 dKey scatter 并减少同步 | 精度失败 | atomic/scatter finalize 顺序导致 NaN 或大误差，保留确定性 V2。 |
| C1-P/C1-SY 拆为两个 task | 回退 | 静态 L0 规划未恢复官方物理复用，未带来稳定收益。 |
| 父分配切片实现 L0 ping-pong | 否决 | L1->L0 描述符可能带父形状，对 MMA/FIX 不安全。 |
| 修改 pipeline_delay | 无结构收益 | 生成 C++ 已让 AIV 跑 V0 时 AIC 前进到 C1 并等待跨核标志。 |
| Query/Rope 常驻 L1 | 精度安全，收益不稳定 | 减少部分 GM 读，但不能消除 C1-SY/C2 串行。 |
| C1 Key/Rope 单槽预取 | 保留 | C++ 确认 MTE1 消费后 MTE2 refill，是小而安全的基线收益。 |
| C1-SY 或 C2 单槽预取 | 持平或变慢 | 控制开销无法突破单 L0/Cube 依赖，已回退。 |
| V1 按分支等待 C1 | 有效但很小 | 不死锁、精度安全，profile 约减少 5 us，可保留。 |
| C1-SY QueryIndex 常驻 L0A | 已保留，收益未量化 | C++ 已确认将 16 次 L1->L0A 装载收敛为一次；精度保持基线。五轮整体中位数约 0.488 ms，未证明稳定加速，但也无回归。 |
| C2 dKey QueryIndex 常驻 L0B | 已回退 | C++ 与精度正确，但一次预载形成 C2 MTE1 前导，未与首轮 reluGrad/KeyIndex GM->L1 重叠；性能波动扩大且无稳定中位数收益。 |
| V0 P/SY 两阶段 gather | 已回退 | C++ 完全按预期先发布 P、后处理 KeyIndex，但五轮中位数为 0.474-0.510 ms，无稳定收益且扩大端到端波动。 |
| C1-P 精确物理双缓冲 | 已回退 | C++ 物理 shape/offset 正确且 `PIPE_M` barrier 已生成，但仍触发 L0C read/write conflict；当前手写 `T.mma` event 模型不能安全复现官方 ownership。 |

## 不应重试

1. 不增加 kernel，启动和 GM 往返成本高于微优化收益。
2. 不删除 MTE3_MTE2 backward WAR 边。
3. 不在已别名的 UB 地址上重试 backward ping-pong。
4. 不将 dKey scatter 放回每任务流水。
5. 不用大 L0 父分配模拟小形状 L0。
6. 不在无完整归约成本模型时使用空闲 core。
7. 不以一次 0.472 ms 接受性能改动。

## 可继续对齐官方的方向

### 优先级 1：精确物理 C1 双缓冲，先过可行性门槛

这是唯一有足够理论空间接近 0.366 ms 的方向，但不能再做源码级别名尝试。
在硬件测试前，生成 C++ 原型必须证明：

1. C1 有两个精确 32 KiB L0A 槽、两个精确 32 KiB L0B 槽、两个精确
   64 KiB L0C 槽。
2. 每个 L1->L0 目的形状等于对应 MMA 的物理矩阵，不存在父形状泄漏。
3. 每槽 MTE2/MTE1、MTE1/M、M/FIX、FIX/M event 均已 seed、成对且 drain。
4. C1-SY 与 C2 后续仍能使用各自精确形状，且没有非法物理重叠。
5. profile 实际显示下一 tile 搬运与当前 MMA/FIX 重叠。

若 TileLang-Ascend 无法从 0721 源码产生该布局，应视为当前 codegen 限制。
强行别名会重复已知的正确性失败。支持 phase-scoped 精确 L0 分配/复用需要
上游能力，且超出当前“不修改 tilelang-ascend”的约束。

### 优先级 2：更细粒度 V0->C1 就绪，先以 profile 证明

V0 在 C1-P 前约独占 30 us。0721 按 1024-key workspace block 发布就绪；
官方有更细的 L1/event 缓冲。可研究更小 K-ready 槽，让 C1 消费当前槽时
V0 生产下一槽。

前提是 ownership 表能证明 V0 不会在 C1 MTE2 消费前覆盖 workspace tile，
并具备 producer-return flag、合法 ping-pong workspace、正确 tail 处理。
还必须确认新增跨核 flag 不会抵消节省的气泡。先审计官方 V0/C1 flag cadence，
不能只按可视化时间线猜测实现。

### 已回退：C1-P 精确物理双缓冲

当前 C1-P 只有一个 128x128 Key L1、L0A、L0B 和 FP32 L0C。MTE1 必须等当前
MMA 归还同一 L0A/L0B 后才能装入下一 128-column operand，这正是 profile 中
C/V WAIT_FLAG 的主要来源。官方 `SLITMatmulService` 则按 d 维轮转两套 L1、
L0A/L0B，按 K inner 轮转两套 L0C。

本实验不使用父分配切片。P scope 分别分配两套精确 `128x128` L1 Key、L0A、
L0B 和 `128x128 fp32` L0C；偶数/奇数 d_idx 分别使用 slot 0/1，并在 MTE1
消费当前 slot 后向另一个 L1 slot 预取下一 operand。偶数/奇数 k_inner 使用
独立 L0C，允许上一输出的 FIX 与下一 K tile 的 MMA 重叠。

L0A/L0B/L0C 容量无法同时容纳 P 的双槽和 SY 的独立精确形状，因此 C1 被拆为
顺序 P scope 与 SY scope。P 在离开前 drain event 0/1；SY 只在后续 scope 分配
原有 64x128/128x128/64x128 buffer，允许 codegen 回收 P 地址。生成 C++ 必须同时
证明两套 P L0 地址彼此不重叠、MMA/FIX 模板仍为 128x128、且 SY 地址来自已结束
的 P lifetime。任何容量、offset 或 event 审计失败都必须立即回退。

两次运行均确认 P L0C 位于 offset 0 和 65536，L0A/L0B 也完全独立，MMA/FIX
模板为正确的 128x128；因此容量、shape 和 parent-slice 均不是根因。第二次 C++
还确认每个中间 P MMA 后都生成了 `PipeBarrier<PIPE_M>`，但设备仍报告相同的
`L0C read/write conflict`。这证明问题不再是 M pipe 内的可见性，而是当前手写
`T.mma` 路径无法表达官方 L0C ping-pong 所需的完整硬件 ownership 协议。继续添加
barrier 或复用 event 只是在同一未建模边界上试错，已回退到精度验证过的单槽 C1。
双槽源码保留为未调用实验宏和 C++ 证据，但主流程不再进入该路径；在不修改
tilelang-ascend 的约束下，此方向应视为不可安全移植。

### 已回退：V0 P/SY 两阶段 gather

官方 `ProcessVector0` 并不在同一稀疏行中同时 gather 三种数据。它先遍历全部
K outer block，gather `Key` 与 `KeyRope` 到 P workspace，并在每个 block 完成后
发布 `SYNC_V0_TO_C1_P_FLAG`；随后才遍历全部 block，gather `KeyIndex` 并发布
`SYNC_V0_TO_C1_SY_FLAG`。C1-P 因而不会等待它根本不读取的 KeyIndex。

0721 原实现将三种流混在同一个 UB ping-pong 循环中，并且在三者都写入 GM 后
才同时发布 P/SY flag。当前改动把该循环拆成两个静态阶段，继续复用既有的两组
UB 槽和 `MTE3_MTE2` event 0/1：P 阶段只读写 Key/KeyRope 并只发布 P flag；SY
阶段只读写 KeyIndex 并只发布 SY flag。无效 sparse index 的零填充分别保留。

验收时必须检查生成 C++：所有 Key/KeyRope 的 GM->UB 与 UB->GM copy 必须位于
首个 P flag 之前；全部 KeyIndex copy 必须位于 P 阶段之后且 SY flag 之前；不得
出现逐 sparse-row 的运行时 phase 分支。该改动不触碰 MMA、归约、输出和 workspace
ABI，预期只缩短 V0 到 C1-P 的依赖前缀；收益应以重复中位数和 profile 为准。

结果表明 codegen 和数值均正确：P flag 0/1 位于所有 KeyIndex GM->UB 之前，
SY flag 2/3 位于其后，精度保持基线。但五轮 TileLang 中位数为
0.510/0.474/0.494/0.491/0.495 ms，整体中位数约 0.491 ms；官方对应
0.286/0.296/0.312/0.273/0.301 ms。它没有相对于 0721 基线证明可重复收益，
因此回退。

原因不是 C1-P 仍在等待 KeyIndex，而是该等待只位于第一个 1024-key block 的
起点。C1-P 必须完成两个 outer block 才能发布给 V1；拆开的 KeyIndex gather
随后与 C1-P 的 GM->L1 搬运并发，不能缩短单槽 C1 的 16 次 MMA/FIX 串行关键
路径，反而可能增加 GM 竞争和测量离散度。官方的两阶段顺序依赖其自身的 gather
批量 DMA 和 C1 真正的 L0/L0C ping-pong，不能单独移植为 0721 的性能突破。

### 优先级 3：减少 V1 backward 的真实工作

低风险的 V1 等待空洞已经利用完。剩余成本是两个 32x64 子块、barrier、
broadcast、cast、reduction、store。32x128 路径或真正独立 UB 双缓冲可能
减少指令数，但必须先以独立最小用例证明当前 TileLang 向量 intrinsic 支持
4096 元素路径，且生成 UB offset 不会不安全别名。在此之前它只是研究方向。

### 已完成：C1-SY 不变左操作数常驻 L0A

当前 shape 的 DQueryIndex=128，因此 C1-SY 的 QueryIndex 只有一个 64x128
块，并对 16 个 K tile 完全不变。原实现每个 K tile 都从 L1 重载一次 L0A；
当前实验仅在 C1-SY 前预载一次，循环内只更新 KeyIndex 的 L0B。该改动不增加
buffer、不改变 MMA 累加或 FIX 顺序，也不改变 V0/C1、C1/V1 的跨核协议。

生成 C++ 已确认 L0A copy 位于 C1-SY K 循环外，循环内只保留 KeyIndex 的
L0B copy、MMA、FIX。五轮测试精度均保持基线；TileLang 中位数为
0.460/0.483/0.488/0.498/0.500 ms，整体中位数 0.488 ms，对应官方整体
中位数 0.300 ms。它没有证明稳定加速，但消除了确定的冗余搬运且未引入回归，
因此保留。它的理论上限只是 15 次 16 KiB L1->L0A 搬运和相应 MTE1 调度，
不能替代物理 C1 双缓冲路线。

### 已回退：C2 dKey QueryIndex 常驻 L0B

C2 在进入 n_idx 循环前已经将完整 QueryIndex 载入 c2_qi_l1，但原实现仍在
每个 n_idx 中重复执行 c2_qi_l1 到 c2_dk_qi_l0b 的 64x128 L1->L0B copy。
c2_dk_qi_l0b 位于 L0B offset 0，而同轮 dQuery 的 c2_dq_gki_l0b 位于另一
物理区域，因此 dKey 的 QueryIndex 可以在循环前一次预载，并跨 16 次 dKey
MMA 保持只读。

该实验不增加 buffer，也不改变 dKey/dQuery 的 MMA、FIX、GM 写回顺序。生成
C++ 与精度均正确，但预载发生在首轮 reluGrad/KeyIndex GM->L1 发射之前，形成
C2 的 MTE1 前导，未获得预期重叠。实测性能波动增大且无稳定中位数收益，因此
已回退；不能将“减少 copy 次数”直接等同于缩短关键路径。

### 优先级 5：Cube 改善后再处理 V2

只有 Cube 流水有真实改善后，才重新按端到端时间 profile V2。不能只为了
减少 SyncAll 移动 scatter；确定性 ownership 模型比推测的 barrier 节省重要。

## 后续步骤与接受标准

1. 保存当前源码和生成 C++，作为 V1 分支等待版本参考。
2. 在相同进程状态下收集匹配的 0721 和官方 msprof trace，记录 C1-P、C1-SY、
   V1-P、V1-SY、backward、C2、V2、WAIT_FLAG 时长。
3. 至少进行 5 轮 warmup=5、bench=10，比较两者中位数、最小值与离散度。
4. 从 trace 确认主导项是 V0 独占前缀还是 C1-SY/C2 Cube 串行，再评估
   优先级 1 或 2。
5. 任何结构性改动都先审计生成 C++；local shape、offset、event ownership
   协议不正确时，应在硬件执行前拒绝。

一个性能补丁只有同时满足以下条件才能接受：精度边界不变、生成 C++ 符合预期
协议、profile 关键路径缩短、重复中位数改善超过噪声、不增加 kernel，且不
修改 tilelang-ascend。
