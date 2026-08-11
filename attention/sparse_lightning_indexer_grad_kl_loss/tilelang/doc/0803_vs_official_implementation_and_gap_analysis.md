# 0803 与官方 AscendC 算子实现对比及差距分析

## 1. 文档范围与结论

本文以当前 `tilelang/0803.py` 为 TileLang-Ascend 实现，以 `op_kernel/arch22/` 和
`op_host/arch22/` 下的官方 AscendC 实现为对照，分析以下内容：

1. 输入、切分、V0/C1/V1/C2/V2 各阶段的实现差异。
2. 差异产生的直接原因、已经验证的失败尝试和 TileLang-Ascend 后端缺口。
3. 当前精度和性能基线，以及达到官方 80% 所需的后续方向。

结论先行：

- `0803.py` 当前是精度基线，最近一次 40 例泛化结果为 `40/40 PASS`；资源受限的 9 例单独统计，不应当记作精度失败。
- 0803 的数学主路径已经对齐：BMM1、BMM2/ReLU、P/SY、backward、C2 梯度、dKey scatter、dQuery/Loss reduction 的结果顺序一致。
- 性能差距主要来自实现结构，而不是单个算子公式：官方把工作按实际稀疏长度平衡到多个 AIC core，并在一个 core 内使用物理双槽流水；0803 以完整 `(B, query)` 为 task，C1 仍是安全的单物理槽路径，V1 通过 GM workspace 交换大块中间结果。
- 不能通过继续删除同步来“碰运气”达到 0.80x。下一次结构性优化必须先证明后端能够表达独立的 L0/L1 slot ownership，并且每次只改变一个层次。

本文将“源码已确认”“日志已观察”“尚未验证”明确区分，避免把性能假设写成事实。

## 2. 算子数学流程

对每个 query，输入的 sparse index 选择 K 个 key 候选。主要计算可以抽象为：

```text
Gather(Key, KeyIndex, KeyRope, SparseIndices)
    -> BMM1 = Query @ GatherKey^T
    -> BMM2 = QueryIndex @ GatherKeyIndex^T
    -> P/SY/Softmax
    -> diff = SoftmaxI - PReduced
    -> dWeight = reduce(BMM2 * diff)
    -> reluGrad = Select(BMM2 > 0, Weight * diff, 0)
    -> C2: dQueryIndex partial + dKey partial
    -> V2: dKey partial 按 SparseIndices atomic scatter
    -> cast/reduce: dQueryIndex, dKeyIndex, dWeights, Loss
```

其中：

- `K` 是输入 SparseIndices 的 ABI 宽度，必须保持原始 stride 语义。
- `KRealSize[b, q]` 是 causal/sparse 规则下该 query 的真实有效列数。
- 0803 还区分 `KExec` 和 `KScatterExec`：前者用于 Cube/V1/C2 静态循环，后者用于 V2 scatter 的静态搬运宽度。
- 尾块无效列必须在 P/SY/backward/scatter 中被排除；无效值不能依赖有限大负数与 `exp` 的近似行为。

## 3. 官方切分和 block_num

### 3.1 官方 host 侧

官方 host 入口位于：

`op_host/arch22/sparse_lightning_indexer_grad_kl_loss_tiling_general.cpp`

官方首先根据 layout 和 actual sequence length 计算每个 `(B, S1)` query 的稀疏有效长度：

```text
s2RealSize = max(actual_K - actual_Q + query_position + 1, 0)
s2RealSize = AlignUp(s2RealSize, 512)
s2RealSize = min(s2RealSize, K)
```

随后构造每个 query 的 `sparseValidArray`，按累计有效 S2 工作量进行 AIC core 切分：

- BSND：按 4D 工作量平衡。
- TND：先按 task 数粗分，再按每个 task 的 sparse 有效长度做粗调和精调。
- 实际使用 core 数为 `min(totalSize, aicNum)`。
- `context_->SetBlockDim(actualUsedCoreNum)` 设置 kernel 的 block 数。
- 每个 AIC core 由 `GetBlockIdx()` 获得自己的区间；同一物理 core 上的 AIV 通过
  `GetTaskRation()` 参与 vector 工作，官方代码注释显示 AIV 数量为 AIC 的两倍。

因此，官方的主要切分维度不是简单的 `B*S1` 均匀切分，而是：

```text
第一层：按 (B, S1) task 切分
第二层：按每个 task 的实际 sparse S2 工作量平衡 AIC core
第三层：每个 task 内按 K/S2 工作块循环
第四层：AIV 负责 Vector0/Vector1/Vector2 子任务，AIC 负责 Cube1/Cube2
```

官方 workspace 计算也体现了这种设计：`pSize`、`sySize`、`bmm1Size`、`bmm2Size`、
`reluGradSize`、`bmm3Size` 等按 `PING_PONG_VALUE` 计算，并按 active AIC core 数复制。
确定性模式还额外保留两份 scatter accumulator。

### 3.2 0803 host 侧

0803 的 `check_param_valid()` 固定了当前实现支持的形状集合，并将每个 `(B, S1)`
视为一个完整 task。`_make_task_schedule()` 的逻辑是：

```text
task_count = B * S1
active_cores = min(max(task_count, v2_core_need, 1), hardware_cube_core_num)
task_id = round * active_cores + core_id
```

也就是说，0803 的 `block_num` 等价于 host 选择的 `launch_core_num`，而不是官方
基于 sparse 有效工作量求得的 AIC block 数。每个 active core 内部由一个 AIC 和两个
AIV 协作，AIV 在当前 task 内按 K 的连续半区分工。

0803 的区别是有意的工程取舍：使用统一 task pipeline 简化了 workspace ownership，
但无法像官方那样把稀疏较短的 query 和稀疏较长的 query 重新平衡到不同 core。

## 4. 分阶段实现对比

| 阶段 | 官方 AscendC 实现 | 0803 TileLang-Ascend 实现 | 主要差异及影响 |
|---|---|---|---|
| Host 校验与 tiling | 检查 layout、dtype、N/D/K 约束；根据 actual length 构造 sparse valid array；按有效 S2 工作量平衡 AIC core | `check_param_valid()` 做 Python 侧约束；根据 host 可见长度选择 `full_valid` 或 `tail-protected`；按 task 数和 V2 需求选 core | 0803 更容易表达和缓存，但失去官方的稀疏工作量平衡，长 S1/TND 用例容易出现负载不均 |
| V0 Gather | `ProcessVector0()` 调用 `MergeKv()`；按 sparse index 做 DataCopyPad，Key/KeyRope 与 KeyIndex 分别写入双缓冲 workspace；每个 S2 block 完成后发布 cross-core flag | `vector0_gather_task()` 中两个 AIV 各处理当前 1024 行 K tile 的一半；每行读取 SparseIndices，再分别 copy Key/KeyIndex/Rope 到 UB，写入连续 workspace；16 行 UB ping-pong | 语义已对齐，但 0803 仍是逐 sparse row 的多路动态 DMA，缺少官方 gather/descriptor 的批量能力；当前 profile 显示 V0 是大瓶颈 |
| V0->C1 同步 | 官方使用 `CrossCoreSetFlag<2, PIPE_MTE3>`，按外层 S2/K block 的奇偶槽发布 | 由于后端不能可靠表达官方 AIV mode-1 wait，使用 AIC mode-2 汇聚两个 AIV 的 readiness flag；slot 为 `(task_slot + k_outer) % 2` | 0803 的协议可正确工作，但增加了 AIC relay/汇聚语义，且必须维护 task generation；不能简单改成全局少量 flag |
| C1 BMM1 | `ComputeMm1()`：Query/QueryRope 先放入 L1；Key gather 进入 L1；L0A/L0B/L0C 均有 ping-pong；MMA 后用 FIXpipe 写 bmm1 workspace；尾部使用 `singleN`/`singleK` | `cube1_task()`：显式 L1、L0A、L0B、L0C allocation；Query 的 4 个 128 维块放 L1；Key 按 128 列进入 L1；使用单物理 P L0C，手工 event 保护 MMA/FIX | 官方能同时 refill 下一个 slot、计算当前 slot、FIX 上一个 slot；0803 的单槽安全路径使 MTE1/MMA/FIX 更串行，这是大 S1/K 用例的主要差距之一 |
| C1 BMM2/SY score | `ComputeMm2()` 使用独立 query-index L1/L0A、KeyIndex L1/L0B、L0C 双槽；FIXpipe 带 ReLU 输出 bmm2 workspace | 仍在 `cube1_task()` 中顺序执行 SY BMM；独立 SY L0C 逻辑上存在，但受 planner/物理 L0C 复用约束，不能安全形成与 P 同时存在的物理双槽 | 公式和输出对齐，但 C1-P/C1-SY 不能复制官方的完整物理并发；增加 P/SY 顺序等待 |
| V1 P | 官方 `VectorP()` 读取 bmm1 block，使用真实 `kRealSize` 和 softmax tiling，生成 PReduced；尾块通过实际 copy/补齐处理 | `vector1_psy_task()` 读取 bmm1Res 和 ValidKMask；完整块走无 mask 路径，部分块 Compare/Select 后使用 `-T.infinity(calc_dtype)`；输出 PReduced workspace | 0803 尾块语义更显式、更容易验证，但 mask workspace 读取和 GM 往返增加了 Vector/MTE2 工作；full-valid specialization 可删除这部分动态路径 |
| V1 SY/Softmax | 官方 `VectorSy()` 读取 bmm2，按权重做 ReduceSumY/Softmax；通过 `psySyncGm` 交换小量统计结果，另有 task/AIV ownership | `vector1_psy_task()` 中 AIV0 计算 P、AIV1 计算 SY，`relay_vector1_psy()` 通过双槽 flag/GM relay 对齐；SoftmaxI 写完整 workspace | 0803 需要等待和 relay，并保留完整 SoftmaxI workspace；官方的局部统计和本地流水更紧凑 |
| V1 backward | 官方 `VectorDwDqDk*()` 以 2K 为分界走不同实现，使用 `reluGrad` ping-pong，MTE3->MTE2 WAR 保护下一个 refill；dWeight、dQuery/dKey 所需中间结果尽量复用本地 buffer | `vector1_backward_task()` 按 Nidx 分块，读取 PReduced/SoftmaxI/bmm2/weights，生成 dWeight、LossPartial、reluGrad；保留已验证的 WAR flag 和 32x64 物理分组 | 0803 的精度保护已经稳定，但完整 K workspace 读取和 GM store 仍多于官方；删除 WAR 或只改逻辑 slot 曾产生随机 NaN/Inf |
| C2 dKey/dQuery | 官方 `ComputeMm5()`、`ComputeMm6()` 使用 Cube L0A/L0B/L0C 双槽；reluGrad 从 GM 按 K outer block 读入；MM5 写 scatter partial，MM6 累积 dQuery | `cube2_task()` 使用显式 MMA/copy/event；读取 reluGrad workspace 和 GatherKeyIndex；写 `dk_matmul_partial` 与 dQueryIndexAccum | 数学一致，但 0803 的 task 级边界和 workspace relay 更粗；官方的 K block 级 double buffer 能与 MTE1/MMA/FIX 重叠 |
| V2 scatter | 官方 vector2 在所有 task/C2 完成后运行；按输出 T2/S2 工作量给 AIV 分区，16/64 行 UB ping-pong；确定性模式合并两份 accumulator，再 cast 写 dKeyIndex；Loss partial 最后归约 | `clear_dkey_accumulator()` 全核清零；两个 AIV 按 K 段读取 `dk_matmul_partial`，16 行批量按 SparseIndices 做 atomic add；最后由 task owner finalize dQuery、dKey、Loss | 0803 已有 KScatterExec 和每 task 一次 KRealSize 读取，尾块搬运明显收缩；但仍使用随机 atomic scatter，且保留独立 clear/sync 阶段；这不是当前最大瓶颈 |

## 5. 数据布局、workspace 和同步差异

### 5.1 workspace

官方和 0803 都需要中间结果，但生命周期不同：

```text
官方：每个 active AIC core 复制 ping-pong workspace，服务内部 LocalTensor 队列
0803：按 (B, S1, KExec) 预分配 caller-managed GM workspace，跨 V0/C1/V1/C2/V2 传递
```

0803 当前主要 workspace 包括：

- `GatherKey_ws`、`GatherKeyRope_ws`、`GatherKeyIndex_ws`
- `bmm1Res`、`bmm2Res`
- `PReduced`、`SoftmaxI`
- `dQueryIndexAccum`、`dk_matmul_partial`
- `reluGradOut`、`LossPartial`
- `dKeyIndexAccum`

这些 workspace 使阶段边界清晰、容易做精度诊断，也使 kernel 的 GM 往返数量高于官方。
尤其是：

1. V0 必须把 gather 结果写 GM，C1 再读回 L1。
2. C1 把 BMM 结果写 GM，V1 再读回 UB。
3. V1 把 PReduced/SoftmaxI/reluGrad 写 GM，后续 V1/C2 再读。
4. C2 把 partial 写 GM，V2 再读并随机 scatter。

### 5.2 同步

官方主要使用两类同步：

- AIC/AIV 之间的 `CrossCoreSetFlag/CrossCoreWaitFlag`，按 task 和 K outer block 发布。
- core 内 MTE2/MTE1/MMA/FIX 的 HardEvent，严格绑定 LocalTensor ping-pong slot。

0803 关闭自动同步，使用显式 `SetFlag/WaitFlag/PipeBarrier`，并使用固定 flag 组：

```text
V0 -> C1 P/SY       0..3
C1 -> V1 P/SY       4..7
V1 -> C2             8..9
V1 PSY -> AIC relay 10..13
```

0803 的协议已经解决过 task-slot 和 K-outer generation 错位问题，但它仍然是软件层面
模拟官方 pipeline。TileLang-Ascend 的 memory planner 不能仅凭逻辑 buffer 名称保证
异步消费者尚未结束时物理地址不被复用，因此每次调整 slot 都必须审计生成 C++ 的实际 offset。

## 6. 尾块和 full-valid 处理

0803 的尾块保护是必要的正确性改动，不是可随意删除的性能装饰：

- `KRealSize` 定义每个 query 的有效列范围。
- `ValidKMask` 只在 tail-protected specialization 中参与部分块保护。
- P/SY 无效列使用 `-T.infinity(calc_dtype)` 参与 max/exp 语义，再显式 Select 为零。
- backward 对完整块、部分块、完全无效块采用不同路径。
- V0 使用统一 GM 零哨兵行，使无效 sparse index 的所有 lane 走同一种 GM->UB gather。
- V2 使用静态 `KScatterExec`，避免对完整接口 K 做无意义 scatter。
- `full_valid` 是 host 已知所有 query 均满足 `KRealSize == K` 时的编译 specialization，生成代码可以静态删除动态 mask 分支。

官方并不依赖完整的 `(B,S1,K)` FP32 `ValidKMask` workspace，而是在 run info 中携带
`kRealSize`，在实际 copy、tail loop 和 vector 计算中使用有效长度。这是两者在尾块
实现上的重要结构差异：0803 更保守、更容易保证跨阶段一致性，但付出 mask DMA 和
workspace 容量。

## 7. 当前精度与性能表现

### 7.1 精度

冻结基线的全量结果：

```text
Summary: total=40 pass=40 fail=0 precision=0 tilelang_error=0 official_error=0 input_error=0 bench_error=0
resource_excluded=9
```

精度验收口径为 `norm_rel <= 4e-3`，同时禁止 NaN、Inf 和跨次随机漂移。资源排除是
设备 workspace 约束：在当前 61 GiB 设备上，`B*S1*K` 约 8.4M 已可能 OOM，TND
的 `S1*K=16.78M` 也会 OOM。这些用例不能作为 kernel 数学精度失败。

### 7.2 性能

下面是冻结基准日志中的代表数据，ratio 定义为 `official_ms / tilelang_ms`：

| 用例 | 0803/基线 TileLang | 官方 | ratio |
|---|---:|---:|---:|
| `bsnd_b1_s1_1_k_1024` | 0.367 ms | 0.185 ms | 0.50x |
| `bsnd_b1_s1_1_k_2048` | 0.379 ms | 0.246 ms | 0.65x |
| `bsnd_b1_s1_64_k_4096` | 1.601 ms | 1.309 ms | 0.82x |
| `bsnd_b1_s1_512_k_4096` | 20.059 ms | 12.557 ms | 0.63x |
| `bsnd_b1_s1_1024_k_5120` | 65.966 ms | 19.537 ms | 0.30x |
| `bsnd_b1_s1_2048_k_2048` | 241.032 ms | 7.909 ms | 0.03x |
| `tnd_s1_1024_k_5120` | 127.284 ms | 3.371 ms | 0.03x |

这些数字说明性能不是一个固定比例：小 task、full-valid、K 较小的用例可能接近或
超过 0.80x；大 S1、多 task、TND 或尾部稀疏长度差异大的用例会明显低于目标。

另一个直接实验是 `GATHER_CHUNK_SIZE=32`：

| 用例 | TileLang | 官方 | ratio |
|---|---:|---:|---:|
| `B=1,S1=1,K=2048` | 0.467 ms | 0.312 ms | 0.67x |
| `B=1,S1=1024,K=5120` | 73.099 ms | 25.239 ms | 0.35x |

该实验没有证明稳定收益，且没有消除逐 sparse row 的随机 DMA，因此生产实现保持
`GATHER_CHUNK_SIZE=16`。

## 8. 已尝试的优化及结论

| 尝试 | 结果 | 结论 |
|---|---|---|
| `KRealSize` 全流程保护、tail-protected/full-valid specialization | 精度通过，full-valid 可删除动态 mask 路径 | 保留，是正确性和部分性能恢复的基础 |
| V2 每 task 只读一次 KRealSize、`KScatterExec` 静态收缩 | 尾部用例从更宽 scatter 收缩到实际静态宽度，精度通过 | 保留，收益集中在 tail；不是大 S1 主瓶颈 |
| V0 GM 零哨兵 | 无效索引统一走 GM->UB，避免同一 chunk 混合路径 | 保留，降低 gather ownership 风险 |
| V0/C1 flag 使用 `(task_slot + k_outer) % 2` | 解决 K outer/task slot generation 错位问题 | 保留，属于同步正确性协议 |
| V1 PSY relay 按 task slot 双缓冲 | 稳定精度通过 | 保留，不能把它当作大性能收益点 |
| C1-P/SY 增加物理 L0C 双槽 | 出现 `L0C read/write conflict`；planner 将逻辑 buffer 复用到冲突地址 | 当前后端和当前 L0C 容量/ownership 下不可合入 |
| C1-P L0A/L0B 双缓冲实验 | 部分精度通过，但大 S1 用例退化，例如 `65.97 -> 76.80 ms` | 只增加 slot 不能形成有效 MTE1/MMA/FIX overlap，回退 |
| 删除 V1 WAR flag | 随机 NaN/Inf，典型为 reluGrad 被下一次 refill 覆盖 | 必须保留 `MTE3 -> MTE2` 保护 |
| V1 backward 逻辑 ping-pong/重排 | 曾出现精度漂移或非有限值 | 物理 buffer lifetime 未被后端可靠表达，不应继续在生产 kernel 试错 |
| 把 ReLU 改成 Compare/Select | mask layout/lowering 曾不稳定；当前近似路径已通过基线 | 在 Compare/Select 后端语义被独立验证前，不作为正式替换 |
| `GATHER_CHUNK_SIZE=32` | 小用例约 0.67x，大 S1 约 0.35x，未显示稳定收益 | 根因是随机 indexed DMA，不是 chunk 数量 |
| 清零前移或在 atomic_add 前按位置清零 | atomic 目标随机且可能被多个 task 命中，无法保证首次写语义 | 保留独立 clear 阶段和同步边界 |
| 删除全核同步、改成局部 wait | 单独减少 barrier 没有稳定量化收益，且容易破坏 C2->V2 完成边界 | 只有确认 producer/consumer 后才能改具体 flag |
| copy/MMA 非标准尾块 shape 实验 | 曾有编译、BufferRegion、runtime/精度不稳定问题 | 不能直接作为 0803 生产路径；需先做独立最小 kernel 能力验证 |

## 9. 性能差距根因

### 9.1 第一根因：任务切分不对齐

官方以实际 sparse S2 工作量平衡 AIC core；0803 以 `(B,S1)` task 粗粒度调度。
当不同 query 的 `KRealSize` 差异较大时，0803 仍给每个 core 固定的 task 序列和静态
`KExec`，会同时产生：

- 某些 core 处理长 task，其他 core 提前空闲。
- 大 S1 时大量 task 经过同一个粗粒度阶段边界。
- V2 需要全 kernel drain 后再 scatter，无法补偿前面阶段的负载偏斜。

这解释了为什么小用例可以达到 0.80x，而 `S1=1024/2048` 或 TND 用例会跌到
0.03x~0.30x。它不是简单的 core 数量不足。

### 9.2 第二根因：C1 没有对齐官方物理流水

官方同时拥有独立的 L1、L0A、L0B、L0C ping-pong slot，并用 HardEvent 维护：

```text
GM -> L1(refill next) -> L0A/L0B -> MMA(current) -> L0C/FIX(previous)
```

0803 当前可稳定交付的是单物理 P L0C/单物理 SY L0C 路径。增加逻辑 slot 或 event
并不会自动产生独立物理地址和合法 ownership；已发生的 L0C conflict 正是证据。
因此 C1 的 MTE1、MMA、FIX 重叠不足，且大 K/S1 时这个串行损失被放大。

### 9.3 第三根因：V1 GM workspace 往返

官方 vector service 使用 task 内的 LocalTensor 和有限的 `psySyncGm` 统计交换；0803
为了跨宏、跨 AIV 和跨 C1/C2 阶段稳定传递结果，保留完整 PReduced、SoftmaxI、reluGrad
workspace。大 S1/K 时，GM 读写和对应 wait 的总量显著增加。

### 9.4 第四根因：V0 是逐行随机 gather

0803 每个 sparse row 都需要读取索引并对 Key、KeyIndex、Rope 分别发起 copy。将 16 行
改成 32 行只减少外层循环次数，不能把非连续索引变成批量 DMA。若后端没有 indexed
gather descriptor，V0 的带宽和指令数都无法对齐官方的专用 MergeKv 路径。

### 9.5 第五根因：V2 不是主瓶颈，但有固定开销

V2 已做静态收缩和双 AIV K ownership，尾块用例收益明显；但独立 clear、全核边界、
随机 atomic scatter 和 finalize 仍存在固定成本。除非 timeline 证明 V2 占主导，继续
微调 V2 不会把大 S1 用例从 0.30x 提升到 0.80x。

## 10. TileLang-Ascend 后端能力缺口

下面是影响本算子对齐官方实现的后端能力，分为已确认问题和需要独立验证的问题：

| 能力缺口 | 直接表现 | 对 0803 的影响 |
|---|---|---|
| AIV-to-AIV mode-1 wait 表达能力不足 | 不能直接复用官方 AIV 间等待协议 | 必须用 AIC mode-2 relay 汇聚 readiness flag，增加协议复杂度 |
| 异步 LocalTensor 的 phase-scoped no-alias/ownership 表达不足 | planner 可能把逻辑不同的 L0C buffer 放到同一物理地址 | C1 L0C 双缓冲出现 read/write conflict |
| 非标准 copy/MMA tail shape lowering 不稳定 | 可能出现动态 BufferRegion 编译错误、非法 shape 或精度问题 | 只能使用静态 128/64/16 分块和显式尾块保护 |
| 动态 BufferRegion 索引检查严格 | 例如 1D `KRealSize` 被二维索引、动态 range 被宏收集拒绝 | full-valid/tail 需要合法占位输入和静态 specialization |
| 不支持 TileLang kernel 中的 C/C++ 风格三元表达式 | 编译阶段无法生成目标 TIR | 动态选择必须改为静态分支或 host 侧 specialization |
| Compare/Select mask 的 dtype/pack 约束严格 | float mask 或错误的 pack 数会造成编译/精度问题 | 需要 uint8 mask，并按元素数/8 计算 mask buffer |
| indexed gather/DMA descriptor 能力不足 | 逐行动态 copy 无法合并为官方式批量 gather | V0 GM 带宽和指令数难以对齐 |
| 手工同步关闭后的事件语义不自动验证 | Set/Wait/slot/generation 任一处错位会随机 NaN/Inf | 必须以生成 C++ 审计为准，不能只看源代码变量名 |
| 大动态 workspace 的内存规划成本高 | 可能 OOM 或引入较大 host/NPU 辅助开销 | 需要资源排除和 workspace 尺寸约束 |

## 11. 后续优化建议和实施顺序

### 11.1 先做性能账本，不直接改生产 kernel

针对以下三个代表用例分别采集关闭 debug、充分 warmup 后的设备 timeline：

1. 小 task/full-valid：`B=1,S1=1,K=2048`。
2. 多 task/大 S1：`B=1,S1=1024,K=5120`。
3. 尾块/非均匀有效长度：现有 tail case。

至少拆出 V0、C1-P、C1-SY、V1-P/SY/backward、C2、V2、host/runtime。没有 timeline
前不应再进行全局 barrier、核数、tile shape 的混合修改。

### 11.2 优先验证 V1 workspace 收缩

这是不依赖 L0C 双槽的结构性方向：

1. 在 full-valid、单 task 小 kernel 中，让两个 AIV 按 K ownership 计算局部 P/SY。
2. 只交换每个 head 的局部 max/sum，不交换完整 `PReduced/SoftmaxI`。
3. 在本地 K 段现场计算 `diff`、reluGrad、Loss partial 和 dWeight partial。
4. 保留现有 C2/V2，不先改 event 协议。
5. 先验证 workspace GM read/write 次数下降，再扩展到 tail 和多 task。

如果后端无法稳定表达局部归约或只支持大块 GM relay，应记录为能力缺口，不继续在
生产 kernel 中叠加 workaround。

### 11.3 再验证 C1 双槽能力

不要直接在完整算子里重新增加多个 L0C buffer。建立独立最小 C1-P kernel，固定标准
128 tile，显式分配两份不同的 L1/L0A/L0B/L0C buffer，并验证：

- 生成 C++ 中每个 slot 有不同物理 offset。
- MTE2/MTE1/MMA/FIX 的 set/wait 顺序闭合。
- 没有 `L0C read/write conflict`、hang 或非法指令。
- p50 确实优于单槽，而不是仅仅增加同步和内存占用。

只有最小实验成功，才把同一 ownership 协议迁移到 C1-P，再逐步迁移 C1-SY/C2。

### 11.4 V0 只在后端提供新能力后重做

`GATHER_CHUNK_SIZE` 不是根本开关。有效方向应是后端支持批量 indexed gather，或将
sparse index 预处理为设备可直接消费的 DMA descriptor。任何 host 侧裁剪复制都必须
计入端到端时间，不能只比较 kernel 事件。

### 11.5 V2 只做低风险局部优化

在 timeline 证明 V2 占比明显时，再尝试：

- 按 `valid_k_size` 跳过完全无效的 16 行 DMA。
- 保持独立 clear，不能让 atomic add 承担首次清零职责。
- 验证局部 flag 是否能替代真正的全阶段边界。

## 12. 合入与验收标准

任何后续优化都必须满足：

1. 40 个非资源排除用例均 `norm_rel <= 4e-3`。
2. 代表失败形状连续运行至少 10 次无 NaN/Inf、无随机漂移。
3. 生成 AscendC 的 buffer offset、extent、MMA/FIX shape 和 event generation 与设计一致。
4. 性能用相同 warmup/iters、关闭 debug/profile，报告中位数和分位数；不能用一次运行。
5. 单项修改的收益必须在对应瓶颈阶段有 timeline 证据。
6. 目标是逐用例达到 `official_p50 / tilelang_p50 >= 0.80`，不能用少数小用例平均值掩盖大 S1/TND 回退。

当前建议的优先级是：

```text
性能 timeline
  -> V1 K ownership / workspace 收缩
  -> 独立最小 C1 双槽后端能力验证
  -> 若能力可用，再迁移 C1 物理双槽
  -> 后端 indexed gather 后再优化 V0
  -> 最后处理 V2 微优化
```

在 V1 或 C1 的最小实验无法证明收益时，应把结果作为后端能力边界提交，而不是继续
增加 0803 的同步、workspace 和分支复杂度。

## 13. 参考源码位置

### TileLang

- `tilelang/0803.py:141-188`：参数检查和 layout 约束。
- `tilelang/0803.py:197-254`：kernel 参数、静态 tile、task/V2 调度参数。
- `tilelang/0803.py:256-327`：V0 gather、零哨兵和 V0->C1 flag。
- `tilelang/0803.py:329-470`：C1 BMM1/BMM2 和 L0/L1 event。
- `tilelang/0803.py:471-929`：V1 P/SY、relay、backward 和 tail mask。
- `tilelang/0803.py:930-1004`：C2 梯度矩阵乘。
- `tilelang/0803.py:1005-1210`：V2 clear、scatter、finalize 和 Loss reduction。
- `tilelang/0803.py:1212-1289`：task pipeline 和阶段边界。
- `tilelang/0803.py:1290-1622`：host layout 包装、specialization、cache 和 workspace。

### 官方 AscendC

- `op_host/arch22/sparse_lightning_indexer_grad_kl_loss_tiling_general.cpp:646-884`：实际
  sparse 长度计算、工作量平衡和 core 区间。
- `op_host/arch22/sparse_lightning_indexer_grad_kl_loss_tiling_general.cpp:894-940`：
  active core、`block_dim`、softmax tiling 和输出分核。
- `op_host/arch22/sparse_lightning_indexer_grad_kl_loss_tiling_general.cpp:968-991`：官方
  workspace 组成和双缓冲规模。
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_base.h:443-560`：官方 task
  pipeline、run info、AIC/AIV 阶段调度和全局同步边界。
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h:421-670`：V0、V1、C2
  前置/后置协作入口。
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h:975-1665`：P、SY、
  ReLU、dWeight、Loss 和 dQ/dK 相关 Vector 实现。
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_service_cube.h:259-290`：官方
  L0A/L0B/L0C 双槽 buffer。
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_service_cube.h:461-735`：官方
  ComputeMm1/Mm2/Mm5/Mm6、MMA/FIX 和跨阶段 flag。
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector2.h:145-245`：官方 V2
  输出 scatter accumulator 合并、cast 和 Loss reduction。
