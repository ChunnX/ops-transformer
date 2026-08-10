# 0803 泛化性能冲刺路线

## 1. 目标与结论

### 1.1 项目目标

本项目使用 TileLang-Ascend 重写
`sparse_lightning_indexer_grad_kl_loss`，目标是：

1. 泛化测试中所有**可执行** BSND/TND 用例精度稳定通过；比较指标为
   `norm_rel <= 4e-3`，无 NaN、Inf 或跨次随机漂移。
2. 对每个可执行用例，TileLang kernel 中位耗时至少达到官方算子的 `0.80x`：

   ```text
   speed_ratio = official_median_ms / tilelang_median_ms >= 0.80
   ```

3. 资源不足而无法分配工作区的形状必须单列为资源约束，不能伪装成精度或性能失败。

截至本文，`0721_baseline.py` 的清理没有改变泛化精度和性能结果。这是正确结果：
本次仅删除了开发期诊断逻辑、整理注释和格式，没有改动正式数学、工作区、tile、
flag 或流水。

### 1.2 当前性能判断

历史稳定基准中，代表性小任务的 TileLang 中位数约为 `0.47~0.49 ms`，官方约为
`0.29~0.30 ms`，即约 `0.60~0.63x`。达到 `0.80x` 时 TileLang 需要约
`0.36~0.38 ms`，仍需缩短约 `0.10~0.12 ms`（约 23%~27%）。

这个缺口大于一次 `wait_flag`、一次小 DMA 或一次 `sync_all` 通常能够贡献的收益。
因此后续不能靠不断删 barrier 冲刺，而应只实施能够缩短关键路径的结构优化。

### 1.3 最重要的结论

当前 0803（与整理后的 0721 baseline 逻辑一致）已经是**精度基线**，但不是
TileLang 当前能力下可证明的性能最优流水：

- 已验证的正确性保护不能删除，例如 V1 backward 中 `reluGrad` store 与下一块
  BMM2 refill 的 `MTE3 -> MTE2` WAR 依赖。
- 官方的主要优势来自 C1 内部多 L1/L0A/L0B/L0C 物理槽的并发，而当前 TileLang
  正式路径只有安全的单槽 Cube 流水。
- 已经两次证明，直接在主算子中手写 C1 L0C ping-pong 会产生 L0C read/write
  conflict；这不是再加一个 pipe barrier 就能修复的问题。

**所以：若不先证明 TileLang 后端能表达官方所需的物理 L0 ownership，不能承诺
仅靠 0803 源码微调就让所有形状稳定达到 0.80x。**

后续路线分为两条并行但互不混杂的线：

1. 在现有后端能力内消除 V1/V2 的确定性 GM 往返，争取中等收益。
2. 用独立最小实验确认或否定 C1 物理双缓冲可行性；若否定，形成明确的后端能力
   需求，而不是继续在生产 kernel 中试错。

## 2. 当前代码结构

主实现文件：`tilelang/0803.py`。

`0721_baseline.py` 是已精简、无调试依赖的冻结基线；0803 当前与它保持相同正式
逻辑，后续性能实验只在 0803 进行。`precision_test_general/test_0803.py` 是单 case
的快速回归脚本，`precision_test_general/test_0721_general.py` 是泛化测试入口。

### 2.1 主机侧

入口 `sparse_lightning_indexer_grad_kl_loss()` 完成以下工作：

1. `check_param_valid()` 校验 BSND/TND、BF16/FP16、`N2=1`、`DQuery=512`、
   `DQueryIndex=128`、`sparse_mode=3` 和 head/K 约束。
2. `_derive_execution_widths()` 根据主机可见的真实长度推导：
   - `K`：SparseIndices 的原始容量和 GM stride。
   - `KRealSize[b,q]`：该 query 真正有效的候选数。
   - `KExec`：C1/V1/C2 的静态执行宽度。
   - `KScatterExec`：V2 以 16 行 batch 执行 scatter 的静态宽度。
3. `_is_static_full_valid()` 选择 specialization：
   - `full-valid`：所有 query 都有完整 K，生成代码不读取 tail mask。
   - `tail-protected`：保留 KRealSize 和局部尾块保护。
4. 缓存 kernel、主机辅助输入和 caller-managed GM workspace；输入只在布局、RoPE
   补齐和 dtype 适配时做必要处理。
5. 对 TND 通过 `sparse_lightning_indexer_grad_kl_loss_tnd()` 包装为单 batch 的
   BSND 执行，再恢复输出形状。

### 2.2 一个 task 的定义与调度

一个 task 是完整的 `(batch, query)`：

```text
task_id = b * S1 + q
task_count = B * S1
task_id = round * launch_core_num + core_id
```

一个物理 AI Core 包含一个 AIC 和两个 AIV。每个 core 处理自己的 task 序列；
`TASK_PIPELINE_DEPTH=2` 使当前 round 的 V0 与上一 round 的 C1/V1/C2 重叠。
V2 在任务流水 drain 后统一执行。

### 2.3 五阶段数据流

```text
V0  -> GatherKey/GatherKeyRope/GatherKeyIndex workspace
C1  -> bmm1Res (Teacher score) + bmm2Res (Indexer score, ReLU)
V1  -> PReduced + SoftmaxI -> dWeight + reluGrad + LossPartial
C2  -> dQueryIndexAccum + dk_matmul_partial
V2  -> dKeyIndexAccum atomic scatter -> dQuery/dKey cast + Loss reduce
```

| 阶段 | 函数 | AIC/AIV 分工 | 当前关键策略 |
| --- | --- | --- | --- |
| V0 | `vector0_gather_task` | 两个 AIV 按 K 外层块各负责一半 | 16 行 ping-pong gather；负 index 映射到 GM 零哨兵行 |
| C1 | `cube1_task` | AIC | 先 MM1，再 MM2；128 列 Cube tile；Key L1 单槽预取 |
| V1-P/SY | `vector1_psy_task` | AIV0 计算 P，AIV1 计算 SY | 通过 AIC relay 交换完成信号；完整块不读 tail mask |
| V1-BW | `vector1_backward_task` | 两个 AIV 按 Nidx 行切半 | 使用已验证的 `32x64` 物理子 tile 和 WAR flag |
| C2 | `cube2_task` | AIC | reluGrad @ KeyIndex、reluGrad^T @ QueryIndex |
| V2 | `vector2_scatter_all_tasks`、`finalize_vector_outputs` | 所有 AIV | 16 行批量 GM->UB，pair-wise atomic add，统一 cast/reduce |

### 2.4 不能随意删除的同步边

| 资源 | 必需依赖 | 原因 |
| --- | --- | --- |
| V0 ping/pong UB | MTE3 -> MTE2 | MTE3 未写完 Gather workspace 前不能覆盖 slot |
| V0 -> C1 | mode-2 cross flag | AIC 必须等两个 AIV 都完成各自的 K 半区 |
| C1 -> V1 | C1 P/SY 分别 ready | AIV0 与 AIV1 可先做自己的 P/SY，backward 前才会合 |
| V1 `bmm2` / `rg_half` 别名 UB | MTE3 -> MTE2 | 下一块 MTE2 refill 不能覆盖仍被 MTE3 读取的 FP16 reluGrad |
| V1 -> C2 | task-slot cross flag | C2 只能消费完整 task 的 reluGrad |
| C2 -> V2 | 全任务阶段边界 | dKey atomic accumulate 前必须先完成全部 partial 写入 |

尤其是 V1 的 WAR 保护是曾经 NaN/Inf 的根因修复。删除它可能短暂变快，但不属于
可接受优化。

## 3. 已验证、应保留的改动

### 3.1 正确性和泛化能力

| 改动 | 结果 | 为什么保留 |
| --- | --- | --- |
| KRealSize 全路径语义 | 成功 | 无效候选不会参与 P/SY/backward/V2；解决 K 大于实际有效长度的尾块错误 |
| KExec/KScatterExec 静态收缩 | 成功 | 不裁剪 SparseIndices 原始 stride，但减少 tail case 的 workspace 和 V2 搬运 |
| full-valid 静态 specialization | 成功 | full-valid 生成代码删除动态 tail 判定与 mask 读写，恢复常规 case 性能 |
| P/SY 尾块显式 Compare/Select | 成功 | 无效列明确为零或 `-T.infinity(calc_dtype)`，不依赖极小 exp 的近似 |
| V0 GM 零哨兵 | 成功 | 所有 lane 统一走 GM->UB gather，避免混合 UB->UB/GM->UB 路径的后端 ownership 问题 |
| V0/C1 flag slot 为 `(task_slot + k_outer) % 2` | 成功 | 多个 1024 K 外层块不会复用错误 generation |
| V1 relay 按 task-slot 双缓冲 | 成功 | P/SY relay 的 producer/consumer generation 对齐 |
| V1 backward `32x64` + MTE3->MTE2 WAR | 必须保留 | 消除了 reluGrad/dQ/dK 的随机 NaN 和异常 FP16 大值 |
| V2 每 task 读取一次 KRealSize，并使用 KScatterExec | 成功 | 保证 tail scatter 正确，同时减少无效 K 的 DMA |
| dWeight/Loss/dKey cast 留在单 kernel | 成功 | 避免 host 侧额外 NPU kernel 和不一致的归约时序 |

### 3.2 已确认但收益有限的性能改动

| 改动 | 观察 | 处置 |
| --- | --- | --- |
| V1 按 AIV 分别等待 C1-P/C1-SY ready | profile 约减少 5 us | 保留；它缩短局部空洞，但不是关键路径重构 |
| C1-P Key/Rope 单槽 L1 预取 | 生成 C++ 可见 GM->L1 与当前 MMA 的部分重叠 | 保留；稳定且不改变物理 L0 ownership |
| C1-SY QueryIndex 常驻 L0A | 删除确定的重复 L1->L0A copy | 保留；整体中位收益未稳定显著，但无回归 |
| V2 静态 scatter 收缩 | tail case 明显改善 | 保留；对 full-valid 的理论收益有限 |

## 4. 已尝试但不应直接重试的方向

以下结论不是“想法不好”，而是对**当前 TileLang-Ascend 版本、当前 buffer
布局和当前算子**已经得到的实测结论。重试前必须先改变前置条件并写出新的 C++
验收证据。

| 尝试 | 结果 | 根因/结论 |
| --- | --- | --- |
| 仅调 `pipeline_delay` 或串行 task 作为正式路径 | 无稳定收益 | 只改变观察方式，未减少真实 MMA/GM 工作；诊断模式不能进入性能基线 |
| V1 backward ping-pong | 精度失败 | UB alias/lifetime 未被安全表达；会重现 MTE3 与 MTE2 覆盖竞争 |
| 删除 V1 WAR flag | 随机 NaN/Inf | MTE2 覆盖仍被 MTE3 读取的 `reluGrad` 区域 |
| 每 task 内就 scatter dKey、试图少一次全核同步 | 精度失败 | atomic/finalize ownership 不完整；dKey 输出需保留统一确定性阶段 |
| V2 先清零即 scatter、以 atomic add 代替整块初始化 | 不成立 | atomic add 不能初始化随机、可能重复命中的行；清零必须先于任意 atomic add |
| V0 拆成 Key/Rope 与 KeyIndex 两阶段 gather | C++ 正确但无稳定收益 | C1 单槽 MMA 是关键路径，拆 gather 反而可能增加 GM 竞争和波动 |
| C1-SY/C2 的单槽预取或 C2 QueryIndex 常驻 L0B | 持平或变慢 | 减少 copy 次数不等于缩短关键路径，常形成 MTE1 前导 |
| 大父 L0 allocation 再切 slice | 非法指令/错误结果 | L1->L0/FIX descriptor 会继承父物理 shape，不能伪造小矩阵 |
| C1-P 精确物理 L0 双缓冲 | L0C read/write conflict | 即使 offset/shape/MMA/FIX 模板正确，当前手写 `T.mma` 路径仍缺少可表达的完整 L0C ownership 协议 |
| 仅使用 `copy(real_*)`、`mma(k_actual/n_actual)` 重写尾块 | 未形成稳定生产版本 | extent 描述尾部 shape，但不自动解决跨 pipe 生命周期或多 slot event 协议 |
| ReLU 改为 Compare/Select | 未作为正式路径保留 | Vector mask 布局/类型 lowering 曾出现不稳定；当前 `1-exp(-10000*x)` 已通过精度，应保留到最小 intrinsic 实验成功 |
| 仅删 sync_all 或换成 cross flag | 收益很小或引入风险 | 有效时应改成特定 producer/consumer flag；不能凭“更少等待”替换真实全局阶段边界 |

## 5. 与官方的结构差距

官方 arch22 的关键不是单个 intrinsic，而是一个物理 service pipeline：

```text
GM(n+2) -> L1(n+1) -> L0A/L0B(n+1) -> MMA(n) -> L0C/FIX(n-1)
```

其 C1 有两套精确物理 L0A/L0B（各约 32 KiB）和 L0C（各约 64 KiB），并按 slot
轮转 event。这样 MTE2、MTE1、MMA、FIX 可以同时处于不同 tile。

当前 TileLang C1-P 的安全路径仍近似为：

```text
GM -> 单 L1 -> 单 L0A/L0B -> MMA -> 单 L0C -> FIX
```

单槽资源必须等待同一 tile 的 consumer 归还后才能装下一块。C1-SY 和 C2 也存在
相似的串行边。这个差异足以解释约 0.10 ms 的长期缺口。

第二层差距是 V1：当前 PReduced 与 SoftmaxI 落 GM 后，backward 的两个 AIV 都要
重新读取完整 K 范围；官方的局部数据交换和流水更紧凑。第三层才是 V2 的批量
scatter/cast 细节。因此优化优先级必须是 **C1 和 V1 > V0 > V2 微调**。

## 6. 达到 0.80x 的执行计划

### 阶段 0：建立每用例性能账本

目的：不再用单个 case 的 `0.60x` 代表 40 个形状。

1. 冻结 0803 的 commit、CANN/TileLang-Ascend 版本、设备 core 数、默认 seed、
   生成 C++ 和 msprof 配置。
2. 对 40 个可执行泛化用例各跑至少 5 轮，记录 warmup 后 `TileLang p50/p10/p90`、
   `Official p50`、ratio、是否 full-valid、`KExec/KScatterExec`、launch core 数。
3. 按以下桶统计，而不是按单一平均数决策：
   - full-valid / tail-protected；
   - 小 task（`B*S1 <= core_num`）/ 多 task；
   - K=1024、2048、4096、5120、8192；
   - BSND / TND。
4. 针对每个桶选择一个最慢 ratio case 和一个接近目标 case，作为下一阶段的 A/B
   最小回归集。

验收：得到一张 40 行账本；每个 case 的目标延迟为 `official_p50 / 0.80`，而非
使用一个全局毫秒阈值。

### 阶段 1：先确认关键路径，而不是先改代码

目的：用同一 case 的 TileLang/Official msprof timeline 量化差距。

对阶段 0 中最慢的三个代表 case（至少包含 full-valid、小尾块、多 task）执行：

1. profile V0、C1-P、C1-SY、V1-P、V1-SY、V1 backward、C2、V2 的耗时；
2. 统计 AIC 的 `WAIT_FLAG`、MTE2/MTE1/M/FIX 活跃区间和 AIV 空闲区间；
3. 对照生成 C++：MMA/FIX 模板、L0 offset、GM copy 数量、跨核 flag cadence；
4. 将时间归入：Cube 串行、V1 GM 往返、V0 随机 gather、V2 scatter、runtime/launch。

验收：每个代表 case 只给出一个第一瓶颈。若 C1 占主导，直接进入阶段 2；若 V1
GM 往返占主导，阶段 3 优先；不要在 V2 上做猜测性微调。

### 阶段 2：C1 双缓冲能力门槛实验（独立最小 kernel）

目的：回答“当前后端能否安全表达官方最关键的物理流水”，而不是再次在 0803
主算子中试错。

实现范围：新建仅用于验证的最小 C1-P kernel，固定 `M=N=K=128`，先无 RoPE、
无 V1/V2、无动态 tail。它必须有两个**独立且精确 shape**的 L1、L0A、L0B、L0C
slot，并显示定义以下 ownership：

```text
MTE1_MTE2 : L1 slot 已被 MTE1 取走，可供下一 GM refill
MTE2_MTE1 : L1 新数据 ready
M_MTE1    : L0A/L0B 已被 MMA 消费，可供下一 L1->L0
MTE1_M    : L0A/L0B 新数据 ready
FIX_M     : L0C 可给 MMA 写入
M_FIX     : L0C 结果可给 FIX 读取
```

每个 slot 在首轮 seed、稳态 set/wait、尾轮 drain 都必须成对。验收必须同时满足：

1. 与单槽参考逐元素相等或在既定精度范围内；
2. 生成 C++ 的两套 L0 地址、物理 shape、MMA/FIX 模板均正确；
3. 无 `L0C read/write conflict`、非法指令或 hang；
4. profile 能看到 MTE1/MMA/FIX 重叠，并且中位数确实优于单槽。

**通过后**才将同一协议移植到 C1-P，再扩展到 C1-SY/C2；每次只接入一个 phase。

**失败后**立即停止源码层面 C1 双缓冲尝试，将结果记录为 TileLang 后端缺少
phase-scoped L0 slot 与 MMA/FIX ownership 的证据，并转阶段 3。不要在主 kernel
继续增加 barrier、修改 slot offset 或猜 event ID。

### 阶段 3：V1 按 K ownership 重构，消除 P/SY GM 往返

这是不依赖 L0 双缓冲、但具有实际收益空间的主要源码路线。它是一次结构改动，
必须先在一个 full-valid 的小 task 上完成，然后扩展 tail 和多 task。

#### 目标状态

当前：

```text
AIV0: 完整 K 计算 P      -> 写 PReduced GM
AIV1: 完整 K 计算 SY     -> 写 SoftmaxI GM
两 AIV: 按 Nidx 行切分 backward，各自读完整 K 的 P/SY/bmm2
```

目标：

```text
AIV0: K 的前半区
AIV1: K 的后半区

各 AIV：局部 P、局部加权 score、局部 max/sum
两 AIV：只交换标量 max/sum（必要时每 head 的小归约）
各 AIV：在自己的 K 半区现场生成 SoftmaxI、Diff、reluGrad、Loss partial
各 AIV：产生 dWeight partial；最后用小尺寸确定性归约合并
```

由此删除或缩小的 GM 流量包括：完整 `SoftmaxI` workspace、backward 对完整
`PReduced/SoftmaxI` 的双 AIV 重读，以及部分 V1 relay。`reluGrad` 仍按 K 半区写入
原 workspace，C2 保持按 K tile 消费，因此可以分阶段接入。

#### 实施顺序

1. 仅 full-valid、`B=1,S1=1,K=1024`：做 K 半区 local softmax 的 max/sum 合并，
   只输出 P/SY 对照 workspace，不接 backward。
2. 加入 `Diff`、Loss 和 dWeight partial；保持原 `reluGrad` 计算公式与 FP32
   归约顺序，新增固定对齐 partial workspace，避免标量 DMA 相互覆盖。
3. 接入 reluGrad/C2，保留现有 C2/V2 不变；逐任务 flag 改为“两个 K 半区 ready”。
4. 加入 tail-protected 的完整块/部分块/完全无效块三条静态路径。
5. 最后扩展多 task，再运行 40 例泛化回归。

每一步的 C++ 验收：GM `PReduced` / `SoftmaxI` copy 数下降，跨 AIV 只增加小
归约而不是新的全 K relay；任何 NaN 先按 Input -> Gather -> BMM -> P/SY ->
reluGrad -> C2 顺序定位，不能通过放宽阈值掩盖。

### 阶段 4：只针对 profile 显示为瓶颈的 V0/V2

#### 4.1 V0

当前 V0 是逐 index 的随机三路 GM gather。若 msprof 显示其在特定 K/S1 桶中为
关键路径，优先做以下低风险 A/B：

1. 仅把 `GATHER_CHUNK_SIZE` 从 16 提到 32；不改变两个 AIV 的 K ownership 和
   flag 协议。
2. 审计 UB 用量、16/32 行 DMA 对齐、生成 C++ 指令数和 MTE2 带宽。
3. 若重复中位收益不足 3%，立即恢复 16。

若 V0 长期只有低带宽，根因是逐行随机 DMA。真正的解法需要 TileLang 后端提供
批量 indexed gather/DMA descriptor；改变 chunk 数不能替代该能力。

#### 4.2 V2

V2 已具备 KScatterExec、16 行批量加载、两个 AIV 非重叠 ownership 和统一
finalize。它不是第一优先级。

仅在 profile 证明 V2 是关键路径时，按如下顺序实验：

1. `valid_k_size` 驱动每 AIV 的有效 batch 上界，跳过完全位于尾部的 GM->UB DMA；
2. 保持当前 16 行 ping-pong，测 32 行批量是否降低 DMA 指令而不恶化 atomic
   粒度；
3. 不移动清零与 scatter 的全局阶段边界，不让 atomic add 兼任清零。

### 阶段 5：尾块 extent 原语接入（正确性优先）

后端新增的 `T.copy(..., real_*)` / `T.mma(..., k_actual/n_actual)` 只在阶段 2
或阶段 3 的**独立最小测试**已经通过时使用。接入顺序为：

1. 单个 128 列部分尾块的 copy 与 MMA 参考比较；
2. full-valid 与 partial-tail 共享同一数学和 event 协议，仅改变 extent；
3. 检查 readiness flag 是否在真实尾 tile 后仍被正确发布；
4. 再推广到 C1、C2、V1。

它能减少尾块的无效计算和搬运，但对 full-valid 用例没有收益，因此不能作为
“所有用例达到 0.80x”的主方案。

## 7. 每次改动的固定验收流程

1. 只改一个架构变量，写明预期减少什么工作、预期生成 C++ 出现什么变化。
2. 本地执行 `ast.parse`、`py_compile`、未定义变量扫描、`git diff --check`。
3. 硬件上先跑代表用例的精度 10 次；出现随机误差先诊断，不 benchmark。
4. 保存该次生成的 C++，检查 buffer offset/shape、DMA extent、MMA/FIX 模板和
   所有 flag 的 seed/set/wait/drain。
5. 对通过精度的补丁进行至少 5 轮 benchmark，使用相同 warmup/iters，报告 p50、
   p10、p90 和 official p50。
6. 只有“精度稳定 + C++ 符合预期 + profile 关键路径缩短 + p50 超过噪声”四项
   同时满足，才合入 0803；否则回退该单一改动。
7. 代表集通过后再跑 40 例精度与性能矩阵；不在 40 例失败后一次修改多个层级。

## 8. 当前推荐的下一步

下一步不是立即修改 0803 的 V2 或同步 flag，而是执行**阶段 0 和阶段 1**：先拿到
40 例的性能账本与三个代表 case 的 TileLang/Official timeline。随后：

- 若 C1 是三个代表 case 的第一瓶颈，启动阶段 2 的最小双缓冲可行性实验；
- 若 V1 GM 往返是第一瓶颈，启动阶段 3 的 K ownership 设计验证；
- 只有 profile 明确显示 V0 或 V2 占主导时，才进行阶段 4 的局部 A/B。

这条路线的核心纪律是：**正确性基线不再承载试验；每个试验都有可证伪的 C++ 和
时间线验收；无法表达的官方物理流水应上升为后端能力问题。**
