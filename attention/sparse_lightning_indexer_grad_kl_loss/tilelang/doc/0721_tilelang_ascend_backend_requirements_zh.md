# 0721 对 TileLang-Ascend 后端能力的需求与性能影响

## 1. 目的与判定口径

本文只记录 0721 为进一步接近官方 AscendC 性能所需要的 TileLang-Ascend 能力，说明每项能力的用途、当前缺口和缺失时造成的实际瓶颈。

状态定义：

- **已具备**：后端已经提供并已在 0721 稳定使用。
- **部分具备**：接口或局部 lowering 已存在，但缺少可直接用于 0721 的稳定物理流水、通用 shape 支持或验证。
- **缺失**：当前后端没有足够的原语、lowering 或内存 ownership 抽象，需要后端新增能力。

当前已知基线约为 TileLang `0.44-0.46 ms`、官方 `0.28 ms`。性能差距主要集中在 Cube 物理流水、V1 backward 大 tile 和稀疏数据搬运，而不是数学公式。

## 2. 能力需求总表

| 优先级 | TileLang-Ascend 能力 | 当前状态 | 为什么需要 | 缺失或不完整时对 0721 的性能影响 | 0721 对应阶段 |
| --- | --- | --- | --- | --- | --- |
| P0 | 可手工驱动的 L1 -> L0 -> MMA -> Fixpipe 路径：`T.copy(real_k, real_n)`、`T.mma(k_actual, n_actual, unit_flag)` | 部分具备 | C1-P 的一个输出块由 4 个 Query K 块和可选 Rope K 块累加；中间块要保留 L0C，最后一块直接交给 Fixpipe | 当前 C1 使用单个 L0A/L0B/L0C，L1->L0、MMA、FIX 大多串行；无法稳定掩盖 MTE1 和 FIX 等待 | C1-P，随后 C1-SY/C2 |
| P0 | phase-scoped 的物理精确 L0A/L0B/L0C ping/pong slot 与生命周期复用 | 缺失 | 官方 Matmul Service 为每个 phase 明确拥有两个 A/B/C 槽，并在 phase 之间复用 L0 容量 | 0721 以前分配过双槽且生成偏移正确，但出现 `L0C read/write conflict`；不能安全实现 MTE1、MMA、FIX 三层重叠 | C1-P、C1-SY、C2 |
| P0 | 经硬件验证的 MMA/FIX 复合 pipeline service 或等价 ownership lowering | 部分具备 | `unit_flag` 已能描述 MMA 与 paired Fixpipe，但手写 event 仍需覆盖 L0C 内部状态、首尾 drain 和跨输出轮换 | 当前需要保守的 `FIX_M/M_FIX` wait；一旦错误删除或重排会死锁、非法指令或 L0C 冲突，只能维持单槽/低重叠路径 | C1、C2 |
| P0 | 跨多个 vector repeat 保持同一 scalar 的 row-expand/broadcast lowering | 缺失 | 官方 backward 可用一个权重覆盖整行的大 tile，如 `4x2048`；0721 的 reluGrad 与 dWeight 都需要这种行广播 | `32x128` 会每 64 列错误切换权重，`4x2048` 的 repeat stride 会截断；只能回退到 `32x64`，增加 Vector 指令、同步和循环次数 | V1 backward |
| P0 | 动态 UB scalar 安全提升为寄存器标量 | 缺失 | 大 backward tile 需要动态读取每行 weight 后复用到整行计算 | 当前动态 `GetValue` 可能产生非 32B 对齐 UB 地址并触发 AICore 异常；`4x2048` 路线无法落地 | V1 backward |
| P1 | 泛化的 indexed gather、indexed scatter，以及运行时批量 DMA descriptor | 缺失 | 官方可按稀疏索引和相邻地址关系合并搬运；dKey 也需要按索引高效归并 | V0 仍逐行随机 gather 后写 GM workspace，V2 仍走确定性 scatter/finalize 和多次全局同步；DMA 发射、GM 流量和同步成本较高 | V0、V2 |
| P1 | 全路径动态尾块：GM/UB/L1/L0 DMA、MMA、Vector 和 scatter 都能使用实际有效 K | 部分具备 | `kRealSize < K` 时应只搬运、计算和写回有效列 | 新增 `real_k/real_n` 只覆盖 L1->L0/MMA 的一部分；0721 仍需要 `KExec`、零 gather、ValidKMask 和尾部 Vector 掩码，额外处理 padded 列 | V0、C1、V1、V2 |
| P1 | 任意物理行宽/stride 的 reduction、Compare、Select | 部分具备 | 尾块和大 tile 常需要在宽父 buffer 的窄列区域做 reduce 与 mask | 现有 reduction 和 Compare/Select 存在特定物理 shape 限制；0721 需要 4x64 分组、额外 mask/临时 buffer 或专用回退，限制大 tile 设计 | V1 P/SY/backward、tail path |
| P2 | 可选择 buffer 的软件流水多版本化，或显式禁止特定 scratch 被自动环化 | 缺失 | `T.Pipelined` 理论上可生成跨迭代流水，但 attention body 中并非所有 UB scratch 都应复制成多版本 | 当前自动流水可能把全部 scratch 按 `num_stages` 扩容，导致 UB 超限或运行时异常；0721 只能维持手工 event，代码复杂且调度保守 | 跨 C/V 的自动流水、V1 |
| P2 | 可审计的物理 buffer slot/offset 可视化与约束接口 | 缺失 | 手写高性能流水必须验证每个 L0/UB 槽没有被 planner 非预期合并或重叠 | 只能依赖生成 C++ 事后人工审计；试错成本高，保守同步和单槽设计被迫保留 | 全部手工流水阶段 |

## 3. 已新增但尚未接入的能力

`T.copy` 当前已支持 `real_k` 和 `real_n`，`T.mma` 已支持 `k_actual`、`n_actual` 和 `unit_flag`。这不是待后端实现的功能。

它允许一个 K 分块序列采用以下语义：

```text
中间 K 块：MMA(unit_flag=0b10)，结果累加并保留在 L0C
最后 K 块：MMA(unit_flag=0b11)，将 L0C 交给 paired Fixpipe
紧随写回：copy(L0C -> GM, unit_flag=0b11)
```

0721 尚未使用该能力，原因不是 API 缺失，而是需要同时解决双 L0A/L0B、双 L0C、phase 生命周期和手工 event ownership。当前记录在 `0721_pending_optimizations_zh.md`，待全量泛化精度稳定后再实施。

特别说明：`real_k/real_n` 描述的是实际矩阵分形形状，不能单独产生流水；物理双槽和正确的 MTE1/MMA/FIX ownership 才是掩盖时间的必要条件。

## 4. 当前性能瓶颈如何由能力缺口形成

### 4.1 C1：已有 GM 到 L1 预取，缺少 L1 到 L0 与 MMA 的稳定重叠

当前 C1-P 已能在当前 MMA 前发起下一块 GM 到 L1 的预取，但 C1 只有单套 L0A/L0B/L0C。稳态实际接近：

```text
GM -> L1(next) 与 MMA(current) 可部分重叠
L1 -> L0(current) 与 MMA(current) 不能重叠
FIX(current output) 与下一 output 的 MMA 不能稳定重叠
```

官方服务使用双物理槽后可形成：

```text
GM -> L1(tile n+2)
L1 -> L0(tile n+1)
MMA(tile n)
FIX(tile n-1 output)
```

因此 C1 的主要损失不是少一条 copy，而是每个 128 列 tile 都暴露 MTE1 和 FIX 等待。P0 的前三项能力共同解决这一问题。

### 4.2 V1 backward：后端约束迫使物理 tile 从官方大 tile 回退到 32x64

当前 `32x64` 是经过精度验证的最小安全选择，但一次 K=2048 要执行大量小块：

```text
每个 128 列 K tile -> 两个 32x64 列子块
每个 query 行组 -> 多轮 row-expand、Compare/Select、Cast、DMA 和 event
```

若 row-expand 能让一个动态寄存器 scalar 覆盖跨多个 repeat 的整行，且支持合法的大 stride，官方式 `4x2048` 或等价大 tile 才可行。其收益来自大幅减少 Vector 发射、临时 buffer 读写和同步，不是单次算术本身更快。

### 4.3 V0/V2：稀疏地址无法被批量 DMA 和原生 indexed 读写充分利用

V0 需要按 `SparseIndices` 读取 Key、KeyRope、KeyIndex。当前实现先逐行 gather 到 UB，再写入连续 GM workspace，保证了正确性，但会产生大量随机 DMA 发射。

V2 为保证 dKey 的确定性结果，采用 accumulator、分段 scatter、finalize 和阶段同步。缺少 indexed scatter/reduce 原语时，不能直接使用官方风格的高效索引归并，只能以更多 workspace 与控制逻辑换取正确性。

### 4.4 尾块：正确性已经补齐，但仍以完整执行宽度换取安全

`kRealSize`、零 gather 和 ValidKMask 已解决无效 sparse index 对精度的破坏，但它们没有让所有阶段都只处理有效列。尾块仍需要：

- 使用对齐后的 `KExec` 保障物理 tile 和 event 协议；
- 对 padded 列做零填充、mask、reduce 与最终 scatter 保护；
- 保留完整 workspace shape，避免下游读取未初始化列。

因此 tail-protected 路径的额外时间主要来自 padded 数据搬运、Vector mask 和更保守的执行宽度。全路径动态尾块能力才能从根源削减这些工作。

## 5. 不应误归因于后端缺失的事项

以下问题也影响性能，但不应要求后端新增功能来解决：

- launch core 数、V2 分区和三次 `SyncAll` 的取舍属于 0721 的调度策略；当前 4 核优于 8/24 核是已测事实。
- full-valid 与 tail-protected 的 host 侧 variant 选择开销很小，不是当前 0.1 ms 级差距的来源。
- C2 当前约十几微秒，即使实现 phase double buffer，理论收益也远小于优先修复 C1 和 V1 大 tile。
- 官方与 TileLang 的部分数值顺序、workspace 排布和核间协作不同；不能把所有性能差都归结为单一原语缺失。

## 6. 建议的后端投入顺序

1. 先将现有 `real_k/real_n/unit_flag` 与可验证的 phase-scoped 物理 L0 slot 组合，完成 C1 的稳定双缓冲服务。
2. 完善动态寄存器 scalar 与跨 repeat row-expand，解锁 V1 backward 大 tile。
3. 提供动态 indexed gather/scatter 和批量 DMA descriptor，减少 V0/V2 的数据搬运与控制开销。
4. 将 reduction、Compare、Select 的物理 shape/stride 支持泛化，减少尾块和大 tile 的专用回退。
5. 最后改进 `T.Pipelined` 的选择性多版本化和物理 slot 可视化，降低手工流水维护与调试成本。

每项后端能力都应先有最小 kernel 的设备精度、生成 C++ 和 timeline 验证，再接入 0721。不能因 API 已暴露就假定硬件重叠已经生效。
