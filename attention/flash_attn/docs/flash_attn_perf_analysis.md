# Flash Attention D=256 基本块切块选择的性能建模

> 适用范围：arch35（ascend950）非量化 FA（Q/K/V 为 fp16/bf16，dV = D = 256）。
> 对应代码：`attention/flash_attn`（config=4/5）；D=256 切块与 UB 优化正式上库于 [PR #9754](https://gitcode.com/cann/ops-transformer/pull/9754)。

## 1. 背景与问题

FA kernel 的性能目标是 **cube bound**：让 Mmad（Cube 单元）成为流水线瓶颈，MTE 搬入、Fixpipe 搬出、Vector softmax 全部被 Cube 计算遮挡。能否达到 cube bound，由**基本块的切块大小**决定：

- **M 维**（`mBaseSize`，Q/gS1 方向每 AIC 承担的块行数，即 bmm 的 M 维）：决定计算/访存比（算术强度）；
- **N 维**（`s2BaseSize`，KV 序列方向的块大小，即 bmm 的 N 维）：决定 Fixpipe/掩码的每行开销与 L1 KV 流水深度；
- **D 维**（`dBaseSize`=256 本身不可选，但 BMM2 输出必须按 dV 分块）：决定 L0C 子缓冲占用与轮转深度。

D=64/128 时按"计算/访存比 + L0C 容量"即可选出切块；**D=256 的特殊矛盾**在于：各级存储占用（UB/L1/L0A/L0B/L0C）随 D 近似线性膨胀，候选切块在空间上互相排斥，切块选择变成"计算/访存比阈值 × 存储层次容量上限"的**联合求解**。

本文先建立通用时间模型（§3-4），再逐维推导 D=256 的切块选择（§5-7），给出空间核算总表（§8），最后用 PR #9754 的实测数据验证（§9）。

**结论先行**：D=256 最终选择 **M=128（sOuter=64）、N=128（sInner=128）+ BMM2 按 dV/128 splitN**（config=4）；M 方向工作量不足的小规模场景（`gSize × maxSeqQ < 64`）保留 **M=64、N=256**（config=5）。

## 2. 硬件规格（单核）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| CoreNum | 32 | 核数 |
| Frequency | 1.65 GHz | 主频 |
| L1 | 512 KB | AIC 侧片上缓存（P/Q/KV 分片） |
| L0A | 64 KB | 左矩阵 A 缓冲（2×32K 子缓冲） |
| L0B | 64 KB | 右矩阵 B 缓冲（2×32K 子缓冲） |
| L0C | 256 KB | cube 输出/累加缓冲（4×64K 子缓冲） |
| UB | 256 KB | 统一缓冲（可用 248 KB） |
| L2Capacity | 128 MB | |
| L2BandWidth | 5.2 TB/s | 折合单核 ~98.5 B/cycle |
| HBM Size | 128 GB | |
| HBM BandWidth | 1.6 TB/s | 折合单核 ~30.3 B/cycle |

> 单核带宽换算：总带宽 ÷ 核数 ÷ 主频。L2：5.2×10¹² ÷ (32 × 1.65×10⁹) ≈ 98.5 B/cycle；HBM：1.6×10¹² ÷ (32 × 1.65×10⁹) ≈ 30.3 B/cycle。
> Cube 单周期完成一次 `16×16×16` mma，吞吐 FMA = 4096 FMA/cycle（模型假设）。

## 3. 基本块参数与代码映射

tiling 侧切块因子与 kernel 侧基本块的映射（**注意方向**：sOuter 是 Q/M 方向，sInner 是 KV/N 方向）：

| tiling 因子 | kernel 参数 | 含义（bmm 视角） | 取值 |
| --- | --- | --- | --- |
| sOuter | `mBaseSize = sOuter × CV_RATIO` | M 维：Q/gS1 方向每 AIC 的块行数 | 32 / 64 → M = 64 / 128 |
| sInner | `s2BaseSize` | N 维：KV 序列方向块大小 | 128 / 256 |

CV_RATIO=2（`KERNEL_TYPE_MIX_AIC_1_2`，1 个 AIC 配 2 个 AIV）：AIC 一次迭代计算 `mBaseSize` 行 Q，2 个 AIV 各负责 `M/2` 行的 softmax（Vec1）与输出累加（Vec2）。

D=256 的 config 与 (M, N, D) 对应（`op_kernel/utils/flash_attn_common_def.h`，host 侧选择见 `op_host/fa_adjust_sinner_souter.h` / `op_host/arch35/flash_attn_tiling.cpp`）：

| config | sOuter | sInner | M（=sOuter×2） | N（=sInner） | D / dV | host 选择条件 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 64 | 128 | 128 | 128 | 256 / 256 | D=256 且 `gSize × maxSeqQ ≥ 64`（prefill 主路径） |
| 5 | 32 | 256 | 64 | 256 | 256 / 256 | D=256 且 `gSize × maxSeqQ < 64`（小 M 场景） |

单次外层迭代（处理一个 KV 块）的两个 matmul：

- BMM1：`S_Q[M,D] × S_K[D,N] → MM1[M,N]`（Q·Kᵀ，L0C fp32 累加）
- BMM2：`P[M,N] × V[N,dV] → MM2[M,dV]`（attn·V）

D=256 时 BMM1 沿 K 维（D）拆 2 个 128 tile（`MatmulK<128,128,128>`）；BMM2 沿 N 维（dV）拆 2 个 128 tile（splitN，见 §7）。

## 4. 时间模型（通用公式）

以下时间均按**单次迭代总量**计（处理一个 N=s2BaseSize 的 KV 块），便于跨阶段一致比较。

### 4.1 Cube 计算时间

单次迭代两个 matmul（bmm1 = Q@Kᵀ、bmm2 = P@V，非量化 dV=D），FMA 总量 `2·M·N·D`：

$$ T_{cube} = \frac{2 \cdot M \cdot N \cdot D}{FMA} $$

### 4.2 MTE2 搬数时间

Q 块（`M·D`）仅在首次搬入一次，S2 足够大时被多次迭代摊薄，忽略；K/V 每次迭代搬入（`2·N·D`）：

$$ T_{mte2} \approx \frac{2 \cdot N \cdot D \cdot \text{sizeof}(kvDtype)}{Mix_{bw}} $$

其中 $Mix_{bw} \in [HBM_{bw},\ L2_{bw})$ 为 L2 命中与 HBM 缺失混合后的有效带宽，$Mix_{bw} = h \cdot L2_{bw} + (1-h) \cdot HBM_{bw}$（h 为 L2 命中率）。

### 4.3 Fixpipe 时间

bmm1/bmm2 结果由 L0C fixpipe 到 UB（fp32）：

$$ T_{fixpipe} = \frac{(M N + M D)\cdot \text{sizeof}(fp32)}{UB_{bw}} $$

### 4.4 Cube Bound 条件

FA 目标做到 cube bound，即 cube 为瓶颈：$T_{cube} > T_{mte2}$ 且 $T_{cube} > T_{fixpipe}$。代入上式化简：

$$ \text{（M 条件）}\quad M > \frac{2 \cdot FMA}{Mix_{bw}} = \frac{8192}{Mix_{bw}} \quad \text{—— 与 D、N 无关} $$

$$ \text{（N 条件）}\quad UB_{bw} > 2 \cdot FMA \cdot \frac{N+D}{N \cdot D} = 8192 \cdot \left(\frac{1}{D} + \frac{1}{N}\right) \quad \text{—— 与 M 无关} $$

算术强度视角（与 M 条件等价）：每 KV 字节的 FMA 数为 `M/2` FMA/B，Cube 满吞吐所需供给带宽为 `8192/M` B/cycle。

## 5. M 维（mBaseSize）的选择：访存比定下限，存储容量定上限

### 5.1 计算/访存比阈值

代入 `FMA = 4096`、`sizeof(kvDtype) = 2B`（fp16/bf16）：

| M | cube bound 所需 Mix_bw | 可达性 |
| --- | --- | --- |
| 64 | ≥ 128 B/cycle | 超过 L2 上界 98.5 → **任何 L2 命中率下均为 mte2 bound** |
| 128 | ≥ 64 B/cycle | L2 命中率 h ≥ ~50% 时可达 cube bound |
| ≥ 271 | ≥ 30.3 B/cycle | 任意带宽下 cube bound，但空间不可行（见 5.2） |

（h=50% 时 Mix_bw = 0.5×98.5 + 0.5×30.3 = 64.4，恰好越过 64 的门槛。）

L2 命中率由 SectionStreamK 的 section 切分保障：按 L2 预算（96MB token 成本）切 section，使同一 section 内各核反复访问的 KV 不被挤出 L2，大 S2 下 h 可稳定超过 50%。

### 5.2 存储容量上限

设 N=128、D=256，考察 M 的空间上界：

| 约束 | 公式 | M=256 时 | 结论 |
| --- | --- | --- | --- |
| L0C 子缓冲（BMM1 结果） | `M·N·4 ≤ 64K` | 128K > 64K ✗ | **M ≤ 128** |
| UB（仅 mmRes+vec2Res） | `2·(M/2)·dV·4 + (M/2)·dV·4` | 256K + 128K > 248K ✗ | M=256 不可行 |
| L1（P+Q+KV） | `3·M·N·2 + 2·M·D·2 + 4·64K` | 192K+256K+256K = 704K > 512K ✗ | M=256 不可行 |

### 5.3 结论：M=128（sOuter=64）

M=128 是"**超过 mte2 阈值（84）**"与"**L0C/UB/L1 空间可行（≤128）**"的唯一交点：既是不小于 84 的最小可用档位（sOuter 模板只有 32/64 两档），也是空间上界的最大值。此时 BMM1 结果 `128×128×4 = 64K` 恰好填满 1 个 L0C 子缓冲，mask `64×128 = 8K` 恰好填满 1 个 mask buffer，零浪费。

M=64 的 config=5 仅用于 `gSize × maxSeqQ < 64` 的场景：M 方向总行数不足 64，M=128 的块会大面积空跑；此时算术强度由 workload 决定，切块无法改善，改选 N=256 换取 KV 迭代次数减半（见 §6.4）。

### 5.4 单次迭代量化验证（M=128, N=128, D=256）

| 项 | 数值 | 周期数 |
| --- | --- | --- |
| cube | 2×128×128×256 = 8.39M FMA | **2048** |
| KV 搬入 | 2×128×256×2 = 131072 B | @98.5: 1330 / @64: 2048（临界）/ @30.3: 4325 |
| fixpipe 搬出 | (128×128 + 128×256)×4 = 196608 B | @128: 1536 / @96: 2048（临界） |
| Q 搬入（一次性） | 128×256×2 = 65536 B | S2=4096 时摊薄 ~2K B/iter，可忽略 |

对照 (M=64, N=256)：`T_cube` 同为 2048 cycles（M×N 相同），但 KV 搬入 `2×256×256×2 = 262144 B`，即使 100% L2 命中也需 2661 cycles > 2048 → **必然 mte2 bound**。这正是 D=256 必须把 mBaseSize 从 64 提升到 128 的根本原因（PR #9754 演进的最后一步）。

另注意：D=256 时 fixpipe 搬出量（196.6K/iter）**大于** KV 搬入量（131.1K/iter），fixpipe 是比 MTE2 更紧的第二约束（UB_bw > 96 才能被遮挡，见 §6.1）。

## 6. N 维（s2BaseSize）的选择：fixpipe 偏好大 N，空间强制 N=128

### 6.1 fixpipe 条件（软约束，偏好大 N）

$$ UB_{bw} > 8192 \cdot \left(\frac{1}{256} + \frac{1}{N}\right) \Rightarrow \text{N=128 时 } UB_{bw} > 96,\quad \text{N=256 时 } UB_{bw} > 64 $$

N 越大，BMM2 输出（`M×D`，与 N 无关）被更多 KV 行摊薄，fixpipe 每行开销越小——方向上偏好大 N。

### 6.2 mask UB（硬约束）

mask buffer 固定 `2×8K`（`UB_MASK_BUF_BYTES = 8192`）：每 AIV 的掩码行 = `(M/2) × N × 1B ≤ 8192`，即：

$$ M \cdot N \le 16384 \Rightarrow \text{M=128 时 } N \le 128;\quad \text{M=64 时 } N \le 256 $$

### 6.3 L0C（硬约束）

BMM1 结果 `M×N×4B ≤ 64K` 子缓冲，同样得到 **M·N ≤ 16384**——与 6.2 是同一个界（L0C 64K/4B = mask 8K/(1B×½)），两条约束在 (128,128) 与 (64,256) 两点上恰好同时贴满。

### 6.4 L1 KV 流水深度（软约束，偏好小 N）

KV 的 L1 slot 大小 = `N × D × 2B`（K/V 分片交替占用）：

| N | L1 KV slot | 槽数 | L0B | 流水 |
| --- | --- | --- | --- | --- |
| 128 | 64K | 4（`L1_KV_BUFCNT=4`） | 双缓冲 | MTE2→MTE1→M 预取深度足（配合 PRELOAD_N=2） |
| 256 | 128K | 2（`L1_KV_LARGE_BUF`） | 单缓冲 | 流水深度减半 |

### 6.5 结论：M=128 时 N=128

(M=128, N=256) 同时违反 mask（64×256=16K > 8K）、L0C（128×256×4=128K > 64K）、L1 KV（128K/slot 仅 2 槽）三条约束 → 不可行。(M=64, N=256) 恰好贴着 `M·N = 16K` 的界（mask 32×256=8K、BMM1 64×256×4=64K 均恰好），作为 config=5 保留给小 M 场景；且 N=256 使 KV 迭代次数减半，对 M 受限的 workload 更有利。

## 7. D 维分块：BMM2 splitN 化解 L0C 越界

D=256 时 BMM2 结果 `M × dV × 4 = 128×256×4 = 128K`，超过 L0C 64K 子缓冲 → 直接写出会越界。

**方案（PR #9754 落地形态，`IterateBmm2l0Split`）**：

- `nLoops = ceil(dV/128) = 2`，每次 `MatmulFull<128,128,s2>` 只算 128 列，结果 `128×128×4 = 64K` 恰好 1 个 L0C 子缓冲；
- 原 `MatmulN` 内部本就按 N-tile 做 2 次 Mmad（写 L0C 的 `[0]` 和 `[64K]` 偏移）→ **拆到外部后 Mmad 次数不变**，只是把内部循环外提到子缓冲轮转层；
- 每 tile 独立 `Mutex` 加锁 + `FixpipeMm2PartialN`（UB 目的偏移 `n×128`，`dstStride=dV` 不变）。

**代价与收益**：

| 项 | D=128（config=2） | D=256（config=4，splitN） |
| --- | --- | --- |
| 每 loop L0C 子缓冲消耗 | 2（BMM1 1 + BMM2 1） | 3（BMM1 1 + BMM2 2） |
| L0C 轮转深度（4 槽/每 loop 消耗） | 2.0 | **1.33** |
| Mmad 次数 | — | 不变 |
| Fixpipe 次数/写入量 | 1 次/64K | 2 次/共 128K（同一 UB slot 的不同 N 区域） |

L0C 轮转深度从 2.0 降到 1.33 是 splitN 的固有代价（M 管线领先 FIX 管线的余量变小），实测未成为新瓶颈（§9）。

**配套的 UB mmRes 槽数**：D>128 时 `UB_MM_RES_BUFCNT = 2`（4×64K = 256K > 248K 放不下），D≤128 时 4 槽。2 槽下用 `mmResBufId_` 单调递增计数器分配 slot，避免 BMM1(L) 与 BMM2(L−2) 映射到相同 slot+flag 产生碰撞。

**L0A/L0B 并发**：32K 子缓冲×2。BMM1（Q k-tile 32K + K tile 32K）与 BMM2（P 32K + V tile 32K）并发各占 1 → 2/2 恰好满载，零余量；Mmad 完成后立即释放（Set M_MTE1），不死锁。

## 8. 空间核算总表

### 8.1 UB（config=4：M=128, N=128, D=256，每 AIV）

| buffer | 公式 | 大小 |
| --- | --- | --- |
| mmRes ×2 | `2 × (M/2) × max(N,dV) × 4` | 128K |
| mask ×2 | 固定 `2 × 8K` | 16K |
| vec2Res | `(M/2) × Align64(dV) × 4` | 64K |
| vec1Res ×2 | `2 × (M/2+1) × N × 2`（+1 为 NZ 格式 stride padding） | 32.5K |
| softmaxMax/Sum/Exp | `3×3 × Align(M/2×4, 256)` | 2.25K |
| lseOut ×2 | `2 × Align(M/2×8×4, 256)` | 4K |
| softmaxTmp | 固定 | 0.5K |
| **合计** | | **247.25K / 248K（余 768B）** |

mB=128 能放进 UB 的前提是 PR #9754 的两步前置优化（演进早期 UB 同为 247.25K 但 mB=64）：

- **vec1Res/vec2Res 动态化**：由硬编码（33024B/32768B）改为按 `(M, N, dV)` 公式推导，小 config 不再为大 config 买单；
- **mm1Res/mm2Res 合并为 mmRes**：slot 取 `max(mm1, mm2) = (M/2)×max(N,dV)×4`，D=256 时节省 64K；同步上用 2-flag 环形链（同步 ID 标识资源而非操作者）替代两条独立链。

### 8.2 L1（config=4，每 AIC）

| buffer | 公式 | 大小 |
| --- | --- | --- |
| L1_P ×3 | `3 × M×N×2` | 96K |
| L1_Q ×2 | `2 × M×D×2` | 128K |
| L1_KV ×4 | `4 × (N×D×2 = 64K)` | 256K |
| **合计** | | **480K / 512K** |

### 8.3 config=4 vs config=5 总览

| 项 | config=4（M=128, N=128） | config=5（M=64, N=256） |
| --- | --- | --- |
| cube bound 可达性 | h ≥ ~50% 可达 | 不可达（需 Mix_bw ≥ 128 > 98.5） |
| UB | 247.25K / 248K | ~149.75K / 248K |
| L1 | 480K / 512K（P 96K + Q 128K + KV 256K） | 416K / 512K（P 96K + Q 64K + KV 2×128K） |
| L1 KV 槽 / L0B | 4 × 64K / 双缓冲 | 2 × 128K / 单缓冲 |
| L0C 每 loop 消耗 / 深度 | 3 槽 / 1.33 | 3 槽 / 1.33（BMM2 每 tile 仅 32K，半槽浪费） |
| 适用场景 | prefill（M 工作量充足） | `gSize×maxSeqQ < 64`（decode/小 batch） |

## 9. 实测验证（PR #9754，ascend950）

演进路径（每步均由上述建模驱动）：

1. **vec1Res/vec2Res 动态化** —— UB 按公式精确分配；
2. **mm1Res/mm2Res 合并 + 2-flag 环形同步** —— D=256 腾出 64K；
3. **s2: 256 → 128** —— L1 KV 恢复 4×64K 深轮转，BMM1 改用 `MatmulK`；
4. **mB: 64 → 128 + BMM2 splitN + L0C 4×64K 共享轮转** —— 越过 §5.1 的 84 阈值；L0C 归一避免 D=128/D=256 路径分叉，并恢复 master 执行顺序（先 BMM1 后 BMM2）保住 M/FIX 流水重叠；
5. **config 重编号** —— 删除死 config，最终 config=4/5。

性能（相对优化前基线）：

| 用例 | 优化前 | 优化后 | 变化 |
| --- | --- | --- | --- |
| D=256 Prefill 4096 | 434.71 us | 245.21 us | **-43.6%** |
| D=256 Prefill 1024 | 33.03 us | 26.58 us | **-19.5%** |
| D=256 Decode 1×2048 | 19.74 us | 18.81 us | -4.7% |
| D=128 Prefill 4096 | 131.77 us | 131.54 us | -0.2%（无退化） |
| D=128 Prefill 1024 | 17.36 us | 17.25 us | -0.6%（无退化） |
| D=128 Decode | 11.82 us | 11.52 us | -2.5% |

D=256 大幅提升且 D=128 无退化，验证了"M=128 越过算术强度阈值 + N=128/D 维 splitN 满足空间约束"的切块选择。

## 10. 结论

- **M 维**：cube bound 要求 `M > 8192/Mix_bw`（L2 全命中下界 84，全 HBM 缺失下界 271）；L0C/UB/L1 空间上界 M ≤ 128。**M=128 是唯一同时满足两者的取值**，配合 section 级 KV 的 L2 复用（h ≥ 50%）达到 cube bound。
- **N 维**：fixpipe 条件 `UB_bw > 8192(1/D + 1/N)` 偏好大 N，但 mask 8K 与 L0C 64K 共同强制 `M·N ≤ 16384` → M=128 时 **N=128**；M=64 时 N=256（config=5，小 M 场景）。
- **D 维**：BMM2 按 dV/128 splitN，以 L0C 轮转深度 2.0→1.33 为代价化解 128K 越界，Mmad 次数不变。
- **D=256 的方法论**：先算每 KV 字节算术强度（M/2 FMA/B）定 M 下限，再逐级核对 L0C→UB→L1 空间上限，最后用 fixpipe/L1 流水深度核对 N 与 D 分块——计算/访存比给出"能选多大"，存储层次给出"只许多大"。
- **待补充**：UB 带宽 `UB_bw` 实测值，用于 N 条件（6.1 节）的数值封闭；D=256 下 fixpipe 搬出量大于 KV 搬入量，是仅次于 MTE2 的第二约束，值得单独 profiling。
