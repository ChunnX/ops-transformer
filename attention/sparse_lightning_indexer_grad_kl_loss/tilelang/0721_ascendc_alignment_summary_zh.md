# 0721 与官方 AscendC 对齐状态及后续优化路线

## 1. 文档范围

本文记录 `tilelang/0721.py` 在 Case 0 下的当前实现状态、与官方 arch22 AscendC
算子的结构差异、已验证优化、失败回退项、TileLang-Ascend 后端能力边界，以及仍可
继续尝试的性能路线。

当前主要测试形状：

```text
B=1
S1=1
S2=2048
K=2048
Query=128x512
QueryIndex=64x128
TopK=128
```

官方参考实现：

- `../op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_service_cube.h`
- `../op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h`
- `../op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_base.h`

当前 4 核、`CompareScalar + Select` 版本的重复测试结果约为：

```text
TileLang 中位数：0.444-0.464 ms，整体中位约 0.455 ms
官方中位数：    0.275-0.288 ms，整体中位约 0.284 ms
相对性能：      约 0.62-0.63x

dQueryIndex max_diff = 6.103516e-05
dKeyIndex   max_diff = 7.629395e-06
dWeights    max_diff = 2.441406e-04
Loss        abs_diff = 0
```

这版精度比此前指数近似 ReLU mask 的基线更接近官方，但距离官方 80% 性能目标
仍有明显差距。若官方为 `0.284 ms`，达到 80% 需要 TileLang 不高于约
`0.355 ms`，当前仍差约 `0.10 ms`。

## 2. 当前 0721 执行结构

0721 是单个混合 AIC + 双 AIV kernel，关闭自动同步并手工管理 event：

```python
TL_ASCEND_AUTO_SYNC = False
```

当前阶段顺序：

1. **V0 gather**：根据 `SparseIndices` gather Key、KeyRope、KeyIndex，写入 GM
   workspace，并发布 P/SY 就绪 flag。
2. **C1-P**：Query/QueryRope 与 gathered Key 做 Cube MMA，写出 `bmm1Res`。
3. **C1-SY**：QueryIndex 与 gathered KeyIndex 做 Cube MMA，ReLU 后写出
   `bmm2Res`。
4. **V1-P/SY**：AIV0 计算 PReduced，AIV1 计算 SoftmaxI；两侧完成后汇合。
5. **V1 backward**：两个 AIV 各处理 32 行，以 `32x64` 物理 tile 计算
   reluGrad、dWeight 和 Loss partial。
6. **C2**：计算 dKey partial 与 dQueryIndexAccum。
7. **V2**：清零 dKey accumulator、多核 deterministic scatter、输出 Cast 和
   Loss 汇总。

当前 launch core 数不是固定硬件核数，而是按主任务和 V2 工作量计算：

```text
task_count = B * S1
scatter_batches = ceil(K / 16)
finalize_batches = ceil(B * S2 / 16)
v2_batches = max(scatter_batches, finalize_batches)
v2_core_need = ceil(v2_batches / (2 AIV/core * 16 batches/AIV))
launch_core_num = min(hardware_core_num,
                      max(task_count, v2_core_need, 1))
```

当前形状选择 4 核。实测 4 核整体中位约 `0.455 ms`，8 核约 `0.461 ms`，
24 核 Compare/Select 版本约 `0.483 ms`。缩核减少了三次 `SyncAll` 的参与者、
V2 atomic 竞争和尾部分区开销。

## 3. 已经对齐官方 AscendC 的部分

### 3.1 数学公式和阶段边界

- PReduced、SoftmaxI、dWeight、reluGrad、dQuery、dKey 和 Loss 的公式已逐阶段
  对照 host 公式。
- gather、BMM1/BMM2、P/SY、reluGrad、dQ/dK 均有可独立观测的 workspace。
- Loss partial 使用 32 字节对齐槽，避免标量 MTE3 写回覆盖相邻 partial。
- dKey 使用 FP32 accumulator，最终统一 Cast 到 FP16。

### 3.2 单 kernel 混合流水

- 与官方一样，主计算保留在一个混合 AIC/AIV kernel 中，没有拆出额外 kernel。
- V0、C1、V1、C2、V2 通过 cross-core flag 和核内 event 衔接。
- 当前 shape 的主 V0/C1/V1/C2 任务只由 core0 执行，额外 core 仅参与 V2。

### 3.3 C1 的不变输入常驻

- Query 的四个 `128` 维 chunk 使用四个物理精确 L1 buffer，每个 task 只从 GM
  读取一次。
- QueryRope 常驻独立 L1 cache。
- C1-SY 的 QueryIndex 对当前 `DQueryIndex=128` 只加载一次到 L0A，循环内只更新
  KeyIndex 的 L0B。

### 3.4 C1-P 单槽软件预取

- 当前 Key/KeyRope tile 从 GM 到 L1 后，MTE1 消费当前块，MTE2 可继续预取下一
  Query 维度或 Rope 块。
- 该路径保留精确 L0A/L0B/L0C 形状，不使用大父分配切片。

### 3.5 V1 P/SY 分工与提前启动

- AIV0 只等待 C1-P 的 flag 后开始 P。
- AIV1 只等待 C1-SY 的 flag 后开始 SY。
- 两侧完成后通过 AIV -> AIC -> AIV relay 汇合，再进入 backward。
- 该结构利用了 C1-SY 期间 AIV0 的空闲窗口，profile 约减少数微秒。

### 3.6 V1-SY 原地缓存 exp

- SoftmaxI 的第二遍将 `exp(ReduceSumY - max)` 原地保存在历史 buffer 中。
- 第三遍直接除以全局 sum，不再重复 `broadcast(max) -> sub -> exp`。
- 精度保持，属于真实 Vector 工作量削减。

### 3.7 官方式 ReLU mask

当前 backward 使用：

```text
CompareScalar(bmm2 > 0)
weight * (SoftmaxI - PReduced)
Select(mask, value, 0)
Cast
```

该路径与官方 `CompareScalar + Select` 语义一致，精度显著优于此前
`1 - exp(-10000 * relu_output)` 近似。

### 3.8 backward 单槽 WAR ownership

生成 C++ 中 `v1_bw_bmm2` 的 FP32 输入和 `v1_bw_rg_half` 的 FP16 输出复用同一
UB 地址。当前使用直接的 `MTE3_MTE2` ownership：

```text
下一轮 MTE2 refill
  必须等待
上一轮 MTE3 完成 reluGrad store
```

这条 WAR 边消除了历史上的非确定性 NaN、Inf 和接近 FP16 最大值的异常数据，
必须保留。

### 3.9 已保留的微优化

- `v1_bw_rg = weight_2d * diff_2d` 直接三操作数 Mul，删除一次 2048-FP32
  UB-to-UB copy 和一个 barrier。
- Loss 直接在 `1x64` 二维 reduce 输入 buffer 中原地计算，删除
  `Muls(..., 1.0)` 复制。
- V2 scatter 和 finalize 使用 UB ping-pong。
- launch core 数按 V2 每 AIV 约 16 个批次计算，当前形状选择 4 核。

## 4. 因 TileLang-Ascend 后端能力缺失而无法对齐的部分

### 4.1 精确物理 L0A/L0B/L0C 双缓冲 ownership

官方 `SLITMatmulService` 明确拥有：

```text
L0A: 2 x 32 KiB
L0B: 2 x 32 KiB
L0C: 2 x 64 KiB
```

并通过独立的 `MTE1_MTE2`、`M_MTE1`、`MTE1_M`、`FIX_M`、`M_FIX` event
轮换物理槽。0721 可以分配相同总容量，但当前裸 `T.mma` 路径无法表达官方
Matmul service 的完整硬件 ownership。已生成过物理 offset 正确的双槽 C++，
设备仍报告 `L0C read/write conflict`。

这使 0721 无法稳定实现：

```text
下一 tile GM->L1
  与 当前 tile L1->L0
  与 当前 MMA
  与 上一 tile FIX
同时重叠
```

### 4.2 phase-scoped 精确 L0 复用

官方在 ComputeMm1/Mm2/Mm5/Mm6 之间复用同一组物理 L0 槽。TileLang 的静态
memory planner 可以复用生命周期不重叠的地址，但不能可靠保证每个 phase 的
物理矩阵 descriptor 和 ownership 都等价于官方 service。

不能使用一个大 L0 父 allocation 再切片为不同形状。当前 codegen 可能让
L1->L0 copy 继承父 allocation 的物理 shape，导致错误 layout、FIX descriptor
或 illegal instruction。

### 4.3 安全寄存器标量

官方 `4x2048` backward 使用 `weightTensor.GetValue()` 得到寄存器标量，再对
一整行执行 Muls。当前 TileLang：

```python
T.tile.mul(row, row, weight_ub[index])
```

会 lowering 为动态 UB `GetValue(index)`。float 权重按 4 字节递增，只有
`index % 8 == 0` 时满足 32 字节对齐；其余索引可能触发 UB 地址不对齐异常。
后端缺少将这种动态 UB scalar 安全提升为寄存器标量的 lowering。

### 4.4 跨多个 Vector repeat 的 row-expand

当前 `row_expand_mul_experiment` 对每行 64 个 FP32 元素有效，因为一行恰好
对应一个 Vector repeat。

对 `32x128`：

- 一行需要两个 repeat；
- weight source 却每个 repeat 前进一次；
- 同一行前后 64 列使用不同权重，reluGrad 错误。

对 `4x2048`：

- repeat stride 可能生成 `256`；
- 底层 immediate 是 `uint8_t`，发生 `256 -> 0` 截断；
- 生成非法 Vector 配置。

因此当前 `32x64` 是已验证的最大完整 backward row-expand tile。

### 4.5 可控的 UB 真双槽

逻辑上分配 ping/pong buffer 不保证生成两个安全物理槽。memory planner 可能在
生命周期分析不完整时合并地址；同时 V、MTE2、MTE3 对同一物理区域存在 RAW/WAR
依赖。当前后端缺少易于审计的“固定地址、固定槽 ownership”抽象。

### 4.6 动态批量 DMA descriptor

官方 V0 可以根据两个 sparse block 的实际地址关系，动态设置 block count、
block length 和 stride；地址不合法时回退为两次 copy。普通 `T.copy` 很难表达
同等的运行时 `DataCopyExtParams` 和 fallback，因此官方的 sparse-block 合并 DMA
不能直接完整迁移。

### 4.7 复合 Matmul service 能力

TileLang 有 `T.mma`、MTE copy 和 event，但缺少与官方 `SLITMatmulService`
等价的、经过硬件验证的复合 pipeline。手工拼接 event 可以表达可见依赖，却不一定
覆盖硬件内部 L0C/FIX 状态机。

## 5. 已尝试但失败或回退的优化

| 尝试 | 结果或错误标志 | 根因 | 当前决策 |
| --- | --- | --- | --- |
| 参考 `0715_bn128.py` | dQ/dK 仍为 NaN | 该文件本身不是精度基线 | 不再参考 |
| caller workspace 标为 `workspace_idx` | debug workspace 全零 | JIT 分配私有 workspace，忽略 caller buffer | 保持 `workspace_idx=[]` |
| V1 backward 双槽 ping-pong | NaN、大误差或不稳定 | UB 别名、V/MTE3/MTE2 ownership 不完整 | 回退单槽 |
| 删除 backward `MTE3_MTE2` | reluGrad NaN/Inf | MTE2 覆盖 MTE3 正在读取的 FP16 输出 | 永久禁止 |
| P/SY 128 列外层缓存 | 首版死锁，修复后无收益 | 漏发 `V_MTE2`；修复后 GM 字节不变且新增 UB copy/event | 回退 |
| backward `32x128` | dQ/dK 大误差，dWeight/Loss 正确 | row-expand 每 64 元素切换权重 | 回退 32x64 |
| backward `4x2048` row-expand | 编译 warning、运行卡住或异常 | stride `256 -> uint8_t 0` | 回退 |
| backward `4x2048` 动态 scalar | UB address not aligned/AICore exception | 动态 `GetValue` 非 32B 对齐 | 回退 |
| C1-P 物理双缓冲 | `L0C read/write conflict` | 裸 `T.mma` 无法复现完整 L0C ownership | 回退，仅保留实验宏 |
| C1-P/C1-SY task 拆分 | 性能下降或无收益 | phase 拆分未恢复官方物理槽复用 | 回退 |
| C1-SY/C2 单槽预取 | 持平或变慢 | 控制前导和 event 成本抵消 MTE2 重叠 | 回退 |
| C2 QueryIndex 常驻 L0B | 精度正确但无收益 | 形成首轮 MTE1 前导，未缩短关键路径 | 回退 |
| V0 P/SY 两阶段 gather | 精度正确但中位数无收益 | 增加 GM 竞争，未改善单槽 C1 关键路径 | 回退 |
| per-task dKey scatter、减少 SyncAll | NaN 或大误差 | atomic/scatter finalize ownership 和阶段顺序错误 | 回退 deterministic V2 |
| 将三次 SyncAll 收敛 | 性能未改善或精度失败 | clear/scatter/finalize 存在真实全局依赖 | 保留三个阶段边界 |
| C2 L1 单槽软件预取 | 无稳定收益 | C2 仅约 13 us，控制开销占比高 | 回退 |
| `Nidx1=8` 的 C2-dK `K=16` 尾部补零审计 | 精度通过；最初曾把生成的 `c2_*_l1[128]` 误判为错误偏移 | L1 使用 Cube `zN` 分形布局，逻辑第 8 行在首个 `16x16` 分形块内的偏移正是 `8*16=128`，不能按 ND row-major 计算为 1024 | 保留当前补零；以后必须结合 L1 layout 和 copy 描述符解释局部地址 |
| launch 24 核 | 约 0.483 ms | 三次 SyncAll、atomic 竞争、V2 尾部分支过多 | 改为自适应 |
| launch 1 核 | 性能倒退 | V2 每 AIV 串行批次过多 | 不作为当前 shape 默认 |
| launch 8 核 | 约 0.461 ms | 已明显优于 24 核，但略慢于 4 核 | 用于校准公式 |
| launch 4 核 | 约 0.455 ms | 每 AIV 16 个批次，分区完全均衡 | 当前默认模型 |

### 5.1 `Nidx1=8` C2-dK 补齐到 `K=16` 的地址审计修正

对应复现命令：

```bash
python attention/sparse_lightning_indexer_grad_kl_loss/tilelang/precision_test_general/test_0721_general.py \
  --case bsnd_n1_32_nidx8 --seed 46 --no-bench
```

该版本的物理 buffer 和 MMA 形状符合预期：`c2_qi_l1`、`c2_rg_l1` 均为
`16x128`，dKey MMA 为 `M=128, N=128, K=16`，dQuery 仍为
`M=8, N=128, K=128`。

源码意图是把零写入尾部 8 行：

```python
c2_qi_l1[8:16, 0:128]
c2_rg_l1[8:16, 0:128]
```

实际生成 C++ 为：

```cpp
copy_gm_to_l1(..., c2_qi_l1[128], C2ZeroPad[0], ...);
copy_gm_to_l1(..., c2_rg_l1[128], C2ZeroPad[0], ...);
```

最初按普通 ND row-major 把 `[128]` 解释为第 1 行，并据此认为正确偏移应为
`8*128=1024`。这个解释是错误的：`alloc_L1` 的矩阵由 Cube copy 路径按 `zN`
分形布局组织。对于 BF16 `16x128`，每个基本分形块为 `16x16`，逻辑第 8 行在
首个分形块内的偏移是 `8*16=128`。后续跨 N 分形块的步进由
`copy_gm_to_l1<16, 128>` 的形状和 copy 参数处理，而不是简单连续写入普通二维
数组。

两个 `Nidx1=8` 用例均保持有限值且精度达标，是当前补零地址有效的直接运行证据。
因此保留现有 `c2_*_l1[128]` lowering，不再实施“完整 `16x128` 清零后覆盖前缀”
的规避方案。以后审计 L1/L0 地址时，必须先确认 `zN/nZ` 等物理布局，不能只按
Python 逻辑 shape 推导线性偏移。

### 5.2 `bsnd_k3072` 生成 C++ 审计记录

对应复现命令：

```bash
python attention/sparse_lightning_indexer_grad_kl_loss/tilelang/precision_test_general/test_0721_general.py \
  --case bsnd_k3072 --seed 49 --no-bench
```

该用例为 `B=1, S1=4, S2=3329, N1=32, Nidx1=8, K=3072`。运行结果精度
达标，与下面的生成 C++ 静态覆盖审计一致。

本次生成 C++ 对 `K=3072` 的覆盖关系是完整的：

- V0 gather 的 `k_outer < 3`，每段覆盖 1024 行；
- C1-P 和 C1-SY 均为 `3 x 8 x 128 = 3072`，输出 GM stride 也是 3072；
- V1-P、V1-SY、V1 backward 和 C2 均循环 24 个 128 列 tile；
- V2 scatter 遍历 4 个主任务，并把每个任务的 3072 行分配到 7 个 launch core
  的两个 AIV；前三个 core 各处理 448 行，后四个 core 各处理 432 行，合计
  `3*448 + 4*432 = 3072`，未发现遗漏或重叠；
- `S2=3329` 使用 `S2_pad=3584` 的 FP32 accumulator，最终由 wrapper reshape 后
  截取前 3329 行，生成 C++ 对 padded 区域的读写不是本用例的越界证据。

该 C++ 中的 `c2_qi_l1[128]` 和 `c2_rg_l1[128]` 是 `zN` 分形布局下逻辑尾部
8 行的起点，不是错误的 row-major 偏移。当前结论修正为：**`Nidx1=8` 的
`K=16` padding 和 `K=3072` 的循环/调度均已通过本用例验证，未发现独立缺陷。**

## 6. 为什么累计很多微优化，整体提升仍有限

已完成的微优化主要删除：

- 若干 UB copy；
- 若干重复 Exp；
- 若干 64 元素 Muls；
- 少量 L1->L0A copy；
- 多余 core 和尾部分支。

它们通常带来 `1-20 us` 级收益。官方与 0721 仍有约 `100 us` 差距，核心并非
某一条冗余指令，而是官方在 C1/V1/C2 使用大 tile、物理 ping-pong 和跨 pipe
重叠。当前后端缺失恰好阻塞了这些结构性优化。

因此，不能用“源码更接近官方”代替“生成 C++ 的物理流水更接近官方”。只有
profile 中真实减少 WAIT_FLAG、让 MTE2/MTE1/MMA/FIX 重叠，才可能获得数量级更大
的收益。

## 7. 当前仍可继续尝试的优化

### 优先级 A：同为 4 核时 A/B ReLU mask

目前已知：

- 24 核下 Compare/Select 比指数近似慢约 15-20 us，但精度更好；
- 4 核 Compare/Select 已达到约 0.455 ms；
- 尚未在完全相同的 4 核调度下重新测试指数近似。

因此最干净的下一项是只切换 mask：

```text
A: CompareScalar + Select
B: Muls(-10000) + Exp + Sub + Mul
```

两侧都保持 4 核、32x64、同一 workspace 和同步。若指数近似在 4 核下仍快且精度
满足验收要求，可由需求决定“官方语义优先”还是“性能优先”。

### 优先级 B：跨 shape 校准自适应核数

当前模型以每个 AIV 约 16 个 V2 批次为目标。需要至少覆盖：

```text
K = 1024 / 2048 / 4096 / 8192
不同 B、S1、S2
```

验收重点是无尾分支、每 AIV 批次均衡，以及三次 SyncAll 后的最长 core 时长。

### 优先级 C：C2 分阶段物理双缓冲原型

当前 C2 同时保留 dK 和 dQ 的 L0C：

```text
dK L0C = 64 KiB
dQ L0C = 32 KiB
总计 = 96 KiB
```

无法直接再复制 dK L0C，因为会达到 `160 KiB > 128 KiB`。若要对齐官方，需要：

1. 先完成 C2-dK phase，使用两套 64 KiB L0C ping-pong；
2. drain dK event；
3. 再复用地址完成 C2-dQ phase。

该方案还要解决 reluGrad L1 在两个 phase 间复用或重复 GM 读取。由于当前 profile
C2 仅约 13 us，理想收益约数微秒，优先级低于 C1 后端能力。

### 优先级 D：V0 动态双 sparse-block DMA

需要先确认当前 TileLang-Ascend 是否能生成运行时 block count/stride。如果只能
展开成两次普通 copy，则不值得继续；如果能直接 lowering 为官方式
`DataCopyExtParams`，可减少 V0 gather 指令和前 30 us 独占区间的一部分。

### 优先级 E：修改 TileLang-Ascend 后端

这是突破 `0.45 ms` 平台期最有价值的方向：

1. 为 `row_expand` 增加“同一 scalar 保持多个 repeat”的 lowering；
2. 将动态 UB scalar 安全提升为寄存器标量；
3. 增加 phase-scoped、物理精确 L0A/L0B/L0C slot API；
4. 提供经过硬件验证的 MMA/FIX ping-pong service；
5. 提供可审计的固定 UB slot/地址标注。

完成这些能力后，才能安全重试官方 `4x2048 backward` 和 C1/C2 物理双缓冲。

## 8. 不建议继续尝试

1. 不再通过简单修改 `bw_col_block` 扩大到 128 或 2048。
2. 不删除已验证的 `MTE3_MTE2` WAR。
3. 不在当前 UB planner 上再次堆叠 backward ping-pong event。
4. 不使用大 L0 父 allocation 切片伪造不同物理矩阵。
5. 不把 deterministic dKey scatter 提前到每任务流水。
6. 不增加额外 kernel 只为删除少量 Vector 指令。
7. 不以单次 min 或单次加速比接受优化；必须比较重复中位数和绝对 TileLang 时间。

## 9. 当前结论

0721 已经对齐了官方数学逻辑、单 kernel 阶段结构、P/SY 分工、Compare/Select
语义、部分不变输入常驻和输出确定性处理。当前主要差距不是公式错误，而是
TileLang-Ascend 无法安全表达官方的物理 Cube 双缓冲、寄存器标量和大 K backward
tile。

在不修改后端的约束下，`0.44-0.46 ms` 是当前较可信的平台区间。继续优化应优先
做严格 A/B 和跨 shape 调度校准；若目标仍是官方 80%，应将工作重点转向
TileLang-Ascend 后端能力，而不是继续叠加 0721 源码级 event。
