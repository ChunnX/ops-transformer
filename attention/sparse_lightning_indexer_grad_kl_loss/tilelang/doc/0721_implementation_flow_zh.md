# 0721 TileLang 实现流程详解

## 1. 文档目的

本文说明 `tilelang/0721.py` 当前版本的完整实现流程，目标读者是不熟悉
`sparse_lightning_indexer_grad_kl_loss` 算子、TileLang 或 Ascend AIC/AIV
流水线的开发者。

读完本文后，应当能够回答以下问题：

1. 这个算子解决什么问题，输出为什么只有三类梯度和一个 loss。
2. `Query/Key` 与 `QueryIndex/KeyIndex` 分别扮演什么角色。
3. `K`、`KRealSize`、`KExec` 和 `KScatterExec` 有什么区别。
4. 主机入口如何选择 full-valid 与 tail-protected 两种静态版本。
5. V0、C1、V1、C2、V2 五个阶段分别计算什么。
6. AIC、两个 AIV、GM、UB、L1、L0A、L0B、L0C 之间如何传递数据。
7. 手工 flag、task slot、全核同步为什么是正确性的一部分。
8. 当前实现为了精度、性能和 TileLang-Ascend 后端限制采用了哪些策略。

本文以以下文件为准：

- `tilelang/0721.py`：主机封装、kernel builder 和五阶段计算主体。
- `tilelang/slig_debug.py`：从主执行文件中拆出的主机侧诊断逻辑。
- `tilelang/precision_test_general/test_0721_general.py`：泛化精度和性能测试。

## 2. 算子要解决的问题

这个算子训练一个“稀疏索引器”。索引器需要从长度为 `S2` 的 Key 序列中，
为每个 Query 位置选出最多 `K` 个候选位置，并让这些候选位置上的索引器分布
尽量接近完整注意力给出的教师分布。

它同时使用两套特征：

- `Query/Key`：维度固定为 512，用于重建完整注意力概率，形成教师分布 `P`。
- `QueryIndex/KeyIndex`：维度固定为 128，用于训练实际的稀疏索引器，形成
  索引器分布 `SoftmaxI`。

`SparseIndices` 已经给出每个 Query 对应的候选 Key 位置。算子不会在这里重新
执行 top-k 搜索，而是：

1. 按 `SparseIndices` gather 候选 Key 和 KeyIndex。
2. 在候选集合上重建教师分布 `P`。
3. 在同一候选集合上计算索引器分布 `SoftmaxI`。
4. 计算 `KL(P || SoftmaxI)`。
5. 输出对 `QueryIndex`、`KeyIndex` 和 `Weights` 的梯度。

`Query/Key` 只用于提供教师概率，本算子不输出它们的梯度。

## 3. 参数、形状与符号

### 3.1 主要维度

| 符号 | 含义 | 当前约束 |
| --- | --- | --- |
| `B` | Batch 数 | 泛化测试覆盖多个取值 |
| `S1` | Query 序列长度 | 每个 `(batch, query)` 是一个主任务 |
| `S2` | Key 序列长度 | 主机侧补齐为 `S2Pad` 后用于 dKey 累加 |
| `N1` | Query 头数 | `32/64/128` |
| `N2` | Key 头数 | MQA 固定为 `1` |
| `G` | 每个 Key 头对应的 Query 头数 | `N1 // N2`，当前等于 `N1` |
| `Nidx1` | QueryIndex 头数 | `8/16/32/64` |
| `Nidx2` | KeyIndex 头数 | 固定等于 `N2=1` |
| `DQuery` | Query/Key 特征维度 | 固定 `512` |
| `DQueryIndex` | QueryIndex/KeyIndex 特征维度 | 固定 `128` |
| `KTop` | 输入 SparseIndices 的容量 | `1024..8192`，且是 `1024` 的倍数 |
| `DRope` | RoPE 附加特征维度 | 可为 0，kernel 内按 128 补齐 |

### 3.2 BSND 输入形状

| 输入 | 形状 | 数据类型 | 作用 |
| --- | --- | --- | --- |
| `Query` | `[B, S1, N1, 512]` | FP16/BF16 | 教师注意力 Query |
| `Key` | `[B, S2, 1, 512]` | 与 Query 相同 | 教师注意力 Key |
| `QueryIndex` | `[B, S1, Nidx1, 128]` | 与 Query 相同 | 稀疏索引器 Query |
| `KeyIndex` | `[B, S2, 1, 128]` | 与 Query 相同 | 稀疏索引器 Key |
| `Weights` | `[B, S1, Nidx1]` | 输入类型或 FP32 | 各索引头的组合权重 |
| `SparseIndices` | `[B, S1, 1, KTop]` | INT32 | 每个 Query 的候选 Key 下标 |
| `SoftmaxMax` | `[B, 1, S1, G]` | FP32 | 完整注意力 softmax 的行最大值 |
| `SoftmaxSum` | `[B, 1, S1, G]` | FP32 | 完整注意力 `exp(score-max)` 的行和 |
| `QueryRope` | `[B, S1, N1, DRope]` | 与 Query 相同 | 可选 RoPE Query 特征 |
| `KeyRope` | `[B, S2, 1, DRope]` | 与 Query 相同 | 可选 RoPE Key 特征 |
| `actual_seq_lengths_query` | `[B]` | INT32 | 每个 batch 的真实 Query 长度 |
| `actual_seq_lengths_key` | `[B]` | INT32 | 每个 batch 的真实 Key 长度 |

`SparseIndices < 0` 表示无效候选。当前 V0 会将这种索引映射到统一的 GM 零哨兵
行；V2 scatter 会跳过它。非负索引由调用方保证小于真实 `S2`。

### 3.3 输出形状

| 输出 | 形状 | 数据类型 | 含义 |
| --- | --- | --- | --- |
| `dQueryIndex` | `[B, S1, Nidx1, 128]` | 输入类型 | QueryIndex 梯度 |
| `dKeyIndex` | `[B, S2, 1, 128]` | 输入类型 | KeyIndex 梯度 |
| `dWeights` | `[B, S1, Nidx1]` | 与原 Weights 一致 | 权重梯度 |
| `Loss` | `[1]` | FP32 | 所有 batch、query 和有效 K 的 KL loss 总和 |

kernel 内部的 `dWeights` 按输入计算类型存储。如果调用方传入 FP32 Weights，
主机封装会在 kernel 返回后把 `dWeights` 转回 FP32。

## 4. 数学语义

下面省略 batch 和 query 下标，只描述一个 `(b, q)` 任务。

### 4.1 Gather 候选数据

对候选位置 `k`：

```text
source = SparseIndices[b, q, 0, k]
GatherKey[k]      = Key[b, source, 0]
GatherKeyRope[k]  = KeyRope[b, source, 0]
GatherKeyIndex[k] = KeyIndex[b, source, 0]
```

若 `source < 0`，三条 gather 结果都写为 0。

### 4.2 教师分布 P

对教师头 `h` 和候选位置 `k`：

```text
TeacherScore[h, k]
    = dot(Query[h], GatherKey[k])
    + dot(QueryRope[h], GatherKeyRope[k])
```

`QueryRope/KeyRope` 不存在时，第二项为 0。C1 只计算未缩放的 score，V1 再使用
外部传入的完整注意力 softmax 统计量：

```text
PHead[h, k]
    = exp(scaleValue * TeacherScore[h, k] - SoftmaxMax[h])
      / max(abs(SoftmaxSum[h]), 1e-8)

P[k] = mean_h(PHead[h, k])
```

这里的 `P` 在代码中保存为 `PReduced`。

### 4.3 稀疏索引器分布 SoftmaxI

对索引头 `g`：

```text
IndexerScore[g, k]
    = ReLU(dot(QueryIndex[g], GatherKeyIndex[k]))
```

C1 的第二个 MMA 在写回时已经执行 ReLU，因此 `bmm2Res >= 0`。

多个索引头通过 Weights 合成为一个候选 logit：

```text
WeightedScore[k] = sum_g(Weights[g] * IndexerScore[g, k])
SoftmaxI[k]      = softmax_k(WeightedScore[k])
```

softmax 只在当前 query 的有效 K 范围内计算。

### 4.4 KL loss 与梯度

```text
Loss = sum_k P[k] * (log(max(P[k], 1e-8))
                     - log(max(SoftmaxI[k], 1e-8)))
```

把教师分布 `P` 视为常量后，索引器 logit 的梯度为：

```text
Diff[k] = SoftmaxI[k] - P[k]
```

权重梯度：

```text
dWeights[g] = sum_k IndexerScore[g, k] * Diff[k]
```

经过 ReLU 与权重后的中间梯度：

```text
ReluGrad[g, k]
    = Weights[g] * Diff[k] * I(IndexerScore[g, k] > 0)
```

当前 kernel 不直接使用 Compare/Select 实现 `I(x>0)`，而使用：

```text
Gate(x) = 1 - exp(-10000 * x)
```

由于 `IndexerScore` 已经过 ReLU，`x=0` 时 Gate 精确为 0，正值时快速趋近 1。
这是当前后端下已经通过泛化精度验证的实现策略。

最后两个矩阵梯度为：

```text
dQueryIndex = ReluGrad @ GatherKeyIndex

dKeyPartial[k] = ReluGrad[:, k]^T @ QueryIndex
```

`dKeyPartial[k]` 仍然对应候选槽位。V2 根据 `SparseIndices[k]` 将它 atomic add
回原始 KeyIndex 行。多个候选或多个 query 指向同一个 Key 位置时会自然累加。

## 5. 四种 K 的区别

这是理解当前尾块实现的关键。

| 名称 | 来源 | 含义 |
| --- | --- | --- |
| `KTop` | `SparseIndices.shape[-1]` | 输入候选容量，也是原始 GM stride |
| `KRealSize[b,q]` | 运行时长度推导 | 当前 query 真正有效的候选数量 |
| `KExec` | 主机侧静态推导 | Cube/Vector 工作区和主循环实际编译宽度 |
| `KScatterExec` | 主机侧静态推导 | V2 需要读取和 scatter 的静态宽度 |

### 5.1 KRealSize

当前 query 位置 `q` 的有效候选数为：

```text
KRealSize[b, q]
    = clamp(actualKeyLen[b] - actualQueryLen[b] + q + 1,
            min=0,
            max=KTop)
```

`ValidKMask[b,q,k] = 1` 当且仅当 `k < KRealSize[b,q]`。

### 5.2 KExec

如果主机能够直接读取真实序列长度：

```text
maxRealK = max_b,q(KRealSize[b,q])
KExec    = max(1024, align_up(maxRealK, 1024))
```

这样输入 `KTop=4096`、最大真实长度约 1025 的用例，只编译和分配 `KExec=2048`
的主计算工作区。`SparseIndices` 仍保持原始 `KTop` stride，kernel 只访问前
`KExec` 列，不产生主机侧裁剪复制。

如果长度只存在于 NPU 上，主机不会为了读取 `maxRealK` 引入同步，`KExec`
保守退化为 `KTop`。

### 5.3 KScatterExec

V2 以 16 行为一个 scatter batch：

```text
KScatterExec = align_up(maxRealK, 16)
```

并限制不超过 `KExec`。它只控制 V2 的静态 DMA 和循环上界；每个 task 仍会读取
一次自己的 `KRealSize`，保护不同 query 的实际尾部。

### 5.4 full-valid 静态版本

当每个 batch 在最早 query 位置都至少拥有完整 `KTop` 个有效候选时：

```text
actualKeyLen - actualQueryLen + 1 >= KTop
```

主机编译 `full_valid=True` 的专用版本。这个布尔值是 kernel builder 的 Python
静态参数，因此生成代码会删除：

- P/SY 的尾块 mask DMA、Compare 和 Select。
- backward 的 Loss mask 乘法。
- V2 对 `KRealSize` 的实际读取。

由于 TileLang 宏收集阶段仍会检查不可达分支中的 BufferRegion，主机仍传入合法
形状的 `KRealSize` 和 `ValidKMask` 占位张量，但 full-valid 生成代码不读取它们。

## 6. 总体执行架构

0721 是一个混合 AIC/AIV 单 kernel。TileLang 后端会把源码中的 `main` 入口规范化
为生成 C++ 中的 `main_kernel`。

每个物理 AI Core 包含：

- 1 个 AIC/Cube 执行单元，负责矩阵乘、L1/L0 和 FIX。
- 2 个 AIV/Vector 执行单元，`vector_id` 分别为 0 和 1。

主任务定义为一个完整的 `(batch, query)`：

```text
taskId = batch * S1 + query
taskCount = B * S1
```

任务按轮次分配给物理核：

```text
task = taskIter * launchCoreNum + coreId
```

端到端数据流如下：

```mermaid
flowchart LR
    Host["主机封装与静态 specialization"] --> V0["V0: 稀疏 Gather"]
    V0 --> GWS["Gather 工作区"]
    GWS --> C1P["C1-P: 教师分数 MMA"]
    GWS --> C1S["C1-SY: 索引器分数 MMA + ReLU"]
    C1P --> VP["AIV0: PReduced"]
    C1S --> VS["AIV1: SoftmaxI"]
    VP --> Relay["AIC relay: P/SY 会合"]
    VS --> Relay
    Relay --> BW["V1 backward: dWeight / ReluGrad / LossPartial"]
    BW --> C2["C2: dQueryAccum / dKeyPartial"]
    C2 --> Barrier1["SyncAll"]
    Barrier1 --> Clear["所有 AIV 清零 dKey FP32 accumulator"]
    Clear --> Barrier2["SyncAll"]
    Barrier2 --> Scatter["V2: atomic scatter dKeyPartial"]
    Scatter --> Barrier3["SyncAll"]
    Barrier3 --> Final["Cast 输出与 Loss 汇总"]
```

`TL_ASCEND_AUTO_SYNC=False`，因此所有局部 pipe event、跨 AIC/AIV flag 和全核
同步都由源码显式维护。同步协议不是性能注释，而是计算正确性的一部分。

## 7. 顶层辅助函数

### 7.1 `_clear_caches`

清空以下主机缓存：

- 已编译 kernel cache。
- 工作区 cache。
- full-valid 辅助输入 cache。
- tail 辅助输入 cache。
- V0 GM 零哨兵行 cache。

测试在需要隔离编译版本或释放缓存时调用它。

### 7.2 `ceil_div`

整数向上取整，供任务、tile、scatter batch 和 loss partial 对齐使用。

### 7.3 `_get_aicore_count`

从当前 NPU device properties 中读取 `cube_core_num`。如果设备不可用或属性无效，
直接抛出异常，不使用硬编码 24 核。

### 7.4 `SparseLightningIndexerGradKLLossTilingData`

保存从输入形状解析出的 `B/S1/S2/N1/N2/Nidx1/K` 等静态信息，并提供若干
派生属性。当前主执行路径主要读取其字段，kernel 内部再计算实际 tile 数量。

### 7.5 `check_param_valid`

完成入口契约检查：

- BSND 输入必须是四维，TND 输入必须是三维。
- Query、Key、QueryIndex、KeyIndex dtype 一致且为 FP16/BF16。
- Weights 可为输入类型或 FP32。
- `sparse_mode` 固定为 3。
- `N2=1`、`DQuery=512`、`DQueryIndex=128`。
- N1、Nidx1 和 K 位于支持集合。

检查完成后返回 tiling dataclass。

### 7.6 `_make_task_schedule`

launch core 数同时考虑主任务数量和 V2 工作量：

```text
scatterBatches  = ceil(KExec / 16)
finalizeBatches = ceil(B * S2Pad / 16)
v2Batches       = max(scatterBatches, finalizeBatches)
v2CoreNeed      = ceil(v2Batches / (2 AIV/core * 16 batches/AIV))

launchCoreNum = min(hardwareCoreNum,
                    max(taskCount, v2CoreNeed, 1))
```

这样小 `B*S1` 用例不会无条件启动所有核，同时又避免 V2 只落到极少数 AIV 上。

### 7.7 长度和辅助输入函数

| 函数 | 作用 |
| --- | --- |
| `_host_sequence_lengths` | 尝试把 list、tuple 或 CPU tensor 转为主机整数列表 |
| `_derive_execution_widths` | 推导 `KExec` 与 `KScatterExec` |
| `_is_static_full_valid` | 判断是否可编译 full-valid 静态版本 |
| `_get_full_valid_aux_inputs` | 缓存 full-valid 的合法占位输入 |
| `_get_tail_aux_inputs` | 缓存 `KRealSize` 和 `ValidKMask` |
| `_get_v0_zero_row` | 缓存 V0 无效索引使用的 GM 零哨兵行 |

### 7.8 Python 参考函数

`_preprocess_gather`、`_compute_vector_outputs` 和 `_compute_d_key_index` 是主机参考
或诊断实现，不是正常 kernel 的生产计算路径：

- `_preprocess_gather` 由 `slig_debug.py` 用来重建 host gather 对照。
- `_compute_vector_outputs` 记录 P、SoftmaxI、dWeight、ReluGrad 和 Loss 公式。
- `_compute_d_key_index` 记录 bmm + scatter_add 的参考语义。

## 8. 主机入口 `sparse_lightning_indexer_grad_kl_loss`

### 步骤 1：检查参数并处理布局

入口先调用 `check_param_valid`。如果 `layout_query="TND"`，转入 TND 包装函数；
否则继续 BSND 路径。

### 步骤 2：选择 full-valid 或 tail-protected

主机根据真实序列长度判断是否所有 query 都完整有效：

- full-valid：保持 `K=KTop`，使用静态无 mask kernel。
- tail-protected：推导 `KExec/KScatterExec`，生成或读取 KRealSize/ValidKMask。

如果真实长度在 CPU 可见，辅助张量会按长度组合缓存；如果只在 NPU 可见，则用
torch NPU 运算动态构造，但不为了求 `maxRealK` 把数据同步回主机。

### 步骤 3：补齐 S2 和 RoPE

```text
S2Pad = align_up(S2, 256)
DRopePad = align_up(DRope, 128)
```

没有 RoPE 时传入最后一维为 0 的张量。有 RoPE 且维度不足 128 对齐时，主机使用
0 补齐 QueryRope 和 KeyRope。

### 步骤 4：处理 dtype

主计算类型固定为 FP32，Cube 输入与大部分中间存储使用 Query 的 FP16/BF16。
Weights 为 FP32 时，kernel 前转换为输入类型，kernel 后再把 dWeights 转回 FP32。

### 步骤 5：生成调度与 cache key

主机读取硬件 core 数，调用 `_make_task_schedule`，然后把所有会改变生成代码的
静态参数加入 kernel cache key，包括：

- 形状、dtype、scale、core 数和 RoPE 宽度。
- `full_valid`、`KExec`、`KScatterExec`。
- 原始 SparseIndices stride。
- 三个诊断 specialization 开关。

cache miss 时调用 `tilelang_grad_kl_loss_single_kernel` 编译，并把生成的 AscendC
源码写入 `0721_kernel_source.cpp`。

### 步骤 6：分配或复用工作区

主机使用独立 workspace cache。关键工作区如下：

| 工作区 | dtype | Producer | Consumer |
| --- | --- | --- | --- |
| `GatherKey_ws` | 输入类型 | V0 | C1-P |
| `GatherKeyRope_ws` | 输入类型 | V0 | C1-P |
| `GatherKeyIndex_ws` | 输入类型 | V0 | C1-SY、C2-dQ |
| `bmm1Res` | FP32 | C1-P | V1-P |
| `bmm2Res` | FP32 | C1-SY | V1-SY、V1 backward |
| `PReduced` | FP32 | V1-P | V1 backward |
| `SoftmaxI` | FP32 | V1-SY | V1 backward |
| `reluGradRes` | 输入类型 | V1 backward | C2 |
| `dQueryIndexAccum` | FP32 | C2 | finalize |
| `dk_matmul_partial` | FP32 | C2 | V2 scatter |
| `dKeyIndexAccum` | FP32 | V2 | finalize |
| `LossPartial` | FP32 | V1 backward | finalize |
| `C2ZeroPad` | 输入类型 | 主机常量 | Nidx1=8 的 C2 |

工作区规模近似与 `B*S1*KExec` 成正比。大 B、大 S1 和大 K 的组合会同时放大
Gather、BMM1、BMM2、ReluGrad 和 dKey partial，是泛化测试中 OOM 的主要来源。

### 步骤 7：调用单 kernel

kernel ABI 共 30 个参数：

- 13 个输入和辅助输入。
- 13 个调用方工作区。
- 4 个输出。

装饰器使用 `out_idx=[-4,-3,-2,-1]` 返回最后四个张量，并保持
`workspace_idx=[]`，确保 kernel 读写的是调用方缓存的工作区，而不是 JIT 私有
分配的同名临时张量。

### 步骤 8：调试、reshape 和返回

需要调试时，主机把张量引用封装成 `DebugContext`，交给 `slig_debug.py`。
正常路径不会构造该上下文。

kernel 返回的 dKey 是按 `S2Pad` 展平的二维张量。主机 reshape 后裁剪回真实 S2：

```text
[B * S2Pad, 128]
  -> [B, S2Pad, 1, 128]
  -> [B, S2, 1, 128]
```

## 9. Kernel builder `tilelang_grad_kl_loss_single_kernel`

该函数接收静态形状和 specialization 参数，返回一个 `@T.prim_func main`。

### 9.1 固定 tile

当前核心 tile 为：

```text
N_SPLIT = 128       # K 方向主计算列块
K_SPLIT = 128       # MMA reduction 分块
S2_WORKSPACE = 1024 # V0/C1 外层 K 分段
GATHER_CHUNK_SIZE = 16
V2_SCATTER_BATCH = 16
```

V1 backward 进一步把 128 列拆成两个 64 列物理块，以适配已经验证正确的
row-expand 和 reduction lowering。

### 9.2 精确物理 buffer 策略

C1/C2 为每种 MMA 形状分配独立、精确的 L0A/L0B/L0C buffer，不使用一个大父
allocation 再切片成不同物理矩阵。原因是后端可能让 L1->L0 或 FIX 继承父
allocation 的物理 shape，产生错误 descriptor。

### 9.3 任务流水延迟

正常版本：

```text
TASK_PIPELINE_DEPTH = 2
taskPipelineLag = 1
```

每轮对当前 task 执行 V0，同时对上一 task 执行 C1/V1/C2。诊断环境变量
`SLIG_SERIAL_TASK_PIPELINE=1` 会把 lag 改为 0，使各阶段处理同一个 task，用于
排查跨 task event generation；它不是性能版本。

## 10. V0：`vector0_gather_task`

### 10.1 输入输出

输入为 SparseIndices、Key、KeyRope、KeyIndex 和统一零哨兵行；输出为三块 GM
Gather workspace。

### 10.2 AIV 分工

每个 1024 行外层 K 段由两个 AIV 平分：

```text
AIV0: 前 512 行
AIV1: 后 512 行
```

每个 AIV 再以 16 行为一个 gather chunk。

### 10.3 Gather 策略

每个 lane 读取一个运行时 SparseIndex：

- 非负索引：从 Key、KeyIndex、KeyRope 的对应 GM 行复制到 UB。
- 负索引：三路都从 `V0ZeroRow` 复制 0。

三路数据始终采用统一的 GM->UB 路径，避免同一 chunk 混用 UB->UB 清零和
GM->UB gather 时产生后端 ownership 歧义。

### 10.4 Ping-pong

正常路径分配两个 UB slot：

1. MTE2 将下一个 16 行 chunk 读入 `nxt`。
2. MTE3 将当前 `cur` 写入 Gather workspace。
3. `MTE3_MTE2` free flag 防止下轮覆盖仍在写出的 slot。

`SLIG_V0_SINGLE_SLOT=1` 会关闭重叠，逐 chunk 串行执行，只用于诊断。

### 10.5 发布给 C1

两个 AIV 完成各自 512 行后，使用 mode-2 cross flag 发布 readiness。AIC 对同一
flag 的 wait 会汇聚两个 AIV，确保 C1 不会看到只完成一半的 workspace。

V0 到 C1 的 flag slot 使用：

```text
flagSlot = (taskSlot + kOuter) % 2
```

因为一个 task 可能包含多个 1024 行外层段，必须同时区分 task generation 和
当前 kOuter。

## 11. C1：`cube1_task`

C1 在 AIC 上依次完成教师分数和索引器分数。

### 11.1 C1-P：TeacherScore

Query 的 512 维固定拆成四个 128 维 L1 cache：

```text
QueryChunk0 = Query[...,   0:128]
QueryChunk1 = Query[..., 128:256]
QueryChunk2 = Query[..., 256:384]
QueryChunk3 = Query[..., 384:512]
```

对每个 128 列 K tile：

1. 等待 V0-P readiness。
2. GatherKey 的一个 `128x128` 块从 GM 搬到 L1。
3. Query chunk 和 Key chunk 搬到 L0A/L0B。
4. 四次 MMA 在同一个 `N1x128` L0C 中累加。
5. 如果有 RoPE，再按 128 维块继续累加。
6. 最后由 FIX 写回 `bmm1Res[B,S1,N1,K]`。

正常路径在当前 MMA 消费 L1 时预取下一个 D 或 RoPE chunk。
`SLIG_C1_SERIAL=1` 禁止该预取，仅用于定位 C1 内部复用问题。

C1-P 完成后立即发布 `C1_TO_V1_P_FLAGS`，AIV0 可以开始计算 P，而 AIC 继续执行
C1-SY。

### 11.2 C1-SY：IndexerScore

`DQueryIndex=128`，因此 QueryIndex 通常只需一个 reduction chunk：

1. QueryIndex 缓存在 L1，并搬到 L0A。
2. 每个 GatherKeyIndex `128x128` tile 搬到 L0B。
3. MMA 得到 `Nidx1x128`。
4. FIX 写回 `bmm2Res` 时启用 ReLU。

C1-SY 完成后发布 `C1_TO_V1_SY_FLAGS`，AIV1 开始 SoftmaxI。

## 12. V1 前半：`vector1_psy_task`

这个宏让两个 AIV 并行处理同一 task 的两条独立分支：

```text
AIV0 -> PReduced
AIV1 -> SoftmaxI
```

### 12.1 AIV0：PReduced

对每个 128 列 K tile：

1. 只在 tile 与 `KRealSize` 相交时读取 bmm1Res。
2. 乘 `scaleValue`。
3. 部分尾块先用 uint8 Compare mask 把无效列 Select 为负无穷。
4. 按行减 SoftmaxMax。
5. 执行 exp。
6. 按行除以 `max(abs(SoftmaxSum), 1e-8)`。
7. 部分尾块再次乘 ValidKMask，明确得到 0。
8. 按 N1 方向 reduce，并除以 N1 得到 PReduced。

P 的行扩展最多按 `64x64` 子块执行，因为后端底层 repeat/stride 字段较窄；
直接处理更大逻辑 tile可能生成参数截断。

N1 方向 reduction 也按最多 64 行拆分。当前后端的 `64x128 -> 128` 快路径已验证
正确，而单次 `128x128` axis-0 reduction 会破坏结果，因此多个 partial 在 UB
中显式相加。

### 12.2 AIV1：SoftmaxI

SoftmaxI 分三遍处理整个 K：

第一遍，生成每个 K tile 的 `WeightedScore` 并求全局 max：

1. Weights 从输入类型 cast 到 FP32。
2. 广播为 `Nidx1x128`。
3. `bmm2Res * weights`。
4. 沿 Nidx1 reduce 得到 128 个候选 logit。
5. 部分尾块无效列 Select 为负无穷。
6. 保存每个 tile 的 logit，并更新全局 max。

第二遍，计算并缓存 `exp(logit-max)`，同时累加全局 sum：

- 部分尾块明确 Select 为 0。
- 完整无效块不执行 exp，直接填 0。

第三遍，用全局 sum 原地归一化并写回 SoftmaxI。

这种实现避免归一化阶段重复执行 `sub + exp`。

### 12.3 P/SY 会合：`relay_vector1_psy`

两个 AIV 完成后都向 AIC 发布同一 task slot 的 `V1_PSY_TO_AIC` flag。AIC 的
relay 宏：

1. 等待两个 AIV 都完成。
2. 向两个 AIV 发布 `V1_PSY_RELAY` flag。

两个 AIV 收到返回 flag 后才能进入 backward，保证 PReduced 和 SoftmaxI 都完整。

## 13. V1 后半：`vector1_backward_task`

### 13.1 AIV 分工

两个 AIV 平分 Nidx1：

```text
rowsPerAIV = Nidx1 / 2
AIV0: 前半索引头
AIV1: 后半索引头
```

每个 128 列主 tile 再拆成两个 64 列 backward tile。

### 13.2 权重广播

Weights 先从输入类型 cast 到 FP32，再广播到 `rowsPerAIV x 64`。

当 `Nidx1=8` 时，每个 AIV 只有 4 行，而 BRCB row expansion 要求行数为 8 的
倍数，因此代码逐行调用 broadcast；其他 Nidx1 使用 row-expand intrinsic。

### 13.3 dWeight 与 ReluGrad

每个 64 列块读取：

- PReduced。
- SoftmaxI。
- 当前 AIV 的 bmm2Res 行。
- tail 版本的 ValidKMask。

随后计算：

```text
Diff = SoftmaxI - PReduced
dWeight += reduce_k(bmm2Res * Diff)
Gate = 1 - exp(-10000 * bmm2Res)
ReluGrad = Weight * Diff * Gate
```

ReluGrad 在 UB 中以 FP32 计算，使用 `CAST_RINT` 转成输入类型后写入 GM workspace。

这一段存在关键 WAR ownership：下一轮 Vector 要覆写的 FP32 buffer 与上一轮
MTE3 正在写出的 FP16 ReluGrad 可能复用同一物理 UB，因此覆写前必须等待
`MTE3_V` free flag。删除这个等待会出现非确定性 NaN 或大误差。

### 13.4 Loss partial

两个 AIV 按 `n_idx % 2` 分担 K 外层块。每个所属 64 列块计算：

```text
P * (log(max(P, 1e-8)) - log(max(SoftmaxI, 1e-8)))
```

tail 版本再乘 ValidKMask。每个 `(task, vectorId)` 产生一个 FP32 loss partial。

每个 partial 在 GM 中占用 8 个 FP32 元素，即 32 字节对齐槽位；只有第一个元素
保存值。这样避免短标量 MTE3 写回覆盖相邻 partial。

### 13.5 发布给 C2

ReluGrad、dWeight 和 LossPartial 写回并 drain 后，两个 AIV发布同一 task slot 的
`V1_TO_C2` mode-2 flag。C2 只有在两半索引头都完成后才开始。

## 14. C2：`cube2_task`

C2 在同一个 K tile 上执行两种 MMA。

### 14.1 dKey partial

```text
ReluGrad[:, kTile]^T @ QueryIndex
```

形状为：

```text
[128, Nidx1] @ [Nidx1, 128] -> [128, 128]
```

每个 K tile 独立写入 `dk_matmul_partial`，暂时不做 sparse scatter。

当 `Nidx1=8` 时，BF16 Cube reduction 维度至少为 16。代码把 QueryIndex 和
ReluGrad 的物理尾部 8 行补 0，再执行 `K=16` 的 MMA，逻辑结果仍只来自前 8 行。

### 14.2 dQueryIndex

```text
ReluGrad @ GatherKeyIndex
```

每个 K tile 产生一个 `Nidx1x128` partial，所有 K tile 在同一个 L0C 中累加，
最后写入 FP32 `dQueryIndexAccum`。

### 14.3 双 MMA 资源

dKey 和 dQuery 使用各自精确的 L0A/L0B/L0C allocation及独立本地 flag，避免
不同物理 shape 互相污染 descriptor。

## 15. V2：清零、scatter 与 finalize

C2 完成后，主流水执行第一次 `T.sync_all()`。从这里开始，不再按 task owner
执行，而是让所有 launch core 的两个 AIV 共同处理全局输出。

### 15.1 `clear_dkey_accumulator`

`dKeyIndexAccum` 是 FP32 atomic add 目标，主机使用 `torch.empty` 分配，因此必须
在 kernel 内清零。

所有 AIV 以 32 行为一个批次均匀分担 `[B*S2Pad, 128]`，每个 AIV 在 UB 中生成
一个零块并写回自己的连续区间。

清零后执行第二次 `T.sync_all()`，确保任何 atomic add 开始前整个 accumulator
都为 0。

### 15.2 `vector2_scatter_all_tasks`

每个物理 core 先获得一段连续 K scatter batch，再把这段在两个 AIV 间拆分。
每个 AIV 对所有 task 遍历同一段 K：

1. 每个 task 只读取一次 KRealSize。
2. 以 16 行为一批，从 `dk_matmul_partial` GM 搬到 ping/pong UB。
3. 每次处理两行，读取两个 SparseIndex。
4. 同时满足 `k < KRealSize` 和 `SparseIndex >= 0` 时执行 atomic add。

目标地址为：

```text
dKeyIndexAccum[batch * S2Pad + SparseIndex]
```

相同目标行可被不同 query、不同候选槽位和不同 AIV并发累加。

scatter 完成后执行第三次 `T.sync_all()`，保证 finalize 读取的是最终 FP32 累加值。

### 15.3 `finalize_vector_outputs`

该宏完成三件事。

第一，dQueryIndex cast：

- 仍由原 task owner core 负责。
- 两个 AIV 交错处理 Nidx1 头。
- FP32 accumulator 使用 `CAST_RINT` 转为输入类型。

第二，dKeyIndex cast：

- 所有 AIV 按 16 行批次分担 `[B*S2Pad,128]`。
- 使用双槽 UB 完成 FP32->输入类型转换和 GM 写回。

第三，Loss 汇总：

- 仅 `core0/AIV0` 执行。
- 每次读取 128 个对齐 partial 元素。
- reduce 后累加为单个 FP32 LossOut。

## 16. `main` 的任务调度

正常 task 流水可概括为：

```text
pipeIter 0: V0(task 0)
pipeIter 1: V0(task 1) + C1/V1/C2(task 0)
pipeIter 2: V0(task 2) + C1/V1/C2(task 1)
...
epilogue :             C1/V1/C2(last task)
```

源码中宏调用看起来是顺序的，但每个宏通过 `T.Scope("C")` 或 `T.Scope("V")`
落到不同执行单元，并由 cross flag约束真实依赖。可重叠的阶段包括：

- 当前 task 的 V0 与上一 task 的 C1/V1/C2。
- C1-SY 与已经收到 P-ready 的 AIV0 P 分支。
- AIV0 的 P 与 AIV1 的 SY。
- C2 内 dKey MMA 与 dQuery MMA 的部分 pipe 活动。

无法跨越的全局边界是：

```text
C2 全部完成
  -> SyncAll
dKey accumulator 全部清零
  -> SyncAll
atomic scatter 全部完成
  -> SyncAll
输出 cast 和 loss finalize
```

三个 `SyncAll` 分别保护真实的全局 RAW 依赖，不能当作普通冗余 barrier 删除。

## 17. 跨 AIC/AIV flag 协议

| Flag | Producer | Consumer | 含义 |
| --- | --- | --- | --- |
| `0/1` | V0 两个 AIV | C1-P | GatherKey/GatherRope 当前 generation 就绪 |
| `2/3` | V0 两个 AIV | C1-SY | GatherKeyIndex 当前 generation 就绪 |
| `4/5` | C1-P FIX | AIV0 | bmm1Res 当前 task 就绪 |
| `6/7` | C1-SY FIX | AIV1 | bmm2Res 当前 task 就绪 |
| `8/9` | V1 backward 两个 AIV | C2 | ReluGrad 两个半区都就绪 |
| `10/11` | V1 P/SY 两个 AIV | AIC relay | PReduced 与 SoftmaxI 都完成 |
| `12/13` | AIC relay | V1 两个 AIV | 允许进入 backward |

大部分 flag 成对出现，分别对应 `taskSlot=0/1`。task slot 是循环 generation 的
一部分，不能只按 core 或固定 event ID 判断数据属于哪个 task。

核内还使用 MTE2、MTE3、MTE1、M、FIX 和 Vector 之间的本地 flag：

- `MTE3_MTE2`：UB 写回结束后，允许下一次 GM->UB 覆写。
- `MTE2_MTE1`：GM->L1 完成，允许 L1->L0。
- `M_MTE1` / `MTE1_M`：MMA 与 L0 load 的 ownership 交换。
- `FIX_M` / `M_FIX`：L0C 在 MMA 和 FIX 之间的 ownership 交换。
- `V_MTE3` / `MTE3_V`：Vector 结果与 UB->GM 写回之间的 ownership 交换。

每个宏末尾都要 drain 自己初始化的本地 event，避免同一 event ID 在下一 task
或下一 phase 被旧 token 污染。

## 18. 精度实现策略

### 18.1 FP32 累加

以下数据使用 FP32：

- bmm1Res、bmm2Res。
- PReduced、SoftmaxI。
- dQueryIndexAccum、dKeyIndexAccum、dKey partial。
- dWeight 的 UB 累加。
- Loss partial 和最终 Loss。

ReluGrad 为节省工作区并匹配 Cube 输入，在 V1 末尾转回输入类型；C2 再用
FP16/BF16 MMA 计算梯度。

### 18.2 softmax 尾块

无效列必须在 exp 前成为负无穷，而不是先 exp 再乘 0。否则上游异常大值可能形成：

```text
exp(large) = inf
inf * 0 = NaN
```

因此 P 和 SY 的部分尾块都使用 uint8 Compare mask 与 Select，完整无效块直接
跳过昂贵计算并填 0。

### 18.3 reduction 分块

逻辑 shape 不等于后端可靠支持的物理 shape。当前实现主动把：

- P row-expand 拆为最多 `64x64`。
- P axis-0 reduction 拆为最多 `64x128`。
- backward 拆为 `rowsPerAIV x 64`。

这些拆分保持已经验证的 lowering 参数范围，避免 repeat 或 stride 窄字段溢出。

### 18.4 Nidx1=8 的 Cube 补零

逻辑 reduction 维度 8 不满足 BF16 Cube 的 16 对齐要求。C2 使用物理 16 行，
后 8 行写 0，不改变逻辑公式。

### 18.5 dKey FP32 atomic accumulator

dKey 可能有大量重复 SparseIndex。直接以输入类型 atomic 累加会扩大舍入误差；
当前实现先在 FP32 accumulator 中完成全部 atomic add，最后统一 cast。

## 19. 性能实现策略

当前已经采用的主要性能策略包括：

1. Gather 留在 kernel 的 V0，不执行每次调用的主机 gather。
2. V0 使用 16 行 UB ping-pong，并让两个 AIV 平分每个 1024 行 K 段。
3. Query 的四个 128 维 chunk 与 QueryRope 在 C1 中驻留 L1。
4. C1-P 正常路径预取下一个 Key/KeyRope reduction chunk。
5. AIV0 P 与 AIV1 SY 并行，并允许 P 与 C1-SY 重叠。
6. SoftmaxI 缓存 exp 结果，归一化阶段不重复 exp。
7. full-valid 静态版本删除所有 tail mask 指令。
8. tail 版本按主机可见 maxRealK 收缩 KExec 和 KScatterExec。
9. V2 使用所有 AIV、16 行批次和 UB ping-pong。
10. kernel、workspace、尾块元数据和零哨兵都跨调用缓存。

当前最难进一步优化的部分是：

- C1 无法稳定表达与官方 Matmul service 等价的完整 L0 ping-pong。
- V2 对重复 SparseIndex 的 atomic scatter 可能产生竞争。
- 三个全局阶段边界具有真实数据依赖。
- 若干 Vector intrinsic 需要为后端 repeat/stride 限制拆成较小物理 tile。

## 20. TND 入口

`sparse_lightning_indexer_grad_kl_loss_tnd` 不实现另一套 kernel。它把 TND 输入
临时增加 batch 维：

```text
[T, N, D] -> [1, T, N, D]
```

然后调用同一个 BSND 主入口。返回时再 squeeze batch 维。

真实 Query/Key 长度来自显式参数；未提供时分别使用 Query 和 Key 的第 0 维。
因此 TND 与 BSND 共享全部 tail、kernel cache、workspace 和五阶段实现。

## 21. 调试与诊断

主机诊断已集中在 `slig_debug.py`，避免主 kernel 文件被大量只在 debug 时执行的
代码淹没。

| 环境变量 | 作用 | 是否可用于性能测试 |
| --- | --- | --- |
| `SLIG_DEBUG=1` | 打印全局 workspace 与 host 公式对比 | 否 |
| `SLIG_DEBUG_TASK=b,q` | 只诊断一个 batch/query | 否 |
| `SLIG_DEBUG_RELU_COORD=h,k` | 在 task 诊断中打印指定 ReluGrad 坐标 | 否 |
| `SLIG_PROFILE=1` | 打印主机 wrapper 事件区间 | 否 |
| `SLIG_SERIAL_TASK_PIPELINE=1` | 关闭跨 task lag | 否，诊断 specialization |
| `SLIG_C1_SERIAL=1` | 关闭 C1-P L1 预取 | 否，诊断 specialization |
| `SLIG_V0_SINGLE_SLOT=1` | 关闭 V0 ping-pong | 否，诊断 specialization |

`SLIG_PROFILE` 当前标签主要测量主机准备、整个 kernel 调用和返回后的 reshape
区间，不能把 `kernel(incl gather)` 再解释成 V0/C1/V1/C2/V2 的内部拆分。要分析
各阶段真实性能，需要设备 timeline。

`slig_debug.py` 的两条主路径是：

- `_run_task_diagnostics`：只抓一个 query，逐阶段比较 Gather、BMM1、BMM2、
  P/SY、ReluGrad 和 C2。
- `_run_full_diagnostics`：检查完整工作区、非有限值、最大误差和 loss partial。

## 22. 一个 task 的完整生命周期

以 `(b, q)` 为例：

1. 主机确定该 query 的 `KRealSize`，选择已经缓存的 kernel variant。
2. V0 的两个 AIV 各 gather 一半 Key/KeyRope/KeyIndex。
3. C1-P 等待完整 GatherKey，计算 N1 个教师 score 行。
4. C1-P 发布结果后，AIV0 开始计算 PReduced。
5. C1-SY 计算 Nidx1 个索引器 score 行，并在 FIX 时 ReLU。
6. C1-SY 发布结果后，AIV1 开始计算 SoftmaxI。
7. AIC relay 等待 P 和 SY 都完成，再释放两个 AIV。
8. 两个 AIV 各处理一半索引头，计算 dWeight、ReluGrad 和 LossPartial。
9. C2 等待两半 ReluGrad 完成。
10. C2 计算本 task 的 dQueryIndexAccum 和每个候选槽的 dKey partial。
11. 所有 task 的 C2 完成后，全核清零 dKey accumulator。
12. 所有 AIV 遍历自己的 K 区间，把该 task 的 dKey partial scatter 到原始 Key 行。
13. 所有 task scatter 完成后，输出 cast 并汇总 loss。
14. 主机把 padded dKey 裁剪到真实 S2，恢复 dWeight dtype，然后返回。

## 23. 维护代码时的边界

修改 0721 时应遵守以下原则：

1. 数学公式、tile shape、event 协议和 task 调度一次只改一类变量。
2. 修改 cross flag 时，同时审计 producer、consumer、task slot、generation 和 drain。
3. 修改 UB/L1/L0 allocation 时，以生成 C++ 的物理 offset、shape 和 descriptor
   为准，不能只看 Python 逻辑 shape。
4. 修改 tail 时同时检查 P、SY、Loss 和 V2 scatter，不能只保护某一阶段。
5. 修改 KExec 时保持 SparseIndices 输入 stride 与执行宽度分离。
6. 修改 dKey 流程时保留 clear -> scatter -> finalize 三个真实全局依赖。
7. 修改 backward 时保留 ReluGrad store 与下一轮 UB 覆写之间的 WAR wait。
8. 修改 full-valid specialization 后检查生成代码中是否真的删除 mask DMA 和分支。
9. 正式性能测试必须关闭所有 `SLIG_DEBUG/PROFILE/SERIAL` 开关。
10. 每次结构变更都先跑代表精度集，再跑全量泛化和重复性能中位数。

## 24. 函数导航表

| 函数 | 执行位置 | 角色 |
| --- | --- | --- |
| `check_param_valid` | 主机 | 输入契约与 tiling 解析 |
| `_make_task_schedule` | 主机 | 自适应 launch core 数 |
| `_derive_execution_widths` | 主机 | KExec/KScatterExec 推导 |
| `_is_static_full_valid` | 主机 | full-valid specialization 判定 |
| `sparse_lightning_indexer_grad_kl_loss` | 主机 | BSND 总入口、缓存、工作区、调用 |
| `sparse_lightning_indexer_grad_kl_loss_tnd` | 主机 | TND 到 BSND 的包装 |
| `tilelang_grad_kl_loss_single_kernel` | 编译期 | 构造静态 TileLang kernel |
| `vector0_gather_task` | AIV | 稀疏 gather |
| `cube1_task` | AIC | TeacherScore 与 IndexerScore |
| `vector1_psy_task` | AIV0/AIV1 | PReduced 与 SoftmaxI |
| `relay_vector1_psy` | AIC | 两个 AIV 的 P/SY 会合 |
| `vector1_backward_task` | AIV0/AIV1 | dWeight、ReluGrad、LossPartial |
| `cube2_task` | AIC | dQueryAccum 与 dKey partial |
| `clear_dkey_accumulator` | 全部 AIV | 清零 FP32 atomic 目标 |
| `vector2_scatter_all_tasks` | 全部 AIV | sparse atomic scatter |
| `finalize_vector_outputs` | 全部 AIV | cast 输出与 loss 汇总 |
| `main` | AIC+AIV | task pipeline 与全局阶段边界 |
| `parse_debug_options` | 主机 | 解析诊断环境变量 |
| `run_debug_diagnostics` | 主机 | 调度单 task 或全量诊断 |

## 25. 总结

0721 的核心设计可以概括为：

```text
主机静态 specialization
  + 单 kernel 混合 AIC/AIV
  + V0/C1/V1/C2 的 task 级流水
  + V2 的全核 FP32 deterministic scatter
  + full-valid 快路径与 tail-protected 路径
  + 显式手工 event ownership
```

它不是简单地把 PyTorch 公式逐句翻译成 TileLang，而是在后端可稳定支持的物理
tile、局部内存布局和同步能力范围内，重建官方 AscendC 算子的计算与梯度语义。
理解和维护它时，应始终同时观察数学公式、工作区数据流、生成 C++ 的物理布局
以及 event generation，任何一层单独正确都不足以保证最终精度和性能。
