# SASG K_OUT Kernel 主流程解释

## 1. 文档范围

本文解释：

```text
attention/generic_block_sparse_attention_grad/op_kernel/arc22/
    generic_block_sparse_attention_grad_k_out_kernel.h
```

说明以当前源码为准，覆盖模板参数、输入参数、任务状态、Packet 构造、TND/BNSD 地址计算、AIC/AIV 协作、Workspace 布局以及每个函数的流程和变量用途。文件中仍保留了少量早期 Q_OUT 路径的辅助函数，本文会明确标出其是否被 K_OUT 主入口调用。

## 2. 算子计算语义

对一个 Q 行块和一个稀疏 K 行块，Kernel 完成 Attention 反向传播的三组矩阵计算。设：

```text
S  = Q * K^T
P  = softmax(S, LSE)
dP = dOut * V^T
D  = sum(dOut * Out)
dS = P * (dP - D)
dQ = dS * K
dK = dS^T * Q
dV = P^T * dOut
```

其中 `S/dP` 以 FP32 保存，`P/dS` 按输入低精度类型保存；最终 dQ、dK、dV 在 FP32 Workspace 中累加，再由 Vector 后处理写回用户输出。

K_OUT 的调度顺序是：

```text
K Task（batch、KV head、K 序列块）
  -> Q head（一个 KV head 对应 groupSize 个 Q head）
    -> 稀疏索引扫描
      -> Q Packet（最多 128 行）
        -> Cube1: QK^T 和 dOut*V^T
        -> Vector: SoftmaxGrad，生成 P 和 dS
        -> Cube2/Cube3: dQ、dV、dK
        -> Vector: 离散 dQ Restore
```

## 3. 编译期结构和平台宏

`GenericBlockSparseAttentionGradKOutKernel` 是模板类，模板参数为三类 Cube MMAD 服务、四类 Vector/Epilogue 服务和 `INPUT_LAYOUT`。类内部通过 `using` 暴露服务中的 Element、Layout、ArchTag 和参数类型。

主函数通过编译宏隔离硬件专属代码：

| 宏 | 执行内容 |
|---|---|
| `__DAV_C220_CUBE__` | K/V 全局 Tensor、三个 BlockMmad、Layout、Cube 事件和 Cube 计算 |
| `__DAV_C220_VEC__` | Pre、D 预计算、Q/dOut Pack、SoftmaxGrad、dQ Restore、Post |
| 无宏区域 | 参数初始化、任务遍历、Packet 构造、布局选择和两侧都需要的索引计算 |

这样 AIC 和 AIV 使用同一套 K->Q->Packet 控制流，但只有对应核编译自己的计算代码；二者通过跨核事件传递中间结果。

### 3.1 Packet 内聚算子

Packet 逻辑位于 `attn_infra/epilogue/block/block_epilogue_qpacket.hpp`，由两个对象分别管理 Gather 和 Scatter 的资源与 LocalTensor：

| 算子 | 职责 |
|---|---|
| `GBSAGGater<InputType>` | 保存稀疏索引参数和跨 Packet cursor；`Aggregate()` 生成最多 128 行的 `QPacket`，`Gather()` 由两个 AIV 分片聚合 Q/dOut。 |
| `GBSAGScateer<InputType>` | 两个 AIV 分片恢复 dQ；`Scatter(..., enableAtomicAdd)` 由调用方选择普通写或原子累加。 |

`QPacket` 和 `QPacketCursor` 同样定义在该头文件，主 Kernel 只构造参数、调用算子并消费输出 metadata。

## 4. 输入、输出与 Workspace

### 4.1 `Params` 参数

| 字段 | 类型 | 用途 |
|---|---|---|
| `dout` | `GM_ADDR` | Attention 输出梯度，Cube3 计算 dV，Vector 计算 D |
| `q` | `GM_ADDR` | Query，Cube1 计算 QK^T，Cube3 计算 dK |
| `k` | `GM_ADDR` | Key，Cube1 作为右矩阵，Cube2 计算 dQ |
| `v` | `GM_ADDR` | Value，Cube1 作为右矩阵计算 dP |
| `out` | `GM_ADDR` | 前向 Attention 输出，Vector 用于 D 计算 |
| `softmaxLse` | `GM_ADDR` | 每个 Q 行的 log-sum-exp，SoftmaxGrad 使用 |
| `rsvdBlockIdx` | `GM_ADDR` | 稀疏块索引，K_OUT 中表示当前 `(batch,Q head,K block)` 关联的 Q block |
| `rsvdBlockCount` | `GM_ADDR` | 每个稀疏索引行的有效 Q block 数 |
| `blockShape` | `GM_ADDR` | Block 形状输入，当前 K_OUT 地址计算使用 tiling 中的 X/Y 值 |
| `attentionMask` | `GM_ADDR` | 接口保留字段，当前主流程不直接读取 |
| `actualQseqlen` | `GM_ADDR` | TND 下每个 batch 的实际 Q 长度 |
| `actualKvseqlen` | `GM_ADDR` | TND 下每个 batch 的实际 KV 长度 |
| `dq/dk/dv` | `GM_ADDR` | 最终输出梯度，Vector Post 从 FP32 Workspace 写回 |
| `workspace` | `GM_ADDR` | S、dP、dQ/dK/dV 中间结果及离散 Packet 缓冲区 |
| `tiling` | `GM_ADDR` | `GenericBlockSparseAttentionGradTilingData`，Host 侧生成的契约数据 |

两个构造函数只负责按参数顺序保存地址，不做设备侧计算。

### 4.2 `TaskInfo` 字段

| 字段 | 含义 |
|---|---|
| `curBatchIdx` | 当前 batch |
| `curHeadIdx` | 当前 Q head（部分旧 Q_OUT 辅助函数使用） |
| `curQSeqIdx` | 当前 Q 起始行 |
| `curQBlcokIdx` | 当前 Q block 索引，源码拼写保持不变 |
| `curCalQSize` | 当前计算的有效 Q 行数 |
| `curCalKVSize` | 当前计算的有效 K/V 行数 |
| `qSeqlen/kvSeqlen` | 当前 batch 的实际或最大序列长度 |
| `qOffset` | Q、dOut、dQ 的元素起始偏移 |
| `kvOffset` | K、V、dK、dV 的元素起始偏移 |
| `sOffset` | 当前双缓冲槽在 Workspace 中的 S 起始偏移 |
| `curKvHeadIdx` | 当前 KV head |
| `curKBlockIdx` | 当前 K block 索引 |
| `curKSeqIdx` | 当前 K 序列起始行 |
| `qBatchBaseOffset` | 当前 batch 的 Q 基地址 |
| `kvBatchBaseOffset` | 当前 batch 的 KV 基地址 |

`taskInfo[2]` 是双槽任务状态：当前任务使用 `i % 2`，下一个任务提前写入另一个槽。

### 4.3 常量、事件和中间区

| 名称 | 含义 |
|---|---|
| `AGGREGATE_M = 128` | 一个 Q Packet 最多聚合 128 行 |
| `MAX_PACKET_SEGMENTS = 128` | 一个 Packet 最多记录 128 个离散 segment |
| `CUBE2VEC` | Cube1 完成 S/dP 后通知 Vector |
| `VEC2CUBE` | Vector 完成 P/dS 后通知 Cube2/Cube3 |
| `CUBE2POST` | Cube 主循环完成后通知 Vector Post |
| `WORKSPACE_BLOCK_SIZE_DB` | 每个逻辑 AIC core 的 S/dP 双缓冲区大小 |
| `WORKSPACE_P16_OFFSET_ELEMENT` | S/dP FP32 区对应的低精度 P/dS 区元素偏移 |
| `L1_SIZE_OFFSET` | Cube L1 中不同矩阵区域的字节偏移 |
| `PINGPONG_OFFSET_2/4` | BlockMmad 内部双缓冲/四缓冲偏移参数 |

Workspace 的核心布局为：

```text
[S FP32][P low precision]
[dP FP32][dS low precision]
[dQ FP32][dK FP32][dV FP32]
[D workspace]
[每个逻辑 core 的 qPack/doutPack/dqPack 双缓冲]
```

`InitCommonParams` 按 `sOutSize/dPOutSize/dQOutSize/dKOutSize/dVOutSize/gradSize` 计算各区起始地址。

## 5. 任务状态和地址函数

### 5.1 `SetFlag()` / `WaitFlag()`

这两个函数成对管理 Cube 侧 MTE1、FIX_M、MTE1_MTE2 事件。入口中在构造 BlockMmad 后调用 `SetFlag()`，主循环结束调用 `WaitFlag()`，确保内部 L1/MTE 事务全部完成后再通知 Post。它们不是跨核事件；跨 AIC/AIV 的事件使用 `CrossCoreSetFlag/WaitEvent`。

### 5.2 `UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, taskInfo)`

计算 Q_OUT 风格任务的有效 Q 行数，并设置 `curQBlcokIdx`：

1. 用 `curQSeqIdx / blockShapeX` 得到 Q block。
2. 计算当前 Q block 的结束位置。
3. 若 basic Q tile 不越过稀疏块边界和 batch 边界，使用 `basicQBlockSize`。
4. 若会越界，截断到 Q block 或当前 batch 的末尾。

K_OUT 主流程不直接调用它；K_OUT 的 Q 行范围由 `GBSAGGater::Aggregate()` 产生。它仍被旧的 `initTaskInfo/updateNextTaskInfo` 使用。

### 5.3 `initTaskInfo(...)`

旧 Q_OUT 路径的任务初始化函数。它从 `beginBatch/beginHead/beginQSeqOffset` 和 `preQSeqLengths/preKVSeqLengths` 读取起点，按 TND 或 BNSD 计算 Q/KV 偏移，再调用 `UpdateTaskInfoCalQSize`。当前 K_OUT 的 `operator()` 使用 `InitKTaskInfo`，因此该函数不是 K_OUT 主路径。

### 5.4 `updateNextTaskInfo(...)`

旧 Q_OUT 路径的任务推进函数：TND 优先遍历 head，BNSD 优先遍历 S 轴；序列完成后切换 head 或 batch，并重新计算 Q block 大小。当前 K_OUT 使用 `UpdateNextKTaskInfo`，该函数属于遗留兼容代码。

### 5.5 `UpdateKTaskCalSize(blockShapeY, basicKVBlockSize, taskInfo)`

计算当前 K Task 的有效 KV 行数：

```text
curKBlockIdx = curKSeqIdx / blockShapeY
sparseEnd    = (curKBlockIdx + 1) * blockShapeY
taskEnd      = curKSeqIdx + basicKVBlockSize
validEnd     = min(sparseEnd, kvSeqlen)
curCalKVSize = min(taskEnd, validEnd) - curKSeqIdx
```

它同时处理普通 KV tile、K block 尾部和实际序列尾部。

### 5.6 `InitKTaskInfo(...)`

初始化 K_OUT 的第一个 K Task：从 `kOutBeginBatch/kOutBeginKvHead/kOutBeginKvSeqOffset` 读取起点；读取当前 batch 的长度；根据 TND/BNSD 计算 batch 基地址和 `kvOffset`；最后调用 `UpdateKTaskCalSize`。

### 5.7 `UpdateNextKTaskInfo(...)`

沿 K_OUT 任务顺序推进。TND 先遍历 KV head 再递增 K 序列，BNSD 先递增 K 序列再切换 KV head；当前 batch 的所有 KV head/K tile 完成后切换 batch，并重新计算 `kvOffset`、`curCalKVSize`。

### 5.8 `GBSAGGater::GetQBlockSize(qBlock)`

返回 Q block 在实际 Q 序列中的有效行数：`min(blockShapeX, qSeqlen - qBlock*blockShapeX)`；Q block 起点越界时返回 0。

## 6. Q Packet 构造与离散搬运

### 6.1 `QPacket` 和 `QPacketCursor`

`QPacket` 保存 `rows`、`segmentCount` 以及每个 segment 的原始起点 `qStart`、连续 Packet 起点 `packStart`、行数 `segmentRows`。`QPacketCursor` 的 `slot` 指向稀疏索引槽，`rowInBlock` 指向当前 Q block 尚未消费的行。

### 6.2 `GBSAGGater::Aggregate()`

按 `rsvdBlockIdx` 顺序构造下一个 `M <= 128` Packet：清零状态，跳过负值/越界索引和空尾块，取当前 block 的有效行数，按剩余 Packet 容量截取行，写入原始到连续布局的映射；block 未消费完时保留 `rowInBlock`，消费完才递增 `slot`。调用次数由 `GetPacketCount()` 驱动。

### 6.3 `GetQPacketRowOffset(...)`

按 TND/BNSD 计算指定 Q head、Q 行的 GM 元素偏移，供 Gather 和 Scatter 逐行访问。

### 6.4 `GBSAGGater::Gather(GatherParams)`

计算当前 AIV 的 Packet 行范围，遍历与该范围相交的 segment，用 `GetQPacketRowOffset` 从原始 Q/dOut 读取整行到类内 UB，再写到 Packet GM 的不相交区间。Q 和 dOut 顺序复用同一个 LocalTensor。

### 6.5 `SimpltSoftmax::LseCopy/CopyDIn(...)`

离散 LSE/D 搬运位于 `block_epilogue_simply_softmax.hpp`。主 Kernel 只把 `QPacket` metadata 和原始 GM 地址传入 `SfmParams`；`LseCopy` 与 `CopyDIn` 分别遍历当前 compute 行区间和 Packet segment 的交集，按原始 Q 行读取 LSE/D。TND 按 `numHeads` 跨距搬运，BNSD 按当前 head 的连续行搬运。

目的行使用 `localRowStart = copyBegin - curS1Idx`，因此两个 AIV 都从各自 2048 B 工作缓冲的第 0 行开始写入。packed LSE 复用 `lseBrocTensor`，packed D 复用 `softmaxGradTensor`；不再分配 `lsePacketLocal/dPacketLocal`，也不再执行循环前的 `PackDiscreteSoftmaxInputs` 和 `PIPE_ALL`。每个 AIV 对 `[vecBegin,vecEnd)` 仍只调用一次 SoftmaxGrad。

### 6.6 `GBSAGScateer::Scatter(Params, QPacket, enableAtomicAdd)`

两个 AIV 各自将所属的连续 `gDqPack` 行读回类内 UB，并按 Packet segment 交集恢复到原始 dQ Workspace。K_OUT 传 `enableAtomicAdd=true`，因此写入前开启 FP32 AtomicAdd，结束后恢复 `SetAtomicNone()`；传 `false` 时执行普通写。

`GBSAGGater` 在自身内部保存 `QPacketCursor` 和当前 `QPacket`，不再需要主 Kernel 的 `CopyQPacket` 辅助函数。

## 7. 初始化、主入口和 Vector 阶段

### 7.1 构造函数

`GenericBlockSparseAttentionGradKOutKernel()` 为空构造函数，实际资源由成员默认构造和 `InitCommonParams` 完成。

### 7.2 `InitCommonParams(params)`

解析 tiling，计算逻辑 core，读取形状/布局/Workspace 字段，校验关键除数非零，绑定 Q、dOut、稀疏索引、实际长度和 dQ Tensor，计算 Packet 双缓冲地址，初始化第一个 K Task、mask block 数和 S 双缓冲偏移。失败时返回 false，主入口直接退出。

### 7.3 `operator()(params)`

这是唯一的 K_OUT 统一入口。公共部分负责初始化、全核同步、K/Q/Packet 循环和任务推进；`__DAV_C220_CUBE__` 区块执行 Cube1/2/3，`__DAV_C220_VEC__` 区块执行 Pre、D、Pack、SoftmaxGrad、Restore、Post。

### 7.4 主循环伪代码

```text
InitCommonParams
VecPre -> SyncAll -> VecPrepareD -> SyncAll
初始化 AIC BlockMmad 和 AIV Softmax/Pack 缓冲
for K task:
  for Q head in 当前 KV head 对应的 groupSize 个 head:
    读取 sparseCount 和 idxBase
    cursor = {0, 0}
    packetCount = ceil((sparseCount * blockShapeX - tailCorrection) / 128)
    # tailCorrection 仅在有序 idx 的最后一项命中尾 Q block 时生效
    for packetIdx in [0, packetCount):
      GBSAGGater::Aggregate() 生成当前 Packet
      两个 AIV 调用 GBSAGGater::Gather() 聚合 Q/dOut -> CUBE2VEC
      Cube1 -> CUBE2VEC
      Vector SoftmaxGrad -> VEC2CUBE
      dQPack + dV + dK，再由两个 AIV 调用 GBSAGScateer::Scatter(..., true) 恢复 dQ
      pingpong 翻转
  UpdateNextKTaskInfo
CUBE2POST -> WaitFlag -> VecPost
```

### 7.5 `VecPrepareD(params)`

调用 `EpilogueFAGSfmg`，对每个 `(batch,qHead)` 计算 `D=sum(dOut*out)`。TND 通过 batch 前缀实际长度定位，BNSD 通过最大长度定位；AIV 以 `usedVecCoreNum` 为步长分配 group。

### 7.6 `VecPost(params)`

构造 `EpilogueFAGPost`，将 Workspace 中 FP32 的 dQ/dK/dV 按实际序列长度、布局和 tiling 写回最终输出。

### 7.7 `VecPre(params)`

构造 `EpilogueFAGPre`，初始化 dQ/dK/dV Workspace，使后续 Cube Atomic 累加有确定的零值起点。

## 8. 主循环局部变量说明

| 变量 | 所在阶段 | 用途 |
|---|---|---|
| `i` | K Task 循环 | 当前逻辑 core 的任务序号 |
| `curInfo` | K Task/Packet | 当前 K Task 的工作副本 |
| `qHeadBegin/qHeadEnd` | Q Head 循环 | 当前 KV head 映射出的 Q head 范围 |
| `qBlockNum` | Q Head 循环 | 当前 batch 的 Q block 总数 |
| `rowOffset` | 稀疏索引读取 | `(batch,qHead,kBlock)` 在 count Tensor 中的行号 |
| `sparseCount` | 稀疏索引读取 | 当前稀疏行的有效 Q block 数，读取后会裁剪到合法范围 |
| `idxBase` | 稀疏索引读取 | 当前稀疏行在 idx Tensor 中的起始元素 |
| `cursor` | Packet 构造 | 跨 Packet 保存的索引槽和 block 内行位置 |
| `pingpongFlag` | Workspace | 选择 0/1 Packet 双缓冲槽 |
| `layoutA1/B1/C1` | Cube1 | QK^T、dOutV^T 的矩阵布局和 stride |
| `layoutA2/B2/C2` | Cube2 | dS*K 的矩阵布局和 stride |
| `layoutA3/B3/C3` | Cube3 | dS^T*Q、P^T*dOut 的矩阵布局和 stride |
| `actualShape1/2/3` | Cube | 传给 BlockMmad 的实际 M/N/K，避免尾块按满 tile 计算 |
| `lowPrecisionSOffset` | Cube2/Cube3 | 将 FP32 S Workspace 行偏移转换为低精度 P/dS 区偏移 |
| `splitRow` | Vector | 一个 Packet 在两个 AIV 之间的行切分点 |
| `vecBegin/vecEnd` | Vector | 当前 AIV 实际处理的 Packet 行范围 |
| `lseBrocTensor/softmaxGradTensor` | Vector | 普通与 packed 路径复用的 LSE/D UB 缓冲 |
| `vecBegin/vecEnd` | Vector | 用于 Packet S/dP 行偏移，并确定本次 LSE/D segment 交集 |
| `coreOffset` | Vector | 当前 AIV 在 Packet 矩阵中的行偏移 |
| `vector32Soffset` | Vector | FP32 S/dP 区的字节地址偏移 |
| `vector16Soffset` | Vector | 低精度 P/dS 区的字节地址偏移 |

## 9. 变量、地址和事件的关键约束

1. `qOffset/kvOffset/sOffset` 均是元素偏移或 Workspace 逻辑偏移，转成 GM 地址时才乘 `sizeof(T)`。
2. TND 的 Q stride 包含 `numHeads * headDim`，KV stride 包含 `kvHeads * headDim`；BNSD 的每个 head 行为连续 headDim。
3. LSE/D 的源地址与 Restore 目标必须使用原始 Q 行号；`LseCopy/CopyDIn` 写入 UB 时使用当前 compute 的局部行号。
4. `gDqPack` 是每个逻辑 AIC core 独占的 FP32 缓冲区，不允许跨 core 共享写入。
5. `pingpongFlag` 只有在当前 Packet 的 Cube/Vector/Restore 依赖完成后才能翻转。
6. `rsvdBlockCount` 被限制到当前 Q/K block 数，索引负值和越界值被跳过。

## 10. 复杂度和性能关注点

每个有效 Q/KV block 对的主要 Cube 工作量约为：

```text
Cube1: 2 * M * N * D
Cube2:     M * N * D
Cube3: 2 * N * M * D
```

离散 Packet 还增加 Q/dOut Pack、dQ Restore、GM DataCopy 和跨核事件。性能分析应优先统计 Packet 有效 M、segment 数、Pack 字节数、AIC/AIV Wait 时间、Atomic 写回时间和各 Core 执行时间。

## 11. 调用关系

```text
operator
  ├─ InitCommonParams
  │   ├─ InitKTaskInfo -> UpdateKTaskCalSize
  │   └─ Workspace/Tensor 初始化
  ├─ VecPre
  ├─ VecPrepareD
  ├─ GBSAGGater::GetPacketCount -> GetQBlockSize
  ├─ GBSAGGater::Aggregate()
  ├─ GBSAGGater::Gather(GatherParams)
  ├─ SimpltSoftmax::LseCopy/CopyDIn
  ├─ Cube1 -> SoftmaxGrad -> Cube2/Cube3
  ├─ GBSAGScateer::Scatter(Params, QPacket, true)
  ├─ UpdateNextKTaskInfo -> UpdateKTaskCalSize
  └─ VecPost
```

`SetFlag/WaitFlag` 管理 Cube 内部事件；`CUBE2VEC/VEC2CUBE/CUBE2POST` 管理 AIC 与 AIV 的跨核依赖；`initTaskInfo/updateNextTaskInfo` 是旧 Q_OUT 辅助路径，不属于当前 K_OUT 主路径。
