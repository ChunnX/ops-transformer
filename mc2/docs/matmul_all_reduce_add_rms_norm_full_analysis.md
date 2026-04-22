# matmul_all_reduce_add_rms_norm 算子全面分析报告

## 1. 算子概述

`matmul_all_reduce_add_rms_norm` 是 matmul_all_reduce 的扩展融合算子，在 **MatMul + AllReduce** 完成后，立即对结果执行 **Residual Add + RMSNorm**，形成四阶段融合流水线。核心优势在于 AllReduce 结果无需写回 HBM 再被 ARN (AddRmsNorm) 读回，直接在 GM 上原地完成 residual add 和 RMS normalization。

### 1.1 算子语义

```
输入:  x1[S, M, K], x2[K, N], residual[S, M, N], gamma[N], bias[M](可选)
计算:
  mm_result = x1 × x2 + bias               (分片MatMul)
  ar_result = AllReduce(mm_result)          (跨TP域归约)
  y = ar_result + residual                  (残差加法)
  norm_out = y / RMS(y) * gamma             (RMS归一化)
输出:  y[S, M, N], norm_out[S, M, N]
```

### 1.2 三种量化变体

| 变体 | x1/x2类型 | 量化参数 | 核心路径 |
|------|----------|---------|---------|
| 无量化 | FP16/BF16 × FP16/BF16 | 无 | MatmulAllReduceAddRmsNorm910General |
| Per-Token量化 | INT8 × INT8 | dequant_scale | MatmulAllReduceAddRmsNormQuantBF16 |
| 权重量化 | FP16/BF16 × INT8/INT4 | antiquant_scale+offset | MatmulAllReduceAddRmsNormWeightQuant |

---

## 2. 目录结构与文件职责

```
matmul_all_reduce_add_rms_norm/
├── op_kernel/
│   ├── matmul_all_reduce_add_rms_norm.cpp             :125  内核入口(第43-46行入口, 第47-124行函数体)
│   ├── matmul_all_reduce_add_rms_norm_tiling_data.h   :74   Tiling数据结构(第35-43行AddRMSNormTilingData, 第45-50行KeyData, 第52-57/59-64/66-71行三套变体)
│   ├── mm_allreduce_add_rms_norm_910_general.h        :113  通用版(第41-42行类定义, 第56-80行Process)
│   ├── mm_allreduce_add_rms_norm_quant.h              :96   Per-Token量化版
│   ├── mm_allreduce_add_rms_norm_weight_quant.h       :94   权重量化版
│   ├── add_rms_norm_kernel.h                          :170  ARN调度器(HCCL Query轮询)
│   ├── add_rms_norm.h                                 :304  标准ARN(逐行处理)
│   ├── add_rms_norm_split_d.h                         :330  大列ARN(D维分块)
│   ├── add_rms_norm_merge_n.h                         :297  小列多行ARN
│   ├── add_rms_norm_single_n.h                        :238  单行/核ARN
│   ├── add_rms_norm_multi_n.h                         :275  FP16对齐多行ARN
│   ├── rms_norm_base.h                                :195  RMSNorm基础工具函数
│   └── reduce_common.h                                :176  归约辅助函数
├── op_host/
│   ├── matmul_all_reduce_add_rms_norm_def.cpp         :161  算子定义(第18行类定义, 第21-156行构造函数, 第159行OP_ADD)
│   ├── matmul_all_reduce_add_rms_norm_infershape.cpp  :195  Shape推导
│   └── op_tiling/
│       ├── common_add_rms_norm_tiling.h               :80   ARN Tiling公共定义
│       ├── common_add_rms_norm_tiling.cpp             :307  ARN模式选择+参数计算
│       ├── matmul_all_reduce_add_rms_norm_tiling.h    :68   通用Tiling(优先级2)
│       ├── matmul_all_reduce_add_rms_norm_tiling.cpp  :213  通用Tiling实现
│       ├── quant_matmul_all_reduce_add_rms_norm_tiling.h/cpp  量化Tiling(优先级0)
│       └── weight_quant_matmul_all_reduce_add_rms_norm_tiling.h/cpp  权重量化Tiling(优先级1)
├── op_api/
│   ├── aclnn_matmul_all_reduce_add_rms_norm.h/cpp            无量化API
│   ├── aclnn_quant_matmul_all_reduce_add_rms_norm.h/cpp      Per-Token量化API
│   ├── aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h/cpp 权重量化API
│   └── matmul_all_reduce_arn_util.h/cpp              公共API工具
└── op_graph/ (proto, gen_task, fallback)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `matmul_all_reduce_add_rms_norm_def.cpp`
- 第18行: 类 `MatmulAllReduceAddRmsNorm` 定义
- 第21-156行: 构造函数 (输入/输出/属性/平台配置)
- 第159行: `OP_ADD(MatmulAllReduceAddRmsNorm)` 注册

**输入张量** (8个):

| 索引 | 名称 | 必选 | 数据类型 | 说明 | 源码行号 |
|------|------|------|---------|------|---------|
| 0 | x1 | REQUIRED | FP16/BF16/INT8 | 矩阵A [S,M,K] 或 [M,K] | 第23-33行 |
| 1 | x2 | REQUIRED | FP16/BF16/INT8/INT4 | 矩阵B [K,N] | 第34-45行 |
| 2 | bias | OPTIONAL | FP16/BF16/INT32 | 偏置 [N] | 第46-56行 |
| 3 | residual | REQUIRED | FP16/BF16 | 残差输入 [S,M,N] | 第57-67行 |
| 4 | gamma | REQUIRED | FP16/BF16 | RMSNorm缩放 [N] | 第68-78行 |
| 5 | antiquant_scale | OPTIONAL | FP16/BF16 | 权重量化缩放因子 | 第79-89行 |
| 6 | antiquant_offset | OPTIONAL | FP16/BF16 | 权重量化偏移 | 第90-100行 |
| 7 | dequant_scale | OPTIONAL | FP16/BF16/UINT64/INT64 | Per-Token反量化缩放 | 第101-111行 |

**输出张量** (2个):

| 索引 | 名称 | 类型 | 说明 | 源码行号 |
|------|------|------|------|---------|
| 0 | y | FP16/BF16 | all_reduce_result + residual | 第113-123行 |
| 1 | norm_out | FP16/BF16 | RMSNorm(y, gamma, epsilon) | 第124-134行 |

**属性** (7个, 第136-142行):

| 序号 | 名称 | 类型 | 默认值 | 说明 | 源码行号 |
|------|------|------|--------|------|---------|
| 0 | group | String | REQUIRED | HCCL通信组名 | 第136行 |
| 1 | reduce_op | String | "sum" | 归约操作 | 第137行 |
| 2 | is_trans_a | Bool | false | A矩阵转置 | 第138行 |
| 3 | is_trans_b | Bool | false | B矩阵转置 | 第139行 |
| 4 | comm_turn | Int | 0 | 通信切分数 | 第140行 |
| 5 | antiquant_group_size | Int | 0 | 权重量化分组大小 | 第141行 |
| 6 | epsilon | Float | 1e-6 | RMSNorm的ε参数 | 第142行 |

**平台配置** (第144-155行): `ascend910b` (动态shape/动态rank/动态格式), `KERNEL_TYPE_MIX_AIC_1_2`

### 3.2 Shape 推导

**文件**: `matmul_all_reduce_add_rms_norm_infershape.cpp` (第28-195行)

```
InferShapeForMC2AddRmsNorm():
  ├── InferShapeForMatmul(): 获取M/N/K, 验证x1(2D/3D), x2(2D), K一致性
  ├── 验证 antiquant_scale shape:
  │   ├── !transB: [N, CeilDiv(K, group_size)]
  │   └── transB: [CeilDiv(K, group_size), N]
  ├── 验证 residual 是 3D [S, M, N]
  ├── y shape = residual shape     (residual add结果)
  └── norm_out shape = residual shape  (RMSNorm结果)

InferDtypeForMC2AddRmsNorm():
  └── y.dtype = norm_out.dtype = residual.dtype
```

### 3.3 Tiling 策略

**三种 Tiling 类优先级** (按序尝试，首个成功即采用):

```
优先级0: QuantMatmulAllReduceAddRmsNormTiling     (Per-Token量化, keys 8/4104)
优先级1: WeightQuantMatmulAllReduceAddRmsNormTiling (权重量化, 12个key)
优先级2: MatmulAllReduceAddRmsNormTiling           (通用, keys 0/256/260)
```

**通用 Tiling 主流程** (`matmul_all_reduce_add_rms_norm_tiling.cpp`, 第57-210行):

```
DoOpTiling():
  ├── 1. MMNTilingTransferHelper.DoOpTiling() → 复用matmul_all_reduce tiling
  │   └── 得到 tileMValue_, tailMValue_, aicCoreNum
  ├── 2. 检查ARN输入 (residual, gamma非空)
  ├── 3. [有tail] Tiling4AddRmsNorm(tileMValue_, tileTiling, tileKey)
  │   └── [无tail] Tiling4AddRmsNorm(tileMValue_, tileTiling, tileKey)
  └── 4. [有tail] Tiling4AddRmsNorm(tailMValue_, tailTiling, tailKey)

Workspace:
  = mc2WorkSpace + (arnWorkSpace - SYS_WORKSPACE_SIZE)
  (16MB系统workspace共享, ARN部分只追加增量)

PostTiling:
  ├── blockDim = aicCoreNum
  ├── 断言: aicCoreNum * 2 >= ARN所需的AIV核数 (1:2 AIC:AIV比例)
  └── SetScheduleMode(batch_mode)  // SyncAll需要所有核同时启动
```

**ARN 模式选择** (`common_add_rms_norm_tiling.cpp`, 第217-246行):

```
SetAddRmsNormTilingData(numRow, numCol, x1Dtype):
  ├── numCol > UB_FACTOR (12288 FP16 / 10240 FP32) → K_SPLIT_D
  ├── blockFactor == 1  → K_SINGLE_N
  ├── numColAlign <= 2000  → K_MERGE_N
  ├── FP16 && 列对齐 → K_MULTI_N
  └── 默认 → K_NORMAL

TilingKey 编码: dtypeKey * 10 + modeKey
  dtypeKey: 1=FP16, 2=FP32, 3=BF16
  modeKey:  0=Normal, 1=SplitD, 2=MergeN, 3=SingleN, 4=MultiN
  → 产生: 10/11/12/13/14(FP16), 30/31/32/33(BF16)

Tiling参数计算:
  ├── numCore = GetCoreNumAIV()
  ├── blockFactor = CeilDiv(numRow, numCore)  (每核处理的行数)
  ├── rowFactor = UB容量相关的行批处理因子
  └── epsilon, avg_factor = 1.0/numCol
```

### 3.4 ACLNN API

三套 API 共享 `InnerMatmulAllReduceAddRmsNormGetWorkspaceSize()` 实现:

```c
// 无量化
aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(
    x1, x2, bias, residual, gamma,
    epsilon, group, reduceOp, commTurn, streamMode,
    y, normOut, &workspaceSize, &executor);

// Per-Token量化 (增加dequantScale)
aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    x1, x2, bias, dequantScale, residual, gamma,
    epsilon, group, reduceOp, commTurn, streamMode,
    y, normOut, &workspaceSize, &executor);

// 权重量化 (增加antiquantScale/Offset/GroupSize)
aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    x1, x2, bias, residual, gamma,
    antiquantScale, antiquantOffset, epsilon,
    group, reduceOp, commTurn, streamMode, antiquantGroupSize,
    y, normOut, &workspaceSize, &executor);
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `matmul_all_reduce_add_rms_norm.cpp`
- 第43-46行: 内核入口函数签名 (GM_ADDR 参数列表)
- 第47-124行: 内核函数体 (Tiling注册 + 三种量化分支分发)

```cpp
extern "C" __global__ __aicore__ void matmul_all_reduce_add_rms_norm(
    GM_ADDR aGM, bGM, biasGM, residualGM, gammaGM,
    antiquantScaleGM, antiquantOffsetGM, dequantGM,
    yGM, normOutGM, workspaceGM, tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 1AIC+2AIV

    // 注册3种Tiling解析器 (根据TilingKey选择)
    REGISTER_TILING_DEFAULT(MatmulAllReduceAddRmsNormTilingData);        // 默认
    REGISTER_TILING_FOR_TILINGKEY(8, QuantMatmul...);                     // Per-Token
    REGISTER_TILING_FOR_TILINGKEY(4104, QuantMatmul...);                  // Per-Token(BF16)
    // Weight-Quant: 12个特定key注册 WeightQuantMatmul...

    // 分发到3种实现
    #ifdef MC2_QUANT_FP16/BF16 → MatmulAllReduceAddRmsNormQuantBF16
    #ifdef MC2_WEIGHT_QUANT → MatmulAllReduceAddRmsNormWeightQuant
    #else → MatmulAllReduceAddRmsNorm910General
}
```

**TilingKey 分发规则**:
- Keys 0, 256, 260: 无量化 → `Mc2MatmulBaseKernel` / `Mc2MatmulBaseUnAlignedKernel`
- Keys 8, 4104: Per-Token量化 → `BmmDequant` (FP16) / `BmmDequantBf16` (BF16)
- 12个权重量化Key: transB × quantType(PER_TENSOR/CHANNEL/GROUP) × offsetFlag

### 4.2 通用版: MatmulAllReduceAddRmsNorm910General

**文件**: `mm_allreduce_add_rms_norm_910_general.h`
- 第41-42行: 类 `MatmulAllReduceAddRmsNorm910General` 定义 (继承 `MatmulAllReduce910General`)
- 第56-80行: `Process()` 方法 (MM+AR 流水线 + ARN 轮询调度)

继承 `MatmulAllReduce910General` (来自 matmul_all_reduce 算子), 增加ARN阶段:

```
MatmulAllReduceAddRmsNorm910General<xType, wType, yType, mmType>
    : public MatmulAllReduce910General<xType, wType, yType, mmType,
                                       Mc2CoreType::ON_CUBE_AND_VECTOR>
```

**Process 流程** (`910_general.h`, 第56-80行):

```
Process():
  ├── [可选] PreProcForBiasOnVector(): BF16 bias FP32预处理
  ├── InnerProcess(false, tileCnt, tileInfo_):  // 主块循环
  │   └── for tile: MatMul → PostProcEachTurn → HCCL Commit
  ├── InnerProcess(true, tailCnt, tailInfo_):   // 尾块循环
  ├── SyncAll<false>():  全核同步(确保AllReduce全部提交)
  │
  ├── [AIV] AddRmsNormKernel::ComputeAddRmsNorm():  ← ARN主循环
  │   ├── while (true):
  │   │   ├── [block0] GetCurFinishedCnt() → hccl_.Query()查询已完成数
  │   │   ├── [block0] DataCacheCleanAndInvalid → 写入GM供所有核可见
  │   │   ├── SyncAll() → 全核同步获取完成数
  │   │   ├── ComputeAddRmsNormInner(tile部分):
  │   │   │   └── AddRmsNorm(tilingData, key, numBlocks, ...)
  │   │   ├── ComputeAddRmsNormInner(tail部分):
  │   │   │   └── AddRmsNorm(tailingData, key, numBlocks, ...)
  │   │   ├── 指针递增: normOutGM += offset*cnt, yGM += offset*cnt
  │   │   └── if (lastCnt >= tileCnt+tailCnt) break
  │
  ├── SyncAll<false>()
  └── hccl_.Finalize()
```

### 4.3 AddRmsNormKernel 调度器

**文件**: `add_rms_norm_kernel.h` (第36-170行)

核心设计: **流式生产者-消费者模式** — AIC 生产 AllReduce 结果分片，AIV 轮询完成后立即执行 ARN:

```
AddRmsNormKernel:
  构造参数: arnAddrs, tPipe, dataSize, tileCnt, tailCnt,
            hccl指针, tileHandleId, tailHandleId

  ComputeAddRmsNorm():
    while (true):
      ├── [blockIdx==0] GetCurFinishedCnt():
      │   ├── tileCnt = hccl_.Query(tileHandleId_)  ← 非阻塞查询
      │   └── return tileCnt + hccl_.Query(tailHandleId_)
      ├── [block0] DataCacheCleanAndInvalid 写入完成数到GM
      ├── SyncAll()  ← 所有核同步
      ├── lastCnt = *rcvCntGM  ← 读取完成数
      ├── [lastCnt >= addRmsNormCount]:
      │   ├── 处理tile部分 (key/numBlocks来自ARNKeyTile/ARNNumBlocksTile)
      │   └── 处理tail部分 (key/numBlocks来自ARNKeyTail/ARNNumBlocksTail)
      └── [lastCnt >= tileCnt+tailCnt] break

  AddRmsNorm():
    └── 根据key分发到5种ARN实现:
        ├── 10 (ADD_RMS_NORM)          → KernelAddRmsNorm<half>
        ├── 30 (ADD_RMS_NORM_BF16)     → KernelAddRmsNorm<bfloat16_t>
        ├── 11 (ADD_RMS_NORM_SPLIT_D)  → KernelAddRmsNormSplitD<half>
        ├── 31 (ADD_RMS_NORM_SPLIT_BF16) → KernelAddRmsNormSplitD<bfloat16_t>
        ├── 12 (ADD_RMS_NORM_MERGE_N)  → KernelAddRmsNormMergeN<half>
        ├── 32 (ADD_RMS_NORM_MERGE_N_BF16) → KernelAddRmsNormMergeN<bfloat16_t>
        ├── 13 (ADD_RMS_NORM_SINGLE_N) → KernelAddRmsNormSingleN<half>
        ├── 33 (ADD_RMS_NORM_SINGLE_N_BF16) → KernelAddRmsNormSingleN<bfloat16_t>
        └── 14 (ADD_RMS_NORM_MULTI_N)  → KernelAddRmsNormMultiN<half>
```

### 4.4 五种 ARN 内核变体

#### 4.4.1 KernelAddRmsNorm (标准模式, key=10/30)

**文件**: `add_rms_norm.h` (第22-304行)

逐行处理, 每行完整执行 Add + RMSNorm:

```
ComputeProcess():
  for row in assigned_rows:
    ├── CopyIn: GM→UB
    │   ├── normOut[i*numCol]
    │   └── residual[i*numCol]
    ├── Add: result = normOut + residual → yGM
    │   ├── half: 直接Add
    │   └── bfloat16_t: Cast(BF16→FP32) → Add → Cast(FP32→BF16)
    ├── RMSNorm:
    │   ├── Cast → FP32
    │   ├── Mul: x²
    │   ├── Muls(avgFactor): Σx²/N
    │   ├── ReduceSumCustom: 行内归约
    │   ├── Adds(epsilon): + ε
    │   ├── Sqrt: √(...)
    │   ├── Div(1/sqrt): rstd = 1/√(...)
    │   ├── Cast → 原始类型
    │   └── Mul(gamma): norm_out = result × rstd × gamma
    └── CopyOut: UB→GM (norm_out)
```

#### 4.4.2 KernelAddRmsNormSplitD (大列模式, key=11/31)

**文件**: `add_rms_norm_split_d.h` (第21-330行)

当 numCol > UB容量时，将列分块处理，三阶段流水线:

```
ComputeProcess():
  for row:
    ├── ComputeFormer():  阶段1 - 累加平方和
    │   ├── for chunk in D:
    │   │   ├── 加载x1+x2 → Add → x² → 部分和
    │   │   └── BlockReduceSumFP32累加
    │   └── 写入部分和到workspace
    ├── ComputeRstd():  阶段2 - 计算rstd
    │   ├── 加载所有部分和 → 总和
    │   └── rstd = 1/√(sum/N + ε)
    └── ComputeLatter():  阶段3 - 归一化
        ├── for chunk in D:
        │   ├── 加载y + gamma_chunk
        │   └── y × rstd × gamma → norm_out_chunk
        └── CopyOut
```

#### 4.4.3 KernelAddRmsNormMergeN (小列多行模式, key=12/32)

**文件**: `add_rms_norm_merge_n.h` (第25-297行)

当 numColAlign <= 2000 时，同时处理多行, 减少每行开销:

```
特点:
  ├── BroadCast: gamma广播到多行
  ├── ReduceSumMultiN: 对(N,D)张量一次完成N行的行内归约
  ├── DOUBLE_BUFFER_QUEUE = 2: 双缓冲流水线
  └── 多行批处理: rowFactor = UB容量 / (numColAlign * sizeof(T) * 16 + 64)
```

#### 4.4.4 KernelAddRmsNormSingleN (单行/核模式, key=13/33)

**文件**: `add_rms_norm_single_n.h` (第26-238行)

当 blockFactor==1 (每个AIV核只处理1行) 时的高度优化版本:

```
特点:
  ├── SINGLE_N_BUFFER_SIZE = 195,584字节 (191 blocks × 1024)
  ├── 异步DataCopy重叠: normOut和residual并发加载
  ├── gamma加载与x输出Copy重叠
  └── 事件驱动: EVENT_ID0/1控制流水线
```

#### 4.4.5 KernelAddRmsNormMultiN (FP16对齐多行模式, key=14)

**文件**: `add_rms_norm_multi_n.h` (第21-275行)

仅支持 FP16 + 列对齐场景, 使用 Gather 收集 rstd 值后批量乘:

```
特点:
  ├── 仅支持 half 类型 (BF16走其他模式)
  ├── rowFactor = (ubSize - 1024 - 256 - numColAlign*2) / (numColAlign*16 + 64)
  └── Gather + 批量Mul with repeat参数
```

### 4.5 Per-Token 量化版

**文件**: `mm_allreduce_add_rms_norm_quant.h` (第33-96行)

```
MatmulAllReduceAddRmsNormQuantBF16<xType, wType, yType, mmType, coreType>
    : public MatmulAllReduceQuantBF16<xType, wType, yType, mmType, coreType, false>

特点:
  ├── coreType可配置:
  │   ├── FP16量化: ON_CUBE_AND_VECTOR (Cube+Vector协同)
  │   └── BF16量化: ON_VECTOR (仅Vector, BF16反量化需Vector)
  ├── Process接收mmType&引用 (非默认构造, 需REGIST_MATMUL_OBJ)
  └── Mc2SyncAll<coreType>() 替代 SyncAll<false>()
```

### 4.6 权重量化版

**文件**: `mm_allreduce_add_rms_norm_weight_quant.h` (第34-94行)

```
MatmulAllReduceAddRmsNormWeightQuant<xType, wType, yType, mmType>
    : public MatmulAllReduceWeightQuant<xType, wType, yType, mmType>

量化类型: Mc2QuantType::PER_TENSOR / PER_CHANNEL / PER_GROUP
12个TilingKey: transB(2) × quantType(3) × offsetFlag(2)
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

**文件**: `matmul_all_reduce_add_rms_norm_tiling_data.h`
- 第35-43行: `AddRMSNormTilingData` 结构体 (num_row, num_col, block_factor, row_factor, ub_factor, epsilon, avg_factor)
- 第45-50行: `AddRMSNormTilingeKeyData` 结构体 (ARNKeyTile, ARNKeyTail, ARNNumBlocksTile, ARNNumBlocksTail)
- 第52-57行: `MatmulAllReduceAddRmsNormTilingData` 结构体 (无量化版, 复用 matmul_all_reduce + ARN)
- 第59-64行: `QuantMatmulAllReduceAddRmsNormTilingData` 结构体 (Per-Token量化版)
- 第66-71行: `WeightQuantMatmulAllReduceAddRmsNormTilingData` 结构体 (权重量化版)

```
// 无量化版 (MatmulAllReduceAddRmsNormTilingData, 第52-57行)
Mc2Tiling::MatmulAllReduce910TilingData  mmTiling;    ← 复用matmul_all_reduce
Mc2Tiling::AddRMSNormTilingData          arnTileTiling;  ← 主块ARN参数
Mc2Tiling::AddRMSNormTilingData          arnTailTiling;  ← 尾块ARN参数
Mc2Tiling::AddRMSNormTilingeKeyData      arnKeyData;     ← ARN模式选择key

// AddRMSNormTilingData 字段 (第35-43行):
uint32_t num_row;       ← 本轮处理的行数
uint32_t num_col;       ← 列数 (N)
uint32_t block_factor;  ← 每核处理的行数
uint32_t row_factor;    ← 每批处理的行数
uint32_t ub_factor;     ← UB单次处理元素数
float    epsilon;       ← RMSNorm ε
float    avg_factor;    ← 1.0/numCol

// AddRMSNormTilingeKeyData 字段 (第45-50行):
uint32_t ARNKeyTile;        ← 主块ARN模式key (10-14/30-33)
uint32_t ARNKeyTail;        ← 尾块ARN模式key
uint32_t ARNNumBlocksTile;  ← 主块ARN使用的核数
uint32_t ARNNumBlocksTail;  ← 尾块ARN使用的核数
```

### 5.2 GM 地址布局

```
// MC2GmAddrs (继承自 matmul_all_reduce)
aGM:      A矩阵 [S,M,K]
bGM:      B矩阵 [K,N]
cGM:      AllReduce输出 (= AllReduce(A×B)) [S,M,N]
outputGM: 最终输出(可指向cGM或WindowIn)
addGM:    bias地址 (如果isAdd)

// QuantGmAddrs (量化场景)
dequantScaleGM:  Per-Token反量化缩放因子
antiquantScaleGM:  权重量化缩放因子
antiquantOffsetGM: 权重量化偏移

// ArnGmAddrs (ARN专用)
residualGM:  残差输入 [S,M,N]
gammaGM:     RMSNorm缩放 [N]
yGM:         残差加法输出 [S,M,N]
normOutGM:   RMSNorm输出 [S,M,N]
```

---

## 6. 数据流

### 6.1 端到端数据流

```
时间线 (MM+AR 与 ARN 的流水线重叠):

AIC: [tile0: MM→AR_commit][tile1: MM→AR_commit][tile2: MM→AR_commit]...
HCCL:                  [----AR tile0----][----AR tile1----][----AR tile2----]...
AIV:                                           [ARN tile0][ARN tile1][ARN tile2]...
                                               ↑ Query(tile0完成) ↑ Query(tile1完成)

详细数据流:
┌──────────────────────────────────────────────────────────┐
│ GM                                                        │
│ ┌─────────┐  ┌─────────┐  ┌────────┐  ┌──────────┐     │
│ │A [S,M,K]│  │B [K,N]  │  │bias[N] │  │residual  │     │
│ │         │  │         │  │        │  │[S,M,N]   │     │
│ └────┬────┘  └────┬────┘  └───┬────┘  └────┬─────┘     │
│      │            │           │             │            │
│      ▼            ▼           ▼             │            │
│ ┌──────────────────────────┐  │             │            │
│ │ AIC: MatMul (分片)       │  │             │            │
│ │ partial_C = A_tile×B+bias│  │             │            │
│ └───────────┬──────────────┘  │             │            │
│             ▼                 │             │            │
│ ┌──────────────────────────┐  │             │            │
│ │ HCCL: AllReduce          │  │             │            │
│ │ C = Σ(partial_C)         │  │             │            │
│ │ (tile-by-tile完成)       │  │             │            │
│ └───────────┬──────────────┘  │             │            │
│             ▼                 │             │            │
│ ┌──────────────────────────┐  │             │            │
│ │ AIV: AddRmsNorm          │  │             │            │
│ │ y = C + residual ────────┼──┼─────────────┘            │
│ │ norm = RMSNorm(y,gamma)  │  │                          │
│ └──────┬──────────┬────────┘  │                          │
│        │          │            │                          │
│        ▼          ▼            │                          │
│ ┌──────────┐  ┌──────────┐   │                          │
│ │y[S,M,N]  │  │norm_out  │   │                          │
│ │(add结果) │  │[S,M,N]   │   │                          │
│ └──────────┘  └──────────┘   │                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 内存层次

```
┌── AIC (Cube) ─────────────────────┐
│ L0: A[m0,k0], B[k0,n0], C[m0,n0] │
│ L1: A/B分片预取                   │
│ L0C → GM (AllReduce输入)          │
└───────────────────────────────────┘

┌── HCCL ───────────────────────────┐
│ GM partial_C → 网络传输 → GM C     │
│ (Window IPC 或 RDMA)              │
└───────────────────────────────────┘

┌── AIV (Vector) ───────────────────┐
│ UB (~256KB):                       │
│ ├── normOut行 + residual行         │
│ ├── gamma行 (部分或全部)           │
│ ├── FP32中间结果 (x², sum, rstd)  │
│ └── 输出缓冲 (y, norm_out)        │
│                                    │
│ BF16场景额外UB开销:               │
│ └── FP32中间缓冲 (Cast BF16↔FP32) │
└───────────────────────────────────┘
```

---

## 7. 性能设计

### 7.1 三重重叠流水线

```
AIC:  [MM tile_i] [MM tile_i+1] [MM tile_i+2]
HCCL:             [AR tile_i  ] [AR tile_i+1  ]
AIV:                            [ARN tile_i   ] [ARN tile_i+1]

三级流水线:
  Level 1: AIC计算 与 HCCL AllReduce 重叠 (继承自matmul_all_reduce)
  Level 2: HCCL完成 与 AIV ARN 重叠 (Query轮询)
  Level 3: tile_i ARN 与 tile_i+1 MM并行
```

### 7.2 ARN 模式自适应选择

```
输入特征                     选择模式          原因
─────────────────────────────────────────────────────
numCol > 12288(FP16)      SplitD           列不-fit-UB,分块处理
blockFactor == 1           SingleN          每核只1行,单行优化
numColAlign <= 2000        MergeN           小列宽,多行合并减少开销
FP16 && 列对齐             MultiN           利用FP16 SIMD效率
默认                       Normal           通用逐行处理
```

### 7.3 BF16 特殊处理

BF16 算术精度不足, 所有中间计算转 FP32:
```
BF16 → Cast(FP32) → Add/Mul/Reduce → Cast(BF16)
额外开销: 每次Cast ~2 cycles, 约30%性能损失 vs FP16
```

---

## 8. 依赖关系

### 8.1 内部依赖

```
matmul_all_reduce_add_rms_norm
├── mc2/matmul_all_reduce/ (继承关系)
│   ├── op_kernel/arch32/matmul_all_reduce_base.h      ← MatmulAllReduceBase
│   ├── op_kernel/arch32/matmul_all_reduce_910_general.h ← MatmulAllReduce910General
│   ├── op_kernel/arch32/*_quant*.h                     ← 量化路径基类
│   └── op_host/op_tiling/matmul_all_reduce_tiling_base.h ← Tiling基类
├── mc2/common/
│   ├── op_kernel/mc2_tiling_struct.h   ← MC2ServerCfg, MC2HcommCfg
│   ├── op_kernel/mc2_common_def.h      ← 通信算法枚举
│   └── op_host/op_tiling/mc2_tiling_utils.h ← 公共工具
└── 自有 kernel文件 (add_rms_norm_*.h, rms_norm_base.h, reduce_common.h)
```

### 8.2 外部依赖

- **HCCL**: AllReduce 通信 (AICPU server type), `hccl_.Query()` 非阻塞查询
- **AscendC Runtime**: KERNEL_TYPE_MIX_AIC_1_2, SyncAll, DataCacheCleanAndInvalid
- **matmul_all_reduce**: 通过继承复用其 MM+AR 全部逻辑

---

## 9. 函数调用关系总览

```
matmul_all_reduce_add_rms_norm (内核入口)
├── REGISTER_TILING_DEFAULT / REGISTER_TILING_FOR_TILINGKEY
│
├── [无量化, keys 0/256/260]
│   └── MatmulAllReduceAddRmsNorm910General
│       ├── Init(): 继承基类 + 解析ARN Tiling
│       └── Process():
│           ├── PreProcForBiasOnVector() [可选]
│           ├── InnerProcess(tileCnt): MM + AllReduce
│           │   └── for tile: CatlassMatmul → PostProcEachTurn → HCCL Commit
│           ├── InnerProcess(tailCnt): MM + AllReduce (尾块)
│           ├── SyncAll()
│           ├── AddRmsNormKernel::ComputeAddRmsNorm()
│           │   ├── hccl_.Query() 轮询完成数
│           │   ├── ComputeAddRmsNormInner(tile):
│           │   │   └── AddRmsNorm() → 9种变体分发
│           │   │       ├── KernelAddRmsNorm        (逐行)
│           │   │       ├── KernelAddRmsNormSplitD   (大列分块)
│           │   │       ├── KernelAddRmsNormMergeN   (小列多行)
│           │   │       ├── KernelAddRmsNormSingleN  (单行/核)
│           │   │       └── KernelAddRmsNormMultiN   (FP16对齐)
│           │   └── ComputeAddRmsNormInner(tail): 同上
│           ├── SyncAll()
│           └── hccl_.Finalize()
│
├── [Per-Token量化, keys 8/4104]
│   └── MatmulAllReduceAddRmsNormQuantBF16
│       └── Process(mmTile, mmTail): 量化MM+AR + ARN
│
└── [权重量化, 12个key]
    └── MatmulAllReduceAddRmsNormWeightQuant
        └── Process(): 反量化+MM+AR + ARN
```

---

## 10. 变体对比

### 10.1 算子族概览

`matmul_all_reduce_add_rms_norm` 属于 `matmul_all_reduce` 算子族，该族包含三个相关算子：

| 特性 | matmul_all_reduce | matmul_all_reduce_add_rms_norm | inplace_matmul_all_reduce_add_rms_norm |
|------|-------------------|-------------------------------|---------------------------------------|
| 计算流水线 | MatMul + AllReduce | MatMul + AllReduce + Add + RMSNorm | MatMul + AllReduce + Add + RMSNorm |
| 输出数 | 1 (y) | 2 (y, norm_out) | 2 (residual(覆写), norm_out) |
| 输入数 | 9 (含x3, pertoken_scale等) | 8 | 8 |
| residual输入 | 无 | 独立输入 | 独立输入 |
| y输出 | AllReduce结果 | AllReduce + residual | 覆写residual张量 |
| 内存量 | 标准开销 | 需额外存储residual和y | 省去y的独立存储 |
| 适用场景 | 纯通信计算融合 | 完整Transformer层后处理 | 内存敏感场景 |
| HCCL组 | group | group | group |
| Kernel类型 | MIX_AIC_1_2 | MIX_AIC_1_2 | MIX_AIC_1_2 |

### 10.2 matmul_all_reduce_add_rms_norm 内部三种量化变体

| 变体 | Kernel类 | TilingKey | x1/x2类型 | 额外参数 | 基类 | 量化场景 |
|------|---------|-----------|----------|---------|------|---------|
| 无量化 | MatmulAllReduceAddRmsNorm910General | 0, 256, 260 | FP16/BF16 x FP16/BF16 | 无 | MatmulAllReduce910General | 标准FP16/BF16训练/推理 |
| Per-Token量化 | MatmulAllReduceAddRmsNormQuantBF16 | 8, 4104 | INT8 x INT8 | dequant_scale | MatmulAllReduceQuantBF16 | INT8量化推理 |
| 权重量化 | MatmulAllReduceAddRmsNormWeightQuant | 12个key (>10000) | FP16/BF16 x INT8/INT4 | antiquant_scale + offset | MatmulAllReduceWeightQuant | W8A16/W4A16量化推理 |

**无量化 vs Per-Token量化 vs 权重量化关键差异**:

```
无量化 (MatmulAllReduceAddRmsNorm910General):
  ├── coreType: ON_CUBE_AND_VECTOR (Cube+Vector协同)
  ├── Process: InnerProcess → SyncAll → AddRmsNormKernel
  ├── 同步: SyncAll<false>()
  └── TilingData: MatmulAllReduce910TilingData + ARN

Per-Token量化 (MatmulAllReduceAddRmsNormQuantBF16):
  ├── coreType: FP16=ON_CUBE_AND_VECTOR, BF16=ON_VECTOR (BF16反量化需Vector)
  ├── Process: 接收mmType&引用 (REGIST_MATMUL_OBJ)
  ├── 同步: Mc2SyncAll<coreType>() 替代 SyncAll<false>()
  ├── FP16路径: BmmDequant (Cube+Vector)
  ├── BF16路径: BmmDequantBf16 (Vector only)
  └── TilingData: QuantMatmulAllReduceTilingData + ARN

权重量化 (MatmulAllReduceAddRmsNormWeightQuant):
  ├── 量化类型: PER_TENSOR / PER_CHANNEL / PER_GROUP
  ├── 12个TilingKey: transB(2) x quantType(3) x offsetFlag(2)
  ├── TilingData: WeightQuantMatmulAllReduceTilingData + ARN
  └── 额外参数: antiquant_group_size (PER_GROUP模式)
```

### 10.3 与 inplace 变体的对比

`inplace_matmul_all_reduce_add_rms_norm` 与 `matmul_all_reduce_add_rms_norm` 的核心区别在于输出策略：

| 对比项 | matmul_all_reduce_add_rms_norm | inplace_matmul_all_reduce_add_rms_norm |
|--------|-------------------------------|---------------------------------------|
| 类名 | MatmulAllReduceAddRmsNorm | InplaceMatmulAllReduceAddRmsNorm |
| 输出0 | y (独立输出张量) | residual (覆写输入residual) |
| 输出1 | norm_out | norm_out |
| 内存开销 | 需要独立的y张量 [S,M,N] | 复用residual张量,省去y的空间 |
| 输入定义 | 与inplace版完全相同 | 与标准版完全相同 |
| 属性定义 | 与inplace版完全相同 | 与标准版完全相同 |
| API命名 | aclnnMatmulAllReduceAddRmsNorm | aclnnInplaceMatmulAllReduceAddRmsNorm |
| 权衡 | 非破坏性,保留原始residual | 节省内存,但原始residual被覆盖 |

---

*分析完成时间: 2026-04-20 (修订: 2026-04-21, 添加源码行号引用及变体对比章节)*
