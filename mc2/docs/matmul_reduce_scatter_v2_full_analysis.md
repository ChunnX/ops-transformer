# matmul_reduce_scatter_v2 算子全面分析报告

## 1. 算子概述

`matmul_reduce_scatter_v2` 实现矩阵乘法 (MatMul) 与 ReduceScatter 通信算子的融合。在每张 NPU 卡上执行 `C = A × B` 的局部矩阵乘法后，将结果沿 M 维度按 rank 数均分，各 rank 对应分片上做跨卡求和归约，最终每张卡只保留属于自己的那一份结果 `[M/rankSize, N]`。

与 V1 版本相比，V2 的核心升级在于：
- **通信路径**：采用 Window IPC + Flag 同步的自研 AIV 通信路径，替代 V1 的 HCCL AICPU 路径，通信延迟更低
- **计算引擎**：使用 Catlass 模板线性代数库替代传统 MatmulImpl，支持分块预取 (Preload) 和 Shuffle-K 优化
- **SmallM 优化**：针对 M ≤ 2048 且 N > 512 的场景，引入列分割通信 (`CommColumnSplitter`) 和 CRTP 派生类 `MatmulReduceScatterAivModeSmallM`
- **量化支持**：INT8 × INT8 输入，支持 PerChannel/PerToken 反量化，以及 Fixpipe 硬件加速路径

### 1.1 算子语义

```
输入:  A[M, K] (每卡本地), B[K, N] (每卡本地)
计算:  C_full[m, n] = A[m, k] × B[k, n]   (每卡独立完成完整 MM)
通信:  C_local[m/rank, n] = ReduceScatter(C_full)  (跨卡沿M维归约+切片)
输出:  y[M/rankSize, N]   (每卡只保留属于自己 rank 的那块)
```

### 1.2 模板参数

```
// TilingKey 模板参数 (matmul_reduce_scatter_v2_tiling_key.h:23-28)
ASCENDC_TPL_ARGS_DECL(
    Mc2MatmulReduceScatterV2,
    ASCENDC_TPL_BOOL_DECL(TPL_ISBIAS, 0, 1),          // 是否有 bias
    ASCENDC_TPL_BOOL_DECL(TPL_IS_TRANSPOSE_A, 0, 1),   // A 矩阵转置
    ASCENDC_TPL_BOOL_DECL(TPL_IS_TRANSPOSE_B, 0, 1),   // B 矩阵转置
    ASCENDC_TPL_BOOL_DECL(TPL_IS_SMALLM, 0, 1),        // 小 M 优化路径
);

// 合法组合: ISBIAS∈{0,1} × TRANS_A=0 × TRANS_B∈{0,1} × SMALLM∈{0,1}
// 注意: 当前 A 矩阵不支持转置 (TRANS_A 固定为 0)
```

---

## 2. 目录结构与文件职责

```
matmul_reduce_scatter_v2/
├── op_kernel/
│   ├── matmul_reduce_scatter_v2.cpp                    :56   内核入口,模板分发
│   ├── matmul_reduce_scatter_v2_tiling_key.h           :43   模板参数声明
│   ├── common_def.h                                    :42   MAX_HANDLE/Block512B/BiasType
│   ├── matmul_reduce_scatter_aiv_mode.h                :548  AIV主类(Catlass MM + AIV Comm)
│   ├── matmul_reduce_scatter_aiv_mode_smallM.h         :658  SmallM优化派生类(CRTP)
│   ├── matmul_reduce_scatter_aiv_mode_util.h           :516  CommBase通信基类/工具函数
│   ├── matmul_reduce_scatter_aiv_mode_tiling.h         :127  Kernel侧Tiling数据结构
│   ├── matmul_reduce_scatter_aiv_mode_dequant.h        :157  DequantRunner反量化
│   ├── matmul_reduce_scatter_aiv_mode_padding.h        :198  PaddingRunner矩阵对齐
│   ├── matmul_reduce_scatter_aiv_mode_block_epilogue.h :      BlockEpilogue反量化块
│   ├── matmul.hpp / matmul_smallM.hpp                  :      Catlass MM kernel封装
│   └── matmul_reduce_scatter_v2_apt.cpp                :      arch35(950)内核入口
├── op_host/
│   ├── matmul_reduce_scatter_v2_def.cpp                :236  算子定义(6输入/2输出/12属性)
│   ├── matmul_reduce_scatter_v2_infershape.cpp         :60   Shape推导
│   └── op_tiling/
│       ├── matmul_reduce_scatter_tiling_base.h         :100  Tiling基类
│       ├── matmul_reduce_scatter_tiling_base.cpp       :      基类实现(公式化Tiling)
│       ├── reduce_scatter_formulaic_tiling.h/cpp       :      ReduceScatter公式化Tiling
│       ├── arch32/
│       │   ├── matmul_reduce_scatter_v2_tiling.h       :65   arch32 Tiling子类
│       │   ├── matmul_reduce_scatter_v2_tiling.cpp     :      入口分发
│       │   ├── matmul_reduce_scatter_v2_aiv_mode_tiling.cpp :1213 AIV模式Tiling主逻辑
│       │   └── matmul_reduce_scatter_v2_aiv_mode_smallm_tiling.h : SmallM Tiling查表
│       └── arch35/
│           └── quant_bmm_reduce_scatter_tiling.h/cpp   :      量化BMM ReduceScatter
├── op_api/
│   ├── aclnn_matmul_reduce_scatter_v2.h                :76   ACLNN两段式API声明
│   └── aclnn_matmul_reduce_scatter_v2.cpp              :      ACLNN实现
└── op_graph/ / tests/ / examples/
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `matmul_reduce_scatter_v2_def.cpp:18-235`

**输入张量** (6个):

| 索引 | 名称 | 必选 | 910b数据类型 | 950数据类型 | 说明 |
|------|------|------|-------------|-------------|------|
| 0 | x1 | REQUIRED | BF16/FP16/INT8 | BF16/FP16/FP8E4M3/FP8E5M2/HIF8 | 矩阵A [M,K] |
| 1 | x2 | REQUIRED | BF16/FP16/INT8(ND/NZ) | FP8系列 | 矩阵B [K,N] |
| 2 | bias | OPTIONAL | BF16/FP16/FP32 | FP32 | 偏置 |
| 3 | x1_scale | OPTIONAL | FP32 | FP32/FP8E8M0(MX) | A矩阵缩放因子 |
| 4 | x2_scale | OPTIONAL | FP32/INT64 | FP32/FP8E8M0(MX) | B矩阵缩放因子 |
| 5 | quant_scale | OPTIONAL | FP32 | FP32 | 输出量化缩放 |

**输出张量** (2个):

| 索引 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y | BF16/FP16/FP32 | ReduceScatter结果 [M/rankSize, N] |
| 1 | amax_out | FP32 (可选) | FP8量化场景的输出最大值 |

**属性** (12个):

| 序号 | 名称 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| 0 | group | String | REQUIRED | HCCL通信组名 |
| 1 | reduce_op | String | "sum" | 归约操作 |
| 2 | is_trans_a | Bool | false | A矩阵转置 |
| 3 | is_trans_b | Bool | false | B矩阵转置 |
| 4 | comm_turn | Int | 0 | 通信数据切分数 |
| 5 | rank_size | Int | 0 | rank数量 |
| 6 | block_size | Int | 0 | 量化block大小 |
| 7 | group_size | Int | 0 | 反量化group大小 |
| 8 | is_amax_out | Bool | false | 是否输出amax |
| 9 | y_dtype | Int | DT_UNDEFINED | 输出数据类型 |
| 10 | comm_mode | String | "aicpu" | 通信模式(aicpu/aiv) |
| 11 | stream_mode | Int | - | 流模式 |

**平台配置** (`def.cpp:114-230`):
- `ascend950`: 29组数据类型组合(含FP8/HIF8), `opFile=matmul_reduce_scatter_v2_apt`
- `ascend910b`: 10组(BF16/FP16/INT8+ND/NZ), AIV模式通信
- `ascend910_93`: 复用910b配置, is910C=true标志区分

### 3.2 Shape 推导

**文件**: `matmul_reduce_scatter_v2_infershape.cpp:25-59`

```
InferShapeMatmulReduceScatterV2():
  ├── InferMatmulReduceScatterCommon(context)  // 复用公共RS shape推导
  │   └── 输出y: [M/rankSize, N]
  └── amax_out: 根据is_amax_out属性决定shape
      ├── true  → [1]
      └── false → [0]

InferDataTypeMatmulReduceScatterV2():
  ├── FP16/BF16输入 → 输出类型与输入一致
  └── FP8输入 → 使用y_dtype属性指定输出类型
```

### 3.3 Tiling 策略

**Tiling入口**: `arch32/matmul_reduce_scatter_v2_tiling.cpp` → `MatmulReduceScatterTilingV2AivModeFunc()`

**AIV 模式 Tiling 主流程** (`matmul_reduce_scatter_v2_aiv_mode_tiling.cpp:1082-1212`):

```
MatmulReduceScatterTilingV2AivModeFunc():
  ├── 1. SetScheduleMode(batch_mode=1)  // SyncAll需要所有核同时启动
  ├── 2. 校验属性 (group/trans_a/trans_b)
  ├── 3. 获取Shape (M, K, N)
  ├── 4. 获取平台信息 (aivNum, ubSize)
  ├── 5. GetTilingKey(isBias, transA, transB, isSmallM)
  │   └── isSmallM = GetAlgorithmPolicy(M, N)
  │       ├── M<=2048 && N>512 && !is910C && !isQuant && output<180MB → SMALL_M
  │       └── 否则 → LARGE_M
  ├── 6. Tiling参数计算
  │   ├── [SMALL_M] SetTilingData_SmallM()
  │   │   └── 按rankSize=2/4/8查表获取最优m0/swizzlCount/swizzlDirect/pValue/ubMoveNum
  │   └── [LARGE_M] SetTilingData()
  │       └── 编码整数值解码为Tiling参数 (ReduceScatterV2DecodeTilingData)
  ├── 7. Workspace计算
  │   ├── 对齐空间 (A/B矩阵Padding)
  │   └── 反量化空间 (INT32中间结果)
  └── 8. 通信配置
      ├── is910C → ReduceScatter=level0:fullmesh
      └── A2 → MultiPut=level0:fullmesh (Window IPC)
```

**Code-Based Tiling 参数编码** (`aiv_mode_tiling.cpp:703-728`):

```
ReduceScatterV2DecodeTilingData(code):
  code = [M0:1bit][swizzlDirect:1bit][swizzlCount:4bit][pValue:4bit]
         [ubMoveNum:8bit][commDirect:1bit][commNpuSplit:5bit][commDataSplit:5bit]

  m0 = (code&M0_MASK)*128 + 128          // 128 or 256
  k0 = 256
  n0 = m0==128 ? 256 : 128               // 互补
```

**MKN 条件查表** (`aiv_mode_tiling.cpp:52-534`): 为每种架构/卡数/量化组合预定义 MKN 范围到最优 code 的映射表:
- `g_matmulReduceScatterA2FourRankFP16CodeMap`: A2/4卡/FP16, 22个code条目
- `g_matmulReduceScatterA2FourRankINT8CodeMap`: A2/4卡/INT8, 28个code条目
- `g_matmulReduceScatterA2EightRankFP16CodeMap`: A2/8卡/FP16, 31个code条目
- `g_matmulReduceScatterA2EightRankINT8CodeMap`: A2/8卡/INT8, 19个code条目
- `g_matmulReduceScatterA3*`: 910_93平台对应的FP16/INT8×4/8卡映射表

### 3.4 ACLNN API

**文件**: `aclnn_matmul_reduce_scatter_v2.h:51-69`

```c
// 第一段: 计算workspace大小
aclnnMatmulReduceScatterV2GetWorkspaceSize(
    x1, x2, bias, x1Scale, x2Scale, quantScale,
    blockSize, group, reduceOp, commTurn, streamMode, groupSize, commMode,
    output, amaxOutOptional,
    &workspaceSize, &executor);

// 第二段: 执行
aclnnMatmulReduceScatterV2(workspace, workspaceSize, executor, stream);
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `matmul_reduce_scatter_v2.cpp:31-56`

```cpp
template<bool TPL_ISBIAS, bool TPL_IS_TRANSPOSE_A, bool TPL_IS_TRANSPOSE_B, bool TPL_IS_SMALLM>
__global__ __aicore__ void matmul_reduce_scatter_v2(
    GM_ADDR aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, quantScaleGM,
    cGM, amaxOutGM, workspaceGM, tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 1AIC+2AIV混合核
    if constexpr (!TPL_IS_SMALLM) {
        INVOKE(MatmulReduceScatterAivMode, FORMAT_X2==NZ, false, TPL_TRANS_B, void);
    } else {
        INVOKE(MatmulReduceScatterAivModeSmallM, FORMAT_X2==NZ, false, TPL_TRANS_B);
    }
}
```

关键设计:
- `KERNEL_TYPE_MIX_AIC_1_2`: 每个物理核包含1个AIC(Cube)和2个AIV(Vector)子核
- `FORMAT_X2 == FORMAT_FRACTAL_NZ`: B矩阵格式通过编译期常量传入
- `Derived=void`: 普通M场景使用静态分发, SmallM使用CRTP

### 4.2 MatmulReduceScatterAivMode 主类

**文件**: `matmul_reduce_scatter_aiv_mode.h:58-548`

**类继承关系**:
```
CommBase                              ← Window IPC通信基类
  └── MatmulReduceScatterAivMode      ← 普通M场景
        └── MatmulReduceScatterAivModeSmallM  ← SmallM场景(CRTP)
```

**模板参数** (8个):
```
typename AType, typename BType, typename biasType,
typename x2ScaleType, typename cType,
bool weight_nz,    // B矩阵NZ格式
bool TA,           // A矩阵转置
bool TB            // B矩阵转置
```

#### Init 流程 (`aiv_mode.h:109-141`)

```
Init(aGM, bGM, biasGM, perTokenScale, perChannelScale, cGM, workspaceGM, tilingGM):
  ├── GET_TILING_DATA(tilingData, tilingGM)
  ├── CommBase::SetArgs(tilingData)
  │   ├── 从HCCL Context读取rank/rankSize
  │   ├── 初始化buff[8] Window IPC地址数组
  │   ├── 计算 m_loop = CeilDiv(m/rankSize, m0) * rankSize
  │   ├── 计算 n_loop, k_loop
  │   └── UB缓冲区分配 (160KB)
  ├── Padding地址映射:
  │   ├── hasAAlign → workspaceGM用于A对齐
  │   ├── hasBAlign → workspaceGM+aAlignSize用于B对齐
  │   └── quantFlag → workspaceGM+a+b用于INT32累加器
  ├── m_align = Block512B<AType>::AlignUp(m)
  ├── k_align, n_align 同理
  ├── AICInit(): 设置Cube初始状态
  │   ├── SetLoadDataPaddingValue(0)
  │   ├── SetFixpipeNz2ndFlag(1,0,0)
  │   └── gm_peer_mem = buff[rank]
  └── AIVInit(): 设置Vector初始状态
      ├── max_ub_ping_pong_size = max_ub_single_dma_size / 2
      ├── core_loop = m_loop * n_loop
      ├── cal_count = CeilDiv(core_loop, loop_num_per_comm)
      └── pingpong_idx = 0
```

#### CatlassMatmul 流程 (`aiv_mode.h:172-324`)

仅在 AIC (Cube) 子核上执行，使用 Catlass 模板线性代数库:

```
CatlassMatmul():  [仅AIC执行]
  ├── quantFlag判断: ElementC = quantFlag ? int32_t : cType
  ├── L1/L0 Tile Shape选择:
  │   ├── quantFlag → L1_K=512, L0_K=128
  │   └── !quantFlag → L1_K=256, L0_K=64
  ├── DispatchPolicy = MmadAtlasA2Preload<false, true>  (启用Shuffle-K)
  ├── B矩阵布局选择:
  │   ├── weight_nz=true → LayoutNZ(nZ/zN)
  │   └── weight_nz=false → LayoutB(RowMajor/ColumnMajor)
  ├── m0分块选择:
  │   ├── m0==128 → GemmShape<128, 256, L1K>
  │   └── m0==256 → GemmShape<256, 128, L1K>
  └── 实例化并执行 MatmulReduceScatterAivMode kernel
      └── 参数: processSize, gm_a_src, gm_b_src, cGM_,
               perChannelScaleGM_, gm_peer_mem, gm_accum,
               p_value, swizzl_count, swizzl_direct, dequant_type,
               rank, rank_size, need_fixpipe
```

**Catlass kernel 内部行为**:
- AIC 计算 `C_full[m,n] = A[m,k] × B[k,n]`
- 结果直接写到 Window IPC 的 peer memory: 每个 m0×n0 块根据目标 rank 写到对应位置
- 目标 rank == 本地 rank → 写到本地输出 y
- 目标 rank != 本地 rank → 写到 `gm_peer_mem` 供其他卡读取
- INT8×INT8 场景: 支持 Fixpipe 硬件级 `int32 × perChannelScale → FP16` 转换

#### AIVComm 流程 (`aiv_mode.h:487-535`)

仅在 AIV (Vector) 子核上执行:

```
AIVComm():  [仅AIV执行]
  ├── Padding(): A/B矩阵对齐到512B边界 (Catlass PaddingMatrix)
  │   └── AIV执行→CrossCoreSetFlag通知AIC
  ├── ResetIpcFlags(2): 清零rank同步标志
  ├── PipeBarrier<PIPE_ALL>: 全核同步
  │
  └── for cal_idx = 0..cal_count:
      ├── flag_idx = cal_idx % MAX_BLOCK_COUNT  (双缓冲,MAX_BLOCK_COUNT=2)
      ├── len_per_rank = actual_loop_num * tile_block_size / rank_size
      │
      ├── WaitEvent(flag_idx)          ← 等待AIC计算完成
      ├── SetAndWaitAivSync(flag_idx)  ← AIV子核间同步
      ├── Dequant(cal_idx, flag_idx)   ← 反量化(PerChannel/PerToken)
      │   ├── needPerChannel = quantFlag && !(isX2ScaleTypeInt64 && cType==FP16)
      │   └── needPerToken = quantFlag && dequant_type==PER_TOKEN
      ├── CrossRankSyncV1(FLAG_ZERO_IDX, cal_idx+1)  ← 跨rank同步(Flag写入)
      ├── SetAndWaitAivSync(flag_idx)
      │
      ├── [aiv_idx==0 && core_idx < comm_npu_split*comm_data_split]:
      │   ├── StartBeforeFirstStep(): SetAtomicAdd<cType>
      │   ├── data_loops循环:
      │   │   └── FirstStepInOutWithSplit():
      │   │       └── 3层循环(rank_per_core × data_loops):
      │   │           └── FirstStepInOut():
      │   │               ├── GM→UB Ping-Pong读取其他rank的peer memory
      │   │               └── UB→GM 写到本地输出(AtomicAdd累加)
      │   └── EndFirstStep(): SetAtomicNone
      │
      └── [cal_idx < cal_count - MAX_BLOCK_COUNT]:
          ├── CrossRankSyncV2(FLAG_ONE_IDX, cal_idx+1)
          └── SetAicSync(flag_idx)  ← 通知AIC下一轮
```

### 4.3 MatmulReduceScatterAivModeSmallM 派生类

**文件**: `matmul_reduce_scatter_aiv_mode_smallM.h:56-658`

使用 CRTP (Curiously Recurring Template Pattern) 继承基类，覆盖 `CatlassMatmul()` 和 `AIVComm()`:

```
MatmulReduceScatterAivModeSmallM<AType,BType,biasType,x2ScaleType,cType,weight_nz,TA,TB>
    : public MatmulReduceScatterAivMode<..., MatmulReduceScatterAivModeSmallM<...>>
```

**核心差异**:

| 特性 | 普通M (基类) | SmallM (派生) |
|------|-------------|--------------|
| 通信方式 | 跨rank读peer memory + AtomicAdd | 两阶段: Notify远端 + 本地Reduce累加 |
| 输出初始化 | 无需(直接写) | InitOutputMatrixCToZero() 全部置0 |
| 列分割 | 无 | CommColumnSplitter按列分批通信 |
| 双缓冲 | MAX_BLOCK_COUNT=2 | MAX_BLOCK_COUNT_SM=4 |
| 反量化 | DequantRunner统一处理 | BlockEpilogue逐块处理 |
| Catlass kernel | MatmulReduceScatterAivMode | MatmulReduceScatterAivModeSmallM |

**SmallM AIVComm 流程** (`smallM.h:620-656`):

```
AIVComm():  [仅AIV执行]
  ├── Padding()
  ├── ResetIpcFlags(rank_size)
  ├── InitOutputMatrixCToZero(): 输出矩阵C全部初始化为0
  ├── CheckBuffFlag: 确认其他卡准备就绪
  ├── SyncAll<true>
  ├── CommColumnSplitter(loop_num_per_comm, m_loop, swizzl_count, n_loop)
  │   └── 按列块索引分配每轮通信的n范围 [n_st, n_ed)
  │
  ├── [aiv_idx==1]:  ← 第二个AIV子核负责通知
  │   ├── NotifyRemoteDataCopy(taskSplit):
  │   │   └── for cal_idx:
  │   │       ├── WaitEvent(flag_idx) ← 等AIC
  │   │       ├── Dequant(...)         ← 反量化
  │   │       ├── SyncAll<true>
  │   │       └── SetBuffFlag(buff[core_idx]+rank, cal_idx+1) ← 通知其他卡数据就绪
  │   ├── SyncAll<true>
  │   └── ComfirmRemoteCopyDone():
  │       └── CheckBuffFlag(buff[rank]+core_idx, cal_count+1) ← 等远端读完
  │
  └── [aiv_idx==0]:  ← 第一个AIV子核负责搬运
      ├── CompleteRemoteDataCopy(taskSplit):
      │   └── for cal_idx:
      │       ├── Dequant(...)
      │       ├── SyncAll<true>
      │       ├── StartBeforeFirstStep(): SetAtomicAdd
      │       ├── loop_copy_count = CeilDiv(m_per_rank, n_repeat)
      │       ├── 按core_idx分配copy任务:
      │       │   └── for rank_idx in rank_size:
      │       │       ├── CheckBuffFlagV2: 首次确保远端数据就绪
      │       │       ├── CopyGmToUbuf: 从远端buff[target_rank]读入UB
      │       │       └── CopyUbufToGmUnknown: 写到本地C (AtomicAdd累加)
      │       └── EndFirstStep()
      ├── SyncAll<true>
      └── SendCopyCompleteSignal(): SetBuffFlag通知远端
```

### 4.4 CommBase 通信基类

**文件**: `matmul_reduce_scatter_aiv_mode_util.h:138-458`

**Window IPC 初始化** (`util.h:142-188`):
```
SetArgs(tilingData):
  ├── HCCL Context获取 (区分910C和A2结构体)
  │   ├── 910C: HcclOpResParam → localUsrRankId, remoteRes[].windowsIn
  │   └── A2: HcclA2CombineOpParam → rankId, windowsIn[rankSize]
  ├── buff[8]数组初始化: 每个rank的Window IPC地址
  │   ├── buff[rank] = 本地Window In地址 (自己的peer memory)
  │   └── buff[other] = 远端Window In地址 (其他卡的peer memory)
  ├── 循环参数: m_loop, k_loop, n_loop
  ├── loop_num_per_comm = p_value * core_num
  └── UB缓冲区: TBuf<VECCALC>, 160KB
```

**Flag同步机制** (`util.h:250-305`):
```
SetBuffFlag(buff, flag):
  ├── ubTensor(0) = flag + RAND_BASE  (加随机基值防止误触发)
  └── CopyUbufToGmAlignB16(buff, ubTensor, 1, 4B)

CheckBuffFlag(buff, flag):
  └── while(true):
      ├── CopyGmToUbufAlignB16(ubTensor, buff)
      ├── SetFlag<HardEvent::MTE2_S>
      ├── WaitFlag<HardEvent::MTE2_S>
      └── if (ubTensor(0) == flag + RAND_BASE) break

CrossRankSyncV1(flag_idx, flag_data):  // rank0广播到所有rank
  ├── [core_idx==rank]: SetBuffFlagByAdd (原子加)
  └── [core_idx<rank_size]: CheckBuffFlag (等待flag_data * FLAG_VALUE)

CrossRankSyncV2(flag_idx, flag_data):  // 所有rank向rank0汇报
  ├── [core_idx<rank_size]: SetBuffFlagByAdd
  └── [core_idx==rank]: CheckBuffFlag(等待rank_size * flag_data)
```

**Block Swizzle** (`util.h:103-136`):

```
GetSwizzledBlockIdx(loop_idx, m_loop, n_loop, swizzl_direction, swizzl_count):
  ├── swizzl_direction==0 (Zn模式):
  │   ├── tile_block_loop = CeilDiv(m_loop, swizzl_count)
  │   ├── 每 swizzl_count 行一组, 组间交替正反序
  │   └── L2 Cache局部性优化: 相邻block在相邻行
  └── swizzl_direction==1 (Nz模式):
      ├── tile_block_loop = CeilDiv(n_loop, swizzl_count)
      ├── 每 swizzl_count 列一组, 组间交替正反序
      └── 适用于N维度较大的场景
```

### 4.5 反量化 DequantRunner

**文件**: `matmul_reduce_scatter_aiv_mode_dequant.h:38-156`

使用 Catlass Epilogue 子系统进行高效反量化:

```
DequantRunner<OutputType>::RunMatmulReduceScatter(...):
  ├── Epilogue配置:
  │   ├── TileShape: <64, 128>
  │   ├── TileRowBroadcastMul: PerChannel scale (行广播)
  │   ├── TileOneBlkColumnBroadcastMul: PerToken scale (列广播)
  │   └── BlockEpilogue: 组合以上Tile操作
  ├── 遍历 p_value * core_num 个循环:
  │   ├── dstRankIdx = loopIdx % rankSize  (目标rank)
  │   ├── GetBlockIdCoord → GetBlockLocCoord → GetBlockSizeCoord
  │   ├── [dstRankIdx==rankIdx]: 写到本地输出y
  │   └── [dstRankIdx!=rankIdx]: 写到peer memory
  └── 反量化模式:
      ├── PerChannel+PerToken: int32 × x2Scale × x1Scale → FP16/BF16
      ├── PerChannel only: int32 × x2Scale → FP16/BF16
      └── PerToken only: peerMem × x1Scale → FP16/BF16
```

### 4.6 Padding Runner

**文件**: `matmul_reduce_scatter_aiv_mode_padding.h:29-198`

```
PaddingRunner<InputType, WeightType>::Run(...):
  ├── 根据 transA × transB 的4种组合选择Layout
  ├── TemplatePadder (Catlass PaddingMatrix):
  │   ├── COMPUTE_LENGTH_A = 96KB / sizeof(ElementA)
  │   └── COMPUTE_LENGTH_B = 96KB / sizeof(ElementB)
  ├── 仅AIV子核执行 (template specialization <AIV>)
  │   ├── alignA: 将A矩阵Padding到512B对齐
  │   ├── alignB: 将B矩阵Padding到512B对齐
  │   └── CrossCoreSetFlag: 通知AIC Padding完成
  └── 无需Padding时: 直接设置flag跳过
```

### 4.7 多核并行

```
物理核结构 (KERNEL_TYPE_MIX_AIC_1_2):
┌──────────────────────────────────┐
│  物理核 block_id                  │
│  ┌────────────┐  ┌────────────┐  │
│  │ AIC (Cube) │  │ AIV-0 (Vec)│  │
│  │            │  │            │  │
│  │ CatlassMat │  │ Complete   │  │
│  │ mul()      │  │ RemoteCopy │  │
│  └────────────┘  ├────────────┤  │
│                  │ AIV-1 (Vec)│  │
│                  │            │  │
│                  │ Notify     │  │
│                  │ RemoteCopy │  │
│                  └────────────┘  │
└──────────────────────────────────┘

core_idx = block_id / GetTaskRation()
aiv_idx = GetSubBlockIdx()  // 0 或 1

任务分配:
- AIC: 执行CatlassMatmul(), 结果写到peer memory
- AIV-0: CompleteRemoteDataCopy() - 从远端读数据并AtomicAdd到本地C
- AIV-1: NotifyRemoteDataCopy() - 反量化并通知其他rank数据就绪
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据传递

```
Host侧 (TilingContext):
  MatmulReduceScatterV2AivModeTilingData {
      Mc2InitTiling mc2InitTiling;           ← HCCL初始化参数
      Mc2CcTiling mc2CcTiling;               ← HCCL通信参数
      MatmulReduceScatterV2AivModeInfo info;  ← 矩阵shape/平台信息
      CoCTiling cocTiling;                    ← 计算通信Tiling参数
  }
        ↓ tilingGM
Kernel侧:
  GET_TILING_DATA_WITH_STRUCT → 解析为相同结构体
  REGISTER_TILING_DEFAULT → 注册默认Tiling解析器
```

### 5.2 Workspace 布局

```
workspaceGM:
├── [0, aAlignSize)              ← A矩阵Padding空间 (如有)
├── [aAlignSize, aAlignSize+bAlignSize) ← B矩阵Padding空间 (如有)
└── [aAlignSize+bAlignSize, ...)  ← INT32累加器空间 (量化场景)

Workspace大小计算 (GetUsrWorkSpaceSize):
  ├── hasAAlign: M*K_elem*elemSize (对齐后)
  ├── hasBAlign: K*N_elem*elemSize (对齐后)
  ├── quantFlag+SmallM: M_align * N_align * sizeof(int32_t)
  ├── quantFlag+LargeM: pValue * numBlocks * m0 * n0 * 2 * sizeof(int32_t)
  └── 总计 = SYSTEM_NEED_WORKSPACE(16MB) + userWorkSpaceSize
```

### 5.3 HCCL Context

```
A2平台 (HcclA2CombineOpParam):
  ├── rankId: 本卡rank编号
  ├── rankNum: 总rank数
  └── windowsIn[rankNum]: 各rank的Window IPC地址

910C平台 (HcclOpResParam):
  ├── localUsrRankId: 本卡rank编号
  ├── rankSize: 总rank数
  ├── localWindowsIn: 本地Window地址
  └── remoteRes[i].nextDevicePtr → HcclRankRelationResV2.windowsIn
```

---

## 6. 数据流

### 6.1 普通 M 场景数据流

```
                    Rank 0                         Rank 1                    Rank R
               ┌──────────────┐              ┌──────────────┐         ┌──────────────┐
 A₀,B₀ ──→    │ AIC: MM计算   │              │ AIC: MM计算   │         │ AIC: MM计算   │
               │ C[m,n]=A×B   │              │ C[m,n]=A×B   │         │ C[m,n]=A×B   │
               └──────┬───────┘              └──────┬───────┘         └──────┬───────┘
                      │                             │                        │
               ┌──────▼───────┐              ┌──────▼───────┐         ┌──────▼───────┐
               │AIC写peer_mem │              │AIC写peer_mem │         │AIC写peer_mem │
               │按dest rank   │              │按dest rank   │         │按dest rank   │
               │分块写入      │              │分块写入      │         │分块写入      │
               └──────┬───────┘              └──────┬───────┘         └──────┬───────┘
                      │                             │                        │
          ┌───────────┼─────────────────────────────┼────────────────────────┤
          │     Window IPC Peer Memory (180MB/rank)  │                        │
          │  buff[0]    buff[1]    ...    buff[R]     │                        │
          └───────────┼─────────────────────────────┼────────────────────────┘
                      │  Flag同步: CrossRankSyncV1/V2 │
               ┌──────▼───────┐              ┌──────▼───────┐         ┌──────▼───────┐
               │AIV: Dequant  │              │AIV: Dequant  │         │AIV: Dequant  │
               │AIV: 读远端   │              │AIV: 读远端   │         │AIV: 读远端   │
               │  peer_mem    │              │  peer_mem    │         │  peer_mem    │
               │  AtomicAdd   │              │  AtomicAdd   │         │  AtomicAdd   │
               │  →本地C      │              │  →本地C      │         │  →本地C      │
               └──────┬───────┘              └──────┬───────┘         └──────┬───────┘
                      │                             │                        │
               ┌──────▼───────┐              ┌──────▼───────┐         ┌──────▼───────┐
  y₀ ←──      │y[M/R, N]     │     y₁ ←──  │y[M/R, N]     │  y_R←─ │y[M/R, N]     │
               └──────────────┘              └──────────────┘         └──────────────┘
```

### 6.2 SmallM 场景数据流

```
Rank 0:                              Rank R:
┌─────────────────────┐              ┌─────────────────────┐
│ AIC: CatlassMatmul  │              │ AIC: CatlassMatmul  │
│ → 写到buff[rank]     │              │ → 写到buff[rank]     │
└─────────┬───────────┘              └─────────┬───────────┘
          │                                    │
┌─────────▼───────────┐              ┌─────────▼───────────┐
│ AIV-1: NotifyRemote │  SetBuffFlag │ AIV-1: NotifyRemote │
│  Dequant + Flag     │────────────→│  Dequant + Flag     │
└─────────┬───────────┘              └─────────┬───────────┘
          │                                    │
┌─────────▼───────────┐              ┌─────────▼───────────┐
│ AIV-0: Init C=0     │              │ AIV-0: Init C=0     │
│ CompleteRemoteCopy:  │              │ CompleteRemoteCopy:  │
│  for rank:           │              │  for rank:           │
│   CheckBuffFlagV2    │              │   CheckBuffFlagV2    │
│   读buff[target_rank]│◄─────────────│   读buff[target_rank]│
│   AtomicAdd → C      │              │   AtomicAdd → C      │
└─────────┬───────────┘              └─────────┬───────────┘
          │ SendCopyCompleteSignal              │
┌─────────▼───────────┐              ┌─────────▼───────────┐
│ ComfirmRemoteCopyDone│              │ ComfirmRemoteCopyDone│
│ → y[M/R, N]         │              │ → y[M/R, N]         │
└─────────────────────┘              └─────────────────────┘
```

### 6.3 内存层次使用

```
┌─────────────────────────────────────────────────┐
│ GM (HBM)                                         │
│ ├── aGM: A矩阵 [M, K]                           │
│ ├── bGM: B矩阵 [K, N]                           │
│ ├── workspaceGM: Padding空间 + INT32累加器       │
│ ├── cGM: 输出y [M/rankSize, N]                   │
│ └── buff[0..R]: Window IPC peer memory (180MB)   │
│                                                   │
│ ┌── L1 (~1MB) ──────────────────────────────┐    │
│ │ ├── A分片 [m0, L1K]                       │    │
│ │ └── B分片 [L1K, n0]                       │    │
│ │   (Preload: 预取下一块到L1)               │    │
│ └────────────────────────────────────────────┘    │
│                                                   │
│ ┌── L0 (Cube局部) ────────────────────────┐      │
│ │ ├── L0A [m0, L0K]                       │      │
│ │ ├── L0B [L0K, n0]                       │      │
│ │ └── L0C [m0, n0] → int32 或 cType       │      │
│ └──────────────────────────────────────────┘      │
│                                                   │
│ ┌── UB (~160KB, Ping-Pong) ──────────────┐       │
│ │ ├── Half-0 (80KB): DMA读缓冲            │       │
│ │ └── Half-1 (80KB): DMA写缓冲            │       │
│ │   (Flag同步也使用UB存放临时值)           │       │
│ └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

---

## 7. 性能设计

### 7.1 计算通信重叠

```
时间线:
AIC:  [----MM Round 0----][----MM Round 1----][----MM Round 2----]
AIV:                      [--Comm 0--Dequant--][--Comm 1--Dequant--]
      ↑                   ↑                   ↑
      Start               Round 0完成          Round 1完成
      AIC先启动           AIV开始通信          AIC继续计算
```

- `loop_num_per_comm = p_value * core_num`: 每轮通信包含的计算块数
- `MAX_BLOCK_COUNT = 2`: 双缓冲深度, AIC计算第N+2轮时AIV处理第N轮
- `cal_count = CeilDiv(core_loop, loop_num_per_comm)`: 总通信轮数

### 7.2 L2 Cache 优化 (Block Swizzle)

```
普通顺序:                    Swizzle优化:
Block 0: row=0, col=0        Block 0: row=0, col=0
Block 1: row=1, col=0        Block 1: row=1, col=0
Block 2: row=2, col=0        Block 2: row=2, col=0
Block 3: row=3, col=0        Block 3: row=0, col=1 (反向)
Block 4: row=0, col=1        Block 4: row=1, col=1
...                          ...

效果: 相邻block在L2 Cache中空间局部性更好
参数: swizzl_direct(0=Zn, 1=Nz), swizzl_count(分组大小)
```

### 7.3 Shuffle-K 优化

Catlass `MmadAtlasA2Preload` 的 `ENABLE_SHUFFLE_K=true`:
- K维度分块时, A矩阵的L0A块通过 Cube 内部 Shuffle 指令在不同 L0 间传递
- 减少从L1到L0的数据搬运次数, 提升Cube利用率

### 7.4 UB Ping-Pong 双缓冲

```
AIV通信中的DMA搬运:
  UB[0..HALF) ←── 读远端peer memory (Ping)
  UB[HALF..END) ←── 读远端peer memory (Pong)
  交替使用 EVENT_ID0 / EVENT_ID1 实现DMA流水线
  max_ub_ping_pong_size = max_ub_single_dma_size / 2
```

---

## 8. 依赖关系

### 8.1 内部依赖

```
matmul_reduce_scatter_v2
├── mc2/common/op_kernel/
│   ├── mc2_tiling_struct.h       ← MC2ServerCfg, MC2HcommCfg, Mc2Msg
│   ├── mc2_common_def.h          ← 通信算法枚举
│   └── moe_distribute_base.h     ← HCCL信号/流结构体
├── mc2/common/op_host/
│   ├── mc2_tiling_utils.h        ← GetRankSize等工具函数
│   ├── matmul_formulaic_tiling.h ← 公式化Tiling
│   └── mc2_common_infershape.h   ← 公共RS shape推导
└── mc2/3rd/template_linear_algebra/  ← Catlass模板库
    ├── arch/arch.hpp
    ├── layout/layout.hpp
    ├── gemm/block/block_mmad.hpp
    ├── gemm/block/block_swizzle.hpp
    ├── gemm/dispatch_policy.hpp
    ├── gemm/kernel/padding_matmul.hpp
    └── epilogue/block/block_epilogue.hpp
```

### 8.2 外部依赖

- **HCCL**: Window IPC共享内存通信 (buff[rankSize]地址数组)
- **AscendC Runtime**: KERNEL_TYPE_MIX_AIC_1_2 核类型, GetHcclContext<0>
- **Catlass**: 模板线性代数库 (vendored in 3rd/)

---

## 9. 函数调用关系总览

```
matmul_reduce_scatter_v2 (内核入口)
├── KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
├── REGISTER_TILING_DEFAULT(MatmulReduceScatterV2AivModeTilingData)
│
├── [!TPL_IS_SMALLM] MatmulReduceScatterAivMode
│   ├── Init()
│   │   ├── CommBase::SetArgs()
│   │   │   ├── GetHcclContext<0>() → Window IPC地址
│   │   │   ├── m_loop/n_loop/k_loop计算
│   │   │   └── TPipe::InitBuffer(uBuf_, 160KB)
│   │   ├── AICInit(): Cube初始化
│   │   └── AIVInit(): Vector初始化
│   │
│   └── Process()
│       ├── CatlassMatmul()  [AIC]
│       │   ├── GemmShape选择 (m0×n0×L1K)
│       │   ├── Layout选择 (NZ/ND, Trans/NoTrans)
│       │   └── MatmulReduceScatterAivMode kernel()
│       │       └── 分块MM → 按dest rank写到peer memory
│       │
│       └── AIVComm()  [AIV]
│           ├── Padding() → Catlass PaddingMatrix
│           ├── ResetIpcFlags(2)
│           ├── for cal_idx:
│           │   ├── WaitEvent(cal_idx%2) ← AIC完成
│           │   ├── Dequant() ← PerChannel/PerToken
│           │   ├── CrossRankSyncV1() ← Flag同步
│           │   ├── FirstStepInOutWithSplit() ← 读远端+AtomicAdd
│           │   └── CrossRankSyncV2() + SetAicSync()
│           └── ResetIpcFlags(1) + CheckBuffFlag
│
└── [TPL_IS_SMALLM] MatmulReduceScatterAivModeSmallM
    ├── Init(): 继承基类, m_loop额外计算
    ├── InitOutputMatrixCToZero(): 全部置0
    ├── CatlassMatmul()  [AIC]
    │   └── MatmulReduceScatterAivModeSmallM kernel()
    │       └── 结果直接写peer memory (不区分dest rank)
    │
    └── AIVComm()  [AIV]
        ├── [aiv_idx==1]: NotifyRemoteDataCopy
        │   ├── for cal_idx:
        │   │   ├── WaitEvent
        │   │   ├── Dequant (BlockEpilogue)
        │   │   └── SetBuffFlag ← 通知数据就绪
        │   └── ComfirmRemoteCopyDone ← 等远端读完
        │
        └── [aiv_idx==0]: CompleteRemoteDataCopy
            ├── for cal_idx:
            │   ├── for rank_idx in rank_size:
            │   │   ├── CheckBuffFlagV2 ← 等远端就绪
            │   │   ├── CopyGmToUbuf ← 读远端peer memory
            │   │   └── CopyUbufToGmUnknown ← AtomicAdd到C
            │   └── EndFirstStep()
            └── SendCopyCompleteSignal()
```

---

*分析完成时间: 2026-04-20*
