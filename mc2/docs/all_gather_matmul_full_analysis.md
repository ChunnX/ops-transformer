# all_gather_matmul 算子全面分析报告

## 1. 算子概述

AllGather+MatMul 融合算子。先执行 AllGather 收集所有 rank 的 A 矩阵分片，再在本地执行 MatMul。用于 Tensor Parallel 的列并行线性层前向传播。

**融合原理**: AllGather 分片接收 → 立即开始本卡 MatMul → 逐片 Wait 远端数据 → 计算远端分片。通信和计算流水线重叠。

**支持平台**: ascend910b (arch32), ascend910_93 (arch35), ascend950

## 2. 目录结构与文件职责

```
all_gather_matmul/
├── op_host/
│   ├── all_gather_matmul_def.cpp          (第18-75行) 算子定义
│   ├── all_gather_matmul_infershape.cpp   (第23-39行) Shape 推导
│   └── op_tiling/
│       ├── all_gather_formulaic_tiling.h     公式化 Tiling 头文件
│       └── all_gather_formulaic_tiling.cpp   公式化 Tiling 实现
├── op_kernel/
│   ├── all_gather_matmul.cpp              Kernel 入口
│   ├── all_gather_matmul_tiling.h         (第28-47行) Tiling 数据结构
│   ├── all_gather_matmul_tiling_key.h     (第23-50行) TilingKey 模板参数
│   ├── all_gather_matmul_base.h           (第33-155行) 基类
│   └── all_gather_matmul_full_mesh.h      (第22-229行) FullMesh 实现
├── op_api/
│   ├── aclnn_all_gather_matmul.h          ACLNN API 声明
│   └── aclnn_all_gather_matmul.cpp        ACLNN API 实现
└── op_graph/
    ├── all_gather_matmul_proto.h          图模式原型
    └── all_gather_matmul_gen_task.cpp     GenTask
```

## 3. Host 侧分析

### 3.1 算子定义 (op_def)

**文件**: `all_gather_matmul_def.cpp`

| 输入/输出 | 名称 | 数据类型 | 格式 | 属性 |
|----------|------|---------|------|------|
| 输入 | x1 | FP16, BF16 | ND | REQUIRED, A 矩阵 |
| 输入 | x2 | FP16, BF16 | ND | REQUIRED, B 矩阵, IgnoreContiguous |
| 输入 | bias | FP16, BF16 | ND | OPTIONAL |
| 输出 | y | FP16, BF16 | ND | REQUIRED, MatMul 结果 |
| 输出 | gather_out | FP16, BF16 | ND | REQUIRED, AllGather 结果 |

**属性**:

| 属性名 | 类型 | 默认值 | 含义 |
|--------|------|--------|------|
| group | String (必选) | - | HCCL 通信域 |
| is_trans_a | Bool | false | A 矩阵是否转置 |
| is_trans_b | Bool | false | B 矩阵是否转置 |
| gather_index | Int | 0 | 0: gather A 矩阵, 1: gather B 矩阵 |
| comm_turn | Int | 0 | 通信分片数 (0=自动) |
| rank_size | Int | 0 | rank 数量 (0=自动) |
| is_gather_out | Bool | true | 是否输出 gather 结果 |

**MC2 配置**: `MC2().HcclGroup("group")` — 使用 `group` 属性作为 HCCL 通信域。

### 3.2 Shape 推导

**文件**: `all_gather_matmul_infershape.cpp` (第23-38行)

调用 `AllGatherMatmulCommonInferShape(context, GATHER_OUT_V1)`，复用公共 infershape 逻辑。输出数据类型与输入一致 (`context->GetInputDataType(0)`)。

### 3.3 ACLNN API

**文件**: `aclnn_all_gather_matmul.h` (第41-56行)

两段式 API:
```cpp
// 第一段: 计算 workspace
aclnnAllGatherMatmulGetWorkspaceSize(x1, x2, bias, group, gatherIndex,
                                      commTurn, streamMode, output, gatherOut,
                                      &workspaceSize, &executor);
// 第二段: 执行
aclnnAllGatherMatmul(workspace, workspaceSize, executor, stream);
```

## 4. Kernel 侧分析

### 4.1 Tiling 数据结构

**文件**: `all_gather_matmul_tiling.h` (第28-47行)

```c
struct AllGatherSoc {
    uint32_t commAlg;    // 通信算法: FULL_MESH / DOUBLE_RING 等
    uint32_t isA3;       // 是否 A3 架构
    uint32_t isStep;     // 是否分步模式
    uint32_t isND2NZ;    // 是否需要 ND→NZ 格式转换
};

class AllGatherMatmulTilingData {
    Mc2InitTiling mc2InitTiling;       // HCCL 初始化参数
    Mc2CcTiling mc2CcTiling;           // HCCL CC Tiling
    TCubeTiling tileTiling;            // 主分片 MatMul Tiling
    TCubeTiling tailTiling;            // 尾块 MatMul Tiling
    TCubeTiling localTiling;           // 本卡数据 MatMul Tiling
    TileL2Tiling tileL2Tiling;         // 主分片 L2 Cache Tiling
    TileL2Tiling tailL2Tiling;         // 尾块 L2 Cache Tiling
    TileL2Tiling localL2Tiling;        // 本卡 L2 Cache Tiling
    RCSTiling param;                   // 公共参数 (rankM/K/N, tileCnt 等)
    AllGatherSoc socParam;             // SoC 特定参数
};
```

**RCSTiling 核心字段** (定义于 `mc2_tiling_struct.h`):
- `rankM, rankK, rankN`: 全局矩阵维度
- `tileCnt, tailCnt`: 主分片和尾块数量
- `dataType, subtype`: HCCL 通信数据类型和规约操作
- `cToFloatLen, nd2NzWorkLen, biasLen, gatherLen`: Workspace 子区大小

### 4.2 TilingKey 编码

**文件**: `all_gather_matmul_tiling_key.h` (第23-50行)

三个布尔模板参数:

| 参数 | 值 0 | 值 1 | 含义 |
|------|------|------|------|
| `ALL_GATHER_MM_FULL_MESH` | 保留 | FullMesh | 是否使用 FullMesh 通信 |
| `ALL_GATHER_MM_ND2NZ_OPT` | 关闭 | 开启 | B 矩阵是否做 ND→NZ |
| `ALL_GATHER_MM_BIAS_CAST` | 关闭 | 开启 | Bias 是否做 BF16→FP32 |

**合法组合** (仅 FullMesh):
- (1,1,0): FullMesh + ND2NZ + 无 Bias Cast
- (1,0,0): FullMesh + 无 ND2NZ + 无 Bias Cast
- (1,1,1): FullMesh + ND2NZ + Bias Cast
- (1,0,1): FullMesh + 无 ND2NZ + Bias Cast

### 4.3 Kernel 入口

**文件**: `all_gather_matmul.cpp` (第45-82行)

```cpp
template<bool IsFullMesh, bool IsNd2Nz, bool IsBias>
__global__ __aicore__ void all_gather_matmul(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
```

**入口流程**:
1. `REGISTER_TILING_DEFAULT(AllGatherMatmulTilingData)` (第48行)
2. 提取 `mc2InitTiling` 和 `mc2CcTiling` 子结构 (第52-53行)
3. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` (第56行) — AIC+AIV 混合核
4. 根据 `IsNd2Nz` 和 `IsBias` 选择 B 矩阵格式和 Bias 类型:
   - `IsNd2Nz=true`: `bType = MatmulType<GM, NZ, B_DTYPE, true>`
   - `IsNd2Nz=false`: `bType = MatmulType<GM, ND, B_DTYPE, true>`
   - `IsBias=true`: `biasType = MatmulType<GM, ND, float>` (BF16→FP32)
   - `IsBias=false`: `biasType = MatmulType<GM, ND, BiasType<BIAS_DTYPE>::type>`
5. `INVOKE_ALL_GATHER_MATMUL_OP_IMPL(AllGatherMatmulFullMesh, IsNd2Nz, IsBias)` (第64-79行)

### 4.4 基类 AllGatherMatmulBase

**文件**: `all_gather_matmul_base.h` (第33-155行)

```
模板参数:
  A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE  — MatmulType<> 类型包装
  BNd2Nz     — 是否 B 矩阵做 ND→NZ
  Bias2Float — 是否 Bias 做 BF16→FP32
```

**成员变量**:

| 变量 | 类型 | 含义 |
|------|------|------|
| `aGM_` | GM_ADDR | A 矩阵地址 |
| `bGM_` | GM_ADDR | B 矩阵地址 |
| `biasGM_` | GM_ADDR | Bias 地址 |
| `cGM_` | GM_ADDR | C 矩阵 (输出) 地址 |
| `gatherGM_` | GM_ADDR | AllGather 输出地址 |
| `workspaceGM_` | GM_ADDR | Workspace 地址 |
| `tilingData_` | AllGatherMatmulTilingData* | Tiling 数据指针 |
| `tPipe_` | TPipe* | 管线指针 |
| `rankId_` | uint32_t | 本卡 Rank ID |
| `rankDim_` | uint32_t | 总 Rank 数 (默认 8) |

**方法详解**:

#### InitBase() (第57-74行)
```
功能: 初始化 GM 地址, 分配 Workspace 子区
步骤:
  1. 存储所有输入地址
  2. 若 gatherLen != 0 或 gatherGM 为空:
     gatherGM_ = workspaceGM_         ← AllGather 输出使用 Workspace 前段
     workspaceGM_ += gatherLen        ← Workspace 后段用于其他临时数据
```

#### Nd2NzBiasCast() (第77-95行)
```
功能: 可选的 B 矩阵 ND→NZ 格式转换和 Bias BF16→FP32 转换
步骤:
  1. TBuf 初始化 (TOTAL_UB_SIZE)
  2. Workspace 子区划分:
     gmNd2NzAddr = workspaceGM_                                    ← ND→NZ 转换区
     cToFloat    = gmNd2NzAddr + nd2NzWorkLen                      ← C 矩阵 float 区
     biasToFloat = cToFloat + cToFloatLen                           ← Bias float 区
  3. [Bias2Float] CastBFtoFloat(biasToFloat, biasGM_, rankN)       ← Bias 转 float
  4. [BNd2Nz] MatrixBtoNZ(bGM_, gmNd2NzAddr, tilingData_->param)  ← B 矩阵 ND→NZ
```

#### MatmulLocalCompute() (第98-114行)
```
功能: 计算本卡数据的 MatMul
步骤:
  1. 初始化 MatmulCompute (SplitType::DEFAULT)
  2. C 矩阵定位: cGM += rankId_ * rankM * N * sizeof(C_TYPE)
     → 本卡输出在完整 C 矩阵的第 rankId_ 个分片位置
  3. InitGlobalTensor(bGM_, biasGM_) → 设置 B、Bias 地址
  4. InitGlobalTensor(aGM, aOffset, cGM, cOffset) → 设置 A、C 偏移
  5. mm.Compute(0, GetBlockIdx()) → 按核 ID 分配计算
  6. mm.End()
```

#### MatmulLocalComputeL2Cache() (第118-134行)
```
功能: 使用 L2 Cache 优化的本卡 MatMul 计算
步骤: 与 MatmulLocalCompute 相同，但使用 SplitType::L2CACHE
  → mm.ComputeWithL2Cache() 代替 mm.Compute()
```

#### MatmulKernelLocal() (第137-154行)
```
功能: 本卡数据计算入口
步骤:
  1. 检查 blockIdx < tiling.usedCoreNum，超出则跳过
  2. 根据 l2Tiling.enableL2Tile 选择:
     [enableL2Tile > 0] → MatmulLocalComputeL2Cache()
     [else]             → MatmulLocalCompute()
```

### 4.5 FullMesh 实现

**文件**: `all_gather_matmul_full_mesh.h` (第22-229行)

```
AllGatherMatmulFullMesh<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>
  └── AllGatherMatmulBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>
```

**新增成员**:

| 变量 | 类型 | 含义 |
|------|------|------|
| `hccl_` | `Hccl<HCCL_SERVER_TYPE_AICPU>` | HCCL 通信实例 |
| `handleId_` | HcclHandle | 主分片 AllGather 句柄 |
| `tailHandleId_` | HcclHandle | 尾块 AllGather 句柄 |

**方法详解**:

#### Init() (第51-60行)
```
步骤:
  1. InitBase() — 初始化基类
  2. hccl_.Init(contextGM, mc2InitTiling) — 初始化 HCCL
  3. hccl_.SetCcTiling(mc2CcTiling) — 设置 CC Tiling
  4. rankId_ = hccl_.GetRankId()
  5. rankDim_ = hccl_.GetRankDim()
```

#### Process() (第63-69行)
```
完整执行流程:
  1. HcclPrepare()      — 下发 AllGather 通信任务
  2. Nd2NzBiasCast()    — 可选格式转换
  3. InnerProcess()     — 本卡计算 + 远端数据计算
  4. HcclFinalize()     — 等待并结束通信
```

#### HcclPrepare() (第72-95行)
```
功能: AIC 核预先下发 AllGather 任务
执行条件: ASCEND_IS_AIC 且非 debugOnlyCalc
步骤:
  1. 提取 cfg, tileTiling, tailTiling
  2. 计算分片大小:
     aTileCnt = tileTiling.M * tileTiling.Ka
     aTileOffset = tileCnt * aTileCnt * sizeof(A_TYPE)
     aRankCnt = rankM * rankK
  3. handleId_ = hccl_.AllGather<true>(
       aGM_, gatherGM_, aTileCnt, dataType, aRankCnt, tileCnt)
     ← 预下发 tileCnt 次 AllGather
  4. [tailCnt > 0] tailHandleId_ = hccl_.AllGather<true>(尾块)
```

**关键**: `AllGather<true>` 的 `true` 表示使用 AICPU 完成通信，返回的 HcclHandle 用于后续 Commit/Wait。

#### InnerProcess() (第98-121行)
```
功能: AIC 核执行核心计算逻辑
步骤:
  1. MatmulKernelLocal() — 计算本卡数据 (不需要等通信)
  2. MatmulKernelGather() — 逐片计算远端数据
     ├── tileCnt 片 (主分片)
     └── [tailCnt > 0] 尾块
```

#### MatmulKernelCompute() (第124-166行)
```
功能: 无 L2Cache 优化时的远端数据计算
参数:
  aGM   — AllGather 后的 A 矩阵数据 (已收集所有 rank)
  cGM   — 输出 C 矩阵
  tiling — 当前分片的 MatMul Tiling
  handleId — AllGather 句柄
  tileCnt — 分片数

步骤:
  1. 计算偏移:
     cOffset = M * N * sizeof(C_T)          ← 单片 C 大小
     aOffset = M * K * sizeof(A_T)          ← 单片 A 大小
     raOffset = rankM * rankK * sizeof(A_T) ← 单个 rank 的 A 大小
     rcOffset = rankM * rankN * sizeof(C_T) ← 单个 rank 的 C 大小
  2. 初始化 MatmulCompute (SplitType::DEFAULT)
  3. for i = 0..tileCnt:
     a. hccl_.Wait(handleId)                ← 等待第 i 片 AllGather 完成
     b. for j = 0..rankDim:
        rank = (rankId_ + j + blockIdx) % rankDim
        [rank == rankId_] continue          ← 跳过本卡 (已算过)
        index = (rank + rankDim - (rankId_+1)) % rankDim + i*(rankDim-1)
        aAddr = aGM + rank * raOffset       ← 远端 rank 的 A 分片
        cAddr = cGM + rank * rcOffset       ← 远端 rank 的 C 分片
        mm.InitGlobalTensor(aAddr, aOffset, cAddr, cOffset)
        mm.Compute(index)                   ← 计算
     c. aGM += aOffset; cGM += cOffset      ← 移到下一片
  4. mm.End()
```

**核负载分配策略**: 
- 所有 AIC 核同时参与计算
- `rank = (rankId_ + j + blockIdx) % rankDim` 确保:
  - 从自己的 rankId_ 开始编号
  - 加上 `blockIdx` 实现核间负载打散
  - 跳过自身 rank (已在 Local 中算过)

#### MatmulKernelComputeL2Cache() (第169-198行)
```
功能: 使用 L2 Cache 优化的远端数据计算
与 MatmulKernelCompute 的差异:
  - 使用 SplitType::L2CACHE 的 MatmulCompute
  - raOffset/rcOffset 乘以 rankDim_ (一次性传入全量数据)
  - mm.ComputeWithL2Cache(i) 代替内层循环
  - L2 Cache 硬件自动管理数据预取
```

#### MatmulKernelGather() (第201-214行)
```
功能: 远端数据计算入口
步骤:
  1. 检查 blockIdx < tiling.usedCoreNum
  2. 根据 l2Tiling.enableL2Tile 选择:
     [>0] → MatmulKernelComputeL2Cache()
     [else] → MatmulKernelCompute()
```

#### HcclFinalize() (第217-227行)
```
功能: AIC 核结束通信
步骤:
  1. 检查非 debugOnlyCalc
  2. CrossCoreSetFlag/WaitFlag(EVENT_ID_6) — 所有 AIC 核同步
  3. hccl_.Finalize() — 结束 HCCL 通信
```

## 5. Host-Kernel 接口

### 5.1 Tiling 数据传递

```
Host (Tiling)                    Kernel
┌────────────────────┐           ┌──────────────────────────────┐
│ Mc2InitTiling      │──tilingGM→│ hccl_.Init(contextGM, ...)   │
│ Mc2CcTiling        │           │ hccl_.SetCcTiling(...)       │
│ tileTiling         │           │ mm.Init(cfg, tiling, ...)    │
│ tailTiling         │           │                              │
│ localTiling        │           │ mm.Compute(blockIdx)         │
│ tileL2Tiling       │           │ mm.ComputeWithL2Cache(i)     │
│ tailL2Tiling       │           │                              │
│ localL2Tiling      │           │                              │
│ RCSTiling param    │           │ 分片偏移计算                   │
│ AllGatherSoc       │           │ 通信算法选择                   │
└────────────────────┘           └──────────────────────────────┘
```

### 5.2 Workspace 布局

```
Workspace:
├── [0, gatherLen)                          ← AllGather 输出缓冲
│   └── gatherGM_ (所有 rank 的 A 矩阵分片)
├── [gatherLen, gatherLen + nd2NzWorkLen)   ← B 矩阵 ND→NZ 转换区
├── [gatherLen + nd2NzWorkLen, + cToFloatLen) ← C 矩阵 float 中间结果
└── [rest]                                  ← Bias float 转换区
```

## 6. 完整数据流

```
AllGather + MatMul 数据流:

          AllGather                   Local MatMul              Remote MatMul
rank0 A ─┐                     ┌── B → C[rank0] (本卡)    ┌─ Wait(0) → B → C[rank0]
rank1 A ─┤→ gatherGM ─────────┤                          ├─ Wait(0) → B → C[rank1]
rank2 A ─┤  (全 A 分片)        │                          ├─ Wait(0) → B → C[rank2]
...      ─┤  按 tile 分片      │                          │  ...
rank7 A ─┘  重叠接收+计算      └── 不需要通信直接算         └─ Wait(N) → B → C[rank7]

时间线:
AIC Core 0: [Local MM] [Wait Tile0] [Remote MM Tile0] [Wait Tile1] [Remote MM Tile1] ...
AIC Core 1: [Local MM] [Wait Tile0] [Remote MM Tile0] [Wait Tile1] [Remote MM Tile1] ...
...
HCCL:                [AllGather Tile0] [AllGather Tile1] ...
                     ↑ overlap with Local MM    ↑ overlap with Remote MM Tile0
```

## 7. 性能相关设计

### 7.1 通信-计算重叠

- **本卡数据先行**: `MatmulKernelLocal()` 不需要等待通信，立即开始计算
- **逐片 Wait+计算**: 每收到一片 AllGather 数据就立即开始对应远端 rank 的 MatMul
- **分片数量**: 由 `commTurn` 属性或自动 tiling 决定

### 7.2 L2 Cache 优化

- 当 `l2Tiling.enableL2Tile > 0` 时使用 L2 Cache 模式
- `ComputeWithL2Cache()` 让硬件自动管理 A 矩阵的 L2 预取
- 减少了手动分块的 GM 访问延迟

### 7.3 ND→NZ 格式转换

- B 矩阵可以从 ND 格式转为 NZ (Fractal NZ) 格式
- NZ 格式对 Cube 计算单元更友好，减少格式转换开销
- 转换在 Workspace 中完成，不影响原始输入

## 8. 依赖关系

```
all_gather_matmul.cpp
├── all_gather_matmul_tiling_key.h      (TilingKey 定义)
├── all_gather_matmul_tiling.h          (TilingData 结构)
└── all_gather_matmul_full_mesh.h       (FullMesh 实现类)
    └── all_gather_matmul_base.h        (基类)
        ├── common/op_kernel/mc2_nd_to_nz.h     (ND→NZ 转换)
        ├── common/op_kernel/mc2_matmul_compute.h (MatMul 计算封装)
        └── adv_api/hccl/hccl.h                  (HCCL 通信 API)
```

## 9. 关键约束与注意事项

1. **仅支持 AIC+AIV 混合核** (`KERNEL_TYPE_MIX_AIC_1_2`)，不支持纯 AIV 或纯 AIC
2. **B 矩阵格式**: ND 或 NZ，NZ 需要 Workspace 空间做转换
3. **通信域**: 通过 `group` 属性指定，必须与训练框架的通信域一致
4. **gather_index**: 默认 0 (gather A 矩阵)，当前仅支持 0
5. **输出两个张量**: `y` (MatMul 结果) 和 `gather_out` (AllGather 中间结果)，后者可用于调试或残差

## 10. 与 all_gather_matmul_v2 的核心差异

`all_gather_matmul` (V1) 与 `all_gather_matmul_v2` (V2) 同属 AllGather+MatMul 融合算子族，但面向不同的精度场景。以下是两者的核心差异对照。

### 10.1 接口签名

| 维度 | V1 (all_gather_matmul) | V2 (all_gather_matmul_v2) |
|------|------------------------|---------------------------|
| 输入数量 | 3 (x1, x2, bias) | 7 (x1, x2, bias, x1_scale, x2_scale, quant_scale, bias_scale) |
| 输出数量 | 2 (y, gather_out) | 3 (y, gather_out, amax_out) |
| 新增输入 | — | x1_scale, x2_scale (输入量化缩放), quant_scale (输出量化缩放) |
| 新增输出 | — | amax_out (绝对值最大值，用于动态量化) |

### 10.2 数据类型与量化

| 维度 | V1 | V2 |
|------|----|----|
| 支持的数据类型 | FP16, BF16 | FP16, BF16, FP8, HIFLOAT8 |
| 量化支持 | 无 | 支持输入量化 (FP8/HIFLOAT8 输入) 和输出量化 |
| 计算精度 | 直接以输入精度计算 | 量化路径涉及反量化→计算→量化流程 |

### 10.3 算子定义结构

| 维度 | V1 | V2 |
|------|----|----|
| def.cpp 类定义 | 简洁构造 (行数较少) | Line 18 类声明, Lines 20-251 构造函数 (含量化参数注册), Line 254 OP_ADD |
| 参数注册 | 基础 x1/x2/bias + 7 个属性 | 基础参数 + 量化相关 scale 输入 + amax 输出 + 扩展属性 |

### 10.4 平台支持与 Tiling 复杂度

| 维度 | V1 | V2 |
|------|----|----|
| 支持平台 | ascend910b, ascend910_93, ascend950 | 与 V1 相同，但平台配置更复杂 (量化参数 per-SoC 调优) |
| Tiling 策略 | 基础 FullMesh + ND2NZ + BiasCast 三参数组合 | 在 V1 基础上增加量化相关 Tiling 维度 |
| Kernel 模板参数 | 3 个 (IsFullMesh, IsNd2Nz, IsBias) | 扩展为包含量化模式的模板参数 |

### 10.5 总结

V1 是基础的 AllGather+MatMul 融合，覆盖 FP16/BF16 精度的主流训练推理场景。V2 在 V1 架构上扩展了量化能力 (FP8/HIFLOAT8)，增加了缩放因子和动态量化输出，面向低精度训练和推理场景。两者的核心通信-计算重叠机制 (FullMesh、逐片 Wait+计算、L2 Cache 优化) 保持一致，V2 的额外复杂度主要集中在量化参数的 Host 侧管理和 Kernel 侧的反量化/量化计算路径。

---

*分析完成时间: 2026-04-17 (修订: 2026-04-21 新增第10节 V1/V2 变体对比)*
