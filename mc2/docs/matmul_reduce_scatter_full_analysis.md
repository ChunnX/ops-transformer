# matmul_reduce_scatter 算子全面分析报告

## 1. 算子概述

`matmul_reduce_scatter` 实现 **MatMul + ReduceScatter** 融合 (V1版本)。矩阵乘法完成后，通过 ReduceScatter 在 TP 域内归约并分片结果，每张卡获得输出的一部分。采用 FullMesh 通信算法，AIC 核逐片计算 MatMul 后 Commit ReduceScatter，实现计算-通信流水线重叠。

```
x1 [s, M, K] × x2 [K, N]
        ↓ MatMul (本地分片计算)
   partial_mm [M, N] (每片)
        ↓ ReduceScatter (TP域, N轴分片, 与计算重叠)
   y [s, M, N/rankSize] (每卡一份)
```

**支持平台**: ascend910b (arch32), ascend910_93 (arch32)

**不支持**: ascend950 (A5) — 内核入口有 `#if __CCE_AICORE__ != 310` 保护，注释明确标注"不支持A5"

### 1.1 模板参数

```cpp
// matmul_reduce_scatter_tiling_key.h (第23-28行)
ASCENDC_TPL_ARGS_DECL(MatmulReduceScatter,
    ASCENDC_TPL_BOOL_DECL(MM_REDUCE_SCATTER_FULL_MESH, 0, 1),   // 是否 FullMesh
    ASCENDC_TPL_BOOL_DECL(MM_REDUCE_SCATTER_ND2NZ_OPT, 0, 1),   // B矩阵是否 ND→NZ
    ASCENDC_TPL_BOOL_DECL(MM_REDUCE_SCATTER_BIAS_CAST, 0, 1));  // Bias是否 BF16→FP32
```

**4种合法组合** (第32-49行):

| Key | FullMesh | ND2NZ | BiasCast | B格式 | Bias类型 |
|-----|----------|-------|----------|-------|---------|
| 默认 | 1 | 1 | 0 | NZ | 原类型 |
| 默认 | 1 | 0 | 0 | ND | 原类型 |
| 默认 | 1 | 1 | 1 | NZ | float |
| 默认 | 1 | 0 | 1 | ND | float |

**注意**: 所有合法组合都要求 `FULL_MESH=1`，即 V1 仅支持 FullMesh 通信算法。

### 1.2 核心类模板

```cpp
// matmul_reduce_scatter_full_mesh.h (第27-28行)
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
class MatmulReduceScatterFullMesh
    : public MatmulReduceScatterBase<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BNd2Nz, Bias2Float>;
```

| 参数 | 含义 |
|------|------|
| `A_TYPE` | A 矩阵类型包装 `MatmulType<GM, ND, half/bfloat16_t, true>` |
| `B_TYPE` | B 矩阵类型包装 `MatmulType<GM, ND/NZ, half/bfloat16_t, true>` |
| `C_TYPE` | C 矩阵类型包装 `MatmulType<GM, ND, half/bfloat16_t>` |
| `BIAS_TYPE` | Bias 类型包装 `MatmulType<GM, ND, half/float>` |
| `BNd2Nz` | B 矩阵是否需要 ND→NZ 格式转换 |
| `Bias2Float` | Bias 是否需要 BF16→FP32 转换 |

---

## 2. 目录结构与文件职责

```
matmul_reduce_scatter/
├── op_kernel/
│   ├── matmul_reduce_scatter.cpp               (4,277字节)  内核入口 + 模板实例化
│   ├── matmul_reduce_scatter_base.h            (3,624字节)  基类 (GM管理 + ND2NZ/BiasCast)
│   ├── matmul_reduce_scatter_full_mesh.h       (11,716字节) FullMesh 主类 (HCCL RS + MM)
│   ├── matmul_reduce_scatter_tiling.h          (1,768字节)  Tiling 数据结构
│   ├── matmul_reduce_scatter_tiling_key.h      (2,288字节)  模板参数声明
│   └── reduce_scatter_nd_to_nz.h               (2,367字节)  B矩阵 ND→NZ 转换 (AIV侧)
├── op_host/
│   ├── matmul_reduce_scatter_def.cpp           (2,891字节)  算子定义
│   ├── matmul_reduce_scatter_infershape.cpp    (1,571字节)  Shape 推导
│   ├── op_tiling/
│   │   └── matmul_reduce_scatter_tiling.cpp    (36,436字节) Tiling 策略
│   └── config/
│       ├── ascend910b/   (binary.json + simplified_key.ini)
│       └── ascend910_93/ (binary.json + simplified_key.ini)
├── op_api/  (ACLNN C-API)
└── op_graph/ (proto, gen_task, fallback)
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `matmul_reduce_scatter_def.cpp` (第18-69行)

**输入张量**:

| 序号 | 名称 | 类型 | 格式 | 说明 |
|------|------|------|------|------|
| 0 | x1 | FP16, BF16 | ND | A 矩阵 [s, M, K] (REQUIRED) |
| 1 | x2 | FP16, BF16 | ND | B 矩阵 [K, N] (REQUIRED, IgnoreContiguous) |
| 2 | bias | FP16, BF16 | ND | 偏置 (OPTIONAL) |

**输出张量**:

| 序号 | 名称 | 类型 | 格式 | 说明 |
|------|------|------|------|------|
| 0 | y | FP16, BF16 | ND | ReduceScatter 后的输出 [s, M, N/rankSize] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| group | String | 必填 | HCCL 通信域标识 |
| reduce_op | String | "sum" | 归约操作 |
| is_trans_a | Bool | false | A 矩阵是否转置 |
| is_trans_b | Bool | false | B 矩阵是否转置 |
| comm_turn | Int | 0 | 通信分片数 (0=自动) |
| rank_size | Int | 0 | Rank 数量 (0=自动) |

**MC2 配置**: `MC2().HcclGroup("group")` — 使用 `group` 属性作为 HCCL 通信域。

**平台配置**: ascend910b, ascend910_93, ascend950 (均支持动态Shape、动态格式、动态Rank)。

### 3.2 Tiling 策略

**文件**: `matmul_reduce_scatter_tiling.cpp` (36,436字节)

#### 常量定义 (第37-53行)

```cpp
const std::map<uint32_t, std::vector<uint32_t>> VALID_RANK = {
    {0, {2, 4, 8}},          // level0: 单机内
    {1, {2, 4, 8, 16, 32}}   // level1: 跨机
};
constexpr uint64_t CALC_ND_BASIC[] = {6144, 4096, 2048};  // ND2NZ 候选 baseD
constexpr uint64_t UB_SIZE = 196352;                       // UB 大小限制
```

#### ND2NZ Tiling 计算 (第64-120行)

```
CalcNd2NzTilingMC2(args, ubSize, dtypeSize, nValue, dValue, baseN, baseD):
├── mata 交织场景 (dValue % 16384 == 0 && nValue >= 7168 && aicCoreNum >= 24):
│   ├── baseN = 96    (经验值)
│   └── baseD = 512   (经验值)
├── dValue <= 2048/dtypeSize:
│   ├── baseD = CeilAlign(dValue, c0)
│   ├── baseN = ubSize / 2 / dtypeSize / baseD
│   └── 溢出检查: 逐次递减 baseN 直到 UB 不溢出
└── dValue > 2048/dtypeSize:
    ├── 遍历 CALC_ND_BASIC = {6144, 4096, 2048}:
    │   ├── baseD = min(dAlignedC0, base/dtypeSize)
    │   ├── 尾块处理: dTail < 512 时合并到主块
    │   ├── baseN = ubSize / 2 / dtypeSize / baseD
    │   └── 溢出检查
    └── 选择最优 baseN/baseD 组合
```

#### TilingKey 计算

```
UpdateTilingKey():
  key = GET_TPL_TILING_KEY(isFullMesh=true, isNd2Nz, isBiasCast)
  → 4 种组合 (FullMesh 固定为 true)
```

### 3.3 Shape 推导

**文件**: `matmul_reduce_scatter_infershape.cpp` (第23-41行)

```
InferShapeMatmulReduceScatter():
├── 调用 InferMatmulReduceScatterCommon(context)  ← 公共 infershape
└── 输出数据类型 = 输入 x1 数据类型
```

输出 Shape: `y = [s, M, N/rankSize]`

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `matmul_reduce_scatter.cpp` (第47-84行)

```cpp
template<bool IsFullMesh, bool IsNd2Nz, bool IsBias>
__global__ __aicore__ void matmul_reduce_scatter(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
```

**入口流程**:
1. `REGISTER_TILING_DEFAULT(MatmulReduceScatterTilingData)` (第50行)
2. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` (第58行) — AIC+AIV 混合核
3. `GetHcclContext<HCCL_GROUP_ID_0>()` (第60行) — 获取 HCCL 上下文
4. 根据 4 种模板组合实例化 B/bias 类型 (第62-82行)
5. `INVOKE_MATMUL_REDUCE_SCATTER_OP_IMPL(MatmulReduceScatterFullMesh, IsNd2Nz, IsBias)` 宏展开

**B矩阵类型选择**:
- `IsNd2Nz=true`: `bType = MatmulType<GM, NZ, B_DTYPE, true>` — NZ 格式 (Fractal)
- `IsNd2Nz=false`: `bType = MatmulType<GM, ND, B_DTYPE, true>` — ND 格式

**Bias类型选择**:
- `IsBias=false`: `biasType = MatmulType<GM, ND, BiasType<BIAS_DTYPE>::type>` — half→half, 其他→float
- `IsBias=true`: `biasType = MatmulType<GM, ND, float>` — 强制 float

### 4.2 基类 MatmulReduceScatterBase

**文件**: `matmul_reduce_scatter_base.h` (第33-90行)

```cpp
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, bool BNd2Nz, bool Bias2Float>
class MatmulReduceScatterBase;
```

**成员变量**:

| 变量 | 类型 | 含义 |
|------|------|------|
| `aGM_` | GM_ADDR | A 矩阵地址 |
| `bGM_` | GM_ADDR | B 矩阵地址 |
| `biasGM_` | GM_ADDR | Bias 地址 |
| `cGM_` | GM_ADDR | C 矩阵 (输出) 地址 |
| `gmToFloat_` | GM_ADDR | C 矩阵 float 中间缓冲 (Workspace 前段) |
| `workspaceGM_` | GM_ADDR | Workspace 基址 |
| `tilingData_` | MatmulReduceScatterTilingData* | Tiling 数据指针 |
| `tPipe_` | TPipe* | 管线指针 |
| `rankId_` | uint32_t | 本卡 Rank ID |
| `rankDim_` | uint32_t | 总 Rank 数 (默认 8) |
| `debugOnlyCalc_` | bool | 调试模式: 只计算不通信 |

#### InitBase() (第56-69行)

```
InitBase(aGM, bGM, biasGM, cGM, workspaceGM, contextGM, tilingData, tPipe):
├── 存储所有 GM 地址和 Tiling 指针
└── gmToFloat_ = workspaceGM_  ← C 矩阵 float 中间结果使用 Workspace 前段
```

#### Nd2NzBiasCast() (第72-89行)

```
Nd2NzBiasCast():
├── TBuf 初始化 (TOTAL_UB_SIZE)
├── Workspace 子区划分:
│   ├── [0, cToFloatLen)                              ← C 矩阵 float 区
│   ├── [cToFloatLen, cToFloatLen + nd2NzWorkLen)     ← ND→NZ 转换区
│   └── [cToFloatLen + nd2NzWorkLen, ...)             ← Bias float 转换区
├── [Bias2Float] CastBFtoFloat(biasToFloat, biasGM_, rankN)
│   └── Bias BF16→FP32 转换
│   └── SyncAll<true>()  ← AIV 完成后同步 AIC
└── [BNd2Nz] MatrixBtoNZMc2(gmNd2NzAddr, bGM_, tilingData_, isTransposeB, totalUbBuf)
    └── B 矩阵 ND→NZ 格式转换 (AIV 侧执行)
```

### 4.3 FullMesh 实现 MatmulReduceScatterFullMesh

**文件**: `matmul_reduce_scatter_full_mesh.h` (第27-253行)

继承关系: `MatmulReduceScatterFullMesh → MatmulReduceScatterBase`

**新增成员**:

| 变量 | 类型 | 含义 |
|------|------|------|
| `hccl_` | `Hccl<HCCL_SERVER_TYPE_AICPU>` | HCCL 通信实例 (使用 AICPU Server) |
| `handleId_` | HcclHandle | 主分片 ReduceScatter 句柄 |
| `tailHandleId_` | HcclHandle | 尾块 ReduceScatter 句柄 |

#### Init() (第54-63行)

```
Init(aGM, bGM, biasGM, cGM, workspaceGM, contextGM, tilingData, tPipe, mc2InitTiling, mc2CcTiling):
├── InitBase() — 初始化基类
├── hccl_.Init(contextGM, mc2InitTiling) — HCCL 初始化
├── hccl_.SetCcTiling(mc2CcTiling) — 设置 CC Tiling
├── rankId_ = hccl_.GetRankId()
└── rankDim_ = hccl_.GetRankDim()
```

#### Process() (第66-72行)

```
Process():
├── Nd2NzBiasCast()      ← 基类: 可选 ND→NZ / Bias 转换
├── HcclPrepare()        ← 预注册 ReduceScatter 任务
├── InnerProcess()       ← MM 计算 + 逐片 Commit + Wait
└── HcclFinalize()       ← 结束 HCCL
```

#### HcclPrepare() (第75-98行)

```
HcclPrepare() [仅 AIC 核]:
├── if (debugOnlyCalc_) return
├── 提取 cfg, tileTiling, tailTiling
├── 计算分片参数:
│   ├── tileDataCnt = tileTiling.M × tileTiling.N          ← 单片 C 矩阵元素数
│   ├── tileDataOff = tileDataCnt × tileCnt × sizeof(C)    ← 尾块偏移
│   └── rankDataCnt = rankM × rankN / rankDim              ← 单卡 RS 数据量
├── handleId_ = hccl_.ReduceScatter<false>(
│     gmToFloat_, cGM_, tileDataCnt, dataType, reduceOp, rankDataCnt, tileCnt)
│   ← 注册 tileCnt 次 ReduceScatter (false=不立即提交)
└── [tailCnt > 0] tailHandleId_ = hccl_.ReduceScatter<false>(
      gmToFloat_ + tileDataOff, cGM_ + tileDataOff, ...)
      ← 注册尾块 ReduceScatter
```

**关键**: `ReduceScatter<false>` 仅注册不提交，后续在计算每片 MM 后手动 `Commit`。

#### InnerProcess() (第101-130行)

```
InnerProcess() [仅 AIC 核]:
├── MatmulKernelReduceScatter(tileTiling, handleId_, tileCnt)
│   ├── 主分片 MM 计算 + 逐片 Commit ReduceScatter
│   └── 包含 [enableL2Tile] 分支
├── [tailCnt > 0]:
│   ├── aGMTail = aGM + M × Ka × tileCnt × sizeof(A)
│   ├── gmToFloatTail = gmToFloat_ + M × N × tileCnt × sizeof(C)
│   └── MatmulKernelReduceScatter(tailTiling, tailHandleId_, tailCnt)
│       └── 尾块 MM 计算 + Commit
├── if (!debugOnlyCalc_):
│   ├── for i = 0..tileCnt: hccl_.Wait(handleId_)
│   └── for i = 0..tailCnt: hccl_.Wait(tailHandleId_)
└── 等待所有 ReduceScatter 完成
```

#### MatmulKernelCompute() (第134-180行) — 标准 MM 计算

```
MatmulKernelCompute(aGM, gmToFloat, tiling, l2Tiling, handleId, tileCnt):
├── if (blockIdx >= usedCoreNum):
│   └── 参与核间同步但跳过计算
├── 计算偏移:
│   ├── aOffset = M × Ka × sizeof(A)           ← 单片 A 大小
│   ├── cOffset = M × N × sizeof(C)            ← 单片 C 大小
│   ├── raOffset = rankM × rankK / rankDim × sizeof(A) ← 单 Rank A 分片
│   └── rcOffset = rankM × rankN / rankDim × sizeof(C) ← 单 Rank C 分片
├── 初始化 MatmulCompute<SplitType::DEFAULT>
│   ├── InitGlobalTensor(bGM_, biasGM_)   ← B 和 Bias 固定
│   └── Init(cfg, tiling, l2Tiling, rankId_)
└── for i = 0..tileCnt:
    ├── for j = 0..rankDim:
    │   ├── rank = (j + blockIdx) % rankDim   ← 核间负载打散
    │   ├── aWorkAddr = aWork + rank × raOffset
    │   ├── cWorkAddr = cWork + rank × rcOffset
    │   ├── InitGlobalTensor(aWorkAddr, cWorkAddr)
    │   └── mm.Compute(rank + i × rankDim)
    ├── CrossCoreSetFlag/WaitFlag(EVENT_ID_6)  ← AIC 核间同步
    ├── hccl_.Commit(handleId)                  ← 提交该片 ReduceScatter
    ├── aWork += aOffset                        ← 移到下一片
    └── cWork += cOffset
```

**核负载分配**: `rank = (j + blockIdx) % rankDim` 确保不同 AIC 核从不同 Rank 开始计算，避免集中访问。

#### MatmulKernelComputeL2Cache() (第184-225行) — L2 Cache 优化 MM

```
MatmulKernelComputeL2Cache(aGM, gmToFloat, tiling, l2Tiling, handleId, tileCnt):
├── 与 MatmulKernelCompute 的差异:
│   ├── 使用 SplitType::L2CACHE 的 MatmulCompute
│   ├── raOffset/rcOffset 乘以 rankDim_ (一次性传入全量数据)
│   └── mm.ComputeWithL2Cache(i) 代替内层循环
│       ← L2 Cache 硬件自动管理数据预取
└── 同样逐片 Commit + 核间同步
```

#### HcclFinalize() (第241-252行)

```
HcclFinalize() [仅 AIC 核]:
├── if (!debugOnlyCalc_):
│   ├── CrossCoreSetFlag/WaitFlag(EVENT_ID_6)  ← 确保所有核计算完成
│   └── hccl_.Finalize()                         ← 结束 HCCL 通信
```

### 4.4 ND→NZ 格式转换

**文件**: `reduce_scatter_nd_to_nz.h` (第43-63行)

```
MatrixBtoNZMc2(workspace, src, tilingData, isTransposeB, tmpBuf):
├── [AIV 核]:
│   ├── if (blockIdx >= usedCoreNum):
│   │   └── SetFlag(3)  ← 未使用的核直接设置同步标志
│   ├── MatrixBtoNZV2(workspace, src, tiling, isTransposeB, tmpBuf, baseBN, baseBD)
│   │   ← 执行实际的 ND→NZ 格式转换
│   │   ← 使用 tiling 中计算的 baseN/baseD 作为分块参数
│   └── CrossCoreSetFlag(3)  ← AIV 完成后通知 AIC
└── [AIC 核]:
    ├── CrossCoreWaitFlag(3)   ← 等待 AIV 完成 ND→NZ
    ├── CrossCoreSetFlag(4)    ← AIC 核间同步
    └── CrossCoreWaitFlag(4)
```

**AIC-AIV 跨核同步**: AIV 完成格式转换后通过 Event 3 通知 AIC，AIC 核间再通过 Event 4 同步一次。

---

## 5. 数据结构

### 5.1 MatmulReduceScatterTilingData

**文件**: `matmul_reduce_scatter_tiling.h` (第37-50行)

```
MatmulReduceScatterTilingData:
├── mc2InitTiling: Mc2InitTiling           ← HCCL 初始化参数
├── mc2CcTiling: Mc2CcTiling               ← HCCL CC Tiling
├── param: RCSTiling                       ← 核心参数
│   ├── rankM, rankK, rankN                ← 矩阵维度
│   ├── tileCnt, tailCnt                   ← 主分片和尾块数量
│   ├── dataType, subtype                  ← HCCL 数据类型和规约操作
│   ├── cToFloatLen                        ← C 矩阵 float 区大小
│   ├── nd2NzWorkLen                       ← ND→NZ Workspace 大小
│   ├── biasLen                            ← Bias 转换区大小
│   └── isTransposeB                       ← B 矩阵转置标志
├── tileTiling: TCubeTiling                ← 主分片 MM Tiling
├── tailTiling: TCubeTiling                ← 尾块 MM Tiling
├── localTiling: TCubeTiling               ← 本地数据 MM Tiling (未使用于V1)
├── tileL2Tiling: TileL2Tiling             ← 主分片 L2 Cache Tiling
├── tailL2Tiling: TileL2Tiling             ← 尾块 L2 Cache Tiling
├── localL2Tiling: TileL2Tiling            ← 本地 L2 Cache Tiling (未使用于V1)
├── socParam: ReduceScatterSoc             ← SoC 特定参数
│   ├── commAlg                            ← 通信算法
│   ├── isA3                               ← 是否 A3 架构
│   ├── isStep                             ← 是否分步模式
│   ├── isND2NZ                            ← 是否 ND→NZ
│   ├── baseBN                             ← ND2NZ baseN 参数
│   └── baseBD                             ← ND2NZ baseD 参数
└── l2cacheUseInfo: Mc2L2cacheUseInfo      ← L2 Cache 使用信息
```

### 5.2 ReduceScatterSoc

```
ReduceScatterSoc:
├── commAlg: uint32  ← 通信算法 (FULL_MESH / DOUBLE_RING 等)
├── isA3: uint32     ← 是否 A3 架构 (910_93)
├── isStep: uint32   ← 是否分步模式
├── isND2NZ: uint32  ← 是否启用 ND→NZ 转换
├── baseBD: uint32   ← ND2NZ 的 baseD 参数
└── baseBN: uint32   ← ND2NZ 的 baseN 参数
```

---

## 6. 内存管理

### 6.1 Workspace 布局

```
Workspace 布局 (按偏移递增):
┌─────────────────────────────────────────────────┐
│ [0, cToFloatLen)                                │
│   C 矩阵 float 中间结果区                        │
│   大小: rankM × rankN × sizeof(C_TYPE)          │
│   用途: MatMul 计算结果暂存                      │
│   注意: gmToFloat_ 指向此区起始                   │
├─────────────────────────────────────────────────┤
│ [cToFloatLen, cToFloatLen + nd2NzWorkLen)       │
│   B 矩阵 ND→NZ 转换区 (条件分配)                  │
│   大小: 由 Tiling 计算                           │
│   注意: 仅 IsNd2Nz=true 时使用                    │
├─────────────────────────────────────────────────┤
│ [cToFloatLen + nd2NzWorkLen, ...)               │
│   Bias float 转换区 (条件分配)                    │
│   大小: rankN × sizeof(float)                   │
│   注意: 仅 Bias2Float=true 时使用                 │
└─────────────────────────────────────────────────┘
```

### 6.2 UB 分配

```
AIV 侧 UB (TOTAL_UB_SIZE ≈ 196KB):
┌─────────────────────────────────────────────────┐
│ TBuf<VECCALC> totalUbBuf (TOTAL_UB_SIZE)       │
│ ├── 用于 ND→NZ 转换: MatrixBtoNZV2()            │
│ └── 用于 Bias 转换: CastBFtoFloat()             │
└─────────────────────────────────────────────────┘

ND→NZ UB 分配策略 (CalcNd2NzTilingMC2):
  baseN × baseD × dtypeSize × 2 ≤ ubSize (196352 字节)
  → Ping-Pong 双缓冲
```

---

## 7. 通信模式

### 7.1 FullMesh ReduceScatter

```
通信架构:
┌─────────────────────────────────────────────────────────┐
│ TP 域 (所有 Rank 通过 HCCL ReduceScatter 归约)           │
│                                                          │
│ Rank 0: MM 结果 [M, N/rankDim] × tileCnt 片              │
│ Rank 1: MM 结果 [M, N/rankDim] × tileCnt 片              │
│ ...                                                      │
│ Rank N: MM 结果 [M, N/rankDim] × tileCnt 片              │
│                                                          │
│ hccl_.ReduceScatter(gmToFloat_, cGM_, tileDataCnt, ...)  │
│   → 注册 tileCnt 次 ReduceScatter                         │
│   → 每片 MM 完成后 Commit 一次                             │
│   → 通信与下一片的 MM 计算重叠                              │
└─────────────────────────────────────────────────────────┘
```

### 7.2 计算与通信流水线时序

```
时间 →
AIC核0: [MM tile0, rank0..N] [Sync] [Commit RS0] [MM tile1, rank0..N] [Sync] [Commit RS1] ...
AIC核1: [MM tile0, rank1..0] [Sync] [Commit RS0] [MM tile1, rank1..0] [Sync] [Commit RS1] ...
HCCL:                        [RS tile0 执行中...]                      [RS tile1 执行中...]
                              ↑ 与 tile1 MM 重叠                       ↑ 与后续计算重叠

最终: [Wait RS0] [Wait RS1] ... [Finalize]
```

### 7.3 同步点汇总

| 同步点 | 类型 | 参与者 | 目的 |
|--------|------|--------|------|
| SyncAll | 全核 | AIV+AIC | BiasCast 完成后同步 |
| CrossCoreSet/WaitFlag(3) | AIC↔AIV | AIC+AIV | ND→NZ 转换完成通知 |
| CrossCoreSet/WaitFlag(4) | AIC核间 | AIC | ND→NZ 后 AIC 核间同步 |
| CrossCoreSet/WaitFlag(EVENT_ID_6) | AIC核间 | AIC | 每片 MM 完成后同步 |
| hccl_.Commit | HCCL | AIC | 提交单次 ReduceScatter |
| hccl_.Wait | HCCL | AIC | 等待单次 RS 完成 |
| CrossCoreSet/WaitFlag(EVENT_ID_6) | AIC核间 | AIC | Finalize 前同步 |

---

## 8. 完整数据流

```
输入: x1 [s, M, K], x2 [K, N], bias (可选)
                                        ┌── AIV 侧 ──┐
                                        │             │
Step 1: [AIV] Nd2NzBiasCast()          │             │
        ├── [Bias2Float] Bias BF16→FP32 │             │
        ├── [BNd2Nz] B矩阵 ND→NZ       │             │
        └── SyncAll                     │             │
                                        └─────────────┘
                                        ┌── AIC 侧 ──┐
Step 2: [AIC] HcclPrepare()            │             │
        └── 注册 tileCnt+tailCnt 次     │             │
            ReduceScatter               │             │
                                        │             │
Step 3: [AIC] InnerProcess()           │             │
        ├── for tile = 0..tileCnt:      │             │
        │   ├── MM(x1+offset, x2, bias) │             │
        │   │   → gmToFloat_ [M, N]     │             │
        │   ├── CrossCoreSync            │             │
        │   └── hccl_.Commit(handleId)  │             │
        ├── [tailCnt] 尾块 MM + Commit  │             │
        └── Wait 所有 RS 完成           │             │
                                        │             │
Step 4: [AIC] HcclFinalize()           │             │
                                        └─────────────┘
                                        ↓
输出: y [s, M, N/rankSize] (ReduceScatter 归约分片后的结果)
```

---

## 9. 与 V2 的核心差异

| 特性 | V1 (matmul_reduce_scatter) | V2 (matmul_reduce_scatter_v2) |
|------|---------------------------|------------------------------|
| 通信算法 | 仅 FullMesh | FullMesh + All2All+VecReduce |
| 数据类型 | FP16/BF16 | FP16/BF16/FP8/HIF8/INT8 |
| ND→NZ | 支持 (AIV侧转换) | 支持 |
| SmallM 优化 | 不支持 | 专用 SmallM 路径 |
| MatMul 引擎 | MatmulImpl (AscendC) | Catlass 模板库 |
| L2 Cache 优化 | 支持 | 支持 |
| 量化 | 不支持 | Fixpipe/PerBlock |
| 平台 | 910B/910_93 | 910B/910_93/950 |
| AICPU Server | HCCL_SERVER_TYPE_AICPU | CCU (950) / AICPU |
| 内核文件大小 | 11,716字节 (FullMesh) | 28,847 + 36,209字节 |
| Padding | 不支持 | 支持 |
| 双通信路径 | 无 | Native RS + All2All+VecReduce |

---

## 10. 函数调用关系

```
matmul_reduce_scatter (内核入口)
├── MatmulReduceScatterFullMesh::Init()
│   ├── MatmulReduceScatterBase::InitBase()
│   │   └── gmToFloat_ = workspaceGM_
│   ├── hccl_.Init(contextGM, mc2InitTiling)
│   ├── hccl_.SetCcTiling(mc2CcTiling)
│   ├── rankId_ = hccl_.GetRankId()
│   └── rankDim_ = hccl_.GetRankDim()
│
└── MatmulReduceScatterFullMesh::Process()
    ├── MatmulReduceScatterBase::Nd2NzBiasCast()
    │   ├── TBuf InitBuffer (TOTAL_UB_SIZE)
    │   ├── [Bias2Float] CastBFtoFloat(biasToFloat, biasGM_)
    │   │   └── SyncAll<true>()
    │   └── [BNd2Nz] MatrixBtoNZMc2(gmNd2NzAddr, bGM_)
    │       ├── [AIV] MatrixBtoNZV2()
    │       ├── [AIV] CrossCoreSetFlag(3)
    │       └── [AIC] CrossCoreWaitFlag(3) + SetFlag(4) + WaitFlag(4)
    │
    ├── HcclPrepare() [AIC]
    │   ├── handleId_ = hccl_.ReduceScatter<false>(gmToFloat_, cGM_, ...)
    │   └── [tailCnt>0] tailHandleId_ = hccl_.ReduceScatter<false>(...)
    │
    ├── InnerProcess() [AIC]
    │   ├── MatmulKernelReduceScatter(tileTiling, handleId_, tileCnt)
    │   │   ├── [enableL2Tile>0] MatmulKernelComputeL2Cache()
    │   │   │   ├── MatmulCompute<SplitType::L2CACHE>
    │   │   │   └── mm.ComputeWithL2Cache(i)
    │   │   └── [enableL2Tile==0] MatmulKernelCompute()
    │   │       ├── for j = 0..rankDim:
    │   │       │   ├── rank = (j + blockIdx) % rankDim
    │   │       │   └── mm.Compute(rank + i × rankDim)
    │   │       └── hccl_.Commit(handleId)  ← 每片提交一次 RS
    │   ├── [tailCnt>0] MatmulKernelReduceScatter(tailTiling, tailHandleId_)
    │   └── Wait 所有 ReduceScatter
    │       ├── for i: hccl_.Wait(handleId_)
    │       └── for i: hccl_.Wait(tailHandleId_)
    │
    └── HcclFinalize() [AIC]
        ├── CrossCoreSetFlag/WaitFlag(EVENT_ID_6)
        └── hccl_.Finalize()
```

---

## 11. 依赖关系

```
matmul_reduce_scatter.cpp (内核入口)
├── matmul_reduce_scatter_tiling.h        ← Tiling 数据结构
├── matmul_reduce_scatter_tiling_key.h    ← 模板参数声明
├── matmul_reduce_scatter_full_mesh.h     ← FullMesh 实现类
│   ├── matmul_reduce_scatter_base.h      ← 基类
│   │   ├── reduce_scatter_nd_to_nz.h     ← ND→NZ 转换
│   │   │   └── 3rd/mat_mul_v3/op_kernel/mat_mul_nd2nz.h
│   │   ├── common/op_kernel/mc2_nd_to_nz.h      ← 公共 ND→NZ
│   │   └── common/op_kernel/mc2_matmul_compute.h ← MatmulCompute 封装
│   └── adv_api/hccl/hccl.h              ← HCCL 通信 API
└── lib/matmul_intf.h                     ← AscendC Matmul 接口
```

---

## 12. 关键约束与注意事项

1. **不支持 A5 (ascend950)**: 内核入口明确排除 AICORE==310 路径
2. **仅支持 FullMesh**: TilingKey 中 FULL_MESH 固定为 true
3. **HCCL Server**: 使用 AICPU 类型 (`HCCL_SERVER_TYPE_AICPU`)
4. **混合核模式**: `KERNEL_TYPE_MIX_AIC_1_2` (1 AIC + 2 AIV)
5. **Rank 数限制**: 单机 {2, 4, 8}，跨机 {2, 4, 8, 16, 32}
6. **B 矩阵 IgnoreContiguous**: 定义中标记 `IgnoreContiguous()`，允许非连续内存
7. **Workspace 必需**: 至少需要 C 矩阵 float 中间结果空间
8. **L2 Cache 选择**: 由 `l2Tiling.enableL2Tile` 控制，自动选择标准/L2Cache 模式

---

*分析完成时间: 2026-04-17*
