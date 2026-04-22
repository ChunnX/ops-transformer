# matmul_all_reduce 算子全面分析报告

## 1. 算子概述

`matmul_all_reduce` 是 mc2 库中**最核心的通信-计算融合算子**，将矩阵乘法 (MatMul) 与集合通信 AllReduce 融合为单次 Kernel 执行。通过将大矩阵沿 M 轴切分为多个 Tile，逐 Tile 执行 MatMul + AllReduce 的流水线，实现计算与通信的重叠，从而隐藏 AllReduce 延迟。

```
x1 [M, K] × x2 [K, N] → MM结果 [M, N]
     ↓ 切分为 tileCnt 个 Tile
Tile_0: MM → AllReduce ──┐
Tile_1: MM → AllReduce ──┤ 流水线重叠
  ...                     │
Tile_T: MM → AllReduce ──┘
     ↓ 合并
输出 y [M, N]  (AllReduce 后的全局求和)
```

### 核心设计特点

- **多量化模式**: 支持非量化 (A16W16)、全量化 (A8W8)、权重伪量化 (A16W8/A16W4)、FP8/MX 等多种量化场景
- **多平台适配**: ascend910B (arch32)、ascend310P (arch31)、ascend950 (arch35)
- **三级 Kernel 入口分发**: `matmul_all_reduce` → `fp_matmul_all_reduce` / `quant_matmul_allreduce` / `weight_quant_matmul_allreduce`
- **模板实例化**: 通过 15 个模板参数编码所有变体，TilingKey 空间覆盖数百种合法组合
- **通信-计算流水线**: 每完成一个 Tile 的 MatMul 即 Commit 一次 AllReduce，后续 Tile 的 MatMul 与前序 AllReduce 重叠执行

### 模板参数体系

```
15 个模板参数:
┌─────────────────────────────────────────────────────────┐
│ AICORE_TYPE       │ 910B / 310P                          │
│ MM_TYPE           │ FP_MM / FP_MM_CUBE_ONLY / QUANT /    │
│                   │ WEIGHT_QUANT                          │
│ EMPTY_INPUT       │ 是否空张量 (K=0)                       │
│ INT8_COMM         │ 是否 INT8 通信量化                     │
│ ENABLE_L2_CACHE   │ 是否开启 L2 Cache 分块                 │
│ SHARE_MM          │ 预留共享 MM                           │
│ MIXND2NZ          │ ND→NZ 转换策略 (仅 FP_MM)             │
│ TRANS             │ B 矩阵转置 (仅 QUANT)                  │
│ PERTOKEN          │ PerToken 量化 (仅 QUANT)               │
│ SUB_ALGORITHM     │ 权重量化算法子类 (仅 WEIGHT_QUANT)      │
│ HAS_ANTIQUANT_OFFSET │ 是否有反量化偏移                     │
│ ANTIQUANT_TYPE    │ 反量化粒度: NONE/PerTensor/PerChannel/ │
│                   │ PerGroup/MX                           │
│ TRANS_A / TRANS_B │ A/B 矩阵转置                          │
│ FORMAT_B          │ 权重格式: ND / NZ                      │
└─────────────────────────────────────────────────────────┘
```

### MM_TYPE 量化模式分类

| MM_TYPE | 值 | 含义 | 数据类型 | 核心 Tiling 结构 |
|---------|---|------|---------|----------------|
| `FP_MM` | 0 | 非量化 FP16/BF16 | x1=FP16/BF16, x2=FP16/BF16 | `MatmulAllReduce910TilingData` |
| `FP_MM_CUBE_ONLY` | 1 | 仅 Cube 计算（无通信） | 同上 | 同上 |
| `QUANT_MATMUL` | 2 | 全量化 A8W8 | x1=INT8, x2=INT8 | `QuantMatmulAllReduceTilingData` |
| `WEIGHT_QUANT_MATMUL` | 3 | 权重伪量化 A16W8/A16W4 | x1=FP16/BF16, x2=INT8/INT4 | `WeightQuantMatmulAllReduceTilingData` |

---

## 2. 目录结构与文件职责

```
matmul_all_reduce/
├── op_kernel/                              (设备侧)
│   ├── common.h                            (4,691B) 共享定义：数据类型宏、GM 地址结构、同步原语
│   ├── matmul_all_reduce.cpp               (主入口 590 行)
│   │   ├── matmul_all_reduce<>             ← 顶层入口，根据 MM_TYPE 分发
│   │   ├── fp_matmul_all_reduce<>          ← 非量化路径 (910B)
│   │   ├── quant_matmul_allreduce<>        ← 全量化路径 (910B: FP16/BF16 输出)
│   │   └── weight_quant_matmul_allreduce<> ← 权重伪量化路径 (910B)
│   ├── matmul_all_reduce_tiling_key.h      (743B) TilingKey 宏定义 + ASCENDC_TPL 声明
│   ├── matmul_all_reduce_apt.cpp           APT (Auto Parallel Tiling) 入口
│   ├── matmul_all_reduce_apt_tiling_key.h  APT TilingKey
│   └── arch32/                             (910B 平台内核实现)
│       ├── matmul_all_reduce_base.h         基类: Init/HcclFinalize/PostProcEachTurn
│       ├── matmul_all_reduce_910_general.h  非量化: MatmulAllReduce910General
│       ├── matmul_all_reduce_quant.h        全量化: BmmDequant
│       ├── matmul_all_reduce_quant_bf16_comm_int8.h  BF16 全量化+INT8 通信
│       ├── matmul_all_reduce_quant_fp16_comm_int8.h  FP16 全量化+INT8 通信
│       ├── matmul_all_reduce_quant_pertoken_comm_int8.h  PerToken 全量化+INT8 通信
│       ├── matmul_all_reduce_weight_quant.h  权重伪量化主类
│       ├── matmul_all_reduce_add_x3.h       残差加法辅助
│       ├── matmul_all_reduce_empty_tensor_k_general.h  空张量处理 (K=0)
│       ├── unquant_matmul_all_reduce_tiling_data.h  非量化 Tiling 数据结构
│       ├── quant_matmul_all_reduce_tiling_data.h    全量化 Tiling 数据结构
│       └── weight_quant_matmul_all_reduce_tiling_data.h  权重伪量化 Tiling 数据结构
├── op_host/                                (Host 侧)
│   ├── matmul_all_reduce_def.cpp           算子定义: 8 输入 + 1 输出 + 8 属性
│   ├── matmul_all_reduce_infershape.cpp    Shape 推导: [S,]M,K × [K,N] → [S,]M,N
│   └── op_tiling/
│       ├── matmul_all_reduce_tiling.cpp     Tiling 入口: 按架构分发
│       ├── matmul_all_reduce_tiling_base.h/cpp  基类: 形状分析、量化检测、MatMul+AR Tiling
│       ├── all_reduce_formulaic_tiling.h/cpp   AllReduce 公式化 Tiling
│       ├── arch32/                          910B 专用 Tiling
│       │   ├── matmul_all_reduce_tiling_910.cpp
│       │   ├── quant_matmul_all_reduce_tiling.cpp
│       │   └── weight_quant_matmul_all_reduce_tiling.cpp
│       ├── arch31/                          310P 专用 Tiling
│       │   ├── matmul_all_reduce_tiling_310_general.cpp
│       │   ├── quant_matmul_all_reduce_tiling_310_general.cpp
│       │   ├── unquant_matmul_all_reduce_tiling_310.cpp
│       │   └── weight_quant_matmul_all_reduce_tiling_310p.cpp
│       └── arch35/                          950 专用 Tiling
│           ├── matmul_all_reduce_tiling_950.cpp
│           ├── quant_matmul_all_reduce_tiling_950.cpp
│           └── weight_quant_matmul_all_reduce_tiling_950.cpp
├── op_api/                                 (ACLNN API)
│   ├── aclnn_matmul_all_reduce.h/cpp       V1 API
│   ├── aclnn_matmul_all_reduce_v2.h/cpp    V2 API
│   ├── aclnn_quant_matmul_all_reduce.h/cpp       量化 V1
│   ├── aclnn_quant_matmul_all_reduce_v2.h/cpp    量化 V2
│   ├── aclnn_quant_matmul_all_reduce_v3.h/cpp    量化 V3
│   ├── aclnn_quant_matmul_all_reduce_v4.h/cpp    量化 V4
│   ├── aclnn_weight_quant_matmul_all_reduce.h/cpp  权重伪量化
│   └── matmul_all_reduce_util.h/cpp        共享工具函数
└── tests/ut/                               单元测试
```

---

## 3. Host 侧分析

### 3.1 算子定义 (`matmul_all_reduce_def.cpp`)

**输入张量** (8 个, 含 6 个可选):

| 序号 | 名称 | 类型 | 必选 | 说明 |
|------|------|------|------|------|
| 0 | x1 | FP16/BF16/INT8/HIFLOAT8/FP8_E4M3/FP8_E5M2/INT4 | 是 | 激活矩阵 [M, K] 或 [S, M, K] |
| 1 | x2 | FP16/BF16/INT8/INT4/HIFLOAT8/FP8_E4M3/FP8_E5M2 | 是 | 权重矩阵 [K, N] |
| 2 | bias | FP16/BF16/INT32/FLOAT | 否 | 偏置 [N] |
| 3 | x3 | FP16/BF16 | 否 | 残差输入 (Add 融合) |
| 4 | antiquant_scale | FP16/BF16/FLOAT | 否 | 反量化缩放因子 |
| 5 | antiquant_offset | FP16/BF16/FLOAT | 否 | 反量化偏移 |
| 6 | dequant_scale | UINT64/FLOAT/BF16/FLOAT8_E8M0 | 否 | 通信量化缩放 |
| 7 | pertoken_scale | FLOAT/FLOAT8_E8M0 | 否 | PerToken 缩放 |

**输出张量** (1 个):

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y | FP16/BF16/FLOAT | AllReduce 后输出 [S, M, N] |

**属性** (9 个):

| 序号 | 名称 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| 0 | group | String | - | HCCL 通信组名 (必需) |
| 1 | reduce_op | String | "sum" | 归约操作 |
| 2 | is_trans_a | Bool | false | A 矩阵转置 (当前不支持) |
| 3 | is_trans_b | Bool | false | B 矩阵转置 |
| 4 | comm_turn | Int | 0 | 通信轮次 |
| 5 | antiquant_group_size | Int | 0 | 反量化分组大小 |
| 6 | group_size | Int | 0 | 通信量化分组大小 |
| 7 | y_dtype | Int | DT_UNDEFINED | 输出数据类型 |
| 8 | comm_quant_mode | Int | 0 | 通信量化模式 |

**平台适配**:
- `ascend950`: 支持 66 种数据类型组合 (含 HIFLOAT8/FP8/E2M1)
- `ascend910b`: 支持 16 种数据类型组合
- `ascend310p`: 支持 6 种数据类型组合 (仅 FP16)

### 3.2 Shape 推导 (`matmul_all_reduce_infershape.cpp`)

```
输入:
  x1: [S, M, K] 或 [M, K]     (dim 2~3)
  x2: [K, N] 或 [N, K] (transB)  (dim 2)

推导:
  K 检查: x1.dim(-1) == x2.dim(transB ? 0 : 1)
  输出: y = [S, M, N] 或 [M, N]

输出数据类型推导:
  - y_dtype 属性指定 → 使用指定类型
  - INT8×INT8 + dequant=BF16 → 输出 BF16
  - INT8×INT8 + 其他 → 输出 FP16
  - 其他 → 同 x1 类型
```

### 3.3 Tiling 策略

**Tiling 入口** (`matmul_all_reduce_tiling.cpp`, 第29-37行):
```cpp
// 按架构分发
if (npuArch == DAV_3510) {
    TilingRegistryArch::GetInstance().DoTilingImpl(context);  // 950: 新架构 Tiling
} else {
    TilingRegistryNew::GetInstance().DoTilingImpl(context);   // 910B/310P: 标准 Tiling
}
```

**Tiling 基类** (`MatmulAllReduceTilingBase`, `matmul_all_reduce_tiling_base.h`):

```
MatmulAllReduceTilingBase 继承 TilingBaseClass:
├── GetPlatformInfo()      → 获取 SoC 版本、AIC/AIV 核数
├── GetShapeAttrsInfo()    → 解析输入 Shape 和属性
│   └── AnalyzeShapeAttr() → 提取 M/K/N/S、量化参数
├── IsCapable()            → 子类实现: 判断当前 Tiling 模板是否适用
├── DoOpTiling()           → 子类实现: 计算 Tiling 参数
│   ├── DoSplitMTiling()       → M 轴切分 (tileCnt + tailCnt)
│   ├── DoMatmulTiling()       → 调用 MultiCoreMatmulTiling
│   ├── DoAllReduceTiling()    → AllReduce 公式化 Tiling
│   └── DoL2CacheTiling()      → L2 Cache 分块参数
├── DoLibApiTiling()       → HCCL API Tiling
├── GetWorkspaceSize()     → 计算 Workspace 大小
└── PostTiling()           → 序列化 TilingData + 设置 TilingKey
```

**场景枚举** (`AllReduceScenario`, 第102-111行):

| 场景 | 枚举值 | 条件 |
|------|--------|------|
| A8W8 | 0 | x1=INT8, x2=INT8 |
| A16W8 | 1 | x1=FP16/BF16, x2=INT8/FP8 |
| A16W4 | 2 | x1=FP16/BF16, x2=INT4/E2M1 |
| FP8HIF8 | 3 | x1=FP8/HIF8, x2=FP8/HIF8 |
| MXFP4 | 4 | MX 格式 FP4 |
| MXFP8 | 5 | MX 格式 FP8 |

### 3.4 ACLNN API

提供 5 个 API 版本:
- `aclnnMatmulAllReduce` / `aclnnMatmulAllReduceV2` — 非量化
- `aclnnQuantMatmulAllReduce` / `V2` / `V3` / `V4` — 全量化
- `aclnnWeightQuantMatmulAllReduce` — 权重伪量化

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与分发 (`matmul_all_reduce.cpp`)

**顶层入口** `matmul_all_reduce<>` (第342-590行):

```
matmul_all_reduce<>(aGM, bGM, biasGM, addGM, ..., tilingGM)
│
├── [workspaceGM == nullptr] → 直接返回 (空算子保护)
├── [userWS == nullptr] → 直接返回
│
├── REGISTER_TILING_DEFAULT(WeightQuantMatmulAllReduceTilingData)  ← 最大结构体
├── REGISTER_TILING_FOR_TILINGKEY(...) ← 7 个条件分支注册不同 TilingData
│
├── TPipe tPipe;  ← 管道管理器
│
├── [910B] ─┬─ MM_TYPE == FP_MM → fp_matmul_all_reduce<>()
│           ├─ MM_TYPE == QUANT → quant_matmul_allreduce<>()
│           └─ MM_TYPE == WEIGHT_QUANT → weight_quant_matmul_allreduce<>()
│
└── [310P] ─┬─ INT8×INT8 → MatmulAllReduceQuantBmm (全量化)
            ├─ FP16×INT8,NZ → MatmulAllReduceWeightQuant310 (权重 NZ)
            ├─ FP16×FP16,NZ → MatmulAllReduceUnquant310 (非量化 NZ)
            └─ 其他 → MatmulAllReduce310General (通用 ND 路径)
```

### 4.2 非量化路径: MatmulAllReduce910General

**文件**: `arch32/matmul_all_reduce_910_general.h`

**模板参数**:
```cpp
template <typename xType, typename wType, typename yType, class mmType, Mc2CoreType coreType>
```

**Init 流程** (继承自 `MatmulAllReduceBase`, `matmul_all_reduce_base.h` 第49-86行):

```
Init():
├── hccl_.Init(GetHcclContext<0>())          ← 初始化 HCCL 上下文
├── OOMInit(context)                          ← 注册地址范围检查
├── [useBufferType == WINDOW_IN] → cGM 重定向到 Window In 地址
├── 计算 tile/tail 的地址偏移:
│   ├── aAddrOffset = M × Ka × sizeof(xType)
│   └── cAddrOffset = M × N  × sizeof(yType)
└── [notifyFlag_] → 注册 AllReduce:
    ├── tile: hccl_.AllReduce(cGM, outputGM, cOffset, HCCL_REDUCE_SUM, tileCnt)
    └── tail: hccl_.AllReduce(cGM+offset, outputGM+offset, cOffset, ...)
```

**Process 流程** (第57-69行):

```
Process():
├── [BF16] PreProcForBiasOnVector()    ← BF16 bias 转 FP32 (需额外 UB)
├── InnerProcess(false, tileCnt, tileInfo_)
│   ├── for i in [0, tileCnt):
│   │   ├── [block_idx < usedCoreNum] → MM Init/UpdateGlobalTensor + Process
│   │   └── PostProcEachTurn()         ← Add + 地址推进 + SyncAll + hccl_.Commit
│   └── 注意: Commit 后 AllReduce 异步执行，下个 Tile 的 MM 可继续
├── [tailFlag_] InnerProcess(true, tailCnt, tailInfo_)
└── HcclFinalize()                     ← Wait + Finalize
```

**核心: 通信-计算流水线** (`PostProcEachTurn`, 第105-121行):

```
Tile_i: MM_i 计算
         ↓
PostProcEachTurn():
├── [addFlag_] MatMul_All_Reduce_Add_X3(cGM, addGM)  ← 残差加法
├── aGM += aOffset, cGM += cOffset                    ← 地址推进
├── Mc2SyncAll<coreType>()                             ← 核间同步
└── [notifyFlag_] hccl_.Commit(handleId)               ← 提交 AllReduce

时间线:
┌─────────────────────────────────────────────────┐
│ Tile_0: [MM_0][Add][Commit_0]                    │
│ Tile_1:           [MM_1][Add][Commit_1]          │
│ Tile_2:                      [MM_2][Add][Commit_2]│
│                                                  │
│ AllReduce_0:    [======= AR_0 ========]          │
│ AllReduce_1:           [======= AR_1 ========]   │
│ AllReduce_2:                    [===== AR_2 ====] │
└─────────────────────────────────────────────────┘
```

### 4.3 全量化路径: BmmDequant / BmmDequantBf16 / BmmDequantPertoken

**文件**: `arch32/matmul_all_reduce_quant.h` 等

**INT8 通信量化** (`INT8_COMM == T`):
- 将 AllReduce 的通信数据从 FP16/BF16 量化为 INT8，减少通信带宽
- 量化公式: `INT8_val = FP_val × comm_quant_scale`
- 反量化: `FP_out = INT8_val × comm_quant_scale`

**核心类型组合** (quant_matmul_allreduce, 第155-257行):

| INT8_COMM | TRANS_B | PERTOKEN | 核心类 | 输出类型 |
|-----------|---------|----------|--------|---------|
| F | F | F | BmmDequant | FP16/BF16 |
| F | T | F | BmmDequant | FP16/BF16 |
| F | F | T | BmmDequantPertoken | FP16/BF16 |
| F | T | T | BmmDequantPertoken | FP16/BF16 |
| T | F | F | MatmulAllReduceQuantFP16CommInt8 / BF16CommInt8 | FP16/BF16 |
| T | T | F | 同上 | FP16/BF16 |
| T | F | T | MatmulAllReduceQuantPertokenInt8 | FP16/BF16 |
| T | T | T | MatmulAllReduceQuantPertokenInt8 | FP16/BF16 |

### 4.4 权重伪量化路径

**文件**: `arch32/matmul_all_reduce_weight_quant.h`

**反量化类型** (`ANTIQUANT_TYPE`):

| 类型 | 说明 | 缩放因子 Shape |
|------|------|---------------|
| PER_TENSOR | 整体缩放 | [1] |
| PER_CHANNEL | 逐通道 | [N] |
| PER_GROUP | 逐分组 | [ceil(K/group), N] |

**Weight NZ 格式** (`SUB_ALGORITHM == WEIGHT_NZ`):
- 权重以 FRACTAL_NZ 格式存储，硬件 Cube 单元可直接计算
- 仅支持 `PER_CHANNEL` 反量化

### 4.5 数据结构与内存布局

#### TilingData 结构层次

```
统一布局 (以 MatmulAllReduce910TilingData 为例):
┌─────────────────────────────────────────────────────────┐
│ Mc2Msg msg                  ← 通信元数据 (rankId, rankDim, ...) │
│ RCSTiling param             ← 流水线参数 (tileCnt, tailCnt, ...) │
│ Mc2MatmulV3TilingData       ← Tile 部分 MM Tiling             │
│   └── TCubeTiling           ← Cube Tiling (M, N, Ka, ...)     │
│ Mc2MatmulV3TilingData       ← Tail 部分 MM Tiling             │
│   └── TCubeTiling           ← Cube Tiling                     │
└─────────────────────────────────────────────────────────┘

QuantMatmulAllReduceTilingData:
├── Mc2Msg msg
├── RCSTiling param
├── Mc2QuantBatchMatmulV3TilingData tilematmulTiling  ← 量化 MM Tile
└── Mc2QuantBatchMatmulV3TilingData tailmatmulTiling  ← 量化 MM Tail

WeightQuantMatmulAllReduceTilingData:
├── Mc2Msg msg
├── RCSTiling param
├── Mc2WeightQuantBatchMatmulV2TilingData tilematmulTiling  ← 权重量化 Tile
└── Mc2WeightQuantBatchMatmulV2TilingData tailmatmulTiling  ← 权重量化 Tail
```

#### GM 地址结构 (`common.h`, 第209-253行)

```
MC2GmAddrs {
    aGM        ← 激活输入 (按 Tile 推进)
    bGM        ← 权重 (固定)
    biasGM     ← 偏置 (BF16 时可能重定向到 workspace)
    addGM      ← 残差输入 (按 Tile 推进)
    cGM        ← MM 输出 / AllReduce 输入
    workspaceGM ← 工作空间 (BF16 bias 转换区)
    outputGM   ← 最终输出
}

MC2TileInfo {
    mmTiling          ← 当前 Tile 的 MM Tiling 指针
    hcclHandleId      ← AllReduce Handle
    aOffset/aAddrOffset ← 输入地址步长
    cOffset/cAddrOffset ← 输出地址步长
}
```

### 4.6 多核并行策略

| 平台 | 核类型 | Mc2CoreType | 说明 |
|------|--------|-------------|------|
| 910B FP_MM | MIX_AIC_1_2 | ON_CUBE_AND_VECTOR | 1 AIC + 2 AIV 协同 |
| 910B FP_MM_CUBE_ONLY | MIX_AIC_1_0 | ON_CUBE | 仅 Cube 计算 |
| 910B QUANT (BF16 out) | MIX_AIC_1_1 | ON_CUBE_AND_VECTOR | 1 AIC + 1 AIV |
| 910B QUANT (INT8 comm) | MIX_AIC_1_2 | ON_VECTOR | 通信量化需要 Vector |
| 910B WEIGHT_QUANT | MIX_AIC_1_2 | ON_CUBE_AND_VECTOR | 标准 1+2 模式 |
| 910B Empty Tensor | MIX_AIV_1_0 | ON_VECTOR | 空张量仅 Vector |
| 310P | MIX_AIC_1_2 | ON_CUBE_AND_VECTOR | 同 910B |

**核角色判定** (`common.h`, 第256-261行):

```cpp
enum class Mc2CoreType {
    ON_CUBE_AND_VECTOR = 0,  // 同时使用 AIC 和 AIV
    ON_VECTOR = 1,           // 仅使用 AIV
    ON_CUBE = 2,             // 仅使用 AIC
};
```

**同步机制** (`Mc2SyncAll`, 第335-346行):

```cpp
Mc2SyncAll():
├── ON_CUBE_AND_VECTOR → SyncAll<false>()  ← 轻量同步 (不等待所有核)
├── ON_VECTOR → SyncAll()                  ← 标准同步
└── ON_CUBE → PipeBarrier + CrossCoreSetFlag(3) + WaitFlagDevLocal(3)
```

### 4.7 BF16 Bias 预处理

**文件**: `common.h`, 第277-327行

BF16 bias 无法直接被 Cube 单元使用，需要在 AIV0 上转换为 FP32:

```
PreProcForBiasOnVector():
├── [biasLen == 0] → 返回
├── TBuf tmpBuf(TOTAL_UB_SIZE)
├── CastBFtoFloatOnAiv0(workspaceGM, biasGM, rankN, tmpBuf):
│   ├── for offset in [0, size, tmpBufCount):
│   │   ├── DataCopyPad(xLocal, gmSrc)      ← BF16 GM → UB
│   │   ├── SetFlag/WaitFlag(MTE2_V)
│   │   ├── Cast(yLocal, xLocal, CAST_NONE)  ← BF16 → FP32
│   │   ├── SetFlag/WaitFlag(V_MTE3)
│   │   └── DataCopyPad(gmDst, yLocal)      ← FP32 UB → GM
├── SyncAll<false>()           ← 等待 AIV0 完成
├── biasGM = workspaceGM       ← 重定向 bias 到 FP32 版本
└── workspaceGM += biasLen     ← 推进 workspace 指针
```

---

## 5. Host-Kernel 接口

### 5.1 TilingData 注册

Kernel 入口通过 `REGISTER_TILING_*` 注册所有可能用到的 TilingData 结构:

```cpp
REGISTER_TILING_DEFAULT(WeightQuantMatmulAllReduceTilingData);  // 最大结构体
REGISTER_TILING_FOR_TILINGKEY("MM_TYPE == QUANT", QuantMatmulAllReduceTilingData);
REGISTER_TILING_FOR_TILINGKEY("...WEIGHT_QUANT... && ND", WeightQuantMatmulAllReduceTilingData);
REGISTER_TILING_FOR_TILINGKEY("...WEIGHT_QUANT... && NZ", WeightQuantMatmulAllReduceNzTilingData);
REGISTER_TILING_FOR_TILINGKEY("...FP_MM && 910B", MatmulAllReduce910TilingData);
REGISTER_TILING_FOR_TILINGKEY("...FP_MM && 310P && ND", UnQuantMatmulAllReduceTilingData);
REGISTER_TILING_FOR_TILINGKEY("...FP_MM && 310P && NZ", MatmulAllReduceTilingData);
```

### 5.2 Workspace 布局

```
Workspace:
┌───────────────────────────────────────────────────────┐
│ HCCL 工作空间 (由 AllReduce 公式化 Tiling 计算)        │
├───────────────────────────────────────────────────────┤
│ [BF16] Bias FP32 转换缓冲区 (biasLen = rankN × 4)     │
├───────────────────────────────────────────────────────┤
│ UserWorkspace (由 GetUserWorkspace 获取)               │
│ ├── MM 工作空间 (MatMul Tiling 计算)                   │
│ └── 量化 MM 工作空间                                   │
└───────────────────────────────────────────────────────┘
```

---

## 6. 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ Host Tiling 阶段                                                 │
│                                                                  │
│ x1[S,M,K], x2[K,N] → AnalyzeShapeAttr() → 提取 M/K/N/S         │
│    → AnalyzeInputs() → 判定量化场景 (A8W8/A16W8/A16W4/FP)         │
│    → DoSplitMTiling() → 计算 tileCnt, tailCnt, tileM, tailM       │
│    → DoMatmulTiling() → 计算 Tile/Tail 的 TCubeTiling            │
│    → DoAllReduceTiling() → 计算 HCCL 参数 (rankDim, dataType)    │
│    → GetWorkspaceSize() → 计算 WS 总量                           │
│    → PostTiling() → 设置 TilingKey + memcpy TilingData           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Kernel 执行阶段                                                  │
│                                                                  │
│ matmul_all_reduce<>                                              │
│ ├── workspace 空指针检查                                         │
│ ├── TPipe tPipe 创建                                             │
│ ├── [910B] 根据 MM_TYPE 分发到 fp/quant/weight_quant 子入口      │
│ │                                                                │
│ │ 以 fp_matmul_all_reduce (非量化) 为例:                          │
│ │                                                                │
│ │ MatmulAllReduce910General.Init():                              │
│ │ ├── HCCL 初始化                                                │
│ │ ├── [BF16] Bias BF16→FP32 预处理                               │
│ │ ├── 计算 Tile/Tail 地址偏移                                     │
│ │ └── 注册 AllReduce Handle                                      │
│ │                                                                │
│ │ MatmulAllReduce910General.Process():                           │
│ │ ├── for i in [0, tileCnt):                                     │
│ │ │   ├── McMatmulBaseBlock.Init/Update + Process()  ← MM 计算  │
│ │ │   └── PostProcEachTurn()                         ← 流水线推进│
│ │ │       ├── [add] 残差加法                                      │
│ │ │       ├── 地址推进 (aGM += aOffset, cGM += cOffset)          │
│ │ │       ├── Mc2SyncAll()                                        │
│ │ │       └── hccl_.Commit()  ← 触发 AllReduce (异步)           │
│ │ │                                                               │
│ │ ├── [tail] 同上流程处理尾部                                     │
│ │ └── HcclFinalize()  ← Wait AllReduce 完成                     │
│ │                                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    y [S, M, N]  (AllReduce 后结果)
```

---

## 7. 性能相关设计

### 7.1 通信-计算流水线

核心优化: 将 M 轴切分为 `tileCnt` 个 Tile，每个 Tile 的 MM 计算与上一个 Tile 的 AllReduce 重叠:

```
无流水线:
[MM全部][==========AllReduce全部==========]

有流水线:
[MM0][MM1][MM2][MM3]
     [AR0 ][AR1 ][AR2 ][AR3 ]
总时间 ≈ MM总 + AR单次 (而非 MM总 + AR总)
```

### 7.2 L2 Cache 优化

通过 `ENABLE_L2_CACHE` 开关，将大矩阵按 L2 Cache 大小进一步分块，提升数据局部性:

```
L2TilePara {
    mTileCntL2;    ← M 轴 L2 分块数
    nTileCntL2;    ← N 轴 L2 分块数
    mTileBlock;    ← 每块 M 大小
    nTileBlock;    ← 每块 N 大小
    calOrder;      ← 计算顺序 (行列优先)
}
```

### 7.3 Window In 输出优化

当 `useBufferType == MC2_BUFFER_TYPE_WINDOW_IN` 时，将 MM 输出直接写入 HCCL Window In 地址，避免一次 GM 拷贝:

```cpp
// Init 中的优化路径:
if (useBufferType == WINDOW_IN && !determinism) {
    cGM = hccl_.GetWindowsInAddr(rankId);  // 输出直接写 Window
}
```

### 7.4 INT8 通信量化

在多机场景下，AllReduce 通信带宽成为瓶颈。通过 `INT8_COMM` 开关将 FP16/BF16 数据量化为 INT8 后通信:

```
MM输出(FP16/BF16) → × comm_quant_scale → INT8 → AllReduce → INT8 → × comm_quant_scale → FP16/BF16
```

通信数据量减半，代价是额外的量化/反量化计算。

### 7.5 Empty Tensor 快速路径

当 K=0 时，无需实际计算，直接执行空 AllReduce:

```
[EMPTY_INPUT == T]:
├── FP_MM → MatMulEmptyTensorKernel
├── WEIGHT_QUANT NZ → WeightQuantEmptyTensorKernel
└── 使用 MIX_AIV_1_0 (仅 Vector，不占用 Cube)
```

---

## 8. 量化与精度分析

### 8.1 量化场景分类

| 场景 | x1 类型 | x2 类型 | MM 累加类型 | 输出类型 | 通信数据类型 |
|------|---------|---------|------------|---------|------------|
| A16W16 (非量化) | FP16/BF16 | FP16/BF16 | FP32 | FP16/BF16 | FP16/BF16 |
| A8W8 (全量化) | INT8 | INT8 | INT32 | FP16/BF16 | FP16/BF16 或 INT8 |
| A16W8 (权重量化) | FP16/BF16 | INT8/FP8 | - | FP16/BF16 | FP16/BF16 |
| A16W4 (INT4 量化) | FP16/BF16 | INT4/E2M1 | - | FP16/BF16 | FP16/BF16 |
| FP8/HIF8 | HIFLOAT8/FP8 | HIFLOAT8/FP8 | - | FLOAT | FLOAT |

### 8.2 全量化 (A8W8) 详细管线

```
量化 MM 管线:
INT8 × INT8 → INT32 (Cube 累加) → × dequant_scale → FP16/BF16 → AllReduce → FP16/BF16

PerToken 动态量化额外步骤:
INT32 → × dequant_scale → × pertoken_scale → FP16/BF16

INT8 通信量化:
FP16/BF16 → × comm_quant_scale_1 → INT8 → AllReduce(INT8) → INT8 → × comm_quant_scale_2 → FP16/BF16
```

### 8.3 权重伪量化 (A16W8/A16W4) 管线

```
INT8/INT4 权重 → × antiquant_scale [+ antiquant_offset] → FP16/BF16 → MM (FP16×FP16/BF16×BF16)
                                                                                ↓
                                                                          AllReduce → 输出

PerTensor: scale 为标量
PerChannel: scale 为 [N] 向量
PerGroup: scale 为 [ceil(K/group), N] 矩阵
```

---

## 9. 关键约束与注意事项

1. **A 矩阵不支持转置**: `is_trans_a` 属性虽然存在但强制为 false (infershape 第81-82行)
2. **BF16 Bias 需额外处理**: Cube 单元不支持 BF16 bias，需 AIV0 预转为 FP32，占用额外 workspace
3. **TilingKey 空间巨大**: 15 个模板参数的组合空间达数千种，但实际合法组合通过 `ASCENDC_TPL_SEL` 精确枚举
4. **310P 架构差异**: 使用 `HcclServer` (RAC) 而非 910B 的 HCCL AICPU 通信
5. **Workspace 使用**: `GetUserWorkspace(workspaceGM)` 获取的用户工作空间为空时整个算子跳过
6. **Window In 优化**: 仅在非确定性模式下启用 (`config.determinism != 1`)

---

## 10. 函数调用关系

```
matmul_all_reduce<> (Kernel 主入口)
├── REGISTER_TILING_DEFAULT + REGISTER_TILING_FOR_TILINGKEY
├── GetUserWorkspace 空指针检查
│
├── [910B, FP_MM]
│   └── fp_matmul_all_reduce<>
│       ├── [CUBE_ONLY] INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseKernel, ON_CUBE)
│       ├── [EMPTY] INVOKE_MC2_EMPTY_TENSOR_OP_IMPL
│       ├── [!MIXND2NZ] INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseKernel, ON_CUBE_AND_VECTOR)
│       └── [MIXND2NZ] INVOKE_MC2_910_OP_IMPL(Mc2MatmulBaseUnAlignedKernel, ON_CUBE_AND_VECTOR)
│
├── [910B, QUANT]
│   └── quant_matmul_allreduce<>
│       ├── [BF16 out, !INT8_COMM, !TRANS, !PERTOKEN] → BmmDequant
│       ├── [BF16 out, !INT8_COMM, TRANS, PERTOKEN] → BmmDequantPertoken
│       ├── [BF16 out, INT8_COMM, !PERTOKEN] → MatmulAllReduceQuantBF16CommInt8
│       └── [BF16 out, INT8_COMM, PERTOKEN] → MatmulAllReduceQuantPertokenInt8
│
├── [910B, WEIGHT_QUANT]
│   └── weight_quant_matmul_allreduce<>
│       ├── [EMPTY] INVOKE_MC2_EMPTY_TENSOR_OP_IMPL
│       ├── [CUSTOM_ANTIQUANT] INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(transB, quantType, hasOffset)
│       │   └── 12 种组合: PerTensor/PerChannel/PerGroup × {有无Offset} × {B转置/非转置}
│       └── [WEIGHT_NZ] INVOKE_MC2_WEIGHT_QUANT_910_OP_IMPL(transB, PerChannel, hasOffset)
│
└── [310P] ─┬─ INT8×INT8 → MatmulAllReduceQuantBmm
            ├─ FP16×INT8,NZ → MatmulAllReduceWeightQuant310
            ├─ FP16×FP16,NZ → MatmulAllReduceUnquant310
            └─ 通用ND → MatmulAllReduce310General (16+ 种组合)

MatmulAllReduceBase<>::Init():
├── hccl_.Init()
├── OOMInit()
├── [WindowIn] cGM 重定向
├── 计算地址偏移
└── hccl_.AllReduce() 注册

MatmulAllReduce910General::Process():
├── [BF16] PreProcForBiasOnVector()
│   └── CastBFtoFloatOnAiv0() → Cast BF16→FP32
├── InnerProcess(false, tileCnt, tileInfo_)
│   ├── for i: McMatmulBaseBlock.Init/Process()
│   └── PostProcEachTurn()
│       ├── [add] MatMul_All_Reduce_Add_X3()
│       ├── 地址推进
│       ├── Mc2SyncAll()
│       └── hccl_.Commit()
├── [tail] InnerProcess(true, tailCnt, tailInfo_)
└── HcclFinalize()
    ├── hccl_.Wait()
    ├── Mc2SyncAll()
    └── hccl_.Finalize()
```

---

## 11. 依赖关系

```
matmul_all_reduce.cpp
├── common.h
│   ├── mc2_tiling_struct.h         ← Mc2Msg, RCSTiling
│   ├── mc2_kernel_utils.h          ← Kernel 工具函数
│   └── lib/hccl/hccl.h            ← HCCL 通信接口
├── matmul_all_reduce_tiling_key.h  ← 模板参数宏 + ASCENDC_TPL 声明
│   ├── arch32/quant_matmul_all_reduce_tiling_data.h
│   ├── arch32/unquant_matmul_all_reduce_tiling_data.h
│   └── arch32/weight_quant_matmul_all_reduce_tiling_data.h
│
├── [910B 非量化]
│   ├── arch32/matmul_all_reduce_910_general.h
│   │   ├── matmul_all_reduce_base.h
│   │   │   └── matmul_all_reduce_add_x3.h
│   │   ├── 3rd/mat_mul_v3/mat_mul_base_kernel.h
│   │   └── 3rd/mat_mul_v3/mat_mul_unaligned_base_kernel.h
│   └── arch32/matmul_all_reduce_empty_tensor_k_general.h
│
├── [910B 全量化]
│   ├── arch32/matmul_all_reduce_quant.h
│   ├── arch32/matmul_all_reduce_quant_bf16_comm_int8.h
│   ├── arch32/matmul_all_reduce_quant_fp16_comm_int8.h
│   └── arch32/matmul_all_reduce_quant_pertoken_comm_int8.h
│
├── [910B 权重量化]
│   └── arch32/matmul_all_reduce_weight_quant.h
│
└── [310P]
    ├── arch31/rac_server.h          ← 310P RAC 通信服务
    ├── arch31/matmul_all_reduce_quant_bmm.h
    ├── arch31/matmul_all_reduce_unquant_310.h
    ├── arch31/matmul_all_reduce_weight_quant_310.h
    └── arch31/matmul_all_reduce_310_general.h
```

---

*分析完成时间: 2026-04-17*
