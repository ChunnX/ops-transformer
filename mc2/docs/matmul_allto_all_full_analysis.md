# matmul_allto_all 算子全面分析报告

## 1. 算子概述

`matmul_allto_all` 实现 **MatMul + AllToAll** 融合。在 TP (Tensor Parallel) 推理场景中，每个 Rank 本地计算部分 MatMul，再通过 AllToAll 将结果沿 N 轴切分分发到对应 Rank。910B 平台采用 **Catlass 模板线性代数库** + **Window IPC 点对点通信**；950/910_93 平台采用 mc2_templates Stage 化架构。

```
x1 [M, K] × x2 [K, N] (本地 MatMul)
        ↓ Catlass / mc2_templates 计算
   mm_result [M, N]
        ↓ AllToAll (N 轴按 rankSize 切分)
   每个 Rank 得到 [M × rankSize, N/rankSize]
```

**设计模式**: 910B 使用 Catlass `MatmulAlltoAllKernel` 将 Cube MM 结果直接写入 Window IPC 共享内存，AIV 核通过 `CopyGMToGM` 完成跨 Rank 数据搬运。

### 核心设计特点

- **Catlass 模板库**: 910B 平台使用第三方模板线性代gebra 库 `catlass`，实现高效的分块 MM + Epilogue + AllToAll
- **Window IPC 通信**: 利用 HCCL 预注册的 Window 共享内存区域，各 Rank 可直接读写对端数据
- **AIC+AIV 协同**: AIC (Cube) 计算 MM，AIV (Vector) 负责数据搬运和同步
- **非量化 + 量化双路径**: 非量化走 `CatlassMatmul()`，量化走 `QuantCatlassMatmul()`
- **Flag 同步机制**: 通过 Window 共享内存中的 Flag 区域实现跨 Rank 同步

### 模板参数体系

```
910B 模板参数 (3 个):
┌──────────────────────────────────────────┐
│ MM_ALLTO_ALL_TRANS_X2   │ B 矩阵转置     │
│ MM_ALLTO_ALL_HAS_BIAS   │ 是否有偏置      │
│ MM_ALLTO_ALL_QUANT_BF16 │ 量化输出 BF16   │
└──────────────────────────────────────────┘

合法组合:
  (TRANS_X2=0/1, HAS_BIAS=0/1, QUANT_BF16=0)  → 4 种非量化
  (TRANS_X2=0/1, HAS_BIAS=1,   QUANT_BF16=1)  → 2 种量化 (必须有 Bias)
```

---

## 2. 目录结构与文件职责

```
matmul_allto_all/
├── op_kernel/
│   ├── matmul_allto_all.cpp               (46 行) Kernel 主入口
│   ├── matmul_allto_all_apt.cpp           APT 入口
│   └── arch32/                             (910B 内核实现)
│       ├── matmul_allto_all.h              (395 行) MatmulAlltoAll 主类: Init/Process
│       ├── matmul_allto_all_tiling.h       Tiling 数据结构定义
│       ├── matmul_allto_all_tiling_key.h   TilingKey 宏定义 (3 参数)
│       ├── matmul_allto_all_util.h         (249 行) CommBase 通信工具类
│       ├── moe_distribute_base.h           HCCL 参数结构
│       ├── matmul.hpp                      Catlass MM 辅助
│       ├── quant_matmul.hpp                Catlass 量化 MM 辅助
│       ├── block_epilogue_dequant.hpp      量化反量化 Epilogue
│       └── tile_broadcast_add.hpp          广播加法 Epilogue
├── op_host/
│   ├── matmul_allto_all_def.cpp           算子定义: 8 输入 + 1 输出 + 10 属性
│   ├── matmul_allto_all_infershape.cpp    Shape 推导: M,K × K,N → M×R, N/R
│   └── op_tiling/
│       ├── matmul_allto_all_tiling.cpp     Tiling 入口
│       ├── matmul_allto_all_tiling_base.h/cpp  基类
│       ├── common/
│       │   ├── allto_all_formulaic_tiling.h/cpp  AllToAll 公式化 Tiling
│       │   └── matmul_allto_all_util_tiling.h/cpp  共享 Tiling 工具
│       ├── arch32/                          910B/910_93 Tiling
│       │   ├── matmul_allto_all_tiling_910b.h/cpp
│       │   └── fp_matmul_allto_all_tiling_910_93.h/cpp
│       └── arch35/                          950 Tiling
│           ├── fp_matmul_allto_all_tiling_base.h/cpp
│           ├── kc_quant_matmul_allto_all_tiling_base.h/cpp
│           └── mx_quant_matmul_allto_all_tiling_base.h/cpp
└── op_api/
    ├── aclnn_matmul_allto_all.h/cpp        非量化 API
    ├── aclnn_quant_matmul_allto_all.h/cpp  量化 API
    └── matmul_allto_all_util.h/cpp         共享工具
```

---

## 3. Host 侧分析

### 3.1 算子定义 (`matmul_allto_all_def.cpp`)

**输入张量** (8 个, 含 6 个可选):

| 序号 | 名称 | 类型 | 必选 | 说明 |
|------|------|------|------|------|
| 0 | x1 | FP16/BF16/INT8/FP8_E4M3/FP8_E5M2/FLOAT4_E2M1 | 是 | 激活矩阵 [M, K] |
| 1 | x2 | FP16/BF16/INT8/FP8_E4M3/FP8_E5M2/FLOAT4_E2M1 | 是 | 权重矩阵 [K, N] |
| 2 | bias | FLOAT/FP16/BF16 | 否 | 偏置 [N] |
| 3 | x1_scale | FLOAT/FLOAT8_E8M0 | 否 | x1 量化缩放 |
| 4 | x2_scale | FLOAT/FLOAT8_E8M0 | 否 | x2 量化缩放 |
| 5 | comm_scale | FLOAT | 否 | 通信量化缩放 |
| 6 | x1_offset | FLOAT | 否 | x1 量化偏移 |
| 7 | x2_offset | FLOAT | 否 | x2 量化偏移 |

**输出张量** (1 个):

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y | FP16/BF16/FLOAT | AllToAll 后输出 [M×rankSize, N/rankSize] |

**属性** (10 个):

| 序号 | 名称 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| 0 | group | String | - | HCCL 通信组 (必需) |
| 1 | world_size | Int | - | Rank 总数 (必需, 支持 2/4/8/16) |
| 2 | all2all_axes | ListInt | [-1,-2] | AllToAll 切分轴: [-1,-2] |
| 3 | y_dtype | Int | DT_UNDEFINED | 输出类型 |
| 4 | x1_quant_mode | Int | 0 | x1 量化模式 (0=无, 3=PerToken, 6=MX) |
| 5 | x2_quant_mode | Int | 0 | x2 量化模式 (0=无, 2=PerChannel, 6=MX) |
| 6 | comm_quant_mode | Int | 0 | 通信量化模式 |
| 7 | comm_quant_dtype | Int | DT_UNDEFINED | 通信量化类型 |
| 8 | transpose_x1 | Bool | false | x1 转置 (不支持) |
| 9 | transpose_x2 | Bool | false | x2 转置 |
| 10 | group_size | Int | 0 | 量化分组大小 |

**平台适配**:
- `ascend950`: 31 种数据类型组合 (含 FP8/E2M1/MX)
- `ascend910_93`: 2 种 (FP16/BF16 非量化)
- `ascend910b`: 5 种 (FP16/BF16/INT8 + 量化)

### 3.2 Shape 推导 (`matmul_allto_all_infershape.cpp`)

```
输入:
  x1: [M, K]     (dim 2)
  x2: [K, N] 或 [N, K] (transX2)

推导:
  M = x1.dim(0), K = x1.dim(1)
  N = transX2 ? x2.dim(0) : x2.dim(1)
  rankSize ∈ {2, 4, 8, 16} (强制校验)

输出:
  y: [M × rankSize, N / rankSize]
  
  注意: 当 M == -1 (动态shape) → y = [-1, N]
```

### 3.3 Tiling 策略

**Tiling 基类** (`MatmulAllToAllTilingBase`):
```
├── GetPlatformInfo()     → 获取 SoC 版本
├── GetShapeAttrsInfo()   → 空实现 (校验在子类 DoOpTiling)
├── DoOpTiling()          → 子类: TileCommAndCompute()
│   └── 计算 m0, k0, n0, p_value, ubMoveNum
├── GetWorkspaceSize()    → 计算 HCCL + User WS
└── GetTilingKey()        → 子类: 根据量化/bias/trans 参数编码
```

**CoCTiling 参数** (`matmul_allto_all_tiling.h`, 第25-31行):

```
CoCTiling {
    m0;    ← M 轴分块大小 (128 或 256)
    k0;    ← K 轴分块大小
    n0;    ← N 轴分块大小
    pValue;    ← Ping-Pong 深度 (每次处理 pValue 行)
    ubMoveNum; ← UB 搬运数据量
}
```

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (`matmul_allto_all.cpp`, 第29-46行)

```
matmul_allto_all<TRANS_X2, HAS_BIAS, QUANT_BF16>(x1, x2, bias, ...)
│
├── REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData)
├── GET_TILING_DATA_WITH_STRUCT
├── KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
│
├── [QUANT_BF16]:
│   MatmulAlltoAll<DTYPE_X1, DTYPE_X2, bfloat16_t, ..., true, TRANS_X2>
│   └── op.Init(...) + op.Process()
│
└── [非量化]:
    MatmulAlltoAll<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, ..., false, TRANS_X2>
    └── op.Init(...) + op.Process()
```

### 4.2 Init 流程 (`matmul_allto_all.h`, 第104-123行)

```
Init(aGM, bGM, biasGM, pertokenScaleGM, scaleGM, cGM, workspaceGM, tilingGM):
│
├── REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData)
├── 保存 GM 地址指针
├── workspaceGM_ = GetUserWorkspace(workspaceGM)
│
└── CommBase::SetArgs(&rank, rank_size, tilingData):
    ├── block_id = GetBlockIdx()
    ├── core_num = GetBlockNum()
    ├── aiv_idx = GetSubBlockIdx()
    ├── core_idx = block_id / GetTaskRation()
    ├── rank_size = tiling.worldSize
    │
    ├── 从 HCCL Context 获取通信参数:
    │   ├── winContext_ = GetHcclContext<HCCL_GROUP_ID_0>()
    │   ├── rank = winContext_->rankId
    │   └── buff[i] = winContext_->windowsIn[i]  ← 各 Rank 的 Window 地址
    │
    ├── SetTiling(info): 提取 m0, k0, n0, p_value
    └── [AIV] TPipe pipe; pipe.InitBuffer(uBuf_, 160KB)
```

### 4.3 Process 流程 (`matmul_allto_all.h`, 第347-394行)

```
Process():
│
├── [x1 == INT8] → QuantCatlassMatmul()  ← 量化路径
│   └── 使用 MmadAtlasA2PreloadAsyncWithCallback + BlockEpilogue
│       INT8 × INT8 → INT32 → × scale → × pertokenScale → BF16 → Window
│
└── [x1 != INT8] → CatlassMatmul() + AIV 通信搬运  ← 非量化路径
    │
    ├── CatlassMatmul() [仅 AIC 执行]:
    │   ├── 定义 Catlass 类型: ArchTag=AtlasA2, LayoutA/B/C
    │   ├── 选择 DispatchPolicy:
    │   │   ├── [has_bias] MmadAtlasA2PingpongBias  ← 有偏置用 Pingpong
    │   │   └── [!has_bias] MmadAtlasA2Preload       ← 无偏置用 Preload
    │   ├── 选择 L1 TileShape:
    │   │   ├── [m0==128] GemmShape<128, 256, 256> + L0 <128, 256, 64>
    │   │   └── [m0==256] GemmShape<256, 128, 256> + L0 <256, 128, 64>
    │   ├── 构建 MatmulAlltoAllKernel
    │   └── matmul_op(params)  ← 执行 MM，结果写 gm_peer_mem
    │       注意: MM 结果直接写入 Window 共享内存 (buff[rank])
    │
    └── [AIV 通信搬运]:
        ├── commCount = DivCeil(m_loop, p_value)
        ├── ResetIpcFlags(2)  ← 清零 Flag
        ├── PipeBarrier<PIPE_ALL>()
        │
        └── for commIdx in [0, commCount):
            ├── 计算 token_total, token_per_rank
            ├── WaitEvent(flag_idx)  ← 等待 AIC 完成该块
            ├── SetAndWaitAivSync(flag_idx)  ← AIV 核间同步
            │
            ├── CrossRankSyncV1(FLAG_ZERO_IDX, commIdx+1)
            │   ← 通过 Window Flag 通知对端 Rank 数据已就绪
            │
            ├── SetAndWaitAivSync(flag_idx)
            │
            ├── [aiv_idx==0 && core_idx < rank_size]:
            │   ├── src = buff[rank] + pingpong_offset + rank_offset
            │   ├── dst = cGM + core_idx*rank_offset + commIdx偏移
            │   └── CopyGMToGM(src, dst, token_per_rank)
            │       ← 从本 Rank Window → 目标 Rank 输出 (跨 Rank)
            │
            ├── SetAndWaitAivSync(flag_idx)
            ├── CrossRankSyncV1(FLAG_ONE_IDX, commIdx+1) ← 通知完成
            ├── SetAndWaitAivSync(flag_idx)
            └── [commIdx < commCount-2] SetAicSync(flag_idx)
                ← 通知 AIC 可以开始下一块
```

### 4.4 量化路径: QuantCatlassMatmul (`matmul_allto_all.h`, 第213-345行)

```
QuantCatlassMatmul():
├── ElementA/B = INT8 (原始类型), ElementC = INT32 (累加)
├── DispatchPolicy = MmadAtlasA2PreloadAsyncWithCallback
│   (preloadStages=1, l1Stages=2, l0AStages=2, l0BStages=2, enableShuffleK=true)
├── Epilogue = BlockEpilogue
│   ├── TileRowBroadcastMul: × scale (反量化)
│   ├── TileRowBroadcastAdd: + bias
│   ├── TileBroadcastOneBlk: × pertoken_scale
│   └── TileOneBlkColumnBroadcastMul: 量化缩放
├── 选择 L1 TileShape:
│   ├── [m0==128] GemmShape<128, 256, 512> + L0 <128, 256, 128>
│   └── [m0==256] GemmShape<256, 128, 512> + L0 <256, 128, 128>
└── QuantMatmulAllToAllKernel(params)
    INT8×INT8 → INT32 → ×scale → ×pertokenScale → FP16/BF16 → Window
```

### 4.5 数据结构与内存布局

#### TilingData (`matmul_allto_all_tiling.h`)

```
MatmulAlltoAllTilingData (8 字节对齐):
┌──────────────────────────────────────────────────┐
│ Mc2InitTiling mc2InitTiling     ← HCCL 初始化参数  │
│ Mc2CcTiling mc2CcTiling         ← 通信计算参数     │
│ MatmulAlltoAllInfo              ← MM 维度: M, K, N │
│ │   uint32_t M;                                   │
│ │   uint32_t K;                                   │
│ │   uint32_t N;                                   │
│ │   uint32_t worldSize;                           │
│ CoCTiling                       ← 分块参数         │
│ │   int32_t m0;    ← M 分块 (128/256)             │
│ │   int32_t k0;    ← K 分块                       │
│ │   int32_t n0;    ← N 分块                       │
│ │   int32_t pValue; ← Ping-Pong 深度               │
│ │   int32_t ubMoveNum; ← UB 搬运量                 │
└──────────────────────────────────────────────────┘
```

#### Window 内存布局 (`CommBase`)

```
Window 共享内存 (每个 Rank 预注册):
┌─────────────────────────────────────────────────────────┐
│ buff[0]: Rank 0 的 Window 区域                          │
│ buff[1]: Rank 1 的 Window 区域                          │
│ ...                                                     │
│ buff[rankSize-1]: Rank N 的 Window 区域                 │
│                                                         │
│ 每个 Window 区域内部:                                     │
│ ┌──────────────────────────────────────────┐            │
│ │ [0..gm_a_pingpong_size × MAX_BLOCK_COUNT]│ ← MM 结果  │
│ │   (Ping-Pong 双缓冲, 最多 2 块)           │            │
│ │ ...                                     │            │
│ │ [FLAG_OFFSET..FLAG_OFFSET+1]             │ ← 同步标志 │
│ │   FLAG_ZERO_IDX: 数据就绪标志             │            │
│ │   FLAG_ONE_IDX: 搬运完成标志              │            │
│ └──────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘

FLAG_OFFSET = 180MB / 4 (以 int32 为单位)
```

#### UB 布局 (AIV 核, 160KB)

```
uBuf_ (160KB):
┌──────────────────────────────────────────┐
│ [0..UB_OFFSET)        copyTensor0        │ ← Ping 缓冲区
│ [UB_OFFSET..)         copyTensor1        │ ← Pong 缓冲区
│                                         │
│ UB_OFFSET = 97440 / sizeof(int16_t)     │
└──────────────────────────────────────────┘

CopyGMToGM 使用双缓冲:
  move_idx 偶数 → copyTensor0
  move_idx 奇数 → copyTensor1
```

### 4.6 多核并行策略

```
核类型: KERNEL_TYPE_MIX_AIC_1_2 (1 AIC + 2 AIV)

角色分工:
┌─────────┬──────────────────────────────────────┐
│  AIC    │ Catlass MM 计算                      │
│         │ 结果写入 gm_peer_mem (Window 区域)    │
├─────────┼──────────────────────────────────────┤
│  AIV-0  │ 核间同步 + 跨 Rank 数据搬运           │
│         │ CopyGMToGM: Window → 输出 GM          │
├─────────┼──────────────────────────────────────┤
│  AIV-1  │ 核间同步 (等待)                       │
│         │ WaitEvent / CheckBuffFlag             │
└─────────┴──────────────────────────────────────┘

同步流程:
  AIC → [完成一块] → SetAicSync(flag) → AIV WaitEvent(flag)
  AIV-0 → SetBuffFlag(本端就绪) → 对端 CheckBuffFlag → 对端读取数据
  对端 → SetBuffFlag(读取完成) → 本端 CheckBuffFlag → 继续
```

### 4.7 通信模式: Window IPC 点对点

**核心通信流程** (AIV 侧):

```
CrossRankSyncV1(flag_idx, flag_data):
├── [core_idx==0 && aiv_idx==0]:
│   SetBuffFlag(buff[rank] + FLAG_OFFSET + flag_idx, flag_data)
│   ← 向本 Rank Window 写入 Flag
└── [core_idx < rank_size && aiv_idx==0]:
    CheckBuffFlag(buff[core_idx] + FLAG_OFFSET + flag_idx, flag_data)
    ← 轮询对端 Rank 的 Flag 直到匹配

SetBuffFlag(buff, flag):
├── ubTensor(0) = flag
└── CopyUbufToGmAlignB16(buff, ubTensor)

CheckBuffFlag(buff, flag):
├── while(true):
│   ├── CopyGmToUbufAlignB16(ubTensor, buff)
│   └── if (ubTensor(0) == flag) break
└── 轮询等待

CopyGMToGM(gm_src, gm_dst, copy_size):
├── Ping-Pong 双缓冲搬运
├── for move_idx:
│   ├── [偶数] copyTensor0 / [奇数] copyTensor1
│   ├── CopyGmToUbuf (GM → UB)
│   └── CopyUbufToGm (UB → GM)
└── max_ub_ping_pong_size = ubMoveNum / MAX_BLOCK_COUNT / n0 × n0
```

---

## 5. Host-Kernel 接口

### 5.1 TilingData 传递

```
Host → Kernel:
  MatmulAlltoAllTilingData (通过 tilingGM 传递)
  ├── mc2InitTiling     ← HCCL 初始化
  ├── mc2CcTiling       ← 通信计算参数
  ├── matmulAlltoAllInfo ← M, K, N, worldSize
  └── cocTiling         ← m0, k0, n0, pValue, ubMoveNum

HCCL Context:
  GetHcclContext<0> → HcclA2CombineOpParam*
  ├── rankId, rankDim
  ├── windowsIn[0..7]  ← 各 Rank Window 地址
  └── windowsOut[0..7]
```

### 5.2 Workspace 布局

```
Workspace:
┌───────────────────────────────────────────────────┐
│ HCCL 工作空间 (公式化 Tiling)                       │
├───────────────────────────────────────────────────┤
│ UserWorkspace:                                     │
│ ├── [量化路径] 量化 MM 中间结果缓冲区               │
│ └── [量化路径] Dequant 工作空间                     │
└───────────────────────────────────────────────────┘
```

---

## 6. 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ Host Tiling 阶段                                                 │
│                                                                  │
│ x1[M,K], x2[K,N] → Check2DMatrixMulShapes()                    │
│    → TileCommAndCompute() → 计算 m0, k0, n0, pValue             │
│    → AllToAll 公式化 Tiling → HCCL 参数                          │
│    → GetWorkspaceSize() → WS 总量                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Kernel 执行阶段                                                  │
│                                                                  │
│ matmul_allto_all<TRANS_X2, HAS_BIAS, QUANT_BF16>               │
│ ├── MatmulAlltoAll::Init()                                       │
│ │   ├── 保存 GM 地址                                              │
│ │   └── CommBase::SetArgs() → 获取 rank, Window 地址             │
│ │                                                                │
│ └── MatmulAlltoAll::Process()                                    │
│     │                                                            │
│     ├── [非量化] CatlassMatmul() [AIC]:                          │
│     │   ├── 构建 Catlass 类型: AtlasA2, RowMajor/NZ              │
│     │   ├── 选择 L1 分块: 128×256×256 或 256×128×256             │
│     │   ├── 选择 DispatchPolicy: Preload / PingpongBias          │
│     │   └── MatmulAlltoAllKernel(params):                        │
│     │       MM(x1, x2) → 结果写入 gm_peer_mem (=buff[rank])     │
│     │                                                            │
│     └── [非量化] AIV 通信搬运:                                    │
│         for commIdx in [0, commCount):                           │
│         ├── WaitEvent → 等待 AIC 完成该块                        │
│         ├── CrossRankSyncV1(FLAG_ZERO) → 通知对端就绪            │
│         ├── CopyGMToGM(buff[rank] → cGM) → 搬运到目标 Rank       │
│         ├── CrossRankSyncV1(FLAG_ONE) → 通知搬运完成              │
│         └── [非最后] SetAicSync → 通知 AIC 继续下一块             │
│                                                                  │
│     ── 或 ──                                                     │
│                                                                  │
│     ├── [量化] QuantCatlassMatmul() [AIC+AIV]:                   │
│     │   ├── MmadAtlasA2PreloadAsyncWithCallback                   │
│     │   ├── BlockEpilogue: INT32 → ×scale → ×pertoken → BF16    │
│     │   └── QuantMatmulAllToAllKernel(params)                    │
│     │       (通信逻辑内嵌在 Catlass Kernel 中)                    │
│     └── SyncAll<false>()                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
          y [M × rankSize, N / rankSize]
```

---

## 7. 性能相关设计

### 7.1 Catlass 模板库

使用 `3rd/template_linear_algebra` 中的 Catlass 库，实现高效分块 MM:
- **BlockMmad**: L1 级分块 MM，自动管理 L0/L1 缓冲区
- **BlockSwizzle**: 3D Block 调度 (GemmIdentityBlockSwizzle<3,0>)
- **Epilogue**: 可选的 Bias/Dequant 后处理

### 7.2 Ping-Pong 双缓冲

MM 结果和 UB 搬运都使用双缓冲:
- `MAX_BLOCK_COUNT = 2`: Window 区域分两块，AIC 写一块时 AIV 搬另一块
- UB `copyTensor0/1`: CopyGMToGM 双缓冲搬运

### 7.3 L1 分块自适应

根据 M 维度选择不同的 L1 分块策略:
- `m0 == 128`: `GemmShape<128, 256, 256>` — 窄 M 场景
- `m0 == 256`: `GemmShape<256, 128, 256>` — 宽 M 场景

### 7.4 量化 MM 流水线

量化路径使用 `MmadAtlasA2PreloadAsyncWithCallback`:
- `enableShuffleK = true`: 启用 K 维度 Shuffle 优化
- `preloadStages = 1, l1Stages = 2`: L1 双缓冲预取
- Epilogue 集成反量化: INT32 → scale × pertoken → FP16/BF16

---

## 8. 与 matmul_all_reduce 的核心差异

| 特性 | matmul_all_reduce | matmul_allto_all |
|------|-------------------|------------------|
| 通信原语 | AllReduce | AllToAll |
| 通信库 | HCCL AICPU | Window IPC + Catlass |
| 输出 Shape | [M, N] (不变) | [M×R, N/R] (展开) |
| 核心 Tiling 结构 | RCSTiling + TCubeTiling | CoCTiling |
| MM 引擎 | MatMulV3 / QBMV3 / WQBMV2 | Catlass 模板库 |
| 同步方式 | hccl_.Commit/Wait | Flag 轮询 + CrossCoreSetFlag |
| 平台 | 910B + 310P + 950 | 910B + 910_93 + 950 |
| 多核类型 | MIX_AIC_1_2 / MIX_AIC_1_0 | MIX_AIC_1_2 |

---

## 9. 关键约束与注意事项

1. **x1 不支持转置**: `transpose_x1` 强制为 false (infershape 第88-90行)
2. **rankSize 限制**: 仅支持 2/4/8/16 (infershape 第51行 `SUPPORT_RANK_NUM`)
3. **量化必须有 Bias**: `QUANT_BF16=1` 时 `HAS_BIAS` 必须为 1 (TilingKey 限制)
4. **Window 大小**: FLAG_OFFSET 在 180MB 处，MM 结果区域需 ≤ 180MB
5. **CopyGMToGM 性能**: AIV 通过 UB 中转搬运，UB 分块大小受 `ubMoveNum` 限制
6. **输出类型必须指定**: `y_dtype` 不能为 DT_UNDEFINED (infershape 第182-183行)
7. **合法量化模式组合**: 仅支持 (0,0) 无量化, (3,2) PerToken+PerChannel, (6,6) MX

---

## 10. 函数调用关系

```
matmul_allto_all<TRANS_X2, HAS_BIAS, QUANT_BF16>() (Kernel 主入口)
├── REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData)
├── KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
└── [QUANT_BF16?]:
    ├── MatmulAlltoAll<INT8, INT8, bf16, ...>::Init()
    │   └── CommBase::SetArgs() → 获取 rank, Window 地址
    └── MatmulAlltoAll::Process()
        └── QuantCatlassMatmul()
            ├── [AIC] 选择 L1/L0 分块
            ├── 构建 QuantMatmulAllToAllKernel
            └── quant_matmul_op(params)

    ── 或 ──

    ├── MatmulAlltoAll<FP16, FP16, ..., false>::Init()
    └── MatmulAlltoAll::Process()
        ├── [AIC] CatlassMatmul()
        │   ├── 选择 DispatchPolicy (Preload / PingpongBias)
        │   ├── 选择 L1 分块 (128×256×256 / 256×128×256)
        │   └── MatmulAlltoAllKernel(params)
        │       └── MM → 写入 Window 共享内存
        │
        └── [AIV] 通信搬运循环:
            for commIdx in [0, commCount):
            ├── WaitEvent(flag_idx)
            ├── SetAndWaitAivSync()
            ├── CrossRankSyncV1(FLAG_ZERO_IDX)
            ├── CopyGMToGM(Window → 输出GM)
            │   └── Ping-Pong 双缓冲 UB 搬运
            ├── CrossRankSyncV1(FLAG_ONE_IDX)
            └── SetAicSync() (非最后块)
```

---

## 11. 依赖关系

```
matmul_allto_all.cpp
├── arch32/matmul_allto_all_tiling_key.h  ← 模板参数宏
├── arch32/matmul_allto_all_tiling.h      ← Tiling 数据结构
└── arch32/matmul_allto_all.h             ← 主类
    ├── matmul_allto_all_util.h            ← CommBase 通信工具
    │   └── matmul_allto_all_tiling.h
    ├── moe_distribute_base.h              ← HCCL 参数结构
    ├── 3rd/template_linear_algebra/       ← Catlass 模板库
    │   ├── catlass.hpp
    │   ├── arch/arch.hpp
    │   ├── layout/layout.hpp
    │   ├── gemm/block/block_mmad.hpp
    │   ├── gemm/block/block_swizzle.hpp
    │   ├── gemm/dispatch_policy.hpp
    │   ├── gemm/kernel/ (MatmulAlltoAllKernel, QuantMatmulAllToAllKernel)
    │   └── epilogue/ (TileCopy, TileBroadcastMul, ...)
    ├── block_epilogue_dequant.hpp          ← 量化反量化 Epilogue
    ├── tile_broadcast_add.hpp              ← 广播加法
    ├── matmul.hpp                          ← 非量化 MM 辅助
    └── quant_matmul.hpp                    ← 量化 MM 辅助
```

---

*分析完成时间: 2026-04-17*
