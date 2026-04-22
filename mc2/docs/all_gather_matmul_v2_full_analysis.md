# all_gather_matmul_v2 算子全面分析报告

## 1. 算子概述

`all_gather_matmul_v2` 是 `all_gather_matmul` 的 V2 版本，核心差异在于采用 **AIV-mode (Window IPC) 通信**替代 V1 的 HCCL API 通信，实现 AllGather 与 MatMul 的深度融合。利用 Catlass 模板线性代数库进行 AIC 侧矩阵乘法，AIV 侧通过 Window 共享内存直接将本卡 A 矩阵数据写入其他卡的 Window，实现零拷贝 AllGather。

### 1.1 模板参数

```cpp
// all_gather_matmul_v2_tiling_key.h (第23-28行)
ASCENDC_TPL_ARGS_DECL(
    Mc2AllGatherMatmulV2,
    ASCENDC_TPL_BOOL_DECL(TPL_ISBIAS, 0, 1),        // 是否有 bias
    ASCENDC_TPL_BOOL_DECL(TPL_IS_TRANSPOSE_A, 0, 1), // A 矩阵是否转置
    ASCENDC_TPL_BOOL_DECL(TPL_IS_TRANSPOSE_B, 0, 1), // B 矩阵是否转置
);
// 有效组合: TPL_ISBIAS=0, TPL_IS_TRANSPOSE_A=0, TPL_IS_TRANSPOSE_B ∈ {0,1}
// 即: 仅支持无 bias、A 不转置，B 可选转置
```

### 1.2 核心类模板

```cpp
// all_gather_matmul_aiv_mode.h (第95-96行)
template <typename X1Type, typename X2Type, typename BiasType,
          typename x2ScaleType, typename YType,
          bool weightNZ, bool TA, bool TB>
class AllGatherMatmulAIVMode;
```

| 参数 | 含义 |
|------|------|
| `X1Type` | A 矩阵数据类型 (BF16/FP16/INT8/INT4) |
| `X2Type` | B 矩阵数据类型 |
| `BiasType` | Bias 数据类型 |
| `x2ScaleType` | 权重量化 scale 类型 (float/int64) |
| `YType` | 输出数据类型 (BF16/FP16) |
| `weightNZ` | 权重是否为 NZ 格式 (FRACTAL_NZ) |
| `TA` | A 矩阵是否转置 (当前固定 false) |
| `TB` | B 矩阵是否转置 |

### 1.3 量化标志 (编译期)

```cpp
// all_gather_matmul_aiv_mode.h (第98-99行)
constexpr static bool quantFlag =
    (std::is_same<X1Type, int8_t>::value && std::is_same<X2Type, int8_t>::value) ||
    (std::is_same<X1Type, AscendC::int4b_t>::value && std::is_same<X2Type, AscendC::int4b_t>::value);
```

---

## 2. 目录结构与文件职责

```
all_gather_matmul_v2/
├── op_kernel/
│   ├── all_gather_matmul_v2.cpp                    (48行)  AIV-mode 内核入口
│   ├── all_gather_matmul_v2_apt.cpp                (107行) arch35 APT 内核入口 (FP8量化)
│   ├── all_gather_matmul_v2_tiling_key.h           (42行)  AIV-mode 模板参数声明
│   ├── all_gather_matmul_v2_apt_tiling_key.h       (85行)  APT 模板参数声明
│   ├── common.h                                    (99行)  公共定义 (MC2GmAddrs, Mc2CoreType, Mc2SyncAll)
│   ├── all_gather_matmul_aiv_mode_tiling.h         (87行)  Tiling 数据结构定义
│   ├── all_gather_matmul_aiv_mode.h                (894行) AIV-mode 主类 (核心实现)
│   ├── all_gather_matmul_aiv_mode_util.h           (220行) 工具函数 (对齐/同步/拷贝)
│   ├── all_gather_matmul_aiv_mode_padding.h        (200行) AIV 侧矩阵 Padding
│   ├── all_gather_matmul_aiv_mode_dequant.h        (151行) AIV 侧反量化 Runner
│   ├── all_gather_matmul_aiv_mode_block_epilogue_dequant.h (464行) UB 反量化 Epilogue
│   ├── gather_matmul.hpp                           (467行) Catlass AllGatherMatmulV2 内核
│   ├── block_mmad_preload_fixpipe.h                (401行) 量化 Fixpipe MatMul
│   ├── copy_gm_to_l1.h                            (52行)  Scale 数据 L1 拷贝特化
│   ├── copy_l0c_to_gm.h                           (62行)  Fixpipe 输出拷贝特化
│   ├── gather_moe_distribute_base.h                (321行) HCCL 通信基础设施/结构体
│   └── arch35/                                     arch35 (910_93) 专用实现
│       ├── all_gather_matmul_tiling_arch35.h       (58行)  arch35 Tiling 数据结构
│       ├── all_gather_matmul_base.h                (207行) arch35 基类
│       ├── all_gather_matmul_fp16_bf16.h           (274行) arch35 FP16/BF16 实现
│       ├── all_gather_quant_bmm.h                  (377行) arch35 FP8 量化实现
│       └── all_gather_quant_bmm_perblock.h         (175行) arch35 PerBlock 量化实现
├── op_host/
│   ├── all_gather_matmul_v2_def.cpp                (256行) 算子定义
│   ├── all_gather_matmul_v2_infershape.cpp         (62行)  Shape 推导
│   └── op_tiling/                                  Tiling 策略
│       ├── arch32/                                 910B Tiling
│       │   ├── all_gather_matmul_aiv_mode_tiling.cpp (618行) AIV-mode Tiling
│       │   ├── all_gather_matmul_tiling_v2.cpp      (64行)  Tiling 路由
│       │   ├── all_gather_matmul_v2_tiling.cpp      (280行) 编码表
│       │   └── tiling_func.h                        (567行) 条件映射表
│       └── arch35/                                 910_93 Tiling
│           ├── all_gather_fit_balance_tiling.cpp    (88行)  Balance Tiling
│           └── all_gather_quant_bmm_tiling.cpp      (859行) 量化 Tiling
├── op_api/                                         ACLNN C-API
├── op_graph/                                       图模式
└── tests/                                          测试
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_def)

**文件**: `all_gather_matmul_v2_def.cpp` (第18-255行)

**输入张量**:

| 序号 | 名称 | 类型 | 格式 | 说明 |
|------|------|------|------|------|
| 0 | x1 | FP16/BF16/FP8_E4M3/FP8_E5M2/HIFLOAT8/INT8/INT4 | ND | A 矩阵输入 |
| 1 | x2 | FP16/BF16/FP8_E4M3/FP8_E5M2/HIFLOAT8/INT8/INT4 | ND/NZ | B 矩阵权重 |
| 2 | bias | FP16/BF16/FLOAT | ND | 偏置 (可选) |
| 3 | x1_scale | FLOAT/FP8_E8M0 | ND | A 矩阵量化 scale (可选) |
| 4 | x2_scale | FLOAT/INT64/FP8_E8M0 | ND | B 矩阵量化 scale (可选) |
| 5 | quant_scale | FLOAT | ND | 输出量化 scale (可选) |

**输出张量**:

| 序号 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 0 | y | FP16/BF16/FLOAT | MatMul 结果 (含所有 rank 的 AllGather 结果) |
| 1 | gather_out | 与 x1 同类型 | AllGather 输出 (M×worldSize×K) |
| 2 | amax_out | FLOAT | 最大绝对值输出 (量化场景) |

**关键属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| group | String | 必填 | HCCL 通信域标识 |
| is_trans_a | Bool | false | A 矩阵是否转置 |
| is_trans_b | Bool | false | B 矩阵是否转置 |
| gather_index | Int | 0 | AllGather 索引 |
| comm_turn | Int | 0 | 通信轮次 |
| rank_size | Int | 0 | 通信域大小 |
| block_size | Int | 0 | 量化块大小 |
| group_size | Int | 0 | 量化分组大小 |
| is_gather_out | Bool | true | 是否输出 gather_out |
| is_amax_out | Bool | false | 是否输出 amax_out |
| y_dtype | Int | DT_UNDEFINED | 输出数据类型覆盖 |
| comm_mode | String | "aicpu" | 通信模式 ("aicpu"/"aiv") |

**平台配置**:
- `ascend950`: 使用 `all_gather_matmul_v2_apt` 内核文件，支持 FP8 量化
- `ascend910b`: 支持 BF16/FP16/INT8/INT4，x2 可为 ND 或 FRACTAL_NZ 格式
- `ascend910_93`: 与 910b 共享配置

### 3.2 Shape 推导

**文件**: `all_gather_matmul_v2_infershape.cpp` (第25-62行)

- `y` 输出 shape: 通过 `AllGatherMatmulCommonInferShape` 公共函数计算，模式为 `GATHER_OUT_V2`
- `gather_out` 输出 shape: 同上
- `amax_out` 输出 shape: 若 `is_amax_out=true` 则为 `[1]`，否则为 `[0]`
- 数据类型推导: BF16/FP16 输入保持原类型；FP8 输入使用 `y_dtype` 属性指定输出类型

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `all_gather_matmul_v2.cpp` (第29-48行)

```cpp
template<bool TPL_ISBIAS, bool TPL_IS_TRANSPOSE_A, bool TPL_IS_TRANSPOSE_B>
__global__ __aicore__ void all_gather_matmul_v2(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scaleInv1,
    GM_ADDR scaleInv2, GM_ADDR scale, GM_ADDR cGM,
    GM_ADDR gatherOut, GM_ADDR amax, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 混合 AIC+AIV 模式
    REGISTER_TILING_DEFAULT(AllGatherMatmulAIVModeTilingData);
    // 实例化: FORMAT_X2==FRACTAL_NZ, TA=false, TB=TPL_IS_TRANSPOSE_B
    INVOKE_ALLGATHERMATMUL_AIV_MODE_OP_IMPL(
        AllGatherMatmulAIVMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false, TPL_IS_TRANSPOSE_B);
}
```

关键点:
- `KERNEL_TYPE_MIX_AIC_1_2`: 混合内核，每个物理核包含 1 个 AIC + 2 个 AIV 子核
- Tiling 数据结构: `AllGatherMatmulAIVModeTilingData`
- `weightNZ` 参数由编译期宏 `FORMAT_X2 == FORMAT_FRACTAL_NZ` 决定

### 4.2 常量定义

```cpp
// all_gather_matmul_aiv_mode.h (第44-62行)
BUFFER_NUM = 2;                      // Ping-Pong 缓冲区数量
CUBE_MATRIX_SIZE_B16 = 256;          // Cube 矩阵单元大小 (16×16)
L0AB_PINGPONG_BUFFER_SIZE = 32768;   // L0 A/B 缓冲区 32KB
MAX_BLOCK_COUNT = 2;                 // 最大并发块数
USED_UB_SIZE = 160 * 1024;           // UB 使用量 160KB
FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t);  // Flag 区域偏移 180MB
TILE_SHAPE_128/256/512/64;           // 矩阵分块形状
```

### 4.3 Init() 方法详解

**位置**: `all_gather_matmul_aiv_mode.h` 第213-303行

```
Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, allgatherGM, workspaceGM, tilingGM)
├── 第217-218行: 从 tilingGM 读取 Tiling 数据
├── 第220-226行: 获取 HCCL Context (contextGM0)，保存各 GM 地址
├── 第228-252行: 从 Tiling 读取维度参数
│   ├── m, k, n (矩阵维度)
│   ├── m0, k0, n0 (分块大小，由 Tiling 寻优确定)
│   ├── m_loop, n_loop, k_loop (分块循环数)
│   ├── pValue (通信流水线粒度)
│   ├── comm_npu_split, comm_data_split, comm_direct (通信切分参数)
│   └── swizzlCount, swizzlDirect (核间负载均衡 swizzle 参数)
├── 第254-268行: 对齐处理
│   ├── m_align = Block512B<X1Type>::AlignUp(m)  (512B 对齐)
│   ├── k_align, n_align 同理
│   └── 分配 workspace 用于 Padding 缓冲区
├── 第270-275行: 量化标志设置
│   ├── needPerChannel: INT4 或 (A8W8 且非 int64+fp16 场景)
│   └── needPerToken: quantFlag && dequantType == PER_TOKEN
├── 第277-295行: HCCL Context 解析 (910C vs A2)
│   ├── [910C]: HcclOpResParam → localUsrRankId, rankSize
│   │   stateAddrPerRank[i] = remoteRes[i].windowsIn
│   └── [A2]:  HcclA2CombineOpParam → rankId, rankNum
│       stateAddrPerRank[i] = windowsIn[i]
└── 第300-302行: 调用 AICInit() 和 AIVInit()
```

**910C vs A2 上下文结构对比**:

```
910C (HcclOpResParam):
  localUsrRankId / rankSize
  localWindowsIn / localWindowsOut
  remoteRes[i].nextDevicePtr → HcclRankRelationResV2.windowsIn
  (需要二级指针解引用)

A2 (HcclA2CombineOpParam):
  rankId / rankNum
  windowsIn[rankId] = 本卡 Window 地址
  windowsIn[other]  = 其他卡 Window 映射地址
  (直接数组访问)
```

### 4.4 AICInit() 方法

**位置**: 第306-314行

```
AICInit()  [仅 AIC 核执行]
├── SetLoadDataPaddingValue(0)    ← 加载数据时补 0
├── SetAtomicNone()               ← 关闭原子操作
├── SetFixpipeNz2ndFlag(1,0,0)    ← 设置 Fixpipe NZ→ND 转换标志
└── gm_peer_mem = stateAddrPerRank[rankId]  ← 本卡 Window 数据区地址
```

### 4.5 AIVInit() 方法

**位置**: 第341-360行

```
AIVInit()  [仅 AIV 核执行]
├── SetAtomicNone() / SetMaskNormImpl() / SetVectorMask(全1)
├── 计算 cal_count = DivCeil(m_loop, pValue)  ← 通信流水线阶段数
├── gm_a_pingpong_size = m0 * k_align * pValue * worldSize  ← Ping-Pong 缓冲大小
├── data_len = m * k_align     ← 每卡 A 矩阵数据量
├── num_per_rank_move = m0 * k_align * pValue  ← 每轮每卡搬运量
├── src_offset = 0             ← 数据搬运偏移
└── rank_offset = rankId * num_per_rank_move  ← 本卡在 Window 中的偏移
```

### 4.6 Process() 方法 — 核心流水线

**位置**: 第770-891行

这是算子的核心执行流程，采用 AIV + AIC 混合流水线模式：

```
Process():
├── [AIV] TPipe pipe; pipe.InitBuffer(uBuf_, 160KB)
├── [AIV] Padding()                    ← A 矩阵/B 矩阵对齐到 512B
├── [AIV] 清零 4 个 flag
├── [AIV] PipeBarrier<PIPE_ALL>()
│
├── for cal_idx = 0..cal_count+1:      ← cal_count+2 次迭代
│   ├── flag_idx = cal_idx % 2         ← Ping-Pong flag 索引
│   │
│   ├── [cal_idx == 1] AllGatherPerTokenScale()  ← 聚合 PerToken Scale
│   │
│   ├── [cal_idx >= 2] WaitEvent(flag_idx)       ← 等待 AIC 上轮完成
│   │
│   ├── SetAndWaitAivSync(flag_idx)              ← AIV 核间同步
│   │
│   ├── [cal_idx < cal_count]
│   │   ├── CrossRankSyncV1(ZERO, cal_idx+1)    ← 跨 rank 同步 (数据就绪)
│   │   ├── SetAndWaitAivSync(flag_idx)
│   │   └── [aivIdx==0, blockIdx<core_count]
│   │       └── MoveWithSplit(A+src_offset, rank_offset, num_per_rank_move)
│   │           ← AIV 核将 A 矩阵分块写入其他卡 Window
│   │
│   ├── [cal_idx > 0, aivIdx==1, blockIdx>=core_count]
│   │   └── MoveResultFromPeerMemToOut()
│   │       ← AIV 核从本卡 Window 读取其他卡数据到 gatherOut
│   │
│   ├── [cal_idx < cal_count]
│   │   ├── SetAndWaitAivSync(flag_idx)
│   │   ├── CrossRankSyncV2(ONE, cal_idx+1)      ← 跨 rank 同步 (搬运完成)
│   │   ├── SetAndWaitAivSync(flag_idx)
│   │   └── SetAicSync(flag_idx)                  ← 通知 AIC 可以开始计算
│   │
│   └── [cal_idx >= 2] Dequant(cal_idx-2)         ← AIV 反量化
│
├── [AIV] 清零 4 个 flag
├── [AIV] CheckBuffFlag 确认所有 rank 的 flag 归零
├── [AIV] PipeBarrier<PIPE_ALL>()
│
├── CatlassMatmul()                    ← AIC 侧矩阵乘法 (所有 cal_count 轮)
└── SyncAll<false>()                   ← 全局同步
```

**流水线时序图**:

```
时间 →
AIV核0:  [Pad] [Flag0] [Move 0→ranks] [Sync1] [Move 1→ranks] [Sync2] ... [Dequant0] [Dequant1]
AIV核1:  [Pad] [Flag0] [Read 0→out]           [Read 1→out]           ... [Dequant0] [Dequant1]
AIC核:   [Wait Pad] ............[MM0+Gather0]...............[MM1+Gather1]............
                 ↕ SetAicSync(flag0)              ↕ SetAicSync(flag1)
```

### 4.7 CatlassMatmul() 方法 — AIC 侧矩阵乘法

**位置**: 第363-500行

这是 AIC 专用的方法，使用 Catlass 模板库实现 AllGather+MatMul 融合：

```
CatlassMatmul() [仅 AIC 核执行]
├── 计算布局参数
│   ├── layoutA: RowMajor {m, k_align}
│   ├── layoutPeerMem: RowMajor {peer_mem_m * 2, k_align}
│   ├── layoutC: RowMajor {m * worldSize, n}
│   └── processSize: GemmCoord{m, n, k}
├── 确定量化输出模式 (need_fixpipe)
│   └── quantFlag && YType==half && x2ScaleType==int64 && !int4
├── 选择分块大小 (m0)
│   ├── m0==128: L1TileShape<128, 256, K>, L0TileShape<128, 256, K>
│   └── m0==256: L1TileShape<256, 128, K>, L0TileShape<256, 128, K>
├── 创建 AllGatherMatmulV2 kernel
│   └── Params: 包含 A/B/C 地址、PeerMem 地址、Scale 地址
│              pValue, swizzlCount, swizzlDirect, rankId, worldSize
└── 执行 matmul_op(params)
```

**AllGatherMatmulV2 内核** (gather_matmul.hpp 第425-440行):

```
operator()<AIC>(params):
├── CrossCoreWaitFlag(flagAivFinishPadding)  ← 等待 AIV 完成 Padding
├── InitArgs(params)                          ← 初始化 GM 缓冲、计算循环参数
├── DoLocalMatmul(params)                     ← 本地 MatMul (rankId 对应的数据块)
│   └── 对每个 M-N 分块:
│       blockMmad(gmA[offset], gmB[offset], gmDst[offset], ...)
│       输出到 gmWorkSpace (量化时) 或 gmC (非量化时)
└── Matmul(params) 或 FixpipeMatmul(params)   ← AllGather 后的远程数据 MatMul
    └── 对每个 cal_idx:
        ├── WaitEvent(flagIdx)                ← 等待 AIV 完成该轮数据搬运
        ├── 对每个 remote rank:
        │   ├── 从 gmPeerMem 或 gmA 读取 A 矩阵数据
        │   ├── blockMmad(peerA, gmB, gmC[dst_rank*mn + ...])
        │   └── 输出按 rank 交错排列: C[rank0行0, rank1行0, ..., rank0行1, ...]
        └── FFTSCrossCoreSync(flagIdx)        ← 通知 AIV 该轮计算完成
```

### 4.8 CrossRankSyncV1/V2 — 跨 Rank 同步

**位置**: 第317-338行

```
CrossRankSyncV1 (数据写入就绪通知):
  本卡 (blockIdx==rankId): SetBuffFlagByAdd → 通知其他 rank 数据已写入本卡 Window
  其他卡 (blockIdx<worldSize): CheckBuffFlag → 等待对应 rank 的数据写入

CrossRankSyncV2 (数据消费完成通知):
  所有卡: SetBuffFlagByAdd → 通知本卡已消费完 Window 数据
  本卡 (blockIdx==rankId): CheckBuffFlag(worldSize) → 等待所有 rank 消费完毕
```

**Flag 地址计算**:
```
flag_addr = stateAddrPerRank[rank] + FLAG_OFFSET + flag_idx
FLAG_OFFSET = 180MB / sizeof(int32_t)
```

### 4.9 MoveWithSplit() — A 矩阵分块写入其他 Rank

**位置**: 第720-767行

```
MoveWithSplit(gm_src, rank_offset, len):
├── data_split = DivCeil(len, len_per_loop)
├── group_num = DivCeil(worldSize, comm_npu_split)
├── for data_block_idx = 0..data_split:
│   ├── data_src = 按 blockIdx 计算本核负责的数据偏移
│   ├── data_len = 按切分计算本核负责的数据量
│   ├── if comm_direct:
│   │   └── MoveToOtherRankWithSkip()  ← NPU 方向优先: 一份数据发所有目标卡
│   └── else:
│       └── 按 rank 分组 MoveResultToDst()  ← 数据方向优先: 所有数据发目标卡0, 再卡1...
```

### 4.10 MoveToOtherRankWithSkip() — 跨 Rank 数据搬移

**位置**: 第598-639行

```
MoveToOtherRankWithSkip(gm_src, rank_offset, len, rank_st, skip_num, group_num, rank_scope):
├── Ping-Pong UB 缓冲 (copyTensor0, copyTensor1, 各 ~97KB)
├── ping_pong_move_count = DivCeil(len, max_ub_ping_pong_size)
├── for move_idx = 0..ping_pong_move_count:
│   ├── WaitFlag → CopyGmToUbufAlignB16 (GM→UB) → SetFlag
│   ├── dst_rank = rank_st % rank_scope
│   ├── for cycle_idx = 0..group_num:
│   │   ├── if dst_rank != rankId && dst_rank < worldSize:
│   │   │   └── CopyUbufToGmAlignB16(UB → stateAddrPerRank[dst_rank] + rank_offset)
│   │   └── dst_rank = (dst_rank + skip_num) % rank_scope  ← 跳跃访问
│   └── SetFlag
└── FreeTensor
```

### 4.11 MoveResultFromPeerMemToOut() — Window 数据读取到 gatherOut

**位置**: 第642-690行

```
MoveResultFromPeerMemToOut(gm_src, gm_dst, actual_m):
├── max_move_m = max_ub_ping_pong_size / max_move_k
├── Ping-Pong 搬运:
│   ├── 外层: 按 M 维切分 (max_move_m)
│   └── 内层: 按 K 维切分 (max_move_k = 20480)
│       ├── GM→UB: 读 k_align 列 (包含 padding)
│       └── UB→GM: 写 k 列 (去除 padding 的 stride 拷贝)
└── 结果: 去除对齐 padding 的紧凑数据写入 gatherOut
```

关键: 这里实现了 **对齐数据的去 padding**：从 Window 中读取 k_align 列，但只写出前 k 列到 gatherOut，通过 stride 参数跳过尾部 padding。

### 4.12 Dequant() — 反量化

**位置**: 第575-595行

由 `DequantRunner` (all_gather_matmul_aiv_mode_dequant.h) 执行，支持三种模式:

| 模式 | 条件 | 操作 |
|------|------|------|
| perChannel + perToken | needPerChannel && needPerToken | C_int32 × perChannelScale × perTokenScale → YType |
| perChannel only | needPerChannel && !needPerToken | C_int32 × perChannelScale → YType |
| perToken only | !needPerChannel && needPerToken | YType × perTokenScale → YType |

### 4.13 AllGatherPerTokenScale() — 跨 Rank Scale 聚合

**位置**: 第543-572行

```
AllGatherPerTokenScale(buff_st):
├── [rankId==blockIdx] 将本卡 x1Scale 拷贝到本卡 Window 对应位置
├── CrossRankSyncV1  ← 等待所有 rank 写入
├── [blockIdx<worldSize] 从各 rank Window 读取 scale 到 gm_scale_workspace
└── CrossRankSyncV2  ← 等待所有 rank 读取完毕
```

### 4.14 Padding() — 矩阵对齐 Padding

**位置**: 第503-529行

```
Padding():
├── if !aligned_a && !aligned_b: 设置 flag 直接返回 (无需 padding)
├── 根据 transA/transB 组合选择 Layout:
│   ├── !transA && !transB: LayoutA=RowMajor, LayoutB=RowMajor
│   ├── !transA && transB:  LayoutA=RowMajor, LayoutB=ColumnMajor
│   ├── transA && !transB:  LayoutA=ColumnMajor, LayoutB=RowMajor
│   └── transA && transB:   LayoutA=ColumnMajor, LayoutB=ColumnMajor
├── TemplatePadder(params):
│   ├── [alignedA] PaddingMatrix: gmA → gmAAlign (补 0 到 512B 对齐)
│   ├── [alignedB] PaddingMatrix: gmB → gmBAlign (补 0 到 512B 对齐)
│   └── CrossCoreSetFlag 通知 AIC padding 完成
└── INT4 类型特殊处理: 维度减半 (2 个 int4 打包在 1 个 int8 中)
```

---

## 5. 数据结构

### 5.1 AllGatherMatmulAIVModeTilingData

**文件**: `all_gather_matmul_aiv_mode_tiling.h` (第80-85行)

```
AllGatherMatmulAIVModeTilingData:
├── mc2InitTiling: Mc2InitTiling         ← HCCL 初始化参数
├── mc2CcTiling: Mc2CcTiling             ← 通用通信计算参数
├── allGatherMatmulInfo:                 ← 算子维度与对齐参数
│   ├── M, K, N (uint32)                ← 矩阵维度
│   ├── worldSize, rankId (uint32)      ← 通信域参数
│   ├── aivNum, totalUbSize (uint32)    ← 硬件资源参数
│   ├── aAlignSize, bAlignSize (uint64) ← Padding 区大小
│   ├── isTransposeX1, isTransposeX2    ← 转置标志
│   ├── hasAAlign, hasBAlign            ← 是否需要 Padding
│   ├── quantFlag                       ← 量化标志
│   ├── is910C                          ← 平台标志
│   ├── isX2ScaleTypeInt64              ← Scale 类型标志
│   └── dequantType (DequantType)       ← 反量化类型
└── cocTiling: CoCTiling                ← 通信-计算切分参数
    ├── m0, k0, n0                      ← 分块大小
    ├── mLoop, kLoop, nLoop             ← 分块循环数
    ├── pValue                          ← 流水线粒度
    ├── ubMoveNum                       ← UB 搬运量
    ├── commNpuSplit                    ← NPU 维度切分
    ├── commDataSplit                   ← 数据维度切分
    ├── commDirect                      ← 通信方向 (0:数据优先, 1:NPU优先)
    ├── lenPerLoop                      ← 每轮数据量
    ├── swizzlCount, swizzlDirect       ← 核间负载均衡参数
    ├── splitK                          ← K 维切分
    ├── write2OtherRank                 ← 是否写其他卡
    └── withSerialMode                  ← 串行模式
```

### 5.2 CoCTiling 字段含义

```
CoCTiling 通信-计算编排参数:

┌──────────────┬───────────────────────────────────────────────┐
│ m0, n0, k0   │ 分块维度 (由 Tiling 寻优确定)                │
│ mLoop        │ M 维分块数 = DivCeil(m, m0)                  │
│ pValue       │ 通信流水线粒度: 每 pValue 个 M 分块为一轮     │
│ commNpuSplit │ NPU 方向切分: 同时向多少个目标卡发送          │
│ commDataSplit│ 数据方向切分: 每个核负责的数据分片数          │
│ commDirect   │ 0=数据优先搬移, 1=NPU 优先搬移               │
│ lenPerLoop   │ 每轮搬移数据量                               │
│ swizzlCount  │ Swizzle 块大小 (改善 L2 Cache 命中)          │
│ swizzlDirect │ Swizzle 方向: 0=Zn, 1=Nz                    │
└──────────────┴───────────────────────────────────────────────┘
```

---

## 6. 内存管理

### 6.1 Workspace 布局

```
Workspace 布局 (按偏移递增):
┌─────────────────────────────────────────────────┐
│ [0, aAlignSize)                                  │
│   A 矩阵 Padding 缓冲 (有条件分配)               │
│   m_align * k_align * sizeof(X1Type)             │
├─────────────────────────────────────────────────┤
│ [aAlignSize, aAlignSize+bAlignSize)              │
│   B 矩阵 Padding 缓冲 (有条件分配)               │
│   k_align * n_align * sizeof(X2Type)             │
├─────────────────────────────────────────────────┤
│ [aAlignSize+bAlignSize,                          │
│  aAlignSize+bAlignSize+m*n*worldSize*sizeof32)   │
│   INT32 累加缓冲 (量化场景，有条件分配)           │
│   m * n * worldSize * sizeof(int32_t)            │
├─────────────────────────────────────────────────┤
│ [gm_scale_workspace_st, ...)                     │
│   PerToken Scale 聚合缓冲 (量化场景)             │
│   m * worldSize * sizeof(float)                  │
└─────────────────────────────────────────────────┘
```

### 6.2 UB 分配

```
AIV 侧 UB (USED_UB_SIZE = 160KB):
┌─────────────────────────────────────────────────┐
│ TBuf<VECCALC> uBuf_ (160KB)                     │
│ ├── copyTensor0 (Ping): ~80KB                    │
│ ├── copyTensor1 (Pong): ~80KB                    │
│ │   (UB_OFFSET = 97440 bytes / sizeof(X1Type))  │
│ └── 用于 MoveWithSplit, MoveResultFromPeerMemToOut│
│     等数据搬移操作的 Ping-Pong 缓冲              │
└─────────────────────────────────────────────────┘

DequantRunner UB (由 Catlass resource 管理):
├── ubCList[2]:     int32_t [TileShape::COUNT] × 2 stages
├── ubScaleList[2]: float [TileShape::COLUMN] × 2 stages
├── ubPerTokenScaleList[2]: float [TileShape::ROW] × 2 stages
├── ubDList[2]:     YType [TileShape::COUNT] × 2 stages
├── ubCFp32:        float [TileShape::COUNT]
├── ubMul:          float [TileShape::COUNT]
└── ubPerTokenScaleBrcb: float [TileShape::ROW * BYTE_PER_BLK]
```

### 6.3 Window (IPC 共享内存) 布局

```
每张卡的 Window 地址: stateAddrPerRank[rankId] / stateAddrPerRank[other]
┌─────────────────────────────────────────────────┐
│ [0, FLAG_OFFSET × sizeof(int32_t))  ≈ 180MB     │
│   数据区 (Ping-Pong):                            │
│   ├── Ping 区 (flag_idx=0):                      │
│   │   ├── Rank0 数据: m0 * k_align * pValue     │
│   │   ├── Rank1 数据: m0 * k_align * pValue     │
│   │   └── ...                                    │
│   │   总计: m0 * k_align * pValue * worldSize   │
│   ├── Pong 区 (flag_idx=1):                      │
│   │   └── 同 Ping 结构                           │
│   └── PerToken Scale 区 (cal_idx==1 时写入):     │
│       ├── Rank0 scale: m * sizeof(float)         │
│       └── ...                                    │
├─────────────────────────────────────────────────┤
│ [FLAG_OFFSET × 4, ...)                           │
│   状态区 (Flag):                                 │
│   ├── FLAG_ZERO_IDX: 数据写入就绪标志            │
│   ├── FLAG_ONE_IDX:  数据消费完成标志            │
│   ├── FLAG_TWO_IDX:  Scale 写入就绪标志          │
│   └── FLAG_THREE_IDX: Scale 消费完成标志         │
└─────────────────────────────────────────────────┘
```

---

## 7. 通信模式分析

### 7.1 AllGather 实现: Window IPC 直写

V2 的核心创新是使用 Window IPC 共享内存实现 AllGather，而非 V1 的 HCCL API:

```
V1 (HCCL API):
  本卡 A → HCCL AllGather → gatherOut[worldSize, M, K]

V2 (Window IPC):
  本卡 AIV → 直接写入其他卡的 Window → 其他卡的 AIC 从 Window 读取
  无需经过 HCCL 协议栈，延迟更低
```

### 7.2 数据流路径

```
                  ┌──── AIV 核 ────┐     ┌──── AIC 核 ────┐
                  │                │     │                │
Rank 0:           │                │     │                │
  A[0:M, 0:K] ──→│ Pad(512B对齐)  │     │                │
                  │ MoveWithSplit  │──→──│                │
                  │ (写入 Window)  │     │                │
                  │                │     │ CatlassMatmul  │
Rank 1:           │                │     │ (从 Window 读取│
  A[0:M, 0:K] ──→│ Pad(512B对齐)  │     │  其他卡 A 数据 │
                  │ MoveWithSplit  │──→──│  + 本卡 B 矩阵│
                  │ (写入 Window)  │     │  → C 矩阵)    │
                  │                │     │                │
                  │ MoveResult     │     │                │
                  │ (Window→Out)   │←────│                │
                  └────────────────┘     └────────────────┘

同时:
  AIV核0: 负责 Pad + MoveWithSplit (将 A 写入其他卡 Window)
  AIV核1: 负责 MoveResultFromPeerMemToOut (从 Window 读取到 gatherOut)
  AIC核:  负责 CatlassMatmul (本地 + 远程 A 矩阵 × B 矩阵)
```

### 7.3 同步点汇总

| 同步点 | 类型 | 参与者 | 目的 |
|--------|------|--------|------|
| SetAndWaitAivSync | AIV 核间 barrier | 所有 AIV 核 | 核内子任务同步 |
| CrossRankSyncV1 | 跨 Rank flag | 所有 rank 的 Window | 通知数据写入就绪 |
| CrossRankSyncV2 | 跨 Rank flag | 所有 rank 的 Window | 确认数据消费完成 |
| SetAicSync/WaitEvent | AIV↔AIC 跨核 | AIV+AIC | AIV 通知 AIC 可以计算 |
| FFTSCrossCoreSync | AIC 核间 | 所有 AIC 核 | MatMul 轮次完成 |
| SyncAll | 全局 | AIV+AIC | 最终同步 |

---

## 8. 量化与精度分析

### 8.1 量化场景判断

```cpp
// 第272-275行
needAivDequant = quantFlag && (dequantType == PER_TOKEN || YType == bfloat16_t);
needPerChannel = is_int4 || (quantFlag && !(int64_scale && fp16_output));
needPerToken = quantFlag && (dequantType == PER_TOKEN);
```

### 8.2 精度路径对比

| 输入类型 | 输出类型 | 累加类型 | 反量化方式 | PerChannel | PerToken |
|----------|----------|----------|-----------|------------|----------|
| BF16/BF16 | BF16 | BF16 | 无 | N/A | N/A |
| FP16/FP16 | FP16 | FP16 | 无 | N/A | N/A |
| INT8/INT8 (scale=float) | FP16/BF16 | int32 | PerChannel × Cast | Y | N |
| INT8/INT8 (scale=int64) | FP16 | int32 | Fixpipe 硬件反量化 | Y (fixpipe) | N |
| INT8/INT8 (scale=int64) | BF16 | int32 | PerChannel+PerToken | Y | Y |
| INT8/INT8 (PER_TOKEN) | FP16/BF16 | int32 | PerChannel × PerToken | Y | Y |
| INT4/INT4 | FP16/BF16 | int32 | PerChannel × Cast | Y | 可选 |

### 8.3 Fixpipe 模式

当 `quantFlag && YType==half && isX2ScaleTypeInt64 && !int4` 时启用 Fixpipe 硬件反量化:
- AIC 的 Fixpipe 单元在 L0C → GM 写出时直接应用 int64 scale
- 无需 AIV 侧额外反量化
- 由 `FixpipeBlockMmad` (block_mmad_preload_fixpipe.h) 实现

### 8.4 AIV Dequant 流水线

由 `BlockEpilogue` (all_gather_matmul_aiv_mode_block_epilogue_dequant.h) 实现:

```
perChannel + perToken 模式:
  GM(int32 C) → UB(int32 C)
                         → Cast FP32
                         → × perChannelScale (RowBroadcastMul)
                         → × perTokenScale (ColumnBroadcastMul)
                         → Cast YType
                         → GM(YType)

UB 双缓冲 (UB_STAGES=2):
  Stage 0: 读C+Scale → 计算
  Stage 1: 计算 → 写D (流水线重叠)
```

---

## 9. 多核并行策略

### 9.1 核角色分配

```
MIX_AIC_1_2 模式 (1 AIC + 2 AIV per 物理核):
┌────────────────────────────────────────────────┐
│ blockIdx: 0, 1, ..., coreNum-1                 │
│ 每个 blockIdx 对应:                             │
│   ├── AIC 子核 (自动)                          │
│   ├── AIV 子核 0 (aivIdx=0):                   │
│   │   负责: Padding + MoveWithSplit + Dequant  │
│   └── AIV 子核 1 (aivIdx=1):                   │
│       负责: MoveResultFromPeerMemToOut          │
│                                                 │
│ AIV 任务分发:                                   │
│   aivIdx==0 && blockIdx < core_count:           │
│     → 执行 MoveWithSplit (将 A 写入其他卡)      │
│   aivIdx==1 && blockIdx >= core_count:          │
│     && blockIdx < core_count + worldSize:        │
│     → 执行 MoveResultFromPeerMemToOut           │
└────────────────────────────────────────────────┘
```

### 9.2 Swizzle 负载均衡

`GetBlockIdx()` (all_gather_matmul_aiv_mode_util.h 第186-219行) 实现 Z 形/N 形 swizzle:
- 方向 0 (Zn): M 维按 swizzlCount 分块，块内 Z 形遍历 N 维
- 方向 1 (Nz): N 维按 swizzlCount 分块，块内 Z 形遍历 M 维
- 目的: 改善 L2 Cache 局部性，避免相邻核访问相同 Cache Line

---

## 10. 与 V1 的核心差异

| 特性 | V1 (all_gather_matmul) | V2 (all_gather_matmul_v2) |
|------|----------------------|--------------------------|
| 通信实现 | HCCL API (AllGather) | Window IPC 直写 |
| MatMul 引擎 | AscendC MatmulImpl | Catlass 模板库 |
| 量化支持 | Static/Dynamic Quant | PerChannel/PerToken/Fixpipe |
| A 矩阵格式 | ND | ND (带可选 Padding) |
| B 矩阵格式 | ND | ND 或 FRACTAL_NZ |
| INT4 支持 | 不支持 | 支持 |
| FP8 支持 | 不支持 | 支持 (APT 内核) |
| PerToken Scale AllGather | 不需要 | 需要 (量化场景) |
| 多核模式 | AIC 独立 | MIX_AIC_1_2 (AIC+2×AIV) |
| 适用平台 | 910B/910_93 | 910B/910_93/950 |
| 核心优势 | 简单通用 | 延迟更低，通信更直接 |

---

## 11. 函数调用关系

```
all_gather_matmul_v2 (内核入口)
├── AllGatherMatmulAIVMode::Init()
│   ├── AICInit()
│   └── AIVInit()
└── AllGatherMatmulAIVMode::Process()
    ├── [AIV] Padding()
    │   └── PaddingRunner::Run()
    │       └── TemplatePadder::operator()<AIV>()
    │           ├── PaddingMatrix<A> (可选)
    │           └── PaddingMatrix<B> (可选)
    ├── [AIV] 主循环:
    │   ├── [cal_idx==1] AllGatherPerTokenScale()
    │   │   ├── MoveResultFromSrcToDst()
    │   │   ├── CrossRankSyncV1()
    │   │   ├── MoveResultFromSrcToDst()
    │   │   └── CrossRankSyncV2()
    │   ├── CrossRankSyncV1()
    │   ├── MoveWithSplit()
    │   │   ├── MoveToOtherRankWithSkip() (comm_direct==1)
    │   │   │   ├── CopyGmToUbufAlignB16()
    │   │   │   └── CopyUbufToGmAlignB16() (× ranks)
    │   │   └── MoveResultToDst() (comm_direct==0)
    │   ├── MoveResultFromPeerMemToOut()
    │   │   └── CopyGmToUbuf + CopyUbufToGmAlignB16 (stride)
    │   ├── CrossRankSyncV2()
    │   ├── SetAicSync()
    │   └── Dequant()
    │       └── DequantRunner::Run()
    │           └── BlockEpilogue::operator()
    │               ├── CopyGmToUbC/Scale/PerTokenScale
    │               ├── Cast(int32 → fp32)
    │               ├── TileRowBroadcastMul
    │               ├── TileBroadcastOneBlk
    │               ├── TileOneBlkColumnBroadcastMul
    │               ├── Cast(fp32 → YType)
    │               └── CopyUbToGmD
    ├── CatlassMatmul()
    │   └── AllGatherMatmulV2::operator()<AIC>()
    │       ├── CrossCoreWaitFlag(padding)
    │       ├── InitArgs()
    │       ├── DoLocalMatmul()
    │       │   └── BlockMmad::operator()
    │       └── Matmul() / FixpipeMatmul()
    │           └── BlockMmad::operator() / FixpipeBlockMmad::operator()
    └── SyncAll()
```

---

## 12. 依赖关系

```
all_gather_matmul_v2.cpp
├── all_gather_matmul_v2_tiling_key.h        ← 模板参数声明
├── lib/matmul_intf.h                         ← AscendC Matmul 接口
├── all_gather_matmul_aiv_mode.h             ← 主实现类
│   ├── adv_api/hccl/hccl.h                  ← HCCL API
│   ├── kernel_tiling/kernel_tiling.h        ← Kernel Tiling 框架
│   ├── gather_moe_distribute_base.h         ← HCCL 通信结构体
│   ├── all_gather_matmul_aiv_mode_tiling.h  ← Tiling 数据结构
│   ├── all_gather_matmul_aiv_mode_util.h    ← 工具函数
│   ├── all_gather_matmul_aiv_mode_padding.h ← Padding
│   ├── all_gather_matmul_aiv_mode_dequant.h ← 反量化
│   ├── 3rd/template_linear_algebra/...      ← Catlass 模板库
│   │   ├── catlass.hpp
│   │   ├── arch/arch.hpp, resource.hpp, cross_core_sync.hpp
│   │   ├── layout/layout.hpp
│   │   ├── gemm/block/block_mmad.hpp, block_swizzle.hpp
│   │   ├── gemm/dispatch_policy.hpp, gemm_type.hpp, gemm_coord.hpp
│   │   ├── gemm/kernel/padding_matmul.hpp
│   │   ├── epilogue/block/block_epilogue.hpp
│   │   ├── epilogue/tile/*.hpp
│   │   └── detail/callback.hpp
│   └── gather_matmul.hpp                    ← AllGatherMatmulV2 Catlass 内核
│       └── block_mmad_preload_fixpipe.h     ← Fixpipe 量化 MatMul
└── basic_api/kernel_basic_intf.h / kernel_operator.h
```

---

*分析完成时间: 2026-04-17*
