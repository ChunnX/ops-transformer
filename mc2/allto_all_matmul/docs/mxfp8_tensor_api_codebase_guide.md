# MX-FP8 Tensor API 分支代码索引指南

## 一、分支定义

MX-FP8 tensor_api 分支是 `allto_all_matmul` 算子中，针对 `fp8_e4m3` / `fp8_e5m2` 数据类型，使用 Tensor API (BlockMmad) 替代 MatMulV3 实现矩阵乘的优化路径。该分支去除了转置节点，使用 GM 自动累加完成跨 rank K 轴求和。

**触发条件**：`ORIG_DTYPE_X1` 和 `ORIG_DTYPE_X2` 均为 `DT_FLOAT8_E4M3FN` 或 `DT_FLOAT8_E5M2`。

---

## 二、文件清单与关键位置

### 2.1 算子专属文件（`mc2/allto_all_matmul/` 下）

| 文件 | 用途 | 关键行号/接口 |
|------|------|-------------|
| `op_kernel/allto_all_matmul_apt.cpp` | Kernel 入口 + 宏定义 | `ALLTO_ALL_MX_TENSOR_MATMUL_IMPL` 宏 (line 112), `TPipe pipe` (line 143), 调用入口 (line 163) |
| `op_kernel/arch35/allto_all_mx_tensor_matmul_arch35.h` | Arch35 封装层 | `Init()` 设置 GM 地址和 stride, `ProcessTile()` (line 122), `ProcessTail()` (line 182), `ProcessScale()` (line 104) |
| `op_kernel/arch35/allto_all_mx_tensor_matmul_pipeline.h` | 流水线编排 | `Process()` (line 78): AIV 通信 → VecNotifyCube → AIC 等待 → 计算, `ProcessScale()` (line 94) |
| `op_kernel/arch35/allto_all_matmul_tiling_data.h` | Tiling 数据结构 | `MxTensorMatmulTilingData`, `AlltoAllQuantMatmulTilingData` |
| `op_host/op_tiling/arch35/allto_all_mx_quant_matmul_tiling_base.h/.cpp` | Host 侧 Tiling 计算 | `FillMxTensorTiling()` (base.cpp), `FillMxTensorTailTiling()` |
| `docs/remove_transpose_optimization.md` | 去除转置优化方案文档 | stride 计算、数据布局详解 |
| `docs/tensor_api_replacement.md` | Tensor API 替换方案文档 | BlockMmad 参考、同步优化 |

### 2.2 公共计算模板（`mc2/common/` 下）

| 文件 | 用途 | 关键行号/接口 |
|------|------|-------------|
| `common/op_kernel/mc2_templates/computation/matmul/tensor_mx_matmul_stage.h` | MX-FP8 计算阶段 thin wrapper | `Process(taskIndex)`: 初始化 BlockMmad + Scheduler, rank 循环 GM 累加, SetMultiRankInfo |
| `common/op_kernel/mc2_templates/computation/matmul/matmul_base.h` | Context 结构体 | `MC2MMContext`: baseData + additionalData + tilingDataPtr + rankDim/globalStride/tileStride |
| `common/op_kernel/mc2_templates/computation/matmul/mc2_mx_quant_matmul.h` | MX quant MatMulV3 路径（非 tensor_api） | ASW Kernel, 仅供 MX-FP4 分支使用 |
| `common/op_kernel/mc2_templates/scheduler/pipeline_builder.h` | Pipeline 基础设施 | MC2AlltoAllContext 等 |

### 2.3 BlockMmad 参考代码（`mc2/common/matmul_ref/` 下）

| 文件 | 用途 |
|------|------|
| `matmul_ref/mx_block_mmad_swat.h` | MX BlockMmad 实现（L0 流水 + 硬件 flag 同步） |
| `matmul_ref/block/quant_matmul_mx_block_scheduler_swat.h` | MX BlockScheduler |
| `matmul_ref/tile/tile_mmad_mx.h` | MX tile 级操作 |
| `matmul_ref/tile/copy_scale_gm_to_l1.h` | Scale GM→L1 |
| `matmul_ref/tile/copy_scale_l1_to_l0a.h` | Scale L1→L0A |
| `matmul_ref/tile/copy_scale_l1_to_l0b.h` | Scale L1→L0B |
| `matmul_ref/tile/pad_mx_kl1.h` | MX K 轴 L1 padding |

---

## 三、数据流

```
┌─────────┐   ┌──────────┐   ┌─────────────────────────────────────────────┐
│ 通信    │ → │ AIV→AIC  │ → │ matmul (AIC)                                │
│ (AIV)   │   │ Sync     │   │ BlockMmad(rank循环, k=rankK, GM累加)       │
│ HCCL-CCU│   │ Flag 8/9 │   │ SetMultiRankInfo → 多rank地址自动切换      │
└─────────┘   └──────────┘   └─────────────────────────────────────────────┘
     ↓              ↓                          ↓
 commOutGM_    CrossCoreFlag             y_ (GM累加输出)
 commX1ScaleGM1_
```

**核间同步**：AIV 通信完成后通过 `CrossCoreSetFlag<0, PIPE_MTE3>(8)` → `CrossCoreWaitFlag(8)` → `CrossCoreSetFlag<2, PIPE_MTE3>(9)` 通知 AIC；AIC 通过 `CrossCoreWaitFlag(9)` 等待。

**多 rank 计算**：BlockMmad 内部通过 `SetMultiRankInfo(rankDim, rankK, aGmAddrs, scaleAGmAddrs)` 自动处理多 rank 地址，无需外层 rank 循环。

---

## 四、TPipe 使用状态（删除后）

**已删除**：mxfp8 tensor_api 分支的计算节点已不再使用 TPipe。

| 原使用位置 | 原操作 | 状态 |
|-----------|--------|------|
| `tensor_mx_matmul_stage.h` 构造函数 | 接收 `TPipe*` | ✅ 已删除 |
| `tensor_mx_matmul_stage.h` 成员 | `TPipe *tPipePtr_` | ✅ 已删除 |
| `tensor_mx_matmul_stage.h:69` | `tPipePtr_->Reset()` | ✅ 已删除 |
| `allto_all_mx_tensor_matmul_arch35.h` Init | 接收 `TPipe*` | ✅ 已删除 |
| `allto_all_mx_tensor_matmul_arch35.h` 成员 | `TPipe *tPipe_` | ✅ 已删除 |
| `allto_all_matmul_apt.cpp` 宏 | `matmulImplName(&pipe)` → `matmulImplName()` | ✅ 已删除 |
| `allto_all_matmul_apt.cpp` 宏 | `op.Init(..., &pipe)` → `op.Init(...)` | ✅ 已删除 |

**仍保留**（其他分支使用）：
- `TPipe pipe;` 在 `allto_all_matmul_apt.cpp:143` — FP/KC-QUANT/MX-QUANT 分支仍通过 `&pipe` 使用

---

## 五、与其他分支的差异

| 分支 | 数据类型 | 矩阵乘引擎 | TPipe 使用 | 转置节点 |
|------|---------|-----------|-----------|---------|
| FP | fp16/bf16 | BlockMmad (tensor_api) | `Reset()` + 传给 MatMulV3 Init | 有 |
| MX-FP8 (tensor_api) | fp8_e4m3/e5m2 | BlockMmad (tensor_api) | **无** (已删除) | 无 (GM累加) |
| MX-FP4 (ASW) | fp4_e2m1 | MatMulASWKernel | `Reset()` + 传给 ASW Init | 有 |
| KC-QUANT | 动态量化 | MatMulV3 + Fp8DynamicQuant | `Reset()` + `InitBuffer` + `AllocEventID` | 有(量化+转置) |