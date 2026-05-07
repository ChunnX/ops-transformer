# AlltoAllMatmul 矩阵乘替换为 Tensor API —— 改动方案记录

## 一、目标

将 AlltoAllMatmul 算子中的矩阵乘从 MC2 模板系统（`Mc2FpMatmul → Mc2MatmulAswKernel` / `MC2MXQuantMatmul → MatMulASWKernel`）全部替换为 Tensor API 实现，同时：
- 通信层（HCCL-CCU AllToAll）、转置层（MC2VecTranspose）、流水线编排、核间同步保持不变
- Tiling 策略简化为固定 `baseM=baseN=baseK=128`
- 与 KC-QUANT 路径零耦合

## 二、Tensor API 依赖升级

原 `mc2/3rd/tensor_api/`（来自 `matmul_kernel` 参考工程）的 `CopyGM2L1` 类型检查不支持 `fp8_e8m0_t`（MX scale 类型），替换为 `cann-sample-master/third_party/tensor_api/`：

- 新增 `CheckGm2L1ScaleDataType()` 函数，使 `CopyGM2L1` → `ScaleAND→Zz` / `ScaleBDN→Nn` 分支持 `fp8_e8m0_t`
- 保留 `mad_mx`（`MmadMxInstr`）和 `MmMadType::MX` 基础设施，供自定义 MmadTrait 调用

## 三、架构对比

```
原架构（FP）：                        新架构（FP）：
  MC2FpMatmul                           TensorMatmulStage (thin wrapper)
    └── Mc2MatmulAswKernel                ├── BlockScheduler（M×N 蛇形切分）
          (3rd/mat_mul_v3/)               └── BlockMmad（tensor_api Mad）

原架构（MX-FP8）：                    新架构（MX-FP8）：
  MC2MXQuantMatmul                       TensorMXQuantMatmulStage
    └── MatMulASWKernel                   └── tensor_api Copy + mad_mx
          (lib/matmul_intf.h)                   (3rd/tensor_api/)
```

## 四、FP 路径实现方式

FP 路径不手写流水线，而是将参考项目 `references/matmul_kernel` 的核心代码**原样拷贝**到 `mc2/common/matmul_ref/`（共 11 个文件），只改 include 路径：

```
mc2/common/matmul_ref/
├── matmul_block_mmad.h          ← BlockMmad 核心（GM→L1→L0→MMAD→GM 流水线）
├── matmul_block_scheduler.h     ← BlockScheduler（M×N 蛇形切分 + while 循环驱动）
├── block_mmad.h                 ← SFINAE 基类
├── block_scheduler_policy.h     ← MatmulSwatScheduler 策略标签
├── block_scheduler_utils.h      ← BlockSchedulerSelector 转发声明
├── common_utils.h               ← L0A_SIZE / CeilDiv / NO_FULL_LOAD_MODE 等
├── layout_utils.h               ← layout::RowMajor / TagToTrans
├── tuple_utils.h                ← Get<> 模板
├── integra_constant.h           ← always_false
├── dispatch_policy.h            ← MatmulMultiBlockPolicy
└── matmul_constant.h            ← FINAL_ACCUMULATION 等常量
```

`tensor_matmul_stage.h` 位于 `mc2/common/op_kernel/mc2_templates/computation/matmul/`，是约 120 行的 thin wrapper：

```cpp
Process(taskIndex):
  mmadOp_.Init(problemShape=(tileM,N,K), tileShape=(128,128,128), enableL0cPingPong)
  bs.Init(problemShape, baseM=128, baseN=128)

  while (bs.GetTileIdx(blockIdx)):      // ← 参考的 BlockScheduler while 循环
      curM, curN = blockIdx[M], blockIdx[N]
      mmadOp_(gmBlockA, gmBlockB, gmBlockC, singleShape)  // ← 参考的 BlockMmad
```

与参考 `MatmulKernel::Process` 逐行对齐：BlockScheduler 做 M×N 双向切分保证每块 ≤128×128，while 循环驱动 BlockMmad，L0C 双缓冲跨块避免 fixpipe WAW 冲突。

## 五、新增文件

### 5.1 `op_kernel/arch35/tensor_matmul_tiling_data.h`

简化 Tiling 数据结构，替代 `Mc2MatMulV3TilingData`（10+ 字段）：

```cpp
struct TensorMatmulTilingData {
    uint32_t tileM;
    uint32_t fullK;
    uint32_t fullN;
};
```

### 5.2 `mc2/common/matmul_ref/` — 参考代码库（11 个文件）

从 `references/matmul_kernel` 原样拷贝，只改内部 `#include` 的相对路径（`kernel_utils/xxx` → `xxx`，`include/tensor.h` → `tensor.h` 等）。语义零变更。

### 5.3 `mc2/common/op_kernel/mc2_templates/computation/matmul/tensor_matmul_stage.h` — FP 路径

提供 `Init / GetContextPtr / Process / End` 接口，替代 `MC2FpMatmul`。内部使用 `Block::BlockMmad` + `BlockScheduler` 的 while 循环，零手写流水线。

### 5.4 `op_kernel/arch35/tensor_mx_matmul_stage.h` — MX-FP8 路径

提供相同接口，替代 `MC2MXQuantMatmul`。内含以下自定义原子（源自 `matmul_story` 样例）：

| 自定义原子 | 功能 | 来源 |
|-----------|------|------|
| `Mc2Mx::CopyL12L0MxScaleA3510` | ScaleA L1(Zz)→L0A，调用 `load_cbuf_to_ca_mx` | `copy_scale_l1_to_l0a.h` |
| `Mc2Mx::CopyL12L0MxScaleB3510` | ScaleB L1(Nn)→L0B，调用 `load_cbuf_to_cb_mx` | `copy_scale_l1_to_l0b.h` |
| `Mc2Mx::MmadMx` | MX MMAD，调用 `mad_mx` intrinsic | `tile_mmad_mx.h` |

### 5.5 `mc2/3rd/tensor_api/` — 依赖升级

从 `cann-sample-master/third_party/tensor_api/` 完整替换。

## 六、修改文件

### 6.1 `op_kernel/arch35/allto_all_matmul_tiling_data.h`

- `include`：`mat_mul_v3/mat_mul_tiling_data.h` → `tensor_matmul_tiling_data.h`
- `AlltoAllMatmulTilingData`：`Mc2MatMulV3TilingData` → `TensorMatmulTilingData`
- `AlltoAllQuantMatmulTilingData`：末尾新增 `TensorMatmulTilingData mxTensorTileTilingData` 和 `mxTensorTailTilingData`（不破坏 KC 路径的偏移）

### 6.2 `op_kernel/arch35/allto_all_matmul_arch35.h`

`ProcessTile` 和 `ProcessTail` 中，将 `tilingDataPtr` 从 `&mc2MmV3TileTilingData` 改为先填入维度再指向 `&tensorMatmulTileTilingData`。

### 6.3 `op_kernel/arch35/allto_all_mx_quant_matmul_arch35.h`

- `ProcessTile` / `ProcessTail` 中同时填入 `mxTensorTileTilingData` 和 `mxTensorTailTilingData`
- 使用 `if constexpr` 编译期判断 context 类型，自动选择指向 `mxTensor*`（tensor API）或 `mc2QuantMm*`（MatMulASWKernel）

### 6.4 `op_kernel/allto_all_matmul_apt.cpp`

| 宏 | 改动 |
|----|------|
| `ALLTO_ALL_MATMUL_APT_FP_IMPL` | `DEFINE_MC2_MATMUL_...` → `TensorMatmulStage<TensorMatmulTilingData>` |
| `ALLTO_ALL_MX_TENSOR_MATMUL_IMPL`（新增） | fp8 MX 路径用 `TensorMXQuantMatmulStage<TensorMatmulTilingData, X2TRANSPOSE>` |
| `ALLTO_ALL_MX_QUANT_MATMUL_IMPL`（保留） | fp4 MX 路径原样使用 `MatMulASWKernel` |

dispatch 逻辑：
```cpp
// FP: fp16 / bf16
ALLTO_ALL_MATMUL_APT_FP_IMPL(...)                        // tensor API

// MX-FP8: fp8_e4m3 / fp8_e5m2
ALLTO_ALL_MX_TENSOR_MATMUL_IMPL(...)                     // tensor API

// MX-FP4: fp4_e2m1
ALLTO_ALL_MX_QUANT_MATMUL_IMPL(...)                      // MatMulASWKernel（原样）

// KC-QUANT: 其余量化组合
ALLTO_ALL_KC_QUANT_MATMUL_IMPL(...)                      // MatMulV3（原样）
```

### 6.5 `op_host/op_tiling/arch35/allto_all_fp_matmul_tiling_base.h/.cpp`

- `.h`：移除 `MatMulV3` 相关 include 和 `mmV3Args_` / `mmV3compileInfo_` 成员变量
- `.cpp`：`DoMMTiling()` 改为直接 `return ge::GRAPH_SUCCESS`（空转）；删除 `DoMatmulV3Tiling` 和 `PrintMMV3TilingData` 定义

### 6.6 `op_host/CMakeLists.txt`

新增 tensor_api include 路径：
```cmake
-I${CMAKE_CURRENT_SOURCE_DIR}/../../3rd/tensor_api/
-I${CMAKE_CURRENT_SOURCE_DIR}/../../3rd/tensor_api/include/
```

## 七、最终状态总览

| 路径 | 数据类型 | 矩阵乘引擎 | 状态 |
|------|---------|-----------|:---:|
| FP | fp16 / bf16 | `TensorMatmulStage`（BlockScheduler + BlockMmad） | ✅ |
| MX-FP8 | fp8_e4m3 / fp8_e5m2 | `TensorMXQuantMatmulStage`（tensor API `mad_mx`） | ✅ |
| MX-FP4 | fp4_e2m1 | `MatMulASWKernel`（原样） | ✅ |
| KC-QUANT | 动态量化 | `MatMulV3`（原样） | ✅ |

## 八、未变部分

以下模块零改动：
- 通信层（`MC2AlltoAllPrimitives`，HCCL-CCU）
- 转置层（`MC2VecTranspose`）
- 流水线编排（`AlltoAllMatmulPipeLine`、`AlltoAllMXQuantMatmulPipeLine`、`AlltoAllKCQuantMatmulPipeLine`）
- 顶层封装（`AlltoAllMatmulArch35`、`AlltoAllMxQuantMatmulArch35`）
- op_api / op_graph / op_host 框架
- 核间同步机制（`CrossCoreSetFlag` / `WaitFlag`）
- KC-QUANT 全部代码路径
- MX-FP4 全部代码路径
