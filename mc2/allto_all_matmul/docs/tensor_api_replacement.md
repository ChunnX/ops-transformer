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
原架构（FP）：                    原架构（MX-FP8）：
  MC2FpMatmul                        MC2MXQuantMatmul
    └── Mc2MatmulAswKernel             └── MatMulASWKernel
          (3rd/mat_mul_v3/)                   (lib/matmul_intf.h)

新架构（FP）：                    新架构（MX-FP8）：
  TensorMatmulStage                  TensorMXQuantMatmulStage
    └── tensor_api Copy/Mad            └── tensor_api Copy + mad_mx
          (3rd/tensor_api/)                 (3rd/tensor_api/)
```

## 四、新增文件

### 4.1 `op_kernel/arch35/tensor_matmul_tiling_data.h`

简化 Tiling 数据结构，替代 `Mc2MatMulV3TilingData`（10+ 字段）：

```cpp
struct TensorMatmulTilingData {
    uint32_t tileM;   // 当前 M 维度（tileM 或 tailM）
    uint32_t fullK;   // 完整 K = rankK * rankDim
    uint32_t fullN;   // 完整 N = rankN
};
```

### 4.2 `op_kernel/arch35/tensor_matmul_stage.h` — FP 路径

提供 `Init / GetContextPtr / Process / End` 接口，替代 `MC2FpMatmul`。

`Process(taskIndex)` 执行流程：

```
GM→L1: A(fp16/Nz) + B(fp16/Zn) 双缓冲 ping-pong
  └→ N-tiling 外循环（baseN=128）
       └→ K-tiling 内循环
            └→ L1→L0A(Nz) / L1→L0B(Zn)  双缓冲 ping-pong
                 └→ Mad(fp16×fp16→fp32 累加到 L0C)
  └→ Fixpipe L0C→GM
```

### 4.3 `op_kernel/arch35/tensor_mx_matmul_stage.h` — MX-FP8 路径

提供相同接口，替代 `MC2MXQuantMatmul`。内含以下自定义原子（源自 `cann-sample-master/Samples/2_Performance/matmul_story/matmul_recipes/include/tile/`）：

| 自定义原子 | 功能 | 来源 |
|-----------|------|------|
| `Mc2Mx::CopyL12L0MxScaleA3510` | ScaleA L1(Zz)→L0A，调用 `load_cbuf_to_ca_mx` | `copy_scale_l1_to_l0a.h` |
| `Mc2Mx::CopyL12L0MxScaleB3510` | ScaleB L1(Nn)→L0B，调用 `load_cbuf_to_cb_mx` | `copy_scale_l1_to_l0b.h` |
| `Mc2Mx::MmadMx` | MX MMAD，调用 `mad_mx` intrinsic | `tile_mmad_mx.h` |

`Process(taskIndex)` 执行流程：

```
GM→L1: A(fp8/Nz) + ScaleA(fp8_e8m0/Zz) + B(fp8/Zn) + ScaleB(fp8_e8m0/Nn)
  └→ N-tiling 外循环
       └→ K-tiling 内循环
            └→ L1→L0A: A(Nz) + ScaleA(Zz→CA via load_cbuf_to_ca_mx)
            └→ L1→L0B: B(Zn) + ScaleB(Nn→CB via load_cbuf_to_cb_mx)
            └→ mad_mx（硬件自动 dequant: data × scale → 累加到 L0C）
  └→ Fixpipe L0C→GM
```

### 4.4 `mc2/3rd/tensor_api/` — 依赖升级

从 `cann-sample-master/third_party/tensor_api/` 完整替换。

## 五、修改文件

### 5.1 `op_kernel/arch35/allto_all_matmul_tiling_data.h`

- `include`：`mat_mul_v3/mat_mul_tiling_data.h` → `tensor_matmul_tiling_data.h`
- `AlltoAllMatmulTilingData`：`Mc2MatMulV3TilingData` → `TensorMatmulTilingData`
- `AlltoAllQuantMatmulTilingData`：末尾新增 `TensorMatmulTilingData mxTensorTileTilingData` 和 `mxTensorTailTilingData`（不破坏 KC 路径的偏移）

### 5.2 `op_kernel/arch35/allto_all_matmul_arch35.h` — FP 路径

`ProcessTile` 和 `ProcessTail` 中，将 `tilingDataPtr` 从 `&mc2MmV3TileTilingData` 改为先填入维度再指向 `&tensorMatmulTileTilingData`。

### 5.3 `op_kernel/arch35/allto_all_mx_quant_matmul_arch35.h` — MX 路径

- `ProcessTile` / `ProcessTail` 中同时填入 `mxTensorTileTilingData` 和 `mxTensorTailTilingData`
- 使用 `if constexpr` 编译期判断 `computationContext->tilingDataPtr` 类型，自动选择指向 `mxTensor*`（tensor API）或 `mc2QuantMm*`（MatMulASWKernel）

### 5.4 `op_kernel/allto_all_matmul_apt.cpp`

| 宏 | 改动 |
|----|------|
| `ALLTO_ALL_MATMUL_APT_FP_IMPL` | `DEFINE_MC2_MATMUL_CONTEXT/MATMUL_FP` → `TensorMatmulStage<TensorMatmulTilingData>` |
| `ALLTO_ALL_MX_TENSOR_MATMUL_IMPL`（新增） | fp8 MX 路径用 `TensorMXQuantMatmulStage<TensorMatmulTilingData, X2TRANSPOSE>` |
| `ALLTO_ALL_MX_QUANT_MATMUL_IMPL`（保留） | fp4 MX 路径原样使用 `MatMulASWKernel` |

dispatch 逻辑：
```cpp
// FP: fp16 / bf16
ALLTO_ALL_MATMUL_APT_FP_IMPL(...)

// MX-FP8: fp8_e4m3 / fp8_e5m2
ALLTO_ALL_MX_TENSOR_MATMUL_IMPL(...)    // tensor API

// MX-FP4: fp4_e2m1
ALLTO_ALL_MX_QUANT_MATMUL_IMPL(...)     // MatMulASWKernel（原样）

// KC-QUANT: 其余量化组合
ALLTO_ALL_KC_QUANT_MATMUL_IMPL(...)     // MatMulV3（原样）
```

### 5.5 `op_host/op_tiling/arch35/allto_all_fp_matmul_tiling_base.h/.cpp`

- `.h`：移除 `MatMulV3` 相关 include（`matmul_v3_tiling_strategy.h`、`matmul_v3_common_advanced.h`）和 `using namespace mc2_matmul_v3_advanced`
- `.h`：移除 `DoMatmulV3Tiling` 方法声明、`PrintMMV3TilingData` 方法声明、`mmV3Args_` / `mmV3compileInfo_` 成员变量
- `.cpp`：`DoMMTiling()` 从调用 `DoMatmulV3Tiling` 改为直接 `return ge::GRAPH_SUCCESS`（空转）
- `.cpp`：删除 `DoMatmulV3Tiling` 函数定义和 `PrintMMV3TilingData` 函数定义
- `.cpp`：`PrintAlltoAllMatmulTilingData` 改为打印 `TensorMatmulTilingData` 字段

### 5.6 `op_host/CMakeLists.txt`

在 `add_ops_compile_options` 中新增 tensor_api include 路径：

```cmake
-I${CMAKE_CURRENT_SOURCE_DIR}/../../3rd/tensor_api/
-I${CMAKE_CURRENT_SOURCE_DIR}/../../3rd/tensor_api/include/
```

## 六、最终状态总览

| 路径 | 数据类型 | 矩阵乘引擎 | 状态 |
|------|---------|-----------|------|
| FP | fp16 / bf16 | `TensorMatmulStage` (tensor API `Mad`) | ✅ |
| MX-FP8 | fp8_e4m3 / fp8_e5m2 | `TensorMXQuantMatmulStage` (tensor API `mad_mx`) | ✅ |
| MX-FP4 | fp4_e2m1 | `MatMulASWKernel` (原样) | ✅ |
| KC-QUANT | 动态量化 | `MatMulV3` (原样) | ✅ |

## 七、未变部分

以下模块零改动：
- 通信层（`MC2AlltoAllPrimitives`，HCCL-CCU）
- 转置层（`MC2VecTranspose`）
- 流水线编排（`AlltoAllMatmulPipeLine`、`AlltoAllMXQuantMatmulPipeLine`、`AlltoAllKCQuantMatmulPipeLine`）
- 顶层封装（`AlltoAllMatmulArch35`、`AlltoAllMxQuantMatmulArch35`）
- op_api / op_graph / op_host 框架
- 核间同步机制（`CrossCoreSetFlag` / `WaitFlag`）
- KC-QUANT 全部代码路径
- MX-FP4 全部代码路径
