# MhcPreBackward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

- **接口功能**：`MhcPreBackward` 对应 `MhcPre` 的反向算子，用于计算 mHC（Manifold-Constrained Hyper-Connections）结构中的反向梯度。
- **主要输出**：`gradX`、`gradPhi`、`gradAlpha`、`gradBias`，以及在 `gamma != nullptr` 时输出 `gradGamma`。
- **前向缓存依赖**：`invRms`、`hMix`、`hPre`、`hPost`。
- **可选输入**：`gamma`、`gradXPostOptional`。

## 输入输出说明

### 输入

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| x | 输入 | mHC层输入数据 | BFLOAT16、FLOAT16 | ND | `(B,S,N,D)` 或 `(T,N,D)` | √ |
| phi | 输入 | mHC参数矩阵 | FLOAT32 | ND | `(2N+N*N, N*D)` | √ |
| alpha | 输入 | mHC缩放参数 | FLOAT32 | ND | `(3)` | - |
| gradHIn | 输入 | 对 `h_in` 的梯度 | BFLOAT16、FLOAT16 | ND | `(B,S,D)` 或 `(T,D)` | √ |
| gradHPost | 输入 | 对 `h_post` 的梯度 | FLOAT32 | ND | `(B,S,N)` 或 `(T,N)` | √ |
| gradHRes | 输入 | 对 `h_res` 的梯度 | FLOAT32 | ND | `(B,S,N,N)` 或 `(T,N,N)` | √ |
| invRms | 输入 | 前向缓存的 `inv_rms` | FLOAT32 | ND | `(B,S)` 或 `(T)` | √ |
| hMix | 输入 | 前向缓存的 `h_mix` | FLOAT32 | ND | `(B,S,2N+N*N)` 或 `(T,2N+N*N)` | √ |
| hPre | 输入 | 前向缓存的 `h_pre` | FLOAT32 | ND | `(B,S,N)` 或 `(T,N)` | √ |
| hPost | 输入 | 前向缓存的 `h_post` | FLOAT32 | ND | `(B,S,N)` 或 `(T,N)` | √ |
| gamma | 可选输入 | RMSNorm缩放因子 | FLOAT32 | ND | `(N,D)` | √ |
| gradXPostOptional | 可选输入 | 来自后续路径的 `grad_x` 累加项 | BFLOAT16、FLOAT16 | ND | `(B,S,N,D)` 或 `(T,N,D)` | √ |
| hcEps | 属性 | `h_pre` sigmoid 后使用的 eps 参数 | FLOAT32/DOUBLE 接口传参 | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gradX | 输出 | `x` 的梯度 | BFLOAT16、FLOAT16 | ND | 与 `x` 相同 |
| gradPhi | 输出 | `phi` 的梯度 | FLOAT32 | ND | `(2N+N*N, N*D)` |
| gradAlpha | 输出 | `alpha` 的梯度 | FLOAT32 | ND | `(3)` |
| gradBias | 输出 | bias整体梯度 | FLOAT32 | ND | `(2N+N*N)` |
| gradGamma | 可选输出 | `gamma` 的梯度 | FLOAT32 | ND | `(N,D)` |

## 约束说明

### 确定性计算

- aclnnMhcPreBackward 默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束

1. 输入/输出 Tensor 的数据格式仅支持 `ACL_FORMAT_ND`。
2. `x`、`phi`、`alpha`、`gradHIn`、`gradHPost`、`gradHRes`、`invRms`、`hMix`、`hPre`、`hPost`、`gradX`、`gradPhi`、`gradAlpha`、`gradBias` 不支持空 Tensor。
3. `gamma` 和 `gradXPostOptional` 为可选输入。
4. `gradGamma` 仅在 `gamma != nullptr` 时需要提供。
5. 非连续 Tensor 无需用户预处理，算子内部会转连续。
6. Workspace 需在 Device 侧申请，并严格使用第一段接口返回值。

### 规格约束

| 规格项 | 规格 | 说明 |
| :--- | :--- | :--- |
| `N` | `4`、`6`、`8` | 当前仅支持这三种取值 |
| `D` | `1~16384` | 且需满足 `64` 元素对齐 |
| 数据格式 | `ND` | 输入输出统一使用 ND |

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_mhc_pre_backward.cpp](examples/test_aclnn_mhc_pre_backward.cpp) | 通过接口方式调用 [aclnnMhcPreBackward](docs/aclnnMhcPreBackward.md) 算子。 |

## 参考文档

- [aclnnMhcPreBackward 文档](docs/aclnnMhcPreBackward.md)
