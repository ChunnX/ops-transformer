# aclnnKvCacheTurboQuant

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- 算子功能：对标准 MHA/GQA 的 KV cache 做在线 TurboQuant 编码。输入 BF16 的 KV 向量，
  输出主量化码字（bit-packed）、QJL 残差符号（bit-packed）、输入范数与原尺度残差范数。
- 计算公式与语义细节（严格大于的量化判定、LSB-first 位打包、BF16 ties-to-even 舍入、
  零向量行为）见 [../README.md](../README.md) 与任务包 `golden.py`。

## 函数原型

算子执行接口为**两段式接口**，必须先调用 `aclnnKvCacheTurboQuantGetWorkspaceSize` 获取
workspace 大小和执行器，再调用 `aclnnKvCacheTurboQuant` 执行计算。

```cpp
aclnnStatus aclnnKvCacheTurboQuantGetWorkspaceSize(
    const aclTensor *kvVectors,
    const aclTensor *rotationMatrix,
    const aclTensor *qjlMatrix,
    int64_t          mseBits,
    const aclTensor *quantIdxOut,
    const aclTensor *quantQjlOut,
    const aclTensor *quantNormOut,
    const aclTensor *quantGammaOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);

aclnnStatus aclnnKvCacheTurboQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

头文件：`aclnnop/aclnn_kv_cache_turbo_quant.h`。

## 参数说明

### 输入

| 参数名 | 数据类型 | 维度 | 非连续支持 | 说明 |
| --- | --- | --- | --- | --- |
| kvVectors | BF16 | `[T, H_kv, 128]` | 由 `AutoContiguous` 处理 | 待编码的 K 或 V 向量，`T` 为 token 数、`H_kv` 为 KV 头数 |
| rotationMatrix | FP32 | `[128, 128]` | 同上 | 正交旋转矩阵 `H`，推理期固定 |
| qjlMatrix | FP32 | `[128, 128]` | 同上 | 残差高斯投影矩阵 `S`，推理期固定 |

### 属性

| 参数名 | 数据类型 | 说明 |
| --- | --- | --- |
| mseBits | INT64 | 主量化位宽，仅支持 2/3/4，默认 3 |

### 输出

| 参数名 | 数据类型 | 维度 | 说明 |
| --- | --- | --- | --- |
| quantIdxOut | UINT8 | `[T, H_kv, 16 * mseBits]` | 主量化码字，8 通道一组、低位在前打包 |
| quantQjlOut | UINT8 | `[T, H_kv, 16]` | QJL 符号，1 bit/通道、低位在前打包 |
| quantNormOut | BF16 | `[T, H_kv]` | 输入向量 L2 范数 |
| quantGammaOut | BF16 | `[T, H_kv]` | 原尺度残差范数 `n · ‖r‖₂` |

### workspace

`workspaceSize` 由 `aclnnKvCacheTurboQuantGetWorkspaceSize` 返回（框架区 + 算子自身的中间
矩阵区）。**同一输入 shape 下大小固定**，可复用；调用方按框架惯例申请与释放。

## 返回值

| 返回值 | 说明 |
| --- | --- |
| `ACLNN_SUCCESS` | 成功 |
| 其他 | 见 `aclnnStatus` 定义与 `aclGetRecentErrMsg()`；非法入参（`mseBits` 越界、`head_dim ≠ 128`、head 数越界等）由算子定义与 tiling 校验后返回失败 |

## 约束说明

- 仅支持标准 MHA/GQA；不支持 MLA、不支持 3.5-bit outlier 混合模式。
- `head_dim = 128`、`qjl_dim = 128`；`num_kv_heads ∈ [4, 32]`；`mse_bits ∈ {2, 3, 4}`。
- 不支持空 tensor。
- 编码在同一条 stream 上完成；算子内部为**单个融合 kernel**（两段 GEMM 与全部向量阶段都在
  一个 kernel 内，MIX_AIC_1_2），不需要调用方做多段编排。

## 调用示例

```cpp
// 1) 准备输入/输出 aclTensor（BF16/FP32/UINT8），见 examples/test_aclnn_kv_cache_turbo_quant.cpp
// 2) 两段式调用
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnKvCacheTurboQuantGetWorkspaceSize(kvVectors, rotationMatrix, qjlMatrix, /*mseBits=*/3,
                                       quantIdx, quantQjl, quantNorm, quantGamma,
                                       &workspaceSize, &executor);
void *workspace = nullptr;
if (workspaceSize > 0) { aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST); }
aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

完整可运行样例与编译脚本见 [../examples/](../examples/)；精度/性能自测见 [../tests/README.md](../tests/README.md)。
