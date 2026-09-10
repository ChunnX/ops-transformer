# aclnnClaGateQuant

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR/Ascend 950DT</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持

<!-- end id3 -->

<!-- npu="310b" id4 -->

- <term>Atlas 200I/500 A2 推理产品</term>：不支持

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas 推理系列产品</term>：不支持

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas 训练系列产品</term>：不支持

<!-- end id6 -->

## 功能说明

- 接口功能：融合算子，实现 CLA（Cross-Layer Attention）两路 head-wise gate 加权合并与双轴动态块量化的组合计算。先对 Global/CLA 分支与 Local/SWA 分支的 Attention 输出分别施加 Sigmoid 门控并加权合并，再将合并结果 reshape 为 [T, K]（K = N×D），同时在 K 方向（[1,32] block，row-wise）和 T 方向（[32,1] block，col-wise）进行基于块的动态量化，输出低精度的 FP8/FP4 张量和对应的 E8M0 缩放因子。量化模式由 `dualAxisFlag` 控制：双轴模式（dualAxisFlag=1，默认）同时输出 row-wise 与 col-wise 两套量化数据；单轴模式（dualAxisFlag=0）仅输出 row-wise 一套量化数据，colDataOut / colScaleOut 传空指针。
- 计算公式：
  
  **阶段1：CLA gate 融合**
  
  - 两路 head-wise Sigmoid 门控（z_g、z_l 为两路 gate logits）：
    
    $$
    s_g = \sigma(z_g), \qquad s_l = \sigma(z_l)
    $$
    
    其中 $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ 为 Sigmoid 函数。
  - CLA gate 融合（逐 token、逐 head）：
    
    $$
    O_{t,n,d} = s_{g,t,n} \cdot O_{g,t,n,d} + s_{l,t,n} \cdot O_{l,t,n,d}
    $$
  - 量化前逻辑矩阵：
    
    $$
    X = \operatorname{reshape}(O, [T, K]), \qquad K = N \times D
    $$
  
  **阶段2：双轴动态块量化**
  
  - **Row-wise 量化（K 方向）**：将 X 在 K 维度上按照32个数进行分组，一组32个数$\{\{V_i\}_{i=1}^{32}\}$量化为$\{mxscale\_row, \{P_i\}_{i=1}^{32}\}$
    
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$
    
    $$
    mxscale\_row = 2^{shared\_exp}
    $$
    
    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale\_row, round\_mode), \space i\space from\space 1\space to\space 32
    $$
  - **Col-wise 量化（T 方向，仅 dualAxisFlag=1）**：将 X 在 T 维度上按照32个数进行分组，一组32个数$\{\{V_j\}_{j=1}^{32}\}$量化为$\{mxscale\_col, \{P_j\}_{j=1}^{32}\}$
    
    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$
    
    $$
    mxscale\_col = 2^{shared\_exp}
    $$
    
    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale\_col, round\_mode), \space j\space from\space 1\space to\space 32
    $$
  - Row-wise 量化后的$P_{i}$按对应的$V_{i}$的位置组成输出rowDataOut，mxscale\_row按对应的 K 维度上的分组组成输出rowScaleOut（每两个相邻分组的 scale 组成一对，存于最后一维，shape 为 [T, ceil(N*D/64), 2]，尾组不足两个时偶数 pad 填充 0）。Col-wise 量化后的$P_{j}$按对应的$V_{j}$的位置组成输出colDataOut，mxscale\_col按对应的 T 维度上的分组组成输出colScaleOut（shape 为 [ceil(T/64), N*D, 2]，同样偶数 pad 填充 0）。
  - emax: 对应数据类型的最大正则数的指数位。
    
    |   DataType    | emax |
| :-----------: | :--: |
|  FLOAT4_E2M1  |  2   |
|  FLOAT4_E1M2  |  0   |
| FLOAT8_E4M3FN |  8   |
|  FLOAT8_E5M2  |  15  |
    
    

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnClaGateQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnClaGateQuant”接口执行计算。

```cpp
aclnnStatus aclnnClaGateQuantGetWorkspaceSize(
  const aclTensor *globalAttn,
  const aclTensor *localAttn,
  const aclTensor *globalGateLogits,
  const aclTensor *localGateLogits,
  char             *roundMode,
  int64_t          scaleAlg,
  int64_t          dstType,
  char            *inputAttnLayout,
  bool            dualAxisFlag,
  const aclTensor *rowDataOut,
  const aclTensor *rowScaleOut,
  const aclTensor *colDataOut,
  const aclTensor *colScaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnClaGateQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnClaGateQuantGetWorkspaceSize

- **参数说明：**
  
  <table><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>参数说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>globalAttn (aclTensor*)</td>
      <td>输入</td>
      <td>Global/CLA 分支 Attention 输出，公式中的 O_g。</td>
      <td><ul><li>shape为[T, N, D]，T 为动态维度（本 rank token 数）。</li><li>数据类型与 localAttn 一致，支持 BF16/FP16。</li><li>数据格式为 ND（逻辑排布 TND，由 inputAttnLayout 入参指定）。</li><li>不支持空 Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>localAttn (aclTensor*)</td>
      <td>输入</td>
      <td>Local/SWA 分支 Attention 输出，公式中的 O_l。</td>
      <td><ul><li>shape与 globalAttn 一致，为 [T, N, D]。</li><li>数据类型与 globalAttn 一致，支持 BF16/FP16。</li><li>数据格式为 ND（逻辑排布 TND）。</li><li>不支持空 Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>globalGateLogits (aclTensor*)</td>
      <td>输入</td>
      <td>Global gate 的 Sigmoid 前值 z_g。</td>
      <td><ul><li>shape为 [T, N]，T 与两路输入一致。</li><li>数据类型与 globalAttn / localAttn 一致。</li><li>数据格式为 ND。</li><li>不支持空 Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>localGateLogits (aclTensor*)</td>
      <td>输入</td>
      <td>Local gate 的 Sigmoid 前值 z_l。</td>
      <td><ul><li>shape为 [T, N]，T 与两路输入一致。</li><li>数据类型与 globalAttn / localAttn 一致。</li><li>数据格式为 ND。</li><li>不支持空 Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>输入</td>
      <td>表示量化数据转换的舍入模式，对应公式中的 round_mode。</td>
      <td><ul><li>当 dstType 为 35/36，对应输出 rowDataOut 和 colDataOut 数据类型为 FLOAT8_E5M2/FLOAT8_E4M3FN 时，仅支持 {"rint"}。</li><li>当 dstType 为 40/41，对应输出 rowDataOut 和 colDataOut 数据类型为 FLOAT4_E2M1/FLOAT4_E1M2 时，支持 {"rint", "floor", "round"}。</li><li>传入空指针时，采用 "rint" 模式。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>输入</td>
      <td>表示 rowScaleOut 和 colScaleOut 的计算方法。</td>
      <td><ul><li>取值范围：{1, 0}，1代表cuBLAS实现, 0代表OCP实现。</li><li>当 dstType 为 40/41（FP4）时仅支持 scaleAlg 为 0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>输入</td>
      <td>表示指定量化转换后 rowDataOut 和 colDataOut 的数据类型。</td>
      <td><ul><li>输入范围为 {35, 36, 40, 41}，分别对应输出 rowDataOut 和 colDataOut 的数据类型为 {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN, 40: FLOAT4_E2M1, 41: FLOAT4_E1M2}。</li><li>当 dstType 为 40/41 时，K=N*D 必须可被 4 整除。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputAttnLayout (char*)</td>
      <td>输入</td>
      <td>输入globalAttn/localAttn tensor 的排布格式。</td>
      <td><ul><li>当前仅支持取值TND。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dualAxisFlag (bool)</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li>取值 true（默认）：双轴量化，同时输出 row-wise 与 col-wise 两套量化数据。</li><li>取值 false：单轴量化，仅输出 row-wise 一套量化数据，此时 colDataOut / colScaleOut 必须传空指针，接口不访问 col 输出；rowDataOut / rowScaleOut 与双轴模式输出逐比特一致。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowDataOut (aclTensor*)</td>
      <td>输出</td>
      <td>表示融合结果 Row-wise 量化后的对应结果，对应公式中的<i>P<sub>i</sub></i>。</td>
      <td><ul><li>shape为 [T, N*D]。</li><li>数据类型由 dstType 决定。</li></ul></td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowScaleOut (aclTensor*)</td>
      <td>输出</td>
      <td>表示 Row-wise 每个 [1,32] 分组对应的量化尺度，对应公式中的 mxscale_row。</td>
      <td><ul><li>shape为 [T, ceil(N*D/64), 2]。</li><li>数据类型为 FLOAT8_E8M0。</li><li>需进行偶数 pad，pad 填充值为 0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>colDataOut (aclTensor*)</td>
      <td>输出</td>
      <td>表示融合结果 Col-wise 量化后的对应结果，对应公式中的<i>P<sub>j</sub></i>。</td>
      <td><ul><li>dualAxisFlag=1 时：shape为 [T, N*D]，数据类型与 rowDataOut 一致（与 rowDataOut 数值编码通常不同）。</li><li>dualAxisFlag=0 时：传空指针，接口不访问该输出。</li></ul></td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>colScaleOut (aclTensor*)</td>
      <td>输出</td>
      <td>表示 Col-wise 每个 [32,1] 分组对应的量化尺度，对应公式中的 mxscale_col。</td>
      <td><ul><li>dualAxisFlag=1 时：shape为 [ceil(T/64), N*D, 2]，数据类型为 FLOAT8_E8M0。</li><li>需进行偶数 pad，pad 填充值为 0。</li><li>dualAxisFlag=0 时：传空指针，接口不访问该输出。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>globalAttn、localAttn、globalGateLogits、localGateLogits、rowDataOut、rowScaleOut、workspaceSize 或 executor 是空指针；dualAxisFlag=1 时 colDataOut 或 colScaleOut 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>globalAttn、localAttn、globalGateLogits、localGateLogits、roundMode、scaleAlg、dstType、rowDataOut、rowScaleOut、colDataOut、colScaleOut 的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>globalAttn、localAttn、globalGateLogits、localGateLogits、rowDataOut、rowScaleOut、colDataOut 或 colScaleOut 的 shape 不满足校验条件（如 globalAttn/localAttn shape 不一致、gate logits 非 [T, N]）。</td>
    </tr>
    <tr>
      <td>roundMode、scaleAlg、dstType不符合当前支持的值（如 dstType 不在 {35, 36, 40, 41}、dstType 为 40/41 时 K 不可被 4 整除或 scaleAlg 非 0、FP8 输出时 roundMode 非 "rint"）。</td>
    </tr>
    <tr>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_ALLOC</td>
      <td>561001</td>
      <td>内存分配失败。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>Tiling 或内部计算错误。</td>
    </tr>
  </tbody></table>

## aclnnClaGateQuant

- **参数说明：**
  
  <table><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnClaGateQuantGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody></table>
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnClaGateQuant 默认确定性实现。
- 输入 globalAttn / localAttn 必须为 3 维张量且 shape 一致，为 [T, N, D]；N ∈ [1,128]，D ∈ {128, 256}。
- 输入 globalGateLogits / localGateLogits 必须为 2 维张量，为 [T, N]，数据类型与 globalAttn / localAttn 一致。
- 当 dstType 为 FLOAT4_E2M1/FLOAT4_E1M2 时，K=N*D 必须可被 4 整除。
- FP8 输出类型（FLOAT8_E5M2/FLOAT8_E4M3FN）仅支持 “rint” 舍入模式。
- 量化 block 固定为 32：row-wise 为 [1,32] block（沿 K 方向）、col-wise 为 [32,1] block（沿 T 方向）；
- 关于 rowScaleOut、colScaleOut 的 shape 约束说明：
  - rowScaleOut.shape[1] = ceil(ceil(N*D/32) / 2) = ceil(N*D/64)。
  - rowScaleOut.shape[2] = 2。
  - colScaleOut.shape[0] = ceil(ceil(T/32) / 2) = ceil(T/64)。
  - colScaleOut.shape[2] = 2。
  - 两者均需偶数 pad，pad 填充值为 0。
- dualAxisFlag=0（单轴模式）时 colDataOut / colScaleOut 必须传空指针，接口不校验、不访问 col 输出；rowDataOut / rowScaleOut 与双轴模式（dualAxisFlag=1）输出逐比特一致。
- 默认支持饱和模式。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cla_gate_mx_quant_with_dual_axis.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnClaGateQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据 API 的接口自定义构造
    // 输入 globalAttn / localAttn shape: [T, N, D] = [256, 64, 128]，K = N*D = 8192
    std::vector<int64_t> outShape = {256, 64, 128};
    // 输入 gate logits shape: [T, N] = [256, 64]
    std::vector<int64_t> logitsShape = {256, 64};
    // 量化数据输出 shape: [T, K] = [256, 8192]
    std::vector<int64_t> dataOutShape = {256, 8192};
    // rowScaleOut shape: [T, ceil(K/64), 2] = [256, 128, 2]
    std::vector<int64_t> rowScaleOutShape = {256, 128, 2};
    // colScaleOut shape: [ceil(T/64), K, 2] = [4, 8192, 2]
    std::vector<int64_t> colScaleOutShape = {4, 8192, 2};

    void* globalAttnDeviceAddr = nullptr;
    void* localAttnDeviceAddr = nullptr;
    void* globalGateLogitsDeviceAddr = nullptr;
    void* localGateLogitsDeviceAddr = nullptr;
    void* rowDataOutDeviceAddr = nullptr;
    void* rowScaleOutDeviceAddr = nullptr;
    void* colDataOutDeviceAddr = nullptr;
    void* colScaleOutDeviceAddr = nullptr;

    aclTensor* globalAttn = nullptr;
    aclTensor* localAttn = nullptr;
    aclTensor* globalGateLogits = nullptr;
    aclTensor* localGateLogits = nullptr;
    aclTensor* rowDataOut = nullptr;
    aclTensor* rowScaleOut = nullptr;
    aclTensor* colDataOut = nullptr;
    aclTensor* colScaleOut = nullptr;

    // 输入数据初始化（BF16）
    std::vector<uint16_t> outHostData(256 * 64 * 128, 0);
    for (int64_t i = 0; i < 256 * 64 * 128; i++) {
        outHostData[i] = static_cast<uint16_t>(i % 100);
    }
    std::vector<uint16_t> logitsHostData(256 * 64, 0);
    for (int64_t i = 0; i < 256 * 64; i++) {
        logitsHostData[i] = static_cast<uint16_t>(i % 50);
    }
    std::vector<uint8_t> rowDataOutHostData(256 * 8192, 0);
    std::vector<uint8_t> rowScaleOutHostData(256 * 128 * 2, 0);
    std::vector<uint8_t> colDataOutHostData(256 * 8192, 0);
    std::vector<uint8_t> colScaleOutHostData(4 * 8192 * 2, 0);

    // 参数设置
    char* roundMode = const_cast<char*>("rint");
    int64_t scaleAlg = 1;        // cuBLAS
    int64_t dstType = 36;        // FLOAT8_E4M3FN
    char* layout = const_cast<char*>("TND");          // TND
    int64_t dualAxisFlag = 1;       // 双轴量化

    // 创建 globalAttn aclTensor
    ret = CreateAclTensor(outHostData, outShape, &globalAttnDeviceAddr, aclDataType::ACL_BF16, &globalAttn);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> globalAttnTensorPtr(globalAttn, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> globalAttnDeviceAddrPtr(globalAttnDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 localAttn aclTensor
    ret = CreateAclTensor(outHostData, outShape, &localAttnDeviceAddr, aclDataType::ACL_BF16, &localAttn);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> localAttnTensorPtr(localAttn, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> localAttnDeviceAddrPtr(localAttnDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 globalGateLogits aclTensor
    ret = CreateAclTensor(logitsHostData, logitsShape, &globalGateLogitsDeviceAddr, aclDataType::ACL_BF16, &globalGateLogits);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> globalGateLogitsTensorPtr(globalGateLogits, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> globalGateLogitsDeviceAddrPtr(globalGateLogitsDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 localGateLogits aclTensor
    ret = CreateAclTensor(logitsHostData, logitsShape, &localGateLogitsDeviceAddr, aclDataType::ACL_BF16, &localGateLogits);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> localGateLogitsTensorPtr(localGateLogits, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> localGateLogitsDeviceAddrPtr(localGateLogitsDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 rowDataOut aclTensor
    ret = CreateAclTensor(rowDataOutHostData, dataOutShape, &rowDataOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &rowDataOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rowDataOutTensorPtr(rowDataOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rowDataOutDeviceAddrPtr(rowDataOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 rowScaleOut aclTensor
    ret = CreateAclTensor(rowScaleOutHostData, rowScaleOutShape, &rowScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &rowScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rowScaleOutTensorPtr(rowScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rowScaleOutDeviceAddrPtr(rowScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 colDataOut aclTensor
    ret = CreateAclTensor(colDataOutHostData, dataOutShape, &colDataOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &colDataOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> colDataOutTensorPtr(colDataOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> colDataOutDeviceAddrPtr(colDataOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 colScaleOut aclTensor
    ret = CreateAclTensor(colScaleOutHostData, colScaleOutShape, &colScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &colScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> colScaleOutTensorPtr(colScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> colScaleOutDeviceAddrPtr(colScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用 aclnnClaGateQuant 第一段接口（双轴量化）
    ret = aclnnClaGateQuantGetWorkspaceSize(globalAttn, localAttn, globalGateLogits, localGateLogits,
        roundMode, scaleAlg, dstType, layout, dualAxisFlag,
        rowDataOut, rowScaleOut, colDataOut, colScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用 aclnnClaGateQuant 第二段接口
    ret = aclnnClaGateQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateQuant failed. ERROR: %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧
    auto size1 = GetShapeSize(dataOutShape);
    auto size2 = GetShapeSize(dataOutShape);
    std::vector<uint8_t> rowDataOutData(size1, 0);
    std::vector<uint8_t> colDataOutData(size2, 0);

    ret = aclrtMemcpy(rowDataOutData.data(), rowDataOutData.size() * sizeof(uint8_t), rowDataOutDeviceAddr,
                      size1 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rowDataOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(colDataOutData.data(), colDataOutData.size() * sizeof(uint8_t), colDataOutDeviceAddr,
                      size2 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy colDataOut from device to host failed. ERROR: %d\n", ret);
              return ret);

    // 打印部分输出结果
    LOG_PRINT("rowDataOut first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size1; i++) {
        LOG_PRINT("rowDataOut[%ld] = %d\n", i, rowDataOutData[i]);
    }
    LOG_PRINT("colDataOut first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size2; i++) {
        LOG_PRINT("colDataOut[%ld] = %d\n", i, colDataOutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream 初始化，参考 acl API 手册
    // 根据自己的实际 device 填写 deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnClaGateQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```

