# aclnnMatmulFt

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

依赖catlass master 分支commit 81014c2590d0e6e6f73d2acd78e36975b6acad70

## 功能说明

- 接口功能：实现基于方差估计自适应门限算法（V-ABFT）的GEMM容错检测算子。在矩阵乘法完成后，对输出矩阵C进行分块ABFT校验，检测并定位静默计算错误，输出逐行的故障检测结果。

- 计算公式：

  $$
    C = A \times B, \quad A \in \mathbb{R}^{M \times K},\; B \in \mathbb{R}^{K \times N}
  $$

  利用ABFT校验和编码，计算C的列校验和与行校验和，通过与编码后的期望校验和比较，基于统计方差估计的自适应阈值门限进行逐行错误检测：

  - 行校验和检测：$|C^{r_1}_i - C^{r_1'}_i| \le Threshold_i$ 时判定第i行无错误，否则该行存在计算错误。
  - 列校验和检测：$|C^{c_1}_j - C^{c_1'}_j| \le Threshold_j$ 时判定第j列无错误，否则该列存在计算错误。

  其中阈值$Threshold_i$由输入矩阵A、B的局部统计特征（均值、标准差）动态估计，无需依赖C矩阵输出结果。

- 算子功能说明：

  - 输入矩阵A、B经过容错算子内部的checksum编码、阈值估计和校验比对流程，输出：
    - 逐行的checksum误差（`z_row` / `z_col`）和定位分母（`d_row` / `d_col`），供定位错误位置。
    - 压缩后的逐行/逐列故障检测结果位流（`comp_row` / `comp_col`），1表示正确，0表示检测到错误。
    - 原始矩阵乘法结果C（`c`）。
    - 动态阈值（`threshold`）及A、B矩阵统计信息（均值、最大值、最小值等），供外部调参或诊断。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMatmulFtGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnMatmulFt”接口执行计算。

```cpp
aclnnStatus aclnnMatmulFtGetWorkspaceSize(
    const aclTensor  *a,
    const aclTensor  *b,
    const aclTensor  *x,
    const aclTensor  *xv,
    const aclTensor  *vxForAe,
    double            eMax,
    int64_t           reduceCores,
    const aclTensor  *zRow,
    const aclTensor  *zCol,
    const aclTensor  *dRow,
    const aclTensor  *dCol,
    const aclTensor  *compRow,
    const aclTensor  *compCol,
    const aclTensor  *c,
    const aclTensor  *threshold,
    const aclTensor  *bMeanAbs,
    const aclTensor  *bMeanSquare,
    const aclTensor  *bVar,
    const aclTensor  *be,
    const aclTensor  *beForAiv,
    const aclTensor  *bMaxSlice,
    const aclTensor  *bMinSlice,
    const aclTensor  *aMax,
    const aclTensor  *aMean,
    const aclTensor  *aMin,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMatmulFt(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMatmulFtGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed;width: 1540px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 118px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th><a href="../../../docs/zh/context/数据格式.md" target="_blank">数据格式</a></th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th><a href="../../../docs/zh/context/非连续的Tensor.md" target="_blank">非连续的Tensor</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>a（aclTensor）</td>
        <td>输入</td>
        <td>矩阵乘法输入A。</td>
        <td>
          <ul>
            <li>维度为2，shape为[M, K]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[M, K]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>b（aclTensor）</td>
        <td>输入</td>
        <td>矩阵乘法输入B。</td>
        <td>
          <ul>
            <li>维度为2，shape为[K, N]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[K, N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x（aclTensor）</td>
        <td>输入</td>
        <td>列编码输入向量，对应ABFT中的行校验和编码向量$r_1$（全1向量）。</td>
        <td>
          <ul>
            <li>维度为1，shape为[N]。</li>
          </ul>
        </td>
        <td>FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>xv（aclTensor）</td>
        <td>输入</td>
        <td>列编码值向量，对应ABFT中的行校验和编码向量$r_2$（加权向量）。</td>
        <td>
          <ul>
            <li>维度为1，shape为[N]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>vxForAe（aclTensor）</td>
        <td>输入</td>
        <td>A矩阵误差估计向量，用于A矩阵每行的checksum编码及统计量估计。</td>
        <td>
          <ul>
            <li>维度为1，shape为[K]。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[K]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eMax（double）</td>
        <td>输入</td>
        <td>误差阈值系数，控制故障检测的灵敏度。</td>
        <td>
          <ul>
            <li>默认值为0.001。</li>
            <li>增大eMax会放宽检测门限，减少误检率但可能漏检。</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>reduceCores（int64_t）</td>
        <td>输入</td>
        <td>规约计算使用的核心数。</td>
        <td>
          <ul>
            <li>默认值为8。</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>zRow（aclTensor）</td>
        <td>输出</td>
        <td>行校验和误差分子$C^{r_2'} - C^{r_2}$，用于定位错误行索引。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>zCol（aclTensor）</td>
        <td>输出</td>
        <td>列校验和误差分子$C^{c_2'} - C^{c_2}$，用于定位错误列索引。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dRow（aclTensor）</td>
        <td>输出</td>
        <td>行校验和误差分母$C^{r_1'} - C^{r_1}$，与zRow联用定位错误列索引。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dCol（aclTensor）</td>
        <td>输出</td>
        <td>列校验和误差分母$C^{c_1'} - C^{c_1}$，与zCol联用定位错误行索引。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compRow（aclTensor）</td>
        <td>输出</td>
        <td>压缩后的行方向故障检测位流输出。</td>
        <td>
          <ul>
            <li>每个bit表示一行一段的检测结果，1表示正确，0表示检测到错误。</li>
          </ul>
        </td>
        <td>UINT8</td>
        <td>ND</td>
        <td>[ceil(M/8) * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compCol（aclTensor）</td>
        <td>输出</td>
        <td>压缩后的列方向故障检测位流输出。</td>
        <td>
          <ul>
            <li>每个bit表示一列一段的检测结果，1表示正确，0表示检测到错误。</li>
          </ul>
        </td>
        <td>UINT8</td>
        <td>ND</td>
        <td>[ceil(N/8)]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>c（aclTensor）</td>
        <td>输出</td>
        <td>矩阵乘法结果C = A x B。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M, N]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>threshold（aclTensor）</td>
        <td>输出</td>
        <td>逐行的自适应检测阈值，基于A、B矩阵统计特征动态估计。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bMeanAbs（aclTensor）</td>
        <td>输出</td>
        <td>B矩阵分块均值绝对值。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[ceil(splitN/8)*8 + 8]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bMeanSquare（aclTensor）</td>
        <td>输出</td>
        <td>B矩阵分块均值平方。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[ceil(splitN/8)*8 + 8]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bVar（aclTensor）</td>
        <td>输出</td>
        <td>B矩阵分块方差。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[ceil(splitN/8)*8 + 8]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>be（aclTensor）</td>
        <td>输出</td>
        <td>B矩阵编码输出$B^r$，即B矩阵的ABFT行校验和编码结果。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[K * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beForAiv（aclTensor）</td>
        <td>输出</td>
        <td>供AIV Core使用的B矩阵编码输出。</td>
        <td>-</td>
        <td>FLOAT16（输入为FLOAT16时）、FLOAT32（其他情况下）</td>
        <td>ND</td>
        <td>[K * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bMaxSlice（aclTensor）</td>
        <td>输出</td>
        <td>B矩阵每分块内的最大值。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[K * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bMinSlice（aclTensor）</td>
        <td>输出</td>
        <td>B矩阵每分块内的最小值。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[K * splitN]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>aMax（aclTensor）</td>
        <td>输出</td>
        <td>A矩阵每行的最大值。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>aMean（aclTensor）</td>
        <td>输出</td>
        <td>A矩阵每行的均值。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>aMin（aclTensor）</td>
        <td>输出</td>
        <td>A矩阵每行的最小值。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[M]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  其中 $splitN = \lceil N / 256\rceil$。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景时报错:

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选输入、输出或者必选属性是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>a、b、x、xv、vxForAe、eMax、reduceCores、zRow、zCol、dRow、dCol、compRow、compCol、c、threshold、bMeanAbs、bMeanSquare、bVar、be、beForAiv、bMaxSlice、bMinSlice、aMax、aMean、aMin的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>a和b的维度不为2。</td>
    </tr>
    <tr>
      <td>a的第1维（K）与b的第0维（K）不相等。</td>
    </tr>
    <tr>
      <td>eMax为负数。</td>
    </tr>
    <tr>
      <td>reduceCores小于0。</td>
    </tr>
  </tbody></table>

## aclnnMatmulFt

- **参数说明：**
    <table>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMatmulFtGetWorkspaceSize获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMatmulFt默认确定性实现。
- 输入矩阵a和b必须为2维，shape分别为[M, K]和[K, N]，且a的第1维（K）与b的第0维（K）必须相等。
- 输入向量x和xv的shape必须为[N]，vxForAe的shape必须为[K]。
- 支持的数据类型组合为：

  | a       | b       | x       | xv      | vxForAe |
  |:-------:|:-------:|:-------:|:-------:|:-------:|
  | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
  | BFLOAT16| BFLOAT16| FLOAT32 | BFLOAT16| BFLOAT16|
  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |

- 不支持的场景：
  - 不支持ND格式以外的数据格式。
  - 不支持非连续的Tensor。
  - 不支持a和b的维度大于2的场景。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_ft.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
        printf(message, ##__VA_ARGS__); \
    } while (0)

using Bf16 = uint16_t;

Bf16 FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<Bf16>(bits >> 16U);
}

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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1.（固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置矩阵参数
    int64_t M = 4096;
    int64_t N = 4096;
    int64_t K = 4096;
    double eMax = 0.001;
    int64_t reduceCores = 8;

    int64_t splitN = (N + 256 - 1) / 256;
    int64_t bStatBlock = (splitN + 8 - 1) / 8;
    int64_t bStatLen = bStatBlock * 8 + 8;
    int64_t rowSplitLen = M * splitN;
    int64_t beLen = K * splitN;

    // 输入shape
    std::vector<int64_t> aShape = {M, K};
    std::vector<int64_t> bShape = {K, N};
    std::vector<int64_t> xShape = {N};
    std::vector<int64_t> xvShape = {N};
    std::vector<int64_t> vxForAeShape = {K};

    // 输出shape
    std::vector<int64_t> zRowColShape = {rowSplitLen};
    std::vector<int64_t> compRowShape = {((M + 7) / 8) * splitN};
    std::vector<int64_t> compColShape = {(N + 7) / 8};
    std::vector<int64_t> cShape = {M, N};
    std::vector<int64_t> bStatShape = {bStatLen};
    std::vector<int64_t> beShape = {beLen};
    std::vector<int64_t> aRedShape = {M};

    // 3. 构造输入数据（BF16示例）
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<Bf16> hostA(GetShapeSize(aShape), 0);
    std::vector<Bf16> hostB(GetShapeSize(bShape), 0);
    std::vector<float> hostX(GetShapeSize(xShape), 1.0F);
    std::vector<Bf16> hostXv(GetShapeSize(xvShape), 0);
    std::vector<Bf16> hostVxForAe(GetShapeSize(vxForAeShape), 0);

    for (auto& v : hostA) {
        v = FloatToBf16(dist(rng));
    }
    for (auto& v : hostB) {
        v = FloatToBf16(dist(rng));
    }
    for (auto& v : hostXv) {
        v = FloatToBf16(1.0F);
    }
    for (auto& v : hostVxForAe) {
        v = FloatToBf16(1.0F / static_cast<float>(K));
    }

    // 4. 构造输出数据（零初始化）
    std::vector<float> hostZRow(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostZCol(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostDRow(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostDCol(GetShapeSize(zRowColShape), 0.0F);
    std::vector<uint8_t> hostCompRow(GetShapeSize(compRowShape), 0);
    std::vector<uint8_t> hostCompCol(GetShapeSize(compColShape), 0);
    std::vector<float> hostC(GetShapeSize(cShape), 0.0F);
    std::vector<float> hostThreshold(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostBMeanAbs(GetShapeSize(bStatShape), 0.0F);
    std::vector<float> hostBMeanSquare(GetShapeSize(bStatShape), 0.0F);
    std::vector<float> hostBVar(GetShapeSize(bStatShape), 0.0F);
    std::vector<Bf16> hostBe(GetShapeSize(beShape), 0);
    std::vector<float> hostBeForAiv(GetShapeSize(beShape), 0.0F);
    std::vector<float> hostBMaxSlice(GetShapeSize(beShape), 0.0F);
    std::vector<float> hostBMinSlice(GetShapeSize(beShape), 0.0F);
    std::vector<float> hostAMax(GetShapeSize(aRedShape), 0.0F);
    std::vector<float> hostAMean(GetShapeSize(aRedShape), 0.0F);
    std::vector<float> hostAMin(GetShapeSize(aRedShape), 0.0F);

    // 5. 创建Device侧Tensor
    void *devA = nullptr, *devB = nullptr, *devX = nullptr, *devXv = nullptr, *devVxForAe = nullptr;
    void *devZRow = nullptr, *devZCol = nullptr, *devDRow = nullptr, *devDCol = nullptr;
    void *devCompRow = nullptr, *devCompCol = nullptr, *devC = nullptr;
    void *devThreshold = nullptr;
    void *devBMeanAbs = nullptr, *devBMeanSquare = nullptr, *devBVar = nullptr;
    void *devBe = nullptr, *devBeForAiv = nullptr;
    void *devBMaxSlice = nullptr, *devBMinSlice = nullptr;
    void *devAMax = nullptr, *devAMean = nullptr, *devAMin = nullptr;

    aclTensor *tA = nullptr, *tB = nullptr, *tX = nullptr, *tXv = nullptr, *tVxForAe = nullptr;
    aclTensor *tZRow = nullptr, *tZCol = nullptr, *tDRow = nullptr, *tDCol = nullptr;
    aclTensor *tCompRow = nullptr, *tCompCol = nullptr, *tC = nullptr;
    aclTensor *tThreshold = nullptr;
    aclTensor *tBMeanAbs = nullptr, *tBMeanSquare = nullptr, *tBVar = nullptr;
    aclTensor *tBe = nullptr, *tBeForAiv = nullptr;
    aclTensor *tBMaxSlice = nullptr, *tBMinSlice = nullptr;
    aclTensor *tAMax = nullptr, *tAMean = nullptr, *tAMin = nullptr;

    CHECK_RET(CreateAclTensor(hostA, aShape, &devA, ACL_BF16, &tA) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostB, bShape, &devB, ACL_BF16, &tB) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostX, xShape, &devX, ACL_FLOAT, &tX) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostXv, xvShape, &devXv, ACL_BF16, &tXv) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostVxForAe, vxForAeShape, &devVxForAe, ACL_BF16, &tVxForAe) == ACL_SUCCESS, return ret);

    CHECK_RET(CreateAclTensor(hostZRow, zRowColShape, &devZRow, ACL_FLOAT, &tZRow) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostZCol, zRowColShape, &devZCol, ACL_FLOAT, &tZCol) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostDRow, zRowColShape, &devDRow, ACL_FLOAT, &tDRow) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostDCol, zRowColShape, &devDCol, ACL_FLOAT, &tDCol) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostCompRow, compRowShape, &devCompRow, ACL_UINT8, &tCompRow) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostCompCol, compColShape, &devCompCol, ACL_UINT8, &tCompCol) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostC, cShape, &devC, ACL_FLOAT, &tC) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostThreshold, zRowColShape, &devThreshold, ACL_FLOAT, &tThreshold) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMeanAbs, bStatShape, &devBMeanAbs, ACL_FLOAT, &tBMeanAbs) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMeanSquare, bStatShape, &devBMeanSquare, ACL_FLOAT, &tBMeanSquare) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBVar, bStatShape, &devBVar, ACL_FLOAT, &tBVar) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBe, beShape, &devBe, ACL_BF16, &tBe) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBeForAiv, beShape, &devBeForAiv, ACL_FLOAT, &tBeForAiv) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMaxSlice, beShape, &devBMaxSlice, ACL_FLOAT, &tBMaxSlice) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMinSlice, beShape, &devBMinSlice, ACL_FLOAT, &tBMinSlice) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostAMax, aRedShape, &devAMax, ACL_FLOAT, &tAMax) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostAMean, aRedShape, &devAMean, ACL_FLOAT, &tAMean) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostAMin, aRedShape, &devAMin, ACL_FLOAT, &tAMin) == ACL_SUCCESS, return ret);

    // 6. 调用专用容错矩阵乘法接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnMatmulFtGetWorkspaceSize(
        tA, tB, tX, tXv, tVxForAe,
        eMax, reduceCores,
        tZRow, tZCol, tDRow, tDCol,
        tCompRow, tCompCol, tC,
        tThreshold,
        tBMeanAbs, tBMeanSquare, tBVar,
        tBe, tBeForAiv,
        tBMaxSlice, tBMinSlice,
        tAMax, tAMean, tAMin,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulFtGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMatmulFt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulFt failed. ERROR: %d\n", ret); return ret);

    // 7.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 8. 获取输出的值（以compRow为例，检查是否所有bit位均为1表示无错误）
    {
        auto size = GetShapeSize(compRowShape);
        std::vector<uint8_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(uint8_t), devCompRow,
                          size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
        bool allCorrect = true;
        for (int64_t i = 0; i < size; i++) {
            if (resultData[i] != 0xFFU) {
                allCorrect = false;
                break;
            }
        }
        LOG_PRINT("Fault detection result: %s\n", allCorrect ? "ALL CORRECT" : "ERROR DETECTED");
    }

    // 9. 释放资源
    aclDestroyTensor(tA); aclDestroyTensor(tB); aclDestroyTensor(tX); aclDestroyTensor(tXv); aclDestroyTensor(tVxForAe);
    aclDestroyTensor(tZRow); aclDestroyTensor(tZCol); aclDestroyTensor(tDRow); aclDestroyTensor(tDCol);
    aclDestroyTensor(tCompRow); aclDestroyTensor(tCompCol); aclDestroyTensor(tC);
    aclDestroyTensor(tThreshold);
    aclDestroyTensor(tBMeanAbs); aclDestroyTensor(tBMeanSquare); aclDestroyTensor(tBVar);
    aclDestroyTensor(tBe); aclDestroyTensor(tBeForAiv);
    aclDestroyTensor(tBMaxSlice); aclDestroyTensor(tBMinSlice);
    aclDestroyTensor(tAMax); aclDestroyTensor(tAMean); aclDestroyTensor(tAMin);

    aclrtFree(devA); aclrtFree(devB); aclrtFree(devX); aclrtFree(devXv); aclrtFree(devVxForAe);
    aclrtFree(devZRow); aclrtFree(devZCol); aclrtFree(devDRow); aclrtFree(devDCol);
    aclrtFree(devCompRow); aclrtFree(devCompCol); aclrtFree(devC);
    aclrtFree(devThreshold);
    aclrtFree(devBMeanAbs); aclrtFree(devBMeanSquare); aclrtFree(devBVar);
    aclrtFree(devBe); aclrtFree(devBeForAiv);
    aclrtFree(devBMaxSlice); aclrtFree(devBMinSlice);
    aclrtFree(devAMax); aclrtFree(devAMean); aclrtFree(devAMin);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyAclOpExecutor(executor);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
