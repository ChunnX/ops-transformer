# aclnnChunkKdaFwd

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- 接口功能：完成不涉及CP切分的KDA分块正向计算，输出注意力结果、最终状态和反向计算所需的中间量。

- 计算公式：

  将每条序列按$C=chunkSize$划分为$M$个chunk。以第$c$个chunk为例，$i$、$j$表示chunk内token下标，$l_c$表示该chunk最后一个有效token下标。以下公式省略batch和head下标；GQA场景中，每个Value head使用对应的Query/Key head。

  令$x_{c,i,d}=g_{c,i,d}+dtBias_d$，未传入`dtBiasOptional`时令$dtBias_d=0$。激活后的gate为：

  $$
  \gamma_{c,i,d}=
  \begin{cases}
  g_{c,i,d}, & useGateInKernel=false,\\
  -\exp(A_{log})\operatorname{softplus}(x_{c,i,d}),
      & useGateInKernel=true,\ safeGate=false,\\
  lowerBound\operatorname{sigmoid}(\exp(A_{log})x_{c,i,d}),
      & useGateInKernel=true,\ safeGate=true.
  \end{cases}
  $$

  `gkOut`为chunk内以2为底的累计gate：

  $$
  gk_{c,i,d}=\frac{1}{\ln 2}\sum_{t=0}^{i}\gamma_{c,t,d}.
  $$

  `qgOut`和`kgOut`分别为gate缩放后的Query和Key：

  $$
  qg_{c,i,d}=q_{c,i,d}2^{gk_{c,i,d}},\qquad
  kg_{c,i,d}=k_{c,i,d}2^{gk_{c,l_c,d}-gk_{c,i,d}}.
  $$

  `aqkOut`为包含对角线的下三角Query-Key系数矩阵：

  $$
  Aqk_{c,i,j}=\mathbb{1}_{j\le i}\cdot scale
  \sum_d q_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}}.
  $$

  定义严格下三角矩阵$L_c$，`akkOut`为$I+L_c$的逆矩阵：

  $$
  L_{c,i,j}=\mathbb{1}_{j<i}\cdot\beta_{c,i}
  \sum_d k_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\qquad
  Akk_c=(I+L_c)^{-1}.
  $$

  定义$w^{seed}$和$u^{seed}$：

  $$
  w^{seed}_{c,i,d}=\beta_{c,i}k_{c,i,d}2^{gk_{c,i,d}},\qquad
  u^{seed}_{c,i,e}=\beta_{c,i}v_{c,i,e}.
  $$

  `wOut`和`uOut`为：

  $$
  w_c=Akk_c\,w^{seed}_c,\qquad u_c=Akk_c\,u^{seed}_c.
  $$

  令$h_c$为第$c$个chunk计算前的状态。传入`initialStateOptional`时$h_0=initialStateOptional$，否则$h_0=0$。`vNewOut`、下一个chunk状态、`hOut`和`finalStateOut`为：

  $$
  v^{new}_c=u_c-w_c\,h_c,
  $$

  $$
  h_{c+1}=2^{gk_{c,l_c}}\odot h_c+kg_c^T\,v^{new}_c,
  $$

  $$
  hOut[c]=h_c,\qquad finalStateOut=h_M.
  $$

  其中，$2^{gk_{c,l_c}}\odot h_c$表示沿$K$维缩放状态矩阵。`attnOut`为：

  $$
  attnOut_c=scale\cdot qg_c\,h_c+Aqk_c\,v^{new}_c.
  $$

- 符号说明：

  | 符号 | 含义 |
  | --- | --- |
  | B | Batch Size。 |
  | S | 四维输入的序列长度。 |
  | T | 三维输入的总token数。 |
  | H | Query和Key的head数。 |
  | HV | Value和gate的head数。 |
  | K | Query和Key的head dim。 |
  | V | Value的head dim。 |
  | N | 逻辑序列数。 |
  | C | chunk大小。 |
  | NC | 未传入`cuSeqlensOptional`时为每条序列的chunk数$\lceil S/C\rceil$；传入时为所有逻辑序列的chunk总数。 |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnChunkKdaFwdGetWorkspaceSize`接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用`aclnnChunkKdaFwd`接口执行计算。

```cpp
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnChunkKdaFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnChunkKdaFwdGetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） | 非连续Tensor |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | q | 输入 | 公式中的Query。 | 不支持空Tensor。shape由`layout`解释。 | FLOAT16、BFLOAT16 | ND | BSND：[B,S,H,K]<br>BNSD：[B,H,S,K]<br>TND：[T,H,K]<br>NTD：[H,T,K] | √ |
  | k | 输入 | 公式中的Key。 | 不支持空Tensor。shape和数据类型必须与`q`相同。 | FLOAT16、BFLOAT16 | ND | 与`q`相同 | √ |
  | v | 输入 | 公式中的Value。 | 不支持空Tensor。数据类型必须与`q`相同。 | FLOAT16、BFLOAT16 | ND | BSND：[B,S,HV,V]<br>BNSD：[B,HV,S,V]<br>TND：[T,HV,V]<br>NTD：[HV,T,V] | √ |
  | g | 输入 | `useGateInKernel=false`时为已激活的自然对数gate；`useGateInKernel=true`时为raw gate。 | 不支持空Tensor。 | FLOAT32、BFLOAT16 | ND | BSND：[B,S,HV,K]<br>BNSD：[B,HV,S,K]<br>TND：[T,HV,K]<br>NTD：[HV,T,K] | √ |
  | beta | 输入 | 公式中的Delta系数$\beta$。 | 不支持空Tensor。 | FLOAT32、BFLOAT16 | ND | BSND：[B,S,HV]<br>BNSD：[B,HV,S]<br>TND：[T,HV]<br>NTD：[HV,T] | √ |
  | aLogOptional | 可选输入 | gate衰减参数$A_{log}$。 | `useGateInKernel=true`时必须传入；否则可以传入nullptr。 | FLOAT32 | ND | [HV] | √ |
  | dtBiasOptional | 可选输入 | gate偏置。 | 可以传入nullptr，表示偏置为0；仅在`useGateInKernel=true`时参与计算。 | FLOAT32 | ND | [HV×K] | √ |
  | initialStateOptional | 可选输入 | 每条逻辑序列的初始状态$h_0$。 | 可以传入nullptr，表示初始状态为0。末两维顺序由`stateVFirst`决定。 | FLOAT32 | ND | `stateVFirst=false`：[N,HV,K,V]<br>`stateVFirst=true`：[N,HV,V,K] | √ |
  | cuSeqlensOptional（aclIntArray*） | 可选输入 | 变长序列累计长度。 | 可以传入nullptr，表示定长序列。传入时元素单调不减，首元素为0，末元素为S或T；四维变长输入要求B=1。 | INT64 | - | [N+1] | - |
  | chunkIndicesOptional（aclIntArray*） | 可选输入 | 按`(seq_id, chunk_id)`展平的chunk索引。 | 可以传入nullptr。传入时必须同时传入`cuSeqlensOptional`，并按sequence-major canonical顺序包含每个chunk。 | INT64 | - | [2×NC] | - |
  | layout | 输入 | 输入布局。 | 支持`BSND`、`BNSD`、`TND`、`NTD`，区分大小写。该参数只描述`q`、`k`、`v`、`g`和`beta`的布局。 | STRING | - | - | - |
  | scale | 输入 | Query缩放系数。 | 通常取$K^{-0.5}$。 | DOUBLE | - | - | - |
  | chunkSize | 输入 | chunk大小。 | 仅支持64、128。 | INT64 | - | - | - |
  | safeGate | 输入 | 是否使用有界gate。 | 取值为true或false。 | BOOL | - | - | - |
  | lowerBound | 输入 | 有界gate下界。 | `safeGate=true`且`useGateInKernel=true`时，取值范围为[-5,0)。 | DOUBLE | - | - | - |
  | useGateInKernel | 输入 | 是否在算子内由raw gate计算激活后的gate。 | 取值为true或false。 | BOOL | - | - | - |
  | stateVFirst | 输入 | 状态张量是否使用(V,K)末两维顺序。 | 取值为true或false。 | BOOL | - | - | - |
  | attnOut | 输出 | 公式中的注意力输出。 | 不支持空Tensor。数据类型必须与`q`相同，固定为sequence-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,S,HV,V]<br>三维输入：[T,HV,V] | × |
  | finalStateOut | 可选输出 | 每条逻辑序列最后一个chunk更新后的状态$h_M$。 | 可以传入nullptr，表示不导出该结果。末两维顺序由`stateVFirst`决定。 | FLOAT32 | ND | `stateVFirst=false`：[N,HV,K,V]<br>`stateVFirst=true`：[N,HV,V,K] | × |
  | gkOut | 可选输出 | chunk内以2为底的累计gate。 | 可以传入nullptr，表示不导出该结果。固定为head-major布局。 | FLOAT32 | ND | 四维输入：[B,HV,S,K]<br>三维输入：[HV,T,K] | × |
  | aqkOut | 输出 | chunk内Query-Key系数矩阵$Aqk$。 | 不支持空Tensor。数据类型必须与`q`相同，固定为head-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,HV,S,C]<br>三维输入：[HV,T,C] | × |
  | akkOut | 输出 | chunk内三角求逆结果$Akk$。 | 不支持空Tensor。数据类型必须与`q`相同，固定为head-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,HV,S,C]<br>三维输入：[HV,T,C] | × |
  | wOut | 可选输出 | 公式中的$w$中间量。 | 可以传入nullptr，表示不导出该结果。传入时数据类型必须与`q`相同，固定为head-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,HV,S,K]<br>三维输入：[HV,T,K] | × |
  | uOut | 可选输出 | 公式中的$u$中间量。 | 可以传入nullptr，表示不导出该结果。传入时数据类型必须与`q`相同，固定为head-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,HV,S,V]<br>三维输入：[HV,T,V] | × |
  | qgOut | 可选输出 | gate缩放后的Query。 | 可以传入nullptr，表示不导出该结果。传入时数据类型必须与`q`相同，固定为head-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,HV,S,K]<br>三维输入：[HV,T,K] | × |
  | kgOut | 可选输出 | gate缩放后的Key。 | 可以传入nullptr，表示不导出该结果。传入时数据类型必须与`q`相同，固定为head-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,HV,S,K]<br>三维输入：[HV,T,K] | × |
  | vNewOut | 可选输出 | 状态校正后的Value。 | 可以传入nullptr，表示不导出该结果。传入时数据类型必须与`q`相同，固定为head-major布局。 | FLOAT16、BFLOAT16 | ND | 四维输入：[B,HV,S,V]<br>三维输入：[HV,T,V] | × |
  | hOut | 可选输出 | 每个chunk计算前的状态$h_c$。 | 可以传入nullptr，表示不导出该结果。传入时数据类型必须与`q`相同，固定为sequence-major布局，末两维顺序由`stateVFirst`决定。 | FLOAT16、BFLOAT16 | ND | 四维输入且`stateVFirst=false`：[B,NC,HV,K,V]<br>四维输入且`stateVFirst=true`：[B,NC,HV,V,K]<br>三维输入且`stateVFirst=false`：[NC,HV,K,V]<br>三维输入且`stateVFirst=true`：[NC,HV,V,K] | × |
  | workspaceSize | 输出 | 返回需要在Device侧申请的workspace大小。 | 单位为Byte。 | - | - | - | - |
  | executor | 输出 | 返回算子执行器。 | 包含算子计算流程。 | - | - | - | - |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回值 | 错误码 | 描述 |
  | --- | --- | --- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | `q`、`k`、`v`、`g`、`beta`、`attnOut`、`aqkOut`或`akkOut`为空指针；或者`useGateInKernel=true`时`aLogOptional`为空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | 输入或输出的shape、数据类型、数据格式、layout、chunk元数据或属性值不在支持范围内。 |

## aclnnChunkKdaFwd

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  | --- | --- | --- |
  | workspace | 输入 | 在Device侧申请的workspace内存地址。`workspaceSize`为0时可以传入nullptr。 |
  | workspaceSize | 输入 | 在Device侧申请的workspace大小，由第一段接口`aclnnChunkKdaFwdGetWorkspaceSize`获取。 |
  | executor | 输入 | op执行器，包含算子计算流程。 |
  | stream | 输入 | 指定执行任务的Stream。 |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - `aclnnChunkKdaFwd`默认确定性实现。
- 输入布局：
  - `layout`仅支持`BSND`、`BNSD`、`TND`、`NTD`。
  - `layout`只描述`q`、`k`、`v`、`g`和`beta`输入；各输出布局按参数表固定。
- 维度约束：
  - B、S、T、H、HV、K和V均必须大于0，不支持空Tensor。
  - H和HV必须满足$0<H\le HV\le128$且$HV\bmod H=0$。
  - K和V的取值范围为[16,256]，且必须为16的倍数。
  - `chunkSize`仅支持64、128。
  - 四维变长输入要求B=1。
  - 变长输入最多支持1024条逻辑序列。
- 参数组合约束：
  - `q`、`k`、`v`的数据类型必须相同。
  - `useGateInKernel=true`时必须传入`aLogOptional`，`dtBiasOptional`可以传入nullptr。
  - `safeGate=true`且`useGateInKernel=true`时，`lowerBound`的取值范围为[-5,0)。
  - 传入`chunkIndicesOptional`时必须同时传入`cuSeqlensOptional`，且chunk索引必须采用sequence-major canonical顺序。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_kda_fwd.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1. device/context/stream初始化，根据实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    int64_t batch = 1;
    int64_t seqLen = 64;
    int64_t headNum = 2;
    int64_t headDim = 128;
    int64_t chunkSize = 64;

    std::vector<int64_t> qkShape = {batch, seqLen, headNum, headDim};       // BSND
    std::vector<int64_t> vShape = {batch, seqLen, headNum, headDim};        // BSND
    std::vector<int64_t> gShape = {batch, seqLen, headNum, headDim};        // BSND
    std::vector<int64_t> betaShape = {batch, seqLen, headNum};              // BSND
    std::vector<int64_t> attnOutShape = {batch, seqLen, headNum, headDim};  // BSND
    std::vector<int64_t> aqkShape = {batch, headNum, seqLen, chunkSize};    // (B, HV, S, C)
    std::vector<int64_t> akkShape = aqkShape;

    int16_t bfloatOne = 16256;  // int16_t的16256二进制对应bfloat16的1.0
    std::vector<int16_t> qHostData(GetShapeSize(qkShape), bfloatOne);
    std::vector<int16_t> kHostData(GetShapeSize(qkShape), bfloatOne);
    std::vector<int16_t> vHostData(GetShapeSize(vShape), bfloatOne);
    std::vector<float> gHostData(GetShapeSize(gShape), 0.5f);
    std::vector<float> betaHostData(GetShapeSize(betaShape), 1.0f);
    std::vector<int16_t> attnOutHostData(GetShapeSize(attnOutShape), bfloatOne);
    std::vector<int16_t> aqkHostData(GetShapeSize(aqkShape), bfloatOne);
    std::vector<int16_t> akkHostData(GetShapeSize(akkShape), bfloatOne);

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *gDeviceAddr = nullptr;
    void *betaDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;
    void *aqkDeviceAddr = nullptr;
    void *akkDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *g = nullptr;
    aclTensor *beta = nullptr;
    aclTensor *attnOut = nullptr;
    aclTensor *aqk = nullptr;
    aclTensor *akk = nullptr;

    ret = CreateAclTensor(qHostData, qkShape, &qDeviceAddr, ACL_BF16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkShape, &kDeviceAddr, ACL_BF16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, ACL_BF16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, ACL_FLOAT, &g);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, ACL_BF16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(aqkHostData, aqkShape, &aqkDeviceAddr, ACL_BF16, &aqk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(akkHostData, akkShape, &akkDeviceAddr, ACL_BF16, &akk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnChunkKdaFwdGetWorkspaceSize(
        q, k, v, g, beta, nullptr, nullptr, nullptr, nullptr, nullptr, "BSND",
        1.0 / std::sqrt(static_cast<double>(headDim)), chunkSize, false, -5.0, false, false,
        attnOut, nullptr, nullptr, aqk, akk, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnChunkKdaFwdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnChunkKdaFwd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkKdaFwd failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("ChunkKdaFwd aclnn call succeeded.\n");

    // 5. 释放资源
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(g);
    aclDestroyTensor(beta);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(aqk);
    aclDestroyTensor(akk);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(gDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(aqkDeviceAddr);
    aclrtFree(akkDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();
    return 0;
}
```
