# aclnnFlashAttentionVarLenScoreV3

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/flash_attention_score)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |


## 功能说明

- 接口功能：训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。跟[aclnnFlashAttentionVarLenScoreV2](./aclnnFlashAttentionVarLenScoreV2.md)的区别是该接口支持query/key多输入，即query、queryRope、key和keyRope作为输入。非多输入场景使用[aclnnFlashAttentionVarLenScoreV2](./aclnnFlashAttentionVarLenScoreV2.md)或其他接口。
- 计算公式：

   注意力的正向计算公式如下：

     $$
     attention\_out=Dropout(Softmax(Mask(scale*(query*key^T + queryRope*keyRope^T) + pse),atten\_mask),keep\_prob)*value
     $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionVarLenScoreV3”接口执行计算。

```c++
aclnnStatus aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *queryRope,
  const aclTensor   *key,
  const aclTensor   *keyRope,
  const aclTensor   *value,
  const aclTensor   *realShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *prefixOptional,
  const aclIntArray *actualSeqQLenOptional,
  const aclIntArray *actualSeqKvLenOptional,
  const aclIntArray *qStartIdxOptional,
  const aclIntArray *kvStartIdxOptional,
  double             scaleValue,
  double             keepProb,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            innerPrecise,
  int64_t            sparseMode,
  int64_t            pseType,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *softmaxOutOut,
  const aclTensor   *attentionOutOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```
```c++

aclnnStatus aclnnFlashAttentionVarLenScoreV3(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```


## aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 213px">
    <col style="width: 115px">
    <col style="width: 169px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>输入</td>
        <td>公式中的query。</td>
        <td>数据类型与key/value的数据类型一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryRope</td>
        <td>输入</td>
        <td>公式中的queryRope。</td>
        <td>数据类型与key/value的数据类型一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的key。</td>
        <td>数据类型与query/value的数据类型一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyRope</td>
        <td>输入</td>
        <td>公式中的keyRope。</td>
        <td>数据类型与query/value的数据类型一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的value。</td>
        <td>数据类型与query/key的数据类型一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>输入</td>
        <td>公式中的pse。</td>
        <td>必须为nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>输入</td>
        <td>公式中的Dropout。</td>
        <td>必须为nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>公式中的atten_mask。</td>
        <td>取值为1代表该位不参与计算，为0代表该位参与计算。</td>
        <td>BOOL、UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv]、[B,1,Sq,Skv]、[1,1,Sq,Skv]、[Sq,Skv] </td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>输入</td>
        <td>代表prefix稀疏计算场景每个Batch的N值。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>输入</td>
        <td>描述了每个Batch对应的query的sequence length。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>输入</td>
        <td>描述了每个Batch对应的key/value的sequence length。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中的scale，代表缩放系数。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>输入</td>
        <td>代表dropMaskOptional中1的比例。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>用于稀疏计算 ，表示slides window的左边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>用于稀疏计算，表示slides window的右边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>输入</td>
        <td>代表单卡的head个数，即输入query的N轴长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>代表输入query、key、value的数据排布格式。</td>
        <td>支持TND。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>用于提升精度。</td>
        <td>暂未使用。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>表示sparse的模式。</td>
        <td>支持配置值为0、1、2、3、4、7、8。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>输入</td>
        <td>控制mul与add计算顺序，仅支持配置值为1。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>输出</td>
        <td>Softmax计算的Max中间结果，用于反向计算。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[N,T,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>输出</td>
        <td>Softmax计算的Sum中间结果，用于反向计算。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[N,T,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>输出</td>
        <td>计算公式的最终输出。</td>
        <td>数据类型和shape类型与query保持一致。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>  
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>query、queryRope、key、keyRope、value、realShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOut、softmaxSumOut、softmaxOutOut、attentionOutOut的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>query、queryRope、key、keyRope、value、realShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOut、softmaxSumOut、softmaxOutOut、attentionOutOut的数据格式不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionVarLenScoreV3

-   **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
    <col style="width: 153px">
    <col style="width: 121px">
    <col style="width: 880px">
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
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
      </tr>
    </tbody>
    </table>

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnFlashAttentionVarLenScoreV3默认确定性实现。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配
- 输入query、queryRope、key、keyRope、value的B：batchsize必须相等。
- 输入query、key、value的D：Head-Dim必须满足(qD == kD && kD >= vD)，D必须是8的整数倍。
- 输入queryRope、keyRope的D：Head-Dim必须满足(qRopeD == kRopeD)，D必须是8的整数倍，且必须小于等于query、key和value的D。
- 输入query、key、value的inputLayout必须为TND。
- 关于数据shape的约束，其中：
    -   T：取值范围为1\~1M。
    -   N：取值范围为1\~256。
    -   D：取值范围为1\~768。
    -   数据shape必须为TND。
    -   KeepProb必须为1。  
- query、key、value数据排布格式仅支持TND，T是B和S合轴紧密排列的数据（每个batch的SeqLenQ和SeqLenKV），其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- sparseMode的约束如下: 
  - 当所有的attenMaskOptional的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量；
  - 配置为1、2、3、5时，用户配置的preTokens、nextTokens不会生效；
  - 配置为0、4时，须保证attenMaskOptional与preTokens、nextTokens的范围一致。
  - 用户不特意指定时建议传入0。
  - sparse不同模式的详细说明请参见[sparse模式说明](../../../docs/zh/context/sparse_mode参数说明.md)。
  - 配置为3时，不支持无效行计算，需要满足每个batch的Sq<=Skv。
  - 配置为7时，不支持可选输入realShiftOptional。
  - 配置为8时，当每个sequence的q、kv等长时支持可选输入realShiftOptional，针对全局做pse生成。支持q方向进行外切，需要外切前每个sequence的q、kv等长，外切后传入的actualSeqQLenOptional。
- 部分场景下，如果计算量过大可能会导致算子执行超时(aicore error类型报错，errorStr为：timeout or trap error)，此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- band场景，preTokens和nextTokens之间必须要有交集。
- prefixOptional稀疏计算场景即sparseMode=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。
[0] - actualSeqKvLenOptional[0] + qStartIdxOptional - kvStartIdxOptional == 0（本功能属实验性功能）。
- actualSeqQLenOptional输入支持某个Batch上的S长度为0，此时不支持可选输入realShiftOptional。actualSeqQLenOptional的长度取值范围为1\~2K。当存在prefixOptional输入的时候，其长度最大支持1K。
- attenMaskOptional输入不支持补pad，即attenMaskOptional中不能存在某一行全1的场景。
- 支持actualSeqQLenOptional中某个Batch上的S长度为0；如果存在S为0的情况，不支持pse输入，
  假设真实的S长度为\[2,2,0,2,2\]，则传入的actualSeqQLenOptional为\[2,4,4,6,8\]。
- pseType只能为0或者1。
- realShiftOptional必须为空。
- dropMaskOptional必须为空。
- attenMaskOptional不能为空。


## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> qRopeShape = {256, 1, 64};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> kRopeShape = {256, 1, 64};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};

  std::vector<int64_t> attentionOutShape = {256, 1, 128};
  std::vector<int64_t> softmaxMaxShape = {256, 1, 8};
  std::vector<int64_t> softmaxSumShape = {256, 1, 8};

  void* qDeviceAddr = nullptr;
  void* qRopeDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* qRope = nullptr;
  aclTensor* k = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* v = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> qRopeHostData(16384, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> kRopeHostData(16384, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> attentionOutHostData(32768, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_BF16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qRopeHostData, qRopeShape, &qRopeDeviceAddr, aclDataType::ACL_BF16, &qRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_BF16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_BF16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_BF16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_BF16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  std::vector<int64_t>  acSeqQLenOp = {256};
  std::vector<int64_t>  acSeqKvLenOp = {256};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;

  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFlashAttentionVarLenScoreV3第一段接口
  ret = aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(
            q, qRope, k, kRope, v, pse, dropMask, padding, attenmask, prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
            scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
            sparseMode, pseType, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFlashAttentionVarLenScoreV3第二段接口
  ret = aclnnFlashAttentionVarLenScoreV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionVarLenScoreV3 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(qRope);
  aclDestroyTensor(k);
  aclDestroyTensor(kRope);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(qRopeDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```