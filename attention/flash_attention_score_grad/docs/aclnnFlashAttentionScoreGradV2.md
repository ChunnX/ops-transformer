# aclnnFlashAttentionScoreGradV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/flash_attention_score_grad)


## 产品支持情况
|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |

## 功能说明

- 接口功能：训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV2](../../flash_attention_score/docs/aclnnFlashAttentionScoreV2.md)的反向计算。**该接口相较于[aclnnFlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad.md)接口，新增psetype参数**：

  - psetype=1时，与[aclnnFlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad.md)实现相同。
  - psetype=其他取值时，需要先mul再add。

- 计算公式：

  已知注意力的正向计算公式为（psetype=1时，与[aclnnFlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad.md)计算公式相同，psetype=其他取值公式如下）：

  $$
  Y=Dropout(Softmax(Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask),keep\_prob)V
  $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S=Mask(\frac{QK^T}{\sqrt{d}}+pse),atten\_mask
  $$

  $$
  P=Dropout(Softmax(S),keep\_prob)
  $$

  $$
  Y=PV
  $$

  则注意力的反向计算公式为：

  $$
  dV=P^TdY
  $$

  $$
  dQ=\frac{((dS)*K)}{\sqrt{d}}
  $$

  $$
  dK=\frac{((dS)^T*Q)}{\sqrt{d}}
  $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFlashAttentionScoreGradV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFlashAttentionScoreGradV2”接口执行计算。

```c++
aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *keyIn,
  const aclTensor   *value,
  const aclTensor   *dy,
  const aclTensor   *pseShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclTensor   *softmaxMaxOptional,
  const aclTensor   *softmaxSumOptional,
  const aclTensor   *softmaxInOptional,
  const aclTensor   *attentionInOptional,
  const aclIntArray *prefixOptional,
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
  const aclTensor   *dqOut,
  const aclTensor   *dkOut,
  const aclTensor   *dvOut,
  const aclTensor   *dpseOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```
```c++
aclnnStatus aclnnFlashAttentionScoreGradV2(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```


## aclnnFlashAttentionScoreGradV2GetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1529px"><colgroup>
    <col style="width: 198px">
    <col style="width: 120px">
    <col style="width: 289px">
    <col style="width: 302px">
    <col style="width: 238px">
    <col style="width: 106px">
    <col style="width: 130px">
    <col style="width: 146px">
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>输入</td>
        <td>公式中的Q。</td>
        <td>数据类型与keyIn/value一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>输入</td>
        <td>公式中的K。</td>
        <td>数据类型与query/value一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的V。</td>
        <td>数据类型与query/keyIn一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dy</td>
        <td>输入</td>
        <td>公式中的dY。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>可选输入</td>
        <td>公式中的pse。</td>
        <td>数据类型与query的数据类型一致,该参数需要与pseType配套使用。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv]、[B,N,1,Skv]、[1,N,Sq,Skv]、[B,N,1024,Skv]、[1,N,1024,Skv]、[B,N]、[N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>可选输入</td>
        <td>公式中的Dropout。</td>
        <td>-</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0、1</td>
        <td>√</td>
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
        <td>softmaxMaxOptional</td>
        <td>可选输入</td>
        <td>注意力正向计算的中间输出。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[B,N,Sq,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOptional</td>
        <td>可选输入</td>
        <td>注意力正向计算的中间输出。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[B,N,Sq,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionInOptional</td>
        <td>可选输入</td>
        <td>注意力正向的最终输出。</td>
        <td>数据类型和shape与query一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>可选输入</td>
        <td>prefix稀疏场景每个Batch的N。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>可选输入</td>
        <td>表示每个Batch的query序列长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqKvLenOptional</td>
        <td>可选输入</td>
        <td>表示每个Batch的kv序列长度。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0、1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dqOut</td>
        <td>输出</td>
        <td>公式中的dQ，query的梯度。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dkOut</td>
        <td>输出</td>
        <td>公式中的dK，keyIn的梯度。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dvOut</td>
        <td>输出</td>
        <td>公式中的dV，value的梯度。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>[BNSD]、[BSND]、[BSH]、[SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dpseOut</td>
        <td>输出</td>
        <td>d(pse)梯度。</td>
        <td>暂未使用。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中的scale缩放系数。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>输入</td>
        <td>dropMask中1的比例。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>稀疏计算窗口左边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>稀疏计算窗口右边界。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>输入</td>
        <td>单卡head个数，对应query的N轴。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>query/key/value的数据排布格式。</td>
        <td>支持BSH、SBH、BSND、BNSD。</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>稀疏模式。</td>
        <td>支持配置值0~6。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回Device侧需要申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回算子执行器，包含计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
      <td>query、keyIn、value、dy、pseShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOptional、softmaxSumOptional、softmaxInOptional、attentionInOptional、dqOut、dkOut、dvOut的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>query、keyIn、value、dy、pseShiftOptional、dropMaskOptional、paddingMaskOptional、attenMaskOptional、softmaxMaxOptional、softmaxSumOptional、softmaxInOptional、attentionInOptional、dqOut、dkOut、dvOut的数据格式不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>


## aclnnFlashAttentionScoreGradV2

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashAttentionScoreGradV2GetWorkspaceSize获取。</td>
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

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnFlashAttentionScoreGradV2默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value、dy的B：batchsize必须相等。
- 输入query、key、value的D：Head-Dim必须满足(qD == kD && kD >= vD)。
- 输入query、key、value、dy的inputLayout必须一致。
- 输入key/value的shape除D外必须一致，在query/key/value的D大小相同的情况下，query/dy的shape必须一致。
- 支持输入query/dy的N和key/value的N不相等，但必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。
- 关于数据shape的约束，以inputLayout的BSND、BNSD为例（BSH、SBH下H=N\*D），其中：
    -   B：取值范围为1\~2M。带prefixOptional的时候B最大支持2K。
    -   N：取值范围为1\~256。
    -   S：取值范围为1\~1M。
    -   D：取值范围为1\~768。
    -   KeepProb: 取值范围为(0, 1].
- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
- pseShiftOptional：如果Sq大于1024的每个batch的Sq与Skv等长且是sparseMode为0、2、3的下三角掩码场景，可使能alibi位置编码压缩，此时只需要输入原始PSE最后1024行，实现内存优化，即alibi_compress = ori_pse[:, :, -1024:, :]，具体如下：
  - 参数每个batch不相同时，shape为BNHSkv(H=1024)。
  - 每个batch相同时，shape为1NHSkv(H=1024)。
  - 如果pseType为2或3的时候，数据类型需为FLOAT32, 对应shape支持范围是[B,N]或[N]。
  - 如果不使能该参数，pseShiftOptional需要传入nullptr，pseType需要传入1。
- innerPrecise: 当前0、1为保留配置值，2为使能无效行计算，其功能是避免在计算过程中存在整行mask进而导致精度有损失，但是该配置会导致性能下降。 如果算子可判断出存在无效行场景，会自动使能无效行计算，例如sparseMode为3，Sq > Skv场景。
- pseType 各个取值含义
    | pseType     | 含义                              |      备注   |
    | ----------- | --------------------------------- | ----------|
    | 0           | 外部传入pse 先mul再add              | - |
    | 1           | 外部传入pse 先add再mul              | 跟[FlashAttentionScoreGrad](./aclnnFlashAttentionScoreGrad.md)实现一致。 |
    | 2           | 内部生成pse 先mul再add              | - |
    | 3           | 内部生成pse 先mul再add再sqrt         | - |
- sparseMode的约束如下: 
  - 当所有的attenMaskOptional的shape小于2048且相同的时候，建议使用default模式，来减少内存使用量；
  - 配置为1、2、3、5时，用户配置的preTokens、nextTokens不会生效；
  - 配置为0、4时，须保证attenMaskOptional与preTokens、nextTokens的范围一致。
  - 用户不特意指定时建议传入0。
  - sparse不同模式的详细说明请参见[sparse模式说明](../../../docs/zh/context/sparse_mode参数说明.md)。
- 部分场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此时建议做轴切分处理，注：这里的计算量会受B、S、N、D等参数的影响，值越大计算量越大。
- 关于softmaxMax与softmaxSum参数的约束：输入格式固定为\[B, N, S, 8\]，TND的输入格式除外，此时为\[T, N, 8\],注：T=B*S。
- headNum的取值必须和传入的Query中的N值保持一致。
- band场景，preTokens和nextTokens之间必须要有交集。
- prefixOptional稀疏计算场景即sparseMode=5或者sparseMode=6，当Sq > Skv时，prefix的N值取值范围\[0, Skv\]，当Sq <= Skv时，prefix的N值取值范围\[Skv-Sq, Skv\]。
- pseShiftOptional Sq大于1024时如果配置BNHS、1NHS，需要Sq和Skv等长。


## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> dxShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};
  std::vector<int64_t> softmaxMaxShape = {1, 1, 256, 8};
  std::vector<int64_t> softmaxSumShape = {1, 1, 256, 8};
  std::vector<int64_t> attentionInShape = {256, 1, 128};

  std::vector<int64_t> dqShape = {256, 1, 128};
  std::vector<int64_t> dkShape = {256, 1, 128};
  std::vector<int64_t> dvShape = {256, 1, 128};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<float> dxHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);
  std::vector<float> attentionInHostData(32768, 1);
  std::vector<float> dqHostData(32768, 0);
  std::vector<float> dkHostData(32768, 0);
  std::vector<float> dvHostData(32768, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT16, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT16, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT16, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  char layOut[5] = {'S', 'B', 'H', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnFlashAttentionScoreGradV2第一段接口
  ret = aclnnFlashAttentionScoreGradV2GetWorkspaceSize(q, k, v, dx, pse, dropMask, padding,
            attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, prefix, qStartIdx, kvStartIdx,
            scaleValue, keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise, sparseMode, pseType,
            dq, dk, dv, dpse, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnFlashAttentionScoreGradV2第二段接口
  ret = aclnnFlashAttentionScoreGradV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(dqShape, &dqDeviceAddr);
  PrintOutResult(dkShape, &dkDeviceAddr);
  PrintOutResult(dvShape, &dvDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```