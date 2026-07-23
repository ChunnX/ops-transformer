# aclnnQkvRmsNormRopeCacheWithKScale

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：`aclnnQkvRmsNormRopeCacheWithKScale`面向大语言模型推理中的PagedAttention KV Cache更新场景。接口从融合输入`qkv`中拆分Q、K、V分量，对Q/K执行RMSNorm、RoPE和共享`rotationOptional`矩阵乘，随后将Q/K动态量化为FP8 E4M3FN；Q分支输出`qOut`和`qScale`，K分支按`slotMapping`写入`kCacheRef`和`kScaleCacheRef`。V分支按`vScaleOptional`缩放后量化为FP8 E4M3FN，并按`slotMapping`写入`vCacheRef`。

- 计算公式：

  按`headNums=[Nq, Nk, Nv]`从`qkv`拆分Q、K、V：

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  Q/K分支分别使用`qGamma`和`kGamma`做RMSNorm：

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  第`b`个batch中第`i`个token的RoPE位置由`queryStartLoc`和`seqLens`确定：

  $$
  position = seqLens[b] - (queryStartLoc[b + 1] - queryStartLoc[b]) + i
  $$

  Q/K分支执行RoPE，`cosSin[..., :D/2]`为cos，`cosSin[..., D/2:]`为sin；V分支不执行RoPE：

  $$
  y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  Q/K共享`rotationOptional`矩阵：

  $$
  q_{rot} = q_{rope} @ rotationOptional,\quad k_{rot} = k_{rope} @ rotationOptional
  $$

  Q/K按每个token和head做动态量化，FP8 E4M3FN最大有限值使用`FP8_E4M3FN_MAX`，该值为448：

  $$
  scale = max(abs(x)) / FP8\_E4M3FN\_MAX,\quad x_{fp8} = cast(x / scale)
  $$

  V分支按`vScaleOptional`缩放后量化：

  $$
  v_{fp8} = cast(v * vScaleOptional)
  $$

  Cache写回位置由`slotMapping`决定：

  $$
  blockId = slotMapping[t] / BlockSize,\quad blockOffset = slotMapping[t]\ \%\ BlockSize
  $$

  $$
  kCacheRef[blockId, nk, blockOffset, :] = k_{fp8}[t, nk, :]
  $$

  $$
  vCacheRef[blockId, nv, blockOffset, :] = v_{fp8}[t, nv, :]
  $$

  $$
  kScaleCacheRef[blockId, nk, blockOffset, 0] = k_{scale}[t, nk]
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnQkvRmsNormRopeCacheWithKScale”接口执行计算。

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor   *qkv,
    const aclTensor   *qGamma,
    const aclTensor   *kGamma,
    const aclTensor   *cosSin,
    const aclTensor   *slotMapping,
    aclTensor         *kCacheRef,
    aclTensor         *vCacheRef,
    aclTensor         *kScaleCacheRef,
    const aclTensor   *queryStartLoc,
    const aclTensor   *seqLens,
    const aclTensor   *rotationOptional,
    const aclTensor   *vScaleOptional,
    const aclIntArray *headNums,
    const char        *layoutQkv,
    const char        *layoutQOut,
    float              epsilon,
    aclTensor         *qOut,
    aclTensor         *qScale,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1312px"><colgroup>
  <col style="width: 158px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 137px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
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
      <th>维度（shape）</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">qkv（const aclTensor*）</td>
      <td>输入</td>
      <td>Q/K/V融合输入，对应公式中的<code>qkv</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td><code>[T,Nq+Nk+Nv,D]</code>或<code>[Nq+Nk+Nv,T,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qGamma（const aclTensor*）</td>
      <td>输入</td>
      <td>Q分支RMSNorm权重，对应Q分支公式中的<code>gamma</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kGamma（const aclTensor*）</td>
      <td>输入</td>
      <td>K分支RMSNorm权重，对应K分支公式中的<code>gamma</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cosSin（const aclTensor*）</td>
      <td>输入</td>
      <td>RoPE位置编码表，对应公式中的<code>cosSin</code>、<code>cos</code>和<code>sin</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>前<code>D/2</code>列为cos，后<code>D/2</code>列为sin。</li><li>第一维需覆盖本次调用会访问的RoPE位置。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[MaxSeqLen,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">slotMapping（const aclTensor*）</td>
      <td>输入</td>
      <td>每个token写入cache的slot索引，对应公式中的<code>slotMapping</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>取值范围应为<code>[0,BlockNum*BlockSize-1]</code>。</li><li>同一次调用内多个token写入同一slot时，最终写入顺序和结果未定义。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[T]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kCacheRef（aclTensor*）</td>
      <td>输出</td>
      <td>KCache写回Tensor，对应公式中的<code>kCacheRef</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>接口基于传入Tensor原地更新。</li><li>支持非连续Tensor，需满足“约束说明”中的stride限制。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td><code>[BlockNum,Nk,BlockSize,D]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vCacheRef（aclTensor*）</td>
      <td>输出</td>
      <td>VCache写回Tensor，对应公式中的<code>vCacheRef</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>接口基于传入Tensor原地更新。</li><li>支持非连续Tensor，需满足“约束说明”中的stride限制。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td><code>[BlockNum,Nv,BlockSize,D]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kScaleCacheRef（aclTensor*）</td>
      <td>输出</td>
      <td>K动态量化scale cache写回Tensor，对应公式中的<code>kScaleCacheRef</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>接口基于传入Tensor原地更新。</li><li>支持非连续Tensor，需满足“约束说明”中的stride限制。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[BlockNum,Nk,BlockSize,1]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">queryStartLoc（const aclTensor*）</td>
      <td>输入</td>
      <td>当前调用内各batch token数的前缀和，对应公式中的<code>queryStartLoc</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>长度需大于等于2。</li><li><code>queryStartLoc[0]</code>应为0，<code>queryStartLoc[Batch]</code>应为<code>T</code>。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[Batch+1]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">seqLens（const aclTensor*）</td>
      <td>输入</td>
      <td>每个batch追加本次token后的实际序列长度，对应公式中的<code>seqLens</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li><li>长度需等于<code>queryStartLoc.shape[0]-1</code>。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[Batch]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">rotationOptional（const aclTensor*）</td>
      <td>可选输入</td>
      <td>Q/K共享矩阵乘权重，对应公式中的<code>rotationOptional</code>。</td>
      <td><ul><li>该参数带Optional后缀，但当前实现不支持传入空指针或空Tensor。</li><li>用于Q/K共享矩阵乘场景。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td><code>[D,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vScaleOptional（const aclTensor*）</td>
      <td>可选输入</td>
      <td>V分支量化缩放因子，对应公式中的<code>vScaleOptional</code>。</td>
      <td><ul><li>该参数带Optional后缀，但当前实现不支持传入空指针或空Tensor。</li><li>用于V分支FP8量化前缩放场景。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[Nv]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">headNums（const aclIntArray*）</td>
      <td>输入</td>
      <td>Q/K/V头数数组，依次映射为公式中的<code>Nq</code>、<code>Nk</code>、<code>Nv</code>。</td>
      <td><ul><li>不支持空指针。</li><li>必须包含3个正整数。</li><li><code>Nv</code>必须等于<code>Nk</code>。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>长度为3</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutQkv（const char*）</td>
      <td>输入</td>
      <td><code>qkv</code>的N/T轴布局标识，对应公式中<code>T</code>、<code>Nq</code>、<code>Nk</code>、<code>Nv</code>所在轴。</td>
      <td><ul><li>默认值为<code>"TND"</code>。</li><li>传入空指针或空字符串时按默认值处理。</li><li>大小写敏感，仅支持<code>"TND"</code>和<code>"NTD"</code>。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutQOut（const char*）</td>
      <td>输入</td>
      <td><code>qOut</code>和<code>qScale</code>的N/T轴布局标识。</td>
      <td><ul><li>默认值为<code>"NTD"</code>。</li><li>传入空指针或空字符串时按默认值处理。</li><li>大小写敏感，仅支持<code>"TND"</code>和<code>"NTD"</code>。</li><li>当前不支持<code>layoutQkv="NTD"</code>、<code>layoutQOut="TND"</code>。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">epsilon（float）</td>
      <td>输入</td>
      <td>RMSNorm防除零参数，对应公式中的<code>epsilon</code>。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qOut（aclTensor*）</td>
      <td>输出</td>
      <td>Q分支FP8 E4M3FN量化输出，对应Q分支动态量化公式中的<code>x_{fp8}</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td><code>[T,Nq,D]</code>或<code>[Nq,T,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qScale（aclTensor*）</td>
      <td>输出</td>
      <td>Q分支每个token/head对应的动态量化scale，对应Q分支动态量化公式中的<code>scale</code>。</td>
      <td><ul><li>不支持空指针或空Tensor。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[T,Nq]</code>或<code>[Nq,T]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>不支持空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>不支持空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td style="white-space: nowrap">ACLNN_ERR_PARAM_NULLPTR</td>
      <td style="white-space: nowrap">161001</td>
      <td><code>qkv</code>、<code>qGamma</code>、<code>kGamma</code>、<code>cosSin</code>、<code>slotMapping</code>、<code>kCacheRef</code>、<code>vCacheRef</code>、<code>kScaleCacheRef</code>、<code>queryStartLoc</code>、<code>seqLens</code>、<code>rotationOptional</code>、<code>vScaleOptional</code>、<code>headNums</code>、<code>qOut</code>、<code>qScale</code>、<code>workspaceSize</code>或<code>executor</code>为空指针。</td>
    </tr>
    <tr>
      <td rowspan="5" style="white-space: nowrap">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5" style="white-space: nowrap">161002</td>
      <td>Tensor为空Tensor。</td>
    </tr>
    <tr>
      <td>Tensor数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>Tensor数据格式为私有格式，或不满足ND格式要求。</td>
    </tr>
    <tr>
      <td>Tensor shape、<code>headNums</code>、<code>layoutQkv</code>或<code>layoutQOut</code>不满足接口约束。</td>
    </tr>
    <tr>
      <td><code>kCacheRef</code>、<code>vCacheRef</code>或<code>kScaleCacheRef</code>非连续Tensor的stride不满足约束。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td style="white-space: nowrap">561101</td>
      <td>创建<code>aclOpExecutor</code>失败。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_NULLPTR</td>
      <td style="white-space: nowrap">561103</td>
      <td>输入Contiguous处理、Cache视图创建、算子任务构图或构图输出检查失败。</td>
    </tr>
  </tbody></table>

## aclnnQkvRmsNormRopeCacheWithKScale

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。当第一段接口返回的<code>workspaceSize</code>为0时，可传入<code>nullptr</code>。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize</code>获取。</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnQkvRmsNormRopeCacheWithKScale默认确定性实现。
- 输入shape限制：
  - 当前实现仅支持`D=128`。
  - `layoutQkv`控制`qkv`的N/T轴布局，默认值为`"TND"`；`layoutQOut`控制`qOut`和`qScale`的N/T轴布局，默认值为`"NTD"`：
    - `layoutQkv="TND"`，`layoutQOut="TND"`：`qkv=[T, Nq+Nk+Nv, D]`，`qOut=[T, Nq, D]`，`qScale=[T, Nq]`。
    - `layoutQkv="TND"`，`layoutQOut="NTD"`：`qkv=[T, Nq+Nk+Nv, D]`，`qOut=[Nq, T, D]`，`qScale=[Nq, T]`。
    - `layoutQkv="NTD"`，`layoutQOut="NTD"`：`qkv=[Nq+Nk+Nv, T, D]`，`qOut=[Nq, T, D]`，`qScale=[Nq, T]`。
  - `kCacheRef`、`vCacheRef`和`kScaleCacheRef`的`BlockNum`和`BlockSize`必须一致。
  - `kCacheRef`、`vCacheRef`和`kScaleCacheRef`均为4维正stride，最后一维stride为1；`kCacheRef`和`vCacheRef`前三维stride必须一致。
- 输入值域限制：
  - `seqLens[b]`必须满足`seqLens[b] >= queryStartLoc[b+1] - queryStartLoc[b]`。若`seqLens[b]`小于该batch本次调用的token数，行为未定义。
  - 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

- <term>Ascend 950PR/Ascend 950DT</term>：

  ```c++
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  struct TensorResource {
      aclTensor *tensor = nullptr;
      void *deviceAddr = nullptr;
  };

  struct AclResource {
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      bool aclInited = false;
      bool deviceSet = false;
      std::vector<TensorResource *> tensors;
      aclIntArray *headNums = nullptr;
      aclOpExecutor *executor = nullptr;
      void *workspaceAddr = nullptr;
  };

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
      }
      return shapeSize;
  }

  std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t> &shape)
  {
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      return strides;
  }

  uint16_t FloatToBf16(float value)
  {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      return static_cast<uint16_t>(bits >> 16);
  }

  int Init(int32_t deviceId, AclResource &resource)
  {
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      resource.aclInited = true;

      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      resource.deviceId = deviceId;
      resource.deviceSet = true;

      ret = aclrtCreateStream(&resource.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return ACL_SUCCESS;
  }

  void FreeResource(AclResource &resource)
  {
      for (auto *tensorResource : resource.tensors) {
          if (tensorResource != nullptr && tensorResource->tensor != nullptr) {
              aclDestroyTensor(tensorResource->tensor);
              tensorResource->tensor = nullptr;
          }
      }
      if (resource.headNums != nullptr) {
          aclDestroyIntArray(resource.headNums);
          resource.headNums = nullptr;
      }
      for (auto *tensorResource : resource.tensors) {
          if (tensorResource != nullptr && tensorResource->deviceAddr != nullptr) {
              aclrtFree(tensorResource->deviceAddr);
              tensorResource->deviceAddr = nullptr;
          }
      }
      if (resource.workspaceAddr != nullptr) {
          aclrtFree(resource.workspaceAddr);
          resource.workspaceAddr = nullptr;
      }
      if (resource.stream != nullptr) {
          aclrtDestroyStream(resource.stream);
          resource.stream = nullptr;
      }
      if (resource.deviceSet) {
          aclrtResetDevice(resource.deviceId);
          resource.deviceSet = false;
      }
      if (resource.aclInited) {
          aclFinalize();
          resource.aclInited = false;
      }
  }

  int ReturnAfterCleanup(int ret, AclResource &resource)
  {
      FreeResource(resource);
      return ret;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                      TensorResource &resource)
  {
      const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
      auto ret = aclrtMalloc(&resource.deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(resource.deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides = GetContiguousStrides(shape);
      resource.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), resource.deviceAddr);
      CHECK_RET(resource.tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return ACL_SUCCESS;
  }

  int main()
  {
      AclResource resource;
      auto ret = Init(0, resource);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
                return ReturnAfterCleanup(ret, resource));

      constexpr int64_t T = 4;
      constexpr int64_t Nq = 16;
      constexpr int64_t Nk = 2;
      constexpr int64_t Nv = 2;
      constexpr int64_t D = 128;
      constexpr int64_t Batch = 1;
      constexpr int64_t MaxSeqLen = 16;
      constexpr int64_t BlockNum = 1;
      constexpr int64_t BlockSize = 16;

      std::vector<int64_t> qkvShape = {T, Nq + Nk + Nv, D};
      std::vector<int64_t> qGammaShape = {D};
      std::vector<int64_t> kGammaShape = {D};
      std::vector<int64_t> cosSinShape = {MaxSeqLen, D};
      std::vector<int64_t> slotMappingShape = {T};
      std::vector<int64_t> kCacheShape = {BlockNum, Nk, BlockSize, D};
      std::vector<int64_t> vCacheShape = {BlockNum, Nv, BlockSize, D};
      std::vector<int64_t> kScaleCacheShape = {BlockNum, Nk, BlockSize, 1};
      std::vector<int64_t> queryStartLocShape = {Batch + 1};
      std::vector<int64_t> seqLensShape = {Batch};
      std::vector<int64_t> rotationShape = {D, D};
      std::vector<int64_t> vScaleShape = {Nv};
      std::vector<int64_t> qOutShape = {T, Nq, D};
      std::vector<int64_t> qScaleShape = {T, Nq};

      std::vector<uint16_t> qkvHostData(GetShapeSize(qkvShape), FloatToBf16(0.125f));
      std::vector<float> qGammaHostData(GetShapeSize(qGammaShape), 1.0f);
      std::vector<float> kGammaHostData(GetShapeSize(kGammaShape), 1.0f);
      std::vector<float> cosSinHostData(GetShapeSize(cosSinShape), 0.0f);
      for (int64_t row = 0; row < MaxSeqLen; ++row) {
          for (int64_t col = 0; col < D / 2; ++col) {
              cosSinHostData[row * D + col] = 1.0f;
          }
      }
      std::vector<int32_t> slotMappingHostData = {0, 1, 2, 3};
      std::vector<uint8_t> kCacheHostData(GetShapeSize(kCacheShape), 0);
      std::vector<uint8_t> vCacheHostData(GetShapeSize(vCacheShape), 0);
      std::vector<float> kScaleCacheHostData(GetShapeSize(kScaleCacheShape), 0.0f);
      std::vector<int32_t> queryStartLocHostData = {0, T};
      std::vector<int32_t> seqLensHostData = {T};
      std::vector<uint16_t> rotationHostData(GetShapeSize(rotationShape), FloatToBf16(0.0f));
      for (int64_t i = 0; i < D; ++i) {
          rotationHostData[i * D + i] = FloatToBf16(1.0f);
      }
      std::vector<float> vScaleHostData(GetShapeSize(vScaleShape), 1.0f);
      std::vector<uint8_t> qOutHostData(GetShapeSize(qOutShape), 0);
      std::vector<float> qScaleHostData(GetShapeSize(qScaleShape), 0.0f);

      TensorResource qkv;
      TensorResource qGamma;
      TensorResource kGamma;
      TensorResource cosSin;
      TensorResource slotMapping;
      TensorResource kCache;
      TensorResource vCache;
      TensorResource kScaleCache;
      TensorResource queryStartLoc;
      TensorResource seqLens;
      TensorResource rotation;
      TensorResource vScale;
      TensorResource qOut;
      TensorResource qScale;
      resource.tensors = {&qkv,         &qGamma,       &kGamma, &cosSin, &slotMapping, &kCache, &vCache,
                          &kScaleCache, &queryStartLoc, &seqLens, &rotation, &vScale,  &qOut,   &qScale};

      ret = CreateAclTensor(qkvHostData, qkvShape, ACL_BF16, qkv);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qGammaHostData, qGammaShape, ACL_FLOAT, qGamma);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kGammaHostData, kGammaShape, ACL_FLOAT, kGamma);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(cosSinHostData, cosSinShape, ACL_FLOAT, cosSin);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(slotMappingHostData, slotMappingShape, ACL_INT32, slotMapping);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kCacheHostData, kCacheShape, ACL_FLOAT8_E4M3FN, kCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(vCacheHostData, vCacheShape, ACL_FLOAT8_E4M3FN, vCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kScaleCacheHostData, kScaleCacheShape, ACL_FLOAT, kScaleCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(queryStartLocHostData, queryStartLocShape, ACL_INT32, queryStartLoc);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(seqLensHostData, seqLensShape, ACL_INT32, seqLens);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(rotationHostData, rotationShape, ACL_BF16, rotation);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(vScaleHostData, vScaleShape, ACL_FLOAT, vScale);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qOutHostData, qOutShape, ACL_FLOAT8_E4M3FN, qOut);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qScaleHostData, qScaleShape, ACL_FLOAT, qScale);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));

      std::vector<int64_t> headNumsVec = {Nq, Nk, Nv};
      resource.headNums = aclCreateIntArray(headNumsVec.data(), headNumsVec.size());
      CHECK_RET(resource.headNums != nullptr,
                LOG_PRINT("aclCreateIntArray failed.\n"); return ReturnAfterCleanup(ACL_ERROR_INVALID_PARAM, resource));

      const char *layoutQkv = "TND";
      const char *layoutQOut = "TND";
      float epsilon = 1e-6f;
      uint64_t workspaceSize = 0;
      aclnnStatus status = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
          qkv.tensor, qGamma.tensor, kGamma.tensor, cosSin.tensor, slotMapping.tensor, kCache.tensor, vCache.tensor,
          kScaleCache.tensor, queryStartLoc.tensor, seqLens.tensor, rotation.tensor, vScale.tensor, resource.headNums,
          layoutQkv, layoutQOut, epsilon, qOut.tensor, qScale.tensor, &workspaceSize, &resource.executor);
      CHECK_RET(status == ACL_SUCCESS,
                LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize failed. ERROR: %d\n", status);
                return ReturnAfterCleanup(static_cast<int>(status), resource));

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&resource.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                    return ReturnAfterCleanup(ret, resource));
      }

      status = aclnnQkvRmsNormRopeCacheWithKScale(resource.workspaceAddr, workspaceSize, resource.executor,
                                                  resource.stream);
      CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale failed. ERROR: %d\n", status);
                return ReturnAfterCleanup(static_cast<int>(status), resource));

      ret = aclrtSynchronizeStream(resource.stream);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                return ReturnAfterCleanup(ret, resource));
      LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale execute success.\n");
      FreeResource(resource);
      return 0;
  }
  ```
