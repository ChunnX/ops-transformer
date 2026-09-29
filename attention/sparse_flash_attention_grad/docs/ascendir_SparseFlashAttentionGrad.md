# Ascendir_SparseFlashAttentionGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：根据sparse_indices（topkIndices）对key和value选取大小为sparse_block_size（selectedBlockSize）的数据重排，接着进行训练场景下计算注意力的反向输出。V2版本新增可选的可学习OSS Sink输入sinks与对应梯度输出dsink。

- 计算公式：根据传入的sparse_indices对key和value选取数量为selectedBlockCount个大小为selectedBlockSize的数据重排，公式如下：

  $$
   selectedKey\text{ }=\text{ }Gather \left( key,sparse\_indices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectedBlockCount\right. \right. \right. \right.
  $$

  $$
   selectedValue\text{ }=\text{ }Gather \left( value,sparse\_indices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectedBlockCount\right. \right. \right. \right.
  $$

  KV merge场景下，若value为空指针，则上述公式中的value按key处理，selectedValue等价于selectedKey。该场景的反向输出不单独返回dv，dk表示合并后的梯度：

  $$
   dk = dK + dV
  $$

  阶段1：根据矩阵乘法导数规则，计算$dP$和$dV$：

  $$
   dP\mathop{{}}\nolimits_{{t,:}}=dO\mathop{{}}\nolimits_{{t,:}}\text{@}V\mathop{{}}\nolimits^{{T}}
  $$

  $$
   dV \left[ u \left] =P\mathop{{}}\nolimits_{{T}}^{{t,:}}\text{@}dO\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

  阶段2：计算$dS$：

  $$
   d\mathop{{S}}\nolimits_{{t,:}}= \left[ P\mathop{{}}\nolimits_{{t,:}}@ \left( dP\mathop{{}}\nolimits_{{t,:}}-FlashSoftmaxGrad \left( dO,O \left)  \left)  \right] \right. \right. \right. \right.
  $$

  阶段3：计算$dQ$与$dK$：

  $$
   d\mathop{{Q}}\nolimits_{{t,:}}=d\mathop{{S}}\nolimits_{{t,:}}@K \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}/\sqrt{{d\mathop{{}}\nolimits_{{k,:}}}}\right. \right.
  $$

  $$
   dK \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}=dS\mathop{{}}\nolimits_{{t,:t}}\mathop{{}}\nolimits^{{T}}\text{@}Q/\sqrt{{d\mathop{{}}\nolimits_{{t,:}}}}\right. \right.
  $$

  阶段4：计算$dSink$（仅当提供sinks输入时）：

  Sink在前向中扮演一个"虚拟key"角色，其对应的"虚拟value"为0，不直接贡献输出O，但通过softmax归一化因子$\ell$间接影响所有注意力权重$P_j$。因此Sink参与前向softmax的max与sum的计算：

  $$
   m \mathop{{}}=\mathop{{}}\nolimits max \left( max \mathop{{}}\nolimits_{{j}} \left( S \mathop{{}}\nolimits_{{t,j}} \right) ,\mathop{{}}\nolimits sink \right)
  $$

  $$
   e \mathop{{}}\nolimits_{{sink}} = exp \left( sink - m \right)
  $$

  $$
   \ell = e \mathop{{}}\nolimits_{{sink}} + \sum \mathop{{}}\nolimits_{{j}} exp \left( S \mathop{{}}\nolimits_{{t,j}} - m \right)
  $$

  引入Sink后，softmax反向的公共项$D_{i}$不变：$D_{i} = \sum \mathop{{}}\nolimits_{{j}} P \mathop{{}}\nolimits_{{t,j}} \cdot dP \mathop{{}}\nolimits_{{t,j}}$（其中$dP \mathop{{}}\nolimits_{{sink}} = dO \text{@} V^{T} = 0$，Sink不直接贡献输出，因此不产生dP）。Sink的梯度如下：

  $$
   dSink \left[ h \right] = P \mathop{{}}\nolimits_{{sink}} \odot \left( dP \mathop{{}}\nolimits_{{sink}} - D_{i} \right) = \sum \mathop{{}}\nolimits_{{b,i}} P \mathop{{}}\nolimits_{{sink}} \left[ b,i,h \right] \cdot \left( - D_{i} \left[ b,i,h \right] \right)
  $$

## Ascend IR定义

Ascend IR定义所在头文件路径为[op_graph/sparse_flash_attention_grad_proto.h](../op_graph/sparse_flash_attention_grad_proto.h)。

```cpp
REG_OP(SparseFlashAttentionGrad)
    .INPUT(query, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(key, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(sparse_indices, TensorType({DT_INT32}))
    .INPUT(d_out, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(out, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(softmax_max, TensorType({DT_FLOAT}))
    .INPUT(softmax_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sinks, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(value, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(d_query, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(d_key, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(d_value, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(d_query_rope, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(d_key_rope, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(d_sinks, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(scale_value, Float)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(deterministic, Bool, false)
    .OP_END_FACTORY_REG(SparseFlashAttentionGrad)
```

### 参数说明

<table style="undefined;table-layout: fixed; width: 1340px"><colgroup>
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 260px">
  <col style="width: 280px">
  <col style="width: 210px">
  <col style="width: 80px">
  <col style="width: 240px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query（tensor）</td>
      <td>输入</td>
      <td>attention结构的输入query。</td>
      <td>query、key、value、sparse_indices、d_out、out、softmax_max、softmax_sum的shape维度保持一致。N1支持128、64、32、16、8、4、2、1（<term>Ascend 950PR&950DT系列产品</term>额外还支持48、24、12、6、3）。</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S1,N1,D]、[T1,N1,D]</td>
    </tr>
    <tr>
      <td>key（tensor）</td>
      <td>输入</td>
      <td>attention结构的输入key。</td>
      <td>-</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S2,N2,D]、[T2,N2,D]</td>
    </tr>
    <tr>
      <td>sparse_indices（tensor）</td>
      <td>输入</td>
      <td>稀疏场景下选择的权重较高的注意力索引。</td>
      <td>K支持1024、2048、2051、3072、4096、5120、6144、7168、8192。</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B,S1,N2,K]、[T1,N2,K]</td>
    </tr>
    <tr>
      <td>d_out（tensor）</td>
      <td>输入</td>
      <td>注意力输出矩阵的梯度。</td>
      <td>-</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S1,N1,D]、[T1,N1,D]</td>
    </tr>
    <tr>
      <td>out（tensor）</td>
      <td>输入</td>
      <td>注意力输出矩阵。</td>
      <td>-</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S1,N1,D]、[T1,N1,D]</td>
    </tr>
    <tr>
      <td>softmax_max（tensor）</td>
      <td>输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>G为N1/N2。</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N2,S1,G]、[N2,T1,G]</td>
    </tr>
    <tr>
      <td>softmax_sum（tensor）</td>
      <td>输入</td>
      <td>注意力正向计算的中间输出。</td>
      <td>-</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N2,S1,G]、[N2,T1,G]</td>
    </tr>
    <tr>
      <td>sinks（tensor）</td>
      <td>可选输入</td>
      <td>可学习OSS Sink输入。</td>
      <td>未提供时不参与计算；传入sinks时必须同时提供dsink输出；仅支持<term>Ascend 950PR&950DT系列产品</term>。</td>
      <td>float32</td>
      <td>ND</td>
      <td>[N1]</td>
    </tr>
    <tr>
      <td>value（tensor）</td>
      <td>可选输入</td>
      <td>attention结构的输入value。</td>
      <td>传入空指针时启用KV merge，内部按value=key处理；此时要求d_value也传入空指针，d_key返回dK+dV的合并结果。</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S2,N2,D]、[T2,N2,D]</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query（tensor）</td>
      <td>可选输入</td>
      <td>每个Batch中，query的有效token数。</td>
      <td>当layout为TND时该变量必须存在；长度与B保持一致；累加和与T1保持一致（EOD场景下累加和可以小于T1）；取值须为非负数。</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B]</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_kv（tensor）</td>
      <td>可选输入</td>
      <td>每个Batch中，key、value的有效token数。</td>
      <td>当layout为TND时该变量必须存在；长度与B保持一致；累加和与T2保持一致（EOD场景下累加和可以小于T2）。</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B]</td>
    </tr>
    <tr>
      <td>query_rope（tensor）</td>
      <td>可选输入</td>
      <td>MLA rope部分：query位置编码的输出。</td>
      <td>Dr为64。</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S1,N1,Dr]、[T1,N1,Dr]</td>
    </tr>
    <tr>
      <td>key_rope（tensor）</td>
      <td>可选输入</td>
      <td>MLA rope部分：key位置编码的输出。</td>
      <td>-</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S2,N2,Dr]、[T2,N2,Dr]</td>
    </tr>
    <tr>
      <td>scale_value（float）</td>
      <td>必要属性</td>
      <td>缩放系数。</td>
      <td>建议值：公式中d开根号的倒数。</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_block_size（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>选择的块的大小。</li>
          <li>默认值为1。</li>
        </ul>
      </td>
      <td><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>支持1、8、16、32、64；<term>Ascend 950PR&950DT系列产品</term>支持1。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout（string）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>输入数据的layout格式。</li>
          <li>默认值为"BSND"。</li>
        </ul>
      </td>
      <td>支持BSND、TND，query与key/value共用同一layout。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>表示sparse的模式。0表示不做mask操作；3表示rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          <li>默认值为3。</li>
        </ul>
      </td>
      <td>仅支持配置值为0、3。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>Attention算子里，对S矩阵的滑窗起始位置。</li>
          <li>默认值为2147483647。</li>
        </ul>
      </td>
      <td>仅支持默认值2147483647。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>Attention算子里，对S矩阵的滑窗终止位置。</li>
          <li>默认值为2147483647。</li>
        </ul>
      </td>
      <td>仅支持默认值2147483647。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deterministic（bool）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>确定性计算。</li>
          <li>默认值为false。</li>
        </ul>
      </td>
      <td>与整网确定性参数use_deterministic_algorithms保持一致。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>d_query（tensor）</td>
      <td>输出</td>
      <td>表示query的梯度。</td>
      <td>与输入query的shape维度保持一致。</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S1,N1,D]、[T1,N1,D]</td>
    </tr>
    <tr>
      <td>d_key（tensor）</td>
      <td>输出</td>
      <td>表示key的梯度。</td>
      <td>KV merge场景下表示key和value的合并梯度，即dK+dV。</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S2,N2,D]、[T2,N2,D]</td>
    </tr>
    <tr>
      <td>d_value（tensor）</td>
      <td>可选输出</td>
      <td>表示value的梯度。</td>
      <td>KV merge场景下不单独返回，与d_key合并输出。</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S2,N2,D]、[T2,N2,D]</td>
    </tr>
    <tr>
      <td>d_query_rope（tensor）</td>
      <td>可选输出</td>
      <td>表示query_rope的梯度。</td>
      <td>-</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S1,N1,Dr]、[T1,N1,Dr]</td>
    </tr>
    <tr>
      <td>d_key_rope（tensor）</td>
      <td>可选输出</td>
      <td>表示key_rope的梯度。</td>
      <td>-</td>
      <td>bf16、fp16</td>
      <td>ND</td>
      <td>[B,S2,N2,Dr]、[T2,N2,Dr]</td>
    </tr>
    <tr>
      <td>d_sinks（tensor）</td>
      <td>可选输出</td>
      <td>表示sinks的梯度。</td>
      <td>仅sinks存在时输出，否则返回空tensor（shape [0]）。</td>
      <td>float32</td>
      <td>ND</td>
      <td>[N1]</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 确定性计算：
  - Ascendir_SparseFlashAttentionGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 入参为空的场景处理：
  - query为空tensor：直接返回。
- 当前只支持value和key完全一致的场景。
- KV merge场景下（<term>Ascend 950PR&950DT系列产品</term>与<term>Atlas A2系列产品</term>支持），value和d_value需传入空指针，d_key返回dK+dV，不再单独返回d_value。
- OSS Sink：传入sinks时未提供d_sinks会返回参数校验失败；sinks仅支持<term>Ascend 950PR&950DT系列产品</term>，其它平台传入sinks会返回运行时错误。
- 参数query中的D和key、value的D值相等为512，参数query_rope中的Dr和key_rope的Dr值相等为64。
- 参数query、key、value的数据类型必须保持一致。
- 仅支持sparse_mode=0或3（无mask或以右顶点为划分的下三角场景）。
- 仅支持BSND或TND layout；关于数据shape的约束如下：
  - B：取值范围1~256。
  - S1支持1~128K，S2支持1~1M；S1、S2支持不等长。
  - N1支持1/2/4/8/16/32/64/128，<term>Ascend 950PR&950DT系列产品</term>额外还支持48、24、12、6、3。
  - N2：仅支持1。
  - D：仅支持512。
  - Drope：仅支持64。
  - topk：1024、2048、2051、3072、4096、5120、6144、7168、8192。不建议topk * sparse_block_size超过100k，由于内部算法硬件限制可能会导致oom。
- sparse_block_size约束：
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持1、8、16、32、64。
  - <term>Ascend 950PR&950DT系列产品</term>：支持1。

## 调用示例

示例代码如下，仅供参考。GE图模式的编译和执行过程请参考[算子调用](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式)，完整代码见[test_geir_sparse_flash_attention_grad](../examples/test_geir_sparse_flash_attention_grad.cpp)。

```c++
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_sparse_flash_attention_grad.cpp
 * \brief GE graph construction sample for SparseFlashAttentionGrad (MLA-absorb mode, TND layout, fp16).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/sparse_flash_attention_grad_proto.h"

#define FAILED (-1)
#define SUCCESS 0

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t kQTokens = 1;                 // T1
constexpr int64_t kQHeads = 16;                 // N1
constexpr int64_t kKvTokens = 2048;             // T2
constexpr int64_t kKvHeads = 1;                 // N2
constexpr int64_t kHeadDim = 512;               // D (MLA latent)
constexpr int64_t kRopeDim = 64;                // Dr
constexpr double kScale = 0.041666666666666664; // 1.0 / sqrt(512 + 64)
constexpr float kRtol = 3e-2f;
constexpr float kAtol = 3e-2f;

std::string GetGeError()
{
    const AscendString errorMessage = GEGetErrorMsgV2();
    return errorMessage.GetString() == nullptr ? "" : errorMessage.GetString();
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// Deterministic value in [-0.45, 0.45]; the magnitude stays in the normal fp16 range.
float GenValue(uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    return ((static_cast<float>(seed & 0xFFU) / 255.0f) - 0.5f) * 0.9f;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    const uint32_t exponent = (value & 0x7C00U) >> 10;
    const uint32_t mantissa = value & 0x03FFU;
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -24) * ((sign != 0) ? -1.0f : 1.0f);
    }
    const uint32_t bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
int32_t AddInput(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, const string& name,
                 int32_t index, const vector<int64_t>& shape, DataType dtype, const vector<T>& hostData)
{
    auto dataOp = op::Data(name).set_attr_index(index - 1);
    TensorDesc desc(Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    const int64_t elementCount = GetShapeSize(shape);
    CHECK_RET(static_cast<int64_t>(hostData.size()) == elementCount,
              LOG_PRINT("[ERROR] %s data size mismatch\n", name.c_str());
              return FAILED);
    Tensor tensor;
    auto ret = tensor.SetTensorDesc(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetTensorDesc failed\n", name.c_str()); return FAILED);
    ret = tensor.SetData(reinterpret_cast<const uint8_t*>(hostData.data()), hostData.size() * sizeof(T));
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s SetData failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_input_desc_x(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_input_desc_x failed\n", name.c_str()); return FAILED);
    ret = dataOp.update_output_desc_y(desc);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] %s update_output_desc_y failed\n", name.c_str()); return FAILED);
    ret = graph.AddOp(dataOp);
    CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] Graph::AddOp failed for %s\n", name.c_str()); return FAILED);
    inputTensors.push_back(tensor);
    inputOps.push_back(dataOp);
    return SUCCESS;
}

struct ForwardStats {
    vector<float> softmaxMax;  // [N2, T1, G]
    vector<float> softmaxSum;  // [N2, T1, G]
    vector<float> attentionIn; // [T1, N1, D]
};

// CPU forward: score = scale * (q @ k^T + qRope @ kRope^T), out = softmax(score) @ v.
void ComputeForward(const vector<float>& q, const vector<float>& k, const vector<float>& v, const vector<float>& qRope,
                    const vector<float>& kRope, ForwardStats& stats)
{
    const int64_t t1 = kQTokens;
    const int64_t n1 = kQHeads;
    const int64_t t2 = kKvTokens;
    const int64_t n2 = kKvHeads;
    const int64_t d = kHeadDim;
    const int64_t dr = kRopeDim;
    const int64_t group = n1 / n2;
    stats.softmaxMax.assign(n2 * t1 * group, 0.0f);
    stats.softmaxSum.assign(n2 * t1 * group, 0.0f);
    stats.attentionIn.assign(t1 * n1 * d, 0.0f);
    for (int64_t i = 0; i < t1; ++i) {
        for (int64_t n = 0; n < n1; ++n) {
            const int64_t kvN = n / group;
            vector<float> score(t2);
            for (int64_t j = 0; j < t2; ++j) {
                float dot = 0.0f;
                for (int64_t t = 0; t < d; ++t) {
                    dot += q[(i * n1 + n) * d + t] * k[(j * n2 + kvN) * d + t];
                }
                for (int64_t t = 0; t < dr; ++t) {
                    dot += qRope[(i * n1 + n) * dr + t] * kRope[(j * n2 + kvN) * dr + t];
                }
                score[j] = static_cast<float>(kScale) * dot;
            }
            float maxScore = score[0];
            for (int64_t j = 1; j < t2; ++j) {
                maxScore = std::fmax(maxScore, score[j]);
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < t2; ++j) {
                score[j] = std::exp(score[j] - maxScore);
                sum += score[j];
            }
            stats.softmaxMax[(kvN * t1 + i) * group + (n % group)] = maxScore;
            stats.softmaxSum[(kvN * t1 + i) * group + (n % group)] = sum;
            for (int64_t t = 0; t < d; ++t) {
                float y = 0.0f;
                for (int64_t j = 0; j < t2; ++j) {
                    y += (score[j] / sum) * v[(j * n2 + kvN) * d + t];
                }
                stats.attentionIn[(i * n1 + n) * d + t] = y;
            }
        }
    }
}

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps,
                   vector<float>& q, vector<float>& k, vector<float>& v, vector<float>& dy, vector<float>& qRope,
                   vector<float>& kRope, ForwardStats& stats)
{
    // TND layout.
    const vector<int64_t> qShape = {kQTokens, kQHeads, kHeadDim};
    const vector<int64_t> kvShape = {kKvTokens, kKvHeads, kHeadDim};
    const vector<int64_t> indicesShape = {kQTokens, kKvHeads, kKvTokens};
    const vector<int64_t> qRopeShape = {kQTokens, kQHeads, kRopeDim};
    const vector<int64_t> kRopeShape = {kKvTokens, kKvHeads, kRopeDim};
    const vector<int64_t> statShape = {kKvHeads, kQTokens, kQHeads / kKvHeads};

    q.resize(GetShapeSize(qShape));
    k.resize(GetShapeSize(kvShape));
    v.resize(GetShapeSize(kvShape));
    dy.resize(GetShapeSize(qShape));
    qRope.resize(GetShapeSize(qRopeShape));
    kRope.resize(GetShapeSize(kRopeShape));
    for (int64_t i = 0; i < GetShapeSize(qShape); ++i) {
        q[i] = GenValue(static_cast<uint64_t>(i) * 7 + 1);
        dy[i] = GenValue(static_cast<uint64_t>(i) * 23 + 3);
    }
    for (int64_t i = 0; i < GetShapeSize(kvShape); ++i) {
        k[i] = GenValue(static_cast<uint64_t>(i) * 13 + 5);
        v[i] = GenValue(static_cast<uint64_t>(i) * 17 + 9);
    }
    for (int64_t i = 0; i < GetShapeSize(qRopeShape); ++i) {
        qRope[i] = GenValue(static_cast<uint64_t>(i) * 31 + 11);
    }
    for (int64_t i = 0; i < GetShapeSize(kRopeShape); ++i) {
        kRope[i] = GenValue(static_cast<uint64_t>(i) * 29 + 7);
    }
    ComputeForward(q, k, v, qRope, kRope, stats);

    vector<uint16_t> qHalf(q.size());
    vector<uint16_t> kHalf(k.size());
    vector<uint16_t> vHalf(v.size());
    vector<uint16_t> dyHalf(dy.size());
    vector<uint16_t> qRopeHalf(qRope.size());
    vector<uint16_t> kRopeHalf(kRope.size());
    vector<uint16_t> outHalf(stats.attentionIn.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qHalf[i] = FloatToHalf(q[i]);
        dyHalf[i] = FloatToHalf(dy[i]);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        kHalf[i] = FloatToHalf(k[i]);
        vHalf[i] = FloatToHalf(v[i]);
    }
    for (size_t i = 0; i < qRope.size(); ++i) {
        qRopeHalf[i] = FloatToHalf(qRope[i]);
    }
    for (size_t i = 0; i < kRope.size(); ++i) {
        kRopeHalf[i] = FloatToHalf(kRope[i]);
    }
    for (size_t i = 0; i < stats.attentionIn.size(); ++i) {
        outHalf[i] = FloatToHalf(stats.attentionIn[i]);
    }
    // Identity selection: every kv token is picked in order.
    vector<int32_t> indicesData(kKvTokens);
    std::iota(indicesData.begin(), indicesData.end(), 0);
    // TND sequence lengths are prefix sums.
    const vector<int32_t> actSeqQData = {static_cast<int32_t>(kQTokens)};
    const vector<int32_t> actSeqKvData = {static_cast<int32_t>(kKvTokens)};

    // Input order follows the IR definition: query, key, sparse_indices, d_out, out, softmax_max, softmax_sum,
    // sinks(optional, skipped), value, actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope.
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query", 1, qShape, DT_FLOAT16, qHalf) == SUCCESS, return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key", 2, kvShape, DT_FLOAT16, kHalf) == SUCCESS, return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "sparse_indices", 3, indicesShape, DT_INT32, indicesData) == SUCCESS,
        return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "d_out", 4, qShape, DT_FLOAT16, dyHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "out", 5, qShape, DT_FLOAT16, outHalf) == SUCCESS, return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "softmax_max", 6, statShape, DT_FLOAT, stats.softmaxMax) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "softmax_sum", 7, statShape, DT_FLOAT, stats.softmaxSum) == SUCCESS,
        return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "value", 9, kvShape, DT_FLOAT16, vHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_query", 10, {1}, DT_INT32, actSeqQData) == SUCCESS,
        return FAILED);
    CHECK_RET(
        AddInput(graph, inputTensors, inputOps, "actual_seq_lengths_kv", 11, {1}, DT_INT32, actSeqKvData) == SUCCESS,
        return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "query_rope", 12, qRopeShape, DT_FLOAT16, qRopeHalf) == SUCCESS,
              return FAILED);
    CHECK_RET(AddInput(graph, inputTensors, inputOps, "key_rope", 13, kRopeShape, DT_FLOAT16, kRopeHalf) == SUCCESS,
              return FAILED);

    auto node = op::SparseFlashAttentionGrad("sparse_flash_attention_grad");
    node.set_input_query(inputOps[0]);
    node.set_input_key(inputOps[1]);
    node.set_input_sparse_indices(inputOps[2]);
    node.set_input_d_out(inputOps[3]);
    node.set_input_out(inputOps[4]);
    node.set_input_softmax_max(inputOps[5]);
    node.set_input_softmax_sum(inputOps[6]);
    node.set_input_value(inputOps[7]);
    node.set_input_actual_seq_lengths_query(inputOps[8]);
    node.set_input_actual_seq_lengths_kv(inputOps[9]);
    node.set_input_query_rope(inputOps[10]);
    node.set_input_key_rope(inputOps[11]);
    node.set_attr_scale_value(kScale);
    node.set_attr_sparse_block_size(1);
    node.set_attr_layout("TND");
    node.set_attr_sparse_mode(0);
    node.set_attr_pre_tokens(2147483647);
    node.set_attr_next_tokens(2147483647);
    node.set_attr_deterministic(false);

    TensorDesc dqDesc(Shape(qShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dkDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dvDesc(Shape(kvShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dqRopeDesc(Shape(qRopeShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dkRopeDesc(Shape(kRopeShape), FORMAT_ND, DT_FLOAT16);
    TensorDesc dSinksDesc(Shape({0}), FORMAT_ND, DT_FLOAT);
    CHECK_RET(node.update_output_desc_d_query(dqDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_d_key(dkDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_d_value(dvDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_d_query_rope(dqRopeDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_d_key_rope(dkRopeDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(node.update_output_desc_d_sinks(dSinksDesc) == GRAPH_SUCCESS, return FAILED);
    CHECK_RET(graph.AddOp(node) == GRAPH_SUCCESS, return FAILED);
    outputOps.push_back(node);
    return SUCCESS;
}

// CPU backward reference of the MLA-absorb sparse attention.
void ComputeBackward(const vector<float>& q, const vector<float>& k, const vector<float>& v, const vector<float>& dy,
                     const vector<float>& qRope, const vector<float>& kRope, const ForwardStats& stats,
                     vector<float>& dq, vector<float>& dk, vector<float>& dv, vector<float>& dqRope,
                     vector<float>& dkRope)
{
    const int64_t t1 = kQTokens;
    const int64_t n1 = kQHeads;
    const int64_t t2 = kKvTokens;
    const int64_t n2 = kKvHeads;
    const int64_t d = kHeadDim;
    const int64_t dr = kRopeDim;
    const int64_t group = n1 / n2;
    dq.assign(t1 * n1 * d, 0.0f);
    dk.assign(t2 * n2 * d, 0.0f);
    dv.assign(t2 * n2 * d, 0.0f);
    dqRope.assign(t1 * n1 * dr, 0.0f);
    dkRope.assign(t2 * n2 * dr, 0.0f);
    for (int64_t i = 0; i < t1; ++i) {
        for (int64_t n = 0; n < n1; ++n) {
            const int64_t kvN = n / group;
            vector<float> prob(t2);
            for (int64_t j = 0; j < t2; ++j) {
                float dot = 0.0f;
                for (int64_t t = 0; t < d; ++t) {
                    dot += q[(i * n1 + n) * d + t] * k[(j * n2 + kvN) * d + t];
                }
                for (int64_t t = 0; t < dr; ++t) {
                    dot += qRope[(i * n1 + n) * dr + t] * kRope[(j * n2 + kvN) * dr + t];
                }
                const float score = static_cast<float>(kScale) * dot;
                prob[j] = std::exp(score - stats.softmaxMax[(kvN * t1 + i) * group + (n % group)]) /
                          stats.softmaxSum[(kvN * t1 + i) * group + (n % group)];
            }
            for (int64_t j = 0; j < t2; ++j) {
                float dP = 0.0f;
                for (int64_t t = 0; t < d; ++t) {
                    dP += dy[(i * n1 + n) * d + t] * v[(j * n2 + kvN) * d + t];
                    dv[(j * n2 + kvN) * d + t] += prob[j] * dy[(i * n1 + n) * d + t];
                }
                float dYdotY = 0.0f;
                for (int64_t t = 0; t < d; ++t) {
                    dYdotY += dy[(i * n1 + n) * d + t] * stats.attentionIn[(i * n1 + n) * d + t];
                }
                const float dS = prob[j] * (dP - dYdotY);
                for (int64_t t = 0; t < d; ++t) {
                    dq[(i * n1 + n) * d + t] += static_cast<float>(kScale) * dS * k[(j * n2 + kvN) * d + t];
                    dk[(j * n2 + kvN) * d + t] += static_cast<float>(kScale) * dS * q[(i * n1 + n) * d + t];
                }
                for (int64_t t = 0; t < dr; ++t) {
                    dqRope[(i * n1 + n) * dr + t] += static_cast<float>(kScale) * dS * kRope[(j * n2 + kvN) * dr + t];
                    dkRope[(j * n2 + kvN) * dr + t] += static_cast<float>(kScale) * dS * qRope[(i * n1 + n) * dr + t];
                }
            }
        }
    }
}

bool CheckOutput(const Tensor& tensor, const string& name, const vector<int64_t>& shape, const vector<float>& expected)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT16 &&
                  tensor.GetSize() == count * sizeof(uint16_t),
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, or data size\n", name.c_str());
              return false);
    vector<uint16_t> halfValues(count);
    std::memcpy(halfValues.data(), tensor.GetData(), count * sizeof(uint16_t));
    size_t mismatches = 0;
    float maxAbsErr = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float value = HalfToFloat(halfValues[i]);
        const float tolerance = kAtol + kRtol * std::fabs(expected[i]);
        const float err = std::fabs(value - expected[i]);
        maxAbsErr = std::fmax(maxAbsErr, err);
        if (err > tolerance) {
            if (mismatches < 4) {
                LOG_PRINT("[CHECK] %s[%zu]=%.7f expected=%.7f tolerance=%.7f\n", name.c_str(), i, value, expected[i],
                          tolerance);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("[CHECK] %s: count=%zu, maxAbsErr=%.7g, mismatches=%zu: %s\n", name.c_str(), count, maxAbsErr, mismatches,
              mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs, const vector<float>& q, const vector<float>& k,
                        const vector<float>& v, const vector<float>& dy, const vector<float>& qRope,
                        const vector<float>& kRope, const ForwardStats& stats)
{
    CHECK_RET(outputs.size() == 6, LOG_PRINT("[CHECK] FAIL: expected 6 outputs, got %zu\n", outputs.size());
              return FAILED);
    vector<float> dq;
    vector<float> dk;
    vector<float> dv;
    vector<float> dqRope;
    vector<float> dkRope;
    ComputeBackward(q, k, v, dy, qRope, kRope, stats, dq, dk, dv, dqRope, dkRope);
    const vector<int64_t> qShape = {kQTokens, kQHeads, kHeadDim};
    const vector<int64_t> kvShape = {kKvTokens, kKvHeads, kHeadDim};
    const vector<int64_t> qRopeShape = {kQTokens, kQHeads, kRopeDim};
    const vector<int64_t> kRopeShape = {kKvTokens, kKvHeads, kRopeDim};
    bool ok = true;
    ok = CheckOutput(outputs[0], "d_query", qShape, dq) && ok;
    ok = CheckOutput(outputs[1], "d_key", kvShape, dk) && ok;
    ok = CheckOutput(outputs[2], "d_value", kvShape, dv) && ok;
    ok = CheckOutput(outputs[3], "d_query_rope", qRopeShape, dqRope) && ok;
    ok = CheckOutput(outputs[4], "d_key_rope", kRopeShape, dkRope) && ok;
    CHECK_RET(outputs[5].GetSize() == 0, LOG_PRINT("[CHECK] d_sinks expected empty\n"); return FAILED);
    LOG_PRINT("[CHECK] total: %s\n", ok ? "PASS" : "FAIL");
    return ok ? SUCCESS : FAILED;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto status = GEInitialize(globalOptions);
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEInitialize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);

    Graph graph("sparse_flash_attention_grad_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    vector<float> q;
    vector<float> k;
    vector<float> v;
    vector<float> dy;
    vector<float> qRope;
    vector<float> kRope;
    ForwardStats stats;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps, q, k, v, dy, qRope, kRope, stats);
    if (ret == SUCCESS) {
        graph.SetInputs(inputOps).SetOutputs(outputOps);
        std::map<AscendString, AscendString> buildOptions;
        Session* session = new (std::nothrow) Session(buildOptions);
        CHECK_RET(session != nullptr, LOG_PRINT("[ERROR] Session allocation failed.\n"); return FAILED);
        constexpr uint32_t graphId = 0;
        std::map<AscendString, AscendString> graphOptions;
        auto addRet = session->AddGraph(graphId, graph, graphOptions);
        CHECK_RET(addRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::AddGraph failed, status=%u, error=%s\n", addRet, GetGeError().c_str());
                  delete session; return FAILED);
        vector<Tensor> outputTensors;
        auto runRet = session->RunGraph(graphId, inputTensors, outputTensors);
        CHECK_RET(runRet == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Session::RunGraph failed, status=%u, error=%s\n", runRet, GetGeError().c_str());
                  delete session; return FAILED);
        LOG_PRINT("SparseFlashAttentionGrad graph run success, output count: %zu\n", outputTensors.size());
        ret = ValidateOutputs(outputTensors, q, k, v, dy, qRope, kRope, stats);
        delete session;
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
```
