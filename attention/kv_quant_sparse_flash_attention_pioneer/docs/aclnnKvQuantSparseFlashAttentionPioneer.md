# aclnnKvQuantSparseFlashAttentionPioneer

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/kv_quant_sparse_flash_attention_pioneer)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |

## 功能说明

- API功能：kvQuantSparseFlashAttentionPioneer在sparseFlashAttention的基础上支持了Per-Token-Head-Tile-128量化输入。引入param sink后，param sink加入到KV的首个基本快中进行Attention计算。

- 计算公式：
    $$
    \tilde{K}=\text{Gather}({Key};{SparseIndice})
    $$

    $$
    \tilde{K}_{\rm Nope}=\text{deQuant}_{8 \rightarrow 16}(\tilde{K}{...,:512};key\_scale)
    $$

    $$
    \tilde{K}_{\rm Rope}=\tilde{K}[...,512:576]
    $$

    $$
    \tilde{K}={\rm concat}[{\tilde{K}_{\rm Nope},\tilde{K}_{\rm Rope}}]
    $$

    $$
    \tilde{K}={\rm concat}\left[key\_sink,\tilde{K}\right]
    $$

    $$
    Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。
    
    本次公布的`kv_quant_sparse_flash_attention_pioneer`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 函数原型

```Cpp
aclnnStatus aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *keySinkOptional,
    const aclTensor *valueSinkOptional,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    const char *layoutQueryOptional,
    const char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnKvQuantSparseFlashAttentionPioneer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```
## aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize

- **参数说明：**

> [!NOTE]
> - query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N表示num\_query\_heads，KV\_N表示num\_key\_value\_heads，Q\_T表示query shape中的T，KV\_T表示key shape中的T。

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 240px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 233px">
  <col style="width: 119px">
  <col style="width: 215px">
  <col style="width: 100px">
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
      <td>表示attention结构的Q输入。</td>
      <td>不支持空tensor。</td>
      <td>bfloat16、float16。</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时shape为[B,S1,Q_N,D]。</li>
                <li>layout_query为TND时，shape为[Q_T,Q_N,D]，其中Q_N支持64/48/32/8/6/4/3/2。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>表示attention结构的K输入。</td>
      <td>不支持空tensor。</td>
      <td>数据类型支持`float8_e4m3fn、hifloat8`。</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为PA_BSND时，shape为[block_num, block_size, KV_N, D]。</li>
                <li>layout_kv为BSND时，shape为[B, S2, KV_N, D]。</li>
                <li>layout_kv为TND时，shape为[KV_T, KV_N, D]，其中KV_N只支持1。</li>
          </ul>
      </td>
      <td>✓</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>表示attention结构的V输入。</td>
      <td>不支持空tensor。</td>
      <td>数据类型支持`float8_e4m3fn、hifloat8`。</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为PA_BSND时，shape为[block_num, block_size, KV_N, D]。</li>
                <li>layout_kv为BSND时，shape为[B, S2, KV_N, D]。</li>
                <li>layout_kv为TND时，shape为[KV_T, KV_N, D]，其中KV_N只支持1。</li>
          </ul>
      </td>
      <td>✓</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>输入</td>
      <td>代表离散取kvCache的索引。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse_size大于0。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape需要传入[B, Q_S, KV_N, sparse_size]。</li>
                <li>layout_query为TND时，shape需要传入[Q_T, KV_N, sparse_size]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyDequantScaleOptional</td>
      <td>输入</td>
      <td>表示key的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>float32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为PA_BSND时，shape为[block_num, block_size, KV_N]。</li>
                <li>layout_kv为BSND时，shape为[B, S2, KV_N]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>valueDequantScaleOptional</td>
      <td>输入</td>
      <td>表示value的反量化系数。</td>
      <td>不支持空tensor。</td>
      <td>float32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_kv为PA_BSND时，shape为[block_num, block_size, KV_N]。</li>
                <li>layout_kv为BSND时，shape为[B, S2, KV_N]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示PageAttention中kvCache存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度为B，第二维长度不小于所有batch中最大的s2对应的block数量，即s2_max / block_size向上取整。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>shape支持(B,S2/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQueryOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKvOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的S长度相同。</li>
                <li> 该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当`layout_kv`为TND或PA_BSND时，该入参必须传入，`layout_kv`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>keySinkOptional</td>
      <td>输入</td>
      <td>表示添加在压缩key的序列维度上的额外参数。</td>
      <td>不支持空tensor。</td>
      <td>bfloat16、float16</td>
      <td>ND</td>
      <td>[sink_num, KV_N, D]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>valueSinkOptional</td>
      <td>输入</td>
      <td>表示添加在压缩value的序列维度上的额外参数。</td>
      <td>不支持空tensor。</td>
      <td>bfloat16、float16</td>
      <td>ND</td>
      <td>[sink_num, KV_N, D]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>输入</td>
      <td>代表缩放系数。</td>
      <td>作为query和key矩阵乘后Muls的scalar值。</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keyQuantMode</td>
      <td>输入</td>
      <td>用于标识输入key的量化模式。</td>
      <td>仅支持传入2，代表per_tile量化模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueQuantMode</td>
      <td>输入</td>
      <td>用于标识输入value的量化模式。</td>
      <td>仅支持传入2，代表per_tile量化模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseBlockSize</td>
      <td>输入</td>
      <td>代表sparse阶段的block大小。</td>
      <td>在计算importance score时使用，支持范围为[1, 16]，且为2的幂次方。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQueryOptional</td>
      <td>输入</td>
      <td>用于标识输入Query的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>当前支持BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional</td>
      <td>输入</td>
      <td>用于标识输入Key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>当前支持PA_BSND、BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>sparse_mode为0时，代表defaultMask模式。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attentionMode</td>
      <td>输入</td>
      <td>表示attention的模式。</td>
      <td>仅支持传入2，表示MLA-absorb模式。即QK的D包含rope和nope两部分，且KV是同一份。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScaleRepoMode</td>
      <td>输入</td>
      <td>表示量化参数的存放模式。</td>
      <td>仅支持传入1，表示combine模式，即量化参数和数据混合存放。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSize</td>
      <td>输入</td>
      <td>表示per_tile时每个参数对应的数据块大小，仅在per_tile时有效。</td>
      <td>仅支持默认值128。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ropeHeadDim</td>
      <td>输入</td>
      <td>表示MLA架构下的ropeHeadDim大小。</td>
      <td>仅在attention\_mode为2时有效，仅支持默认值64。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示attention输出。</td>
      <td>不支持空tensor。</td>
      <td>bfloat16、float16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为"BSND"时输出shape为[B, S1, N1, D]。</li>
                <li>layout_query为"TND"时输出shape为[T1, N1, D]。</li>
          </ul>
      </td>
      <td>x</td>
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

  第一段接口会完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>query、key、value、sparseIndices、keyDequantScaleOptional、valueDequantScaleOptional、actualSeqLengthsQueryOptional、actualSeqLengthsKvOptional、keySinkOptional、valueSinkOptional、scaleValue、keyQuantMode、valueQuantMode、layoutQueryOptional、layoutKvOptional、sparseMode、attentionMode、quantScaleRepoMode、out的数据类型和数据格式不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnKvQuantSparseFlashAttentionPioneer

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 参数query中的N值为64/48/32/8/6/4/3/2，key、value的N支持1。
- 参数query中的D值为576，即nope+rope=512+64。
- 参数key、value中的D值为656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。
- 支持sparse_block_size整除block_size。
- 当前版本sink_num维仅支持128。
- layout_query为TND或BSND，sink场景layout_kv仅支持PA_BSND。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
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
 * \file test_kv_quant_sparse_flash_attention_pioneer.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_quant_sparse_flash_attention_pioneer.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* keyDequantScaleDeviceAddr = nullptr;
    void* valueDequantScaleDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* keyDequantScaleTensor = nullptr;
    aclTensor* valueDequantScaleTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;

};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 1, 4, 576};
    std::vector<int64_t> keyShape = {1, 1, 4, 656};
    std::vector<int64_t> valueShape = {1, 1, 4, 656};
    std::vector<int64_t> sparseIndicesShape = {1, 1, 4, 2048};
    std::vector<int64_t> keyDequantScaleShape = {1, 1, 4};
    std::vector<int64_t> valueDequantScaleShape = {1, 1, 4};
    std::vector<int64_t> attentionOutShape = {1, 1, 4, 512};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t keyDequantScaleShapeSize = GetShapeSize(keyDequantScaleShape);
    int64_t valueDequantScaleShapeSize = GetShapeSize(valueDequantScaleShape);
    int64_t attentionOutShapeSize = GetShapeSize(attentionOutShape);

    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> keyHostData(keyShapeSize, 1);
    std::vector<float> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<int32_t> keyDequantScaleHostData(keyDequantScaleShapeSize, 1);
    std::vector<int32_t> valueDequantScaleHostData(valueDequantScaleShapeSize, 1);
    std::vector<float> attentionOutHostData(attentionOutShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyDequantScaleHostData, keyDequantScaleShape, &resources.keyDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.keyDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueDequantScaleHostData, valueDequantScaleShape, &resources.valueDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.valueDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &resources.attentionOutDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteKvQuantSparseFlashAttentionPioneer(TensorResources& resources, aclrtStream stream,
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t keyQuantMode = 0;
    int64_t valueQuantMode = 0;
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    double scaleValue = 1.0/24.0;
    int64_t sparseBlockSize = 1;
    int64_t attentionMode = 2;
    int64_t quantScaleRepoMode = 1;
    int64_t tileSize = 128;
    int64_t ropeHeadDim = 64;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKey[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.valueTensor, resources.sparseIndicesTensor,
                        resources.keyDequantScaleTensor, resources.valueDequantScaleTensor, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keyQuantMode,
                        valueQuantMode, sparseBlockSize, layoutQuery, layoutKey, sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode,
                        tileSize, ropeHeadDim, resources.attentionOutTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnKvQuantSparseFlashAttentionPioneer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnKvQuantSparseFlashAttentionPioneer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.keyDequantScaleTensor) {
      aclDestroyTensor(resources.keyDequantScaleTensor);
    }
    if (resources.valueDequantScaleTensor) {
      aclDestroyTensor(resources.valueDequantScaleTensor);
    }
    if (resources.attentionOutTensor) {
      aclDestroyTensor(resources.attentionOutTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.keyDequantScaleDeviceAddr) {
      aclrtFree(resources.keyDequantScaleDeviceAddr);
    }
    if (resources.valueDequantScaleDeviceAddr) {
      aclrtFree(resources.valueDequantScaleDeviceAddr);
    }
    if (resources.attentionOutDeviceAddr) {
      aclrtFree(resources.attentionOutDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> attentionOutShape = {1, 1, 4, 512};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteKvQuantSparseFlashAttentionPioneer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    PrintOutResult(attentionOutShape, &resources.attentionOutDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
