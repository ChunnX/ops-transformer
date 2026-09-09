# aclnnTurboQuantSparseFlashAttention

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/turbo_quant_sparse_flash_attention)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
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

面向DeepSeek Sparse Attention（DSA）推理场景，对TQ4量化存储的分页KV cache执行融合反量化与稀疏Attention计算。输入query包含512维latent与64维rope，输出为512维latent。K/V共享latent；rope仅参与QK计算。量化字节结构、scale定义及latent坐标域约定见[《算子说明》](../README.md#功能说明)。

接口按两阶段调用：先调用aclnnTurboQuantSparseFlashAttentionGetWorkspaceSize获取workspace大小和执行器，再调用aclnnTurboQuantSparseFlashAttention向指定流异步下发计算。

## 函数原型

头文件：`aclnnop/aclnn_turbo_quant_sparse_flash_attention.h`。

```cpp
aclnnStatus aclnnTurboQuantSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTable,
    const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKv,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    bool returnSoftmaxLse,
    const aclTensor *attentionOutOut,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnTurboQuantSparseFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnTurboQuantSparseFlashAttentionGetWorkspaceSize

### 参数说明

所有Tensor均采用ACL_FORMAT_ND。张量数据位于同一设备，输出空间由调用方分配。B表示batch数，T表示所有batch的query token总数，N表示query head数，Bn表示物理页数，Bs表示每页token数，K表示每个query的稀疏块索引槽位数，maxBlocks表示每个batch的页表容量。

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| query（const aclTensor*） | 输入 | 包含latent和rope的查询张量。 | 不可为空指针；支持T = 0的空Tensor；N为[1,128]内2的幂。 | FLOAT16、BFLOAT16 | ND | [T,N,576] | √ |
| key（const aclTensor*） | 输入 | 打包的分页KV cache。 | 不可为空指针或空Tensor；仅支持首轴非连续，详见约束说明。 | INT8 | ND | PA_BSND：[Bn,Bs,1,386]；PA_BNBD：[Bn,1,Bs,386] | √ |
| value（const aclTensor*） | 输入 | 与key共享latent的值张量。 | 传入与key相同的张量。 | 与key一致 | ND | 与key一致 | √ |
| sparseIndices（const aclTensor*） | 输入 | batch内局部稀疏块索引。 | 不可为空指针；T = 0时支持空Tensor；有效前缀后以-1填充。 | INT32 | ND | [T,1,K] | √ |
| keyDequantScaleOptional（const aclTensor*） | 可选输入 | key的独立反量化系数。 | 可传nullptr或空Tensor；当前COMBINE模式不读取其数值。 | FLOAT | ND | 不限制 | √ |
| valueDequantScaleOptional（const aclTensor*） | 可选输入 | value的独立反量化系数。 | 可传nullptr或空Tensor；当前COMBINE模式不读取其数值。 | FLOAT | ND | 不限制 | √ |
| blockTable（const aclTensor*） | 输入 | KV cache物理页映射表。 | 不可为空指针或空Tensor；页号必须有效。 | INT32 | ND | [B,maxBlocks] | √ |
| actualSeqLengthsQuery（const aclTensor*） | 输入 | 各batch的query长度累加和。 | 不可为空指针或空Tensor；非负、单调不减，末项等于T。 | INT32 | ND | [B] | √ |
| actualSeqLengthsKv（const aclTensor*） | 输入 | 各batch的有效KV长度。 | 不可为空指针或空Tensor；允许值为0；每项不超过对应页表行的容量。 | INT32 | ND | [B] | √ |
| scaleValue（double） | 输入 | QK分数的缩放系数。 | 通常按模型设置，如1/sqrt(512)。 | - | - | - | - |
| keyQuantMode（int64_t） | 输入 | key的量化模式。 | 仅支持3（TQ4）。 | - | - | - | - |
| valueQuantMode（int64_t） | 输入 | value的量化模式。 | 仅支持3（TQ4）。 | - | - | - | - |
| sparseBlockSize（int64_t） | 输入 | 每个稀疏索引对应的token数。 | 仅支持1、2、4、8、16，必须整除Bs。 | - | - | - | - |
| layoutQueryOptional（char*） | 输入 | query的逻辑布局。 | 传入TND；可传nullptr使用默认值TND。 | - | - | - | - |
| layoutKvOptional（char*） | 输入 | key和value的逻辑布局。 | 传入PA_BSND或PA_BNBD；可传nullptr使用默认值PA_BSND。 | - | - | - | - |
| sparseMode（int64_t） | 输入 | 稀疏注意力的mask模式。 | 0表示无因果mask，3表示右下对齐因果mask。 | - | - | - | - |
| preTokens（int64_t） | 输入 | 历史token窗口属性。 | 仅支持INT64_MAX。 | - | - | - | - |
| nextTokens（int64_t） | 输入 | 未来token窗口属性。 | 仅支持INT64_MAX。 | - | - | - | - |
| attentionMode（int64_t） | 输入 | Attention计算模式。 | 仅支持2（MLA-absorb）；本接口适用于DSA推理。 | - | - | - | - |
| quantScaleRepoMode（int64_t） | 输入 | 反量化系数存储模式。 | 仅支持1（COMBINE），scale位于KV slot尾部。 | - | - | - | - |
| tileSize（int64_t） | 输入 | 量化粒度属性。 | 仅支持128。 | - | - | - | - |
| ropeHeadDim（int64_t） | 输入 | rope分量维度。 | 仅支持64。 | - | - | - | - |
| returnSoftmaxLse（bool） | 输入 | 是否输出softmax_max和softmax_sum。 | true返回两个统计量，false返回两个空Tensor；不直接返回log-sum-exp。 | - | - | - | - |
| attentionOutOut（const aclTensor*） | 输出 | Attention计算结果。 | 不可为空指针；支持T = 0的空Tensor。 | 与query一致 | ND | [T,N,512] | × |
| softmaxMaxOut（const aclTensor*） | 输出 | softmax逐行最大值。 | 不可为空指针；returnSoftmaxLse为false时传入空Tensor描述。 | FLOAT | ND | true时[1,T,N]，false时[0] | × |
| softmaxSumOut（const aclTensor*） | 输出 | softmax逐行指数和。 | 不可为空指针；returnSoftmaxLse为false时传入空Tensor描述。 | FLOAT | ND | true时[1,T,N]，false时[0] | × |
| workspaceSize（uint64_t*） | 输出 | 第二阶段所需设备内存字节数。 | 不可为空指针。 | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 第二阶段使用的执行器。 | 不可为空指针；不可跨输入复用。 | - | - | - | - |

### 返回值

aclnnStatus：返回状态码，具体参见[《aclnn返回码》](../../../docs/zh/context/aclnn_return_code.md)。

第一阶段进行元数据校验，常见异常如下。设备上的索引和长度数值不属于此表的校验范围。

| 返回值 | 错误码 | 描述 |
| --- | --- | --- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选Tensor参数传入空指针，例如query为nullptr。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入或输出dtype组合不在支持范围内，例如query为FLOAT。 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | 输入元数据或属性不满足tiling约束，例如sparseBlockSize为3或layoutKvOptional为BNBD。 |

## aclnnTurboQuantSparseFlashAttention

### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| workspace | 输入 | Device侧workspace地址，按第一阶段结果分配；workspaceSize为0时可传nullptr。 |
| workspaceSize | 输入 | Device侧workspace字节数，使用第一阶段返回值。 |
| executor | 输入 | 第一阶段创建的执行器。 |
| stream | 输入 | 张量所在设备上由调用方管理的有效ACL流。 |

### 返回值

aclnnStatus：返回状态码，具体参见[《aclnn返回码》](../../../docs/zh/context/aclnn_return_code.md)。

两阶段返回码均需检查。第二阶段返回ACLNN_SUCCESS表示任务成功下发，异步错误还可能在流同步时报告。

## 约束说明

- 该接口仅支持推理前向。
- 确定性说明：aclnnTurboQuantSparseFlashAttention默认确定性实现。
- B、Bn和maxBlocks必须大于0；Bs为[16,1024]内16的倍数。KV head数固定为1，slot宽度固定为386字节。实际KV长度为0不代表支持空的物理KV cache。
- cache仅支持第0轴非连续，页内连续；多页时stride(0)至少为Bs × 386字节，不支持重叠页、负stride或页内非连续。单元素轴不约束stride。地址跨度不得超过INT64_MAX，调用方保证底层存储覆盖访问范围。
- 创建带偏移的aclTensor时传入真实strides、offset、底层storage地址及storage shape；不能将view数据地址与非零offset重复相加。三个输出Tensor必须连续且不与输入重叠；调用方应按此要求分配，接口不保证拦截非连续输出。
- sparseIndices中的索引i表示本batch中从i × sparseBlockSize开始的稀疏块，并非物理页号。有效索引必须位于每行前缀，无效后缀填-1；最后一个块可以只包含部分有效token。
- 调用方保证长度数组、索引和页表数值合法：query累计长度单调不减且末项等于T，KV长度非负且不超过对应页表容量，物理页号落在cache范围内。接口不读取这些设备数值做合法性检查。
- 某batch的query长度为0时不产生输出行；KV长度为0时该batch的attention输出为0，请求统计量时max为-2e38、sum为0。整体T = 0返回空输出，其余元数据仍须符合约束。
- KV长度非零但稀疏索引全为-1时，attention输出为0，max为-2e38，sum不保证为0。不能仅用sum判断行有效性。
- 输入、输出和workspace必须存活到流上任务完成，读取输出前需同步。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。本例使用BF16 query、PA_BSND cache，包含1个query token、8个query head和1个有效KV token。

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_sparse_flash_attention.h"

bool Check(int status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        std::cerr << operation << " failed: " << status << '\n';
        return false;
    }
    return true;
}

struct TensorData {
    void *data = nullptr;
    aclTensor *tensor = nullptr;
};

struct Resources {
    TensorData query, kv, indices, blockTable, queryLengths, kvLengths, output, maximum, sum;
    void *workspace = nullptr;
    aclrtStream stream = nullptr;
    bool initialized = false;
    bool deviceSet = false;

    ~Resources()
    {
        // 保证异步任务结束后再释放张量和 workspace；失败返回时也执行清理。
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto *item : {&query, &kv, &indices, &blockTable, &queryLengths, &kvLengths,
                           &output, &maximum, &sum}) {
            if (item->tensor != nullptr) {
                aclDestroyTensor(item->tensor);
            }
            if (item->data != nullptr) {
                aclrtFree(item->data);
            }
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

template <typename T>
bool CreateTensor(const std::vector<T> &host, const std::vector<int64_t> &shape,
                  aclDataType dtype, TensorData &result)
{
    const size_t bytes = host.size() * sizeof(T);
    if (!Check(aclrtMalloc(&result.data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc") ||
        !Check(aclrtMemcpy(result.data, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")) {
        return false;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    result.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                    ACL_FORMAT_ND, shape.data(), shape.size(), result.data);
    if (result.tensor == nullptr) {
        std::cerr << "aclCreateTensor failed\n";
        return false;
    }
    return true;
}

int main()
{
    // 1. 初始化设备与流。可通过 ASCEND_RT_VISIBLE_DEVICES 选择物理设备。
    Resources r;
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    r.initialized = true;
    if (!Check(aclrtSetDevice(0), "aclrtSetDevice")) {
        return 1;
    }
    r.deviceSet = true;
    if (!Check(aclrtCreateStream(&r.stream), "aclrtCreateStream")) {
        return 1;
    }

    // 2. 创建输入与输出。本例用 uint16_t 保存 BF16 位模式，query 为全零。
    // cache 为 [1,16,1,386]：latent 的 nibble 均为 code 8，rope 为零，FP16 scale 为 1。
    std::vector<uint8_t> cache(16 * 386, 0);
    for (size_t token = 0; token < 16; ++token) {
        std::fill_n(cache.data() + token * 386, 256, uint8_t{0x88});
        cache[token * 386 + 385] = 0x3c;
    }
    if (!CreateTensor(std::vector<uint16_t>(8 * 576, 0), {1, 8, 576}, ACL_BF16, r.query) ||
        !CreateTensor(cache, {1, 16, 1, 386}, ACL_INT8, r.kv) ||
        !CreateTensor(std::vector<int32_t>{0}, {1, 1, 1}, ACL_INT32, r.indices) ||
        !CreateTensor(std::vector<int32_t>{0}, {1, 1}, ACL_INT32, r.blockTable) ||
        !CreateTensor(std::vector<int32_t>{1}, {1}, ACL_INT32, r.queryLengths) ||
        !CreateTensor(std::vector<int32_t>{1}, {1}, ACL_INT32, r.kvLengths) ||
        !CreateTensor(std::vector<uint16_t>(8 * 512, 0), {1, 8, 512}, ACL_BF16, r.output) ||
        !CreateTensor(std::vector<float>(8, 0.0F), {1, 1, 8}, ACL_FLOAT, r.maximum) ||
        !CreateTensor(std::vector<float>(8, 0.0F), {1, 1, 8}, ACL_FLOAT, r.sum)) {
        return 1;
    }

    // 3. 获取 workspace 大小。key 与 value 传同一个张量，两个可选 scale 传 nullptr。
    char layoutQuery[] = "TND";
    char layoutKv[] = "PA_BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    if (!Check(aclnnTurboQuantSparseFlashAttentionGetWorkspaceSize(
                   r.query.tensor, r.kv.tensor, r.kv.tensor, r.indices.tensor, nullptr, nullptr,
                   r.blockTable.tensor, r.queryLengths.tensor, r.kvLengths.tensor,
                   1.0, 3, 3, 1, layoutQuery, layoutKv, 3, INT64_MAX, INT64_MAX, 2, 1, 128, 64,
                   true, r.output.tensor, r.maximum.tensor, r.sum.tensor, &workspaceSize, &executor),
               "aclnnTurboQuantSparseFlashAttentionGetWorkspaceSize")) {
        return 1;
    }
    if (workspaceSize != 0 &&
        !Check(aclrtMalloc(&r.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation")) {
        return 1;
    }

    // 4. 下发计算并同步。
    if (!Check(aclnnTurboQuantSparseFlashAttention(r.workspace, workspaceSize, executor, r.stream),
               "aclnnTurboQuantSparseFlashAttention") ||
        !Check(aclrtSynchronizeStream(r.stream), "aclrtSynchronizeStream")) {
        return 1;
    }

    // 5. 拷回结果，展示首个 attention 元素与首行 softmax 结果。
    std::vector<uint16_t> output(8 * 512);
    std::vector<float> maximum(8), sum(8);
    if (!Check(aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), r.output.data,
                          output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "attention D2H") ||
        !Check(aclrtMemcpy(maximum.data(), maximum.size() * sizeof(float), r.maximum.data,
                          maximum.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "maximum D2H") ||
        !Check(aclrtMemcpy(sum.data(), sum.size() * sizeof(float), r.sum.data,
                          sum.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), "sum D2H")) {
        return 1;
    }
    const uint32_t bits = static_cast<uint32_t>(output[0]) << 16;
    float first;
    std::memcpy(&first, &bits, sizeof(first));
    std::cout << "attention_out[0] = " << first << ", softmax_max[0] = " << maximum[0]
              << ", softmax_sum[0] = " << sum[0] << '\n';

    // 6. Resources 析构时释放张量、workspace、流及设备资源。
    return 0;
}
```
