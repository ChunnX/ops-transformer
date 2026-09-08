# aclnnTurboQuantSparseFlashAttention

## 支持的产品

Atlas A2 训练/推理系列、Atlas A3 训练/推理系列。支持前向推理。

## 功能说明

面向 DeepSeek Sparse Attention（DSA）推理场景，对 TQ4 量化存储的分页 KV cache 执行融合反量化与稀疏 Attention 计算。输入 query 包含 512 维 latent 与 64 维 rope，输出为 512 维 latent。K/V 共享 latent；rope 仅参与 QK 计算。量化存储、布局、数值约束见[算子说明](../README.md)。

接口按两阶段调用：第一阶段校验输入元数据、创建执行器并计算 workspace 大小，第二阶段向调用方提供的流异步下发任务。

## 接口原型

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

## 参数说明

所有 tensor 均采用 `ACL_FORMAT_ND`。除两个可选 scale 外，tensor 指针不可为空。张量数据需在与 `stream` 相同的设备上，输出空间由调用方分配。

| 参数 | 类型、shape 与约束 |
| --- | --- |
| query | FLOAT16/BFLOAT16，`[T, N, 576]`；`N` 为 `[1,128]` 内 2 的幂 |
| key、value | INT8；PA_BSND 为 `[Bn, Bs, 1, 386]`，PA_BNBD 为 `[Bn, 1, Bs, 386]`；value 请传入与 key 相同的张量 |
| sparseIndices | INT32，`[T, 1, K]`；值为 batch 内局部稀疏块索引，有效前缀之后用 `-1` 填充 |
| keyDequantScaleOptional、valueDequantScaleOptional | 可为 `nullptr`；传入时为 FLOAT32。当前 COMBINE 模式不读取其数值 |
| blockTable | INT32，`[B, maxBlocks]`；每个有效值为 cache 的物理页号 |
| actualSeqLengthsQuery | INT32，`[B]`，各 batch query 长度的累加和，非负单调不减，末项等于 `T` |
| actualSeqLengthsKv | INT32，`[B]`，每个 batch 的实际 KV 长度，非负且不超过对应表行容量 |
| scaleValue | QK 分数缩放系数，常取 `1/sqrt(512)` |
| keyQuantMode、valueQuantMode | 均为 `3` |
| sparseBlockSize | `1,2,4,8,16` 之一，必须整除 `Bs` |
| layoutQueryOptional | 传入 `"TND"` |
| layoutKvOptional | 传入 `"PA_BSND"` 或 `"PA_BNBD"`，与输入轴顺序一致；无 PA_BNSD/BNBD 别名 |
| sparseMode | `0` 无因果 mask，`3` 右下对齐因果 mask |
| preTokens、nextTokens | 均为 `INT64_MAX` |
| attentionMode、quantScaleRepoMode | 分别为 `2`、`1` |
| tileSize、ropeHeadDim | 分别为 `128`、`64` |
| returnSoftmaxLse | 是否返回两个 softmax 中间结果，并非直接返回 log-sum-exp |
| attentionOutOut | `[T, N, 512]`，与 query 同型 |
| softmaxMaxOut、softmaxSumOut | FLOAT32；请求时为 `[1,T,N]`，否则为 `[0]`，仍需传入 tensor 描述 |
| workspaceSize（第一阶段输出） | 第二阶段所需设备内存字节数 |
| executor（第一阶段输出） | 第二阶段使用的执行器，不可跨输入复用 |
| workspace、workspaceSize（第二阶段） | 按第一阶段结果分配，大小为 0 时 workspace 可为 `nullptr` |
| stream | 由调用方管理的有效 ACL 流 |

`B`、`Bn` 和 `maxBlocks` 必须大于 0，`Bs` 是 `[16,1024]` 内 16 的倍数。KV slot 为 256 字节 nibble、128 字节与 query 同型的 `rope/scale`、2 字节 FLOAT16 scale；低 nibble 对应偶数维。

cache 支持第 0 轴非连续与合法 storage offset，页内必须连续，多页 stride 至少为 `Bs*386` 字节。创建 `aclTensor` 时传入真实 strides、offset、底层 storage 地址和 storage shape；不要把 view 数据地址与非零 offset 重复相加。调用方保证底层存储覆盖访问范围，输出不与输入重叠。

某个 batch 的 query 长度为 0 时不产生输出行；KV 长度为 0 时该 batch 输出为 0，softmax max 为 `-2e38`、sum 为 0。整体 `T=0` 返回空输出，元数据和物理 cache 仍须满足接口约束。非空 KV 但索引全为 `-1` 时，sum 不保证为 0，详见算子说明。

接口不校验长度、索引及页表中的设备数值，调用方须保证其合法性。输入、输出和 workspace 必须存活到流上任务完成，读取输出前需正确同步。

## 返回值

`ACLNN_SUCCESS` 表示当前阶段成功；其他值表示参数、执行环境或算子执行错误，具体原因见 ACLNN 错误日志。两阶段返回码均需检查；第二阶段返回成功只表示任务成功下发，异步错误还可能在流同步时报告。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。本例使用 BF16 query、PA_BSND cache，包含 1 个 query token、8 个 query head 和 1 个有效 KV token。

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
