# aclnnDispatchLayout

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

计算MoE分布式分发所需的布局信息，包括每个rank的token数量、每个expert的token数量、token到rank的映射关系以及发送token索引。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 "aclnnDispatchLayoutGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnDispatchLayout"接口执行计算。

```cpp
aclnnStatus aclnnDispatchLayoutGetWorkspaceSize(
    const aclTensor* topkIdx,
    int64_t num_tokens,
    int64_t num_ranks,
    int64_t num_experts,
    int64_t num_topk,
    int64_t local_ranksize,
    aclTensor* numTokensPerRank,
    aclTensor* numTokensPerExpert,
    aclTensor* isTokenInRank,
    aclTensor* notifySendData,
    aclTensor* sendTokenIdxSmall,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnDispatchLayout(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnDispatchLayoutGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1600px"> <colgroup>
    <col style="width: 170px">
    <col style="width: 140px">
    <col style="width: 350px">
    <col style="width: 350px">
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 150px">
    <col style="width: 150px">
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
    </tr></thead>
    <tbody>
    <tr>
    <td>topkIdx(aclTensor*)</td>
    <td>输入</td>
    <td>每个token选择的top-k expert索引。</td>
    <td>无</td>
    <td>INT64</td>
    <td>ND</td>
    <td>[num_tokens, num_topk]</td>
    <td>√</td>
    </tr>
    <tr>
    <td>num_tokens(int64_t)</td>
    <td>输入</td>
    <td>token数量。</td>
    <td>取值范围：A2场景[1,4096]，A3场景[1,8000]</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>num_ranks(int64_t)</td>
    <td>输入</td>
    <td>分布式通信域中rank数量。</td>
    <td>取值范围：[1,384]</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>num_experts(int64_t)</td>
    <td>输入</td>
    <td>expert总数。</td>
    <td>取值范围：[1,512]，且必须能被num_ranks整除</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>num_topk(int64_t)</td>
    <td>输入</td>
    <td>每个token选择的top-k数量。</td>
    <td>取值范围：[1,16]</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>local_ranksize(int64_t)</td>
    <td>输入</td>
    <td>多机场景下的单机rank数量。</td>
    <td>取值范围：[1,8]，且必须是num_ranks的整数倍</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numTokensPerRank(aclTensor*)</td>
    <td>输出</td>
    <td>每个rank接收的token数量。</td>
    <td>无</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[num_ranks]</td>
    <td>√</td>
    </tr>
    <tr>
    <td>numTokensPerExpert(aclTensor*)</td>
    <td>输出</td>
    <td>每个expert处理的token数量。</td>
    <td>无</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[num_experts]</td>
    <td>√</td>
    </tr>
    <tr>
    <td>isTokenInRank(aclTensor*)</td>
    <td>输出</td>
    <td>token到rank的映射矩阵，值为1表示该token会被发送到对应rank。</td>
    <td>无</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[num_tokens, num_ranks]</td>
    <td>√</td>
    </tr>
    <tr>
    <td>notifySendData(aclTensor*)</td>
    <td>输出</td>
    <td>临时缓冲区，用于通知发送数据。</td>
    <td>无</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[num_tokens, num_experts]</td>
    <td>√</td>
    </tr>
    <tr>
    <td>sendTokenIdxSmall(aclTensor*)</td>
    <td>输出</td>
    <td>发送token索引。</td>
    <td>无</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[num_tokens, num_topk]</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize(uint64_t*)</td>
    <td>输出</td>
    <td>返回workspace大小。</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>
    
- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>输入的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>参数的取值不在支持的范围内，或数据类型不支持。</td>
    </tr>
    </tbody></table>

## aclnnDispatchLayout

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnDispatchLayoutGetWorkspaceSize</code>获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnDispatchLayout默认确定性实现。

- 数据类型约束：
  - topkIdx必须是INT64类型。
  - 所有输出Tensor必须是INT32类型。

- 参数约束：
  - num_experts必须能被num_ranks整除。
  - A2多机场景下，num_tokens不能超过4096。
  - A3场景下，num_tokens不能超过8000。
  - num_topk不能超过16。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dispatch_layout.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1. 初始化ACL
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

    // 2. 设置Device
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);

    // 3. 创建Context和Stream
    aclrtContext context;
    ret = aclrtCreateContext(&context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);

    // 4. 准备输入数据
    int64_t num_tokens = 128;
    int64_t num_ranks = 8;
    int64_t num_experts = 64;
    int64_t num_topk = 2;
    int64_t local_ranksize = 8;

    std::vector<int64_t> topkIdxShape{num_tokens, num_topk};
    std::vector<int64_t> numTokensPerRankShape{num_ranks};
    std::vector<int64_t> numTokensPerExpertShape{num_experts};
    std::vector<int64_t> isTokenInRankShape{num_tokens, num_ranks};
    std::vector<int64_t> notifySendDataShape{num_tokens, num_experts};
    std::vector<int64_t> sendTokenIdxSmallShape{num_tokens, num_topk};

    // 初始化topkIdx数据，假设每个token选择expert 0和1
    std::vector<int64_t> topkIdxHostData;
    for (int64_t i = 0; i < num_tokens; i++) {
        topkIdxHostData.push_back(0);
        topkIdxHostData.push_back(1);
    }

    void *topkIdxDeviceAddr = nullptr;
    void *numTokensPerRankDeviceAddr = nullptr;
    void *numTokensPerExpertDeviceAddr = nullptr;
    void *isTokenInRankDeviceAddr = nullptr;
    void *notifySendDataDeviceAddr = nullptr;
    void *sendTokenIdxSmallDeviceAddr = nullptr;

    aclTensor *topkIdx = nullptr;
    aclTensor *numTokensPerRank = nullptr;
    aclTensor *numTokensPerExpert = nullptr;
    aclTensor *isTokenInRank = nullptr;
    aclTensor *notifySendData = nullptr;
    aclTensor *sendTokenIdxSmall = nullptr;

    // 创建输入Tensor
    ret = CreateAclTensor(topkIdxHostData, topkIdxShape, &topkIdxDeviceAddr, aclDataType::ACL_INT64, &topkIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出Tensor
    std::vector<int32_t> zeroData(num_ranks, 0);
    ret = CreateAclTensor(zeroData, numTokensPerRankShape, &numTokensPerRankDeviceAddr, aclDataType::ACL_INT32, &numTokensPerRank);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    zeroData.resize(num_experts);
    ret = CreateAclTensor(zeroData, numTokensPerExpertShape, &numTokensPerExpertDeviceAddr, aclDataType::ACL_INT32, &numTokensPerExpert);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    zeroData.resize(num_tokens * num_ranks);
    ret = CreateAclTensor(zeroData, isTokenInRankShape, &isTokenInRankDeviceAddr, aclDataType::ACL_INT32, &isTokenInRank);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    zeroData.resize(num_tokens * num_experts);
    ret = CreateAclTensor(zeroData, notifySendDataShape, &notifySendDataDeviceAddr, aclDataType::ACL_INT32, &notifySendData);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    zeroData.resize(num_tokens * num_topk);
    ret = CreateAclTensor(zeroData, sendTokenIdxSmallShape, &sendTokenIdxSmallDeviceAddr, aclDataType::ACL_INT32, &sendTokenIdxSmall);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnDispatchLayoutGetWorkspaceSize(topkIdx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize,
                                               numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData,
                                               sendTokenIdxSmall, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnDispatchLayoutGetWorkspaceSize failed. ret = %d\n", ret); return ret);

    // 6. 申请workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d\n", ret); return ret);
    }

    // 7. 调用第二段接口
    ret = aclnnDispatchLayout(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnDispatchLayout failed. ret = %d\n", ret); return ret);

    // 8. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret = %d\n", ret); return ret);

    LOG_PRINT("[INFO] aclnnDispatchLayout execute successfully.\n");

    // 9. 释放资源
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(topkIdx);
    aclDestroyTensor(numTokensPerRank);
    aclDestroyTensor(numTokensPerExpert);
    aclDestroyTensor(isTokenInRank);
    aclDestroyTensor(notifySendData);
    aclDestroyTensor(sendTokenIdxSmall);
    aclrtFree(topkIdxDeviceAddr);
    aclrtFree(numTokensPerRankDeviceAddr);
    aclrtFree(numTokensPerExpertDeviceAddr);
    aclrtFree(isTokenInRankDeviceAddr);
    aclrtFree(notifySendDataDeviceAddr);
    aclrtFree(sendTokenIdxSmallDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```