# aclnnMoeEpDispatch

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_ep_dispatch)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>     |    √     |

## 功能说明

- 接口功能：需与aclnnMoeEpCombine和aclnnMoeEpDispatchEpilogue配套使用，完成MoE的Expert Parallelism（EP）并行部署下的token dispatch第一阶段。该接口根据每个token的topK专家索引，进行slot分配和EP域的alltoallv通信，输出中间结果供epilogue步骤处理。
  - 支持cached模式，即第二次dispatch时可复用第一次的handle，跳过slot分配阶段；
  - 支持CPU同步等待（doCpuSync）。

- 计算公式：

    - 非cached模式：

    $$alltoallXOut = AllToAllV(X)$$

    $$dstBufferSlotIdx = SlotAssignment(topkIdx)$$

    $$recvSrcMetadataInternal = MetadataAssignment(alltoallXOut, topkIdx)$$

    - cached模式：

    $$alltoallXOut = AllToAllV(X)$$

> 注意该接口必须与aclnnMoeEpCombine和aclnnMoeEpDispatchEpilogue配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnMoeEpDispatchGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMoeEpDispatch"接口执行计算。

```cpp
aclnnStatus aclnnMoeEpDispatchGetWorkspaceSize(
    const aclTensor*  context,
    const aclTensor*  x,
    const aclTensor*  topkIdx,
    const aclTensor*  topkWeights,
    const aclTensor*  cachedHandleDstBufferSlotIdx,
    int64_t           epWorldSize,
    int64_t           epRankId,
    int64_t           numExperts,
    int64_t           numMaxTokensPerRank,
    int64_t           cclBufferSize,
    int64_t           expertAlignment,
    bool              doCpuSync,
    aclTensor*        numRecvTokensPerRankHostPinned,
    aclTensor*        numRecvTokensPerExpertHostPinned,
    aclTensor*        dstBufferSlotIdx,
    aclTensor*        recvSrcMetadataInternal,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```cpp
aclnnStatus aclnnMoeEpDispatch(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeEpDispatchGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>context（aclTensor*）</td>
    <td>输入</td>
    <td>HCCL通信域上下文。</td>
    <td>-</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>x（aclTensor*）</td>
    <td>输入</td>
    <td>表示计算使用的token数据，需根据topkIdx来发送给其他卡，对应公式中X。</td>
    <td><ul><li>不支持空Tensor。</li><li>数据类型仅支持BFLOAT16、FLOAT16。</li></ul></td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>topkIdx（aclTensor*）</td>
    <td>输入</td>
    <td>表示每个token的topK个专家索引，决定每个token要发给哪些专家，对应公式中topkIdx。</td>
    <td><ul><li>Optional参数，cached模式下传入空指针。</li><li>非cached模式下必选。</li><li>张量里value取值范围为[0, numExperts)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>topkWeights（aclTensor*）</td>
    <td>输入</td>
    <td>表示每个token对应的topK专家权重。</td>
    <td><ul><li>Optional参数，cached模式下传入空指针。</li><li>非cached模式下必选。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>cachedHandleDstBufferSlotIdx（aclTensor*）</td>
    <td>输入</td>
    <td>表示cached模式下复用的slot索引张量。</td>
    <td><ul><li>Optional参数，非cached模式下传入空指针。</li><li>cached模式下必选。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>epWorldSize（int64_t）</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td><ul><li>取值范围[2, 768]。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId（int64_t）</td>
    <td>输入</td>
    <td>EP域中的rank编号。</td>
    <td><ul><li>取值范围[0, epWorldSize)。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numExperts（int64_t）</td>
    <td>输入</td>
    <td>MoE专家总数量。</td>
    <td><ul><li>取值范围[1, 2048]，且满足numExperts % epWorldSize = 0。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numMaxTokensPerRank（int64_t）</td>
    <td>输入</td>
    <td>每张卡上的最大token数量上限。</td>
    <td><ul><li>必须大于0。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>cclBufferSize（int64_t）</td>
    <td>输入</td>
    <td>HCCL通信域缓存区大小（字节）。</td>
    <td><ul><li>必须大于0。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertAlignment（int64_t）</td>
    <td>输入</td>
    <td>专家对齐数。</td>
    <td><ul><li>Optional参数，默认为1。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>doCpuSync（bool）</td>
    <td>输入</td>
    <td>是否进行CPU同步等待。</td>
    <td><ul><li>Optional参数，非cached模式默认为true，cached模式必须为false。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numRecvTokensPerRankHostPinned（aclTensor*）</td>
    <td>输出</td>
    <td>各卡接收token数量的host pinned内存张量。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(epWorldSize,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>numRecvTokensPerExpertHostPinned（aclTensor*）</td>
    <td>输出</td>
    <td>每个本地专家接收token数量的host pinned内存张量。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(numLocalExperts,)。</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>dstBufferSlotIdx（aclTensor*）</td>
    <td>输出</td>
    <td>slot索引张量，供aclnnMoeEpDispatchEpilogue和aclnnMoeEpCombine使用。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(BS, K)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>recvSrcMetadataInternal（aclTensor*）</td>
    <td>输出</td>
    <td>内部接收元数据，供aclnnMoeEpDispatchEpilogue使用。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(A, 3)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize（uint64_t*）</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor（aclOpExecutor**）</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1000px"> <colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
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
    <td>context、x等必选参数存在空指针。</td>
    </tr>
    <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>x的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
    <td>x的shape维度不为2。</td>
    </tr>
    <tr>
    <td>numExperts不能被epWorldSize整除。</td>
    </tr>
    </tbody></table>

## aclnnMoeEpDispatch

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1000px"> <colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
    </colgroup>
    <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMoeEpDispatchGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnMoeEpDispatch默认确定性实现。
- 该接口必须与aclnnMoeEpCombine和aclnnMoeEpDispatchEpilogue配套使用。
- cached模式下，topkIdx和topkWeights必须为空指针，cachedHandleDstBufferSlotIdx为必选参数，doCpuSync必须为false；非cached模式下，topkIdx和topkWeights为必选参数，cachedHandleDstBufferSlotIdx必须为空指针。
- 调用接口过程中使用的numExperts、numMaxTokensPerRank参数取值所有卡需保持一致，且和aclnnMoeEpCombine对应参数也保持一致。
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，A = epWorldSize \* numMaxTokensPerRank。
    - H：表示hidden size隐藏层大小。取值为[1024, 8192]。
    - BS：表示batch sequence size，即本卡的token数量。
    - K：表示选取topK个专家，取值范围为1 < K ≤ 32。
    - numLocalExperts：表示本卡专家数量，numLocalExperts = numExperts / epWorldSize，应满足0 < numLocalExperts \* epWorldSize ≤ 2048。
- HCCL通信域缓存区大小：
    调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
    - 通信域缓存区大小可通过调用MoEElasticBuffer.get_moe_ep_ccl_buffer_size接口计算。
    - 设置大小要求 >= 2 \* (epWorldSize \* numMaxTokensPerRank \* Align512(Align32(2 \* H)))。
    - 其中 Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。
- 通信域使用约束：
    - 一个模型中的aclnnMoeEpDispatch、aclnnMoeEpDispatchEpilogue和aclnnMoeEpCombine算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
- <term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>：仅支持BFLOAT16数据类型的x输入，FLOAT16暂不支持。

## 调用示例

<term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>：

- 环境变量配置：

    ```bash
    # 运行前需设置环境变量ENV_DEV_NUM，根据当前机器的卡数设置该变量，以2卡为例设置为2
    export ENV_DEV_NUM=2
    ```

- 算子编译执行：
    在所有机器上编译算子，算子编译命令如下，moe_ep_dispatch、moe_ep_dispatch_epilogue和moe_ep_combine算子都需要编译，这三个算子需要配套执行：
    ```bash
    bash build.sh --pkg --soc=ascend910_93 --ops=moe_ep_dispatch,moe_ep_dispatch_epilogue,moe_ep_combine
    ```

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

> **说明**：本示例三算子入参中的`context`为HCCL通信上下文tensor，需在调用前通过`GetMc2Context`创建。该函数内部涉及HCCL KFC（Kernel-level Fast Communication）API动态加载、通信域上下文创建、各卡HCCL Buffer地址收集等步骤，完整实现请参考torch_extension源码[`update_context.cpp`](../../../torch_extension/npu_ops_transformer/ops/csrc/update_context.cpp)。为突出算子调用流程，示例中将`GetMc2Context`作为外部utility函数调用，不展开其内部实现。

```Cpp
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_moe_ep_dispatch.h"
#include "aclnnop/aclnn_moe_ep_dispatch_epilogue.h"
#include "aclnnop/aclnn_moe_ep_combine.h"

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

struct Args {
    uint32_t rankId;
    HcclComm hcclEpComm;
    aclrtStream dispatchStream;
    aclrtStream combineStream;
    aclrtContext context;
};

constexpr uint32_t EP_WORLD_SIZE = 2;
constexpr uint32_t DEV_NUM = EP_WORLD_SIZE;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void DestroyTensor(aclTensor *tensor) {
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
}

void FreeDeviceAddr(void *deviceAddr) {
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

// GetMc2Context: 创建HCCL通信上下文，返回context INT32 tensor和cclBufferSize。
// 内部涉及HCCL KFC API动态加载（dlopen/dlsym）、通信域OpResCtx创建、各卡Buffer地址收集，
// 完整实现参考 torch_extension/npu_ops_transformer/ops/csrc/update_context.cpp
extern int32_t GetMc2Context(void *mc2ContextHost, int64_t epWorldSize, int64_t &cclBufferSize, const char *groupEpStr);
extern int CreateContextTensor(void *mc2ContextHost, void **contextDeviceAddr, aclTensor **contextTensor);

int LaunchOneProcessDispatchEpilogueCombine(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank=%d, hcomEpName=%s, dispatchStream=%p, combineStream=%p, context=%p\n",
        args.rankId, hcomEpName, args.dispatchStream, args.combineStream, args.context);

    // 创建context通信上下文tensor
    void *mc2ContextHost = nullptr;
    int64_t cclBufferSize = 0;
    int32_t contextRet = GetMc2Context(mc2ContextHost, EP_WORLD_SIZE, cclBufferSize, hcomEpName);
    CHECK_RET(contextRet == 0, LOG_PRINT("[ERROR] GetMc2Context failed\n"); return -1);

    void *contextDeviceAddr = nullptr;
    aclTensor *contextTensor = nullptr;
    ret = CreateContextTensor(mc2ContextHost, &contextDeviceAddr, &contextTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 设置场景参数
    int64_t BS = 8;
    int64_t H = 4096;
    int64_t K = 8;
    int64_t numExperts = EP_WORLD_SIZE * 4;
    int64_t numMaxTokensPerRank = 128;
    int64_t expertAlignment = 1;
    bool doCpuSync = true;
    int64_t numLocalExperts = numExperts / EP_WORLD_SIZE;
    int64_t upperA = EP_WORLD_SIZE * numMaxTokensPerRank;

    // 声明device侧变量
    void *xDeviceAddr = nullptr;
    void *topkIdxDeviceAddr = nullptr;
    void *topkWeightsDeviceAddr = nullptr;
    void *numRecvPerRankHpDeviceAddr = nullptr;
    void *numRecvPerExpertHpDeviceAddr = nullptr;
    void *dstSlotDeviceAddr = nullptr;
    void *recvSrcMetaInternalDeviceAddr = nullptr;
    void *recvXDeviceAddr = nullptr;
    void *recvTopkIdxDeviceAddr = nullptr;
    void *recvTopkWeightsDeviceAddr = nullptr;
    void *recvSrcMetaDeviceAddr = nullptr;
    void *numRecvPerRankDeviceAddr = nullptr;
    void *numRecvPerExpertDeviceAddr = nullptr;
    void *combinedXDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *topkIdx = nullptr;
    aclTensor *topkWeights = nullptr;
    aclTensor *numRecvPerRankHp = nullptr;
    aclTensor *numRecvPerExpertHp = nullptr;
    aclTensor *dstSlot = nullptr;
    aclTensor *recvSrcMetaInternal = nullptr;
    aclTensor *recvX = nullptr;
    aclTensor *recvTopkIdx = nullptr;
    aclTensor *recvTopkWeights = nullptr;
    aclTensor *recvSrcMeta = nullptr;
    aclTensor *numRecvPerRank = nullptr;
    aclTensor *numRecvPerExpert = nullptr;
    aclTensor *combinedX = nullptr;

    // 定义各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> topkIdxShape{BS, K};
    std::vector<int64_t> topkWeightsShape{BS, K};
    std::vector<int64_t> numRecvPerRankHpShape{EP_WORLD_SIZE};
    std::vector<int64_t> numRecvPerExpertHpShape{numLocalExperts};
    std::vector<int64_t> dstSlotShape{BS, K};
    std::vector<int64_t> recvSrcMetaInternalShape{upperA, 3};
    std::vector<int64_t> recvXShape{upperA, H};
    std::vector<int64_t> recvTopkIdxShape{upperA, K};
    std::vector<int64_t> recvTopkWeightsShape{upperA, K};
    std::vector<int64_t> recvSrcMetaShape{upperA, 3};
    std::vector<int64_t> numRecvPerRankShape{EP_WORLD_SIZE};
    std::vector<int64_t> numRecvPerExpertShape{numLocalExperts};
    std::vector<int64_t> combinedXShape{BS, H};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t topkIdxShapeSize = GetShapeSize(topkIdxShape);
    int64_t topkWeightsShapeSize = GetShapeSize(topkWeightsShape);
    int64_t recvXShapeSize = GetShapeSize(recvXShape);

    // 构造host侧数据
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> topkIdxHostData;
    for (int32_t tokenId = 0; tokenId < BS; tokenId++) {
        for (int32_t kId = 0; kId < K; kId++) {
            topkIdxHostData.push_back(kId % numExperts);
        }
    }
    std::vector<float> topkWeightsHostData(topkWeightsShapeSize, 0.125f);
    std::vector<int32_t> numRecvPerRankHpHostData(EP_WORLD_SIZE, 0);
    std::vector<int64_t> numRecvPerExpertHpHostData(numLocalExperts, 0);
    std::vector<int32_t> dstSlotHostData(BS * K, 0);
    std::vector<int32_t> recvSrcMetaInternalHostData(upperA * 3, 0);
    std::vector<int16_t> recvXHostData(recvXShapeSize, 0);
    std::vector<int32_t> recvTopkIdxHostData(upperA * K, 0);
    std::vector<float> recvTopkWeightsHostData(upperA * K, 0);
    std::vector<int32_t> recvSrcMetaHostData(upperA * 3, 0);
    std::vector<int32_t> numRecvPerRankHostData(EP_WORLD_SIZE, 0);
    std::vector<int64_t> numRecvPerExpertHostData(numLocalExperts, 0);
    std::vector<int16_t> combinedXHostData(xShapeSize, 0);

    // 构造device侧变量
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkIdxHostData, topkIdxShape, &topkIdxDeviceAddr, aclDataType::ACL_INT32, &topkIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkWeightsHostData, topkWeightsShape, &topkWeightsDeviceAddr, aclDataType::ACL_FLOAT, &topkWeights);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numRecvPerRankHpHostData, numRecvPerRankHpShape, &numRecvPerRankHpDeviceAddr, aclDataType::ACL_INT32, &numRecvPerRankHp);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numRecvPerExpertHpHostData, numRecvPerExpertHpShape, &numRecvPerExpertHpDeviceAddr, aclDataType::ACL_INT64, &numRecvPerExpertHp);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dstSlotHostData, dstSlotShape, &dstSlotDeviceAddr, aclDataType::ACL_INT32, &dstSlot);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(recvSrcMetaInternalHostData, recvSrcMetaInternalShape, &recvSrcMetaInternalDeviceAddr, aclDataType::ACL_INT32, &recvSrcMetaInternal);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(recvXHostData, recvXShape, &recvXDeviceAddr, aclDataType::ACL_BF16, &recvX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(recvTopkIdxHostData, recvTopkIdxShape, &recvTopkIdxDeviceAddr, aclDataType::ACL_INT32, &recvTopkIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(recvTopkWeightsHostData, recvTopkWeightsShape, &recvTopkWeightsDeviceAddr, aclDataType::ACL_FLOAT, &recvTopkWeights);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(recvSrcMetaHostData, recvSrcMetaShape, &recvSrcMetaDeviceAddr, aclDataType::ACL_INT32, &recvSrcMeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numRecvPerRankHostData, numRecvPerRankShape, &numRecvPerRankDeviceAddr, aclDataType::ACL_INT32, &numRecvPerRank);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(numRecvPerExpertHostData, numRecvPerExpertShape, &numRecvPerExpertDeviceAddr, aclDataType::ACL_INT64, &numRecvPerExpert);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(combinedXHostData, combinedXShape, &combinedXDeviceAddr, aclDataType::ACL_BF16, &combinedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 声明workspace和executor变量
    uint64_t dispatchWorkspaceSize = 0;
    aclOpExecutor *dispatchExecutor = nullptr;
    void *dispatchWorkspaceAddr = nullptr;
    uint64_t epilogueWorkspaceSize = 0;
    aclOpExecutor *epilogueExecutor = nullptr;
    void *epilogueWorkspaceAddr = nullptr;
    uint64_t combineWorkspaceSize = 0;
    aclOpExecutor *combineExecutor = nullptr;
    void *combineWorkspaceAddr = nullptr;

    /*********************************** 调用dispatch算子 ***********************************/
    ret = aclnnMoeEpDispatchGetWorkspaceSize(
        contextTensor, x, topkIdx, topkWeights, nullptr,
        EP_WORLD_SIZE, args.rankId, numExperts, numMaxTokensPerRank,
        cclBufferSize, expertAlignment, doCpuSync,
        numRecvPerRankHp, numRecvPerExpertHp, dstSlot, recvSrcMetaInternal,
        &dispatchWorkspaceSize, &dispatchExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeEpDispatchGetWorkspaceSize failed. ret = %d\n", ret); return ret);
    if (dispatchWorkspaceSize > 0) {
        ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc dispatch workspace failed. ret = %d\n", ret); return ret);
    }
    ret = aclnnMoeEpDispatch(dispatchWorkspaceAddr, dispatchWorkspaceSize, dispatchExecutor, args.dispatchStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeEpDispatch failed. ret = %d\n", ret); return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeEpDispatch execute successfully.\n", args.rankId);

    /*********************************** 调用epilogue算子 ***********************************/
    ret = aclnnMoeEpDispatchEpilogueGetWorkspaceSize(
        contextTensor, dstSlot, recvSrcMetaInternal,
        numRecvPerRankHp, numRecvPerExpertHp,
        EP_WORLD_SIZE, args.rankId, numExperts, numMaxTokensPerRank,
        cclBufferSize, expertAlignment,
        recvX, recvTopkIdx, recvTopkWeights, recvSrcMeta,
        numRecvPerRank, numRecvPerExpert,
        &epilogueWorkspaceSize, &epilogueExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeEpDispatchEpilogueGetWorkspaceSize failed. ret = %d\n", ret); return ret);
    if (epilogueWorkspaceSize > 0) {
        ret = aclrtMalloc(&epilogueWorkspaceAddr, epilogueWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc epilogue workspace failed. ret = %d\n", ret); return ret);
    }
    ret = aclnnMoeEpDispatchEpilogue(epilogueWorkspaceAddr, epilogueWorkspaceSize, epilogueExecutor, args.dispatchStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeEpDispatchEpilogue failed. ret = %d\n", ret); return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] epilogue aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeEpDispatchEpilogue execute successfully.\n", args.rankId);

    /*********************************** 调用combine算子 ***********************************/
    ret = aclnnMoeEpCombineGetWorkspaceSize(
        contextTensor, recvX, dstSlot, recvSrcMeta, numRecvPerRank,
        nullptr, nullptr, nullptr,
        EP_WORLD_SIZE, args.rankId, numExperts, numMaxTokensPerRank,
        cclBufferSize,
        combinedX, nullptr,
        &combineWorkspaceSize, &combineExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnMoeEpCombineGetWorkspaceSize failed. ret = %d\n", ret); return ret);
    if (combineWorkspaceSize > 0) {
        ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc combine workspace failed. ret = %d\n", ret); return ret);
    }
    ret = aclnnMoeEpCombine(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeEpCombine failed. ret = %d\n", ret); return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] combine aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeEpDispatch, aclnnMoeEpDispatchEpilogue and aclnnMoeEpCombine execute successfully.\n", args.rankId);

    // 释放device资源
    if (dispatchWorkspaceSize > 0) {
        aclrtFree(dispatchWorkspaceAddr);
    }
    if (epilogueWorkspaceSize > 0) {
        aclrtFree(epilogueWorkspaceAddr);
    }
    if (combineWorkspaceSize > 0) {
        aclrtFree(combineWorkspaceAddr);
    }
    DestroyTensor(contextTensor);
    DestroyTensor(x);
    DestroyTensor(topkIdx);
    DestroyTensor(topkWeights);
    DestroyTensor(numRecvPerRankHp);
    DestroyTensor(numRecvPerExpertHp);
    DestroyTensor(dstSlot);
    DestroyTensor(recvSrcMetaInternal);
    DestroyTensor(recvX);
    DestroyTensor(recvTopkIdx);
    DestroyTensor(recvTopkWeights);
    DestroyTensor(recvSrcMeta);
    DestroyTensor(numRecvPerRank);
    DestroyTensor(numRecvPerExpert);
    DestroyTensor(combinedX);

    FreeDeviceAddr(contextDeviceAddr);
    FreeDeviceAddr(xDeviceAddr);
    FreeDeviceAddr(topkIdxDeviceAddr);
    FreeDeviceAddr(topkWeightsDeviceAddr);
    FreeDeviceAddr(numRecvPerRankHpDeviceAddr);
    FreeDeviceAddr(numRecvPerExpertHpDeviceAddr);
    FreeDeviceAddr(dstSlotDeviceAddr);
    FreeDeviceAddr(recvSrcMetaInternalDeviceAddr);
    FreeDeviceAddr(recvXDeviceAddr);
    FreeDeviceAddr(recvTopkIdxDeviceAddr);
    FreeDeviceAddr(recvTopkWeightsDeviceAddr);
    FreeDeviceAddr(recvSrcMetaDeviceAddr);
    FreeDeviceAddr(numRecvPerRankDeviceAddr);
    FreeDeviceAddr(numRecvPerExpertDeviceAddr);
    FreeDeviceAddr(combinedXDeviceAddr);

    HcclCommDestroy(args.hcclEpComm);
    aclrtDestroyStream(args.dispatchStream);
    aclrtDestroyStream(args.combineStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[])
{
    const char *env_dev_num = std::getenv("ENV_DEV_NUM");
    if (!env_dev_num) {
        LOG_PRINT("[WARNING] Please set environment variable ENV_DEV_NUM.\n");
        return 0;
    }
    int actual_env_dev_num = std::stoi(std::string(env_dev_num));
    if (actual_env_dev_num < DEV_NUM) {
        LOG_PRINT("[INFO] ENV_DEV_NUM=%d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
        return 0;
    }

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

    aclrtStream dispatchStream[DEV_NUM];
    aclrtStream combineStream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&dispatchStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&combineStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devicesEp[EP_WORLD_SIZE];
    for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        devicesEp[epId] = epId;
    }
    HcclComm commsEp[EP_WORLD_SIZE];
    ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclEpComm = commsEp[rankId];
        args[rankId].dispatchStream = dispatchStream[rankId];
        args[rankId].combineStream = combineStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessDispatchEpilogueCombine, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    return 0;
}
```