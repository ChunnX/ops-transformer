# aclnnBandwidthTest

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |

## 功能说明

- 接口功能：实现MOE（Mixture of Experts）场景下的AlltoAll Dispatch通信，用于测试NPU设备间的通信带宽。每个rank根据`dstRankId`将token发送到对应的目标rank，实现数据的分布式分发。

    该算子支持两种运行模式：
    1. **mode=0（精度模式/完整流程）**：执行完整的通信流程，包括数据发送、状态同步、等待接收和数据拷贝输出。
    2. **mode=1（纯发模式）**：仅执行数据发送和状态同步，用于单向带宽测试。

- 计算公式：

    - 情形1：如果mode=0（完整流程）：

    $$
    y = AllToAllV(x, dstRankId) \\
    receiveCnt = CountTokensByRank(dstRankId)
    $$

    其中，`y`为接收到的token数据，`receiveCnt`为从各rank接收的token数量。

    - 情形2：如果mode=1（纯发模式）：

    仅执行发送操作，不产生有效输出数据。

## 函数原型

每个算子分为两段式接口，必须先调用 "aclnnBandwidthTestGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBandwidthTest"接口执行计算。

```cpp
aclnnStatus aclnnBandwidthTestGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* dstRankId,
    const char*      group,
    int64_t          worldSize,
    int64_t          maxBs,
    int64_t          mode,
    int64_t          aivNum,
    const char*      commAlg,
    aclTensor*       y,
    aclTensor*       receiveCnt,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnBandwidthTest(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnBandwidthTestGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1200px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 250px">
    <col style="width: 150px">
    <col style="width: 80px">
    <col style="width: 180px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>本卡发送的token数据。</td>
    <td>要求2D Tensor。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td><code>(BS, H)</code>（BS=batch size，H=hidden size）</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dstRankId</td>
    <td>输入</td>
    <td>每个token的目标rank索引。</td>
    <td>要求1D Tensor。每个元素表示对应token要发送到的目标rank ID。</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(BS,)</code></td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>通信域名称。</td>
    <td>字符串长度范围为<code>[1, 128)</code>，不能为空。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>worldSize</td>
    <td>输入</td>
    <td>通信域大小（参与通信的rank数量）。</td>
    <td>取值范围: 需大于0。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maxBs</td>
    <td>输入</td>
    <td>最大batch size大小。</td>
    <td>取值范围: 需大于0。用于确定输出tensor的空间大小。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>mode</td>
    <td>输入</td>
    <td>运行模式。</td>
    <td>取值范围<code>[0, 1]</code>：<ul><li>0：精度模式（完整流程），执行发送、同步、接收和输出拷贝。</li><li>1：纯发模式，仅执行发送和同步，用于单向带宽测试。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>aivNum</td>
    <td>输入</td>
    <td>使用核数。</td>
    <td>可选参数，用于指定核数。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>输入</td>
    <td>通信算法。</td>
    <td>可选参数，用于指定通信算法类型。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>接收到的token数据。</td>
    <td>要求为2D Tensor。mode=0时有有效输出；mode=1时不产生有效数据。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td><code>(maxBs * worldSize, H)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>receiveCnt</td>
    <td>输出</td>
    <td>从各rank接收的token数量。</td>
    <td>要求为1D Tensor。mode=0时有有效输出；mode=1时不产生有效数据。</td>
    <td>INT32</td>
    <td>ND</td>
    <td><code>(worldSize,)</code></td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回Device侧需申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回包含算子计算流程的op执行器。</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    - <term>Ascend 950PR/Ascend 950DT</term>：
        - `x`的数据类型支持FLOAT16、BFLOAT16、FLOAT32。
        - `dstRankId`中的元素值需满足<code>[0, worldSize)</code>范围。
        - `mode`取值范围为<code>[0, 1]</code>。
        - `y`的数据类型需与`x`保持一致。
        - `receiveCnt`的数据类型为INT32。

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。  

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 700px">
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
    <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>输入和输出的shape不在支持的范围内。</td>
    </tr>
    <tr>
        <td>参数的取值不在支持的范围内。</td>
    </tr>
    </tbody>
    </table>

## aclnnBandwidthTest

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 550px">
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
    <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnBandwidthTestGetWorkspaceSize</code>获取。</td>
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

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- **确定性计算**：
  - aclnnBandwidthTest默认确定性实现。

- **驱动约束**：
  - 算子通信域各节点的驱动版本应当相同。

- **参数一致性约束**：
  - 调用接口过程中使用的`group`、`worldSize`、`maxBs`参数所有卡需保持一致。

- **产品特定约束**：
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持该产品型号。

- **Shape变量约束**：

  | 变量         | 定义与取值范围                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | BS           | 表示本卡发送的token数量，取值范围<code>(0 < BS ≤ maxBs)</code>。|
  | H（hidden size） | 表示hidden size隐藏层大小</code>。|
  | worldSize    | 表示通信域大小</code>。|

- **通信域使用约束**：
  - 通信域内各rank需在同一超节点内。
  - 一个通信域内不允许同时使用其他通信算子。

- **环境变量约束**：
  - **HCCL_BUFFSIZE**：
      调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB：
      - <term>Ascend 950PR/Ascend 950DT</term>：设置大小要求<code>(≥ 2)</code>且满足<code>(≥ 2 * (maxBs * worldSize * Align512(Align32(2 * H) + 64)))</code>（`Align512(x) = ((x + 512 - 1) / 512) * 512`；`Align32(x) = ((x + 32 - 1) / 32) * 32`）。

- **其他约束**：
  - 公式中的"/"表示整除。
  - `dstRankId`中的每个元素值需在<code>[0, worldSize)</code>范围内，表示对应token发送到的目标rank。

## 调用示例

- 环境变量配置：

    ```bash
    # 运行前需设置环境变量
    ## ENV_DEV_NUM说明：根据当前机器的卡数设置该变量
    export ENV_DEV_NUM=2
    ```

- <term>Ascend 950PR/Ascend 950DT</term>：

    无需配置ranktable文件以及环境变量RANK_TABLE_FILE、FIRST_RANK_ID。

    算子编译命令如下：
    ```bash
    bash build.sh --pkg --soc=ascend950 --ops=bandwidth_test
    ```

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <cstdio>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_bandwidth_test.h"

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
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

const uint32_t WORLD_SIZE = 2;

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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, 
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
    );
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

int launchBandwidthTest(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    // 设置场景参数
    int64_t BS = 8;
    int64_t H = 128;
    int64_t maxBs = BS;
    int64_t worldSize = WORLD_SIZE;
    int64_t mode = 0;  // 精度模式（完整流程）
    std::string commAlg = "";

    // 声明device侧输入输出变量
    void *xDeviceAddr = nullptr;
    void *dstRankIdDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *receiveCntDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *dstRankId = nullptr;
    aclTensor *y = nullptr;
    aclTensor *receiveCnt = nullptr;

    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> dstRankIdShape{BS};
    std::vector<int64_t> yShape{maxBs * worldSize, H};
    std::vector<int64_t> receiveCntShape{worldSize};

    long long xShapeSize = GetShapeSize(xShape);
    long long dstRankIdShapeSize = GetShapeSize(dstRankIdShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long receiveCntShapeSize = GetShapeSize(receiveCntShape);

    // 构造host侧变量
    std::vector<op::fp16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> dstRankIdHostData;
    for (int32_t token_id = 0; token_id < dstRankIdShape[0]; token_id++) {
        // 每个token发送到rank (token_id % worldSize)
        dstRankIdHostData.push_back(token_id % worldSize);
    }
    std::vector<op::fp16_t> yHostData(yShapeSize, 0);
    std::vector<int32_t> receiveCntHostData(receiveCntShapeSize, 0);

    // 构造device侧变量
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FP16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dstRankIdHostData, dstRankIdShape, &dstRankIdDeviceAddr, aclDataType::ACL_INT32, &dstRankId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FP16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(receiveCntHostData, receiveCntShape, &receiveCntDeviceAddr, aclDataType::ACL_INT32, &receiveCnt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 声明算子执行必需变量
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    // 调用算子第一阶段接口
    ret = aclnnBandwidthTestGetWorkspaceSize(
        x, dstRankId, 
        hcomName, worldSize, maxBs, mode, 
        commAlg.c_str(),
        y, receiveCnt, 
        &workspaceSize, &executor
    );
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnBandwidthTestGetWorkspaceSize failed. ret = %d\n", ret); return ret
    );

    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }

    // 调用算子第二阶段接口
    ret = aclnnBandwidthTest(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBandwidthTest failed. ret = %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
    );

    LOG_PRINT("[INFO] device_%d aclnnBandwidthTest execute successfully.\n", args.rankId);

    // 释放device资源
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensor(x);
    DestroyTensor(dstRankId);
    DestroyTensor(y);
    DestroyTensor(receiveCnt);

    FreeDeviceAddr(xDeviceAddr);
    FreeDeviceAddr(dstRankIdDeviceAddr);
    FreeDeviceAddr(yDeviceAddr);
    FreeDeviceAddr(receiveCntDeviceAddr);

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    
    return 0;
}

int main()
{
    // 初始化ACL和HCCL
    aclInit(nullptr);
    HcclInit(nullptr, 0);

    std::vector<Args> args(WORLD_SIZE);
    std::vector<std::thread> threads;

    for (uint32_t i = 0; i < WORLD_SIZE; i++) {
        threads.emplace_back([i, &args]() {
            args[i].rankId = i;
            aclrtSetDevice(i);
            aclrtCreateContext(&args[i].context, i);
            aclrtCreateStream(&args[i].stream);
            HcclCommInitRootInfo(WORLD_SIZE, nullptr, i, &args[i].hcclComm);
            launchBandwidthTest(args[i]);
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    HcclFinalize();
    aclFinalize();
    return 0;
}
```