# aclnnFFNToAttentionV2

## 修订记录

| 日期 | 版本 | 修改说明 |
| --- | --- | --- |
| 2026-08-31 | 1.0 | 首次发布FFNToAttentionV2 ACLNN接口文档。 |

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

`aclnnFFNToAttentionV2`通过URMA将FFN Worker产生的token数据写入目标Attention Worker的通信window。
每个有效token由`sessionIds`、`microBatchIds`、`tokenIds`和`expertOffsets`确定目标rank及window槽位。
算子没有Host侧可见的输出Tensor，执行结果保存在目标Attention Worker的通信window中。

与`aclnnFFNToAttention`相比，V2接口增加以下能力：

- 显式接收MC2通信`context`。
- 显式接收`cclBufferSize`并校验通信window容量。
- 在<term>Ascend 950PR/Ascend 950DT</term>上使用URMA通信路径。

## 函数原型

该算子使用[两段式接口](../../../docs/zh/context/two_phase_api.md)。必须先调用
`aclnnFFNToAttentionV2GetWorkspaceSize`获取workspace大小和执行器，再调用`aclnnFFNToAttentionV2`执行计算。

```cpp
aclnnStatus aclnnFFNToAttentionV2GetWorkspaceSize(
    const aclTensor   *context,
    const aclTensor   *x,
    const aclTensor   *sessionIds,
    const aclTensor   *microBatchIds,
    const aclTensor   *tokenIds,
    const aclTensor   *expertOffsets,
    const aclTensor   *actualTokenNum,
    const aclTensor   *attnRankTable,
    const char        *group,
    int64_t            worldSize,
    const aclIntArray *tokenInfoTableShape,
    const aclIntArray *tokenDataShape,
    int64_t            cclBufferSize,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnFFNToAttentionV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnFFNToAttentionV2GetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  | --- | --- | --- | --- | --- | --- | --- | :---: |
  | context（aclTensor*） | 输入 | MC2通信上下文，包含本rank、远端window地址及URMA通信句柄。 | 不支持空Tensor；必须通过MC2 context管理接口创建，且在所有rank执行完成前保持有效。 | INT32 | ND | 1 | × |
  | x（aclTensor*） | 输入 | 本rank待发送的token数据。 | 不支持空Tensor；shape为`[Y,H]`，`1024<=H<=8192`。 | FLOAT16、BFLOAT16 | ND | 2 | √ |
  | sessionIds（aclTensor*） | 输入 | 每个token对应的Attention Worker编号。 | 不支持空Tensor；shape为`[Y]`；使用`attnRankTable`时取值范围为`[0,A)`，否则对应目标rank。 | INT32 | ND | 1 | √ |
  | microBatchIds（aclTensor*） | 输入 | 每个token对应的micro batch编号。 | 不支持空Tensor；shape为`[Y]`，取值范围为`[0,microBatchNum)`。 | INT32 | ND | 1 | √ |
  | tokenIds（aclTensor*） | 输入 | 每个token在目标micro batch内的batch偏移。 | 不支持空Tensor；shape为`[Y]`，取值范围为`[0,BS)`。 | INT32 | ND | 1 | √ |
  | expertOffsets（aclTensor*） | 输入 | 每个token对应的专家槽位偏移。 | 不支持空Tensor；shape为`[Y]`，取值范围为`[0,expertNumPerToken)`。 | INT32 | ND | 1 | √ |
  | actualTokenNum（aclTensor*） | 输入 | 本rank实际发送的有效token数量。 | 不支持空Tensor；shape为`[1]`，取值范围为`[0,Y]`。 | INT64 | ND | 1 | √ |
  | attnRankTable（aclTensor*） | 输入 | Attention Worker编号到通信rank的映射表。 | 可选输入；传入空指针时`sessionIds`直接表示目标rank；非空时shape为`[A]`且`A<worldSize`。 | INT32 | ND | 1 | √ |
  | group（char*） | 输入 | HCCL通信域名称。 | 不能为空字符串，长度范围为`[1,128)`；所有rank必须一致。 | - | - | - | - |
  | worldSize（int64_t） | 输入 | 通信域rank数量。 | 取值范围为`[2,768]`，必须与通信域实际rank数量一致。 | - | - | - | - |
  | tokenInfoTableShape（aclIntArray*） | 输入 | 目标token信息表shape。 | 长度为3，内容为`[microBatchNum,BS,expertNumPerToken]`；`microBatchNum=1`，`1<=BS<=512`。 | INT64 | - | 1 | - |
  | tokenDataShape（aclIntArray*） | 输入 | 目标token数据表shape。 | 长度为4，内容为`[microBatchNum,BS,expertNumPerToken,HS]`；前三维与`tokenInfoTableShape`相同，`H<=HS<=8320`。 | INT64 | - | 1 | - |
  | cclBufferSize（int64_t） | 输入 | 每个rank可用的通信window大小，单位为Byte。 | 必须大于0，且不小于“通信window容量”中计算的最小值。 | - | - | - | - |
  | workspaceSize（uint64_t*） | 输出 | 返回Device侧workspace大小。 | 不能为空。 | - | - | - | - |
  | executor（aclOpExecutor**） | 输出 | 返回包含算子执行流程的执行器。 | 不能为空。 | - | - | - | - |

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，常见错误如下：

  | 返回值 | 错误码 | 描述 |
  | --- | --- | --- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选Tensor、`group`、shape数组、`workspaceSize`或`executor`存在空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | dtype、format、shape、属性值或参数间关系不满足约束。 |
  | ACLNN_ERR_RUNTIME_ERROR | 361001 | MC2 context、HCCL通信资源或当前运行平台不满足执行条件。 |

## aclnnFFNToAttentionV2

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  | --- | --- | --- |
  | workspace | 输入 | Device侧workspace地址；`workspaceSize`为0时可传入空指针。 |
  | workspaceSize | 输入 | Device侧workspace大小，由第一段接口获取。 |
  | executor | 输入 | 第一段接口返回的执行器。 |
  | stream | 输入 | 执行算子的ACL stream。 |

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 通信window容量

设：

- `M=microBatchNum`
- `B=BS`
- `E=expertNumPerToken`
- `S=HS`

通信window所需的最小容量按以下方式计算：

$$
tokenInfoBytes=M\times B\times E\times 4
$$

$$
tokenDataBytes=M\times B\times E\times S\times 2
$$

$$
cclBufferSize_{min}=AlignUp(AlignUp(tokenInfoBytes,512)+tokenDataBytes,2\mathrm{MB})
$$

所有乘加和对齐计算均应使用64位整数并检查溢出。传给接口的`cclBufferSize`不得大于context实际注册的
通信window大小。

## 约束说明

- 所有rank使用的`group`、`worldSize`、`tokenInfoTableShape`、`tokenDataShape`和`cclBufferSize`必须一致。
- 所有rank必须先完成MC2 context初始化；FFN Worker执行算子期间，Attention Worker必须保持context和通信window有效。
- 同一目标rank内，多个有效token不得映射到相同的
  `(microBatchId,tokenId,expertOffset)`槽位，否则不同写入之间存在覆盖竞争。
- `attnRankTable`中的rank必须位于`[0,worldSize)`。
- 该通信域在算子执行期间不得被不兼容的其他通信流程复用。
- 确定性说明：`aclnnFFNToAttentionV2`默认非确定性实现，不支持通过`aclrtCtxSetSysParamOpt`开启确定性。

## 调用示例

调用流程如下：

1. 创建HCCL通信域，并在所有rank上创建MC2 context。
2. 根据token表shape计算并检查`cclBufferSize`。
3. 在FFN Worker上创建输入Tensor并调用两段式ACLNN接口。
4. 等待FFN Worker stream执行完成后，再释放所有rank的context、window和通信域。

核心调用代码如下：

```cpp
uint64_t hcclBufferSize = 0;
aclTensor *context = nullptr;
// 所有rank先使用同一通信域创建MC2 context，并在算子执行完成前保持context有效。
auto ret = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(
    groupName, "ffn_to_attention_v2", hcclBufferSize, context);

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
// 第一段接口完成参数校验，并返回workspace大小和执行器。
ret = aclnnFFNToAttentionV2GetWorkspaceSize(
    context, x, sessionIds, microBatchIds, tokenIds, expertOffsets,
    actualTokenNum, attnRankTable, groupName, worldSize,
    tokenInfoTableShape, tokenDataShape, static_cast<int64_t>(hcclBufferSize),
    &workspaceSize, &executor);

void *workspace = nullptr;
if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
// 第二段接口仅由FFN Worker调用；Attention Worker需要继续保持通信window有效。
if (ret == ACL_SUCCESS) {
    ret = aclnnFFNToAttentionV2(workspace, workspaceSize, executor, stream);
}
if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
}
```

完整多卡示例见
[test_aclnn_ffn_to_attention_v2.cpp](../examples/arch35/test_aclnn_ffn_to_attention_v2.cpp)。具体编译和执行过程请参考
[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。
