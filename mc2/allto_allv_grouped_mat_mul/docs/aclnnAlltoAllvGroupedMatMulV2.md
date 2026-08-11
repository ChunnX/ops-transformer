# aclnnAlltoAllvGroupedMatMulV2

[查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/allto_allv_grouped_mat_mul)

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950DT | 是 |
| Atlas A3 训练/推理系列产品 | 否 |
| Atlas A2 训练/推理系列产品 | 否 |

## 功能说明

该接口通过调用方已经初始化的 CANN SHMEM/UDMA 通信域完成非均匀 AlltoAllV，并立即对接收结果执行本地专家 GroupedMatMul。V2 接口仅保留源算子能力：BF16、TP=1、非转置权重、无共享专家 MatMul、无独立 permute 输出。

对于本卡专家 $e$ 和来源 rank $r$，计算可表示为：

$$
Y_{e,r} = \operatorname{AlltoAllV}(X)_{e,r} W_e
$$

## 函数原型

该算子使用两段式 aclnn 接口。首先构造 executor 并取得 workspace 大小，然后在同一 SHMEM 生命周期内执行算子。

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX,
    const aclTensor *gmmWeight,
    void *shmemContext,
    int64_t epWorldSize,
    const aclIntArray *sendCounts,
    const aclIntArray *recvCounts,
    aclTensor *gmmY,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 第一段接口参数

| 参数 | 输入/输出 | 类型 | 说明 |
|---|---|---|---|
| `gmmX` | 输入 | `const aclTensor *` | BF16、ND、二维张量，shape 为 `[M, K]`。存储必须由当前 SHMEM 对称堆分配。按目标 rank 分段，同一目标 rank 内按目标本地专家排列。 |
| `gmmWeight` | 输入 | `const aclTensor *` | BF16、ND、三维张量，shape 为 `[E, K, N]`，其中 `E` 为本卡专家数。权重不转置。 |
| `shmemContext` | 输入 | `void *` | 已初始化 SHMEM 运行时的设备 FFTS 配置地址，例如 `reinterpret_cast<void *>(shmemx_get_ffts_config())`。接口仅借用该地址，不接管生命周期。 |
| `epWorldSize` | 输入 | `int64_t` | EP rank 数，取值范围 `[1, 8]`，执行时必须等于 `shmem_n_pes()`。 |
| `sendCounts` | 输入 | `const aclIntArray *` | 长度为 `epWorldSize * E`。展平顺序为 `[目标 rank, 目标本地专家]`，元素为非负 uint32 范围整数，元素之和必须等于 `M`。 |
| `recvCounts` | 输入 | `const aclIntArray *` | 长度为 `epWorldSize * E`。展平顺序为 `[来源 rank, 本地专家]`，元素为非负 uint32 范围整数。 |
| `gmmY` | 输出 | `aclTensor *` | BF16、ND、二维张量，shape 为 `[sum(recvCounts), N]`。 |
| `workspaceSize` | 输出 | `uint64_t *` | 返回第二段接口需要的完整 workspace 字节数。 |
| `executor` | 输出 | `aclOpExecutor **` | 返回包含算子执行流程的 executor。 |

## 第二段接口参数

| 参数 | 输入/输出 | 类型 | 说明 |
|---|---|---|---|
| `workspace` | 输入 | `void *` | 必须使用 `aclshmem_calloc` 或等价 SHMEM 对称堆接口，按第一段返回的 `workspaceSize` 分配。 |
| `workspaceSize` | 输入 | `uint64_t` | 第一段接口返回的完整大小，不允许缩小。 |
| `executor` | 输入 | `aclOpExecutor *` | 第一段接口返回的 executor。 |
| `stream` | 输入 | `aclrtStream` | 执行流。各 rank 必须遵循相同的集合通信调用顺序。 |

## 支持的数据类型

`gmmX`、`gmmWeight` 和 `gmmY` 仅支持 `BFLOAT16`。

## Shape 与输出顺序

- `gmmX`: `[M, K]`
- `gmmWeight`: `[E, K, N]`
- `gmmY`: `[R, N]`，其中 `R = sum(recvCounts)`
- `sendCounts` 与 `recvCounts` 的存储都是 rank-major；每个 rank 段内按本地专家编号递增。
- `gmmY` 保留源算子的 expert-major 输出顺序：先按本地专家编号递增，再在每个专家内按来源 rank 递增。每个 `(expert, source rank)` 分组内保持通信输入顺序。

例如 `EP=2, E=2` 时，`recvCounts` 存储顺序为：

```text
[rank0/expert0, rank0/expert1, rank1/expert0, rank1/expert1]
```

而 `gmmY` 的行分组顺序为：

```text
[expert0/rank0, expert0/rank1, expert1/rank0, expert1/rank1]
```

## SHMEM 与 workspace 约束

- 调用 `GetWorkspaceSize` 前必须完成 `aclshmemx_init_attr`，并将通信引擎设置为 `ACLSHMEM_DATA_OP_UDMA`。
- `gmmX` 和第二段接口的 `workspace` 必须来自同一个已初始化 SHMEM 对称堆；普通 `aclrtMalloc` workspace 不满足要求。
- 算子 user workspace 固定包含 `400 MiB + 4096 B`：前 400 MiB 是双缓冲通信窗口，末尾页面保存跨 rank 信号与对齐空间。第一段返回值还可能包含框架 system workspace，调用方必须按返回的完整字节数进行 SHMEM 分配。
- 源实现约束 `gmmX` 与固定 user workspace 的总量不超过 1004 MiB。调用方设置的 SHMEM heap 还必须容纳框架附加空间及其他对称分配。
- `shmemContext`、`gmmX`、workspace 和 SHMEM runtime 的生命周期必须覆盖 executor 构造、kernel 执行及流同步。
- 各 rank 的 `epWorldSize`、`E`、`K`、`N` 与集合通信调用顺序必须一致；发送表和对端接收表必须全局匹配。

## 其他约束

- 仅支持 Ascend 950（DAV_3510）。
- `1 <= E <= 64`，`1 <= epWorldSize <= 8`。
- `K > 0`、`N > 0`，且 `gmmX.shape[1] == gmmWeight.shape[1]`。
- 每个 peer 的发送总量和接收总量必须可由 uint32 表示。
- 不支持 FP16、权重转置、共享专家 MatMul 和 permute 独立输出。
- 本接口允许破坏旧 V2 ABI；旧的 `group`、`commMode` 和可选张量参数不再存在。

## 返回值

两个接口均返回 `aclnnStatus`：

- `ACLNN_SUCCESS`：接口调用成功。
- `ACLNN_ERR_PARAM_NULLPTR`：必选 tensor、context、workspace 元数据或 executor 指针为空。
- `ACLNN_ERR_PARAM_INVALID`：平台、dtype、shape、EP、counts 或 workspace 不符合约束。

kernel 启动后的 SHMEM/UDMA 异常通过流同步和运行时最近错误信息返回。

## 调用示例

完整 C++ 示例见 [Ascend 950 SHMEM V2 示例](../examples/arch35/test_aclnn_allto_allv_grouped_mat_mul.cpp)。示例采用“一进程一 rank”，使用 `aclshmem_malloc` 分配 `gmmX`，使用 `aclshmem_calloc` 分配第一段接口返回的完整 workspace，并将 `shmemx_get_ffts_config()` 传入 V2。

编译支持以下两种 SHMEM SDK 布局：

- 源算子布局：设置 `SHMEM_HOME_PATH`，使用 `${SHMEM_HOME_PATH}/shmem/include`，示例链接 `${SHMEM_HOME_PATH}/shmem/lib/libshmem.so`。
- HCCL 工程集成布局：设置 `SHMEM_HOME`，使用 `${SHMEM_HOME}/include` 与 `${SHMEM_HOME}/src/device`，示例通常链接 `${SHMEM_HOME}/build/lib/libshmem.so`。

若 SHMEM 运行时承载在 HCCL 网络环境上，启动前应按实际机器配置 `HCCL_SOCKET_IFNAME`，并按集群情况设置 `HCCL_CONNECT_TIMEOUT`、`HCCL_EXEC_TIMEOUT` 与 SHMEM rendezvous 地址。各 rank 必须继承一致的运行时库路径和 rendezvous 配置；不要把 IP 或网卡名硬编码进算子仓。
