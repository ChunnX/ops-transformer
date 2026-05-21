# MC2 通信上下文设计文档

> **兼容性声明**：本文档仅用于说明 MC2 通信上下文的设计，不保证 MC2 通信上下文相关接口的兼容性，该类接口为内部接口，可能随版本迭代发生变更，请以实际代码为准。
>
> **AI 辅助生成声明**：本文档部分内容由 AI 辅助生成和润色，可能存在不准确之处，请结合源码审阅。

## 1. 概述

本文档描述 MC2 场景下通信上下文的创建与管理机制。通信上下文是分布式通信算子执行的前提条件，它封装了通信域内所有 rank 的 buffer 地址、rank 编号、通信句柄等信息，供后续 dispatch/combine/megamoe 等算子在数据面直接使用。

系统支持两种通信上下文流程：

- **KFC（Kernel Function Call）模式**：基于 HCCL KFC 控制面 API 创建通信资源，通过 `HcclCreateOpResCtx` 创建操作上下文，通过 `HcclGetRemoteIpcHcclBuf` 获取远端 buffer 地址。适用于 Ascend910B 单节点、跨节点， Ascend910_93 超节点、跨超节点、双机直连组网 AIV 内存语义和 AIV 直调 RMDA 通信场景。
- **HCCL Channel 模式**：基于 HCCL EngineCtx 控制面 API 创建通信资源，通过 `HcclEngineCtxCreate` 创建引擎上下文，通过 `HcclChannelAcquire` 建立 Channel 通信通道，基于 Channel 查询远端 buffer 地址。适用于 AscendC950 超节点 AIV 通信引擎 + UB_MEM 内存语义通信场景。

两种后端对外暴露统一的 `CommContextManager` 接口，通过 `backend` 参数在构造时指定，后续调用 `create_context()` 或 `update_context()` 时自动路由到对应后端，调用方无需感知内部差异。

---

## 2. 整体架构

系统分为四层：

1. **Python 调用层**：`deep_ep.py` 的 `MoeDistributeBuffer` 和 `mega_moe.py` 的 `SymmBuffer`，负责构造 `CommContextManager` 并获取通信上下文。
2. **Python-C++ 桥接层**：`update_context.py` 加载 C++ 扩展模块，将 `CommContextManager` 类导出给 Python。
3. **C++ 核心层**：`CommContextManager` 作为统一入口，将请求分发到 `KfcContextBuilder` 或 `HcclChannelContextBuilder`。
4. **动态库加载层**：`hccl_common.h` 从 `libhccl.so` / `libhccl_fwk.so` 加载 HCCL 控制面 API，`acl_common.h` 从 `libacl_rt.so` 加载 `aclrtGetSocName` 用于芯片型号查询。

### 2.1 后端路由机制

`CommContextManager` 构造时保存 `backend` 参数，但后端解析延迟到首次调用 `create_context()` 或 `update_context()` 时执行。`EnsureResolved` 检查 `mode_` 是否为 `UNINITIALIZED`，若是则调用 `ResolveBackend` 将 `backend` 参数解析为 `BackendMode` 枚举（`KFC` 或 `CHANNEL`）。（延迟解析是为了避免在构造函数中可能抛出异常的问题）。

解析完成后，`DispatchBuild` 根据枚举值路由到对应的后端 Builder（`KfcContextBuilder` 或 `HcclChannelContextBuilder`），每个 Builder 在栈上构造并直接调用 `Build` 方法创建通信域上下文。

### 2.2 后端选择方式

`backend` 参数支持两种输入形式：

- **字符串**：直接指定后端，`"kfc"` 对应 KFC 模式，`"channel"` 对应 HCCL Channel 模式。
- **字典**：根据当前运行环境的芯片型号自动选择后端。字典的 key 为芯片型号的子串（如 `"Ascend910B"`、`"Ascend910_93"`、`"Ascend950"`），value 为对应的后端字符串。系统调用 `aclrtGetSocName()` 获取当前芯片的完整型号名称，然后遍历字典，使用 `std::strstr` 进行子串匹配。第一个匹配成功的 key 对应的 value 即为所选后端。

---

## 3. 通信上下文数据结构

`CommContext` 是两种后端共用的通信上下文结构体，包含以下字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `epRankId` | `uint32_t` | 本 rank 在通信域中的编号 |
| `rankSizePerServer` | `uint32_t` | 单节点内的 rank 数量（Channel 模式使用，KFC 模式为 0） |
| `kfcContextAddr` | `uint64_t` | 预留字段，当前为 0 |
| `hcclContextAddr` | `uint64_t` | KFC 模式下由 `HcclCreateOpResCtx` 返回的操作资源上下文地址；Channel 模式下为 0 |
| `epHcclBuffer_[]` | `uint64_t[1024]` | 通信域内所有 rank 的 CCL Buffer 地址。本卡通过 `HcclGetHcclBuffer` 获取，远端卡的获取方式因后端而异 |

该结构体在 Host 侧填充完成后，通过 `CopyContextToTensor` 拷贝到 NPU 设备侧的 int32 Tensor 中，供后续通算算子的 Kernel 直接读取。Tensor 的大小等于 `sizeof(CommContext)` 向上取整到 `sizeof(int32_t)` 的倍数，由 `create_context()` 内部自动计算和分配。

---

## 4. KFC 模式

KFC 模式的 MC2 通信上下文创建流程为如下。

### 4.1 初始化通信函数

调用 `InitHcclFunctions()` 从 `libhccl.so` 和 `libhccl_fwk.so` 加载 KFC 模式所需的全部 HCCL 控制面 API 函数指针。该函数使用 `static bool` 标志保证只执行一次。

### 4.2 创建通信配置对象

调用 `HcclKfcAllocOpArgs` 分配一个通信配置对象 `opArgs`，用于承载后续的算法配置和通信引擎设置。然后调用 `HcclKfcOpArgsSetCommEngine` 将通信引擎设置为 AIV（`COMM_ENGINE_AIV = 4`），表示通信算子将在 Vector Core 上执行。

### 4.3 算法配置选择

根据当前芯片型号和通信域大小选择算法配置和操作类型：

- **Ascend910B 且 worldSize > 8**：算法配置为 `"BatchWrite=level1:hierarchy"`，操作类型为 18（BatchWrite）。这种配置适用于Ascend910B 跨节点 RDMA 通信的场景。
- **其他场景**（包括 Ascend910B 单节点、Ascend910_93 超节点 等）：算法配置为 `"AlltoAll=level0:fullmesh;level1:pairwise"`，操作类型为 8（AlltoAllV）。在 level0 使用 fullmesh 算法、level1 使用 pairwise 算法。

芯片型号通过 `GetSocName()` 获取，使用 `std::strstr` 匹配 `"Ascend910B"` 子串。

### 4.4 创建 HCCL 操作上下文

`CreateHcclContext` 完成以下操作：

1. **设置算法配置**：调用 `HcclKfcOpArgsSetAlgConfig` 将上一步选择的算法配置字符串设置到通信配置对象中。
2. **获取通信组句柄**：调用 `HcclCommGetHandleWithName` 通过通信组名称获取 `HcclComm` 句柄。
3. **创建操作资源上下文**：调用 `HcclCreateOpResCtx` 创建对应操作类型的资源上下文，返回的地址存入 `CommContext::hcclContextAddr`。
4. **校验 rank 信息**：调用 `HcclGetRankSize` 和 `HcclGetRankId` 获取通信域大小和本卡编号，校验 `rankId < worldSizeHccl` 且 `worldSize == worldSizeHccl`。

创建完成后，调用 `HcclKfcFreeOpArgs` 释放配置对象。

### 4.5 收集 Rank Buffer 地址

`CollectRankBuffers` 遍历通信域内所有 rank，收集每个 rank 的 CCL Buffer 地址：

1. 调用 `HcclGetRankId` 获取本卡编号，存入 `CommContext::epRankId`。
2. **Ascend910B 且 worldSize > 8 时跳过收集**：在此场景下，`HcclGetRemoteIpcHcclBuf` 不支持在存在RDMA链路的通信域中调用，故直接返回，`epHcclBuffer_[]` 数组保持全零。后续通算算子kernel侧需要获取每个 rank 的 CCL Buffer 地址时，通过解析hcclContext中的ibverbsData进行获取。
3. 对于本卡，调用 `HcclGetHcclBuffer` 获取 buffer 地址和大小，buffer 大小即为 `cclBufferSize`。
4. 对于远端卡，调用 `HcclGetRemoteIpcHcclBuf` 获取远端 buffer 的虚拟首地址。

所有 buffer 地址存入 `CommContext::epHcclBuffer_[]` 数组。

---

## 5. HCCL Channel 模式

Channel 模式使用 HCCL EngineCtx 控制面 API，遵循 AIV 通信算子开发流程如下。

### 5.1 初始化通信函数

调用 `InitHcclEngineCtxFunctions()` 从 `libhccl_fwk.so` 加载 Channel 模式所需的全部 HCCL 控制面 API 函数指针。

### 5.2 获取通信组句柄

调用 `HcomGetCommHandleByGroup` 通过通信组名称获取 `HcclComm` 句柄。该接口是 HCCL 框架层 API，用于从已初始化的通信组中获取底层通信句柄。

### 5.3 协议协商

Channel 模式要求所有 rank 之间支持 UB_MEM（Unified Buffer Memory）内存语义通信协议。`GetCommProtocol` 执行协议协商：

1. 调用 `HcclRankGraphGetLayers` 查询通信域的拓扑层级编号列表和层级数量。
2. 如果只有 1 层（`layerNum == HCCL_COMM_LAYERS_MTE_CCU`），说明是单层拓扑，直接使用 UB_MEM 协议。
3. 如果有多层拓扑，调用 `CheckProtocolSupport` 逐层逐 rank 检查链路是否支持 UB_MEM 协议：
   - 调用 `HcclGetRankId` 获取本卡编号。
   - 对每一层，调用 `HcclRankGraphGetRanksByLayer` 获取该层内的 rank 列表。
   - 对每个远端 rank，调用 `HcclRankGraphGetLinks` 查询本卡与该 rank 之间的通信链路列表。
   - 遍历链路列表，检查是否存在 `linkProtocol == COMM_PROTOCOL_UB_MEM` 的链路。
   - 将支持 UB_MEM 的远端 rank 及其所在层级记录到 `layerMap_` 中，供后续 Channel 创建时确定使用哪一层拓扑。

### 5.4 获取或创建上下文

`GetOrCreateContext` 实现上下文的懒创建机制：

1. 调用 `HcclEngineCtxGet` 尝试获取已存在的上下文。如果成功，说明之前已创建过资源，只需调用 `HcclGetHcclBuffer` 获取本卡 buffer 大小即可。
2. 如果获取失败（首次调用），进入 `CreateContext` 创建新上下文。

### 5.5 创建上下文

`CreateContext` 完成上下文的完整创建流程：

1. **申请 Context 内存**：调用 `HcclEngineCtxCreate` 在 HCCL 引擎中申请一块大小为 `sizeof(CommContext)` 的上下文内存，用于存储通信资源信息。`mc2ContextTag` 由通信组名称拼接固定后缀生成，作为上下文的唯一标识。
2. **查询 rank 信息**：调用 `HcclGetRankId` 和 `HcclGetRankSize` 获取本卡编号和通信域大小，分别存入 `CommContext::epRankId` 和局部变量 `rankSize`。
3. **获取通信资源**：调用 `GetHcclCommResource` 收集 Channel 和 Buffer 资源（详见 5.6 节）。
4. **拷贝上下文到设备**：调用 `HcclEngineCtxCopy` 将 Host 侧填充好的 `CommContext` 结构体拷贝到步骤 1 申请的设备侧上下文内存中。

### 5.6 获取通信资源

`GetHcclCommResource` 负责收集通信域内所有 rank 的 CCL Buffer 地址，流程如下：

1. 调用 `GetHcclCommChannel` 建立 Channel 通信通道（详见 5.7 节），获取 Channel 句柄列表和每节点 rank 数量。
2. 遍历通信域内所有 rank：
   - 对于本卡，调用 `HcclGetHcclBuffer` 获取 buffer 地址和大小。
   - 对于远端卡，通过对应的 Channel 句柄调用 `HcclChannelGetHcclBuffer` 获取远端 buffer 在本端的映射地址。Channel 的索引通过 `(rank < srcRank) ? rank : (rank - 1)` 计算，因为本卡不占 Channel。
3. 所有 buffer 地址存入 `CommContext::epHcclBuffer_[]` 数组。

### 5.7 建立 Channel 通信通道

`GetHcclCommChannel` 负责在当前 rank 与所有远端 rank 之间建立 Channel 通信通道：

1. 调用 `HcclRankGraphGetLayers` 获取拓扑层级信息。
2. 调用 `HcclRankGraphGetRankSizeByLayer` 查询第 0 层的 rank 数量，得到每节点 rank 数，存入 `CommContext::rankSizePerServer`。
3. 调用 `InitHcclChannel` 初始化 Channel 描述符（详见 5.8 节）。
4. 调用 `HcclChannelAcquire` 根据描述符列表获取 Channel 句柄。该接口会触发本端与远端的 Channel 建链，建链过程中会交换两端注册的内存信息。

### 5.8 初始化 Channel 描述符

`InitHcclChannel` 为每个远端 rank 填充一个 `HcclChannelDesc` 描述符：

1. 调用 `HcclChannelDescInit` 初始化所有描述符。
2. 调用 `HcclRankGraphGetLayers` 获取拓扑层级信息。
3. 遍历所有远端 rank，对每个远端 rank：
   - 确定使用的拓扑层级：如果只有 1 层拓扑，使用第 0 层；否则使用协议协商阶段记录在 `layerMap_` 中的层级。
   - 调用 `GetHcclCommLink` 查询本卡与远端 rank 之间支持 UB_MEM 协议的链路，获取本端和远端的 Endpoint 描述符。
   - 填充 Channel 描述符的 `channelProtocol`（UB_MEM）、`remoteRank`、`notifyNum`、`localEndpoint`、`remoteEndpoint` 字段。

### 5.9 查询通信链路

`GetHcclCommLink` 查询两个 rank 之间支持指定协议的通信链路：

1. 调用 `HcclRankGraphGetLinks` 获取源 rank 和目的 rank 之间的所有链路列表。
2. 遍历链路列表，找到 `linkProtocol` 匹配指定协议的链路。
3. 返回该链路的指针，其中包含本端和远端的 Endpoint 描述符。

---

## 6. 动态库加载

HCCL 和 ACL 的 API 不在编译期链接，而是在运行时通过 `dlopen` / `dlsym` 动态加载。这样做的好处是避免了对特定版本 HCCL 库的编译依赖，同时支持按需加载——只有实际使用某种后端时才加载对应的库。

| 头文件 | 加载的库 | 加载的 API | 加载时机 |
|--------|---------|-----------|---------|
| `hccl_common.h` | `libhccl.so` | KFC 模式控制面 API（`HcclKfc*`、`HcclCreateOpResCtx`、`HcclGetHcclBuffer` 等） | `KfcContextBuilder::GetMc2Context()` 首次调用 |
| `hccl_common.h` | `libhccl_fwk.so` | Channel 模式控制面 API（`HcclEngineCtx*`、`HcclChannel*`、`HcclRankGraph*` 等） | `HcclChannelContextBuilder::Build()` 首次调用 |
| `acl_common.h` | `libacl_rt.so` | `aclrtGetSocName` | `GetSocName()` 首次调用时 |

`GetSocName()` 在以下场景中被调用：后端解析（`ResolveBackend` 收到字典参数时）、KFC 模式的算法配置选择、KFC 模式的 Buffer 收集判断。首次调用时触发 `InitAclRtFunctions()` 加载 `libacl_rt.so` 并获取 `aclrtGetSocName` 函数指针，返回值通过 `static const char *` 缓存，后续调用直接返回缓存值。

每个 HCCL 初始化函数使用 `static bool` 标志保证只执行一次，后续调用直接跳过。加载失败时通过 `TORCH_CHECK` 抛出异常。

---

## 7. Python 侧接口

### 7.1 CommContextManager

```python
class CommContextManager:
    def __init__(self, group: str, worldSize: int, backend: Union[str, dict] = "kfc"): ...
    def create_context(self) -> torch.Tensor: ...
    def update_context(self, contextTensor: torch.Tensor) -> None: ...
    @property
    def ccl_buffer_size(self) -> int: ...
```

- **`create_context()`**：在 NPU 上分配一个 int32 Tensor，填充通信上下文数据，返回该 Tensor。用于首次创建通信上下文。首次调用时会触发后端解析，根据 `backend` 参数确定使用 KFC 还是 Channel 模式。
- **`update_context(tensor)`**：将通信上下文数据填充到已有的 Tensor 中。用于更换通信组后重新填充上下文（Tensor 可复用，无需重新分配）。
- **`ccl_buffer_size`**：返回 CCL Buffer 的大小，供后续 dispatch/combine 算子分配通信缓冲区时使用。

### 7.2 使用示例

```python
# 字符串方式 - 直接指定后端
manager = CommContextManager(group_name, world_size, backend="kfc")
ctx = manager.create_context()

# 字典方式 - 根据芯片自动选择
manager = CommContextManager(group_name, world_size, backend={
    "Ascend910B": "kfc",
    "Ascend910_93": "kfc",
    "Ascend950": "channel"
})
ctx = manager.create_context()

# 更新已有 context（换通信组后）
manager = CommContextManager(new_group_name, world_size, backend=backend)
manager.update_context(existing_tensor)
```

---

## 8. 错误处理

所有错误均通过 `TORCH_CHECK` 抛出 Python 异常，不返回错误码。错误信息包含具体的 HCCL 返回值，便于定位问题。

| 错误场景 | 错误信息 |
|---------|---------|
| 无效后端字符串 | `backend string must be 'kfc' or 'channel', got 'xxx'` |
| 空字典配置 | `backend dict must not be empty` |
| 无效字典值 | `backend dict value must be 'kfc' or 'channel', got 'xxx'` |
| SoC 不匹配任何字典 key | `No matching SoC name found for 'xxx' in backend dict` |
| backend 参数类型错误 | `backend must be a string ('kfc' or 'channel') or a dict` |
| HCCL API 调用失败 | 含 API 语义描述和 `ret` 错误码，如 `Create HCCL context memory failed, ret: xxx` |
| 动态库加载失败 | `getAclrtGetSocName failed.` 等 |
| 上下文标签过长 | `Mc2ContextTag is too long, max size is 255, got xxx` |
| 链路不支持 UB_MEM 协议 | `No HCCL links support UB_MEM srcRankID xxx, dstRankID xxx, layer xxx` |

---

## 9. 文件清单

| 文件 | 作用 |
|------|------|
| `common/inc/hccl_common.h` | HCCL 控制面 API 动态加载（`libhccl.so` / `libhccl_fwk.so`） |
| `common/inc/acl_common.h` | ACL 运行时 API 动态加载（`libacl_rt.so`，`aclrtGetSocName`） |
| `ops/csrc/update_context.cpp` | 核心实现：`CommContextManager`、`KfcContextBuilder`、`HcclChannelContextBuilder` |
| `ops/update_context.py` | Python 桥接：加载 C++ 扩展，导出 `CommContextManager` |
| `ops/deep_ep.py` | 调用方：`MoeDistributeBuffer`，低延迟 MoE 通信 |
| `ops/mega_moe.py` | 调用方：`SymmBuffer`，MegaMoE 通信 |
