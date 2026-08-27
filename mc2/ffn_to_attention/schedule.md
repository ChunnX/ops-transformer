# FFNToAttention：2 小时学习计划与代码解析

`FFNToAttention` 不做 Attention 计算，而是根据每个 token 的路由元数据，把 FFN 输出写入目标 Attention Worker 的 HCCL 通信窗口。

## 2 小时学习计划

| 时间 | 阅读内容 | 目标 |
|---|---|---|
| 0–15 分钟 | README、接口文档 | 明确算子解决什么问题、输入分别表示什么 |
| 15–30 分钟 | API、算子注册、调用样例 | 看懂用户调用如何进入算子 |
| 30–55 分钟 | Tiling 数据结构与 Host Tiling | 掌握参数校验、窗口大小、TilingKey、核数设置 |
| 55–90 分钟 | Kernel `Init` 和 `Process` | 看懂 token 如何分核、寻址并写入远端窗口 |
| 90–105 分钟 | Graph Task、HCCL 配置 | 理解 MC2/KFC 如何建立通信执行环境 |
| 105–120 分钟 | 用一个 token 手算地址，完成复盘 | 能脱离代码解释完整数据流 |

## 第一阶段：先建立整体模型（0–15 分钟）

阅读：

- [README.md](./README.md)
- [aclnnFFNToAttention.md](./docs/aclnnFFNToAttention.md)

### 算子输入

设本卡最多有 `Y` 个待发送 token：

| 输入 | Shape | 含义 |
|---|---:|---|
| `x` | `[Y, H]` | FFN 输出，每行是一个 token |
| `sessionIds` | `[Y]` | token 属于哪个 Attention Worker |
| `microBatchIds` | `[Y]` | 目标 micro batch |
| `tokenIds` | `[Y]` | token 在 batch 中的位置 |
| `expertOffsets` | `[Y]` | token 对应的 expert/top-k 槽位 |
| `actualTokenNum` | `[1]` | 本次真正有效的 token 数量 |
| `attnRankTable` | `[A]`，可选 | Attention Worker ID 到物理 rank 的映射 |

两个重要属性：

```text
tokenInfoTableShape = [microBatchNum, BS, expertNumPerToken]
tokenDataShape      = [microBatchNum, BS, expertNumPerToken, HS]
```

其中：

- `H`：真正复制的 token 数据长度。
- `HS`：通信窗口中每个槽位的跨度，通常可包含 `hidden + scale`。
- 因此一般要求 `HS >= H`；Kernel 写 `H` 个元素，但用 `HS` 计算槽位地址。

### 输出在哪里？

这个算子没有普通输出 Tensor。

结果直接写入目标 Attention rank 的通信窗口：

```text
目标 rank 的 Window
├── tokenInfoTable：对应槽位置 1，表示数据已就绪
└── tokenData：存放 x[token, 0:H]
```

这也解释了为什么 infer-shape 函数什么都没做。

## 第二阶段：调用入口（15–30 分钟）

这一阶段不需要平均阅读四个文件。按照“2 个精读、1 个扫读、1 个跳读”处理即可：

| 文件 | 阅读方式 | 建议时间 |
|---|---|---:|
| [test_aclnn_ffn_to_attention.cpp](./examples/test_aclnn_ffn_to_attention.cpp) | 只看参数构造和两段式调用 | 7 分钟 |
| [aclnn_ffn_to_attention.cpp](./op_api/aclnn_ffn_to_attention.cpp) | 精读两个公开函数 | 5 分钟 |
| [ffn_to_attention_def.cpp](./op_host/ffn_to_attention_def.cpp) | 扫一遍输入、属性和 MC2 注册 | 2 分钟 |
| [ffn_to_attention_infershape.cpp](./op_host/ffn_to_attention_infershape.cpp) | 直接看结论 | 1 分钟 |

### 2.1 调用样例：只看三段

文件：[test_aclnn_ffn_to_attention.cpp](./examples/test_aclnn_ffn_to_attention.cpp)

不需要从第一行开始逐行看。重点从 `LaunchOneProcessFFN2Attention()` 开始。

#### 第一段：构造逻辑 Shape

```cpp
int64_t micro_batch_num = 1;
int64_t Y = 8;
int64_t H = 7168;
int64_t K = 7;
int64_t sharedExpertNum = 1;

int64_t expert_num_per_token = K + sharedExpertNum;

int64_t Token_info_shape[] = {
    micro_batch_num,
    Y,
    expert_num_per_token
};

int64_t Token_data_shape[] = {
    micro_batch_num,
    Y,
    expert_num_per_token,
    H
};
```

在这个例子中：

```text
tokenInfoTableShape = [1, 8, 8]
tokenDataShape      = [1, 8, 8, 7168]
```

意味着目标 Attention Window 为每个 micro batch 准备：

```text
8 个 token 位置 × 每 token 8 个 expert 位置
```

每个数据槽能放 `7168` 个 BF16 元素。

#### 第二段：构造输入 Tensor

重点关注路由元数据的初始化：

```cpp
sessionIdsHostData[i] = i % attention_worker_num;
tokenIdsHostData[i] = i % Y;
expertOffsetsHostData[i] = i % expert_num_per_token;
```

对于第 `i` 个 token：

```text
目标 Attention Worker = i % 8
目标 token 槽位        = i % 8
目标 expert 槽位       = i % 8
```

例如 `i = 3`：

```text
sessionId    = 3
tokenId      = 3
expertOffset = 3
```

它会被写入 Attention Worker 3 的：

```text
[microBatch=0, token=3, expert=3]
```

`microBatchIdsHostData` 初始化后没有再修改，因此全部为 `0`。

随后，`CreateAclTensor()` 将 Host 数据搬到 Device，并封装成 `aclTensor`。这部分理解目的即可，不必逐行研究内存申请。

#### 第三段：两段式调用

这是样例中最应该精读的部分：

```cpp
ret = aclnnFFNToAttentionGetWorkspaceSize(
    x,
    sessionIds,
    microBatchIds,
    tokenIds,
    expertOffsets,
    actualTokenNum,
    attnRankTable,
    hcomName,
    WORLD_SIZE,
    tokenInfoTableShape,
    tokenDataShape,
    &workspaceSize,
    &executor);
```

第一阶段并不执行 Kernel，它负责：

```text
校验参数
→ 准备算子执行流程
→ 完成 Host Tiling
→ 返回 workspaceSize 和 executor
```

之后根据返回值申请 workspace：

```cpp
aclrtMalloc(&workspaceAddr, workspaceSize, ...);
```

第二阶段：

```cpp
ret = aclnnFFNToAttention(
    workspaceAddr,
    workspaceSize,
    executor,
    stream);
```

这里才把准备好的任务提交到 stream。

#### 样例尾部只需知道什么？

```cpp
HcclCommInitAll(WORLD_SIZE, devices, comms);
```

它为 16 个 rank 建立 HCCL 通信器。随后，程序为每个 rank 启动一个执行 `LaunchOneProcessFFN2Attention()` 的线程。

```text
主线程
  ├─ 初始化 16-rank HCCL 通信域
  ├─ rank 0 线程
  ├─ rank 1 线程
  ├─ ...
  └─ rank 15 线程
```

内存释放和错误处理宏可以跳过。

### 2.2 API 实现：理解“薄封装”

文件：[aclnn_ffn_to_attention.cpp](./op_api/aclnn_ffn_to_attention.cpp)

这个文件虽然是公开 API 的实现，但本身不包含搬运算法。

#### 参数检查

`CheckNullStatus()` 检查：

- 六个必选 Tensor 不能为空；
- `group` 不能为空或空字符串；
- `attnRankTable` 没有检查，因为它是可选输入。

`CheckParams()` 继续检查通信域名称长度。

注意：Shape、dtype、format 等复杂校验不在这里，而是在 Host Tiling 中。

#### 第一段接口

```cpp
aclnnFFNToAttentionGetWorkspaceSize(...)
{
    CheckParams(...);
    return aclnnInnerFFNToAttentionGetWorkspaceSize(...);
}
```

它只负责公开接口的轻量校验，然后转发到内部生成的 `aclnnInnerFFNToAttentionGetWorkspaceSize()`：

```text
公开 aclnn API
    ↓
参数非空检查
    ↓
aclnnInner 接口
    ↓
框架构造 executor
    ↓
Host Tiling
```

#### 第二段接口

```cpp
aclnnFFNToAttention(...)
{
    NnopbaseSetHcclServerType(
        executor,
        NNOPBASE_HCCL_SERVER_TYPE_MTE);

    return aclnnInnerFFNToAttention(...);
}
```

`NnopbaseSetHcclServerType(..., MTE)` 告诉执行框架，该 MC2 算子的 HCCL Server 使用 MTE 类型。随后，`aclnnInnerFFNToAttention()` 将 executor 中准备好的任务提交到 stream。

整个 API 文件可以总结成：

> 第一段检查参数并构建 executor；第二段选择 MTE 通信模式并提交 executor。

### 2.3 算子定义：只看接口契约

文件：[ffn_to_attention_def.cpp](./op_host/ffn_to_attention_def.cpp)

这里定义的是框架如何认识这个算子。

#### 输入注册

```text
x                  FP16/BF16，必选
session_ids        INT32，必选
micro_batch_ids    INT32，必选
token_ids          INT32，必选
expert_offsets     INT32，必选
actual_token_num   INT64，必选
attn_rank_table    INT32，可选
```

全部要求 ND，并通过 `AutoContiguous()` 转换成连续内存。

#### 属性注册

```text
group
world_size
token_info_table_shape
token_data_shape
```

这四项均为必选属性。

#### 两个注册点

```cpp
this->MC2().HcclGroup({"group"});
```

表示 `group` 是 MC2/HCCL 通信域属性。

```cpp
OP_ADD(FFNToAttention);
```

把算子注册进框架。其余 `DynamicFormatFlag`、`jitCompile.flag` 等配置，在两小时学习中可以跳过。

### 2.4 InferShape：看完三行即可

文件：[ffn_to_attention_infershape.cpp](./op_host/ffn_to_attention_infershape.cpp)

Shape 和 dtype 推导函数都直接返回：

```cpp
return ge::GRAPH_SUCCESS;
```

原因是该算子没有输出 Tensor：

```text
普通算子：输入 Tensor → 计算 → 输出 Tensor
本算子：输入 Tensor → 搬运 → 修改 HCCL Window
```

因此：

- 没有输出 Shape 需要推导；
- 没有输出 dtype 需要推导；
- 保留这两个空函数只是为了满足框架注册流程。

读到这里即可关闭文件。

### 第二阶段压缩记忆

```text
样例准备 Tensor 和 HCCL 通信域
          ↓
GetWorkspaceSize 检查参数、构造 executor
          ↓
算子定义告诉框架输入和通信域是什么
          ↓
Host Tiling 完成详细校验与配置
          ↓
aclnnFFNToAttention 设置 MTE 模式并提交任务
          ↓
Kernel 写远端 HCCL Window
```

如果只有 10 分钟，精读样例的 `LaunchOneProcessFFN2Attention()` 和 API 的两个公开函数；另外两个文件知道上述结论即可。

## 第三阶段：Host Tiling（30–55 分钟）

### 发货类比：Host Tiling 是发货调度员

从这一阶段开始，可以把整个算子想象成一次跨仓库发货：

| 代码概念 | 发货类比 |
|---|---|
| Host Tiling | 发货调度员 |
| Kernel / AIV | 真正搬货的工人 |
| token | 一件货物 |
| 路由元数据 | 货物上的收货标签 |
| Attention rank | 目标仓库 |
| HCCL Window | 目标仓库的收货区 |
| TilingData | 调度员交给搬运工的作业单 |

Host Tiling 自己不搬数据。它在发货前完成以下六步：

```text
读取并检查发货订单
        ↓
检查货物和收货标签
        ↓
确认目标仓库容量
        ↓
准备本地临时操作区
        ↓
规划跨仓库运输网络
        ↓
选择搬运方案并安排工人
```

重点阅读：

- [ffn_to_attention_tiling.h](./op_kernel/ffn_to_attention_tiling.h)
- [ffn_to_attention_tilling.cpp](./op_host/op_tiling/ffn_to_attention_tilling.cpp)
- [ffn_to_attention_tilling_key.h](./op_kernel/ffn_to_attention_tilling_key.h)

注意仓库中的文件名确实拼成了 `tilling`。

### Tiling 传给 Kernel 的内容

核心结构为：

```cpp
struct FFNToAttentionInfo {
    uint32_t H;
    uint32_t A;
    uint32_t microBatchNum;
    uint32_t BS;
    uint32_t expertNumPerToken;
    uint32_t HS;
    uint32_t aivNum;
    uint32_t worldSize;
    bool isInputRankTable;
    uint8_t windowType;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};
```

把它理解成 Kernel 的运行配置：

- 地址计算参数：`microBatchNum / BS / expertNumPerToken / HS`
- 复制长度：`H`
- 并行参数：`aivNum`
- 路由模式：`isInputRankTable`
- 通信配置：`worldSize / windowType`

### Host Tiling 的主流程

`FFNToAttentionTilingFunc` 做六件事：

```text
校验并提取属性
      ↓
校验输入 shape / dtype / format
      ↓
计算通信窗口所需大小
      ↓
设置 workspace
      ↓
生成 HCCL All-to-All 配置
      ↓
选择 TilingKey 并设置 AIV 核数
```

六步的形象含义为：

| Host Tiling 步骤 | 发货系统中的含义 | 主要产出 |
|---|---|---|
| 校验并提取属性 | 阅读订单，确认通信车队和目标仓库布局 | `group`、`worldSize`、`BS`、`HS` 等 |
| 校验输入 | 检查货物数量、包装和收货标签是否合法 | `H`、路由模式及合法输入 |
| 计算 Window 大小 | 确认收货方仓库放得下全部货物 | 所需及可用 Window 大小 |
| 设置 Workspace | 给发货中心预留本地临时操作场地 | Device Workspace 大小 |
| 生成 HCCL 配置 | 规划各仓库之间的运输网络 | MC2/HCCL All-to-All 配置 |
| 选择 TilingKey 和 AIV | 选择搬运程序并安排搬运工数量 | Kernel 版本、Block 数、`aivNum` |

窗口需求近似为：

```text
info bytes =
    microBatchNum × BS × expertNumPerToken × sizeof(int32)

data bytes =
    microBatchNum × BS × expertNumPerToken × HS × sizeof(xType)
```

Host 会确认两者之和没有超过 HCCL 可用窗口。

#### 关键代码 1：提取并校验属性

**发货类比：读取发货订单。** 调度员首先要知道使用哪个通信车队、共有多少个仓库，以及收货区的货架如何排列。

`CheckAndSetAttrs()` 先从算子属性中取得通信域、rank 数和两张逻辑表的 Shape：

```cpp
auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);
auto tokenInfoTableShape =
    attrs->GetListInt(ATTR_TOKEN_INFO_TABLE_SHAPE_INDEX)->GetData();
auto tokenDataShape =
    attrs->GetListInt(ATTR_TOKEN_DATA_SHAPE)->GetData();
```

这里读到的不是输入 Tensor Shape，而是调用者传入的四个算子属性。随后检查两张表前三维一致：

```cpp
tokenInfoTableShape[0] == tokenDataShape[0];
tokenInfoTableShape[1] == tokenDataShape[1];
tokenInfoTableShape[2] == tokenDataShape[2];
```

这是因为 `tokenInfoTable[m, b, e]` 和 `tokenData[m, b, e, :]` 必须描述同一个槽位。校验通过后，将维度写入 TilingData：

```cpp
tilingData.ffnToAttentionInfo.worldSize = *worldSizePtr;
tilingData.ffnToAttentionInfo.microBatchNum = tokenDataShape[0];
tilingData.ffnToAttentionInfo.BS = tokenDataShape[1];
tilingData.ffnToAttentionInfo.HS = tokenDataShape[3];
tilingData.ffnToAttentionInfo.expertNumPerToken =
    tokenInfoTableShape[2];
```

这些字段之后会原样传到 Kernel，用于远端 Window 寻址。

可以把两张表理解成同一套货架对应的“登记表”和“实物区”：

```text
tokenInfoTable[m, b, e]    该格是否已经到货
tokenData[m, b, e, :]      该格存放的实际 token 数据
```

两张表前三维不一致，就像到货登记表与真实货架编号对不上，不能继续发货。

#### 关键代码 2：校验输入并确定路由模式

**发货类比：检查货物及其标签。** `x` 是货物，四张 ID 表是贴在每件货物上的收货标签；数量、类型或排列方式不合法时，搬运工无法安全工作。

```cpp
const gert::StorageShape *attnRankTableShape =
    context->GetOptionalInputShape(INPUT_ATTN_RANK_TABLE_INDEX);
bool isInputRankTable = (attnRankTableShape != nullptr);

tilingData.ffnToAttentionInfo.isInputRankTable = isInputRankTable;
```

可选输入是否存在，会决定后面选择哪个 Kernel 模板：

```text
不存在：sessionId 就是物理 rank
存在：  物理 rank = attnRankTable[sessionId]
```

`CheckInputDim0Dim1()` 还会把 `x.shape[1]` 保存为 `H`：

```cpp
tilingData.ffnToAttentionInfo.H = xDim1;
```

因此 `H` 来自输入 `x`，而 `HS` 来自通信窗口属性 `tokenDataShape`。

其中：

```text
H  = 每件货物实际有多大
HS = 目标仓库为每件货物预留的格子有多大
```

`attnRankTable` 则像地址簿：没有地址簿时，`sessionId` 直接是仓库编号；有地址簿时，要先查询真实仓库地址。

#### 关键代码 3：计算通信窗口大小

**发货类比：确认目标仓库容量。** 在卡车出发前，调度员先计算登记区和实物区一共需要多少空间，避免搬到一半才发现收货区放不下。

```cpp
uint64_t tokenDataNeedWinSize =
    microBatchNum * BS * expertNumPerToken * HS * TOKEN_DATA_SIZE;

uint64_t tokenInfoTableNeedWinSize =
    microBatchNum * BS * expertNumPerToken * TOKEN_INFO_SIZE;

neededSize =
    (tokenDataNeedWinSize + tokenInfoTableNeedWinSize) / MB_SIZE + 1;
```

其中 `TOKEN_DATA_SIZE = 2`，因为 FP16/BF16 都占 2 字节；`TOKEN_INFO_SIZE = 4`，因为 ready 标志是 INT32。若 `neededSize` 大于 HCCL 可用 Window，Tiling 直接失败，Kernel 不会启动。

注意这里是在检查已有 HCCL Window 是否足够，并不是在这一步创建远端 Window。

#### 关键代码 3.5：设置 Workspace

**发货类比：准备本地临时操作场地。**

```cpp
size_t *workSpaces = context->GetWorkspaceSizes(1);
auto ascendcPlatform =
    platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
```

三个空间不要混淆：

```text
HCCL Window = 收货方的远端仓库
Workspace   = 发货中心的本地临时操作区
UB          = 每个搬运工手里的推车
```

该 Kernel 虽然没有大量直接使用 `workspaceGM`，但 AscendC/MC2 运行库仍可能需要系统临时空间，所以 Host 必须按平台设置 Workspace 大小。

#### 关键代码 4：配置通信和执行核数

**发货类比：先规划运输网络，再安排搬运工。** HCCL 配置决定各 rank 之间如何运输；AIV 配置决定本卡派多少名工人并行搬货。

```cpp
std::string algConfigAllToAllStr =
    "AlltoAll=level0:fullmesh;level1:pairwise";

AscendC::Mc2CcTilingConfig mc2CcTilingConfig(
    group, OP_TYPE_ALL_TO_ALL, algConfigAllToAllStr);

mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);
```

这段为 MC2 生成 HCCL All-to-All 通信配置。随后读取设备 AIV 核数：

```cpp
uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
uint32_t numBlocks =
    ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

context->SetBlockDim(numBlocks);
tilingData->ffnToAttentionInfo.aivNum = aivNum;
```

Host 在这里决定启动多少个 Block，并把实际 AIV 数告诉 Kernel，Kernel 再据此分摊 token。

### TilingKey

**发货类比：选择搬运操作手册。** Host 不在运行时临时发明搬法，而是从预先编译好的 Kernel 版本中选出符合当前硬件和路由模式的版本。

当前只生成 A3 的两个 Kernel 模板实例：

```text
RankTableMode = 0：sessionId 直接作为目标 rank
RankTableMode = 1：目标 rank = attnRankTable[sessionId]
```

因为是否有 `attnRankTable` 是 Host 侧已知条件，用模板分支可以避免 Kernel 循环中的运行时判断。

对应的关键代码是：

```cpp
bool rankTableMode =
    tilingData->ffnToAttentionInfo.isInputRankTable;

const uint64_t tilingKey =
    GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A3);

context->SetTilingKey(tilingKey);
```

可以把 TilingKey 理解为 Host 对 Kernel 说：“请选择 A3 架构下，带或不带 RankTable 的那个编译版本。”

## 第四阶段：Kernel（55–90 分钟）

### 发货类比：Kernel 是照作业单搬货的工人

Host 已经检查订单、规划空间并安排人手；Kernel 不再重复做全局规划，只按照 TilingData 执行：

```text
确认自己负责哪些 token
        ↓
读取每件货物的收货标签
        ↓
找到目标 rank 的远端 Window
        ↓
把 token 写入指定 Data 格子
        ↓
把对应 Info 格标记为已到货
```

重点文件：

- [ffn_to_attention.cpp](./op_kernel/arch22/ffn_to_attention.cpp)
- [ffn_to_attention.h](./op_kernel/ffn_to_attention.h)

Kernel 入口只做三件事：

```cpp
FFNToAttention<DTYPE_X, RankTableMode> op;
op.Init(...);
op.Process();
```

真正逻辑全部在头文件里。

入口中的模板参数也值得看懂：

```cpp
template <bool RankTableMode, uint8_t ArchTag>
__global__ __aicore__ void ffn_to_attention(...)
```

- `RankTableMode`：是否需要查询 `attnRankTable`；
- `ArchTag`：硬件架构；当前实现只在 `TILINGKEY_TPL_A3` 分支实例化算子；
- `DTYPE_X`：由编译框架根据输入选择 FP16 或 BF16。

### `Init` 做什么？

`Init`：

1. 获取当前 AIV 编号及 HCCL Context；
2. 绑定所有 GlobalTensor；
3. 从 GM 读取 `actualTokenNum`；
4. 给 token 搬运申请双缓冲；
5. 计算地址寻址所需的步长。

关键步长：

```cpp
batchSizeSendCnt_ = BS * expertNumPerToken;

perMicroBatchOffset_ =
    BS * expertNumPerToken * HS;

perExpertOffset_ =
    expertNumPerToken * HS;
```

通信窗口前面放 info 表，因此 data 区开始位置是：

```cpp
winTokenInfoTableSize_ =
    Align512(microBatchNum * BS * expertNumPerToken * sizeof(int32_t));
```

这里的 512 字节对齐非常重要。

#### `Init` 关键代码解析

**发货类比：工人领取作业单、推车和仓库地址簿。** `aivId_` 是工人编号，HCCL Context 是远端仓库地址信息，双缓冲队列则是工人使用的两辆周转推车。

```cpp
aivId_ = GetBlockIdx();
winContext_ = (__gm__ HcclOpResParam *)
    AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
```

`aivId_` 是当前执行核编号；`winContext_` 保存 HCCL 通信资源，稍后通过它寻找目标 rank 的远端 Window。

```cpp
xGMTensor_.SetGlobalBuffer((__gm__ xType *)x);
sessionIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sessionIds);
// 其余输入同理
```

`SetGlobalBuffer()` 只是把裸 GM 地址包装成可通过下标访问的 `GlobalTensor`，此时尚未批量搬运数据。

```cpp
DataCacheCleanAndInvalid<int64_t, ...>(actualTokenNumGMTensor_);
actualTokenNum_ = actualTokenNumGMTensor_.GetValue(0);
```

先刷新缓存，再读取实际 token 数，避免读到 CPU 或其他执行单元留下的旧缓存值。

```cpp
tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hSize_);
tpipe_->InitBuffer(statusBuf_, UB_ALIGN);
statusTensor_(0) = 1;
```

`xQueue_` 为 token 搬运申请双缓冲；`statusTensor_` 保存固定值 `1`，用来写 ready 标志。

### `Process` 如何分核？

**发货类比：调度员按编号轮流分货。** 不是给每名工人一整段连续货物，而是按 AIV 编号交错分配，以便让工作量尽可能均衡。

每个 AIV 处理交错 token：

```cpp
YOffset = aivId + tokenCnt * aivNum;
```

也就是：

```text
AIV 0：token 0, N, 2N ...
AIV 1：token 1, N+1, 2N+1 ...
```

其中 `N = aivNum`。

这是一种循环划分，不是每核分到一段连续 token。

对应源码先计算每核循环次数：

```cpp
uint32_t sendTokenNum = actualTokenNum_ / aivNum_;
uint32_t remainderRankNum = actualTokenNum_ % aivNum_;

if (aivId_ < remainderRankNum) {
    sendTokenNum += 1;
}
```

前 `remainderRankNum` 个 AIV 多处理一个 token，从而使各核工作量最多只相差 1。循环内再通过：

```cpp
uint32_t YOffset = aivId_ + tokenCnt * aivNum_;
```

取得当前核负责的 token 编号。

### 每个 token 的处理过程

**发货类比：读标签、找仓库、找货架、放货。** 四张元数据表共同给出一件 token 应该送到哪个物理仓库，以及仓库内的哪个格子。

```text
读取 x[YOffset, :]
        ↓
读取四项路由元数据
        ↓
解析目标 Attention rank
        ↓
取得目标 rank 的远端 Window 地址
        ↓
计算 info 和 data 槽位
        ↓
先写 token data
        ↓
再写 info 标志 1
```

目标 rank：

```cpp
targetRank = sessionIds[YOffset];

if constexpr (isInputRankTable) {
    targetRank = attnRankTable[targetRank];
}
```

源码中的 `ReadTokenMetaData()` 对应这一步：

```cpp
metaDataStruct.curAttenWorkIds = sessionIdsGMTensor_(YOffset);
metaDataStruct.curAttenWorkRank = metaDataStruct.curAttenWorkIds;

if constexpr (isInputRankTable) {
    metaDataStruct.curAttenWorkRank =
        attnRankTableGMTensor_(metaDataStruct.curAttenWorkIds);
}

metaDataStruct.curMicroBatchIds = microBatchIdsGMTensor_(YOffset);
metaDataStruct.curTokenBatchOffset = tokenIdsGMTensor_(YOffset);
metaDataStruct.curTokenTopkOffset = expertOffsetsGMTensor_(YOffset);
```

四张一维元数据表共同把 `x[YOffset]` 映射为：

```text
(目标物理 rank, microBatch, token 槽, expert 槽)
```

#### 获取远端 Window

**发货类比：根据目标仓库编号找到收货区大门。** `GetWindowAddr()` 返回的不是普通输出 Tensor 地址，而是目标 rank 暴露的通信内存基址。

```cpp
if (winContext_->userMemType == 0) {
    return remoteRes[targetRank].windowsIn;
} else {
    return userMemRes[targetRank].addr;
}
```

`GetWindowAddr()` 根据内存模式，从 HCCL Context 中取得目标 rank 的远端内存基址。后面的 `DataCopyPad()` 因而不是写普通输出，而是在写通信窗口。

目标逻辑坐标：

```text
(microBatchId, tokenId, expertOffset)
```

数据槽的元素偏移：

```text
dataIndex =
    microBatchId × BS × expertNumPerToken × HS
  + tokenId      × expertNumPerToken × HS
  + expertOffset × HS
```

即：

```text
dataIndex =
    ((microBatchId × BS + tokenId)
      × expertNumPerToken + expertOffset)
    × HS
```

实际地址：

```text
windowBase
+ alignedInfoTableBytes
+ dataIndex × sizeof(xType)
```

Info 槽偏移：

```text
infoIndex =
    (microBatchId × BS + tokenId)
    × expertNumPerToken
    + expertOffset
```

对应的源码地址计算是：

```cpp
uint64_t tokenInfoTableOffset =
    curMicroBatchId * batchSizeSendCnt_;

uint64_t tokenDataOffset =
    (curMicroBatchId * perMicroBatchOffset_
     + curTokenBatchOffset * perExpertOffset_
     + curTokenTopkOffset * axisHS_)
    * sizeof(xType);

tokenDataGMTensor.SetGlobalBuffer(
    (__gm__ xType *)(curRankWinAddr
                     + winTokenInfoTableSize_
                     + tokenDataOffset));
```

先跳过按 512 字节对齐的 Info 区，再加 Data 槽偏移，最终定位到目标 token 的写入地址。

### 为什么先写 data、再写 info？

**发货类比：必须先把货放进货架，再在登记表上签收。** 如果先写 ready 标志，Attention 侧可能看到“已到货”后立即读取，但此时 Data 仍未搬完。

Kernel 中顺序是：

```cpp
DataCopyPad(tokenData, x);
PipeBarrier<PIPE_MTE3>();
DataCopyPad(tokenInfo, statusOne);
```

`tokenInfo = 1` 相当于发布标志。Attention 侧看到它时，token 数据应该已经完成写入。

可以把它理解为一个生产者协议：

```text
FFN：写数据 → 发布 ready
Attention：检查 ready → 读取数据
```

这是整个 Kernel 最重要的同步语义。

完整搬运片段还包含队列操作：

```cpp
xTmpTensor_ = xQueue_.AllocTensor<xType>();
DataCopyPad(xTmpTensor_, xGMTensor_[YOffset * axisH_], xCopyParams, ...);
xQueue_.EnQue(xTmpTensor_);

xTmpTensor_ = xQueue_.DeQue<xType>();
DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams);
xQueue_.FreeTensor<xType>(xTmpTensor_);

PipeBarrier<PIPE_MTE3>();
DataCopyPad(tokenInfoTableGMTensor, statusTensor_, dataCopyParams);
```

它体现了完整的数据通路：

```text
输入 GM → AIV 的 UB 队列 → 目标 rank Window 的 Data 区
                              → Barrier
                              → 目标 rank Window 的 Info 区写 1
```

## 第五阶段：Graph Task 与通信环境（90–105 分钟）

### 发货类比：Graph Task 是运输公司的后台调度系统

Host Tiling 写的是本次发货作业单，Kernel 负责搬货，而 Graph Task 负责让通信服务、运输流和运行时任务真正进入可调度状态：

```text
KFC Server       = 运输调度中心
kfc_stream       = 可复用的运输通道
MC2 tasks        = 运行时运输任务单
GenerateTask     = 将计划转换成可执行任务
```

阅读：

- [ffn_to_attention_gen_task.cpp](./op_graph/ffn_to_attention_gen_task.cpp)

这里不是数据搬运算法本身，而是 MC2 的执行编排：

- `CommonKFCMc2CalcParamFunc`：准备或复用 KFC 通信 stream；
- `Mc2MoeGenTaskCallbackV2`：生成 MC2/MoE 通信相关任务；
- Host Tiling 中将通信算法配置为 All-to-All；
- Kernel 再通过 `GetHcclContext` 获取各 rank 的 Window。

### 关键代码 1：准备 KFC 通信资源

**发货类比：启动或复用运输调度中心及专用车道。** 这一步只准备通信执行环境，还没有搬运任何 token。

```cpp
ge::Status FFNToAttentionCalcParamFunc(
    gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";

    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(
        context, name, reuseKey);
}
```

这里准备 MC2 所需的 KFC Server，并通过 `kfc_stream` 复用通信流。它负责通信执行环境，不负责计算 token 地址。

### 关键代码 2：生成 MC2 任务

**发货类比：把调度计划转换成运输系统真正能执行的任务单。** 运行时之后才能按这些任务启动通信服务和 AICore Kernel。

```cpp
ge::Status FFNToAttentionGenTaskFunc(
    const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(
        context, tasks);
}
```

这里把算子转换成运行时可调度的 MC2/MoE 任务。最后通过：

```cpp
IMPL_OP(FFNToAttention)
    .CalcOpParam(FFNToAttentionCalcParamFunc)
    .GenerateTask(FFNToAttentionGenTaskFunc);
```

把“资源准备”和“任务生成”两个回调注册到 `FFNToAttention`。

所以完整调用链是：

```text
aclnn 两段式 API
  → 算子图注册
  → Host Tiling
  → KFC/MC2 通信任务
  → AICore Kernel
  → 直接写目标 rank 的 HCCL Window
```

## 最后 15 分钟：手算一个 token

### 发货类比：亲手填写一张货物入库单

这一阶段选择一件 token，按照它的三个货架坐标计算最终位置：

```text
microBatchId  = 仓库分区
tokenId       = token 货架编号
expertOffset  = 货架内的 expert 格编号
```

手算目的不是记公式，而是验证自己能从“收货标签”一路找到远端 Window 中唯一的 Data 格和 Info 格。

假设：

```text
microBatchNum = 1
BS = 8
expertNumPerToken = 2
H = 4096
HS = 4224

microBatchId = 0
tokenId = 3
expertOffset = 1
```

则槽位编号：

```text
slot = (0 × 8 + 3) × 2 + 1 = 7
```

Info：

```text
tokenInfoTable[7] = 1
```

Data 起始元素：

```text
dataIndex = 7 × 4224 = 29568
```

Kernel 从这里开始写入 `H=4096` 个元素：

```text
tokenData[29568 : 29568 + 4096]
```

剩余：

```text
HS - H = 128
```

个元素不会被本 Kernel 写入。这通常对应额外的 scale/预留区域，也解释了代码为何同时需要 `H` 和 `HS`。

### 将手算结果代回关键代码

对于 `microBatchId=0`、`tokenId=3`、`expertOffset=1`：

```cpp
perMicroBatchOffset_ = 8 * 2 * 4224;
perExpertOffset_ = 2 * 4224;

tokenDataOffset =
    (0 * perMicroBatchOffset_
     + 3 * perExpertOffset_
     + 1 * 4224)
    * sizeof(xType);
```

化简得到：

```text
tokenDataOffset = 29568 × 2 bytes
                = 59136 bytes
```

因此最终地址是：

```text
目标 rank Window 基址
+ 对齐后的 Info 区大小
+ 59136 字节
```

Kernel 从这个位置复制 `4096 × 2 = 8192` 字节，然后对 MTE3 建立 Barrier，最后把对应 Info 槽写为 `1`。能够将这一步和源码逐项对应，就说明已经真正看懂了核心路径。

## 两小时结束时，你应该能回答

1. 为什么该算子没有输出 Tensor？
2. `sessionIds` 和 `attnRankTable` 有什么区别？
3. `H` 与 `HS` 为什么不是同一个概念？
4. Window 为什么分为 info 和 data 两部分？
5. 如何由 `(microBatchId, tokenId, expertOffset)` 算出目标地址？
6. 为什么必须先写 data，后写 info？
7. AIV 如何分摊 `actualTokenNum` 个 token？
8. Host Tiling 为什么要检查 HCCL Window 大小？

若时间不够，可以略读 `CMakeLists.txt`、InferShape 单测和 API 参数空指针单测；不要略过 Kernel 的 `Process()` 和 Host Tiling 的 `CheckAndSetAttrs()`、`CalWinSize()`、`FFNToAttentionTilingFunc()`。
