# GenericBlockSparseAttention 支持 PA 与非 PA 的设计方案

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 算子 | `GenericBlockSparseAttention` |
| 目标 | 在保持现有 Paged Attention（PA）能力与性能的前提下，新增原始 TND KV（非 PA）执行模式 |
| 参考实现 | `attention/fused_infer_attention_score/op_kernel/arch22/flash_attention_regular.h` |
| 影响范围 | ACLNN/torch 接口校验、Host Tiling、TilingKey、A2/A5 Kernel、Metadata 前置算子、测试与用户文档 |
| 文档状态 | 设计稿 |

## 2. 背景与问题

当前 `GenericBlockSparseAttention` 的公开接口已经预留了 `layout_kv`、`cu_seq_lengths_kv`、`seqused_kv` 和可选 `block_table`，但实际实现固定按 PA 执行：

- Host Tiling 只接受 `layout_q="TND"`、`layout_kv="PA_BBND"`；
- `block_table` 和 `seqused_kv` 被强制要求非空，`cu_seq_lengths_kv` 被强制要求为空；
- Key/Value 固定解释为 `[num_blocks, block_size, Nkv, D]`；
- Kernel 将 `sparse_block_idx` 中的逻辑块号经 `block_table` 转换为物理页号，再计算 K/V 地址；
- TilingKey 只注册了 PA_BBND 对应的 Kernel；
- Metadata 接口的 Host 校验同样只放行 PA_BBND。

这使算子无法直接处理训练后预填充、离线推理或无需分页管理场景中的原始 KV。调用方必须额外构造分页 KV 和 `block_table`，增加数据准备成本，也限制了算子的复用范围。

本方案参考 `flash_attention_regular.h` 的处理方式，以编译期 `PAGED_CACHE_FLAG` 区分 PA 与非 PA，仅在序列长度解析和 K/V 地址解析处做模式分流，复用 QK、Softmax、PV、RescaleO 以及 FD 调度主体。

## 3. 设计目标与非目标

### 3.1 设计目标

1. 保持现有 `TND + PA_BBND` 行为、TilingKey 和性能不变。
2. 新增 `TND + TND` 原始 KV 模式，不要求 `block_table`。
3. 两种模式保持相同的稀疏块语义、因果边界、输出格式和 Metadata 调度协议。
4. PA/非 PA 在 Host 侧确定，在 Kernel 中通过模板参数静态分派，避免热点循环中的运行时判断。
5. 覆盖 Atlas A2（AICore 220）和 Atlas A5（AICore 310）；A5 现有 FP8 全量化路径同步具备非 PA 地址能力。
6. 保留后续扩展 `BSND`、`BNSD`、`PA_BNBD` 的结构空间。

### 3.2 本期非目标

- 不新增 `layout_q`，Query 仍只支持 `TND`。
- 不新增 `layout_kv=BSND/BNSD/PA_BNBD`。
- 不修改 `block_shape=[1, 128]`、`D=128`、`is_packed_gqa=1`、`mask_type=1` 等现有限制。
- 不改变稀疏块选择算法，不在主算子中生成 `sparse_block_idx` 或 `sparse_block_count`。
- 不修改 QK、Softmax、PV、RescaleO 的数学实现及片上流水结构。

## 4. 对外语义

### 4.1 支持矩阵

| `layout_q` | `layout_kv` | `block_table` | KV 形状 | 模式 | 本期状态 |
| --- | --- | --- | --- | --- | --- |
| TND | PA_BBND | 非空 | `[num_blocks, block_size, Nkv, D]` | PA | 保持支持 |
| TND | TND | 空 | `[total_kv_tokens, Nkv, D]` | 非 PA | 新增支持 |
| TND | PA_BBND | 空 | - | 配置矛盾 | 报错 |
| TND | TND | 非空 | - | 配置矛盾 | 报错 |
| 其他 | 任意 | 任意 | - | 本期未支持 | 报错 |

是否启用 PA 以 `block_table` 是否存在为事实来源，`layout_kv` 用于描述内存排布，两者必须交叉校验：

```text
is_paged = (block_table != nullptr)

valid = ( is_paged && layout_kv == "PA_BBND") ||
        (!is_paged && layout_kv == "TND")
```

不建议仅根据 `layout_kv` 推断并忽略 `block_table`，否则错误配置可能进入错误地址路径并产生越界访问。

### 4.2 序列长度输入

| 模式 | `cu_seq_lengths_q` | `seqused_q` | `cu_seq_lengths_kv` | `seqused_kv` |
| --- | --- | --- | --- | --- |
| PA_BBND | 必选，`[B+1]` | 可选，`[B]` | 必须为空 | 必选，`[B]` |
| TND 原始 KV | 必选，`[B+1]` | 可选，`[B]` | 必选，`[B+1]` | 可选，`[B]` |

TND 原始 KV 中：

- `cu_seq_lengths_kv[b+1] - cu_seq_lengths_kv[b]` 表示第 `b` 个 batch 的 KV 存储长度；
- `seqused_kv[b]` 非空时表示实际有效长度，且必须小于等于对应存储长度；
- `seqused_kv` 为空时，实际有效长度等于存储长度；
- 每个 batch 的 padding 位于其 TND 存储片段尾部，与 Q 侧现有双长度语义一致。

### 4.3 稀疏块语义

两种模式下 `sparse_block_idx` 的元素均表示“当前 batch 内、沿逻辑 KV 序列的块号”，而不是物理页号：

```text
logical_block_start = sparse_block_idx[...] * block_shape_y
```

- PA：先计算逻辑 token 位置，再由 `logical_token / blockSize` 查 `block_table`，并保留页内 `logical_token % blockSize` 偏移；当前稀疏块大小恰好等于页大小时，两种块号才相等。
- 非 PA：逻辑块号直接映射到当前 batch 的 TND KV 片段。

此定义保证同一组 `sparse_block_idx/sparse_block_count` 可用于内容相同的 PA 与非 PA 输入，也便于构造两种模式的等价性测试。

### 4.4 默认行为与兼容性

- ACLNN 函数签名不变。
- Torch 扩展的 `layout_kv` 默认值继续保持 `"PA_BBND"`，避免已有调用在省略参数时改变执行模式。
- OpDef 保持原定义不变，`layout_kv` 默认值仍为 `"TND"`。Torch 扩展保留自己的 `"PA_BBND"` 默认值；Torch 非 PA 调用需显式传入 `layout_kv="TND"` 且不传 `block_table`。
- 现有 PA TilingKey 数值不修改，只新增非 PA TilingKey。

## 5. 总体架构

```text
                              Host
  Inputs/Attrs
       |
       v
  Parse layout + block_table existence
       |
       +-----------------------+
       |                       |
       v                       v
  PA_BBND validation      TND KV validation
  seqUsedKv required      cuSeqKv required
  page stride parsing     contiguous stride parsing
       |                       |
       +-----------+-----------+
                   v
          Generate mode-specific TilingKey
                   |
                   v
                              Kernel
       GbsaRegularKernel<..., PAGED_CACHE_FLAG>
                   |
       +-----------+-------------------+
       |                               |
       v                               v
  ResolvePagedBlock()             ResolveTndBlock()
  logical -> physical page        logical -> TND token offset
       |                               |
       +---------------+---------------+
                       v
              Shared QK -> SM -> PV -> RescaleO
```

关键原则是只把“KV 序列描述”和“KV 块地址解析”模式化，计算流水保持一份实现。

## 6. Host Tiling 设计

### 6.1 状态字段

实现复用 `blockTablePresent_` 区分模式，不增加重复状态。

`kStride0_`、`vStride0_` 保持真正的张量 dim0 stride（元素数）：

- PA_BBND：物理页 stride，允许大于 `blockSize * Nkv * D`；
- TND：token stride，即 `Nkv * D`，本期要求 KV 连续。

三个尺寸字段各自只表达一个概念：

| 字段 | 含义 | 当前值 |
| --- | --- | --- |
| `blockShapeY` | 稀疏索引描述的逻辑 KV 块宽度 | 128 |
| `blockSize` | PA 物理页包含的 token 数 | PA 为 128；TND 为 0 |
| `kvBaseTile` | QK/PV 计算 tile 的 KV 宽度 | 独立选择为 128 |

TilingData 字段顺序、大小及已有 PA 字段值不变。TND 不再构造逻辑页 stride，
MMAD 及 buffer 尺寸使用 `kvBaseTile`，避免让物理页大小决定计算形状。

编译期模式已经由 TilingKey 表达，因此无需在 TilingData 中重复新增 `pagedCacheFlag` 字段。

### 6.2 解析顺序

当前 batch 来自 `block_table.shape[0]`，非 PA 时无法成立。建议重排为：

1. 解析 `layout_q/layout_kv` 和 `block_table` 是否存在，确定 `pagedCacheFlag_`；
2. 由 `cu_seq_lengths_q.shape[0] - 1` 得到 batch；
3. 解析 Query 与稀疏索引形状，得到 `Nq/D/Nkv/topK/totalQBlocks`；
4. 按 PA 或非 PA 分支解析 Key/Value 形状；
5. 校验长度输入组合；
6. 解析 KV stride；
7. 生成 Workspace 与 TilingKey。

### 6.3 Layout 与可选输入校验

将 `GetInputLayout` 和 `ParseBlockTable` 合并为一致的模式校验，伪代码如下：

```cpp
pagedCacheFlag_ = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr;

if (layoutQ_ != "TND") {
    return GRAPH_FAILED;
}
if (pagedCacheFlag_) {
    CHECK(layoutKv_ == "PA_BBND");
    CHECK(blockTable is 2D && blockTable.shape[0] == batch_);
    maxBlocksPerBatch_ = blockTable.shape[1];
} else {
    CHECK(layoutKv_ == "TND");
    maxBlocksPerBatch_ = 0;
}
```

长度输入校验调整为：

```cpp
CHECK(cuSeqLengthsQ != nullptr);

if (pagedCacheFlag_) {
    CHECK(sequsedKv != nullptr);
    CHECK(cuSeqLengthsKv == nullptr);
} else {
    CHECK(cuSeqLengthsKv != nullptr);
    // sequsedKv optional
    CHECK(cuSeqLengthsKv.shape == [batch + 1]);
    if (sequsedKv != nullptr) {
        CHECK(sequsedKv.shape == [batch]);
    }
}
```

Host Tiling 无法读取 `cu_seq_lengths_kv` 和 `seqused_kv` 的元素值，因此前缀和单调性、首元素为 0、实际长度不超过存储长度仍由 ACLNN/上游保证；若后续引入 AICPU 值校验，可将该检查前移。

### 6.4 Key/Value 形状校验

PA_BBND：

```text
key.shape == value.shape == [num_blocks, block_size, Nkv, D]
block_size == block_shape_y == 128
Nkv == sparse_block_idx.shape[0]
D == query.shape[2] == 128
```

TND 非 PA：

```text
key.shape == value.shape == [total_kv_tokens, Nkv, D]
Nkv == sparse_block_idx.shape[0]
D == query.shape[2] == 128
```

`blockSize_` 在非 PA 模式下为 0，表示无物理页；计算 tile 从 `kvBaseTile` 读取。

### 6.5 连续性和 stride

- PA_BBND 保留现有能力：仅 dim0 可非连续，页内 `[block_size, Nkv, D]` 必须连续；`kStride0_/vStride0_` 取实际 dim0 stride。
- TND 非 PA 本期要求 Key/Value 全连续；ACLNN 路径调用 `l0op::Contiguous`，直接图调用由 Tiling 校验完整 stride。
- TND 下设置：

```text
kStride0 = vStride0 = Nkv * D            // TND dim0 为 token 轴
kv_row_stride = Nkv * D                  // Kernel 内计算
```

暂不支持 TND KV 的任意 stride，是为了保持现有 MMAD `RowMajor/ColumnMajor` 搬运假设不变。后续若支持任意 stride，需要明确 token/head 两个轴的步长；不能把页 stride 当成 token stride。

### 6.6 TilingKey

沿用现有十进制字段编码：

- Q layout：TND=`2`；
- KV layout：TND=`30`，PA_BBND=`70`；
- PagedCache：非 PA=`0`，PA=`1`；
- dtype、softmax precision、LSE、架构字段保持不变。

必须冻结现有 PA key。例如 AICore 220 FP16、高精度 Softmax、maskType=1 的 PA key 仍为：

```text
9200000071001002
```

对应新增的非 PA key 为：

```text
9200000030001002
```

建议增加独立常量：

```cpp
constexpr uint64_t GBSA_KV_LAYOUT_TND_OFFSET = 30000000ULL;
constexpr uint64_t GBSA_KV_LAYOUT_PA_BBND_OFFSET = 70000000ULL;
constexpr uint64_t GBSA_PAGED_CACHE_OFFSET = 1000000ULL;
```

由同一公式生成 Host key，并在 UT 中断言全部已注册 key 的精确值与唯一性，避免十进制字段相加产生碰撞。

## 7. Kernel 设计

### 7.1 编译期模式分派

参考 `FAInferKernel<..., bool PAGED_CACHE_FLAG, ...>`，为 GBSA Kernel 增加模板参数：

```cpp
template <class BlockMmadQK, class EpilogueOnlineSoftmax,
          class BlockMmadPV, class EpilogueRescaleO,
          bool PAGED_CACHE_FLAG>
class GbsaRegularKernelArch22;

template <class BlockMmadQK, class EpilogueOnlineSoftmax,
          class BlockMmadPV, class EpilogueRescaleO,
          Format qFormat, Format kvFormat,
          bool PAGED_CACHE_FLAG>
class GbsaRegularKernelArch35;
```

A5 FP8 的 `GbsaFullQuantKernelArch35` 同样增加该模板参数。Kernel 入口由 TilingKey 实例化 `<true>` 或 `<false>`，热点循环中使用 `if constexpr`，不引入运行时分支。

### 7.2 KV 序列描述

Kernel 初始化以下 GlobalTensor：

- `gCuSeqLengthsQ`：两种模式均使用；
- `gCuSeqLengthsKv`：仅 `PAGED_CACHE_FLAG=false` 时设置；
- `gSequsedKv`：PA 必选，非 PA 可选；
- `gBlockTable`：仅 `PAGED_CACHE_FLAG=true` 时设置。

每个 task 解析第 `b` 个 batch：

```cpp
uint32_t kvStorageStart;
uint32_t kvStorageLen;
uint32_t kvActualLen;

if constexpr (PAGED_CACHE_FLAG) {
    kvStorageStart = 0;
    kvStorageLen = gSequsedKv.GetValue(b);
    kvActualLen = kvStorageLen;
} else {
    kvStorageStart = gCuSeqLengthsKv.GetValue(b);
    kvStorageLen = gCuSeqLengthsKv.GetValue(b + 1) - kvStorageStart;
    kvActualLen = hasSequsedKv ? gSequsedKv.GetValue(b) : kvStorageLen;
}
```

在参与无符号运算前检查前缀和非负、非递减，以及 `0 <= kvActualLen <= kvStorageLen`。失败 task 写空输出或直接跳过，避免整数下溢和越界地址。

### 7.3 统一逻辑块过滤

`KvBlockResolver::ResolveSparseRange` 先把稀疏块号转换为与存储无关的 token 范围：

```cpp
historyLen = kvActualLen - qActualLen;
causalEnd = historyLen + qTokenInBatch + 1; // 左闭右开
logicalStart = uint64_t(logicalBlockId) * blockShapeY;
range.count = min(blockShapeY, causalEnd - logicalStart);
```

先检查块号非负、`logicalStart < causalEnd`，再执行减法；乘法使用 64 位。
TND 的实际长度已受存储长度约束。PA 的表边界在下一步按逻辑页号校验，
不能用稀疏块号代替逻辑页号。

### 7.4 token、页与 tile 分层寻址

参考 FIA `op_kernel/arch22/flash_attention_regular_decode.h` 的 batch base + token offset 思路，
GBSA 在外层解析稀疏选择，再让两种存储模式进入同一 QK/PV 流水。
GBSA 的 `cu_seq_lengths_kv` 是 `[B+1]` 且以 0 开始；不照搬 FIA 的前缀和索引约定。

生产 helper 位于 `op_kernel/generic_block_sparse_attention_kv_resolver.h`：

- `KvTokenRange{start, count}` 表示 batch 内逻辑 KV token 范围；
- `ResolveTile` 输出 `KvTile{pageId, tokenOffset, tokenCount}`，长度不超过范围、
  `kvBaseTile` 和当前 PA 页的剩余长度；TND 不读取页参数或 block table；
- `BatchHeadOffset` 在每个任务进入 QK/PV 循环前计算一次 batch/head 基址；
- `Offset` 利用已缓存的 tile 分别计算 K/V 地址，PA 不重复查表。

```text
PA:
  logicalPage = logicalToken / pageSize
  tokenInPage = logicalToken % pageSize
  physicalPage = blockTable[b, logicalPage]
  offsetK = physicalPage * kPageStride + tokenInPage * (Nkv * D) + head * D

TND:
  batchHeadOffset = cuSeqLengthsKv[b] * (Nkv * D) + head * D
  offsetK = batchHeadOffset + logicalToken * (Nkv * D)
```

V 使用自己的页 stride。TND 的地址与稀疏块宽度的关系仅在稀疏选择阶段体现，
地址公式本身不需要页大小或计算 tile 大小。

例如稀疏块宽 192、PA 页宽 256、计算 tile 宽 128，选择块 1 并截断到 token 350：
逻辑范围是 `[192, 350)`。PA 分为页 0 内的 64 tokens 和页 1 内的 94 tokens；
TND 分为 128 和 30 tokens。这个例子由 CPU 地址层测试覆盖，尚未对外开放该尺寸组合。

### 7.5 复用现有 MMAD 与后续泛化边界

当前 Host 仍分别要求 `blockShapeY=128`、PA `blockSize=128`，并独立选择 `kvBaseTile=128`。
三条 Kernel 路径使用 `ResolveSingleTileBlock`，明确保证每个稀疏块只输出一个完整 tile；
若范围需要多个 tile，该适配器返回失败，避免只计算范围前半段。

1. 外层将 K/V GlobalTensor 指向 `KvTile` 的起点；
2. MMAD layout、搬运参数及 L1/L0 计算形状使用 `kvBaseTile`；
3. `actualBlockShapeQK/PV` 使用 `tile.tokenCount` 处理尾部；
4. QK、Online Softmax、PV、RescaleO 流水复用，底层 MMAD 无需区分 PA/TND。

后续开放其他尺寸时，需要同时完成：

- 用 `ResolveTile` 消费整个 `KvTokenRange`（每次将 start 加上、count 减去 `tokenCount`）；
- 将稀疏块列表展开为计算 tile 流；当前长度 256 的数组限制的是稀疏条目数，
  不能直接用作展开后的 tile 容量，需要容量规划或流式缓存；
- FD 的 `rawBegin/rawEnd` 继续切分原始稀疏列表，保持重复项和顺序，
  但工作量估算、split 阈值及 MMAD/Softmax 的尾 tile 同步需按展开后的 tile 重新验证；
- 计算 tile 本身改成非 128 时，还要核对 L1/L0、UB 容量、Vector 支持宽度和同步 stage；
- `blockShapeX` 的泛化还涉及多 Q token 的逐行 causal mask，不能仅放开 Host 校验。

地址层具备这些坐标表达能力，不等于完整算子已经支持任意尺寸。

### 7.6 FD 调度

Metadata 中的 SA/FD 调度由 Q 实际 token 数、KV head 数和每个 task 的有效稀疏块数量决定，不依赖 KV 的物理存储方式，因此 Metadata 二进制协议和 FD combine 算法无需变化。

需要确认并测试：

- FD 分割后的 `rawBegin/rawEnd` 在两种模式下均先作用于逻辑稀疏块列表；
- 每个 partial task 使用同一个地址解析器；
- 空块、非法块被过滤后，neutral partial 和空输出行为一致；
- PA/非 PA 的 combine 结果逐元素等价。

## 8. 接口与 Metadata 前置算子改造

### 8.1 ACLNN 主算子

`aclnn_generic_block_sparse_attention.cpp` 需要：

1. `ValidateParams` 放行 `layout_kv in {"TND", "PA_BBND"}`；
2. 增加 `layout_kv` 与 `block_table` 的组合校验；
3. PA 模式沿用 `NormalizeDim0KvTensor`；
4. 非 PA 模式对 Key/Value 调用普通 `l0op::Contiguous`，不创建“仅 dim0 可空洞”的 stride-aware view；
5. 保持函数签名和输出构造不变。

建议将 KV 归一化入口改成显式模式参数：

```cpp
NormalizeKvTensor(key, value, pagedCacheFlag, executor);
```

避免未来新增布局时继续使用带有 PA 假设的函数名和策略。

### 8.2 Torch 扩展

Torch schema 已包含可选 `cu_seq_lengths_kv`、`seqused_kv`、`block_table` 和 `layout_kv`，无需修改 schema。需要：

- 保持默认 `layout_kv="PA_BBND"`；
- meta/fake tensor 路径放行 `layout_kv="TND"`；
- 增加非 PA 调用示例；
- 保证 Metadata wrapper 与主算子使用相同的 layout 和长度输入组合。

### 8.3 GenericBlockSparseAttentionMetadata

Metadata AICPU Kernel 当前只依据 Q 序列和 `sparse_block_count` 生成调度，不读取 KV 数据和 `block_table`。因此设备侧调度实现无需修改，主要改动为：

- `CheckGbsaLayouts` 放行 `kvInputLayout="TND"`；
- 沿用现有 `CheckGbsaSeqLengthInputs` 中已经预留的 TND KV 规则；
- 增加 PA/非 PA 两套 Host UT；
- 更新文档中“当前仅支持 PA_BBND”的描述。

Metadata 与主算子必须接收完全一致的 `cu_seq_lengths_q/seqused_q` 和稀疏块张量。KV 模式不改变 task 编号，因此 Metadata 可在两种存储模式间复用，但推荐调用方仍传入一致的 `layout_kv`，由接口完成配置校验。

## 9. 文件级改动清单

| 文件/模块 | 主要改动 |
| --- | --- |
| `op_host/generic_block_sparse_attention_tiling.h` | 增加模式状态，更新 stride 字段语义 |
| `op_host/generic_block_sparse_attention_tiling.cpp` | 支持两种 layout/input 组合；按模式解析 batch、KV shape、长度和 stride；生成非 PA key |
| `op_host/op_api/aclnn_generic_block_sparse_attention.cpp` | 放行 TND KV；条件化 KV contiguous/stride-aware 处理 |
| `op_kernel/generic_block_sparse_attention_tilingkey.h` | 新增 TND 非 PA key 宏，保留原 PA key |
| `op_kernel/generic_block_sparse_attention.cpp` | 注册非 PA key，并实例化 `PAGED_CACHE_FLAG=false` Kernel |
| `op_kernel/generic_block_sparse_attention_kernel_interface.cpp` | 接口模板透传 `PAGED_CACHE_FLAG` |
| `op_kernel/arch22/generic_block_sparse_attention_kernel_arch22.h` | 条件初始化 KV 长度和 block table；统一块过滤和地址解析 |
| `op_kernel/arch35/generic_block_sparse_attention_kernel_arch35.h` | 同上 |
| `op_kernel/arch35/generic_block_sparse_attention_kernel_arch35_full_quant.h` | FP8 路径同步模式化地址解析 |
| `op_kernel/kernel_common.hpp` | 更新 TilingData 字段注释；如采用重命名则同步结构 |
| `generic_block_sparse_attention_metadata/op_host/...check.h` | 放行 TND KV 及对应长度组合 |
| 主算子与 Metadata 的 UT/示例/README/API 文档 | 增加非 PA 用例并更新约束 |

## 10. 测试设计

### 10.1 Host Tiling UT

正向用例：

- `TND + PA_BBND + block_table + seqused_kv`，确认原 key 不变；
- `TND + TND + cu_seq_lengths_kv`，确认新增 key；
- 非 PA 同时传 `seqused_kv`，shape 合法；
- B>1、GQA/MQA、不同 topK、Q/KV 尾块非对齐；
- A2 FP16/BF16 与 softmax precision 组合；
- A5 FP16/BF16/FP8 组合；
- FD 开启与关闭边界。

反向用例：

- PA_BBND 缺少 `block_table` 或 `seqused_kv`；
- TND KV 错传 `block_table` 或缺少 `cu_seq_lengths_kv`；
- PA 误传 `cu_seq_lengths_kv`；
- Key/Value shape、Nkv、D、dtype 不一致；
- PA 页内非连续；
- 非 PA TND 任一维非连续的直接图调用；
- `cu_seq_lengths_kv/seqused_kv` shape 与 batch 不匹配；
- 未支持 layout。

### 10.2 ACLNN 与 Metadata UT

- 主算子和 Metadata 对同一配置给出一致的通过/拒绝结果；
- 非 PA Key/Value 被正确 contiguous；
- PA dim0-strided view 仍保持零拷贝路径；
- Torch 默认参数仍进入 PA；显式 `layout_kv="TND"` 进入非 PA。

### 10.3 Kernel 精度测试

每个支持的 SoC/dtype 至少覆盖：

- B=1 和 B>1；
- `Nq=Nkv`、GQA、MQA；
- topK=1、边界值和 FD 阈值上下；
- 稀疏块乱序、重复块、尾部无效槽（由 `sparse_block_count` 截断）；
- Q/KV 实际长度小于存储长度；
- KV 最后半块、因果边界落在块中间；
- 零有效 KV、零有效稀疏块；
- A5 FP8 当前支持的输出 dtype。

精度验证采用两类基准：

1. 与 CPU/torch 稀疏 Attention 标杆比较；
2. 将同一份逻辑 KV 分别组织为 TND 和 PA_BBND，使用同一稀疏索引，比较两种模式输出和 LSE。该等价性测试是本功能最重要的回归门禁。

### 10.4 性能与回归

- PA 关键 shape 与改造前对比，Kernel 时延回退目标不超过 1%；
- 非 PA Kernel 内不得读取 `block_table`；
- 通过反汇编或 Kernel 编译产物确认 `if constexpr` 已消除无关分支；
- 非 PA 与“调用方先转 PA 再执行”的端到端时延对比，验证省去重排和 block table 构造后的收益；
- Workspace 大小不因新增模式增加，FD partial workspace 规则保持一致。

## 11. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| batch 当前依赖 block table | 非 PA 无法解析 batch | 统一从 `cu_seq_lengths_q` shape 获取，再与 PA block table 或 TND cuKv 校验 |
| `kStride0/vStride0` 名称带有 PA 语义 | 容易误用 | 保持张量 dim0 stride 语义；PA 为页，TND 为 token；计算 tile 单独命名 |
| TND 存储长度与实际长度混用 | padding、因果边界或地址错误 | Kernel 显式维护 `kvStorageStart/kvStorageLen/kvActualLen` 三个量 |
| 有符号长度转无符号后下溢 | 越界访问 | 转换前校验前缀和、长度关系和 `kvActualLen >= qActualLen` |
| PA key 被重算为新值 | 已有 Kernel 无法命中 | 固定现有 key 常量，新增精确值和唯一性 UT |
| 模板实例数量翻倍 | 编译时间和二进制增大 | 仅对已支持 dtype/precision 注册两种模式，不展开未支持 layout |
| Metadata 与主算子校验不一致 | Metadata 成功但主算子失败 | 抽取相同规则或建立成对 UT；设备 Metadata 协议保持不变 |
| A5 FP8 路径遗漏地址改造 | 非 PA FP8 精度或越界问题 | 地址解析 helper 被 regular/full-quant 共用，并单列 FP8 等价性用例 |

## 12. 实施步骤

1. 先完成 Host/ACLNN/Metadata 校验和新增 TilingKey UT，使两种模式能稳定分流。
2. 引入共用 `KvBlockAddressResolver`，改造 A2 regular 路径并通过 PA/非 PA 等价性测试。
3. 改造 A5 regular 路径，验证片上流水和 FD。
4. 改造 A5 FP8 full-quant 路径。
5. 补齐 torch、示例、README 和 ACLNN 文档。
6. 执行全量 Host UT、Kernel 精度、FD、非连续 PA 和性能回归后合入。

## 13. 验收标准

- `layout_q="TND", layout_kv="TND", block_table=nullptr` 可在支持的 SoC/dtype 上正确执行；
- 现有 `TND + PA_BBND` 用例全部通过，TilingKey 和输出行为不变；
- PA 与非 PA 的等价逻辑输入在误差阈值内输出一致；
- 两种模式的长度、shape、layout 和可选输入错误均在 Host 阶段被明确拒绝；
- 非 PA Kernel 编译产物中不存在 block table 访问；
- PA 关键 shape 性能回退不超过 1%，Workspace 不增加；
- 主算子、Metadata、ACLNN、Torch 文档中的支持矩阵和约束一致。
