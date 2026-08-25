# MoeEpDispatchEpilogue Metadata 输出设计

## 1. 文档目的

本文档约定 `MoeEpDispatchEpilogue` 的输出布局，以及首次执行、cached 执行和后续
Combine 消费 metadata 时必须遵守的数据关系。

本文档是实现指导，不描述具体代码修改位置。本阶段的代码实施范围仅包含
`MoeEpDispatchEpilogue` 及其 Torch 扩展适配；Combine 的处理方式只作为后续实现约束，
不在本阶段修改。

### 1.1 修订记录

| 版本 | 状态 | 修改说明 |
|---|---|---|
| 0.1 | 设计确认稿 | 约定五字段 metadata、首次执行和 cached 执行语义，以及后续 Combine 消费方式。 |
| 0.2 | 设计确认稿 | 明确首次执行和 cached 执行各自需要修改的校验范围。 |

## 2. 核心设计

Epilogue 输出的 token 数据继续按本地 expert 排列，metadata 改为按源 rank、源 token
排列。两种排列通过 metadata 中的 `recv_x_idx` 建立映射。

该设计需要同时满足以下目标：

- `recv_x`、`recv_scales` 和 `recv_topk_weights` 保持 expert-order，供专家计算使用。
- `recv_src_metadata` 的有效前缀保持 rank/token-order，供 Combine 按 rank 连续处理。
- cached 执行复用首次执行产生的 metadata，不重新生成排序关系。
- 不新增对外的 `rank_offsets` 输出。Combine 在消费 metadata 时自行生成 rank 边界。
- `do_cpu_sync` 只影响输出张量的分配容量，不引入新的 Epilogue Kernel 执行分支。

## 3. Metadata 约定

### 3.1 Shape 和数据类型

```text
recv_src_metadata: [A_alloc, 5], int32
```

每条 metadata 记录只包含 5 个 `int32` 字段。输出张量采用紧凑的 5 列布局，不增加
保留字段，也不对外暴露用于内部对齐的 padding。

内部 UB Buffer 可以按硬件要求对齐，但写回 GM 时必须保持 `[A_alloc, 5]` 的逻辑布局，
不能覆盖相邻记录。

### 3.2 字段定义

| 列索引 | 字段 | 含义 |
|---:|---|---|
| 0 | `src_rank` | token 的源 EP rank。 |
| 1 | `src_token_idx` | token 在源 rank 输入中的行号。 |
| 2 | `src_topk_idx` | 当前 expert 在源 token TopK 路由中的位置。 |
| 3 | `src_slot_idx` | 当前源 rank 在本 rank Dispatch window 中的 slot，用于 cached Epilogue 重新读取 token。 |
| 4 | `recv_x_idx` | 当前记录对应的 expert-order `recv_x` 行号。 |

### 3.3 有效长度

统一使用以下定义：

```text
A_valid = sum(num_recv_tokens_per_expert)
```

只有 `recv_src_metadata[0:A_valid]` 有效。不得使用 `A_alloc` 或
`sum(num_recv_tokens_per_rank)` 代替 `A_valid`：`num_recv_tokens_per_rank` 统计的是接收
slot 数，一个 slot 可能命中多个本地 expert，从而产生多条 metadata 记录。

`recv_x[0:A_valid]`、`recv_scales[0:A_valid]` 和
`recv_topk_weights[0:A_valid]` 同样是有效前缀。容量尾部的内容未定义，任何路径都不得
读取或解释该区域。

### 3.4 排序与映射不变量

metadata 有效前缀采用以下稳定顺序：

```text
src_rank 升序 → src_token_idx 升序 → src_topk_idx 升序
```

`recv_x` 有效前缀继续按本地 expert 升序分段，同一 expert 内保持确定的 token 顺序。

对于任意 `i`，记录 `recv_src_metadata[i]` 描述的 token 数据位于：

```text
recv_x[recv_src_metadata[i, 4]]
```

有效 metadata 中的 `recv_x_idx` 必须构成 `[0, A_valid)` 的一一映射，不能重复、遗漏
或指向容量尾部。scales 和 topk weights 与 `recv_x` 使用相同的行号。

### 3.5 本次校验范围

首次执行和 cached 执行都需要适配五字段 metadata，并非只修改 cached 校验。

两条路径共同修改的输出校验为：

- `recv_src_metadata` 的数据类型为 `int32`。
- `recv_src_metadata` 为二维张量，dim1 等于 5。
- `recv_src_metadata` 的 dim0 等于 `recv_x` 的 dim0，即两者使用同一个 `A_alloc`。
- `recv_src_metadata` 的 dim0 不超过既有的接收容量上界。

首次执行没有 `cached_recv_src_metadata` 输入，因此只更新上述输出 metadata 校验，不增加
新的 cached 输入校验，也不扩大其他参数的校验范围。

cached 执行在共同输出校验之外，还需要修改以下与 metadata 复用直接相关的校验：

- `cached_recv_src_metadata` 的数据类型为 `int32`。
- `cached_recv_src_metadata` 为二维张量，dim1 等于 5。
- `cached_recv_src_metadata` 的 dim0 等于本次 `recv_src_metadata` 和 `recv_x` 的 dim0。
- 当前输入 token 数等于 handle 中 `topk_idx` 的行数，保证缓存路由和 `recv_x_idx` 映射
  仍适用于本次输入。

`recv_x_idx` 的值位于 `[0, A_valid)` 是新的数据不变量。该值来自设备侧 metadata，不能
为了校验而触发 CPU 同步；cached Kernel 需要在使用该字段写 GM 前进行边界保护。

本次不借机扩展 `num_experts`、dtype 支持范围、拓扑参数或其他既有参数的校验。cached
模式已有的 `do_cpu_sync=True` 禁止条件和 topk 参数限制保持不变，不属于新增校验。

## 4. 首次执行路径

首次执行是未传入 cached handle 的 Dispatch/Epilogue 路径。Epilogue 需要从 Dispatch
window 读取 token 和路由信息，同时建立两种输出顺序。

### 4.1 处理流程

1. 等待 Dispatch window 数据和通知状态就绪。
2. 根据 `num_recv_tokens_per_expert` 建立 expert-order `recv_x` 行起点。
3. 统计每个 AI Core 对各本地 expert 的命中数，用于确定各 Core 的 `recv_x` 写入区间。
4. 统计每个 AI Core 对各源 rank 的 metadata 记录数，用于确定各 Core 的 metadata
   写入区间。该 rank 前缀仅用于 Epilogue 内部紧凑写回，不作为算子输出。
5. 按 `src_rank → slot → topk` 的顺序扫描有效 window slot。
6. 每发现一次本地 expert 命中：
   - 将 token、scale 和可选 topk weight 写到对应的 expert-order 行；
   - 将 5 个 metadata 字段写到对应的 rank/token-order 行；
   - 在 metadata 第 5 列记录本次 token 实际写入的 `recv_x_idx`。
7. 仅保证 `[0, A_valid)` 内的输出有效，不初始化容量尾部。

Dispatch 的 slot 由连续 token 范围的计数前缀分配，因此同一源 rank 内按 slot 升序扫描
可以保持源 token 顺序；同一 token 命中多个本地 expert 时，再按 `src_topk_idx` 升序输出。

### 4.2 `do_cpu_sync=True`

CPU 等待设备侧接收计数后，Torch 扩展按实际长度分配输出：

```text
A_alloc = A_valid
```

Epilogue 仍按设备侧 `num_recv_tokens_per_expert` 计算有效长度和行号，不需要单独的
CPU-sync Kernel 分支。

### 4.3 `do_cpu_sync=False`

CPU 不读取实际接收计数，Torch 扩展按安全上界分配输出：

```text
A_alloc = ep_world_size
          * num_max_tokens_per_rank
          * min(topk, num_local_experts)
```

Epilogue 的处理逻辑与 `do_cpu_sync=True` 完全相同，仍只写 `[0, A_valid)`。输出容量尾部
保持未定义，后续 cached 路径和 Combine 都必须继续使用 `A_valid`，不能扫描完整容量。

## 5. Cached 执行路径

cached 执行复用首次执行保存的 slot、路由和 metadata。它只允许更新 token 内容及相关
scale；路由关系、输入 token 数量和 MoE 配置必须与生成 handle 时一致。

### 5.1 Cached 前置条件

cached 路径必须满足：

- 使用 handle 中保存的 `topk_idx` 和 slot/route 信息，不接受新的路由关系。
- 当前输入 token 数量等于 handle 中 `topk_idx` 的行数。
- cached metadata 为 `[A_alloc, 5] int32`，其 dim0 与本次输出容量一致。
- `do_cpu_sync=True` 仍为非法组合；cached 路径固定不进行 CPU 同步。

这些条件保证本次 `num_recv_tokens_per_expert` 与首次执行一致，因而首次 metadata 中保存的
`recv_x_idx` 仍然有效。

### 5.2 有效长度和容量

本次执行仍由设备侧重新得到：

```text
A_valid = sum(current_num_recv_tokens_per_expert)
```

输出容量沿用 cached metadata 的容量：

```text
A_alloc = cached_recv_src_metadata.shape[0]
```

如果首次执行使用 `do_cpu_sync=True`，该容量通常等于实际长度；如果首次执行使用
`do_cpu_sync=False`，该容量是安全上界。两种情况下 cached Kernel 都只能读取 metadata
有效前缀 `[0, A_valid)`。

### 5.3 处理流程

1. 等待本次 cached Dispatch 将新 token 内容写入既有 window slot。
2. 按连续区间切分 cached metadata 的 `[0, A_valid)`，不重新计数或排序。
3. 对每条 metadata 读取 `src_rank`、`src_topk_idx`、`src_slot_idx` 和 `recv_x_idx`。
4. 使用 `src_rank` 和 `src_slot_idx` 定位本次 window token；使用 `src_topk_idx` 定位可选
   topk weight。
5. 将 token、scale 和可选 topk weight 写到 `recv_x_idx` 指定的 expert-order 行，而不是
   当前 metadata 行号。
6. 将 metadata 记录按原行号复制到本次输出，保持首次执行建立的 rank/token 顺序和
   `recv_x_idx` 映射。
7. 不读取或复制 `[A_valid, A_alloc)` 的尾部内容。

cached 路径必须对 `recv_x_idx` 做边界保护，确保其位于 `[0, A_valid)`，防止无效 handle
导致越界写入。

Torch `ElasticBuffer` 的 cached 接口当前不接受新的 `topk_weights`，因此常规 Torch 路径
不会产生 cached topk weight 输出；如果底层 ACLNN 模板保留 cached 且带权重的组合，权重
也必须按 `recv_x_idx` 写入，不能按 metadata 行号写入。

### 5.4 Cached 路径结果

cached 执行完成后必须继续满足：

- metadata 有效前缀与首次执行具有相同顺序和字段含义。
- `recv_x` 仍为 expert-order。
- 每条 metadata 的 `recv_x_idx` 仍准确指向本次输出中的对应 token。
- 新输出可以继续作为下一次 cached Dispatch 的 handle 内容。

## 6. 场景闭环

| 执行方式 | 输出容量 | Metadata 来源 | Metadata 顺序 | `recv_x_idx` 来源 |
|---|---:|---|---|---|
| 首次执行，CPU sync | `A_valid` | 本次扫描 window 生成 | rank/token-order | 本次 expert 前缀计算 |
| 首次执行，无 CPU sync | 安全上界 | 本次扫描 window 生成 | 有效前缀为 rank/token-order | 本次 expert 前缀计算 |
| cached，首次容量精确 | handle 容量 | 复制 cached 有效前缀 | 保持不变 | 复用首次结果 |
| cached，首次容量为上界 | handle 安全上界 | 只复制 cached 有效前缀 | 保持不变 | 复用首次结果 |

非法组合包括：

- cached 时显式设置 `do_cpu_sync=True`；
- cached 输入 token 数与 handle 路由行数不同；
- cached metadata 不是 `[A_alloc, 5] int32`。

## 7. Combine 使用指导

Combine 不属于本阶段代码修改范围，但后续实现必须按本节消费新的 metadata。

### 7.1 计算有效长度

Combine 首先计算：

```text
A_valid = sum(num_recv_tokens_per_expert)
```

只处理 `recv_src_metadata[0:A_valid]`，不能使用输入张量 dim0 作为实际记录数。

### 7.2 建立 Rank Offset

由于 metadata 已按 `src_rank` 排序，Combine 可以扫描第 1 列建立：

```text
rank_offsets[0:ep_world_size + 1]
```

其中 `[rank_offsets[r], rank_offsets[r + 1])` 是源 rank `r` 的连续 metadata 区间，且：

```text
rank_offsets[0] = 0
rank_offsets[ep_world_size] = A_valid
```

没有记录的 rank 对应空区间。`rank_offsets` 由 Combine 自己生成和管理，不加入
Epilogue 输出或 `EPHandle`。

### 7.3 发送 Token

对 metadata 有效前缀中的每条记录：

1. 从第 5 列读取 `recv_x_idx`。
2. 从 `recv_x[recv_x_idx]` 读取 expert 输出，而不是使用 metadata 行号读取 `recv_x`。
3. 如果存在 topk weight，从 `topk_weights[recv_x_idx]` 读取对应权重。
4. 使用 `src_rank`、`src_token_idx` 和 `src_topk_idx` 确定返回目标及目标 slot。
5. `src_slot_idx` 只服务于 cached Epilogue，Combine 不依赖该字段定位 `recv_x`。

按 rank offset 切分后，每组 Core 可以直接处理一个源 rank 的连续 metadata 区间，无需
对整个 metadata 反复扫描并过滤 rank。

## 8. 验收不变量

实现和测试需要共同验证以下条件：

```text
A_valid = sum(num_recv_tokens_per_expert)

recv_x[0:A_valid]
    为 expert-order

recv_src_metadata[0:A_valid]
    为 src_rank/src_token_idx/src_topk_idx 的升序排列

recv_src_metadata[:, 4]
    在有效前缀内构成 [0, A_valid) 的一一映射

recv_x[metadata[i, 4]]
    与 metadata[i] 描述的源 token 和 TopK 路由一致
```

首次执行、无 CPU sync、cached 以及连续多次 cached 执行都必须保持以上不变量。空接收
场景下 `A_valid=0`，所有路径均不得读取输出容量中的任何记录。
