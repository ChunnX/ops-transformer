# MoE EP 本地 `recv_x` 反向索引优化：PR 修改前后对比

## 1. PR 概要

本文档基于提交 `af4e079ff7872ea989a47c1ff78f6f7f11144ffe`（`localinfo`）相对其父提交的差异整理。

本次 PR 的核心目标是：

> 将 `(本卡原始 token, topK 位置) -> 本卡 recv_x 行号` 的反向索引，从 combine epilogue 阶段扫描 metadata 临时构建，前移到 dispatch epilogue 阶段生成，并存放在现有 `recv_src_metadata` packed buffer 的新增尾部区域。

因此，`moe_ep_combine_epilogue` 不再由每个 AIV 重复扫描本 rank 的 metadata 区间，而是直接按自己负责的 token 范围连续加载反向索引。

本次修改没有增加新的算子输入或输出参数，但改变了 `recv_src_metadata` 的内部 packed ABI 和实际分配长度。dispatch、combine、combine epilogue、C++ Torch 扩展和 Python Torch 扩展必须使用同一版本。

提交共修改 10 个文件，统计为 122 行新增、113 行删除。

## 2. 背景：为什么 combine 需要反向索引

dispatch epilogue 输出的五列 metadata 每一行描述一条已经接收并放入 `recv_x` 的路由：

| 列号 | 字段 | 含义 |
| --- | --- | --- |
| 0 | `srcRankId` | 该 token 原来属于哪个 EP rank |
| 1 | `tokenIdx` | token 在源 rank 输入中的行号 |
| 2 | `topkIdx` | 该 token 的第几个 topK 路由 |
| 3 | `slotIdx` | 通信窗口中的槽位号 |
| 4 | `recvXIdx` | 该路由在目的 rank `recv_x` 中的行号 |

metadata 的物理行顺序主要服务于接收、expert 分组和通信流程，并不是简单的 `tokenIdx * topK + topkIdx` 顺序。因此，combine 在处理本卡原始 token 时，不能根据 `(tokenIdx, topkIdx)` 通过一个固定算式推导 `recvXIdx`。

对于路由到本 rank 本地 expert 的条目，combine epilogue 需要从本地 `x`/`recv_x` 读取 expert 输出。读取前必须知道对应的 `recvXIdx`，所以需要如下反向关系：

```text
(tokenIdx, topkIdx) -> recvXIdx
```

对于路由到其他 rank expert 的条目，结果通过 combine 通信窗口返回，仍按 token/topK 对应的窗口槽读取，不依赖本地 `recvXIdx`。

## 3. 修改前的逻辑

### 3.1 packed metadata 布局

修改前的 `recv_src_metadata` 由两段组成，每一段按 512 Byte 对齐：

```text
+-----------------------------------------------+
| metadata rows: [A_alloc, 5], int32            |
+-----------------------------------------------+ <- Align512(A_alloc * 5 * 4)
| rank offsets: [ep_world_size + 1], int32      |
+-----------------------------------------------+ <- packed buffer 结束
```

其中：

- `A_alloc` 是 `recv_x` 的容量，不是本轮实际有效行数；
- `rank_offsets[r]` 和 `rank_offsets[r + 1]` 给出源 rank `r` 对应 metadata 行区间；
- `rank_offsets[ep_world_size]` 是实际有效 metadata 行数 `actualA`。

旧布局总字节数为：

```text
oldBytes = Align512(A_alloc * 5 * sizeof(int32))
         + Align512((ep_world_size + 1) * sizeof(int32))
```

### 3.2 combine epilogue 如何构建索引

修改前，`moe_ep_combine_epilogue` 的每个 AIV 执行以下工作：

1. 加载完整的 `rank_offsets` 到 UB；
2. 读取 `actualA`；
3. 得到本 rank metadata 区间：

   ```text
   localMetadataBegin = rank_offsets[rankId]
   localMetadataEnd   = rank_offsets[rankId + 1]
   ```

4. 将当前 AIV 负责的 token 范围对应的 UB 反向索引初始化为 `-1`；
5. 以最多 128 行为一个 chunk，将本 rank 的五列 metadata 从 GM 搬到 UB；
6. 逐行读取 `tokenIdx`、`topkIdx`、`recvXIdx`；
7. 如果 `tokenIdx` 属于当前 AIV 的输出 token 范围，则写入：

   ```text
   localRecvIdx[(tokenIdx - tStart) * topK + topkIdx] = recvXIdx
   ```

8. 后续处理每个 token/topK 时，通过该 UB 索引决定从本地 `recv_x` 还是 combine 通信窗口读取结果。

旧流程可概括为：

```text
每个 AIV
  -> 加载 rank offsets
  -> 扫描同一份本 rank metadata
  -> 只保留属于自己 token 分片的条目
  -> 得到本 AIV 的 localRecvIdx
  -> 执行 combine reduce
```

### 3.3 修改前的主要问题

metadata 虽然只需要扫描一次而不是“每处理一个 token 扫一次”，但扫描发生在每个参与计算的 AIV 上。

假设：

- 本 rank 有效 metadata 行数为 `A_local`；
- 有 `C` 个活跃 AIV；
- 每行 metadata 有 5 个 `int32`。

则 combine epilogue 为构建索引产生的总体 metadata 读取量近似为：

```text
C * A_local * 5 * sizeof(int32)
```

每个 AIV 都会扫描相同的 `[localMetadataBegin, localMetadataEnd)`，只是最终保留的 token 范围不同。这带来：

- 重复 GM -> UB 搬运；
- 重复逐行标量判断；
- rank offsets 和 metadata chunk 占用额外 UB；
- combine epilogue 的前处理耗时随 `A_local` 和活跃 AIV 数量增长。

## 4. 修改后的总体方案

新方案在 `recv_src_metadata` 尾部追加一个二维逻辑表：

```text
local_recv_index[tokenIdx][topkIdx] = recvXIdx
```

其一维下标为：

```text
lookupIndex = tokenIdx * topK + topkIdx
```

表长为：

```text
routeSlots = numTokens * topK = topk_idx.numel()
```

无本地接收行的槽位保持 `-1`。

新的职责划分是：

```text
dispatch epilogue
  -> 初始化整个 local_recv_index 为 -1
  -> 在生成/复制 metadata 时，顺手记录本 rank 源 token 的 recvXIdx

combine epilogue
  -> 按当前 AIV 的 token 分片连续加载 local_recv_index
  -> 不再扫描五列 metadata
  -> 直接用 recvXIdx 选择本地 recv_x 或通信窗口
```

## 5. packed metadata ABI 前后对比

### 5.1 新布局

```text
+--------------------------------------------------+
| metadata rows: [A_alloc, 5], int32               |
+--------------------------------------------------+ <- metadataRankOffsetsOffset
| rank offsets: [ep_world_size + 1], int32         |
+--------------------------------------------------+ <- localRecvIndexOffset
| local recv index: [numTokens, topK], int32       |
+--------------------------------------------------+ <- packed buffer 结束
```

三段各自按 512 Byte 对齐。对应公式为：

```text
metadataRankOffsetsOffset = Align512(A_alloc * 5 * 4)

localRecvIndexOffset = metadataRankOffsetsOffset
                     + Align512((ep_world_size + 1) * 4)

newBytes = localRecvIndexOffset
         + Align512(numTokens * topK * 4)

packedElements = newBytes / sizeof(int32)
```

Python 侧等价地以 128 个 `int32` 元素对齐，因为：

```text
128 * sizeof(int32) = 512 Byte
```

### 5.2 内存增量

相对旧版本，新增空间为：

```text
extraBytes = Align512(numTokens * topK * sizeof(int32))
```

示例：

- `A_alloc = 1024`
- `ep_world_size = 8`
- `numTokens = 256`
- `topK = 8`

则：

```text
metadata rows = 1024 * 5 * 4 = 20480 Byte
rank offsets  = Align512(9 * 4) = 512 Byte
local index   = Align512(256 * 8 * 4) = 8192 Byte

旧总大小 = 20480 + 512        = 20992 Byte
新总大小 = 20480 + 512 + 8192 = 29184 Byte
```

该例中每个 packed metadata buffer 增加 8192 Byte。

## 6. 反向索引构建示例

假设当前 rank 为 0，`numTokens = 2`，`topK = 2`，dispatch epilogue 得到以下有效 metadata：

| metadata 行 | `srcRankId` | `tokenIdx` | `topkIdx` | `slotIdx` | `recvXIdx` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 0 | 5 | 3 |
| 1 | 1 | 0 | 1 | 8 | 4 |
| 2 | 0 | 0 | 1 | 9 | 7 |

索引初始化后为：

```text
[token0/topk0, token0/topk1, token1/topk0, token1/topk1]
[-1,           -1,           -1,           -1]
```

dispatch epilogue 只记录 `srcRankId == 当前 rank` 的两行：

```text
metadata 行 0:
lookupIndex = 1 * 2 + 0 = 2
local_recv_index[2] = 3

metadata 行 2:
lookupIndex = 0 * 2 + 1 = 1
local_recv_index[1] = 7
```

最终结果为：

```text
local_recv_index = [-1, 7, 3, -1]
```

含义是：

| token/topK | 索引值 | combine epilogue 行为 |
| --- | ---: | --- |
| token 0, topK 0 | -1 | 没有本地 `recv_x` 行；根据 expert 位置走远端窗口或跳过无效本地路由 |
| token 0, topK 1 | 7 | 从 `recv_x[7]` 读取本地 expert 结果 |
| token 1, topK 0 | 3 | 从 `recv_x[3]` 读取本地 expert 结果 |
| token 1, topK 1 | -1 | 没有本地 `recv_x` 行 |

metadata 行 1 的 `srcRankId` 是 1，它属于 rank 1 的原始 token。当前 rank 0 的 combine 输出只处理 rank 0 自己的原始 token，因此不需要为该行建立本地反向索引。

## 7. dispatch epilogue 的具体修改

文件：`mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h`

### 7.1 新增状态

新增：

- `numTokens_`：从 tiling 配置读取原始 token 数；
- `localRecvIndexGm_`：指向 packed metadata 第三段；
- `InitLocalRecvIndex()`：并行初始化反向索引；
- `ubStageLocalIndexRing_`：普通路径使用的 32 Byte 对齐 local-index stage ring，与 metadata/weight stage ring 同步轮转；
- `ubLocalIndexValue_`：cached 路径使用的 32 Byte 对齐单值 UB 暂存区。

初始化时先按每 8 个 `int32`（32 Byte）组成一个 block，再使用 `SplitToCore(blockCount, aivNum, ...)` 将连续 block 分给各 AIV。每个 AIV 将自己的区间写为 `-1`，既避免多个核重复初始化同一区域，也保证每个核调用 `InitOutput` 时的 GM 起始地址满足 32 Byte 对齐；最后一个 block 可以只处理有效尾部元素。

### 7.2 普通（non-cached）路径

普通路径原本就会构造五列 metadata，并且此时同时拥有：

- `srcRankMeta`
- `tokenIdxMeta`
- `topkIdx`
- `recvXRow`

因此在写 metadata 后增加：

```text
if srcRankMeta == epRankId:
    lookupIndex = tokenIdxMeta * topK + topkIdx
    local_recv_index[lookupIndex] = recvXRow
```

这不是再次遍历 metadata，而是在既有处理循环中额外写一个 `int32`。实现时不能直接把 `ubStageMetaRing_[rowStart + 4]` 作为 `DataCopyPad` 的 UB 源：第五列相对 32 Byte 对齐行首偏移 16 Byte，不满足非原子 MTE 指令的地址对齐要求。

普通路径会将 `recvXRow` 额外写入当前 `stageBuf` 对应的 `ubStageLocalIndexRing_[stageOff + i * ELEM_ALIGN]`。该地址与 metadata 行首一样按 32 Byte 对齐。loop B 在已有的一次 `S -> MTE3` 握手后连续发出 token weight、metadata 和 local index 的写操作，不再为每个 local index 单独调用 `SyncFunc<S_MTE3>()` 和 `SyncFunc<MTE3_S>()`。三个 stage 输出共同由 `stageEvtMte3ToS_[stageBuf]` 保护，只有同一个 ring slot 再次使用前才等待此前的 MTE3 完成。

```text
同一个 stageBuf
  ├─ ubStageMetaRing_       -> metadata 第一段
  ├─ ubStageWeightsRing_    -> recv_topk_weights
  └─ ubStageLocalIndexRing_ -> packed metadata 第三段 local_recv_index

一次 S -> MTE3 提交，复用 stageBuf 前统一等待 MTE3 -> S
```

### 7.3 cached 路径

cached 路径从 `cached_recv_src_metadata` 分块加载旧的五列 metadata，并按其中的 `recvXIdx` 重建当前输出。

本次修改在已有逐行循环中增加：

```text
if srcRankId == epRankId
   and 0 <= tokenIdx < numTokens
   and 0 <= srcTopkIdx < topK:
    lookupIndex = tokenIdx * topK + srcTopkIdx
    local_recv_index[lookupIndex] = recvXIdx
```

cached 路径不会直接复用 cached packed buffer 尾部的 local index，而是根据五列 metadata 重建当前输出 buffer 的 local index。这样当前输出仍由本轮 dispatch epilogue 完整生产。

cached metadata 每行只有 5 个 `int32`，行跨度为 20 Byte，因此第五列的 UB 地址同样不保证 32 Byte 对齐。cached 路径从 UB 读取出标量 `recvXIdx` 后，也通过 `ubLocalIndexValue_` 暂存并写出，不能直接从 packed metadata 的第五列发起非原子 MTE3 搬运。有 topK weight 时，weight 与 local index 在同一次 `S -> MTE3` 后连续发出，并在两者都提交后只等待一次 `MTE3 -> S`，避免为 local index 再增加一组同步；无 topK weight 时仍需用一组同步保护单值暂存区复用。

### 7.4 初始化与写入顺序

`Process()` 开头先调用 `InitLocalRecvIndex()`，随后普通路径或 cached 路径都会经历已有的等待/全核同步，再进入 metadata 搬运和索引写入。因此索引初值先建立，命中的本地路由再覆盖为有效 `recvXIdx`。

## 8. combine epilogue 的具体修改

文件：`mc2/moe_ep_combine_epilogue/op_kernel/moe_ep_combine_epilogue.h`

### 8.1 删除的逻辑和资源

删除了 combine epilogue 自建索引所需的内容：

- metadata 五列字段常量；
- metadata chunk 最大行数常量；
- `recvSrcMetadataGm_` 和 `recvRankOffsetsGm_`；
- `actualA_`；
- `localMetadataBegin_` / `localMetadataEnd_`；
- `metadataChunkTokens_`；
- rank offsets UB buffer；
- metadata chunk UB buffer；
- `BuildLocalRecvIndex()` 的整段扫描和逐行筛选逻辑。

这减少了 combine epilogue 的初始化工作和固定 UB 占用。

### 8.2 新增直接加载

新增 `localRecvIndexGm_`，直接指向 packed metadata 第三段。

`LoadLocalRecvIndex()` 按当前 AIV 的 token 起点加载连续区间：

```text
global start = tStart * topK
copy count   = maskTokenNum
```

其中 `maskTokenNum` 等于该 AIV 负责的 token 数乘以 `topK`。各 AIV 只读取自己后续会查询的索引，不再读取其他 AIV 的 token 索引，也不再读取五列 metadata。

### 8.3 索引使用和防御性检查

处理每个 `(token, topK)` 时：

```text
localRecvXIdx = localRecvIdxTensor[lookupIndex]
```

新增了范围保护：

```text
if localRecvXIdx >= recvCapacity:
    localRecvXIdx = -1
```

随后：

- `localRecvXIdx >= 0`：从本地 `xGm_[localRecvXIdx]` 读取 token 结果，topK weight 也从本地对应行读取；
- `localRecvXIdx < 0` 且 expert 属于本 rank：跳过，表示该本地路由没有有效接收行；
- `localRecvXIdx < 0` 且 expert 属于远端 rank：从本 rank 的 combine 通信窗口对应 token/topK 槽读取结果。

### 8.4 执行顺序

修改后的入口顺序为：

```text
BuffInit()
LoadTopkIds()
LoadLocalRecvIndex()
MaskCheck()
RecvPhaseReduce()
```

`RecvPhaseReduce()` 中保留了原有逻辑：0 号 AIV 等待 combine 数据完成、清完成标志，然后全核同步并执行输出归约。此次优化只替换本地 `recv_x` 行号的获得方式，没有删除 combine 数据到达同步。

## 9. 普通 `moe_ep_combine` 的修改范围

需要注意，本次 PR 对普通 `mc2/moe_ep_combine` 的修改仅涉及 tiling/shape 契约：

- 在 `MoeEpCombineInfo` 中增加 `localRecvIndexOffset`；
- 按三段新布局校验 `recv_src_metadata` 的 packed 长度。

普通 `moe_ep_combine/op_kernel/moe_ep_combine.h` 本次没有改为读取新索引，仍然保留：

- 加载 rank offsets；
- 扫描本 rank metadata；
- 在 UB 中建立局部 `recvXIdx` 映射。

之所以仍要修改普通 combine 的输入长度校验，是因为 dispatch 输出和 Python handle 现在统一使用新 packed ABI。如果普通 combine 仍按旧长度校验，同一个 dispatch handle 将无法传给它。

## 10. host tiling 修改

以下三个 tiling 结构新增 `localRecvIndexOffset`：

- `MoeEpDispatchEpilogueInfo`
- `MoeEpCombineEpilogueInfo`
- `MoeEpCombineInfo`

对应 host tiling 统一完成：

1. 根据 `A_alloc` 计算 `metadataRankOffsetsOffset`；
2. 根据 `ep_world_size` 计算 `localRecvIndexOffset`；
3. 根据 `numTokens * topK` 计算 packed buffer 新总长度；
4. 严格校验输入或输出 storage shape 等于新长度。

因此旧版两段式 packed buffer 会在 shape 校验阶段失败，而不是让 kernel 越界读取第三段。

## 11. Torch 扩展修改

### 11.1 C++ 扩展

文件：`mc2/common/torch_extension/csrc/elastic_buffer.cpp`

`CheckMoeEpMetadataTensor()` 新增 `routeSlots` 参数，并增加：

- `routeSlots` 的 `int32` 范围校验；
- 第三段 offset 和总长度计算；
- 对新 packed 长度的严格检查。

调用侧统一传入：

```text
routeSlots = topkIdx.numel()
```

覆盖 dispatch epilogue、普通 combine 和 combine epilogue。

### 11.2 Python 扩展

文件：`mc2/common/torch_extension/elastic_buffer.py`

主要变化：

- `_metadata_buffer_layout(capacity, ep_world_size)` 增加 `route_slots` 参数；
- 返回值从 `(rank_offsets_offset, total)` 变为：

  ```text
  (rank_offsets_offset, local_index_offset, total)
  ```

- `_metadata_buffer_views()` 和 `_checked_handle_metadata_buffer()` 同步增加 `route_slots`；
- metadata buffer 分配长度包含第三段；
- 所有 handle 检查统一使用 `handle.topk_idx.numel()`；
- 对外仍只返回 metadata rows 和 rank offsets 两个 view，local index 保持内部实现细节，不增加 Python API 输出。

## 12. 关键流程前后对照

| 阶段 | 修改前 | 修改后 |
| --- | --- | --- |
| metadata buffer | 五列 metadata + rank offsets | 五列 metadata + rank offsets + local reverse index |
| 索引初始化 | combine epilogue 每个 AIV 初始化自己的 UB | dispatch epilogue 各 AIV并行初始化 GM 表为 `-1` |
| 索引构建 | 每个 combine AIV 扫描本 rank metadata | dispatch 写 metadata 时顺手写本地映射 |
| combine 读取 | 读取 rank offsets 和五列 metadata chunk | 连续读取自己 token 分片对应的 index |
| UB 占用 | rank offsets buffer + metadata chunk buffer + local index buffer | 只保留 local index buffer |
| packed buffer 大小 | 两段 | 三段，增加 `Align512(numTokens * topK * 4)` |
| 公开算子参数 | 不变 | 不变 |
| 内部 ABI | 旧两段格式 | 新三段格式，不向后兼容 |

## 13. 性能收益与代价

### 13.1 预期收益

combine epilogue 侧从“每个 AIV 扫描完整本 rank metadata”变为“每个 AIV只加载自己的连续索引分片”。总体索引读取量接近：

```text
numTokens * topK * sizeof(int32)
```

而且各 AIV 读取范围互不重叠。预期收益包括：

- 减少 combine epilogue 的重复 GM 读取；
- 删除 metadata chunk 的逐行标量判断；
- 减少固定 UB 占用，为 token 数据双缓冲留下更多空间；
- combine epilogue 前处理耗时不再直接乘以活跃 AIV 数量。

### 13.2 新增代价

该优化不是零成本，新增：

- 每个 handle/buffer 的 `numTokens * topK` 个 `int32` 存储；
- dispatch epilogue 对整张索引表的一次 `-1` 初始化；
- 每个本地来源有效路由的一次 `int32` GM 写；
- 普通路径增加 `STAGE_RING * topK * 32 Byte` 的对齐 UB 暂存空间；local index 复用 metadata stage ring 的已有 Scalar/MTE3 同步，不再逐条排空 MTE3；
- cached 路径保留一个 32 Byte 单值暂存块；有 topK weight 时复用 weight 写出的同步，无 topK weight 时本地索引写仍需要单独同步；
- combine epilogue 对索引分片的一次连续 GM 读。

因此小 token、小 topK 或本 rank metadata 很少时，dispatch 侧新增开销可能抵消部分收益。最终性能应按典型训练 shape 分别测量 dispatch epilogue、combine epilogue和端到端耗时。

## 14. 兼容性与升级要求

### 14.1 ABI 不兼容

旧 buffer：

```text
[metadata rows][rank offsets]
```

新 buffer：

```text
[metadata rows][rank offsets][local recv index]
```

新 host tiling 和 Torch 扩展会严格要求新总长度，所以不能将旧版持久化 handle 或旧版手工打包的 metadata 直接传给新算子。

### 14.2 必须配套升级的组件

至少需要同时更新并重新编译/部署：

- `moe_ep_dispatch_epilogue` host + kernel；
- `moe_ep_combine_epilogue` host + kernel；
- `moe_ep_combine` host tiling；
- C++ Torch extension；
- Python `elastic_buffer.py`。

如果存在仓外测试脚本、缓存文件或自行构造 `recv_src_metadata` 的调用方，也必须按新公式分配第三段。

## 15. 建议的验证项

### 15.1 正确性

- non-cached 与 cached dispatch 路径结果一致；
- `topK = 1` 和较大 topK；
- 存在无效 expert id（mask 为 false）；
- token 的所有 expert 都是本地、全部远端以及本地/远端混合；
- 某些本地路由被丢弃，索引保持 `-1`；
- `numTokens` 不能被 AIV 数整除；
- `numTokens * topK` 小于 AIV 数；
- `numTokens * topK` 不能被 8 整除，验证尾 block 初始化及各核 GM 起始地址对齐；
- 有/无 topK weight；
- FP16/BF16 以及项目支持的 FP8/scales 路径；
- cached handle 使用新 packed 长度并正确重建索引。

### 15.2 边界与异常

- `recvXIdx == recvCapacity - 1` 可正常读取；
- `recvXIdx >= recvCapacity` 被 combine epilogue 降级为 `-1`；
- 旧 packed 长度被 host/C++/Python 一致拒绝；
- `routeSlots` 越界时扩展侧报错；
- `A_alloc` 大于实际有效行数时 offset 仍按 `A_alloc` 计算。

### 15.3 性能

建议分别记录：

- dispatch epilogue kernel 时间；
- combine epilogue kernel 时间；
- dispatch + expert + combine 端到端时间；
- 不同 `numTokens`、`topK`、本地路由比例和 EP world size；
- 对比逐条同步版本与 stage-ring 版本，确认 non-cached 路径不再随本地索引条数线性增加 MTE3 排空次数；
- cached 路径分别覆盖有/无 topK weight，确认合并同步后的耗时变化；
- GM 带宽、AIV 利用率和 UB 使用情况。

## 16. 评审时需要重点关注的实现假设

1. 普通 non-cached dispatch 路径写 local index 时，直接使用 `tokenIdxMeta` 和 `topkIdx` 计算下标。该路径依赖上游 metadata 保证二者合法；cached 路径则显式做了范围检查。
2. 同一个 `(tokenIdx, topkIdx)` 按语义应最多对应一个本地有效 `recvXIdx`。若产生重复记录，后写入者会覆盖先写入者。
3. 初始化 `-1` 与有效索引写入之间必须保持全核时序，避免迟到的初始化覆盖有效值。
4. dispatch、combine 和 Python/C++ 扩展的 512 Byte 对齐公式必须始终一致。
5. 非原子 MTE 搬运的 UB 地址必须满足 32 Byte 对齐。五列 metadata 的第五列位于行首 `+16 Byte`，cached packed 行的跨度还是 20 Byte，二者都不能直接作为单个 `int32` 写 GM 的 UB 源，必须经过对齐暂存区。
6. `ubStageLocalIndexRing_` 的生命周期必须与 `ubStageMetaRing_` 一致：写出前受同一个 `S -> MTE3` 握手约束，复用前受同一个 `stageEvtMte3ToS_[stageBuf]` 约束，不能在 MTE3 尚未消费时被 Scalar 覆盖。
7. 普通 `moe_ep_combine` 尚未消费新索引；如果后续也改为直接加载，可进一步删除它自身的 metadata 扫描逻辑，但那不属于本次提交。

## 17. 结论

本次 PR 的本质是用少量、规则的空间换取 combine epilogue 更直接的本地寻址：

```text
修改前：combine epilogue 扫描 metadata -> 临时建立映射 -> 查 recv_x

修改后：dispatch epilogue 生成映射 -> combine epilogue 直接加载 -> 查 recv_x
```

其主要优化对象是多 AIV 对同一段本 rank metadata 的重复扫描。公开接口数量没有变化，但 `recv_src_metadata` 的内部布局和长度发生了不兼容变更，需要全链路同步升级与重新编译。
