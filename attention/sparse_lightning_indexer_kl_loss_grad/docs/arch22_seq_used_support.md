# SparseLightningIndexerKLLossGrad arch22 seqUsedQ/seqUsedK 支持修改说明

## 1. 文档目的

本文说明 `SparseLightningIndexerKLLossGrad` 在 arch22 实现中支持 `seqUsedQOptional` 和
`seqUsedKOptional` 的代码修改、接口约束和运行行为。

本文面向以下读者：

- 算子调用者：了解两个可选输入的语义、传参约束、输出行为和兼容性。
- 框架接入人员：了解主算子与 metadata 算子的调用关系。
- 算子开发和维护人员：了解 host tiling、模板 tiling key、AI Core kernel 和 AICPU metadata
  各层修改的作用。
- 测试人员：了解需要覆盖的布局、可选输入组合、边界值和确定性分支。

本次修改范围仅为 arch22 路径。`arch35` 目录下的实现未修改。

## 2. 变更结论

arch22 kernel 现在可以按 batch 使用独立的 Q、K 有效序列长度：

- `seqUsedQOptional[b]` 表示 batch `b` 中实际参与计算的 Q 长度。
- `seqUsedKOptional[b]` 表示 batch `b` 中实际参与计算的 K 长度。
- 两个参数可以同时传入，也可以只传其中一个。
- 未传入的一侧继续使用原来的分配长度。
- `seqUsed` 只改变有效计算范围，不改变输入、输出 Tensor 的物理 shape 和 batch 起始地址。
- arch22 下只要传入任意一个 `seqUsed`，就必须同时传入使用相同参数生成的 `metadataOptional`。
- 传入任意一个 `seqUsed` 后，kernel 会先将完整物理范围的 `dq`、`dw` 和 `softmaxOut` 初始化为 0；
  有效任务随后覆盖对应输出，因此 Q padding 行或因 K 有效长度缩短而跳过的 Q 行保持为 0。
- 原有不传 `seqUsed` 的调用方式保持不变，并继续选择原有 kernel 模板分支。

## 3. 核心概念：分配长度与使用长度

支持 `seqUsed` 的关键是区分“分配长度”和“使用长度”。

### 3.1 分配长度

分配长度决定 Tensor 的物理存储、batch 起始地址和输出 shape。

对于 batch `b`：

```text
BSND:
  allocQ[b] = S1
  allocK[b] = S2

TND:
  allocQ[b] = cuSeqLensQ[b + 1] - cuSeqLensQ[b]
  allocK[b] = cuSeqLensK[b + 1] - cuSeqLensK[b]
```

### 3.2 使用长度

使用长度决定 kernel 实际遍历多少个 Q token，以及 K 侧有效范围和 sparse mask 的计算边界。

```text
usedQ[b] = seqUsedQ[b]，如果传入 seqUsedQOptional
           allocQ[b]，否则

usedK[b] = seqUsedK[b]，如果传入 seqUsedKOptional
           allocK[b]，否则
```

必须满足：

```text
0 <= usedQ[b] <= allocQ[b]
0 <= usedK[b] <= allocK[b]
```

### 3.3 物理地址不压缩

`seqUsed` 不会把不同 batch 的物理数据重新拼接。

```text
BSND batch b 的 Q 物理起始行 = b * S1
BSND batch b 的 K 物理起始行 = b * S2

TND batch b 的 Q 物理起始行 = cuSeqLensQ[b]
TND batch b 的 K 物理起始行 = cuSeqLensK[b]
```

传入 `seqUsedQOptional` 时，metadata 中的任务编号会按 `usedQ` 压缩；只传
`seqUsedKOptional` 时，Q 任务编号和任务总数保持原逻辑。kernel 将压缩任务编号映射回 batch 后，仍按
上述物理地址访问 Q、K 和输出。这样可以跳过 Q padding 计算，同时保持既有 Tensor shape 和地址协议
不变。

## 4. 接口行为与传参要求

### 4.1 参数说明

| 参数                 | 类型与 shape                      | 含义                         | 未传入时的行为                         |
| -------------------- | --------------------------------- | ---------------------------- | -------------------------------------- |
| `seqUsedQOptional` | `INT32`，一维，长度至少为 `B` | 每个 batch 实际使用的 Q 长度 | 使用该 batch 的 Q 分配长度             |
| `seqUsedKOptional` | `INT32`，一维，长度至少为 `B` | 每个 batch 实际使用的 K 长度 | 使用该 batch 的 K 分配长度             |
| `metadataOptional` | `INT32`，由 metadata 算子生成   | arch22 的分核和压缩任务信息  | 仅在两个`seqUsed` 都未传入时仍可省略 |

源码内部部分变量使用 `sequsedQ`、`sequsedK` 或 `seqused_q`、`seqused_k` 命名；它们分别对应公开接口中的
`seqUsedQOptional` 和 `seqUsedKOptional`。

### 4.2 四种可选输入组合

| `seqUsedQOptional` | `seqUsedKOptional` | Q 计算范围    | K 有效范围    | metadata 总任务数 |
| -------------------- | -------------------- | ------------- | ------------- | ----------------- |
| 不传                 | 不传                 | `allocQ[b]` | `allocK[b]` | 原有逻辑          |
| 传                   | 不传                 | `usedQ[b]`  | `allocK[b]` | `sum(usedQ)`    |
| 不传                 | 传                   | `allocQ[b]` | `usedK[b]`  | 原有 Q 总行数     |
| 传                   | 传                   | `usedQ[b]`  | `usedK[b]`  | `sum(usedQ)`    |

只传 `seqUsedKOptional` 时不会减少 Q 行数，但每个 Q 行的有效 K 范围、mask 边界和估算负载会使用
`usedK[b]`。如果 K 有效长度使部分 Q 行无有效计算任务，这些 Q 行对应的 `dq`、`dw` 和
`softmaxOut` 保持为 0。

### 4.3 metadata 条件必传

arch22 下，当 `seqUsedQOptional` 或 `seqUsedKOptional` 任意一个存在时，`metadataOptional` 必须存在。
否则 host tiling 在 `CheckContext()` 中返回失败，并输出：

```text
metadata is required when seqused_q or seqused_k is provided on arch22.
```

原因是 host tiling 只能看到 Tensor 的 shape，不能读取每个 batch 的 `seqUsed` 数值。实际使用长度会影响：

- Q 任务总数；
- 压缩任务编号到 `(batch, s1)` 的映射；
- 每个 Q 行的 K 侧有效负载；
- 多核负载均衡的切分位置。

这些信息由 AICPU metadata 算子读取 `seqUsed` 后生成，主 kernel 再使用同一份 metadata 执行。

调用者必须保证生成 metadata 和调用主算子时传入的 `cuSeqLensQOptional`、`cuSeqLensKOptional`、
`seqUsedQOptional`、`seqUsedKOptional`、布局、mask 参数及 shape 属性相互一致。代码不会验证两次调用的
参数内容是否一致。

### 4.4 TND 额外要求

当前 arch22 host 要求 `layoutQ` 和 `layoutK` 的 rank 相同，因此只支持 `BSND/BSND` 和 `TND/TND`
组合，不支持 `BSND/TND` 或 `TND/BSND` 混合布局。

`seqUsed` 不能替代 `cuSeqLens`。使用 `TND/TND` 时仍需要：

- 同时传入 `cuSeqLensQOptional` 和 `cuSeqLensKOptional`；
- 两个 `cuSeqLens` 的长度相同且大于 1；
- `cuSeqLens` 继续用于确定 batch 的物理起始地址和分配长度；
- `seqUsed` 仅在对应 batch 的分配区间内部截取有效前缀。

## 5. 总体执行流程

```text
调用者
  |
  |-- 1. 调用 SparseLightningIndexerKLLossGradMetadata
  |      输入 cuSeqLens / seqUsed / layout / mask 等信息
  |      校验 0 <= seqUsed <= allocLen
  |      传入 seqUsedQ 时，以 usedQ 构造压缩 Q 任务空间
  |      只传 seqUsedK 时，保持原 Q 任务空间
  |      以 usedQ、usedK 估算每个任务的负载
  |      输出多核切分 metadata
  |
  `-- 2. 调用 SparseLightningIndexerKLLossGrad
         host 根据 seqUsed 是否存在选择模板 tiling key
         kernel 绑定 seqUsed GM
         按 metadata 取得每个核的压缩任务区间
         存在 seqUsedQ 时，将压缩任务编号映射回 batch 和 s1
         用分配长度计算物理地址
         用使用长度限制 Q/K 计算范围
         写回有效行，并将所有跳过行的 Q 侧输出保持为 0
```

## 6. 各文件修改说明

### 6.1 `op_host/sparse_lightning_indexer_kl_loss_grad_tiling_general.cpp`

#### `CheckContext()`

新增 arch22 的参数组合检查：只要 `seqused_q` 或 `seqused_k` 的可选输入 shape 存在，metadata 的可选输入
shape 就必须存在。

作用：

- 阻止 kernel 在没有匹配 metadata 的情况下解释压缩任务编号；
- 避免 host 按分配长度切分任务、kernel 却按使用长度计算所造成的 batch/s1 映射错误；
- 将错误提前到 tiling 阶段，而不是让错误地址或错误计算结果进入 AI Core 执行阶段。

#### `GetTilingKey()`

新增 `hasSequsedQ` 和 `hasSequsedK` 两个布尔值，分别通过可选输入 Tensor 是否存在获得，并加入
`GET_TPL_TILING_KEY` 参数。

作用：

- 为四种 `seqUsedQ/seqUsedK` 组合生成不同的模板 tiling key；
- 让 kernel 使用编译期常量选择相应代码路径；
- 在未传 `seqUsed` 时保留原有模板实例和行为。

### 6.2 `op_kernel/sparse_lightning_indexer_kl_loss_grad_template_tiling_key.h`

模板参数声明和模板参数选择列表中新增：

```text
HAS_SEQUSED_Q = 0 或 1
HAS_SEQUSED_K = 0 或 1
```

作用：

- 正式把两个可选输入的“是否存在”纳入 arch22 tiling key 协议；
- 允许模板系统为两个布尔参数的所有组合匹配合法 kernel；
- 保证 host 产生的 tiling key 与 kernel 入口模板参数顺序一致。

### 6.3 `op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_common.h`

`SLIType` 模板新增 `HasSequsedQ`、`HasSequsedK` 参数，并暴露为：

```cpp
static constexpr bool hasSequsedQ = HasSequsedQ;
static constexpr bool hasSequsedK = HasSequsedK;
```

作用：

- 将 tiling key 中的可选输入状态传递给 base、vector 等内部实现类；
- 内部实现可以使用 `if constexpr` 在编译期裁剪无关分支；
- 避免在每个 Q 行的热路径上反复做运行时空指针判断。

### 6.4 `op_kernel/sparse_lightning_indexer_kl_loss_grad.cpp`

arch22 的全局 kernel 模板入口新增 `HasSequsedQ`、`HasSequsedK` 两个参数，并在 FP16、BF16 两个
`SLI_OP_IMPL` 实例化分支中传给 `SLIType`。

作用：

- 打通“host tiling key -> 全局 kernel 模板 -> `SLIType` -> 内部服务类”的参数传递链路；
- FP16 和 BF16 使用相同的 `seqUsed` 语义；
- 修改位于 `__CCE_AICORE__ != 310` 的 arch22 入口，arch35 的模板参数和实现逻辑未改。

文件末尾同时补充了换行符，该变化不影响运行行为。

### 6.5 `op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_base.h`

该文件负责 arch22 主流程、任务定位、物理地址计算和各计算服务的协调，是本次 kernel 修改的核心。

#### 新增编译期状态

从 `SLIType` 取得 `hasSequsedQ`、`hasSequsedK`。

作用：在 base 层决定是否绑定输入、是否压缩 Q 任务映射、是否使用 K 有效长度，以及是否需要全量清零
Q 侧输出。

#### 新增 GM Tensor

新增 `seqUsedQueryGm`、`seqUsedKeyGm`。

`Init()` 中：

- 输入非空时按 `B` 个 `INT32` 元素绑定；
- 输入为空时绑定 size 为 0 的 GlobalTensor；
- 将两个 GlobalTensor 继续传给 vector service。

原来对 `sequsedQ`、`sequsedK` 的 `(void)` 忽略处理被移除，表示 arch22 kernel 开始实际消费这两个输入。

#### 新增 `GetUsedSeqLens()`

该辅助函数统一实现有效长度读取：

- GlobalTensor size 为 0 时返回调用方给出的 `defaultLens`；
- 否则返回 `seqUsedGm[bIdx]`。

作用：让 BSND 和 TND、Q 和 K 共用相同的 fallback 规则，并把空输入处理集中在一个位置。

#### 修改 `InitWorkspace()`

当 `hasSequsedQ == true` 或 `hasSequsedK == true` 时，按物理 Q 总行数并行初始化以下输出：

- `dq`；
- `dw`；
- `softmaxOut`。

作用：`seqUsedQ` 会直接裁掉 Q padding 行，较短的 `seqUsedK` 也可能因 no-mask 下 K 长度为 0 或 causal
边界变化而使部分 Q 行不产生任务。如果不预先清零，这些行可能保留输出内存中的历史数据。任意
`seqUsed` 存在时全量初始化，可以保证所有跳过行的结果确定为 0。

`dk` 继续使用原有 `scatterAddRes` 全量清零、累加和最终类型转换写回路径，本次没有为 `dk` 增加第二套
初始化机制。

#### 新增 `FindBIndexBySeqUsed()`

metadata 中的 Q 任务编号是压缩编号，其范围为 `[0, sum(usedQ))`。新函数通过 `usedQ` 前缀和把压缩编号
映射为：

```text
batch 索引 bIdx
batch 内 Q 索引 s1Idx = compressedIndex - sum(usedQ[0:bIdx])
```

对于 TND，函数仍使用 `cuSeqLensQ` 的相邻差值取得每个 batch 的 Q 分配长度，再以 `seqUsedQ` 覆盖其
有效长度。对于 BSND，默认分配长度为固定 `S1`。

#### 修改 `CalcMultiCoreOffset()`

当存在 `seqUsedQ` 时，TND 和 BSND 都使用 `FindBIndexBySeqUsed()` 解释 metadata 给出的每核起止任务编号。
不存在 `seqUsedQ` 时继续使用原有逻辑：

- TND 按 `cuSeqLensQ` 累积长度定位 batch；
- BSND 按固定 `S1` 做除法和取余。

作用：保证 metadata 的压缩任务空间和 kernel 的 batch/s1 定位使用同一套坐标系。

#### 修改 `Process()`

TND 分支现在分别保存：

- `allocatedSeqLensQ/K`：来自 `cuSeqLens`，用于计算 `accumS1Len/accumS2Len` 等物理偏移；
- `actualSeqLensQ/K`：存在对应 `seqUsed` 时取 `seqUsed`，否则取分配长度，用于控制计算范围。

BSND 分支中：

- `actualSeqLensQ` 可取 `seqUsedQ[b]`；
- `actualSeqLensK` 可取 `seqUsedK[b]`；
- 非末 batch 的 `s1EndIdxThisBatch` 从固定 `S1` 改为当前 batch 的 `actualSeqLensQ`。

作用：跳过 Q padding 行，并让 K 侧 gather、sparse mask 和后续矩阵计算使用有效 K 长度，同时保持物理
地址按原 shape 计算。

#### 修改 `GetRunInfo()`

BSND 分支中的 `runInfo.actS1Size` 和 `runInfo.actS2Size` 不再固定写入 `S1/S2`，而是使用上层传入的
`actualSeqLensQ/K`。

作用：将 `seqUsed` 的有效边界传递到每个任务的 sparse 长度计算、mask 计算和 vector/cube 处理阶段。

### 6.6 `op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_vector.h`

该文件负责 vector 侧的 gather、softmax/梯度处理，以及确定性 scatter-add 路径中的任务重放。

#### 新增编译期状态和 GM Tensor

新增 `hasSequsedQ`、`hasSequsedK`，以及 `seqUsedQueryGm`、`seqUsedKeyGm`。

作用：vector service 使用与 base 完全一致的模板状态和有效长度数据，避免不同流水阶段对同一 batch 的
长度理解不一致。

#### 新增 `InitSeqUsedGM()`

base 在初始化 vector service 时调用该函数，把两个 `seqUsed` GlobalTensor 传入 vector 层。

作用：保持原有 `InitVector0GM()`、`InitVector1GM()` 的职责不变，单独补充本次新增的共享输入。

#### 新增 `GetUsedSeqLens()`

与 base 中同名函数保持相同 fallback 规则：未绑定 `seqUsed` 时返回分配长度，已绑定时读取当前 batch
的有效长度。

#### 修改 vector `GetRunInfo()`

TND 下：

- 先从 `cuSeqLens` 获取分配长度及物理累积偏移；
- 再按模板状态选择是否使用 `seqUsedQ/K` 覆盖 `actS1Size/actS2Size`；
- `accumS1Idx`、`accumS2Idx` 仍保持物理坐标。

BSND 下：

- `actS1Size/actS2Size` 可使用 `seqUsedQ/K`；
- 物理偏移仍为 `b * S1 + s1` 和 `b * S2`。

作用：vector 侧的 `s2SparseLen`、真实 top-k 长度和后续数据搬运使用有效边界，而不是 padding 后的
分配边界。

#### 修改 `ProcessDeterVector2()`

确定性 scatter-add 路径会按核顺序重新读取 metadata 任务，因此也必须重新完成压缩任务编号到
`(batch, s1)` 的映射。

当存在 `seqUsedQ` 时，新逻辑：

1. 遍历 batch 并累计 `usedQ`；
2. 找到压缩任务所属 batch；
3. 计算该 batch 内的 `s1Idx`；
4. 取得对应的 `usedQ` 和 `usedK`；
5. 如果没有找到合法 batch，则跳过该任务。

同时将 `bIdx`、`s1Idx` 初始化为 0，并增加 `foundSeq` 检查，避免非法 metadata 任务导致未初始化索引被
使用。

内部模板仍保留按 Q/K layout 分支选择 K 分配长度的代码，但当前 host 会拒绝 Q/K rank 不同的混合
布局。因此对外可达的 arch22 路径只有 `TND/TND` 和 `BSND/BSND`。

### 6.7 `sparse_lightning_indexer_kl_loss_grad_metadata/op_kernel_aicpu/arch22/`

`sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch22.h`

metadata AICPU kernel 负责读取每个 batch 的实际长度并构建 AI Core 分核信息。

#### 新增 `ParamsCheck()` 校验

新增检查包括：

- `seqused_q` 存在时，第一维长度至少为 `B`；
- `seqused_k` 存在时，第一维长度至少为 `B`；
- 对每个 batch 检查 `0 <= usedQ <= allocQ`；
- 对每个 batch 检查 `0 <= usedK <= allocK`。

作用：防止负长度、越界长度和数组长度不足进入任务构造阶段。该检查位于能够读取 Tensor 数据的 AICPU
侧，因此可以校验每个元素的值。

#### 新增 `GetSeqUsedLen()`

若对应 Tensor 无效则返回 `defaultLen`，否则读取第 `bIdx` 个 `INT32` 值。

作用：统一 metadata 侧“可选输入不存在时退回分配长度”的规则。

#### 新增 `GetS1SeqSize()` 和 `GetS2SeqSize()`

两个函数分别计算当前 batch 用于任务构造的 Q、K 有效长度：

- TND 默认值来自 `cuSeqLens` 相邻元素之差；
- BSND 默认值来自固定 `S1/S2`；
- 对应 `seqUsed` 存在时覆盖默认值。

#### 修改 `CalcTotalSize()`

存在 `seqUsedQ` 时，metadata 总任务数改为：

```text
totalSize = sum(seqUsedQ[b]), b in [0, B)
```

不存在 `seqUsedQ` 时保持原逻辑：TND 使用 `cuSeqLensQ[B]`，BSND 使用 `B * S1`。

作用：从任务层面彻底跳过 Q padding 行，减少无效 AI Core 调度。

#### 修改 `BuildSparseValidArray()`

TND 和 BSND 两个分支都改为按 `GetS1SeqSize()`、`GetS2SeqSize()` 构造每个 Q 任务的估算负载。

作用：

- 数组长度与压缩后的 `totalSize` 一致；
- 每个 Q 行的负载估算使用真实 K 有效长度；
- 多核切分位置基于实际计算量，而不是 padding 后的最大长度；
- 只传 `seqUsedK` 时虽然 Q 总任务数不变，负载均衡仍能反映 K 长度变化。

## 7. 布局示例

### 7.1 BSND 示例

假设：

```text
B = 2, S1 = 8, S2 = 16
seqUsedQ = [5, 7]
seqUsedK = [12, 9]
```

行为如下：

| batch | Q 物理区间  | Q 有效区间  | K 物理区间   | K 有效区间   |
| ----- | ----------- | ----------- | ------------ | ------------ |
| 0     | `[0, 8)`  | `[0, 5)`  | `[0, 16)`  | `[0, 12)`  |
| 1     | `[8, 16)` | `[8, 15)` | `[16, 32)` | `[16, 25)` |

metadata 的 Q 任务总数为 `5 + 7 = 12`。压缩任务 5 是 batch 1 的第 0 个 Q，但它对应的物理 Q 行是
8，而不是 5。batch 0 的 Q 行 `[5, 8)` 和 batch 1 的 Q 行 `[15, 16)` 不参与计算，其 Q 侧输出保持
为 0。

### 7.2 TND 示例

假设：

```text
cuSeqLensQ = [0, 8, 18]
cuSeqLensK = [0, 16, 30]
seqUsedQ   = [5, 7]
seqUsedK   = [12, 9]
```

两个 batch 的 Q 分配长度分别为 8 和 10，K 分配长度分别为 16 和 14。metadata 仍只生成 12 个 Q
任务，但 batch 1 的 Q、K 物理起始地址分别保持为 `cuSeqLensQ[1] = 8` 和 `cuSeqLensK[1] = 16`。

因此 `seqUsed` 是对每个 TND batch 分配区间取有效前缀，不会改变 `cuSeqLens` 定义的拼接位置。

## 8. 输出语义

### 8.1 `dq`、`dw` 和 `softmaxOut`

传入任意一个 `seqUsed` 时，kernel 会先把完整物理 Q 范围内的 `dq`、`dw` 和 `softmaxOut` 初始化为 0。

传入 `seqUsedQOptional` 时：

- `s1 < usedQ[b]` 的行按正常流程计算和写回；
- `usedQ[b] <= s1 < allocQ[b]` 的行不参与计算；
- 被跳过行在 kernel 初始化阶段被置 0。

只传 `seqUsedKOptional` 时，Q 任务总数不压缩；但如果 `usedK[b]` 使某些 Q 行没有有效 K 范围，这些
Q 行不会进入计算，其输出保持为初始化后的 0。

### 8.2 `dk`

`dk` 仍通过原有 scatter-add workspace 聚合。workspace 按物理 K 输出范围初始化为 0，仅有效任务和有效
sparse index 对相应 K 行累加梯度，之后再写回 `dk`。

调用者应保证 `sparseIndices` 中的有效索引位于当前 batch 的 `usedK` 范围内；无效 top-k 位置继续使用
`-1`。本次修改没有增加逐个 sparse index 的 host 侧值校验。

## 9. 兼容性和性能影响

### 9.1 向后兼容

当两个 `seqUsed` 都不传时：

- metadata 总任务数和负载计算保持原逻辑；
- kernel 选择 `HAS_SEQUSED_Q=0`、`HAS_SEQUSED_K=0` 模板；
- Q/K 有效长度等于原有分配长度；
- 不触发本次新增的 Q 侧全量输出初始化。

### 9.2 性能影响

传入 `seqUsedQ` 的影响：

- 收益：metadata 只为 `sum(usedQ)` 个 Q 行生成任务，可跳过 Q padding 的 gather、矩阵计算和 vector
  处理；
- 收益：多核负载按 `usedQ/usedK` 重新估算，长度差异较大的 batch 切分更合理；
- 代价：压缩任务编号映射 batch 时需要计算 `usedQ` 前缀和，确定性路径也需要重新映射任务。

传入 `seqUsedK` 但不传 `seqUsedQ` 时，Q 任务数不减少；当 `usedK` 小于 top-k 计算上限或改变 causal
有效范围时，可以减少 K 侧有效计算量并改进负载估算。

传入任意一个 `seqUsed` 都会增加一项固定代价：为保证所有跳过行输出为 0，kernel 启动阶段会初始化
完整物理范围的 `dq`、`dw` 和 `softmaxOut`。

### 9.3 模板数量

tiling key 新增两个布尔维度，理论上形成四种 `seqUsedQ/seqUsedK` 模板组合。这样增加了编译实例数量，
但将可选输入判断移出热点运行路径。

## 10. 边界和限制

- `seqUsedQOptional`、`seqUsedKOptional` 的数据类型必须为 `INT32`。
- 两者存在时长度至少为 `B`。
- 每个值必须位于 `[0, allocLen]`。
- 单个 batch 的使用长度允许为 0；但如果 `seqUsedQ` 的总和为 0，metadata 当前要求
  `totalSize > 0`，因此会返回参数错误。
- TND 下 `cuSeqLens` 仍然必需，且其长度至少为 `B + 1`。
- 当前 arch22 只支持 `BSND/BSND` 和 `TND/TND`，不支持混合 Q/K layout。
- 任意 `seqUsed` 存在时，arch22 主算子要求 metadata 存在。
- 本次没有修改 arch35 目录，也不改变 arch35 已有行为。
- arch22 kernel 内部确定性分支已适配 `seqUsed` 任务映射，但当前正式 API 约束仍说明 Atlas A2/A3
  不支持通过 `aclrtCtxSetSysParamOpt` 开启确定性计算。本文不将内部代码分支等同于用户可用能力。

## 11. 验证情况

已完成以下静态和编译验证：

```bash
git diff --check -- \
  attention/sparse_lightning_indexer_kl_loss_grad \
  attention/sparse_lightning_indexer_kl_loss_grad_metadata
```

检查通过，没有空白符错误。

已在 Ascend 910B 构建配置下编译主算子和 metadata 算子：

```bash
source /data/zyf/CANN/Ascend/ascend-toolkit/latest/set_env.sh
bash build.sh --opkernel --soc=ascend910b \
  --ops=sparse_lightning_indexer_kl_loss_grad,sparse_lightning_indexer_kl_loss_grad_metadata \
  --aicpu_kernel -j8
```

构建成功，`ops_transformer_kernel` 目标无编译错误。

当前尚未完成 NPU 运行精度验证。现有 pytest 用例将物理分配长度和实际长度绑定在一起，不能直接验证
“分配长度大于 `seqUsed`”的 padding 语义。

建议后续至少覆盖：

| 维度     | 建议用例                                                               |
| -------- | ---------------------------------------------------------------------- |
| layout   | BSND/BSND、TND/TND                                                     |
| 可选输入 | 都不传、只传 Q、只传 K、Q/K 都传                                       |
| dtype    | FP16、BF16                                                             |
| mask     | `maskMode=0`、`maskMode=3`，不同 `cmpRatio`                      |
| 执行模式 | 非确定性；确定性分支仅做内部代码验证，不作为 A2/A3 公开能力            |
| 长度边界 | `used=alloc`、`used=1`、单 batch `used=0`、多 batch 长短差异明显 |
| 非法输入 | 数组长度小于 B、负值、值大于分配长度、缺少 metadata                    |
| 输出检查 | 有效区域与 golden 一致，Q/K padding 对应梯度和辅助输出为 0             |

## 12. 修改文件清单

| 文件                                                                      | 修改作用                                                          |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `op_host/sparse_lightning_indexer_kl_loss_grad_tiling_general.cpp`      | 增加 metadata 条件必传校验，并将两个可选输入状态编码到 tiling key |
| `op_kernel/sparse_lightning_indexer_kl_loss_grad_template_tiling_key.h` | 声明并选择`HAS_SEQUSED_Q/K` 模板参数                            |
| `op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_common.h`       | 在`SLIType` 中保存两个编译期状态                                |
| `op_kernel/sparse_lightning_indexer_kl_loss_grad.cpp`                   | 将两个状态从全局 kernel 入口传入 arch22 实现                      |
| `op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_base.h`         | 读取有效长度、压缩 Q 任务映射、保持物理偏移、清零所有跳过行输出   |
| `op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_vector.h`       | vector 流程使用有效长度，并修正确定性任务重映射                   |
| `sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch22.h`         | 校验`seqUsed`，按有效长度生成任务总数和多核负载                 |
| `docs/aclnnSparseLightningIndexerKLLossGrad.md`                         | 更新 A2/A3 的 seqUsed 参数约束和 metadata 条件必传规则            |

## 13. 总结

本次修改在不改变 Tensor 物理布局和原有无 `seqUsed` 调用方式的前提下，为 arch22 增加了按 batch 控制
Q/K 有效序列长度的能力。存在 `seqUsedQ` 时，metadata 会压缩 Q 任务空间；只存在 `seqUsedK` 时，Q
任务空间保持不变，但负载估算和 K 有效边界使用 `usedK`。模板 tiling key 将可选输入状态传入 kernel，
base、vector 和内部确定性 scatter-add 路径统一使用有效长度。对调用者最重要的新增约束是：arch22 下
传入任意 `seqUsed` 时，必须传入由相同参数生成的 metadata。
