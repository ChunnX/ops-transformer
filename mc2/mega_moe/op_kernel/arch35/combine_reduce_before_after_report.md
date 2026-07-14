# CombineReduce 抽取前后逻辑说明

## 1. 先说结论

CombineReduce 这次修改的核心是“抽取模块”，不是改掉数学算法。

修改前，CombineReduce 逻辑以 `Unpermute*` 成员函数的形式写在 `mega_moe.h` 的 `MegaMoe` 类里，直接读取 `m_`、`k_`、`topK_`、`params_` 等类成员。

修改后，这部分逻辑被放到独立文件 `mega_moe_combine_reduce.h` 中，改成了一组单核函数。调用方把分核信息、shape、GM 地址和同步信息显式传进去。

不变的核心公式是：

```text
最终输出[token]
    = expert输出[token][0] * topK权重[token][0]
    + expert输出[token][1] * topK权重[token][1]
    + ...
    + expert输出[token][topK-1] * topK权重[token][topK-1]
```

如果通信中的 expert 输出是量化数据，则先反量化为 FP32，再做上面的加权累加。

## 2. 初学者需要知道的几个概念

### 2.1 token

token 可以先简单理解为一条需要处理的输入数据。每个 token 有 `k` 个数值，所以一个 token 可以看成长度为 `k` 的向量。

### 2.2 expert 和 TopK

MoE 中有很多 expert。一个 token 通常不会交给所有 expert，而是交给得分最高的 `topK` 个 expert。

每个 expert 都会返回一个长度为 `k` 的结果，路由阶段还会为每个 expert 产生一个权重。CombineReduce 的任务就是把这 `topK` 个结果加权合并。

### 2.3 GM 和 UB

- GM：全局内存，容量大，但访问成本高。
- UB：核上的局部缓冲区，容量小，但计算时使用更方便。

所以代码不是一次把所有 token 都放进 UB，而是每次处理最多 1024 个 token 的一个 batch。

### 2.4 AIC 和 AIV

- AIC 主要负责矩阵计算。
- AIV 主要负责向量计算、数据搬运和 CombineReduce。

`CrossRankSyncSingleCore` 中对 AIC 直接返回，因为这个单核同步逻辑是给 AIV 走的。

## 3. CombineSend 和 CombineReduce 不是一件事

这两个名字很像，但职责不同：

| 阶段 | 作用 | 简单比喻 |
| --- | --- | --- |
| CombineSend | 把各 expert 的输出送回 token 所在位置 | 快递员把多个包裹送回来 |
| CombineReduce | 读取送回的 `topK` 份 expert 结果，按权重合并 | 收件人把多份材料按比例混合成最终结果 |

整体数据流可以看成：

```text
原始 token
   |
   v
路由到 topK 个 expert
   |
   v
expert 计算（GMM1 / GMM2）
   |
   v
CombineSend：将每个 expert 结果送回
   |
   v
expandedX：[token][topK][k]
   |
   v
CombineReduce：反量化 + 乘 topK weight + 累加
   |
   v
output：[token][k]
```

## 4. 修改前的逻辑

修改前，CombineReduce 在 `mega_moe.h` 中叫做 `Unpermute`。它是 `MegaMoe` 类内的一组成员函数。

### 4.1 修改前的调用顺序

```text
MegaMoe::Process
   |
   +-- CrossRankSyncInWorldSize()
   |      等待各 rank 都进入对应阶段
   |
   +-- Unpermute()
          |
          +-- TilingByCore()
          |      按物理 AIV 核划分 token 范围
          |
          +-- UnpermuteBuffInit()
          |      在 UB 中准备数据、权重和 scale buffer
          |
          +-- 按最多 1024 token 分 batch
                 |
                 +-- UnpermuteLoadWeights()
                 |      加载当前 batch 的 topK 权重
                 |
                 +-- 遍历 batch 中的 token
                        |
                        +-- UnpermuteProcessToken()
                        |      遍历 topK expert，反量化并加权累加
                        |
                        +-- FP32 -> BF16
                        |
                        +-- 写入最终 output
```

### 4.2 修改前的特点

`Unpermute*` 函数可以直接使用下面这些类成员：

- `m_`：token 总数。
- `k_`：每个 token 的隐藏维度。
- `topK_`：每个 token 选中的 expert 数量。
- `params_.peermemInfo.combineSendPtr`：各 expert 送回的结果。
- `params_.probsGmAddr`：topK weight。
- `params_.y2GmAddr`：最终输出地址。
- `dataResTensor_` 等 LocalTensor 成员：UB 中间 buffer。
- `aivCoreIdx_` 和 `blockAivNum_`：分核信息。

优点是写起来直接；缺点是函数与 `MegaMoe` 大类绑得很紧，想单独复用或测试很困难。

## 5. 修改后的逻辑

修改后，`mega_moe.h` 只负责组装参数和调度，真正的 CombineReduce 处理放在 `mega_moe_combine_reduce.h` 中。

### 5.1 修改后的调用顺序

```text
MegaMoe::Process
   |
   +-- 构造 CombineReduceTileSchedule
   |      tileId  = aivCoreIdx_
   |      tileNum = blockAivNum_
   |
   +-- 构造 CombineReduceSyncContext
   |      rankSyncAddr + rankId + worldSize
   |
   +-- CrossRankSyncSingleCore()
   |
   +-- 构造 CombineReduceShapeContext
   |      m + k + topK
   |
   +-- 构造 CombineReduceBufferContext
   |      expandedXAddr + topKWeightsAddr + outputAddr
   |
   +-- ProcessCombineReduceSingleCore()
          |
          +-- ComputeCombineReduceTileRange()
          +-- InitCombineReduceWorkspace()
          +-- 按最多 1024 token 分 batch
                 +-- LoadCombineReduceWeights()
                 +-- ProcessCombineReduceToken()
                 +-- FP32 -> BF16
                 +-- 写入 output
```

### 5.2 新旧函数映射

| 修改前 | 修改后 | 职责是否变化 |
| --- | --- | --- |
| `TilingByCore` | `ComputeCombineReduceTileRange` | 基本不变，但新函数使用外部传入的逻辑 tile |
| `UnpermuteBuffInit` | `InitCombineReduceWorkspace` | 不变，将类成员 Tensor 改成 `workspace` 字段 |
| `UnpermuteLoadWeights` | `LoadCombineReduceWeights` | 不变，将类成员地址改成 context 参数 |
| `UnpermuteProcessToken` | `ProcessCombineReduceToken` | 数学计算不变 |
| `Unpermute` | `ProcessCombineReduceSingleCore` | 不变，仍是单核总入口 |
| `CrossRankSyncInWorldSize` | `CrossRankSyncSingleCore` | 同步协议不变，核 ID/核数改为显式调度参数 |

## 6. 新模块中每个部分是做什么的

### 6.1 `CombineReduceTileSchedule`

描述“当前单核是第几个逻辑 tile，一共有多少个 tile”。

```cpp
struct CombineReduceTileSchedule {
    uint32_t tileId;
    uint32_t tileNum;
};
```

这是本次抽取的关键：单核逻辑不再自己读物理核 ID，而是只处理调用方交给它的连续逻辑 ID。

### 6.2 `CombineReduceShapeContext`

保存计算形状：

- `m`：token 数量。
- `k`：每个 token 的数据长度。
- `topK`：每个 token 需要合并的 expert 数量。

修改前对应 `m_`、`k_`、`topK_` 三个类成员。

### 6.3 `CombineReduceBufferContext`

保存三个 GM 地址：

- `expandedXAddr`：CombineSend 送回的 expert 结果。
- `topKWeightsAddr`：路由产生的 topK 权重。
- `outputAddr`：CombineReduce 最终输出。

数据布局可以简化理解为：

```text
expandedXAddr     -> [m][topK][k]
topKWeightsAddr  -> [m][topK]
outputAddr       -> [m][k]
```

### 6.4 `CombineReduceSyncContext`

保存跨 rank 同步需要的信息：

- `rankSyncAddr`：同步区域基地址。
- `rankId`：当前 rank。
- `worldSize`：通信域中的 rank 总数。

### 6.5 `CombineReduceWorkspace`

将修改前散落在 `MegaMoe` 类里的 LocalTensor 集中在一个对象中：

| buffer | 作用 |
| --- | --- |
| `dataResTensor` | 保存 BF16 输入和最后的 BF16 搬出结果 |
| `dataResFp32Tensor` | 保存 FP32 中间数据和加权累加结果 |
| `topKWeightsTensor` | 保存当前 batch 的 FP32 topK weight |
| `topKWeightsBf16Tensor` | BF16 weight 的搬运中转 buffer |
| `bf16ScaleTensor` | MXFP8 反量化中的 BF16 scale 中间结果 |
| `fp32ScaleTensor` | MXFP8 反量化中的 FP32 scale |

### 6.6 `ComputeCombineReduceTileRange`

把 `m` 个 token 分给 `tileNum` 个逻辑 tile。

例如 `m = 10`、`tileNum = 3`：

```text
lengthPerTile = ceil(10 / 3) = 4

tile 0 -> token [0, 4)  ，4 个
tile 1 -> token [4, 8)  ，4 个
tile 2 -> token [8, 10) ，2 个
```

这与修改前 `TilingByCore(m_, ..., align=1)` 的划分方式一致。新函数还显式检查了 `tileNum == 0`、`tileId` 越界和当前 tile 无 token 的情况。

### 6.7 `InitCombineReduceWorkspace`

根据 `k`、`topK`、权重类型和量化模式在 UB 中划分 buffer。

它还会返回 `batchLength`：

```text
batchLength = min(1024, 当前 tile 负责的 token 数)
```

这样 topK weight buffer 只需容纳当前 batch，不需要容纳全部 `m` 个 token。

### 6.8 `LoadCombineReduceWeights`

从 GM 加载当前 batch 的 topK weight：

- 权重原本是 FP32：直接搬到 `topKWeightsTensor`。
- 权重原本是 BF16：先搬到 `topKWeightsBf16Tensor`，再 Cast 为 FP32。

后面的乘法和累加统一使用 FP32 weight。

### 6.9 `ProcessCombineReduceToken`

这是数学计算的核心，一次只处理一个 token。

它对 `expertId = 0 ... topK-1` 做循环：

1. 取出当前 expert 的 topK weight。
2. 从 `expandedX` 搬入当前 expert 输出。
3. 非量化模式下，将 BF16 输出 Cast 为 FP32。
4. 量化模式下，读取 FP8 token 和 scale，反量化为 FP32。
5. 用 expert 输出乘以对应 weight。
6. 累加到 `dataResFp32Tensor`。

`EVENT_ID0` 和 `EVENT_ID1` 对应两块搬运 buffer，一块在计算时，另一块可为下一轮数据做准备。

### 6.10 `ProcessCombineReduceSingleCore`

这是单核模块的总入口，它自己不关心当前逻辑 tile 是如何映射到物理核的。

它负责：

1. 计算当前 tile 的 token 范围。
2. 初始化 UB workspace。
3. 建立 expandedX 和 output 的 GlobalTensor 视图。
4. 按 batch 加载权重。
5. 逐 token 调用 `ProcessCombineReduceToken`。
6. 将 FP32 累加结果 Cast 回 BF16。
7. 写入 `output[tokenIdx * k]`。

### 6.11 `CrossRankSyncSingleCore`

这个函数是阶段间的“集合点”。它要确保通信域中对应 rank 都进入当前阶段，避免有的 rank 还没送完数据，另一个 rank 就开始 Reduce。

关键地址布局：

```text
rankSyncAddr + [0, 48 KiB)       : rank 间通知区
rankSyncAddr + 48 KiB + tile*64B : 每个逻辑 tile 的本地 syncCnt
```

每个 tile 按下面的方式负责 rank：

```text
rankIndex = tileId, tileId + tileNum, tileId + 2*tileNum, ...
```

该函数在 `MegaMoe::Process` 中使用两次：

1. 前面的全卡同步，确保初始化和数据准备阶段对齐。
2. CombineReduce 前再同步，确保 CombineSend 已完成。

同一个 tile 的 `syncCnt` 每次调用加 1，所以两次同步可以共用地址，又不会把上一轮的通知当成本轮通知。

## 7. 一个具体的数学例子

假设某个 token 的 `topK = 2`、`k = 4`：

```text
expert 0 输出 = [ 1,  2,  3,  4]
expert 1 输出 = [10, 20, 30, 40]

expert 0 权重 = 0.7
expert 1 权重 = 0.3
```

CombineReduce 的计算是：

```text
输出 = expert0 * 0.7 + expert1 * 0.3
     = [1*0.7 + 10*0.3,
        2*0.7 + 20*0.3,
        3*0.7 + 30*0.3,
        4*0.7 + 40*0.3]
     = [3.7, 7.4, 11.1, 14.8]
```

代码中使用 FP32 完成乘法和累加，最后再 Cast 回 BF16 输出。

## 8. 修改前后哪些是等价的

### 8.1 token 分核等价

修改前：

```text
TilingByCore(m_, coreLen, coreOffset, 1)
coreIdx = GetBlockIdx()
coreNum = GetBlockNum() * 2
```

修改后调用方传入：

```text
tileId  = aivCoreIdx_  = GetBlockIdx()
tileNum = blockAivNum_ = GetBlockNum() * 2
```

两者都按 `ceil(m / 核数)` 分配连续 token 区间，所以当前 `MegaMoe` 调用方下分核结果等价。

### 8.2 权重处理等价

- FP32 weight 仍是直接搬运。
- BF16 weight 仍是搬运后 Cast 为 FP32。
- 权重索引仍是 `localIdx * topK + expertId`。

### 8.3 expert 累加等价

- expert 0：用 `Muls` 建立第一份加权结果。
- expert 1 到 topK-1：先 `Muls`，再 `Add` 到累加 buffer。
- 最后仍是 FP32 转 BF16 后搬出。

### 8.4 跨 rank 同步等价

修改前的 `aivCoreIdx_` / `blockAivNum_` 被替换为 `tileSchedule.tileId` / `tileSchedule.tileNum`。当前调用方传入的值完全一致，同步计数地址和 rank 分工方式不变。

## 9. 真正发生变化的地方

| 方面 | 修改前 | 修改后 |
| --- | --- | --- |
| 代码位置 | `mega_moe.h` 大类内 | 独立 `mega_moe_combine_reduce.h` |
| 状态来源 | 隐式读取类成员 | 通过 context 显式传入 |
| UB Tensor | `MegaMoe` 的成员变量 | `CombineReduceWorkspace` 中的局部字段 |
| 分核 | 函数内读物理核 ID | 调用方传入连续逻辑 tile ID |
| 边界检查 | 依赖 `TilingByCore` | 显式检查 tileNum、tileId 和空 tile |
| 可复用性 | 必须依附 `MegaMoe` 类 | 可由其他调度器构造 context 后调用 |

所以，这次改动的主要收益是解耦和复用，不是更换 CombineReduce 算法。

## 10. 相关文件分工

### `mega_moe.h`

普通 MegaMoe 路径的总编排器。它串联 dispatch、GMM1、GMM2、CombineSend 和 CombineReduce。现在该文件不再包含 CombineReduce 的具体实现，而是构造 context 并调用单核模块。

### `mega_moe_impl.h`

放置 MegaMoe 通用实现和 GMM 相关辅助逻辑，并引入 `mega_moe_combine_send.h`。它不是 CombineReduce 的总入口。

### `mega_moe_combine_send.h`

负责 CombineSend：读取 GMM2 输出，根据三元组找到目标 token/rank，必要时量化，再把 expert 结果发回去。

`mega_moe_combine_reduce.h` 还会复用该文件中的 `DeQuantMxFp8`，用于将通信中的 MXFP8 数据恢复为 FP32。

### `mega_moe_combine_reduce.h`

负责本报告描述的最终合并：分 token、分 batch、加载 topK weight、反量化、加权累加、写回 output，并提供跨 rank 同步函数。

### `mega_moe_layered.h`

分层/URMA 路径的总编排器。按当前工作区状态，该文件已恢复为主线实现，仍使用自己的 `Unpermute` 和 `CrossRankSyncInWorldSize`，本次不调用新的 CombineReduce 单核模块。

## 11. 当前需要注意的风险点

1. `tileId` 必须是 `[0, tileNum)` 内连续的逻辑 ID。如果外部映射有空洞，token 范围可能被遗漏。
2. 所有参与核必须对 `tileNum`、`rankSyncAddr` 布局和同步轮次有一致认识，否则可能在 barrier 处一直等待。
3. `expandedXAddr` 的布局必须与 CombineSend 一致。非量化和量化路径的单 token 步长不同。
4. UB 地址是手工连续划分的。增加 workspace 字段时，必须重新检查大小、对齐和地址是否重叠。
5. 抽取前后的静态逻辑已对照，但本地没有完成昇腾设备上的编译和数值对比。最终验证建议包含非量化、MXFP8 E4M3/E5M2、FP32/BF16 weight 以及 `m < tileNum` 等边界场景。

## 12. 一句话总结

修改前是“`MegaMoe` 大类自己做 Unpermute”；修改后是“调用方把单核所需的信息打包传入，独立 CombineReduce 模块完成同样的反量化、TopK 加权和累加”。

## 13. 两张卡时 CombineReduce 怎么执行

### 13.1 先记住最重要的一点

两张卡时，CombineReduce 不是把“卡0的总结果”和“卡1的总结果”相加。

正确流程是：

1. token 可以被发给两张卡上的 expert。
2. 两张卡各自计算本卡 expert 负责的部分。
3. CombineSend 把 expert 结果送回该 token 原本所属的卡。
4. 跨卡同步确认两边都送完。
5. 每张卡只对属于自己的 token 做 CombineReduce。

可以把这理解为“分布式计算，回到原地汇总”。

### 13.2 例子设定

假设有两张卡：

```text
卡0 = rank 0，放置 expert 0，拥有原始 token A
卡1 = rank 1，放置 expert 1，拥有原始 token B

worldSize = 2
topK = 2
k = 2
```

token A 和 token B 都同时选中 expert 0 和 expert 1，但权重不同：

```text
token A: expert 0 权重 = 0.6，expert 1 权重 = 0.4
token B: expert 0 权重 = 0.25，expert 1 权重 = 0.75
```

### 13.3 第一步：Dispatch 把 token 送到 expert

token A 原本属于卡0：

- A 发给 expert 0，expert 0 在卡0，可以本地处理。
- A 发给 expert 1，expert 1 在卡1，需要跨卡发送。

token B 原本属于卡1：

- B 发给 expert 0，expert 0 在卡0，需要跨卡发送。
- B 发给 expert 1，expert 1 在卡1，可以本地处理。

```text
                 +------------------+
token A (属于卡0) |---> expert 0 (卡0)
                 +---> expert 1 (卡1)

                 +---> expert 0 (卡0)
token B (属于卡1) |---> expert 1 (卡1)
```

### 13.4 第二步：两张卡各自执行 expert 计算

假设最后得到下面四个结果：

```text
expert 0 对 token A 的结果 = [ 1,  2]   在卡0上算出
expert 1 对 token A 的结果 = [10, 20]   在卡1上算出

expert 0 对 token B 的结果 = [ 3,  4]   在卡0上算出
expert 1 对 token B 的结果 = [30, 40]   在卡1上算出
```

此时，token A 的两份结果分散在两张卡上，token B 也是一样。还不能直接做 CombineReduce。

### 13.5 第三步：CombineSend 把结果送回 token 归属卡

CombineSend 根据三元组中的 rank、tokenIndex 和 topkIndex 把结果放回正确位置：

```text
卡0上 expert 0 算出的 A 结果 -> 保存到卡0的 A 槽位
卡1上 expert 1 算出的 A 结果 -> 跨卡发送到卡0的 A 槽位

卡0上 expert 0 算出的 B 结果 -> 跨卡发送到卡1的 B 槽位
卡1上 expert 1 算出的 B 结果 -> 保存到卡1的 B 槽位
```

CombineSend 完成后，两张卡的 `expandedX` 可以简化表示为：

```text
卡0 expandedX（只关心属于卡0的 token A）
    A, topk 0 -> [ 1,  2]   来自卡0的 expert 0
    A, topk 1 -> [10, 20]   来自卡1的 expert 1

卡1 expandedX（只关心属于卡1的 token B）
    B, topk 0 -> [ 3,  4]   来自卡0的 expert 0
    B, topk 1 -> [30, 40]   来自卡1的 expert 1
```

### 13.6 第四步：两卡同步

`CrossRankSyncSingleCore` 确认两张卡都完成了 CombineSend。

在 `worldSize = 2` 的情况下，逻辑 tile 会分工处理 rank 0 和 rank 1 的通知。每张卡会：

1. 将本轮 `syncCnt` 通知对应 rank。
2. 等待本卡同步区中对应的计数达到本轮值。
3. 所有本卡 AIV 再做一次 `SyncAll<true>()`。

只有这一步通过后，CombineReduce 才可以安全读取 `expandedX`。

### 13.7 第五步：每张卡本地做 CombineReduce

卡0只处理属于自己的 token A：

```text
A 最终输出
  = [1, 2] * 0.6 + [10, 20] * 0.4
  = [0.6, 1.2] + [4, 8]
  = [4.6, 9.2]
```

卡1只处理属于自己的 token B：

```text
B 最终输出
  = [3, 4] * 0.25 + [30, 40] * 0.75
  = [0.75, 1] + [22.5, 30]
  = [23.25, 31]
```

最后：

```text
卡0 output -> token A = [4.6, 9.2]
卡1 output -> token B = [23.25, 31]
```

卡0和卡1不会再把 A 和 B 相加，因为它们是两个不同 token 的最终结果。

### 13.8 两卡例子的完整流程图

```text
                 Dispatch                      Expert 计算

卡0 token A -----------------------> 卡0 expert 0 --- A0=[1,2]
       |                                      ^
       |                                      |
       +------------------------------> 卡1 expert 1 --- A1=[10,20]

卡1 token B -----------------------> 卡1 expert 1 --- B1=[30,40]
       |                                      ^
       |                                      |
       +------------------------------> 卡0 expert 0 --- B0=[3,4]


                 CombineSend：按 token 归属卡送回

     A0 留在卡0 -------------------+
     A1 从卡1发到卡0 ----------+----> 卡0 expandedX[A][0..1]

     B0 从卡0发到卡1 ----------+
     B1 留在卡1 -------------------+----> 卡1 expandedX[B][0..1]


                 CrossRankSyncSingleCore

               卡0 <------ 互相通知 ------> 卡1
                    确认 CombineSend 都完成


                 各卡本地 CombineReduce

卡0: A = A0*0.6  + A1*0.4  = [4.6, 9.2]
卡1: B = B0*0.25 + B1*0.75 = [23.25, 31]
```

### 13.9 两卡场景的一句话总结

两张卡可以分别计算同一个 token 选中的不同 expert；CombineSend 把这些 expert 结果收集回 token 的原始卡，同步完成后，原始卡再用 CombineReduce 做 TopK 加权求和。
