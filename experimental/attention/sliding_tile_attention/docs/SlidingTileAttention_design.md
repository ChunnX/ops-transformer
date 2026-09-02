# SlidingTileAttention 算子设计

## 1. 目标与范围

本实现面向 Atlas A2/A3，提供 solution-owned Ascend C SlidingTileAttention。输入为 rank-4 BNSD
FP16/BF16 Tensor，非空计算支持 `B > 0`、`1 <= N <= 128`、任意正的 8 倍数 `D`，以及长度 3 广播或
长度 `3*N` 的逐 head window；每个三元组的三个值均须为正奇数。shape 一致的任意空 Tensor 在 ACLNN
层直接成功返回。

语义为：

```text
softmax((Q @ K^T) / sqrt(D), masked_by_sliding_tile_window) @ V
```

## 2. 接口与调用链

```text
aclnnSlidingTileAttentionGetWorkspaceSize
  -> rank/dtype/shape/attribute/window 契约校验
  -> 空 Tensor：workspaceSize=0，直接 ACLNN_SUCCESS
  -> q/k/v 显式 Contiguous materialize
  -> window 规范化为 3*N
  -> L0 SlidingTileAttention
  -> ADD_TO_LAUNCHER_LIST_AICORE(SlidingTileAttentionFai)
  -> op_host tiling/workspace/blockDim
  -> runtime metadata 选择固定快路径或动态通用路径
  -> solution-owned MIX_AIC_1_2 Ascend C kernel
```

公开接口只有一组标准两段式 ACLNN。非连续输入由 executor 管理的 `Contiguous` 节点转为连续 BNSD；
空 Tensor 不进入 L0，因此不发生无意义的 tiling、workspace 分配或 device launch。

## 3. Runtime dispatch

分派只读取公开运行时 metadata，不读取 case id、输入值、数据分布或 timing 特征。

| 条件 | tiling key / 路径 | 设计目的 | 不支持时行为 |
| --- | --- | --- | --- |
| `dtype=BF16 && D=64` | key 1 / BF16 D64 lean 变体 | 编译期删除该 dtype/D 不需要的通用控制流，保留 runtime B/N/sequence/window 解码 | 不适用 |
| 已证明的 FP16 packed/grouped family | key 2/3/5/7/8/9 | 复用固定栈或 grouped 数据流，同时保留 runtime work/base/stride 所有权 | 不适用 |
| D256 padded-text 分片 | key 10/11 | partial kernel 生成分片状态，merge kernel 合并并写回 | 不适用 |
| 其他合法 dtype/D | key 0 / 通用变体 | runtime head/work mapping、动态 scale、D 分块、逐 head window；D72 full-window 使用双概率 PV | 不适用 |
| 非法 rank/dtype/shape/window/属性 | 无 kernel | 保护接口契约和内存安全 | L2/op_host fail loudly |

所有 key 均由当前源码预编译并进入 A2/A3 包，不调用 framework/reference/vendor whole-op。每个变体内部仍
依据合法 runtime metadata 选择固定 384-row task decode、grouped/compact 或动态通用数据流；例如满足
已实现画布条件的 `N=8,D=64` 输入可复用固定 task decode。该路由是 case-independent 的 runtime metadata
优化，任何未来输入只要满足同一条件都会进入同一路径。

SoC 闭合范围为 `ascend910b` 与 `ascend910_93`：OpDef、binary config、CMake compile target 和 package
路径保持一致。

## 4. Host tiling 与 workspace

画布枚举如下：

| `seqShape` | 画布 | image token 数 |
| ---: | --- | ---: |
| 1 | `30x48x80` | 115200 |
| 2 | `36x48x48` | 82944 |
| 3 | `18x48x80` | 69120 |

`seqShape=1` 的 storage sequence 可以大于 `image_tokens+textLength`，但 image 后的 capacity 和
`textLength` 均不超过 512，且 `textLength` 不超过该 capacity；padding query 仍产生输出，但 padding 行不进入
K/V 集合。`seqShape=2/3` 为无 text 的固定画布长度。tile 固定为 `[6,8,8]`，每 tile 384 token；query chunk
处理 128 行。`blockDim` 从 runtime `B*N*queryChunks` 与平台 AIC core 数计算，不再假定 batch 仅为 1/2或
head 固定为 8。

workspace 按 active core 和 runtime D 计算，承载 CATLASS ring stage、softmax recurrence、输出归一化，
以及 BF16 V staging。每个乘法与加法均在 host 侧做容量/溢出检查；合法空输入在这些计算前快返。

## 5. Window 与 mask

L2 接受两种 window 表示：

- 长度 3：一个 `[t,h,w]` 三元组，广播到 N 个 head。
- 长度 `3*N`：每个 head 一个独立三元组。

每个窗口值必须为正奇数。规范化结果携带完整 runtime head 数，设备侧以 `head*3` 读取，不能用固定
24 元素或固定窗口模式编号替代。

query token 首先映射到 tile id，再还原 `(t,h,w)` 坐标。每个轴使用 `radius=window/2`，将中心限制在
合法范围，从而在首尾 tile 保持 clamp 语义；窗口大于画布 extent 时覆盖该轴全部 tile。KV 以连续
width segment 枚举，减少离散 GM 访问。

- image query：局部 image window；有 text 时再追加 text segment。
- text query：完整 image + text 序列。
- 非 causal，无额外上三角 mask。

## 6. Kernel 数据流

### 6.1 通用计算

1. AIC 通过 CATLASS 分块完成 QK，K 维按最多 128 列的块遍历 runtime D。
2. AIV 以 fp32 执行 online softmax，scale 为 runtime `1/sqrt(D)`，维护跨 KV segment 的 row max/sum。
3. AIC 完成 P@V；V 的输出列按 runtime D 分块。
4. AIV 对各 D 列块执行 rescale、最终归一化和写回，避免最终 UB 临时 Tensor 随 D 无界增长。

`D=64` 是同一数学数据流的单 D-block 实例，其中 BF16 使用 key 1 lean 编译变体；更大的 D 通过多个列块
覆盖，尾块处理所有正的 8 倍数。runtime batch/head/query work id 决定 GM 基址，输出所有权为一个
`(batch,head,queryChunk)`，不同 core 不重叠写。

### 6.2 FP16

Q/K/V 直接参与 CATLASS QK/PV，online softmax 与 recurrence 使用 fp32 状态，最终转换为 FP16 写回。

### 6.3 BF16

BF16 不使用设备外 `Cast(Q/K/V) + FAI + Cast(O)` 多 kernel 链。Q/K 保持 BF16 输入，当前 CATLASS
PV lowering 所需的完整 V tile 在同一次 STA kernel 的私有 workspace 中 staging 为 half；之后沿标准
online-softmax/PV/rescale 数据流完成计算并直接写回 BF16。

FP16 D72 full-window 的通用路径将 FP32 softmax probability 分解为 FP16 main 与缩放 residual，执行两次
solution-owned PV 并在 FP32 recurrence 中回加 residual；两次 PV 使用同一 unit-flag L0C completion family。
内部 BF16-to-half staging 不是独立 Cast 算子，也没有 framework、CPU 或 vendor whole-op 替代路径。

## 7. 同步与数据所有权

每个 active core 拥有独立 ring workspace。AIC 发布只读 QK/PV 中间结果，AIV 负责 recurrence、rescale、
归一化和 copy-out；event id 使用命名常量并限制在 FFTS 消息编码范围。动态路径的 D 列块顺序复用同一
core-owned 临时区，前一块完成写回后才复用，不与其他 core 共享可变状态。

## 8. 验证与证据边界

- Host contract test 覆盖 heads 边界 `1/128`、多个动态 D、window 广播/逐 head、非法 window、空 batch
  和 sequence/text 一致性。
- taskbook runner 逐项复现原始 `all_sta.json` 的 50 个 case，字段级覆盖 dtype、B/N/S/D、window、text、
  hasText 和 seqShape；padded storage case 额外检查最后 query 行，全部 case 检查同输入 bitwise 确定性。
- reviewer runner 保留原 18 个 `N=8,D=64` 固定快路径 case，并增加 case 19、23、24、25 的动态
  N/D/window、`B=2/3`、BF16 padded-D 和非连续 layout 代表性检查。runner 固定执行这 22 个 case，
  未列入自验证表的 workload 不由该日志给出性能结论。非零、非均匀 Q/K 使 runtime scale 可观测。
- 发布结论引用当前 PR head 的 fresh taskbook 50-case 和 reviewer 22-case build/runner 原始日志与
  [`docs/evidence`](evidence) 中的 CSV、源码摘要和校验清单。
- 当前 PR 源码树摘要
  `716922e9a9d522a2a3c768ea89935e2518630c869c90a163133de6f21a57984d` 的任务书结果为 50/50，
  reviewer22 结果为 22/22（warmup3/repeat5）；72 个设备 case 均为 bitwise deterministic，两个 suite 的
  空 Tensor workspace0/API 检查均通过。任务书旧性能值和本地 runner 延迟均不用于性能达标声明。
- 同一算法的 OpForge final-source `public_v6` correctness-only gate
  `eval_20260831_051713` 为 95/95，且 dispatch、当前 binary digest、Ascend C 实现和 no-fallback receipt
  全部匹配。该外部开发闭环不是 PR 自验证的运行依赖，也没有新增 timing/speedup 结论。

## 9. 无 fallback 边界

生产路径只调用当前目录构建出的 ACLNN custom op。源码中没有 Python reference、CPU、PyTorch、
FastVideo、Triton/CUDA、vendor whole-op、peer 或其他 backend fallback；不支持的 metadata 直接失败。
