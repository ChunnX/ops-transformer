# 0721 TileLang 与 arch22 AscendC 实现差异

## 1. 文档范围

本文记录当前 `0721.py` 与官方 arch22 AscendC 实现之间的结构差异、已验证取舍和未决风险。对比重点是数学语义、数据所有权、异步流水和物理内存，而不是 API 名称是否逐字一致。

官方实现主要位于：

- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_base.h`
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h`
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector2.h`
- `op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_service_cube.h`

TileLang 源码只表达逻辑意图。涉及 UB/L1/L0 复用、异步 DMA、MMA/FIX 和跨核事件时，生成的 `0721_kernel_source.cpp` 才是最终执行语义，必须同时审计。

## 2. 当前实现概览

当前 kernel 以 `(batch, query)` 为 task，按运行时 `launch_core_num` 分配到 AIC/AIV：

1. **V0 Gather**：两个 AIV 分片读取稀疏索引，gather Key、KeyRope 和 KeyIndex，写入连续 GM workspace。
2. **C1 Forward Matmul**：使用显式 L1/L0A/L0B/L0C 和 `T.mma` 计算 BMM1、BMM2；L1 到 L0 的转置在 copy 阶段完成。
3. **V1 P/SY**：AIV0 计算 PReduced，AIV1 计算 ReduceSumY 和 SoftmaxI，并通过 GM workspace 交换结果。
4. **V1 Backward**：两个 AIV 各处理一半 QueryIndex 头，计算 dWeight、reluGrad 和 Loss partial。
5. **C2 Gradient Matmul**：使用显式 `T.mma` 计算 dKey partial 和 dQuery accumulation。
6. **V2 Finalize**：按稀疏索引将 dKey partial 原子累加，转换最终输出，并在 kernel 内完成 Loss reduction。

`TL_ASCEND_AUTO_SYNC` 保持关闭。当前流水依赖显式 RAW/WAR flag、跨核 flag、task slot 和固定深度调度。

## 3. K 维语义

当前实现中存在四个不同概念，不能混用：

| 名称 | 含义 | 决定方式 | 使用范围 |
| --- | --- | --- | --- |
| `K`（接口宽度） | SparseIndices 的 ABI/top-k 宽度 | 输入 shape | 原始输入 stride 和算子契约 |
| `KRealSize` | 每个 `(batch, query)` 的真实有效列数 | 运行时序列长度 | 尾块判断、mask 和 scatter 保护 |
| `KExec` | Cube、workspace 和主 Vector 阶段的静态执行宽度 | `max(KRealSize)` 按 `S2_WORKSPACE` 对齐，且至少一个 workspace block | kernel 编译 shape、C1/V1/C2 循环上界 |
| `KScatterExec` | V2 实际扫描的静态宽度 | `max(KRealSize)` 按 16 行对齐 | dKey scatter，避免扫描 `KExec` 的全部 padding |

当序列长度在主机侧不可见时，不能为取得 `max(KRealSize)` 引入设备同步；此时两个执行宽度保守退化到接口宽度。

主机侧选择两个静态 specialization：

- **full-valid**：所有 query 都满足 `KRealSize == K`，编译时删除尾部 mask 和动态有效长度读取。
- **tail-protected**：至少一个 query 存在无效尾列，保留 `KRealSize`、`ValidKMask` 和各阶段边界保护。

这不是 kernel 内每个元素的动态二选一，而是 cache key 区分的两个已编译版本。当前 TileLang 宏收集仍会检查未发射分支的 BufferRegion，因此 full-valid 版本仍需传入 rank 合法的占位张量。

## 4. 与官方实现的主要差异

| 区域 | 官方 arch22 | 当前 0721 | 原因与影响 |
| --- | --- | --- | --- |
| 本地内存队列 | `TPipe`/LocalTensor 队列具有明确物理 slot 和归还时机 | `T.alloc_ub`/`T.alloc_L1`/`T.alloc_L0*` 由静态 planner 分配 | 异步消费者完成前，逻辑生命周期不足以证明物理地址可复用 |
| V0 Gather | `DataCopyPad`、队列和实现私有的标量/索引访问 | 16 行、双 UB slot 的显式 gather，再写连续 GM workspace | 指令和事件协议不同；普通流水下仍有偶发 workspace 首错风险 |
| Cube service | Matmul Service 管理 phase、slot 和 MMA/FIX ownership | 显式 L1/L0A/L0B/L0C、`T.mma` 和手动 flag | `gemm_v0` 带隐藏同步，不能与当前全手动协议混用 |
| P/SY 尾部 | 队列与有效长度共同约束尾部 | 部分块用 Compare/Select 置负无穷，exp 后显式置零；完整无效块跳过 exp | 数学语义已对齐，但需要 mask DMA、Vector 指令和静态 padding |
| ReLU gradient | `bmm2 > 0` 后 Select | 同样使用 Compare/Select；16 行路径拆成四组 `4x64` | 分组适配后端 shape 限制，不改变 dWeight reduction |
| V1 backward 队列 | ping/pong LocalTensor 在固定物理地址复用 | 使用可验证的连续小 tile，依赖 planner 和显式 WAR | 逻辑双 buffer 不能证明物理隔离 |
| V2 Scatter | 可使用实现私有的本地 scatter/搬运能力 | 16 行批量读 partial，按非连续目标逐项 `atomic_add` | 随机目标不能合并成普通连续 atomic region |
| Loss partial | 按官方内部对齐协议归约 | 每个 scalar partial 独占 32B GM 槽，并补齐到完整 128-value chunk | 规避短 GM region/DataCopyPad lowering 边界问题 |
| 跨 task 流水 | run-info、task ratio 和 queue 管理多级重叠 | task slot、cross flag 和固定深度主循环 | 多 task 正确性依赖 event generation 与物理 buffer 生命周期 |

## 5. 当前数学语义

### 5.1 PReduced

P 路径先对 BMM1 缩放。部分尾块在减 SoftmaxMax 和 exp 之前，用 `ValidKMask > 0` 的 Compare/Select 将无效列设为 `-T.infinity(calc_dtype)`，避免上游异常值在后续形成 `inf * 0 = NaN`。exp 和除 SoftmaxSum 后再次乘有效 mask，确保无效列严格为零。

完整无效块不读取 BMM1，直接输出零。full-valid specialization 不生成这些尾部操作。

### 5.2 SoftmaxI

SY 路径先按 head 权重归约 BMM2，再在部分尾块上用 Compare/Select 将无效列设为负无穷。全局最大值确定后计算 exp；部分尾块的 exp 结果再次通过 Compare/Select 明确置零，完整无效块直接置零。

当前实现不再依赖 `exp(-1e30 - max)` 或其他有限大负数近似无效列。`-T.infinity(calc_dtype)` 是 softmax 最大值阶段的语义值。

### 5.3 ReLU gradient 与 dWeight

backward 的逻辑为：

```text
diff = SoftmaxI - PReduced
dWeight = reduce_sum(bmm2 * diff, K)
reluGrad = Select(bmm2 > 0, weight * diff, 0)
```

ReLU 判定直接使用 Compare/Select，不使用 `Exp`、倒数或其他近似。`rows_per_aiv == 16` 时，Compare/Select 被拆为四个静态对齐的 `4x64` 子区域；dWeight 仍对完整 `16x64` tile 做 reduction。其他行数路径使用完整 tile Compare/Select。

## 6. 已确认的后端约束

### 6.1 多维子区域必须保留父 stride

二维 parent buffer 的列切片必须保留父行 stride。例如 `[rows, 128]` 的 `[:, 64:128]` 是逐行取后 64 列，不是从线性 offset 64 开始连续取 `rows*64` 个元素。当前后端曾错误 lowering 该类区域，因此更大 parent tile 不能直接拆成多个列子块使用。

### 6.2 Vector 指令存在窄字段边界

部分 repeat/stride 参数在生成 AscendC 时使用 `uint8_t`。逻辑 repeat 达到 256 时会截断为 0。当前代码通过最多 64 行的 row-expand 分组以及 `4x64` Compare/Select 避免越界，但这只是源码适配。

### 6.3 Reduction 和 Compare/Select 对物理 shape 有额外约束

部分数学合法的 shape 不能直接进入 fast path，或会得到错误 lowering。P 的 axis-0 reduction 因此按 64 行 partial 分组后累加；backward 的 16 行 Compare/Select 采用四组 `4x64`。

### 6.4 异步本地内存复用缺少可证明的 ownership

`T.alloc_ub` 声明逻辑 buffer，但不能表达“该物理 slot 在 MTE3 完成前禁止分配给其他 buffer”。仅增加 barrier 或延长源码变量作用域，不能替代 pipe-aware lifetime 和 no-alias 约束。

### 6.5 高层 intrinsic 可能包含隐藏同步

`gemm_v0` 等高层 intrinsic 会插入内部 event。只要其 event ownership 没有正式契约，就不能和外部手动 `set_flag`/`wait_flag` 混合。

### 6.6 Frontend 和宏收集限制

- TileLang-Ascend 当前不支持 Python 三元表达式，必须使用普通静态分支或在宏外计算。
- `T.macro` 会收集并校验未实际发射分支中的 BufferRegion；静态 specialization 仍需 rank 正确的占位 buffer。
- `atomic_add` 对 region rank、local/global 方向和 extent 有严格限制，不能把任意非连续索引集合描述成一个 atomic region。

## 7. 已验证取舍与失败实验

当前保留：

- 显式 MMA、copy 时转置和手动 event；
- P/SY 的双 AIV 职责划分；
- P、SY 和 ReLU 的严格 Compare/Select 语义；
- full-valid/tail-protected 静态 specialization；
- V2 的 16 行批量读取和 `KScatterExec` 静态收缩；
- kernel 内完成 dWeight、dKey、dQuery 和 Loss 输出。

当前不采用：

- 将带隐藏同步的 `gemm_v0` 混入全手动 Cube 协议；
- 仅凭逻辑变量名实现 V1 backward 双 slot ping/pong；
- 把宽二维 parent buffer 的列 slice 当作连续小 tile；
- 用有限大负数的 exp 结果近似 softmax 无效列；
- 用指数链近似 ReLU 掩码；
- 让设备回传 `max(KRealSize)` 以换取更窄执行宽度。

## 8. 当前未决风险

当前泛化精度问题不是 P/SY 或 ReLU 数学公式的已知缺口。已有诊断曾显示错误 query 的 BMM1 有效区首先出现异常大值，并随后污染 PReduced 和 dWeight；另一些运行未出现非有限值，但 dQ/dK/dW 同时发生较大 norm-relative 误差。

现有证据把调查范围缩小到 V0 Gather 写 GM workspace 与 C1 读取之间的时序、task generation 或索引映射，而不是单纯的尾块 mask：

- 串行 task pipeline 或 `SLIG_C1_SERIAL` 只改变错误位置或让个别 query 恢复，不能证明 C1 数学计算有误；
- 错误通常覆盖某个 `(batch, query)` 的完整 head 组，更像 task 级生产/消费问题；
- `SLIG_C1_SERIAL` 是诊断开关，不是正式实现，也不能用于性能结论。

下一步诊断应保持数学路径不变，依次使用 V0 单槽串行、单 task gate 和 workspace 首错定位，确认是 event generation、物理 slot 提前复用，还是 task 到输入区域的映射错误。一次只改变一个同步变量。

## 9. 后续修改原则

1. 泛化精度优先于性能；任何优化必须先通过重复运行和多 task 用例。
2. 同步修改必须同时审计 RAW、WAR、首轮初始化、末轮 drain 和 event ID generation。
3. 本地 buffer 复用必须在生成 C++ 中核对物理 offset、extent 和最后消费者。
4. 尾块修改必须分别验证完整块、部分块和完整无效块，并覆盖 DMA、Vector、MMA、reduction 和 scatter。
5. 性能只使用关闭诊断开关后的多次设备中位数和 timeline；JIT、输入准备和官方算子异常不得计入 kernel 对比。
