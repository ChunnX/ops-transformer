# sparse_lightning_grad_kl_loss 0715 性能优化方案

本文档整理当前 `0715.py` 的性能分析、近期实验结论，以及基于 A3/CANN 9.0 `arch22` 官方 AscendC 实现制定的后续优化方案。

## 1. 当前结论

当前版本精度稳定，但性能距离官方仍有明显差距。

近期测试中，TileLang 中位数约为 `0.45~0.51 ms`，官方算子约为 `0.27~0.30 ms`，稳定加速比约为 `0.58~0.62x`，偶发低至约 `0.52x`。达到官方 80% 需要将 TileLang 延迟降低到约 `0.35~0.37 ms`，还需要约 25% 到 30% 的收益。

当前主要问题不是单个 barrier 或单次搬运，而是任务划分和 AIV 工作分工与官方 `arch22` 存在结构性差异：

1. 当前按 topK block launch，官方按 `(B,S1)` 任务 launch。
2. 当前两个 AIV 都重复执行完整的 V1 PReduced、ReduceSumY 和 Softmax，官方两个 AIV 分别承担 P 和 SY。
3. 当前 dKey scatter 逐行经过 GM，官方采用批量 GM 到 UB，再按两行进行 scatter。
4. 当前 dWeight、Loss 和 dKey Cast 仍有 kernel 外的 Tensor 操作，官方在 kernel 内完成。

单纯继续增加 barrier、调 dQuery 的 4/8 行写回粒度，预计无法补齐剩余性能差距。

## 2. 架构参考的修正

之前曾按照 `arch35` 分析官方实现，提出过 `L0C -> 双 AIV UB` 和直接消除 dKey GM 中转等方案。结合 A3 实际使用的官方代码后，该参考应修正为 `arch22`。

算子入口中只有 `__CCE_AICORE__ == 310` 才走 `arch35`；A3 当前应以 `arch22` 实现为准。主要参考文件如下：

- [0715.py](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/0715.py)
- [0715_kernel_source.cpp](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/0715_kernel_source.cpp)
- [arch22 kernel base](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_base.h)
- [arch22 vector service](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h)
- [arch22 vector2 service](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector2.h)
- [arch22 cube service](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_service_cube.h)

`arch22` 的关键特点不是单纯减少 kernel 数量，而是：

- launch 数量为 `min(B*S1, AIC 数量)`；
- 一个逻辑任务完整处理 `K=2048`；
- Gather 外层按 `s2BaseSize=1024` 分段；
- Cube 内部主要按 `128` 的 M/N/K tile 计算；
- 两个 AIV 分别执行 P 和 SY；
- P 与 SY 结果通过 `psySyncGm` 交换；
- V0、C1/V1/C2、V2 组成跨任务三级流水；
- dKey 仍保留 GM workspace 和最终批量 Cast。

## 3. 当前 0715 的具体问题

### 3.1 launch 维度错误

当前代码在 [0715.py:185](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/0715.py:185) 计算：

```python
block_num = B * s1_tile_num * n_num
```

当前测试中 `n_num=32`，生成代码为 `main_kernel<<<32>>>`。而 A3 物理 AIC 数量约为 24，因此会出现两波逻辑任务。

当前 Case 的 `B*S1` 很小，官方只需要一个逻辑任务，而 TileLang 却将同一个 `(B,S1)` 拆成 32 个 topK 任务。这会带来第二波核执行、重复初始化和同步，以及更明显的调度长尾。

### 3.2 V1 两个 AIV 重复完整计算

当前 V1 在 [0715.py:427](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/0715.py:427) 合并了所有 Vector 操作，但 `vid` 分工只在 dWeight 循环附近生效。

因此在此之前，两个 AIV 都会执行完整 PReduced、32 个 ReduceSumY block、全局 Softmax max/sum 和当前 block 的 SoftmaxI。

官方 `arch22` 在 [vector.h:651](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h:651) 中根据 `calcP` 让一个 AIV 执行 `VectorP`，另一个 AIV 执行 `VectorSy`，随后交换两份长度为 K 的结果。

### 3.3 ReduceSumY 双缓冲没有形成有效重叠

当前代码在 [0715.py:518](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/0715.py:518) 中先完整执行当前 block 的 Vector 计算，再发起下一 block 的 MTE2 copy，并在下一轮立即 wait。

这只是 buffer 复用保护，不是有效的 MTE2 与 Vector overlap。真正的 overlap 应该让下一 block 的 copy 早于当前 block 的 reduce/统计计算发出，中间不能立即等待。

### 3.4 V1 指令和 UB 压力过大

当前生成 C++ 中存在：

- `Muls(..., 1.0)` 作为归约前的形状中转；
- 全局 Softmax 第二遍中的 `Muls(..., 1.0)`；
- Loss 归约前的 `Muls(..., 1.0)`；
- dWeight 路径中 `weight_2d` 到 `rg_chunk` 的 32 条 UB-to-UB row copy；
- 当前 ReLU mask 使用 `Muls(-10000) + Exp + Sub` 近似生成。

生成代码的 UB 分配约为 `196352 B`，接近约 `192 KiB` 上限，几乎没有空间直接增加 pipeline stage。必须先消除中间张量和冗余指令。

### 3.5 dKey 粒度过小

当前 dKey 路径在 [0715.py:789](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/0715.py:789) 将 MM 结果写入 `dk_matmul_partial`，之后在 V2 中按行 GM-to-UB，再 atomic add 到 dKeyIndex。

`arch22` 也保留 GM workspace，但在 [vector2.h:197](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector2.h:197) 先批量搬入 UB，再按 `topKSplitSize=2` 进行 scatter，并使用较大的 ping-pong buffer。当前每行一次搬运会导致 V2 后段 MTE3 和 wait 数量过多。

### 3.6 kernel 外仍有额外归约和 Cast

当前 wrapper 在 [0715.py:1234](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/0715.py:1234) 执行：

```python
d_weight_partial.sum(dim=2)
loss_out.sum()
d_key_index_out.to(input_dtype)
```

这些操作会产生额外 NPU kernel。官方 `arch22` 在主 kernel 尾部同步后完成最终 dKey Cast，并在 Vector 阶段完成 dWeight 和 Loss 的累积。

## 4. 已验证的实验结论

有效改动：

- 去掉 V1 `ReduceSumY` 的冗余 GM 写回；
- 恢复 PReduced/SoftmaxI 的必要 GM 交互后精度稳定；
- 保留 Broadcast 后必要的 V barrier；
- dWeight/reluGrad 按两个 AIV 各处理 32 行。

无收益或失败的改动：

- 独立使用 `block_n=128`，曾出现 repeat 参数溢出、NaN 和严重精度错误；
- 将全局 Softmax 改成一次 2048 元素二维归约，性能下降；
- dQueryIndex 每次 8 行批量写回，没有明确收益；
- 直接把 PReduced/SoftmaxI 改成 UB 直用，出现同步和精度风险；
- 仅增加手动 barrier，通常不能减少关键路径，过多时会使性能恶化。

`block_n=128` 的问题本质上与当前完整 N1 维度下生成的 repeat 参数超过 `uint8_t` 范围有关。后续只有在 AIV 已按 N1/Nidx 维度分工、并确认生成的 repeat 参数不超过硬件范围后，才重新尝试 128 tile。

## 5. 修正后的优化方案

### Phase 0：固定基线和测量方法

保留当前精度正确的 0715 版本，不修改算法语义。每次只做一个独立 A/B，并同时记录：

- `msprof op` 的 kernel 时间；
- V1 Vector、Scalar、MTE2、MTE3 时间；
- V2 GM-to-UB 和 UB-to-GM 时间；
- kernel 外 dWeight、Loss、dKey Cast 时间；
- 20 次以上的 p50、p90 和最小值。

Python 计时只用于观察整体趋势，不能替代 `msprof`。

### Phase 1：V1 低风险优化

第一项建议实现官方一致的 ReLU mask：

```text
Compare(bmm2Res > 0) -> Select(subRes, mask, subRes, 0)
```

删除当前的：

```text
Muls(-10000) -> Exp -> 1 - Exp
```

这项改动同时减少 Vector 指令和 UB 占用，并与 `arch22` 的 [ReLUGrad](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h:1163) 保持一致。

第二项再处理 V1 的冗余物化：

- 让归约输入直接使用二维 UB，删除 `Muls(..., 1.0)`；
- 删除 dWeight 中 32 条 `weight_2d -> rg_chunk` row copy；
- 保留当前 PReduced/SoftmaxI 的 GM 写回读回，避免重新引入之前的同步问题。

这阶段预期收益有限，目标是减少 V1 的 Vector/Scalar 气泡并释放 UB，为后续双缓冲提供空间。

### Phase 2：按 arch22 方式重构任务所有权

将任务从 topK block 改为 `(B,S1)`：

```text
launch_core_num = min(B * S1, physical_aic_num)
每个物理核在核内循环处理多个 (B,S1) 任务
每个任务完整处理 K=2048
```

workspace 改为按物理核和 `taskId % 2` 分配双缓冲，参考 [arch22 base workspace](F:/WorkSpace/Code/ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_base.h:269)。

当前 Case 中 `B*S1` 很小，应该优先验证实际使用一个逻辑任务的情况；对于更大批量，再让多个物理核循环分摊任务。不要把 24 个空闲核强行加入当前单任务 Case。

### Phase 3：复现 arch22 的 AIV 分工和三级流水

单个 `(B,S1)` 任务内部采用：

```text
V0: Gather Key/KeyRope/KeyIndex
C1: MM1 + MM2
V1: P 或 SY，以及 dWeight/reluGrad/Loss
C2: MM5 + MM6
V2: dKey scatter
```

跨任务按三阶段重叠：

```text
当前任务:     V0
前一任务:          C1 -> V1 -> C2
前两任务:                    V2
```

两个 AIV 的职责应参考 `calcP`：

- AIV0/AIV1 分别计算 P 和 SY；
- P 与 SY 各写一份 K 长度结果到 `psySyncGm`；
- 通过 cross flag 交换另一份结果；
- 后续 dWeight/reluGrad 按两个 AIV 各处理一半 Nidx。

这一步是当前性能提升的主要来源，因为它消除了两个 AIV 对完整 V1 的重复计算。

实现上优先手写 `taskId % 2` buffer 和 cross flag，先不要把整个大循环直接套入 `T.Pipelined`。FlashAttention 示例已经说明，自动 pipeline 可能为所有 UB scratch 产生额外版本，导致 UB 超限；当前 0715 的 UB 使用率更高，必须先检查生成 C++。

### Phase 4：对齐 arch22 的梯度输出路径

保留 arch22 的 dKey GM workspace 设计，但将当前逐行路径改为：

- 每次批量搬入 32 或 64 行到 UB；
- UB 内按两行处理 topK scatter；
- 使用 ping-pong buffer 形成 MTE2 与 MTE3 重叠；
- 最终统一批量 Cast 到 dKeyIndex。

dQueryIndex 则在 MM6 中跨完整 K 累加，最后一次 Fixpipe FP32->FP16，删除当前 `dQueryIndexAccum -> GM -> UB -> Cast` 的 V2 路径。

dWeight 和 Loss 应在 kernel 内跨 K 累积，最后各写一次，从而删除 wrapper 中的 `.sum(dim=2)` 和 `.sum()`。

## 6. 每阶段验收条件

精度必须继续满足当前基线：

- dQueryIndex、dKeyIndex、dWeights 不出现 NaN/Inf；
- 相对误差不明显超过当前版本；
- Loss 保持当前误差水平。

生成 C++ 需要重点检查：

- launch 不再是 `<<<32>>>` 的 topK block 模式；
- 两个 AIV 不再重复完整 V1；
- `Muls(..., 1.0)` 和 ReLU Exp 链消失；
- dKey GM-to-UB 不再每行发起一次；
- UB 总分配没有因 pipeline 版本化而超过上限；
- kernel 外不再生成 dWeight/Loss/dKey 的额外归约或 Cast kernel。

## 7. 推荐的下一步

下一次只实现 Phase 1 的第一项：将当前 ReLU mask 改成 `Compare + Select`，不修改 launch、workspace、P/SY 数据流和 dKey 路径。完成精度和 C++ 指令核对后，再进行 dKey 32/64 行批量搬运 A/B。

达到 0.8x 的希望主要来自 Phase 2 到 Phase 4 的结构重构；Phase 1 只用于先降低 V1 成本和释放 UB，不应期待单独达到目标。
