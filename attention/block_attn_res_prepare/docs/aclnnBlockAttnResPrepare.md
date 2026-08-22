# aclnnBlockAttnResPrepare

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_attn_res_prepare)

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持。

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持

<!-- end id3 -->

<!-- npu="310b" id4 -->

- <term>Atlas 200I/500 A2 推理产品</term>：不支持

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas 推理系列产品</term>：不支持

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

* 接口功能：Attention Residuals的历史残差注意力两段式计算的阶段一：一次性并行完成全部 S 层的块间注意力计算；同时返回输出结果与 softmax 统计量(O, M, L)，供阶段二融合当前残差并形成下一子层输入。
* 计算公式：



$$
\operatorname{block\_attn\_res\_prepare}
\left(
V,Q,\operatorname{valid\_blocks};\epsilon
\right)
\longrightarrow
(O,M, L).
$$

其中：

$$
V_{t,n,:}\in\mathbb R^D
$$

表示 token \(t\) 的第 \(n\) 个有效历史来源。

RMS 归一化因子，每个 token、每个历史来源独立计算：

$$
R_{t,n}
=
\sqrt{
\frac1D
\sum_{d=0}^{D-1}V_{t,n,d}^2
+\epsilon
},
$$

$$
I_{t,n}=R_{t,n}^{-1}.
$$

历史来源 logits， 对每个 slot、token 和历史来源：

$$
Z_{s,t,n}
=
I_{t,n}
\sum_{d=0}^{D-1}
Q_{s,d}V_{t,n,d}.
$$

历史 softmax 统计量，对当前有效历史 source 范围 (N_v) 取最大值：

$$
m_{s,t}
=
\max_{0\le n<N_v}Z_{s,t,n}.
$$

计算 max-shift 后的指数：

$$
E_{s,t,n}
=
\exp\left(Z_{s,t,n}-m_{s,t}\right).
$$

历史指数和：

$$
L_{s,t}
=
\sum_{n=0}^{N_v-1}E_{s,t,n}.
$$

历史加权输出：

$$
O_{s,t,d}
=
\sum_{n=0}^{N_v-1}
E_{s,t,n}V_{t,n,d}.
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttnResPrepareGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBlockAttnResPrepare”接口执行计算。

```cpp
aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockRes, const aclTensor *validBlocks,
    const aclTensor *pseudoQuery, aclTensor *numerator,
    aclTensor *logitMax, aclTensor *expSum, float eps,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBlockAttnResPrepare(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

接口采用两段式调用：先获取 workspace 大小和执行器，再下发执行。

## 参数说明


统一符号：

| 符号                                                                                                                                                                                                                                                                                                                                          | 含义                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $T$                                                   | 扁平 token 数；原始 padded 布局下可理解为$\left(B\times T_{\mathrm{seq}}\right)$。$B$：$\mathrm{batch\_size}$，批次大小，一次送入模型多少条样本；$T_{\mathrm{seq}}$：$\mathrm{sequence\ length}$，序列长度，单条样本 token 数目                                           |                                                |
| $N$                                                                                                                                                                                                                                                                                                                                         | 历史 block/source 缓冲区的固定容量             |
| $N_v$                                                                                                                                                                                                                                                                                                                                       | 当前有效历史 block/source 数                   |
| $D$                                                                                                                                                                                                                                                                                                                                         | hidden dimension，隐藏层维度                   |
| $S$                                                                                                                                                                                                                                                                                                                                         | 当前$\mathrm{AttnRes}$ block 的目标 slot 数  |
| $t$                                                                                                                                                                                                                                                                                                             | token 下标，$0\le t<T$ |                                                |
| $n$     | 有效历史 source 下标，$0\le n<N_v\le N$                                                                                                                                                                                                                                                                                         |                                                |
| $d$     | hidden channel 下标，$0\le d<D$                                                                                                                                                                                                                                                                                                 |                                                |
| $s$     | slot 下标，$0\le s<S$                                                                                                                                                                                                                                                                                                           |                                                |

| 参数              | 输入/输出 | 类型               | 格式 | Shape       | 说明                         |
| ----------------- | --------- | ------------------ | ---- | ----------- | ---------------------------- |
| `blockRes`      | 输入      | FLOAT              | ND   | `[T,N,D]` | 残差块。                     |
| `validBlocks`   | 输入      | INT64              | ND   | `[1]`     | 有效块数，裁剪至`[0,N]`。  |
| `pseudoQuery`   | 输入      | FLOAT              | ND   | `[S,D]`   | 伪查询。                     |
| `numerator`     | 输出      | FLOAT              | ND   | `[S,T,D]` | softmax 加权分子。           |
| `logitMax`      | 输出      | FLOAT              | ND   | `[S,T]`   | softmax 最大值。             |
| `expSum`        | 输出      | FLOAT              | ND   | `[S,T]`   | softmax 指数和。             |
| `eps`           | 输入      | FLOAT              | -    | -           | RMS 稳定项，必须为有限正数。 |
| `workspaceSize` | 输出      | UINT64             | -    | -           | Device workspace 字节数。    |
| `executor`      | 输出      | `aclOpExecutor*` | -    | -           | 第二段接口的执行器。         |
| `workspace`     | 输入      | `void*`          | -    | -           | Device workspace。           |
| `stream`        | 输入      | `aclrtStream`    | -    | -           | 执行流。                     |

## 约束说明

- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- `blockRes` 与 `pseudoQuery` 的最后一维必须相同。
- 输入输出支持非连续 Tensor，接口内部执行必要的连续化和 ViewCopy。
- `T == 0` 或 `S == 0` 时返回对应形状的空输出。
- 仅支持 Ascend 950。

## 返回值

| 返回值                      | 说明                                   |
| --------------------------- | -------------------------------------- |
| `ACLNN_SUCCESS`           | 成功。                                 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必选参数为空。                         |
| `ACLNN_ERR_PARAM_INVALID` | dtype、rank、shape 或 eps 不满足约束。 |
| `ACLNN_ERR_RUNTIME_ERROR` | 平台不支持或执行失败。                 |

## 调用示例

完整示例见 [`examples/test_aclnn_block_attn_res_prepare.cpp`](../examples/test_aclnn_block_attn_res_prepare.cpp)。

```bash
bash build.sh --run_example block_attn_res_prepare eager cust --vendor_name=custom --soc=ascend950
```
