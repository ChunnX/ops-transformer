# BlockAttnResPrepare

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：完成 Attention Residuals 历史残差注意力两阶段计算的第一阶段，一次并行计算全部 $S$ 个目标 slot 的块间注意力，并返回 softmax 加权分子及统计量 $(O,M,L)$，供下一阶段融合当前残差。

- 计算公式：

$$
\operatorname{block\_attn\_res\_prepare}
\left(V,Q,\operatorname{valid\_blocks};\epsilon\right)
\longrightarrow (O,M,L)
$$

对 token $t$ 的第 $n$ 个有效历史来源计算 RMS 归一化因子：

$$
R_{t,n}=\sqrt{\frac{1}{D}\sum_{d=0}^{D-1}V_{t,n,d}^{2}+\epsilon}
$$

对每个目标 slot、token 和有效历史来源计算 logit：

$$
Z_{s,t,n}=R_{t,n}^{-1}\sum_{d=0}^{D-1}Q_{s,d}V_{t,n,d}
$$

设 $N_v$ 为 `valid_blocks` 指定的有效历史来源数，则：

$$
M_{s,t}=\max_{0\le n<N_v}Z_{s,t,n}
$$

$$
E_{s,t,n}=\exp\left(Z_{s,t,n}-M_{s,t}\right)
$$

$$
L_{s,t}=\sum_{n=0}^{N_v-1}E_{s,t,n}
$$

$$
O_{s,t,d}=\sum_{n=0}^{N_v-1}E_{s,t,n}V_{t,n,d}
$$

当 `valid_blocks[0] == 0` 时，不执行上述 softmax 计算，直接返回：

$$
O=\mathbf{0}_{S\times T\times D},\qquad
M=m_{\min}\mathbf{1}_{S\times T},\qquad
L=\mathbf{1}_{S\times T},\qquad
m_{\min}=-3.4028234663852886\times10^{38}
$$

即 `numerator` 为 shape `[S, T, D]` 的全 0 Tensor，`logit_max` 为 shape `[S, T]` 且所有元素均为 FLOAT32 最小有限值 $m_{\min}$ 的 Tensor，`exp_sum` 为 shape `[S, T]` 的全 1 Tensor。该结果表示 online softmax 的空状态；`valid_blocks[0] < 0` 时按 0 处理，返回结果相同。

其中：

- $V\in\mathbb{R}^{T\times N\times D}$ 表示历史残差块，$Q\in\mathbb{R}^{S\times D}$ 表示伪查询。
- $T$、$N$、$S$、$D$ 分别表示 token 数、历史来源缓冲区容量、目标 slot 数和 hidden size。
- $N_v$ 表示参与计算的有效历史来源数，取值由 `valid_blocks` 经区间 $[0,N]$ 截断后得到。
- $R$ 表示 RMS 归一化因子，$Z$ 表示归一化注意力 logit，$E$ 表示 max-shift 后的指数值。
- $O$、$M$、$L$ 分别表示 softmax 加权分子、logit 最大值和指数和，对应输出 `numerator`、`logit_max`、`exp_sum`。
- $\epsilon$ 表示 RMS 归一化的数值稳定项，对应属性 `eps`。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 490px">
  <col style="width: 120px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>block_res</td>
      <td>输入</td>
      <td>历史残差块，对应公式中的V，shape为[T, N, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valid_blocks</td>
      <td>输入</td>
      <td>当前有效历史来源数，shape为[1]。值为0时，numerator返回全0 Tensor、logit_max返回所有元素均为FLOAT32最小有限值的Tensor、exp_sum返回全1 Tensor；值小于0时按0处理，大于N时按N处理。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pseudo_query</td>
      <td>输入</td>
      <td>伪查询，对应公式中的Q，shape为[S, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>输出</td>
      <td>softmax加权分子，对应公式中的O，shape为[S, T, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logit_max</td>
      <td>输出</td>
      <td>有效历史来源logit的最大值，对应公式中的M，shape为[S, T]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_sum</td>
      <td>输出</td>
      <td>max-shift后的指数和，对应公式中的L，shape为[S, T]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>RMS归一化的数值稳定项，可选，默认值为1e-6，必须为有限正数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 仅支持 Ascend 950PR/Ascend 950DT。
- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- `block_res.shape[2]` 必须等于 `pseudo_query.shape[1]`。
- `valid_blocks` 的 shape 必须为 `[1]`；`valid_blocks[0] == 0` 时，`numerator` 返回全 0 Tensor、`logit_max` 返回所有元素均为 FLOAT32 最小有限值的 Tensor、`exp_sum` 返回全 1 Tensor；`valid_blocks[0] < 0` 时返回结果相同。
- `eps` 必须为有限正数。
- `T == 0` 或 `S == 0` 时，返回对应 shape 的空输出。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_block_attn_res_prepare](./examples/test_aclnn_block_attn_res_prepare.cpp) | 快速调用示例如下，仅供参考，详细算子编译运行方法参见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)。

```bash
# 在ops-transformer仓库根目录执行
bash build.sh --pkg --soc=ascend950 --ops=block_attn_res_prepare
bash build.sh --run_example block_attn_res_prepare eager cust --soc=ascend950 --vendor_name=custom
```
