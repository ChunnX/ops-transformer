# aclnnRecurrentKda

[📄 查看源码](https://gitcode.com/ziqiyang/flash-linear-attention-npu/tree/master/fla/ops/ascendc/kda/recurrent_kda)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 接口功能：aclnnRecurrentKda 是基于 Gated Delta Rule 的线性注意力前向计算接口的递归（recurrent）实现，面向 decode / MTP（Multi-Token Prediction）短序列场景。该接口按 token 逐步推进递归状态 $S_t$，支持变长输入（配合 `cuSeqlensOptional`）、MTP 槽位复用（配合 `ssmStateIndicesOptional` 与 `numAcceptedTokensOptional`），并可选在 kernel 内将 `gate` 解释为 raw gate、对 q/k 做 L2 normalize、对 beta 做 sigmoid 等数值稳定处理。
- 计算公式：

$$
\begin{aligned}
g_t &= \begin{cases}
\exp(\text{gate}_t) & \text{useGateInKernel} = \text{false} \\
\exp(\text{softplus}(A_{\log}) \cdot \text{gate}_t + \text{dt\_bias}_t) & \text{useGateInKernel} = \text{true}
\end{cases} \\
\beta_t &= \begin{cases}
\beta_t & \text{useBetaSigmoidInKernel} = \text{false} \\
\sigma(\beta_t) \cdot (2 \text{ if } \text{allowNegEigval} \text{ else } 1) & \text{useBetaSigmoidInKernel} = \text{true}
\end{cases} \\
\tilde{K}_t &= K_t \odot g_t \\
\tilde{V}_t &= V_t - \tilde{K}_t \odot (\beta_t \cdot S_{t-1}) \\
S_t &= S_{t-1} + \tilde{K}_t^{\top} \tilde{V}_t \\
O_t &= Q_t \cdot S_{t-1} \cdot \text{scale}
\end{aligned}
$$

其中 $S_t$ 为递归状态（按 `stateVFirst=True` 布局存储为 `[seq_num, H_v, V_dim, K_dim]`，`stateVFirst=False` 布局存储为 `[seq_num, H_v, K_dim, V_dim]`），$\text{scale}$ 为缩放系数（通常为 $K_{\text{dim}}^{-0.5}$）。`safeGate=True` 时对 raw gate 分支施加下界 `lowerBound` 以保证数值稳定。

- 符号说明
  | 符号        | 含义                                                                               |
  | ----------- | ---------------------------------------------------------------------------------- |
  | B           | Batch Size，输入样本批量大小                                                       |
  | T           | Sequence Length，输入样本序列长度（dense 为单条序列长度，varlen 为所有序列总长度） |
  | H_k         | Query/Key 头数                                                                     |
  | H_v         | Value/Gate 头数（要求 H_v 能被 H_k 整除）                                          |
  | K_dim       | Key 每一个头的维度                                                                 |
  | V_dim       | Value 每一个头的维度                                                               |
  | seq_num     | 逻辑序列数（定长时等于 B，变长时等于 `cuSeqlensOptional` 长度减 1）                |
  | scale       | 缩放系数，通常为 K_dim**-0.5                                                       |
  | gate        | gate 张量。`useGateInKernel=true` 时为 raw gate 输入，由算子内部按公式转换为递归 gate $g_t$；`useGateInKernel=false` 时为外部已预计算好的 step log gate |
  | $g_t$       | 递归 gate（exp 后），用于对 key 做 $\tilde{K}_t = K_t \odot g_t$ 衰减               |
  | A_log       | raw gate 分支的对角项（softplus 后参与 gate 计算）                                |
  | dtBias      | raw gate 偏置                                                                      |
  | beta        | Delta 更新系数                                                                     |
  | $S_t$       | 递归状态矩阵，按 `stateVFirst=True` 布局存储为 `[seq_num, H_v, V_dim, K_dim]`，`stateVFirst=False` 布局存储为 `[seq_num, H_v, K_dim, V_dim]`     |
  | $\tilde{K}_t$ | 衰减后的 key，$\tilde{K}_t = K_t \odot g_t$                                      |
  | $\tilde{V}_t$ | 更新后的 value，$\tilde{V}_t = V_t - \tilde{K}_t \odot (\beta_t \cdot S_{t-1})$  |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRecurrentKdaGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRecurrentKda”接口执行计算。

```Cpp
aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor     *query,
    const aclTensor     *key,
    const aclTensor     *value,
    const aclTensor     *gate,
    const aclTensor     *beta,
    const aclTensor     *initialStateRef,
    const aclIntArray   *cuSeqlensOptional,
    const aclTensor     *ssmStateIndicesOptional,
    const aclTensor     *aLogOptional,
    const aclTensor     *dtBiasOptional,
    const aclTensor     *numAcceptedTokensOptional,
    const char          *layout,
    double               scale,
    bool                 outputFinalState,
    bool                 inplaceFinalState,
    bool                 useQkL2normInKernel,
    bool                 useGateInKernel,
    bool                 useBetaSigmoidInKernel,
    bool                 allowNegEigval,
    bool                 safeGate,
    double               lowerBound,
    bool                 stateVFirst,
    const aclTensor     *attnOut,
    const aclTensor     *finalState,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnRecurrentKda(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnRecurrentKdaGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 500px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 400px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续的Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query（aclTensor）</td>
      <td>输入</td>
      <td>Query 张量。</td>
      <td>按 layout 为 BSND: [B,T,H_k,K_dim] 或 TND: [T,H_k,K_dim]。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_k,K_dim] 或 [T,H_k,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key（aclTensor）</td>
      <td>输入</td>
      <td>Key 张量。</td>
      <td>与 query 形状相同，数据类型与 query 相同。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_k,K_dim] 或 [T,H_k,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>value（aclTensor）</td>
      <td>输入</td>
      <td>Value 张量。</td>
      <td>按 layout 为 BSND: [B,T,H_v,V_dim] 或 TND: [T,H_v,V_dim]。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v,V_dim] 或 [T,H_v,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gate（aclTensor）</td>
      <td>输入</td>
      <td>预计算 step log gate 或 raw gate。</td>
      <td>按 layout 为 BSND: [B,T,H_v,K_dim] 或 TND: [T,H_v,K_dim]。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v,K_dim] 或 [T,H_v,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（aclTensor）</td>
      <td>输入</td>
      <td>Delta 更新系数。</td>
      <td>按 layout 为 BSND: [B,T,H_v] 或 TND: [T,H_v]。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v] 或 [T,H_v]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>initialStateRef（aclTensor）</td>
      <td>输入/输出</td>
      <td>初始状态。</td>
      <td>stateVFirst=true 时 Shape: [seq_num,H_v,V_dim,K_dim]；stateVFirst=false 时 Shape: [seq_num,H_v,K_dim,V_dim]。</td>
      <td>BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>[seq_num,H_v,V_dim,K_dim] 或 [seq_num,H_v,K_dim,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOptional（aclIntArray）</td>
      <td>输入</td>
      <td>变长序列累计长度。</td>
      <td>Shape: [N+1]，aclIntArray。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[N+1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ssmStateIndicesOptional（aclTensor）</td>
      <td>输入</td>
      <td>MTP decode 状态槽索引。</td>
      <td>可选；Shape: [&gt;=T]，必须与 numAcceptedTokensOptional 一起传。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>[&gt;=T]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>aLogOptional（aclTensor）</td>
      <td>输入</td>
      <td>A_log 参数。</td>
      <td>useGateInKernel=True 时必选；Shape: [H_v]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[H_v]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dtBiasOptional（aclTensor）</td>
      <td>输入</td>
      <td>raw gate 偏置。</td>
      <td>Shape: [H_v*K_dim] 或 [H_v,K_dim]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[H_v*K_dim] 或 [H_v,K_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numAcceptedTokensOptional（aclTensor）</td>
      <td>输入</td>
      <td>已接受的 token 数量。</td>
      <td>必须与 ssmStateIndicesOptional 一起传；Shape: [seq_num]。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>[seq_num]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>layout（char）</td>
      <td>输入</td>
      <td>query、key、value 的数据 Layout 格式。</td>
      <td>枚举值: BSND | TND。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale（double）</td>
      <td>输入</td>
      <td>缩放系数。</td>
      <td>乘到 query 上，通常为 K_dim**-0.5。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputFinalState（bool）</td>
      <td>输入</td>
      <td>是否输出有效最终状态。</td>
      <td>为 true 时 非原地更新</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inplaceFinalState（bool）</td>
      <td>输入</td>
      <td>状态是否原地更新。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>useQkL2normInKernel（bool）</td>
      <td>输入</td>
      <td>是否对 q/k 做 L2 normalize。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>useGateInKernel（bool）</td>
      <td>输入</td>
      <td>是否将 gate 解释为 raw gate。</td>
      <td>为 true 时 aLogOptional 必选。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>useBetaSigmoidInKernel（bool）</td>
      <td>输入</td>
      <td>是否在 kernel 内计算 sigmoid(beta)。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>allowNegEigval（bool）</td>
      <td>输入</td>
      <td>beta sigmoid 后是否乘 2。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>safeGate（bool）</td>
      <td>输入</td>
      <td>raw gate 的 safe 分支开关。</td>
      <td>与 useGateInKernel 配合使用。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lowerBound（double）</td>
      <td>输入</td>
      <td>safe gate 下界。</td>
      <td>safeGate=True 时生效，取值范围 [-5,0)。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stateVFirst（bool）</td>
      <td>输入</td>
      <td>状态布局标志。</td>
      <td>output_final_state=True 时生效，支持 true（[seq_num,H_v,V_dim,K_dim]）/ false（[seq_num,H_v,K_dim,V_dim]），默认为 false。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOut（aclTensor）</td>
      <td>输出</td>
      <td>Recurrent KDA 输出。</td>
      <td>Shape 与 value 相同。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[B,T,H_v,V_dim] 或 [T,H_v,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>finalState（aclTensor）</td>
      <td>输出</td>
      <td>最终状态。</td>
      <td>stateVFirst=true 时 Shape: [seq_num,H_v,V_dim,K_dim]；stateVFirst=false 时 Shape: [seq_num,H_v,K_dim,V_dim]，数据类型与 initialState 一致。</td>
      <td>BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>[seq_num,H_v,V_dim,K_dim] 或 [seq_num,H_v,K_dim,V_dim]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td rowspan="20">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="20">161002</td>
            </tr>
            <tr><td>query/key/value dtype 不一致且不属于 {BFLOAT16}</td></tr>
            <tr><td>gate/beta dtype 不在 {FLOAT, BFLOAT16, FLOAT16} 范围内</td></tr>
            <tr><td>initialState/finalState dtype 不在 {BFLOAT16, FLOAT} 范围内或两者不一致</td></tr>
            <tr><td>ssmStateIndicesOptional/numAcceptedTokensOptional dtype 不在 {INT32, INT64} 范围内</td></tr>
            <tr><td>aLogOptional/dtBiasOptional dtype 不为 FLOAT</td></tr>
            <tr><td>layout 不在 {BSND, TND} 集合内或与 rank 不匹配</td></tr>
            <tr><td>K_dim 不等于 128；V_dim 不属于 {128, 256}</td></tr>
            <tr><td>H_v 不能被 H_k 整除</td></tr>
            <tr><td>dense 序列长度或 varlen 单序列长度大于 8</td></tr>
            <tr><td>BSND varlen 下 B≠1</td></tr>
            <tr><td>initialState/finalState shape 与 stateVFirst 不匹配（stateVFirst=true 时为 [seq_num,H_v,V_dim,K_dim]，stateVFirst=false 时为 [seq_num,H_v,K_dim,V_dim]）</td></tr>
            <tr><td>ssmStateIndicesOptional 长度小于 T</td></tr>
            <tr><td>numAcceptedTokensOptional 长度不等于 seq_num</td></tr>
            <tr><td>useGateInKernel=true 时 aLogOptional 为空</td></tr>
            <tr><td>useGateInKernel=false 时传入了 aLogOptional/dtBiasOptional 或 safeGate=true</td></tr>
            <tr><td>aLogOptional shape 不为 [H_v]</td></tr>
            <tr><td>dtBiasOptional shape 不为 [H_v*K_dim] 或 [H_v,K_dim]</td></tr>
            <tr><td>safeGate=true 时 lowerBound 不在 [-5,0) 范围内</td></tr>
            <tr><td>numAcceptedTokensOptional 未与 ssmStateIndicesOptional 同时提供</td></tr>
        </tbody>
    </table>

## aclnnRecurrentKda

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRecurrentKdaGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：aclnnRecurrentKda 默认确定性实现。
- 该接口面向 decode / MTP 短序列场景，dense 单序列长度与 varlen 单序列长度均必须小于等于 8。
- query、key、value 数据类型必须保持一致，且仅支持 BFLOAT16；out 与 value 同 dtype。
- gate、beta 支持 FLOAT、BFLOAT16、FLOAT16（接口内部会转换为 FLOAT 参与计算）。
- initialState 与 finalState 数据类型必须一致，支持 BFLOAT16 或 FLOAT；out 与 value 同 dtype。
- ssmStateIndicesOptional、numAcceptedTokensOptional 支持 INT32 或 INT64（接口内部会转换为 INT64）。
- aLogOptional、dtBiasOptional 必须为 FLOAT。
- layout 取值仅支持 BSND、TND（必须大写）；BSND varlen（cuSeqlensOptional 非 None）场景下要求 B = 1。
- K_dim 仅支持 128，V_dim 仅支持 128 或 256。
- H_v 必须能被 H_k 整除。
- stateVFirst 支持 true 和 false（默认 false）：为 true 时 initialState 与 finalState 的 shape 必须满足 `[seq_num, H_v, V_dim, K_dim]`；为 false 时必须满足 `[seq_num, H_v, K_dim, V_dim]`。其中 seq_num 为逻辑序列数（dense 时等于 B，varlen 时等于 cuSeqlensOptional 长度减 1）。
- cuSeqlensOptional 首元素必须为 0、末元素必须等于总 token 数 T，且非递减；每段变长序列长度必须小于等于 8。
- ssmStateIndicesOptional 必须为 1D tensor，长度大于等于总 token 数 T；numAcceptedTokensOptional 必须为 1D tensor，长度等于 seq_num；两者必须同时提供或同时为空，单独提供 numAcceptedTokensOptional 会报错。
- useGateInKernel=true 时 aLogOptional 必选；useGateInKernel=false 时传入了 aLogOptional、dtBiasOptional 或 safeGate=true 均会报错。
- aLogOptional shape 必须为 [H_v]；dtBiasOptional shape 必须为 [H_v*K_dim] 或 [H_v,K_dim]。
- safeGate=true 时 lowerBound 必须落在 [-5, 0) 区间内。
- 输入 tensor 支持非连续，接口内部会调用 Contiguous 处理；finalState 支持非连续输出，接口内部通过 ViewCopy 写回。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_recurrent_kda.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* gateDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* initialStateDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* finalStateDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* gateTensor = nullptr;
    aclTensor* betaTensor = nullptr;
    aclTensor* initialStateTensor = nullptr;
    aclTensor* outTensor = nullptr;
    aclTensor* finalStateTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    // BSND layout: q/k [B,T,H_k,K], v [B,T,H_v,V], gate [B,T,H_v,K], beta [B,T,H_v]
    // state [seq_num,H_v,V,K]
    int64_t B = 1;
    int64_t T = 1;
    int64_t Hk = 2;
    int64_t Hv = 2;
    int64_t K = 128;
    int64_t V = 128;

    std::vector<int64_t> queryShape = {B, T, Hk, K};
    std::vector<int64_t> keyShape = {B, T, Hk, K};
    std::vector<int64_t> valueShape = {B, T, Hv, V};
    std::vector<int64_t> gateShape = {B, T, Hv, K};
    std::vector<int64_t> betaShape = {B, T, Hv};
    std::vector<int64_t> initialStateShape = {B, Hv, V, K};
    std::vector<int64_t> outShape = {B, T, Hv, V};
    std::vector<int64_t> finalStateShape = {B, Hv, V, K};

    int64_t querySize = GetShapeSize(queryShape);
    int64_t keySize = GetShapeSize(keyShape);
    int64_t valueSize = GetShapeSize(valueShape);
    int64_t gateSize = GetShapeSize(gateShape);
    int64_t betaSize = GetShapeSize(betaShape);
    int64_t initialStateSize = GetShapeSize(initialStateShape);
    int64_t outSize = GetShapeSize(outShape);
    int64_t finalStateSize = GetShapeSize(finalStateShape);

    std::vector<float> queryHostData(querySize, 1);
    std::vector<float> keyHostData(keySize, 1);
    std::vector<float> valueHostData(valueSize, 1);
    std::vector<float> gateHostData(gateSize, 0);
    std::vector<float> betaHostData(betaSize, 0);
    std::vector<float> initialStateHostData(initialStateSize, 0);
    std::vector<float> outHostData(outSize, 0);
    std::vector<float> finalStateHostData(finalStateSize, 0);

    // Create query aclTensor.
    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                             aclDataType::ACL_BF16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                         aclDataType::ACL_BF16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr,
                         aclDataType::ACL_BF16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create gate aclTensor.
    ret = CreateAclTensor(gateHostData, gateShape, &resources.gateDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.gateTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create beta aclTensor.
    ret = CreateAclTensor(betaHostData, betaShape, &resources.betaDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.betaTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create initialState aclTensor.
    ret = CreateAclTensor(initialStateHostData, initialStateShape, &resources.initialStateDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.initialStateTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr,
                         aclDataType::ACL_BF16, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    // Create finalState aclTensor.
    ret = CreateAclTensor(finalStateHostData, finalStateShape, &resources.finalStateDeviceAddr,
                         aclDataType::ACL_FLOAT, &resources.finalStateTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteRecurrentKda(TensorResources& resources, aclrtStream stream,
                        void** workspaceAddr, uint64_t* workspaceSize) {
    constexpr const char layoutStr[] = "BSND";
    double scale = 1.0 / sqrt(128.0);
    bool outputFinalState = true;
    bool useQkL2normInKernel = false;
    bool useGateInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool stateVFirst = false;
    aclOpExecutor* executor;

    int ret = aclnnRecurrentKdaGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.valueTensor,
        resources.gateTensor, resources.betaTensor, resources.initialStateTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        layoutStr, scale, outputFinalState, useQkL2normInKernel, useGateInKernel,
        useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound, stateVFirst,
        resources.outTensor, resources.finalStateTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnRecurrentKdaGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnRecurrentKda(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnRecurrentKda failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclBfloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclBfloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.gateTensor) {
      aclDestroyTensor(resources.gateTensor);
    }
    if (resources.betaTensor) {
      aclDestroyTensor(resources.betaTensor);
    }
    if (resources.initialStateTensor) {
      aclDestroyTensor(resources.initialStateTensor);
    }
    if (resources.outTensor) {
      aclDestroyTensor(resources.outTensor);
    }
    if (resources.finalStateTensor) {
      aclDestroyTensor(resources.finalStateTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.gateDeviceAddr) {
      aclrtFree(resources.gateDeviceAddr);
    }
    if (resources.betaDeviceAddr) {
      aclrtFree(resources.betaDeviceAddr);
    }
    if (resources.initialStateDeviceAddr) {
      aclrtFree(resources.initialStateDeviceAddr);
    }
    if (resources.outDeviceAddr) {
      aclrtFree(resources.outDeviceAddr);
    }
    if (resources.finalStateDeviceAddr) {
      aclrtFree(resources.finalStateDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }

    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {1, 1, 2, 128};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteRecurrentKda(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    printf("-----------out输出-----------\n");
    PrintOutResult(outShape, &resources.outDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
