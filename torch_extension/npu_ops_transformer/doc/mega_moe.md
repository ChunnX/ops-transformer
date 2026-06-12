# mega\_moe

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>            |    √    |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

    Mega MoE 将 EP Dispatch、Linear1、SwiGLU、Linear2 和 EP Combine 融合为单个算子，实现计算和通信的重叠

-   API功能：

    需与[get\_symm\_buffer\_for\_mega\_moe](#get_symm_buffer_for_mega_moe接口说明)配套使用，该算子将 `MoE（Mixture of Experts）`以及 `FFN（Feed-Forward Network）`的完整计算流程融合为单个算子，实现 `Dispatch + Linear1 + SwiGLU + Linear2 + Combine` 的端到端融合计算。

-   基础计算公式如下（详细公式请参见计算公式）：
    ### 输入与维度约定

    - $\mathbf{x} \in \mathbb{R}^{\text{num\_tokens} \times \text{hidden}}$
    - $\text{topk\_idx} \in \mathbb{Z}^{\text{num\_tokens} \times \text{num\_topk}}$
    - $\text{topk\_weights} \in \mathbb{R}^{\text{num\_tokens} \times \text{num\_topk}}$
    - $\text{l1\_weights} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden} \times 2\cdot\text{intermediate\_hidden}}$
    - $\text{l2\_weights} \in \mathbb{R}^{\text{num\_experts} \times \text{intermediate\_hidden} \times \text{hidden}}$

    ---

    ### 对 token $i\;(0 \le i < \text{num\_tokens})$

    记它选择的专家编号与路由权重为  

    $$
    e_k = \text{topk\_idx}[i,k],\qquad
    w_k = \text{topk\_weights}[i,k],\qquad k = 1,\dots,\text{num\_topk}
    $$

    ---

    #### 🔹 Dispatch（分发）

    Token $i$ 的特征向量 $\mathbf{x}_i \in \mathbb{R}^{\text{hidden}}$ 被**复制**给 $\text{num\_topk}$ 个专家 $e_k$，每个专家关联一个权重 $w_k$。

    ---

    #### 🔹 专家 $e_k$ 的内部计算（针对该单一 token）

    **Linear1 投影（gate + up）**

    $$
    \mathbf{h}^{(k)} = \mathbf{x}_i \cdot \text{l1\_weights}[e_k]
    \quad\bigl(\in \mathbb{R}^{2\cdot\text{intermediate\_hidden}}\bigr)
    $$

    **拆分 gate 与 up**

    $$
    \mathbf{h}_{\text{gate}}^{(k)} = \mathbf{h}^{(k)}\bigl[:\text{intermediate\_hidden}\bigr],\qquad
    \mathbf{h}_{\text{up}}^{(k)} = \mathbf{h}^{(k)}\bigl[\text{intermediate\_hidden}:\bigr]
    $$

    **SwiGLU 激活**

    $$
    \mathbf{a}^{(k)} = \text{SiLU}\bigl(\mathbf{h}_{\text{gate}}^{(k)}\bigr) \;\odot\; \mathbf{h}_{\text{up}}^{(k)}
    \quad\bigl(\in \mathbb{R}^{\text{intermediate\_hidden}}\bigr)
    $$

    其中 $\text{SiLU}(z) = z \cdot \sigma(z)$，$\odot$ 为逐元素乘法。

    **Linear2 投影（down projection）**

    $$
    \mathbf{y}^{(k)} = \mathbf{a}^{(k)} \cdot \text{l2\_weights}[e_k]
    \quad\bigl(\in \mathbb{R}^{\text{hidden}}\bigr)
    $$

    ---

    #### 🔹 Combine（聚合）

    Token $i$ 将所有专家的输出按其门控权重加权求和：

    $$
    \mathbf{y}_i = \sum_{k=1}^{\text{num\_topk}} w_k \;\cdot\; \mathbf{y}^{(k)}
    \qquad\bigl(\in \mathbb{R}^{\text{hidden}}\bigr)
    $$

    所有 token 的结果堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{num\_tokens} \times \text{hidden}}$。

    ---

    ### 流程说明

    - **Dispatch**：同一个 $\mathbf{x}_i$ 被同时送入 $\text{num\_topk}$ 个专家，展开多条独立计算流。
    - **专家计算**：每个专家依次执行 Linear1（合并 gate/up）、SwiGLU 激活、Linear2（down projection）。
    - **Combine**：各专家的输出 $\mathbf{y}^{(k)}$ 通过门控权重 $w_k$ 加权累加，恢复为 token $i$ 的最终表示。
    $$y = (\text{SwiGLU}(x \cdot l1\_weights)) \cdot l2\_weights$$

    MegaMoe 是一个多阶段融合的 MoE 算子，在专家并行场景下依次完成 Token 路由分发与量化、分组矩阵乘法 + SwiGLU 激活、二次分组矩阵乘法与跨 Rank 聚合、以及加权求和，完整计算流程可以分解为以下四个阶段。
    
    
    第一阶段：对输入 Token 按专家分组收集，按量化模式对输入做量化（A16W16 不量化）：

    $$
    \hat{X}_e,\ S_{X,e} = \mathrm{Q}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}-1}
    $$

    第二阶段：对每个专家执行 GMM1 矩阵乘法、SwiGLU 激活和量化：

    $$
    Z_e^{(x)} = \mathrm{DQ}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}(W_{1,e}^{(x)}, S_{1,e}^{(x)}) + b_{1,e}^{(x)}, \quad Z_e^{(y)} = \mathrm{DQ}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}(W_{1,e}^{(y)}, S_{1,e}^{(y)}) + b_{1,e}^{(y)}
    $$

    $$
    U_e = Z_e^{(x)} \odot \sigma\!\left(Z_e^{(x)}\right) \odot Z_e^{(y)}
    $$

    $$
    \hat{U}_e,\ S_{U,e} = \mathrm{Q}(U_e)
    $$

    第三阶段：对每个专家执行 GMM2 矩阵乘法，并将结果按目标 Rank 分发：

    $$
    O_e = \mathrm{DQ}(\hat{U}_e, S_{U,e}) \cdot \mathrm{DQ}(W_{2,e}, S_{2,e}) + b_{2,e}
    $$

    第四阶段：对所有 Token 按路由权重加权求和：

    $$
    Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)]
    $$

    其中，$X$ 表示参数 `x`，$W$ 表示参数 `topk_weights`，$W_1$ 表示参数 `weight1`，$W_2$ 表示参数 `weight2`，$Y$ 表示参数 `y`，$E_{\text{local}}$ 表示属性 `moe_expert_num / ep_world_size`（每个 Rank 的专家数），$K$ 表示 `topk_ids` 的第二维度（top-K 值）。

    上述公式中 $\mathrm{Q}$、$\mathrm{DQ}$、$S$、$b$ 的含义随量化模式不同而不同，A16W16 模式下 $\mathrm{Q}$ 和 $\mathrm{DQ}$ 为恒等变换（不量化），$S$ 和 $b$ 不存在。各量化模式差异如下：

    #### A16W16 模式（非量化）

    权重和激活均不做量化，直接使用 BF16/FP16 进行计算。上述公式中：
    - $\mathrm{Q}(\cdot)$ 为恒等变换，$\hat{X}_e = X_e$，$\hat{U}_e = U_e$，$S_{X,e}$ 和 $S_{U,e}$ 不存在。
    - $\mathrm{DQ}(\cdot)$ 为恒等变换，权重和激活直接参与计算。
    - $b_{1,e}^{(x)} = 0$，$b_{1,e}^{(y)} = 0$，$b_{2,e} = 0$。

    #### A8W8 模式（激活 per-token INT8 量化 + 权重 per-channel INT8 量化）

    激活采用 per-token 动态量化（在算子内 Dispatch 通信前自动完成，将 BF16 激活量化为 INT8），权重采用 per-channel INT8 量化（缩放因子数据类型为 `float`）。上述公式中：
    - $\mathrm{Q}(\cdot) = \mathrm{Q}_{\text{per-token}}(\cdot)$：逐 token 动态量化，对每个 token 提取绝对值最大值后量化为 INT8，输出 INT8 数据和 float 缩放因子。
    - $\mathrm{DQ}(\hat{X}_e, S_{X,e}) = \mathrm{DQ}_{\text{per-token}}(\hat{X}_e, S_{X,e})$：per-token 反量化，在 matmul 内部隐式执行。
    - $\mathrm{DQ}(W, S) = \mathrm{DQ}_{\text{per-channel}}(W, S)$：per-channel 反量化，在 matmul 内部隐式执行。
    - $S_{X,e}$、$S_{U,e}$：float 类型的 per-token 缩放因子。
    - $S_{1,e}^{(x)}$、$S_{1,e}^{(y)}$、$S_{2,e}$：float 类型的 per-channel 权重缩放因子。
    - $b_{1,e}^{(x)} = 0$，$b_{1,e}^{(y)} = 0$，$b_{2,e} = 0$。

    #### A8W4 模式（激活 per-token INT8 量化 + 权重 per-channel INT4 量化）

    激活量化与 A8W8 相同，权重采用 per-channel INT4 量化（缩放因子数据类型为 `float`），INT4 量化精度较低需要 bias 补偿量化误差。上述公式中：
    - $\mathrm{Q}(\cdot)$、$\mathrm{DQ}(\cdot)$、$S$ 的含义与 A8W8 模式相同。
    - $b_{1,e}^{(x)}$、$b_{1,e}^{(y)}$：GMM1 的偏置（float），来自参数 `l1_bias`。
    - $b_{2,e}$：GMM2 的偏置（float），来自参数 `l2_bias`。

    #### MXFP 量化模式

    dispatch 阶段使用 MX 逐组量化（group size = 32），量化缩放因子类型为 FLOAT8_E8M0。上述公式中：
    - $\mathrm{Q}(\cdot) = \mathrm{Q}_{\text{MX}}(\cdot)$：MX 逐组量化（block size = 32），对每组 32 个元素提取共享指数后量化为 FP8 目标类型（FLOAT8_E5M2 或 FLOAT8_E4M3FN），输出 FP8 数据和 FLOAT8_E8M0 缩放因子。
    - $\mathrm{DQ}(\cdot) = \mathrm{DQ}_{\text{MX}}(\cdot)$：MX 逐组反量化，在 matmul 内部隐式执行。
    - $S_{X,e}$、$S_{U,e}$：FLOAT8_E8M0 类型的 MX 缩放因子。
    - $S_{1,e}^{(x)}$、$S_{1,e}^{(y)}$、$S_{2,e}$：FLOAT8_E8M0 类型的 MX 权重缩放因子。
    - $b_{1,e}^{(x)} = 0$，$b_{1,e}^{(y)} = 0$，$b_{2,e} = 0$。

    #### 局部变量说明

    - $\mathcal{T}_e$：被路由到专家 $e$ 的 Token 索引集合，由 `topk_ids` 排序后确定。
    - $\hat{X}_e$：专家 $e$ 的量化输入，第一阶段中间结果。A16W16 下为 BF16/FP16；A8W8/A8W4 下为 INT8；MXFP 下为 FP8。
    - $S_{X,e}$：专家 $e$ 的输入缩放因子。A8W8/A8W4 下为 float（per-token）；MXFP 下为 FLOAT8_E8M0（MX）。
    - $W_{1,e}^{(x)}$、$W_{1,e}^{(y)}$：$W_1$ 对应专家 $e$ 的前 $N/2$ 列和后 $N/2$ 列子矩阵，由 `weight1` 按 SwiGLU 拆分推导。
    - $S_{1,e}^{(x)}$、$S_{1,e}^{(y)}$：$W_{1,e}^{(x)}$ 和 $W_{1,e}^{(y)}$ 对应的权重缩放因子,从 `weightScales1` 按维度截取。
    - $S_{2,e}$：$W_{2,e}$ 对应的权重缩放因子，来自参数 `weightScales2`，类型同上。
    - $b_{1,e}^{(x)}$、$b_{1,e}^{(y)}$：GMM1 的偏置（float），仅 A8W4 模式需要，来自参数 `l1_bias`。其余模式下为 0。
    - $b_{2,e}$：GMM2 的偏置（float），仅 A8W4 模式需要，来自参数 `l2_bias`。其余模式下为 0。
    - $Z_e^{(x)},\ Z_e^{(y)}$：GMM1 的两路矩阵乘法输出（Swish 分支和门控分支），中间结果。
    - $U_e$：SwiGLU 激活输出，维度 $m_e \times N/2$，中间结果。
    - $\hat{U}_e$：量化后的 SwiGLU 输出，中间结果。A16W16 下为 BF16/FP16；A8W8/A8W4 下为 INT8；MXFP 下为 FP8。
    - $S_{U,e}$：SwiGLU 输出缩放因子。A8W8/A8W4 下为 float（per-token）；MXFP 下为 FLOAT8_E8M0（MX）。
    - $O_e$：GMM2 的专家级输出，维度 $m_e \times H$，中间结果。
    - $\pi(i, k)$：Token $i$ 的第 $k$ 个 top-k 专家在展开排序后的行索引，由路由排序确定。

    各产品支持的量化模式如下：

    | 量化模式 | 激活量化 | 权重量化 | <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | <term>Ascend 950PR/Ascend 950DT</term> |
    | --- | --- | --- | :---: | :---: | :---: |
    | A16W16 | 无 | 无 | √ | √ | - |
    | A8W8 | per-token INT8 | per-channel INT8 | √ | √ | - |
    | A8W4 | per-token INT8 | per-channel INT4 | √ | √ | - |
    | MXFP | MX 逐组 FP8 | MX 逐组 FP8/FP4 | - | - | √ |

## 函数原型

```
mega_moe(x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer, *, l1_weights_sf=None, l2_weights_sf=None, l1_bias=None, l2_bias=None, x_active_mask=None, activation="swiglu", activation_clamp=None, weight1_type=None, weight2_type=None) -> (Tensor, Tensor)
```

## 参数说明

<table>
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
   <td>x</td>
   <td>输入</td>
   <td>表示计算使用的token数据，需根据topk_idx来发送给其他卡。要求为2维张量，shape为(BS, H)，表示有BS个token。支持非连续的Tensor。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topk_idx</td>
   <td>输入</td>
   <td>表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为(BS, K)。张量里value取值范围为[0, num_experts)，且同一行中的K个value不能重复。支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topk_weights</td>
   <td>输入</td>
   <td>表示每个token的topK个专家的权重，要求为2维张量，shape为(BS, K)，其中共享专家不需要乘权重系数，直接相加即可。支持非连续的Tensor。<ul><li>Atlas A2/A3：FLOAT32、BFLOAT16</li><li>Ascend 950：FLOAT32</li></ul></td>
   <td>FLOAT32、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>l1_weights</td>
   <td>输入</td>
   <td>GroupMatmul1计算的右矩阵，用于计算SwiGLU激活前的线性变换。支持非连续的Tensor。<ul><li>Atlas A2/A3：3维张量，shape为(expertPerRank, H, N)；INT8/INT4权重还支持FRACTAL_NZ格式。</li><li>Ascend 950：3维张量，shape为(expertPerRank, N, H)。</li></ul></td>
   <td><ul><li>Atlas A2/A3：BFLOAT16、INT8、INT4</li><li>Ascend 950：FLOAT8_E5M2、FLOAT8_E4M3FN</li></ul></td>
   <td>ND、FRACTAL_NZ</td>
  </tr>
  <tr>
   <td>l2_weights</td>
   <td>输入</td>
   <td>GroupMatmul2计算的右矩阵，用于计算SwiGLU激活后的线性变换。数据类型与l1_weights一致。支持非连续的Tensor。<ul><li>Atlas A2/A3：3维张量，shape为(expertPerRank, N/2, H)；INT8/INT4权重还支持FRACTAL_NZ格式。</li><li>Ascend 950：3维张量，shape为(expertPerRank, H, N/2)。</li></ul></td>
   <td>同l1_weights</td>
   <td>ND、FRACTAL_NZ</td>
  </tr>
  <tr>
   <td>sym_buffer</td>
   <td>输入</td>
   <td>由get_symm_buffer_for_mega_moe接口创建的结构体，封装了以下入参：<ul><li>context：本卡通信域信息（INT32）。</li><li>ep_world_size：专家并行通信域大小（INT64）。</li><li>ccl_buffer_size：CCL通信缓冲区大小（INT64）。</li><li>num_experts：MoE模型的总专家数量（INT64）。</li><li>max_recv_token_num：每个Rank最大可接收Token数，默认值为0表示自动计算（INT64）。</li><li>num_max_tokens_per_rank：每张卡上的token数量，当每个rank的BS不同时，最大的BS大小（INT64）。</li><li>dispatch_quant_mode：dispatch通信时量化模式（INT64）。Atlas A2/A3支持0（非量化，A16W16模式）和2（per-token量化，A8W8/A8W4模式）；Ascend 950仅支持4（MX模式）。</li><li>dispatch_quant_out_dtype：dispatch量化后输出的数据类型（INT64）。Atlas A2/A3 A8W8/A8W4模式下传入torch.int8；Ascend 950支持23（torch.float8_e5m2）、24（torch.float8_e4m3fn）。</li><li>combine_quant_mode：暂不支持该参数，使用默认值即可。</li><li>comm_alg：暂不支持该参数，使用默认值即可。</li></ul></td>
   <td>SymmBuffer</td>
   <td>-</td>
  </tr>
  <tr>
   <td>l1_weights_sf</td>
   <td>可选输入</td>
   <td>量化场景需要，GroupMatmul1右矩阵反量化参数。<ul><li>Atlas A2/A3：A8W8和A8W4模式下必选，2维张量shape为(expertPerRank, N)，per-channel量化缩放因子。</li><li>Ascend 950：MXFP量化场景必选，4维张量shape为(expertPerRank, N, CeilDiv(H, 64), 2)，其中CeilDiv(H, 64) = (H + 63) / 64。支持非连续的Tensor。</li></ul></td>
   <td><ul><li>Atlas A2/A3：FLOAT32</li><li>Ascend 950：FLOAT8_E8M0</li></ul></td>
   <td>ND</td>
  </tr>
  <tr>
   <td>l2_weights_sf</td>
   <td>可选输入</td>
   <td>量化场景需要，GroupMatmul2右矩阵反量化参数。<ul><li>Atlas A2/A3：A8W8和A8W4模式下必选，2维张量shape为(expertPerRank, H)，per-channel量化缩放因子。</li><li>Ascend 950：MXFP量化场景必选，4维张量shape为(expertPerRank, H, CeilDiv(N/2, 64), 2)，其中CeilDiv(N/2, 64) = (N/2 + 63) / 64。支持非连续的Tensor。</li></ul></td>
   <td><ul><li>Atlas A2/A3：FLOAT32</li><li>Ascend 950：FLOAT8_E8M0</li></ul></td>
   <td>ND</td>
  </tr>
  <tr>
   <td>l1_bias</td>
   <td>可选输入</td>
   <td>GroupMatmul1的偏置，用于补偿INT4权重量化误差。<ul><li>Atlas A2/A3：A8W4模式下必选，每个元素为1维张量shape为(N,)。</li><li>Ascend 950：不支持该参数。</li></ul></td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>l2_bias</td>
   <td>可选输入</td>
   <td>GroupMatmul2的偏置，用于补偿INT4权重量化误差。<ul><li>Atlas A2/A3：A8W4模式下必选，每个元素为1维张量shape为(H,)。</li><li>Ascend 950：不支持该参数。</li></ul></td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_active_mask</td>
   <td>可选输入</td>
   <td>表示token是否参与通信。<ul><li>Atlas A2/A3：1维张量，shape为(BS,)。</li><li>Ascend 950：暂不支持该参数，使用默认值即可。</li></ul></td>
   <td>INT8</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activation</td>
   <td>可选属性</td>
   <td>激活函数类型，默认值为"swiglu"。当前仅支持"swiglu"。</td>
   <td>STRING</td>
   <td>-</td>
  </tr>
  <tr>
   <td>activation_clamp</td>
   <td>可选属性</td>
   <td>激活函数截断值，默认值为None（表示不截断）。取值需大于等于0。</td>
   <td>FLOAT</td>
   <td>-</td>
  </tr>
  <tr>
   <td>weight1_type</td>
   <td>可选属性</td>
   <td>暂不支持该参数，使用默认值即可。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
  <tr>
   <td>weight2_type</td>
   <td>可选属性</td>
   <td>暂不支持该参数，使用默认值即可。</td>
   <td>INT64</td>
   <td>-</td>
  </tr>
 </tbody>
</table>

## 输出说明

<table>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>y_out</td>
   <td>输出</td>
   <td>表示本卡收到的token数据，2维张量，shape为(BS, H)。支持非连续的Tensor。</td>
   <td>BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_token_nums</td>
   <td>输出</td>
   <td>本卡每个专家实际收到的token数量，1维张量，shape为(local_expert_num,)。支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

## 约束说明

- **参数一致性约束**：
    - 调用算子过程中使用的 `ep_world_size`、`globalBs`、`HCCL_BUFFSIZE`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域使用约束**：
    - 所有卡的 `moe_expert_num`、`ep_world_size`、`ccl_buffer_size`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_dtype`、`num_max_tokens_per_rank` 参数取值需保持一致。

### Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas A3 推理系列产品 约束

- **参数约束**：
    - BS（x.dim0）范围 [1, 8192]。
    - H（x.dim1）范围 [1024, 10240]，且必须为32的倍数。
    - topK（topk_ids.dim1）范围 [1, 32]。
    - expertPerRank（weight1.dim0）范围 [1, 128]。
    - ep_world_size 仅支持 2, 4, 8, 16, 32, 64, 128。
    - moe_expert_num 范围 [ep_world_size, 1024]。
    - globalBs 为 0 或满足 BS <= globalBs 且 globalBs 为 ep_world_size 的倍数。
    - combineQuantMode 必须为 0，commAlg 必须为空字符串 ""。
    - y 的数据类型与 x 相同。
    - weight1 的最后一维（N）必须为偶数，且 weight1 的 dim1 必须等于 weight2 的 dim2 的二倍，这是因为 SwiGLU 激活需要将中间维度从 N 减半为 N/2。
    - weight1 和 weight2 的数据类型必须一致。
    - expertPerRank = moe_expert_num / ep_world_size，必须为整数。

- **量化模式约束**：
    - dispatch_quant_mode 由 weight1 数据类型决定：weight1 为 `int8` 或 `int4` 时，dispatch\_quant\_mode 必须为 2（per-token 量化）；weight1 为 `bfloat16` 或 `float16` 时，dispatch\_quant\_mode 必须为 0（非量化）。
    - dispatch_quant_out_dtype 由 weight1 数据类型决定：weight1 为 `int8` 或 `int4` 时，dispatch\_quant\_out\_dtype 必须为 `torch.int8`；A16W16 模式下不需要指定。

- **A16W16 场景约束**：
    - weight1 shape 为 (expertPerRank, H, N)，weight2 shape 为 (expertPerRank, N/2, H)。
    - weight1 和 weight2 数据类型为 `bfloat16` 或 `float16`。
    - 不需要 l1\_weights\_sf、l2\_weights\_sf、l1\_bias、l2\_bias。

- **A8W8 场景约束**：
    - weight1 shape 为 (expertPerRank, H, N)，weight2 shape 为 (expertPerRank, N/2, H)，数据类型为 `int8`。
    - l1\_weights\_sf shape 为 (expertPerRank, N)，l2\_weights\_sf shape 为 (expertPerRank, H)，数据类型为 `float`。
    - l1\_weights\_sf 和 l2\_weights\_sf 不可为空。
    - 不需要 l1\_bias、l2\_bias。

- **A8W4 场景约束**：
    - weight1 shape 为 (expertPerRank, H, N)，weight2 shape 为 (expertPerRank, N/2, H)，数据类型为 `int4`。
    - l1\_weights\_sf shape 为 (expertPerRank, N)，l2\_weights\_sf shape 为 (expertPerRank, H)，数据类型为 `float`。
    - l1\_weights\_sf 和 l2\_weights\_sf 不可为空。
    - l1\_bias 不可为空，每个元素 shape 为 (N,)，数据类型为 `float`。
    - l2\_bias 不可为空，每个元素 shape 为 (H,)，数据类型为 `float`。

### Ascend 950PR/Ascend 950DT 约束

- **参数约束**：
    - BS（x.dim0）范围 [1, 512]。
    - H（x.dim1）仅支持 4096、5120、7168。
    - topK（topk_ids.dim1）仅支持 6 或 8。
    - expertPerRank（weight1.dim0）范围 [1, 16]。
    - N（weight1.dim1）仅支持 1024、2048、3072、4096、7168。
    - ep_world_size 范围 [2, 768]。
    - moe_expert_num 范围 [ep_world_size, 1024]，且 moe_expert_num % ep_world_size == 0。
    - maxRecvTokenNum 范围 [0, BS × ep_world_size × min(topK, expertPerRank)]。
    - dispatchQuantOutDtype 仅支持 23（FLOAT8_E5M2）或 24（FLOAT8_E4M3FN）。
    - globalBs 为 0 或满足 BS × ep_world_size <= globalBs 且 globalBs % ep_world_size == 0。
    - 当前版本仅支持 MXFP 量化模式（dispatchQuantMode = 4），dispatch 阶段使用 MX 逐组量化（group size = 32），量化缩放因子类型为 FLOAT8_E8M0。
    - xActiveMask 参数当前版本必须传入空指针，不支持非空输入。
    - combineQuantMode 必须为 0，commAlg 必须为空字符串 ""。
    - y 的数据类型与 x 相同。
    - weight1 的 dim1（N）必须等于 weight2 的 dim2 的二倍，这是因为 SwiGLU 激活需要将中间维度从 N 减半为 N/2。
    - weightScales1 和 weightScales2 不可为空指针。
    - expertPerRank = moe_expert_num / ep_world_size，必须为整数且在 [1, 16] 范围内。

- **MXFP量化场景约束**：
    - weight1 shape 为 (expertPerRank, N, H)，weight2 shape 为 (expertPerRank, H, N/2)。
    - weightScales1 shape 为 (expertPerRank, N, CeilDiv(H, 64), 2)，其中 CeilDiv(H, 64) = (H + 63) / 64。
    - weightScales2 shape 为 (expertPerRank, H, CeilDiv(N/2, 64), 2)，其中 CeilDiv(N/2, 64) = (N/2 + 63) / 64。
    - weightScales1 的 dim3 和 weightScales2 的 dim3 必须等于 2。
    - MXFP 场景下，dispatchQuantOutDtype=23 时 weight1 和 weight2 必须为 FLOAT8_E5M2，dispatchQuantOutDtype=24 时必须为 FLOAT8_E4M3FN。
    - xActiveMask 必须为空指针。

## get\_symm\_buffer\_for\_mega\_moe接口说明
```
get_symm_buffer_for_mega_moe(group, num_experts: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, intermediate_hidden: int, *, max_recv_token_num: int = 0, dispatch_quant_mode: int = 0, dispatch_quant_out_dtype: int = 28, combine_quant_mode: int = 0, comm_alg: str = "") -> SymmBuffer:
```
## 参数说明

<table>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>group</td>
   <td>输入</td>
   <td>EP通信域名称（专家并行通信域）。</td>
   <td>STRING</td>
  </tr>
  <tr>
   <td>num_experts</td>
   <td>输入</td>
   <td>MoE 模型的总专家数量。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>num_max_tokens_per_rank</td>
   <td>输入</td>
   <td>每张卡上的token数量，当每个rank的BS不同时，最大的BS大小。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>num_topk</td>
   <td>输入</td>
   <td>表示每个token发送的专家数。预留参数，暂不支持。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>hidden</td>
   <td>输入</td>
   <td>表示每个token大小。预留参数，暂不支持。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>intermediate_hidden</td>
   <td>输入</td>
   <td>表示中间层投影维度。预留参数，暂不支持。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>max_recv_token_num</td>
   <td>可选属性</td>
   <td>每个Rank最大可接收Token数，默认值为0表示自动计算。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>dispatch_quant_mode</td>
   <td>可选属性</td>
   <td>dispatch通信时量化模式。Atlas A2/A3支持0（非量化，A16W16模式）和2（per-token量化，A8W8/A8W4模式）；Ascend 950仅支持4（MX模式）。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>dispatch_quant_out_dtype</td>
   <td>可选属性</td>
   <td>dispatch量化后输出的数据类型。Atlas A2/A3 A8W8/A8W4模式下传入torch.int8；Ascend 950支持23（torch.float8_e5m2）、24（torch.float8_e4m3fn）。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>combine_quant_mode</td>
   <td>可选属性</td>
   <td>暂不支持该参数，使用默认值即可。</td>
   <td>INT</td>
  </tr>
  <tr>
   <td>comm_alg</td>
   <td>可选属性</td>
   <td>暂不支持该参数，使用默认值即可。</td>
   <td>STRING</td>
  </tr>
 </tbody>
</table>

## 输出说明

-   **sym\_buffer** (`SymmBuffer`)：用于封装输入参数并生成`context`、`ep_world_size`和`ccl_buffer_size`。

## 调用示例

-   Atlas A2/A3 A8W8 模式调用

    ```python
    import os
    import torch
    import torch_npu
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch.multiprocessing as mp
    from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

    E = 4
    BS = 256
    H = 4096
    N = 2048
    topK = 6
    num_experts = 8

    server_num = 1
    rank_per_dev = 2
    world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
        print(f"current device set: {torch_npu.npu.current_device()}")

    def init_hccl_comm(rank):
        print(f'[INFO] device_{rank} 创建HCCL通信链路')
        master_ip = '127.0.0.1'
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
        print(f"device_{rank} init_process_group success")

        print(f"device {rank} 初始化EP域")
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group

        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        return ep_hcomm_info, ep_group

    def get_megamoe_a8w8_kwargs(
        x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        x = x.to(torch.bfloat16).npu()
        expert_ids = expert_ids.to(torch.int32).npu()
        weights1 = weights1.to(torch.int8).npu()
        weights_scales1 = weights_scales1.to(torch.float32).npu()
        weights2 = weights2.to(torch.int8).npu()
        weights_scales2 = weights_scales2.to(torch.float32).npu()
        expert_scales = expert_scales.to(torch.float32).npu()

        return {
            'x': x,
            'topk_ids': expert_ids,
            'topk_weights': expert_scales,
            'l1_weights': [weights1],
            'l1_weights_sf': [weights_scales1],
            'l2_weights': [weights2],
            'l2_weights_sf': [weights_scales2],
        }

    def run_megamoe_npu(
        queue, rank, x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        print(f"{os.getpid()=}{rank=}")
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        print(f'[INFO] device_{rank} 构造megamoe算子输入数据')
        megamoe_kwargs = get_megamoe_a8w8_kwargs(
            x=x,
            expert_ids=expert_ids,
            weights1=weights1,
            weights_scales1=weights_scales1,
            weights2=weights2,
            weights_scales2=weights_scales2,
            expert_scales=expert_scales,
        )
        distribute_buffer = get_symm_buffer_for_mega_moe(
            ep_group, num_experts=num_experts,
            num_max_tokens_per_rank=0, num_topk=topK,
            hidden=H, intermediate_hidden=0,
            dispatch_quant_mode=2, dispatch_quant_out_dtype=torch.int8
        )
        y, expert_token_nums = mega_moe(**megamoe_kwargs, sym_buffer=distribute_buffer)

        torch.npu.synchronize()
        print(f"[INFO] device_{rank} finish\n")
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank} npu finished! \n')

        queue.put([
            rank,
            [
                y.cpu(), expert_token_nums.cpu()
            ]
        ])

    def gen_npu(target_func, **server_kwargs):
        def parse_rank_input(target_func, result_queue, rank, server_kwargs):
            ep_id = rank // 1
            if target_func == run_megamoe_npu:
                return {
                    "queue": result_queue,
                    "rank": rank,
                    "x": server_kwargs["x_list"][ep_id],
                    "expert_ids": server_kwargs["expert_ids_list"][ep_id],
                    "weights1": server_kwargs["weights1_list"][ep_id],
                    "weights_scales1": server_kwargs["weights_scales1_list"][ep_id],
                    "weights2": server_kwargs["weights2_list"][ep_id],
                    "weights_scales2": server_kwargs["weights_scales2_list"][ep_id],
                    "expert_scales": server_kwargs["expert_scales_list"][ep_id]
                }

        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
        print(f"rank list is: {rank_list}")

        proc_list = []
        manager = Manager()
        result_queue = manager.Queue()
        mp.set_start_method("forkserver", force=True)
        for rank in rank_list:
            rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
            proc = Process(target=target_func, kwargs=rank_kwargs)
            proc.start()
            proc_list.append(proc)

        rank_outputs = [None] * rank_per_dev
        for proc in proc_list:
            rank_id, rank_output = result_queue.get()
            local_rank_id = rank_id - server_index * rank_per_dev
            rank_outputs[local_rank_id] = rank_output

        for proc in proc_list:
            proc.join()

        if None in rank_outputs:
            print("[ERROR] Task failed! Please check the detailed error logs printed by the subprocesses.")
            exit(1)

        category_outputs = []
        category_num = len(rank_outputs[0])
        for category_id in range(category_num):
            specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
            category_outputs.append(specific_category_output)

        return category_outputs

    if __name__ == "__main__":
        x_shape = [BS, H]
        expert_idx_shape = [BS, topK]
        weight_shape = [E, H, N]
        weight_scale_shape = [E, N]
        weight2_shape = [E, N // 2, H]
        weight2_scale_shape = [E, H]
        expert_scales_shape = [BS, topK]

        x = torch.randn(x_shape, dtype=torch.bfloat16)
        expert_scales = torch.randn(expert_scales_shape, dtype=torch.float32)
        expert_ids = torch.stack(
            [torch.randperm(num_experts)[:topK] for _ in range(BS)]
        ).to(torch.int32)
        weight1 = torch.randint(-128, 127, weight_shape, dtype=torch.int8)
        weight_scales1 = torch.randn(weight_scale_shape, dtype=torch.float32)
        weight2 = torch.randint(-128, 127, weight2_shape, dtype=torch.int8)
        weight_scales2 = torch.randn(weight2_scale_shape, dtype=torch.float32)

        golden_x_list = [x.clone() for _ in range(rank_per_dev)]
        golden_expert_ids_list = [expert_ids.clone() for _ in range(rank_per_dev)]
        golden_weights1_list = [weight1.clone() for _ in range(rank_per_dev)]
        golden_weights_scales1_list = [weight_scales1.clone() for _ in range(rank_per_dev)]
        golden_weights2_list = [weight2.clone() for _ in range(rank_per_dev)]
        golden_weights_scales2_list = [weight_scales2.clone() for _ in range(rank_per_dev)]
        golden_expert_scales_list = [expert_scales.clone() for _ in range(rank_per_dev)]

        [y, expert_token_nums] = gen_npu(
            run_megamoe_npu,
            x_list=golden_x_list,
            expert_ids_list=golden_expert_ids_list,
            weights1_list=golden_weights1_list,
            weights_scales1_list=golden_weights_scales1_list,
            weights2_list=golden_weights2_list,
            weights_scales2_list=golden_weights_scales2_list,
            expert_scales_list=golden_expert_scales_list,
        )
    ```

-   Atlas A2/A3 A8W4 模式调用

    ```python
    import os
    import torch
    import torch_npu
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch.multiprocessing as mp
    from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

    E = 4
    BS = 256
    H = 4096
    N = 2048
    topK = 6
    num_experts = 8

    server_num = 1
    rank_per_dev = 2
    world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
        print(f"current device set: {torch_npu.npu.current_device()}")

    def init_hccl_comm(rank):
        print(f'[INFO] device_{rank} 创建HCCL通信链路')
        master_ip = '127.0.0.1'
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
        print(f"device_{rank} init_process_group success")

        print(f"device {rank} 初始化EP域")
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group

        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        return ep_hcomm_info, ep_group

    def get_megamoe_a8w4_kwargs(
        x, expert_ids, weights1, weights_scales1, bias1_data,
        weights2, weights_scales2, bias2_data, expert_scales
    ):
        x = x.to(torch.bfloat16).npu()
        expert_ids = expert_ids.to(torch.int32).npu()
        weights1 = weights1.to(torch.int8).npu()
        weights_scales1 = weights_scales1.to(torch.float32).npu()
        bias1_data = bias1_data.to(torch.float32).npu()
        weights2 = weights2.to(torch.int8).npu()
        weights_scales2 = weights_scales2.to(torch.float32).npu()
        bias2_data = bias2_data.to(torch.float32).npu()
        expert_scales = expert_scales.to(torch.float32).npu()

        return {
            'x': x,
            'topk_ids': expert_ids,
            'topk_weights': expert_scales,
            'l1_weights': [weights1],
            'l1_weights_sf': [weights_scales1],
            'l1_bias': [bias1_data],
            'l2_weights': [weights2],
            'l2_weights_sf': [weights_scales2],
            'l2_bias': [bias2_data],
        }

    def run_megamoe_npu(
        queue, rank, x, expert_ids, weights1, weights_scales1, bias1_data,
        weights2, weights_scales2, bias2_data, expert_scales
    ):
        print(f"{os.getpid()=}{rank=}")
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        print(f'[INFO] device_{rank} 构造megamoe算子输入数据')
        megamoe_kwargs = get_megamoe_a8w4_kwargs(
            x=x,
            expert_ids=expert_ids,
            weights1=weights1,
            weights_scales1=weights_scales1,
            bias1_data=bias1_data,
            weights2=weights2,
            weights_scales2=weights_scales2,
            bias2_data=bias2_data,
            expert_scales=expert_scales,
        )
        distribute_buffer = get_symm_buffer_for_mega_moe(
            ep_group, num_experts=num_experts,
            num_max_tokens_per_rank=0, num_topk=topK,
            hidden=H, intermediate_hidden=0,
            dispatch_quant_mode=2, dispatch_quant_out_dtype=torch.int8
        )
        y, expert_token_nums = mega_moe(**megamoe_kwargs, sym_buffer=distribute_buffer)

        torch.npu.synchronize()
        print(f"[INFO] device_{rank} finish\n")
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank} npu finished! \n')

        queue.put([
            rank,
            [
                y.cpu(), expert_token_nums.cpu()
            ]
        ])

    def gen_npu(target_func, **server_kwargs):
        def parse_rank_input(target_func, result_queue, rank, server_kwargs):
            ep_id = rank // 1
            if target_func == run_megamoe_npu:
                return {
                    "queue": result_queue,
                    "rank": rank,
                    "x": server_kwargs["x_list"][ep_id],
                    "expert_ids": server_kwargs["expert_ids_list"][ep_id],
                    "weights1": server_kwargs["weights1_list"][ep_id],
                    "weights_scales1": server_kwargs["weights_scales1_list"][ep_id],
                    "bias1_data": server_kwargs["bias1_list"][ep_id],
                    "weights2": server_kwargs["weights2_list"][ep_id],
                    "weights_scales2": server_kwargs["weights_scales2_list"][ep_id],
                    "bias2_data": server_kwargs["bias2_list"][ep_id],
                    "expert_scales": server_kwargs["expert_scales_list"][ep_id]
                }

        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
        print(f"rank list is: {rank_list}")

        proc_list = []
        manager = Manager()
        result_queue = manager.Queue()
        mp.set_start_method("forkserver", force=True)
        for rank in rank_list:
            rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
            proc = Process(target=target_func, kwargs=rank_kwargs)
            proc.start()
            proc_list.append(proc)

        rank_outputs = [None] * rank_per_dev
        for proc in proc_list:
            rank_id, rank_output = result_queue.get()
            local_rank_id = rank_id - server_index * rank_per_dev
            rank_outputs[local_rank_id] = rank_output

        for proc in proc_list:
            proc.join()

        if None in rank_outputs:
            print("[ERROR] Task failed! Please check the detailed error logs printed by the subprocesses.")
            exit(1)

        category_outputs = []
        category_num = len(rank_outputs[0])
        for category_id in range(category_num):
            specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
            category_outputs.append(specific_category_output)

        return category_outputs

    if __name__ == "__main__":
        x_shape = [BS, H]
        expert_idx_shape = [BS, topK]
        weight_shape = [E, H, N]
        weight_scale_shape = [E, N]
        bias1_shape = [E, N]
        weight2_shape = [E, N // 2, H]
        weight2_scale_shape = [E, H]
        bias2_shape = [E, H]
        expert_scales_shape = [BS, topK]

        x = torch.randn(x_shape, dtype=torch.bfloat16)
        expert_scales = torch.randn(expert_scales_shape, dtype=torch.float32)
        expert_ids = torch.stack(
            [torch.randperm(num_experts)[:topK] for _ in range(BS)]
        ).to(torch.int32)
        weight1 = torch.randint(-8, 7, weight_shape, dtype=torch.int8)
        weight_scales1 = torch.randn(weight_scale_shape, dtype=torch.float32)
        bias1 = torch.randn(bias1_shape, dtype=torch.float32)
        weight2 = torch.randint(-8, 7, weight2_shape, dtype=torch.int8)
        weight_scales2 = torch.randn(weight2_scale_shape, dtype=torch.float32)
        bias2 = torch.randn(bias2_shape, dtype=torch.float32)

        golden_x_list = [x.clone() for _ in range(rank_per_dev)]
        golden_expert_ids_list = [expert_ids.clone() for _ in range(rank_per_dev)]
        golden_weights1_list = [weight1.clone() for _ in range(rank_per_dev)]
        golden_weights_scales1_list = [weight_scales1.clone() for _ in range(rank_per_dev)]
        golden_bias1_list = [bias1.clone() for _ in range(rank_per_dev)]
        golden_weights2_list = [weight2.clone() for _ in range(rank_per_dev)]
        golden_weights_scales2_list = [weight_scales2.clone() for _ in range(rank_per_dev)]
        golden_bias2_list = [bias2.clone() for _ in range(rank_per_dev)]
        golden_expert_scales_list = [expert_scales.clone() for _ in range(rank_per_dev)]

        [y, expert_token_nums] = gen_npu(
            run_megamoe_npu,
            x_list=golden_x_list,
            expert_ids_list=golden_expert_ids_list,
            weights1_list=golden_weights1_list,
            weights_scales1_list=golden_weights_scales1_list,
            bias1_list=golden_bias1_list,
            weights2_list=golden_weights2_list,
            weights_scales2_list=golden_weights_scales2_list,
            bias2_list=golden_bias2_list,
            expert_scales_list=golden_expert_scales_list,
        )
    ```

-   Ascend 950 MXFP 模式调用

    ```python
    import os
    import torch
    import torch_npu
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch.multiprocessing as mp
    from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    import torchair

    E = 4
    BS = 256
    H = 4096
    N = 1024
    topK = 6
    num_experts = 8

    server_num = 1
    rank_per_dev = 2
    world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def ceil(a, b):
        return (a + b - 1) // b

    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
        print(f"current device set: {torch_npu.npu.current_device()}")

    def init_hccl_comm(rank):
        print(f'[INFO] device_{rank} 创建HCCL通信链路')
        master_ip = '127.0.0.1'
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
        print(f"device_{rank} init_process_group success")

        print(f"device {rank} 初始化EP域")
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group

        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        return ep_hcomm_info, ep_group

    def get_megamoe_kwargs(
        x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        x = x.to(torch.bfloat16).npu()
        expert_ids = expert_ids.to(torch.int32).npu()
        weights1 = weights1.to(torch.float8_e5m2).npu()
        weights_scales1 = weights_scales1.to(torch.float8_e8m0fnu).npu()
        weights2 = weights2.to(torch.float8_e5m2).npu()
        weights_scales2 = weights_scales2.to(torch.float8_e8m0fnu).npu()
        expert_scales = expert_scales.to(torch.bfloat16).npu()

        return {
            'x': x,
            'topk_ids': expert_ids,
            'topk_weights': expert_scales,
            'l1_weights': [weights1],
            'l1_weights_sf': [weights_scales1],
            'l2_weights': [weights2],
            'l2_weights_sf': [weights_scales2],
        }

    def run_megamoe_npu(
        queue, rank, x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        print(f"{os.getpid()=}{rank=}")
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        print(f'[INFO] device_{rank} 构造megamoe算子输入数据')
        megamoe_kwargs = get_megamoe_kwargs(
            x=x,
            expert_ids=expert_ids,
            weights1=weights1,
            weights_scales1=weights_scales1,
            weights2=weights2,
            weights_scales2=weights_scales2,
            expert_scales=expert_scales,
        )
        distribute_buffer = get_symm_buffer_for_mega_moe(
            ep_group, num_experts=num_experts,
            num_max_tokens_per_rank=0, num_topk=topK,
            hidden=H, intermediate_hidden=0,
            dispatch_quant_mode=4, dispatch_quant_out_dtype=23
        )
        y, expert_token_nums = mega_moe(**megamoe_kwargs, sym_buffer=distribute_buffer)

        torch.npu.synchronize()
        print(f"[INFO] device_{rank} finish\n")
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank} npu finished! \n')

        queue.put([
            rank,
            [
                y.cpu(), expert_token_nums.cpu()
            ]
        ])

    def gen_npu(target_func, **server_kwargs):
        def parse_rank_input(target_func, result_queue, rank, server_kwargs):

            ep_id = rank // 1

            if target_func == run_megamoe_npu:
                return {
                    "queue": result_queue,
                    "rank": rank,
                    "x": server_kwargs["x_list"][ep_id],
                    "expert_ids": server_kwargs["expert_ids_list"][ep_id],
                    "weights1": server_kwargs["weights1_list"][ep_id],
                    "weights_scales1": server_kwargs["weights_scales1_list"][ep_id],
                    "weights2": server_kwargs["weights2_list"][ep_id],
                    "weights_scales2": server_kwargs["weights_scales2_list"][ep_id],
                    "expert_scales": server_kwargs["expert_scales_list"][ep_id]
                }


        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
        print(f"rank list is: {rank_list}")

        proc_list = []
        manager = Manager()
        result_queue = manager.Queue()
        mp.set_start_method("forkserver", force=True)
        for rank in rank_list:
            rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
            proc = Process(target=target_func, kwargs=rank_kwargs)
            proc.start()
            proc_list.append(proc)


        rank_outputs = [None] * rank_per_dev
        for proc in proc_list:
            rank_id, rank_output = result_queue.get()
            local_rank_id = rank_id - server_index * rank_per_dev
            rank_outputs[local_rank_id] = rank_output


        for proc in proc_list:
            proc.join()

        if None in rank_outputs:
            print("[ERROR] Task failed! Please check the detailed error logs printed by the subprocesses.")
            exit(1)

        category_outputs = []
        category_num = len(rank_outputs[0])
        for category_id in range(category_num):
            specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
            category_outputs.append(specific_category_output)

        return category_outputs

    if __name__ == "__main__":
        x_shape = [BS, H]
        expert_idx_shape = [BS, topK]
        weight_shape = [E, N, H]
        weight_scale_shape = [E, N, ceil(H, 64), 2]
        output_shape = [BS, N//2]
        weight2_shape = [E, H, N//2]
        weight2_scale_shape = [E, H, ceil(N//2, 64), 2]
        expert_scales_shape = [BS, topK]
        x = torch.randn(x_shape, dtype=torch.bfloat16)
        expert_scales = torch.randn(expert_scales_shape, dtype=torch.bfloat16)
        expert_ids = torch.stack(
            [torch.randperm(num_experts)[:topK] for _ in range(BS)]
        ).to(torch.int32)
        weight1 = torch.randn(weight_shape, dtype=torch.float32).to(torch.float8_e5m2)
        weight_scales1 = torch.randint(125, 130, weight_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        weight2 = torch.randn(weight2_shape, dtype=torch.float32).to(torch.float8_e5m2)
        weight_scales2 = torch.randint(125, 130, weight2_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        golden_x_list = [x.clone() for _ in range(rank_per_dev)]
        golden_expert_ids_list = [expert_ids.clone() for _ in range(rank_per_dev)]
        golden_weights1_list = [weight1.clone() for _ in range(rank_per_dev)]
        golden_weights_scales1_list = [weight_scales1.clone() for _ in range(rank_per_dev)]
        golden_weights2_list = [weight2.clone() for _ in range(rank_per_dev)]
        golden_weights_scales2_list = [weight_scales2.clone() for _ in range(rank_per_dev)]
        golden_expert_scales_list = [expert_scales.clone() for _ in range(rank_per_dev)]

        [y, expert_token_nums] = gen_npu(
            run_megamoe_npu,
            x_list=golden_x_list,
            expert_ids_list=golden_expert_ids_list,
            weights1_list=golden_weights1_list,
            weights_scales1_list=golden_weights_scales1_list,
            weights2_list=golden_weights2_list,
            weights_scales2_list=golden_weights_scales2_list,
            expert_scales_list=golden_expert_scales_list,
        )
    ```

-   图模式调用（暂不支持）
