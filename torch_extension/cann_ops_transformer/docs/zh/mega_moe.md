# mega_moe

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    Mega MoE算子 将 MoE 层的专家 FFN 的完整计算流程及前后数据通信（即 Dispatch + Linear1 + SwiGLU + Linear2 + Combine）融合为单个算子，实现了通信和计算的掩盖。
    该算子提供了[mega_moe](#mega_moe接口说明)、[get_mega_moe_ccl_buffer_size](#get_mega_moe_ccl_buffer_size接口说明)与 [get_symm_buffer_for_mega_moe](#get_symm_buffer_for_mega_moe接口说明)等接口，这些接口需配套使用。

- 计算公式：
    - 输入：
        - $\mathbf{X} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$：激活矩阵，对应入参 `x`。$\text{total\_num\_tokens}$ 是全局总 token 数，$\text{hidden}$ 是隐藏层维度。
        - $\mathbf{E} \in \mathbb{Z}^{\text{total\_num\_tokens} \times \text{num\_topk}}$：token 选择的专家编号矩阵，对应入参 `topk_ids`。$\text{num\_topk}$ 是每个 token 选择的专家数量。
        - $\mathbf{G} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{num\_topk}}$：token 选择的专家的门控权重矩阵，对应入参 `topk_weights`。
        - $\mathbf{W}_1 \in \mathbb{R}^{\text{num\_experts} \times \text{hidden} \times (2 \text{intermediate\_hidden})}$：Linear1 的权重矩阵，对应入参 `l1_weights`。$\text{num\_experts}$ 是专家数量，$\text{intermediate\_hidden}$ 是中间层维度。
        - $\mathbf{W}_2 \in \mathbb{R}^{\text{num\_experts} \times \text{intermediate\_hidden} \times \text{hidden}}$：Linear2 的权重矩阵，对应入参 `l2_weights`。
    - 输出：
        - $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$：最终输出矩阵，对应出参 `y`。
    - 约定：
        - $⋅$ 表示矩阵乘法，$⊙$ 表示逐元素乘法。
        - $\left \lfloor z\right \rceil$ 表示将 $z$ 四舍五入到最近的整数，$\left \lfloor z\right \rfloor$ 表示将 $z$ 向下取整。
        - $|z|$ 表示取绝对值，$\max(z)$ 表示取最大值。
        - 全体 token 的集合为 $\{ \text{token}_i \mid i \in \{0, 1, \dots, \text{total\_num\_tokens} - 1\} \}$。
        - $\text{token}_i$ 的 token 表示（即隐藏状态向量）为 $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{hidden}}$，且 $\mathbf{x}_i = \mathbf{X}[i,:]$。
        - $\text{token}_i$ 的专家索引为 $e_{i,k} = \mathbf{E}[i,k],\quad k \in \{0,\dots,\text{num\_topk} - 1\},\quad e_{i,k} \in \{0,\dots,\text{num\_experts} - 1\}$。
        - 激活函数 $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1+e^{-z}}$，其中 $\sigma$ 为 Sigmoid 函数。
        - $\mathbb{Z}_4 = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z}_8^{\text{sym}} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}, \quad \mathbb{Z}_{32} = \{ x \in \mathbb{Z} \mid -2^{31} \le x \le 2^{31}-1 \}$。其中 $\mathbb{Z}_8^{\text{sym}}$ 的上标 $\text{sym}$ 表示对称量化值域区间：其值域关于 $-127$ 与 $127$ 对称取整，与标准 INT8 的 $[-128, 127]$ 值域不同，故以 $\text{sym}$ 上标区分。
        - 张量切片操作采用 Python 风格的 `start:stop:step` 表示法，例如，$[0::2, :]$ 代表取偶数行、$[1::2, :]$ 代表取奇数行。
        - $\mathrm{bitcast}_{T}(\mathbf{Z})$ 表示二进制重解释操作，将张量$\mathbf{Z}$的底层二进制数据按目标类型 $T$ 重新解释。
    - 计算过程

        各产品支持的**Linear计算时**的激活矩阵 A 和权重矩阵 W 的数据类型如下：

        | 场景名    |  A |  W   |<term>Atlas A2 训练系列产品<br>Atlas A2 推理系列产品</term> | <term>Atlas A3 训练系列产品<br>Atlas A3 推理系列产品</term> | <term>Ascend 950PR<br>Ascend 950DT</term> |
        | ---      | :---:                      | :---:                    | :---: | :---: | :---: |
        | A16W16   | BFLOAT16                   |BFLOAT16                  | √ | √ | - |
        | A8W8-INT | INT8                       | INT8                     | √ | √ | - |
        | A8W8-FP  | FLOAT8_E4M3FN、FLOAT8_E5M2 |FLOAT8_E4M3FN、FLOAT8_E5M2 | - | - | √ |
        | A8W4-INT | INT8                       | INT4                     | √ | √ | - |

        <details>
        <summary> A16W16 非量化场景</summary>

        - **EP Dispatch**

            在 Dispatch 阶段，每个 $\text{token}_i$ 将其token表示 $\mathbf{x}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$。即对于每个 $k$，专家 $e_{i,k}$ 接收一份 $\mathbf{x}_i$。

            记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数，则有 $\mathbf{X}_e \in \mathbb{R}^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的token表示 $\mathbf{x}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示。

            对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{x}_i$。该映射记录了 $\mathbf{x}_i$ 在专家 $e_{i,k}$ 的输入矩阵 $\mathbf{X}_{e_{i,k}}$ 中的位置。

        - **Expert Compute**

            在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。整个计算过程分为如下三个子步骤。

            **1. Linear1 投影**

            Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。其计算公式为
            $$
            \mathbf{H}_e = \mathbf{X}_e \cdot \mathbf{W}_1[e]
            \quad\bigl(\in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}\bigr)
            $$
            **2. SwiGLU 激活**

            首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
            $$
            \mathbf{H}_{\text{gate}} = \mathbf{H}_e\bigl[:,\;\text{:intermediate\_hidden}\bigr], \qquad
            \mathbf{H}_{\text{up}} = \mathbf{H}_e\bigl[:,\;\text{intermediate\_hidden:}\bigr]
            $$
            $$
            \mathbf{A}_e = \text{SiLU}\bigl(\mathbf{H}_{\text{gate}}\bigr) \;\odot\; \mathbf{H}_{\text{up}}
            \quad\bigl(\in \mathbb{R}^{N_e \times \text{intermediate\_hidden}}\bigr)
            $$

            **3. Linear2 投影**

            Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
            $$
            \mathbf{Y}_e = \mathbf{A}_e \cdot \mathbf{W}_2[e]
            \quad\bigl(\in \mathbb{R}^{N_e \times \text{hidden}}\bigr)
            $$

            经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。
        - **Token Combine**

            Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：

            $$
            \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\;
            \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
            \qquad\bigl(\in \mathbb{R}^{1 \times \text{hidden}}\bigr)
            $$

            其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

            所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。
        </details>

        <details>
        <summary> A8W8-INT 量化场景</summary>

        - **输入**
            - $\mathbf{S}^{W1} \in \mathbb{R}^{\text{num\_experts} \times (2\times \text{intermediate\_hidden})}$：Linear1 权重矩阵的逐通道缩放因子，对应入参 `l1_weights_sf`。
            - $\mathbf{S}^{W2} \in \mathbb{R}^{\text{num\_experts} \times \text{hidden}}$：Linear2 权重矩阵的逐通道缩放因子，对应入参 `l2_weights_sf`。

        - **EP Dispatch**

            在 Dispatch 通信之前，首先将原始 BF16 激活矩阵 $\mathbf{X}$ 量化为 INT8。对每个 $\text{token}_i$，计算其逐 token 缩放因子：
            $$
            s^{X}_i = \frac{\max(|\mathbf{X}[i,:]|)}{127} \in \mathbb{R}
            $$
            然后量化得到 INT8 表示：
            $$
            \mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[i,:]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{hidden}}
            $$

            在 Dispatch 通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\mathbf{q}_i$ 和缩放因子 $s^{X}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,\text{num\_topk}-1}$（$e_{i,k} = \mathbf{E}[i,k]$）。

            记 $I_e = \{\, i \mid \exists k,\ \mathbf{E}[i,k]=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $N_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数。则有 $\mathbf{Q}_e \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{hidden}}$ 是由所有满足 $i \in I_e$ 的 $\mathbf{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\mathbf{s}^{X}_e \in \mathbb{R}^{N_e}$，其元素由所有满足 $i \in I_e$ 的 $s^{X}_i$ 按与 $\mathbf{Q}_e$ 相同的行顺序堆叠而成。

            对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,N_{e_{i,k}}-1\}$，使得 $\mathbf{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \mathbf{q}_i$。该映射记录了 $\mathbf{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

        - **Expert Compute**

            在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 SwiGLU 结构以提升表达能力。在A8W8场景下，两个线性层都使用 INT8 输入和 INT8 权重进行矩阵乘，得到 INT32 中间结果并反量化。具体分为三个子步骤。

            **1. Linear1 投影（INT8 矩阵乘 + 反量化）**

            Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。计算时执行 INT8 矩阵乘法，得到 INT32 计算结果：
            $$
            \mathbf{C}_e^{\text{int32}} = \mathbf{Q}_e \cdot \mathbf{W}_1[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times 2\cdot\text{intermediate\_hidden}}
            $$
            然后反量化为预激活值 $\mathbf{H}_e$：
            $$
            \mathbf{H}_e = \left( \mathbf{C}_e^{\text{int32}} \odot \mathbf{s}^{W1}_e \right) \odot \mathbf{s}^{X}_e \quad \in \mathbb{R}^{N_e \times 2\cdot\text{intermediate\_hidden}}
            $$

            **2. SwiGLU 激活**

            首先将 $\mathbf{H}_e$ 沿列维度拆分为 **gate 部分** $\mathbf{H}_{\text{gate}}$ 和 **up 部分** $\mathbf{H}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\mathbf{A}_e$。
            $$
            \mathbf{H}_{\text{gate}} = \mathbf{H}_e[:, :\text{intermediate\_hidden}], \qquad
            \mathbf{H}_{\text{up}} = \mathbf{H}_e[:, \text{intermediate\_hidden}:]
            $$
            $$
            \mathbf{A}_e = \operatorname{SiLU}(\mathbf{H}_{\text{gate}}) \odot \mathbf{H}_{\text{up}}
            $$

            **3. Linear2 投影（量化 + INT8 矩阵乘 + 反量化）**

            Linear2 投影作为第二层线性变换，将中间激活表示 $\mathbf{A}_e$ 从高维空间投影回原始的隐藏维度 $\text{hidden}$，使专家输出能够与残差连接等后续操作兼容。
            在 A8W8 场景下，需要将激活值量化为 INT8，因此先对 $\mathbf{A}_e$ 的每一行（每个 token）计算缩放因子：
            $$
            s^{A_e}_i = \frac{\max(|\mathbf{A}_e[i,:]|)}{127}, \quad i=0,\dots,N_e-1
            $$
            得到专家 $e$ 在 Linear2 计算时的激活缩放因子 $\mathbf{s}^{A_e}_e \in \mathbb{R}^{N_e}$。然后量化：
            $$
            \mathbf{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\mathbf{A}_e[i,:]}{s^{A_e}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{N_e \times \text{intermediate\_hidden}}
            $$
            再执行 INT8 矩阵乘法并反量化：
            $$
            \mathbf{D}_e^{\text{int32}} = \mathbf{A}_e^{\text{int8}} \cdot \mathbf{W}_2[e]^{\text{int8}} \quad \in \mathbb{Z}_{32}^{N_e \times \text{hidden}}
            $$
            $$
            \mathbf{Y}_e = \left( \mathbf{D}_e^{\text{int32}} \odot \mathbf{s}^{W2}_e \right) \odot \mathbf{s}^{A_e}_e \quad \in \mathbb{R}^{N_e \times \text{hidden}}
            $$

            经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\mathbf{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

        - **Token Combine**

            Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：
            $$
            \mathbf{y}_i = \sum_{k=0}^{\text{num\_topk} - 1} w_k \;\cdot\; \mathbf{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times \text{hidden}}
            $$
            其中 $w_k = \mathbf{G}[i,k]$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

            所有 token 的输出按输入顺序堆叠为最终输出 $\mathbf{Y} \in \mathbb{R}^{\text{total\_num\_tokens} \times \text{hidden}}$。

        </details>

## 函数原型

```python
cann_ops_transformer.mega_moe(x, topk_ids, topk_weights, l1_weights, l2_weights, sym_buffer, *, l1_weights_sf=None, l2_weights_sf=None, l1_bias=None, l2_bias=None, x_active_mask=None, activation="swiglu", activation_clamp=None, weight1_type=None, weight2_type=None) -> (Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>计算使用的token数据，需根据`topk_ids`发送给其他卡，对应公式中的X。要求为2维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>bfloat16</td>
        <td>(BS,H)</td>
    </tr>
    <tr>
        <td>topk_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token的topK个专家索引，决定每个token要发给哪些专家，对应公式中的E。要求为2维张量，数据格式为ND，支持非连续的Tensor。张量里value取值范围为[0,num_experts)，且同一行中的K个value不能重复。</td>
        <td>int32</td>
        <td>(BS,K)</td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token的topK个专家的权重，对应公式中的G。要求为2维张量，其中共享专家不需要乘权重系数，直接相加即可。数据格式为ND，支持非连续的Tensor。</td>
        <td>float</td>
        <td>(BS,K)</td>
    </tr>
    <tr>
        <td>l1_weights</td>
        <td>List[Tensor]</td>
        <td>必选</td>
        <td>GroupMatmul1计算的右矩阵，用于计算SwiGLU激活前的线性变换，对应公式中的W<sub>1</sub>。支持3维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>float8_e5m2、float8_e4m3fn</td>
        <td>(expertPerRank,N,H)</td>
    </tr>
    <tr>
        <td>l2_weights</td>
        <td>List[Tensor]</td>
        <td>必选</td>
        <td>GroupMatmul2计算的右矩阵，用于计算SwiGLU激活后的线性变换，对应公式中的W<sub>2</sub>。支持3维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>float8_e5m2、float8_e4m3fn</td>
        <td>(expertPerRank,H,N/2)</td>
    </tr>
    <tr>
        <td>sym_buffer</td>
        <td>SymmBuffer</td>
        <td>必选</td>
        <td>由[get_symm_buffer_for_mega_moe](#get_symm_buffer_for_mega_moe接口说明)接口创建的结构体，封装context（本卡通信域信息，int32）、ep_world_size（专家并行通信域大小，int64）、ccl_buffer_size（CCL通信缓冲区大小，int64）、num_experts（MoE模型的总专家数量，int64）、max_recv_token_num（每个Rank最大可接收Token数，默认值为0表示自动计算，int64）、num_max_tokens_per_rank（每张卡上的token数量，当每个rank的BS不同时为最大的BS大小，int64）、dispatch_quant_mode（dispatch通信时量化模式，目前仅支持4（MX模式），int64）、dispatch_quant_out_dtype（dispatch量化后输出的数据类型，支持输入23（torch.float8_e5m2）、24（torch.float8_e4m3fn），int64）、combine_quant_mode（暂不支持该参数，使用默认值即可）、comm_alg（暂不支持该参数，使用默认值即可）等入参。</td>
        <td>SymmBuffer</td>
        <td>-</td>
    </tr>
    <tr>
        <td>l1_weights_sf</td>
        <td>List[Tensor]</td>
        <td>可选</td>
        <td>量化场景需要，GroupMatmul1右矩阵反量化参数，对应公式中的S<sup>W1</sup>。当前支持4维张量，shape中CeilDiv(H,64)=(H+63)/64。数据格式为ND，支持非连续的Tensor。默认值为None。</td>
        <td>float8_e8m0</td>
        <td>(expertPerRank,N,CeilDiv(H,64),2)</td>
    </tr>
    <tr>
        <td>l2_weights_sf</td>
        <td>List[Tensor]</td>
        <td>可选</td>
        <td>量化场景需要，GroupMatmul2右矩阵反量化参数，对应公式中的S<sup>W2</sup>。当前支持4维张量，shape中CeilDiv(N/2,64)=(N/2+63)/64。数据格式为ND，支持非连续的Tensor。默认值为None。</td>
        <td>float8_e8m0</td>
        <td>(expertPerRank,H,CeilDiv(N/2,64),2)</td>
    </tr>
    <tr>
        <td>l1_bias</td>
        <td>List[Tensor]</td>
        <td>可选</td>
        <td>GroupMatmul1计算的偏置项。暂不支持该参数，使用默认值即可。默认值为None。</td>
        <td>float</td>
        <td>(expertPerRank,2intermediate_hidden)</td>
    </tr>
    <tr>
        <td>l2_bias</td>
        <td>List[Tensor]</td>
        <td>可选</td>
        <td>GroupMatmul2计算的偏置项。暂不支持该参数，使用默认值即可。默认值为None。</td>
        <td>float</td>
        <td>(expertPerRank,H)</td>
    </tr>
    <tr>
        <td>x_active_mask</td>
        <td>Optional[Tensor]</td>
        <td>可选</td>
        <td>表示token是否参与通信。暂不支持该参数，使用默认值即可。默认值为None。</td>
        <td>bool</td>
        <td>(BS,)</td>
    </tr>
    <tr>
        <td>activation</td>
        <td>str</td>
        <td>可选</td>
        <td>专家网络使用的激活函数类型，当前支持"swiglu"。默认值为"swiglu"。</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activation_clamp</td>
        <td>Optional[float]</td>
        <td>可选</td>
        <td>激活函数的截断阈值。暂不支持该参数，使用默认值即可。默认值为None。</td>
        <td>float</td>
        <td>-</td>
    </tr>
    <tr>
        <td>weight1_type</td>
        <td>Optional[int]</td>
        <td>可选</td>
        <td>暂不支持该参数，使用默认值即可。默认值为None。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>weight2_type</td>
        <td>Optional[int]</td>
        <td>可选</td>
        <td>暂不支持该参数，使用默认值即可。默认值为None。</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡收到的token数据，对应公式中的Y，数据类型与输入`x`保持一致。要求为2维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>bfloat16</td>
        <td>(BS,H)</td>
    </tr>
    <tr>
        <td>expert_token_nums</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡每个专家实际收到的token数量。要求为1维张量，数据格式为ND，支持非连续的Tensor。</td>
        <td>int32</td>
        <td>(local_expert_num,)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式调用。
- get_mega_moe_ccl_buffer_size、get_symm_buffer_for_mega_moe、mega_moe必须配套使用。

- Shape变量定义：
  - BS：x的第0维（x.dim0），表示本卡token数量。
  - H：x的第1维（x.dim1），表示hidden size隐藏层大小。
  - K：topk_ids的第1维（topkIds.dim1），表示每个token选取的topK个专家。
  - expertPerRank：weight1的第0维（weight1.dim0），表示每个Rank的专家数，expertPerRank = moeExpertNum / epWorldSize。
  - N：weight1的第1维（weight1.dim1），表示中间层维度。
  - num_experts：MoE模型的总专家数量。
  - epWorldSize：专家并行通信域大小。
  - local_expert_num：本卡专家数量，等于moeExpertNum / epWorldSize。

- 各张量参数的list[Tensor]长度、是否转置、是否支持非连续Tensor约束如下：

<table style="undefined;table-layout: fixed; width:1000px"><colgroup>
<col style="width: 160px">
<col style="width: 220px">
<col style="width: 120px">
<col style="width: 220px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>list[Tensor]长度</th>
    <th>是否转置</th>
    <th>是否支持非连续Tensor</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>-</td>
        <td>否</td>
        <td>支持</td>
    </tr>
    <tr>
        <td>topk_ids</td>
        <td>-</td>
        <td>否</td>
        <td>支持</td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>-</td>
        <td>否</td>
        <td>支持</td>
    </tr>
    <tr>
        <td>l1_weights</td>
        <td>=expertPerRank（每个Rank的专家数）</td>
        <td>否</td>
        <td>支持</td>
    </tr>
    <tr>
        <td>l2_weights</td>
        <td>=expertPerRank（每个Rank的专家数）</td>
        <td>否</td>
        <td>支持</td>
    </tr>
    <tr>
        <td>l1_weights_sf</td>
        <td>=expertPerRank（每个Rank的专家数）</td>
        <td>否</td>
        <td>支持</td>
    </tr>
    <tr>
        <td>l2_weights_sf</td>
        <td>=expertPerRank（每个Rank的专家数）</td>
        <td>否</td>
        <td>支持</td>
    </tr>
</tbody>
</table>

- **参数一致性约束**：
    - 调用算子过程中使用的`epWorldSize`、`globalBs`、`HCCL_BUFFSIZE`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域使用约束**：
    - 仅支持`EP`域，无`TP`域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性。
    - 所有卡的`moe_expert_num`、`ep_world_size`、`ccl_buffer_size`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_dtype`、`global_bs`参数取值需保持一致。

- **参数约束**：
    - BS（x.dim0）范围 [1, 512]。
    - H（x.dim1）仅支持4096、5120、7168。
    - topK（topkIds.dim1）仅支持6或8。
    - expertPerRank（weight1.dim0）范围 [1, 16]。
    - N（weight1.dim1）仅支持1024、2048、3072、4096、7168。
    - epWorldSize范围 [2, 768]。
    - moeExpertNum范围 [epWorldSize, 1024]，且moeExpertNum % epWorldSize == 0。
    - maxRecvTokenNum范围 [0, BS × epWorldSize × min(topK, expertPerRank)]。
    - dispatchQuantOutType仅支持23（FLOAT8_E5M2）或24（FLOAT8_E4M3FN）。
    - globalBs为0或满足BS × epWorldSize <= globalBs且globalBs % epWorldSize == 0。
    - 当前版本仅支持MXFP量化模式（dispatchQuantMode = 4），dispatch阶段使用MX逐组量化（group size = 32），量化缩放因子类型为FLOAT8_E8M0。
    - xActiveMask和scales参数当前版本必须传入空指针，不支持非空输入。
    - combineQuantMode必须为0，commAlg必须为空字符串""。
    - y的数据类型与x相同。
    - weight1的dim1（N）必须等于weight2的dim2的二倍，这是因为SwiGLU激活需要将中间维度从N减半为N/2。
    - weightScales1和weightScales2不可为空指针。
    - expertPerRank = moeExpertNum / epWorldSize，必须为整数且在 [1, 16] 范围内。
    - weightScales1和weightScales2不可为空指针。

- **MXFP量化场景约束**：
    - weight1 shape为(expertPerRank, N, H)，weight2 shape为(expertPerRank, H, N/2)。
    - weightScales1 shape为(expertPerRank, N, CeilDiv(H, 64), 2)，其中CeilDiv(H, 64) = (H + 63) / 64。
    - weightScales2 shape为(expertPerRank, H, CeilDiv(N/2, 64), 2)，其中CeilDiv(N/2, 64) = (N/2 + 63) / 64。
    - weightScales1的dim3和weightScales2的dim3必须等于2。
    - MXFP场景下，dispatchQuantOutType=23时weight1和weight2必须为FLOAT8_E5M2，dispatchQuantOutType=24时必须为FLOAT8_E4M3FN。
    - xActiveMask和scales必须为空指针。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import os
  import torch
  import torch_npu
  from torch.multiprocessing import Process, Manager
  import torch.distributed as dist
  from torch.distributed import ReduceOp
  import torch.multiprocessing as mp
  from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
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
      # 创建HCCL通信链路并初始化EP域
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
      # 步骤1：构造distribute_buffer（内部调用get_mega_moe_ccl_buffer_size计算buffer大小）
      distribute_buffer = get_symm_buffer_for_mega_moe(
          ep_group, num_experts=num_experts,
          num_max_tokens_per_rank=0, num_topk=topK,
          hidden=H, intermediate_hidden=0,
          dispatch_quant_mode=4, dispatch_quant_out_dtype=23
      )
      # 步骤2：运行mega_moe，传入上一步构造的sym_buffer
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

      # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
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
      # 构造输入
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

## get_mega_moe_ccl_buffer_size接口说明

### 功能说明

- 接口功能：

    需与[mega_moe](#mega_moe接口说明)配套使用，用于计算mega_moe算子所需的HCCL通信buffer_size大小（单位：MB）。

- 计算公式：

    - 计算dispatch阶段存储所有专家接收到整个EP域中的Token数量空间大小：

        $$dispatch\_token\_per\_expert = ep\_world\_size * Align128(ep\_world\_size * local\_moe\_expert\_num) * sizeof(int32)$$

        其中ep\_world\_size为专家并行通信域大小

    - 计算dispatch量化后输出的token及scales存放需要的空间大小：

        $$quant\_token\_scales = Align512(num\_max\_token\_per\_rank * topK * (hidden + hidden / 32))$$

        其中num\_max\_token\_per\_rank为每张卡上的token数量
    - 计算combine阶段处理发回数据所需空间大小：

        $$combine\_out = Align512(num\_max\_token\_per\_rank * topK * y\_out\_dtype\_size)$$

        其中y\_out\_dtype\_size为输出的token类型转为字节数
    - 计算ccl\_buffer\_size大小：

        $$ccl\_buffer\_size = Align2(Align1MB(peermem\_data\_offset + dispatch\_token\_per\_expert + quant\_token\_scales + combine\_out) / 1MB) / 2$$
        其中peermem\_data\_offset为全卡软同步所需空间大小`60 * 1024B`

### 函数原型

```python
cann_ops_transformer.get_mega_moe_ccl_buffer_size(ep_world_size, moe_expert_num, num_max_tokens_per_rank, num_topk, hidden, dispatch_quant_mode=0, dispatch_quant_out_dtype=28, combine_quant_mode=0, comm_alg="") -> int
```

### 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>ep_world_size</td>
        <td>int</td>
        <td>必选</td>
        <td>通信域的大小。取值范围[2,768]。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moe_expert_num</td>
        <td>int</td>
        <td>必选</td>
        <td>MoE专家数量，取值范围[1,1024]。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_tokens_per_rank</td>
        <td>int</td>
        <td>必选</td>
        <td>每张卡上的token数量。当每个rank的BS不同时，为最大的BS大小。取值范围[1,512]。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_topk</td>
        <td>int</td>
        <td>必选</td>
        <td>选取topK个专家，目前仅支持6或8。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>hidden</td>
        <td>int</td>
        <td>必选</td>
        <td>hidden size隐藏层大小。目前仅支持4096、5120、7168。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dispatch_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>dispatch通信时量化模式，目前仅支持4（MX模式）。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dispatch_quant_out_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>dispatch量化后输出的数据类型，支持输入23（torch.float8_e5m2）、24（torch.float8_e4m3fn）。默认值为28。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>combine_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>暂不支持该参数，使用默认值即可。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>可选</td>
        <td>暂不支持该参数，使用默认值即可。默认值为""。</td>
        <td>str</td>
        <td>-</td>
    </tr>
</tbody>
</table>

### 返回值说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>ccl_buffer_size</td>
        <td>int</td>
        <td>必选</td>
        <td>计算得到的ccl_buffer_size大小，单位为MB。</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

### 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式调用。
- get_mega_moe_ccl_buffer_size、mega_moe必须配套使用。
- 本接口公式中的“/”表示整除。

### 确定性计算

默认支持确定性计算。

### 调用说明

- 单算子模式调用：

  ```python
  import os
  import torch
  import torch_npu
  from cann_ops_transformer.ops import get_mega_moe_ccl_buffer_size

  server_num = 1
  rank_per_dev = 2
  world_size = server_num * rank_per_dev
  num_experts = 8  # moe专家数
  BS = 256  # token数量
  H = 4096  # 每个token的长度
  topK = 6

  # 计算mega_moe算子所需的HCCL通信buffer_size大小（单位：MB）
  buffer_size = get_mega_moe_ccl_buffer_size(world_size, num_experts, BS, topK, H)
  os.environ['HCCL_BUFFSIZE'] = f'buffer_size'
  print(f"buffer_size is {buffer_size}")
  print("run ccl_buffer_size success.")
  ```

## get_symm_buffer_for_mega_moe接口说明

### 功能说明

- 接口功能：

    需与[mega_moe](#mega_moe接口说明)配套使用，用于封装输入参数并创建SymmBuffer结构体，生成`context`、`ep_world_size`和`ccl_buffer_size`等mega_moe算子运行所需信息。

### 函数原型

```python
cann_ops_transformer.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, *, max_recv_token_num=0, dispatch_quant_mode=0, dispatch_quant_out_dtype=None, combine_quant_mode=0, comm_alg="") -> SymmBuffer
```

### 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>group</td>
        <td>str</td>
        <td>必选</td>
        <td>EP通信域名称（专家并行通信域）。</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_experts</td>
        <td>int</td>
        <td>必选</td>
        <td>MoE模型的总专家数量。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_tokens_per_rank</td>
        <td>int</td>
        <td>必选</td>
        <td>每张卡上的token数量，当每个rank的BS不同时，为最大的BS大小。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_topk</td>
        <td>int</td>
        <td>必选</td>
        <td>每个token发送的专家数。预留参数，暂不支持。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>hidden</td>
        <td>int</td>
        <td>必选</td>
        <td>每个token大小。预留参数，暂不支持。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>intermediate_hidden</td>
        <td>int</td>
        <td>必选</td>
        <td>中间层投影维度。预留参数，暂不支持。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>max_recv_token_num</td>
        <td>int</td>
        <td>可选</td>
        <td>每个Rank最大可接收Token数，默认值为0表示自动计算。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dispatch_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>dispatch通信时量化模式，目前仅支持4（MX模式）。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dispatch_quant_out_dtype</td>
        <td>Optional[int]</td>
        <td>可选</td>
        <td>dispatch量化后输出的数据类型，支持输入23（torch.float8_e5m2）、24（torch.float8_e4m3fn）。默认值为None。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>combine_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>暂不支持该参数，使用默认值即可。默认值为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>可选</td>
        <td>暂不支持该参数，使用默认值即可。默认值为""。</td>
        <td>str</td>
        <td>-</td>
    </tr>
</tbody>
</table>

### 返回值说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>sym_buffer</td>
        <td>SymmBuffer</td>
        <td>必选</td>
        <td>用于封装输入参数并生成`context`、`ep_world_size`和`ccl_buffer_size`。</td>
        <td>SymmBuffer</td>
        <td>-</td>
    </tr>
</tbody>
</table>

### 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式调用。
- get_symm_buffer_for_mega_moe、mega_moe必须配套使用。

### 确定性计算

默认支持确定性计算。

### 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import torch.distributed as dist
  from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe

  num_experts = 8  # moe专家数
  H = 4096  # 每个token的长度
  topK = 6

  # 构造SymmBuffer，传入EP通信域ep_group及各项配置
  sym_buffer = get_symm_buffer_for_mega_moe(
      ep_group, num_experts=num_experts,
      num_max_tokens_per_rank=0, num_topk=topK,
      hidden=H, intermediate_hidden=0,
      dispatch_quant_mode=4, dispatch_quant_out_dtype=23
  )
  # 后续将sym_buffer传入mega_moe接口使用
  ```
