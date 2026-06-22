# mega\_moe

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>            |    √    |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

Mega MoE 将 EP Dispatch、Linear1、SwiGLU、Linear2 和 EP Combine 融合为单个算子，实现计算和通信的掩盖。

- API 功能：

    [mega\_moe](#mega_moe接口说明)与 [get\_symm\_buffer\_for\_mega\_moe](#get_symm_buffer_for_mega_moe接口说明) 配套使用，该算子将 MoE 层的专家 FFN 的完整计算流程及前后数据通信（即 Dispatch + Linear1 + SwiGLU + Linear2 + Combine）融合为单个算子，实现了通信和计算的掩盖。

- 计算公式：
    - 符号约定：
        - **维度符号**：$n$（总 token 数，动态变量），$H$（隐藏层维度，静态超参数），$K$（每个 token 选择的专家数量，静态超参数），$E$（专家总数，静态超参数），$D$（中间层维度，静态超参数）。
        - **记法约定**：矩阵用斜体粗体大写字母（如 $\boldsymbol{X}$），向量用斜体粗体小写字母（如 $\boldsymbol{x}$），标量用斜体非粗体小写字母（如 $s$）。
        - **输入/输出张量与算子入参的对应关系**：

            | 符号 | 形状 | 含义 | 算子入参 |
            | :---: | :---: | :---: | :---: |
            | $\boldsymbol{X}$ | $\mathbb{R}^{n \times H}$ | 激活矩阵 | `x` |
            | $\boldsymbol{I}$ | $\mathbb{Z}^{n \times K}$ | token 选择的专家编号 | `topk_idx` |
            | $\boldsymbol{G}$ | $\mathbb{R}^{n \times K}$ | token 选择的专家的门控权重 | `topk_weights` |
            | $\boldsymbol{W}_1$ | $\mathbb{R}^{E \times H \times 2D}$ | Linear1 的权重矩阵 | `l1_weights` |
            | $\boldsymbol{W}_2$ | $\mathbb{R}^{E \times D \times H}$ | Linear2 的权重矩阵 | `l2_weights` |
            | $\boldsymbol{Y}$ | $\mathbb{R}^{n \times H}$ | 最终输出矩阵 | `y_out` |

    - 输入：
        - $\boldsymbol{X} \in \mathbb{R}^{n \times H}$：激活矩阵，$n$ 是总 token 数，$H$ 是隐藏层维度。
        - $\boldsymbol{I} \in \mathbb{Z}^{n \times K}$：token 选择的专家编号，$K$ 是每个 token 选择的专家数量。
        - $\boldsymbol{G} \in \mathbb{R}^{n \times K}$：token 选择的专家的门控权重。
        - $\boldsymbol{W}_1 \in \mathbb{R}^{E \times H \times 2D}$：Linear1 的权重矩阵，$E$ 是专家数量，$D$ 是中间层维度。
        - $\boldsymbol{W}_2 \in \mathbb{R}^{E \times D \times H}$：Linear2 的权重矩阵。
    - 输出：
        - $\boldsymbol{Y}$：最终输出矩阵。
    - 约定：
        - $⋅$ 表示矩阵乘法，$⊙$ 表示逐元素乘法。
        - $\left \lfloor z\right \rceil$ 表示将 $z$ 四舍五入到最近的整数，$\left \lfloor z\right \rfloor$ 表示将 $z$ 向下取整。
        - 全体 token 的集合为 $\{ \text{token}_i \mid i \in \{0, 1, \dots, n - 1\} \}$。
        - $\text{token}_i$ 的 token 表示（即隐藏状态向量）为 $\boldsymbol{x}_i \in \mathbb{R}^{1 \times H}$，且 $\boldsymbol{x}_i = \boldsymbol{X}[i,:]$。
        - $\text{token}_i$ 的专家索引为 $e_{i,k} = I_{i,k},\quad k \in \{0,\dots,K - 1\},\quad e_{i,k} \in \{0,\dots,E - 1\}$。
        - 激活函数 $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1+e^{-z}}$
        - $\mathbb{Z_4} = \{ x \in \mathbb{Z} \mid -8 \le x \le 7 \}, \quad \mathbb{Z_8} = \{ x \in \mathbb{Z} \mid -127 \le x \le 127 \}$。
        - 张量切片操作采用 Python 风格的 `start:stop:step` 表示法，例如，$[0::2, :]$ 代表取偶数行、$[1::2, :]$ 代表取奇数行。
    - 计算过程

        各产品支持的**Linear计算时**的激活矩阵 A 和权重矩阵 W 的数据类型如下：
        | 场景名    |  A |  W   |<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | <term>Ascend 950PR/Ascend 950DT</term> |
        | ---      | :---:                      | :---:                    | :---: | :---: | :---: |
        | A16W16   | BFLOAT16                   |BFLOAT16                  | √ | √ | - |
        | A8W8-INT | INT8                       | INT8                     | √ | √ | - |
        | A8W8-FP  | FLOAT8_E4M3FN、FLOAT8_E5M2 |FLOAT8_E4M3FN、FLOAT8_E5M2 | - | - | √ |
        | A8W4-INT | INT8                       | INT4                     | √ | √ | - |
        <details>
        <summary> A16W16 非量化场景</summary>

        - **EP Dispatch**

            在 Dispatch 阶段，每个 $\text{token}_i$ 将其token表示 $\boldsymbol{x}_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,K-1}$。即对于每个 $k$，专家 $e_{i,k}$ 接收一份 $\boldsymbol{x}_i$。

            记 $I_e = \{\, i \mid \exists k,\ I_{i,k}=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $m_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数，则有 $\boldsymbol{X}_e \in \mathbb{R}^{m_e \times H}$ 是由所有满足 $i \in I_e$ 的token表示 $\boldsymbol{x}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示。

            对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,m_{e_{i,k}}-1\}$，使得 $\boldsymbol{X}_{e_{i,k}}[\operatorname{row}(i,k), :] = \boldsymbol{x}_i$。该映射记录了 $\boldsymbol{x}_i$ 在专家 $e_{i,k}$ 的输入矩阵 $\boldsymbol{X}_{e_{i,k}}$ 中的位置。

        - **Expert Compute**

            在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。整个计算过程分为如下三个子步骤。

            **1. Linear1 投影**

            Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。其计算公式为
            $$
            \boldsymbol{Z}_e = \boldsymbol{X}_e \cdot \boldsymbol{W}_{1,e}
            \quad\bigl(\in \mathbb{R}^{m_e \times 2D}\bigr)
            $$
            **2. SwiGLU 激活**

            首先将 $\boldsymbol{Z}_e$ 沿列维度拆分为 **gate 部分** $\boldsymbol{Z}_{\text{gate}}$ 和 **up 部分** $\boldsymbol{Z}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\boldsymbol{A}_e$。
            $$
            \boldsymbol{Z}_{\text{gate}} = \boldsymbol{Z}_e\bigl[:,\; :D\bigr], \qquad
            \boldsymbol{Z}_{\text{up}} = \boldsymbol{Z}_e\bigl[:,\; D:\bigr]
            $$
            $$
            \boldsymbol{A}_e = \text{SiLU}\bigl(\boldsymbol{Z}_{\text{gate}}\bigr) \;\odot\; \boldsymbol{Z}_{\text{up}}
            \quad\bigl(\in \mathbb{R}^{m_e \times D}\bigr)
            $$

            **3. Linear2 投影**

            Linear2 投影作为第二层线性变换，将中间激活表示 $\boldsymbol{A}_e$ 从高维空间投影回原始的隐藏维度 $H$，使专家输出能够与残差连接等后续操作兼容。
            $$
            \boldsymbol{Y}_e = \boldsymbol{A}_e \cdot \boldsymbol{W}_{2,e}
            \quad\bigl(\in \mathbb{R}^{m_e \times H}\bigr)
            $$

            经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\boldsymbol{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。
        - **Token Combine**

            Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：

            $$
            \boldsymbol{y}_i = \sum_{k=0}^{K - 1} w_k \;\cdot\;
            \boldsymbol{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
            \qquad\bigl(\in \mathbb{R}^{1 \times H}\bigr)
            $$

            其中 $w_k = G_{i,k}$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

            所有 token 的输出按输入顺序堆叠为最终输出 $\boldsymbol{Y} \in \mathbb{R}^{n \times H}$。
        </details>

        <details>
        <summary> A8W8-INT 量化场景</summary>

        - **输入**
            - $\boldsymbol{S}_1 \in \mathbb{R}^{E \times 2D}$：Linear1 权重矩阵的逐通道缩放因子（对应 `l1_weights_sf`）
            - $\boldsymbol{S}_2 \in \mathbb{R}^{E \times H}$：Linear2 权重矩阵的逐通道缩放因子（对应 `l2_weights_sf`）

        - **EP Dispatch**

            在 Dispatch 通信之前，首先将原始 BF16 激活矩阵 $\boldsymbol{X}$ 量化为 INT8。对每个 $\text{token}_i$，计算其逐 token 缩放因子：
            $$
            s_i = \frac{\max(|\boldsymbol{X}[i,:]|)}{127} \in \mathbb{R}
            $$
            然后量化得到 INT8 表示：
            $$
            \boldsymbol{q}_i = \left\lfloor \frac{\boldsymbol{X}[i,:]}{s_i} \right\rceil \quad \in \mathbb{Z}_8^{1 \times H}
            $$

            在 Dispatch 通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\boldsymbol{q}_i$ 和缩放因子 $s_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,K-1}$（$e_{i,k} = I_{i,k}$）。

            记 $I_e = \{\, i \mid \exists k,\ I_{i,k}=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $m_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数。则有 $\boldsymbol{Q}_e \in \mathbb{Z}_8^{m_e \times H}$ 是由所有满足 $i \in I_e$ 的 $\boldsymbol{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\boldsymbol{s}_e \in \mathbb{R}^{m_e}$，其元素由所有满足 $i \in I_e$ 的 $s_i$ 按与 $\boldsymbol{Q}_e$ 相同的行顺序堆叠而成。

            对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,m_{e_{i,k}}-1\}$，使得 $\boldsymbol{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \boldsymbol{q}_i$。该映射记录了 $\boldsymbol{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

        - **Expert Compute**

            在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 SwiGLU 结构以提升表达能力。在A8W8场景下，两个线性层都使用 INT8 输入和 INT8 权重进行矩阵乘，得到 INT32 中间结果并反量化。具体分为三个子步骤。

            **1. Linear1 投影（INT8 矩阵乘 + 反量化）**

            Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。计算时执行 INT8 矩阵乘法，得到 INT32 计算结果：
            $$
            \boldsymbol{C}_e^{\text{int32}} = \boldsymbol{Q}_e \cdot \boldsymbol{W}_{1,e}^{\text{int8}} \quad \in \mathbb{Z}_{32}^{m_e \times 2D}
            $$
            然后反量化为预激活值 $\boldsymbol{Z}_e$：
            $$
            \boldsymbol{Z}_e = \left( \boldsymbol{C}_e^{\text{int32}} \odot \boldsymbol{S}_{1,e} \right) \odot \boldsymbol{s}_e \quad \in \mathbb{R}^{m_e \times 2D}
            $$

            **2. SwiGLU 激活**

            首先将 $\boldsymbol{Z}_e$ 沿列维度拆分为 **gate 部分** $\boldsymbol{Z}_{\text{gate}}$ 和 **up 部分** $\boldsymbol{Z}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\boldsymbol{A}_e$。
            $$
            \boldsymbol{Z}_{\text{gate}} = \boldsymbol{Z}_e[:, :D], \qquad
            \boldsymbol{Z}_{\text{up}} = \boldsymbol{Z}_e[:, D:]
            $$
            $$
            \boldsymbol{A}_e = \operatorname{SiLU}(\boldsymbol{Z}_{\text{gate}}) \odot \boldsymbol{Z}_{\text{up}}
            $$

            **3. Linear2 投影（量化 + INT8 矩阵乘 + 反量化）**

            Linear2 投影作为第二层线性变换，将中间激活表示 $\boldsymbol{A}_e$ 从高维空间投影回原始的隐藏维度 $H$，使专家输出能够与残差连接等后续操作兼容。
            在 A8W8 场景下，需要将激活值量化为 INT8，因此先对 $\boldsymbol{A}_e$ 的每一行（每个 token）计算缩放因子：
            $$
            t_i = \frac{\max(|\boldsymbol{A}_e[i,:]|)}{127}, \quad i=0,\dots,m_e-1
            $$
            得到专家 $e$ 在 Linear2 计算时的激活缩放因子 $\boldsymbol{t}_e \in \mathbb{R}^{m_e}$。然后量化：
            $$
            \boldsymbol{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\boldsymbol{A}_e[i,:]}{t_i} \right\rceil \quad \in \mathbb{Z}_8^{m_e \times D}
            $$
            再执行 INT8 矩阵乘法并反量化：
            $$
            \boldsymbol{D}_e^{\text{int32}} = \boldsymbol{A}_e^{\text{int8}} \cdot \boldsymbol{W}_{2,e}^{\text{int8}} \quad \in \mathbb{Z}_{32}^{m_e \times H}
            $$
            $$
            \boldsymbol{Y}_e = \left( \boldsymbol{D}_e^{\text{int32}} \odot \boldsymbol{S}_{2,e} \right) \odot \boldsymbol{t}_e \quad \in \mathbb{R}^{m_e \times H}
            $$

            经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\boldsymbol{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

        - **Token Combine**

            Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：
            $$
            \boldsymbol{y}_i = \sum_{k=0}^{K - 1} w_k \;\cdot\; \boldsymbol{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr] \quad \in \mathbb{R}^{1 \times H}
            $$
            其中 $w_k = G_{i,k}$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

            所有 token 的输出按输入顺序堆叠为最终输出 $\boldsymbol{Y} \in \mathbb{R}^{n \times H}$。

        </details>

        <details>
        <summary> A8W4-INT 量化场景</summary>

        - **输入**
            - $\boldsymbol{S}_1 \in \mathbb{R}^{E \times 2D}$：Linear1 权重矩阵的逐通道缩放因子（对应 `l1_weights_sf`）
            - $\boldsymbol{S}_2 \in \mathbb{R}^{E \times H}$：Linear2 权重矩阵的逐通道缩放因子（对应 `l2_weights_sf`）
            - $\boldsymbol{B}_1 \in \mathbb{R}^{E \times 2D}$：Linear1 的偏置矩阵，由 INT4 量化过程离线生成（对应 `l1_bias`）
            - $\boldsymbol{B}_2 \in \mathbb{R}^{E \times H}$：Linear2 的偏置矩阵，由 INT4 量化过程离线生成（对应 `l2_bias`）

        - **EP Dispatch**

            在 Dispatch 通信之前，首先将原始 BF16 激活矩阵 $\boldsymbol{X}$ 量化为 INT8。对每个 $\text{token}_i$，计算其逐 token 缩放因子：
            $$
            s_i = \frac{\max(|\boldsymbol{X}[i,:]|)}{127} \in \mathbb{R}
            $$
            然后量化得到 INT8 表示：
            $$
            \boldsymbol{q}_i = \left\lfloor \frac{\boldsymbol{X}[i,:]}{s_i} \right\rceil \quad \in \mathbb{Z}_8^{1 \times H}
            $$

            在 Dispatch 通信阶段，每个 $\text{token}_i$ 将其量化后的向量 $\boldsymbol{q}_i$ 和缩放因子 $s_i$ 发送给专家 $e_{i,0}, e_{i,1}, \dots, e_{i,K-1}$

            记 $I_e = \{\, i \mid \exists k,\ I_{i,k}=e \,\}$ 为所有被分派给专家 $e$ 的 token 索引集合，集合 $I_e$ 的大小 $m_e = |I_e|$ 即为专家 $e$ 需要处理的 token 总数。则有 $\boldsymbol{Q}_e \in \mathbb{Z}_8^{m_e \times H}$ 是由所有满足 $i \in I_e$ 的 $\boldsymbol{q}_i$ **按任意固定顺序行堆叠**而成的矩阵，该矩阵即为专家 $e$ 经过 Dispatch 后收到的全部 token 表示；同理，对应的专家 $e$ 收到的缩放因子向量记为 $\boldsymbol{s}_e \in \mathbb{R}^{m_e}$，其元素由所有满足 $i \in I_e$ 的 $s_i$ 按与 $\boldsymbol{Q}_e$ 相同的行顺序堆叠而成。

            对于每个 $\text{token}_i$ 及其选中的第 $k$ 个专家 $e_{i,k}$，存在唯一的行索引 $\operatorname{row}(i,k) \in \{0,\dots,m_{e_{i,k}}-1\}$，使得 $\boldsymbol{Q}_{e_{i,k}}[\operatorname{row}(i,k), :] = \boldsymbol{q}_i$。该映射记录了 $\boldsymbol{q}_i$ 在专家 $e_{i,k}$ 的输入矩阵中的位置。

        - **Expert Compute**

            在 MoE 层中，每个专家本质上是一个独立的前馈网络（FFN），采用 **SwiGLU** 结构以提升表达能力。在A8W4-INT场景下，两个线性层都采用MSD（Mixed-precision Split-activation Decomposition，混合精度激活拆分分解）方案进行矩阵乘，通过将 INT8 值拆为高4位和低4位两个有符号 INT4，使得 INT8×INT4 的矩阵乘可分解为两个 INT4×INT4 的矩阵乘，从而利用硬件的 INT4 矩阵乘加速。该方案的数学原理和实现逻辑可参阅[GroupedMatmul W4A8量化与MSD方案](https://gitcode.com/cann/ops-transformer/wiki/GMM--GroupedMatmul%E9%87%8F%E5%8C%96%E6%9E%81%E8%87%B4%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96-%E6%8E%A8%E7%90%86%E6%8F%90%E5%8D%87%E7%99%BE%E5%88%86%E4%B9%8B%E4%B8%89%E5%8D%81)。

            **1. 生成精度补偿的偏置矩阵（离线生成，在算子外完成，并作为算子输入）**

            MSD 方案的核心是将量化后的 INT8 激活值二进制重解释地拆分为两个 INT4 分量。由于低位 INT4 分量按 $(\boldsymbol{X}^{\text{int8}} \mathbin{\&} 0x0F) - 8$ 定义，其数值范围被映射到 $[-8, 7]$，这相当于在原始无符号低 4 位（0~15）的基础上减去了 8。当将高、低位的 INT4 分别与权重矩阵做矩阵乘并合并时，该偏移会引入一个常数项，需要在最终结果中予以补偿。

            记原始 INT8 激活矩阵为 $\boldsymbol{X}^{\text{int8}}$，拆分后的高位和低位 INT4 激活矩阵分别为 $\boldsymbol{X}_1^{\text{int4}}$ 和 $\boldsymbol{X}_2^{\text{int4}}$，权重矩阵为 $\boldsymbol{W}$。恢复关系为：

            $$
            \boldsymbol{X}^{\text{int8}} = 16 \times \boldsymbol{X}_1^{\text{int4}} + (\boldsymbol{X}_2^{\text{int4}} + 8 \cdot \boldsymbol{1}_{\text{mat}}),
            $$

            其中 $\boldsymbol{1}_{\text{mat}}$ 为与 $\boldsymbol{X}_2^{\text{int4}}$ 形状相同的全 1 矩阵，用于逐元素加 8。令 $\boldsymbol{1}$ 为形状与输入特征维度一致的全 1 列向量，则矩阵乘展开为：

            $$
            \boldsymbol{X}^{\text{int8}} \cdot \boldsymbol{W} = 16 \cdot (\boldsymbol{X}_1^{\text{int4}} \cdot \boldsymbol{W}) + (\boldsymbol{X}_2^{\text{int4}} \cdot \boldsymbol{W}) + 8 \cdot (\boldsymbol{1}^\top \cdot \boldsymbol{W}).
            $$

            这里 $\boldsymbol{1}^\top \cdot \boldsymbol{W}$ 表示对权重矩阵 $\boldsymbol{W}$ 的每一列求和，其结果为一个行向量（形状与输出维度相同）。在具体实现中，该行向量会被广播到批次维度，加到所有 token 的输出上。可见，若只计算前两项，会缺失一个正数项。为了精确恢复结果，需预先计算补偿偏置：

            $$
            \text{bias} = 8 \cdot (\boldsymbol{1}^\top \cdot \boldsymbol{W}),
            $$

            $\boldsymbol{B}_1$ 和 $\boldsymbol{B}_2$ 均按此法，分别使用对应的全 1 列向量（维度分别为 $H$ 和 $D$）与各自的权重矩阵相乘，得到形状匹配的偏置行向量。这些偏置在离线阶段计算完成后传入算子，在合并高低位结果时直接参与加法，从而消除偏移引入的误差。

            **2. Linear1 投影（INT4 × INT4 矩阵乘 + 反量化）**

            Linear1 投影是专家网络的第一层线性变换，同时产生 **gate 部分** 和 **up 部分** 所需的预激活值。

            **(1) 激活重解释为INT4**

            将 INT8 激活张量 $\boldsymbol{Q}_e^{\text{int8}} \in \mathbb{Z}_8^{m_e \times H}$ 二进制重解释为两个 INT4 交替拼接的视图：

            $$
            \boldsymbol{Q}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2m_e \times H}} \left( \boldsymbol{Q}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2m_e \times H}
            $$

            其中高位和低位分量直接由 $\boldsymbol{Q}_e^{\text{int4}}$ 的偶数行和奇数行给出：

            $$
            \boldsymbol{Q}_e^{\text{high}} = \boldsymbol{Q}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{m_e \times H}, \quad
            \boldsymbol{Q}_e^{\text{low}} = \boldsymbol{Q}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{m_e \times H}
            $$

            它们在数值上可以由原 INT8 经如下计算得到：

            $$
            \boldsymbol{Q}_e^{\text{high}} = \left\lfloor \frac{\boldsymbol{Q}_e^{\text{int8}}}{16} \right\rfloor, \quad
            \boldsymbol{Q}_e^{\text{low}} = (\boldsymbol{Q}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
            $$

            恢复关系为 $\boldsymbol{Q}_e^{\text{int8}} = 16\boldsymbol{Q}_e^{\text{high}} + (\boldsymbol{Q}_e^{\text{low}} + 8)$。由于 $\mathrm{bitcast}$ 仅改变类型视图，$\boldsymbol{Q}_e^{\text{int4}}$ 与 $\boldsymbol{Q}_e^{\text{int8}}$ 共享底层物理内存，无需任何数据重排或拷贝。

            **(2) INT4 × INT4 矩阵乘与权重反量化**

            将重解释后的 INT4 激活视图 $\boldsymbol{Q}_e^{\text{int4}} \in \mathbb{Z}_4^{2m_e \times H}$ 与权重矩阵 $\boldsymbol{W}_{1,e} \in \mathbb{R}^{H \times 2D}$ 执行矩阵乘，并应用权重缩放因子 $\boldsymbol{S}_{1,e}$ 进行反量化，得到结果：

            $$
            \boldsymbol{C}_e = \bigl( \boldsymbol{Q}_e^{\text{int4}} \cdot \boldsymbol{W}_{1,e} \bigr) \odot \boldsymbol{S}_{1,e}
            \;\in\; \mathbb{R}^{2m_e \times 2D}
            $$

            将结果按偶数行和奇数行分别记为如下的矩阵视图，它们即对应于$\boldsymbol{Q}_e^{\text{high}}$和$\boldsymbol{Q}_e^{\text{low}}$的计算结果：

            $$
            \boldsymbol{C}_e^{\text{high}} = \boldsymbol{C}_e[0::2, :] \in \mathbb{R}^{m_e \times 2D}, \qquad
            \boldsymbol{C}_e^{\text{low}}  = \boldsymbol{C}_e[1::2, :] \in \mathbb{R}^{m_e \times 2D}
            $$

            **(3) 精度补偿与激活反量化**

            利用偏置 $\boldsymbol{b}_{1,e}$ 和激活缩放因子 $\boldsymbol{s}_e$（行向量）分别进行精度补偿和激活反量化，最终预激活值为：

            $$
            \boldsymbol{Z}_e = \Bigl( 16 \cdot \boldsymbol{C}_e^{\text{high}} + \boldsymbol{C}_e^{\text{low}} + \boldsymbol{b}_{1,e} \Bigr) \odot \boldsymbol{s}_e \quad\in \mathbb{R}^{m_e \times 2D}
            $$

            其中 $\odot \boldsymbol{s}_e$ 表示将矩阵的每一行乘以 $\boldsymbol{s}_e$ 中对应的标量。

            **3. SwiGLU 激活**

            首先将 $\boldsymbol{Z}_e$ 沿列维度拆分为 **gate 部分** $\boldsymbol{Z}_{\text{gate}}$ 和 **up 部分** $\boldsymbol{Z}_{\text{up}}$，然后对 gate 部分应用 SiLU 激活函数，再与 up 部分逐元素相乘，得到专家 $e$ 的中间激活表示 $\boldsymbol{A}_e$。
            $$
            \boldsymbol{Z}_{\text{gate}} = \boldsymbol{Z}_e[:,\; :D], \qquad
            \boldsymbol{Z}_{\text{up}} = \boldsymbol{Z}_e[:,\; D:]
            $$
            $$
            \boldsymbol{A}_e = \operatorname{SiLU}(\boldsymbol{Z}_{\text{gate}}) \odot \boldsymbol{Z}_{\text{up}}
            \quad\in \mathbb{R}^{m_e \times D}
            $$

            **4. Linear2 投影（量化 + INT4 × INT4 矩阵乘 + 反量化）**

            Linear2 投影作为第二层线性变换，将中间激活表示 $\boldsymbol{A}_e$ 从高维空间投影回原始的隐藏维度 $H$，使专家输出能够与残差连接等后续操作兼容。
            在 A8W4 场景下，首先将激活值量化为 INT8，再通过二进制重解释为两个 INT4 的拼接视图，以一次矩阵乘完成计算，过程如下。

            **(1) 激活量化**

            对 $\boldsymbol{A}_e$ 的每一行计算缩放因子并量化至 INT8：
            $$
            t_i = \frac{\max(|\boldsymbol{A}_e[i,:]|)}{127}, \quad i=0,\dots,m_e-1
            $$
            得到专家 $e$ 在 Linear2 计算时的激活缩放因子 $\boldsymbol{t}_e \in \mathbb{R}^{m_e}$，并计算量化结果：
            $$
            \boldsymbol{A}_e^{\text{int8}}[i,:] = \left\lfloor \frac{\boldsymbol{A}_e[i,:]}{t_i} \right\rceil \quad \in \mathbb{Z}_8^{m_e \times D}
            $$

            **(2) 激活重解释为 INT4**

            将 INT8 激活张量 $\boldsymbol{A}_e^{\text{int8}} \in \mathbb{Z}_8^{m_e \times D}$ 二进制重解释为两个 INT4 交替拼接的视图：

            $$
            \boldsymbol{A}_e^{\text{int4}} = \mathrm{bitcast}_{\mathbb{Z}_4^{2m_e \times D}} \left( \boldsymbol{A}_e^{\text{int8}} \right) \in \mathbb{Z}_4^{2m_e \times D}
            $$

            其中高位和低位分量直接由 $\boldsymbol{A}_e^{\text{int4}}$ 的偶数行和奇数行给出：

            $$
            \boldsymbol{A}_e^{\text{high}} = \boldsymbol{A}_e^{\text{int4}}[0::2, :] \in \mathbb{Z}_4^{m_e \times D}, \quad
            \boldsymbol{A}_e^{\text{low}} = \boldsymbol{A}_e^{\text{int4}}[1::2, :] \in \mathbb{Z}_4^{m_e \times D}
            $$

            它们在数值上可以由原 INT8 经如下计算得到：

            $$
            \boldsymbol{A}_e^{\text{high}} = \left\lfloor \frac{\boldsymbol{A}_e^{\text{int8}}}{16} \right\rfloor, \quad
            \boldsymbol{A}_e^{\text{low}} = (\boldsymbol{A}_e^{\text{int8}} \mathbin{\&} 0x0F) - 8
            $$

            恢复关系为 $\boldsymbol{A}_e^{\text{int8}} = 16\boldsymbol{A}_e^{\text{high}} + (\boldsymbol{A}_e^{\text{low}} + 8)$。由于 $\mathrm{bitcast}$ 仅改变类型视图，$\boldsymbol{A}_e^{\text{int4}}$ 与 $\boldsymbol{A}_e^{\text{int8}}$ 共享底层物理内存，无需任何数据重排或拷贝。

            **(3) INT4 × INT4 矩阵乘与权重反量化**

            将重解释后的 INT4 激活视图 $\boldsymbol{A}_e^{\text{int4}} \in \mathbb{Z}_4^{2m_e \times D}$ 与权重矩阵 $\boldsymbol{W}_{2,e} \in \mathbb{R}^{D \times H}$ 执行矩阵乘，并应用权重缩放因子 $\boldsymbol{S}_{2,e}$ 进行反量化，得到结果：

            $$
            \boldsymbol{D}_e = \bigl( \boldsymbol{A}_e^{\text{int4}} \cdot \boldsymbol{W}_{2,e} \bigr) \odot \boldsymbol{S}_{2,e}
            \;\in\; \mathbb{R}^{2m_e \times H}
            $$

            将结果按偶数行和奇数行分别记为如下的矩阵视图，它们即对应于 $\boldsymbol{A}_e^{\text{high}}$ 和 $\boldsymbol{A}_e^{\text{low}}$ 的计算结果：

            $$
            \boldsymbol{D}_e^{\text{high}} = \boldsymbol{D}_e[0::2, :] \in \mathbb{R}^{m_e \times H}, \qquad
            \boldsymbol{D}_e^{\text{low}}  = \boldsymbol{D}_e[1::2, :] \in \mathbb{R}^{m_e \times H}
            $$

            **(4) 精度补偿与激活反量化**

            利用偏置 $\boldsymbol{b}_{2,e}$ 和激活缩放因子 $\boldsymbol{t}_e$（行向量）分别进行精度补偿和激活反量化，最终输出为：

            $$
            \boldsymbol{Y}_e = \Bigl( 16 \cdot \boldsymbol{D}_e^{\text{high}} + \boldsymbol{D}_e^{\text{low}} + \boldsymbol{b}_{2,e} \Bigr) \odot \boldsymbol{t}_e
            \quad\in \mathbb{R}^{m_e \times H}
            $$

            其中 $\odot \boldsymbol{t}_e$ 表示将矩阵的每一行乘以 $\boldsymbol{t}_e$ 中对应的标量。

            经过以上计算，专家 $e$ 的每一行输出对应其批次中的一个 token。对于 $\text{token}_i$，它在专家 $e_{i,k}$ 中的输出行即为 $\boldsymbol{Y}_{e_{i,k}}\bigl[\operatorname{row}(i,k),\,:\bigr]$。

        - **Token Combine**

            Combine 负责收集所有专家计算出的输出向量，按照每个 token 原先分配到的专家权重进行加权求和，最终为每个 token 生成一个融合后的输出。利用之前记录的位置索引 $\operatorname{row}(i,k)$，从专家 $e_{i,k}$ 的输出矩阵中收回属于 $\text{token}_i$ 的行，并与门控权重相乘后求和：
            $$
            \boldsymbol{y}_i = \sum_{k=0}^{K - 1} w_k \;\cdot\; \boldsymbol{Y}_{e_{i,k}}\!\bigl[\,\operatorname{row}(i,k),\,:\,\bigr]
            \qquad\bigl(\in \mathbb{R}^{1 \times H}\bigr)
            $$
            其中 $w_k = G_{i,k}$ 为 $\text{token}_i$ 对专家 $e_{i,k}$ 的门控权重。

            所有 token 的输出按输入顺序堆叠为最终输出 $\boldsymbol{Y} \in \mathbb{R}^{n \times H}$。

        </details>

        <details>
        <summary> A8W8-FP 量化场景</summary>

        第一阶段对输入 Token 按专家分组收集后做 MXFP8 量化，生成各专家的量化输入与缩放因子：

        $$
        \hat{\boldsymbol{X}}_e,\ \boldsymbol{S}_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(\boldsymbol{X}[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
        $$

        说明：根据 `topkIds` 将 Token 按专家排序收集，$\mathcal{T}_e$ 为分配到专家 $e$ 的 Token 索引集合，$E_{\text{local}}$ 表示当前 Rank 的专家数（即 $E / \text{ep\_world\_size}$），$\boldsymbol{X}[\mathcal{T}_e]$ 为对应的子矩阵。$\mathrm{Q}_{\text{MX}}$ 表示 MX 逐组量化（group size = 32），对每组 32 个元素提取共享指数后量化为 FP8 目标类型（FLOAT8_E5M2 或 FLOAT8_E4M3FN），同时输出 FLOAT8_E8M0 缩放因子。量化后的数据将作为 GMM1 的输入。

        第二阶段对每个专家执行 GMM1 矩阵乘法（将 $\boldsymbol{W}_1$ 沿列方向分为两半分别计算）、SwiGLU 激活和 MX 量化：

        $$
        \boldsymbol{Z}_e^{(x)} = \mathrm{DQ}_{\text{MX}}(\hat{\boldsymbol{X}}_e, \boldsymbol{S}_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(\boldsymbol{W}_{1,e}^{(x)}, \boldsymbol{S}_{1,e}^{(x)}), \quad \boldsymbol{Z}_e^{(y)} = \mathrm{DQ}_{\text{MX}}(\hat{\boldsymbol{X}}_e, \boldsymbol{S}_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(\boldsymbol{W}_{1,e}^{(y)}, \boldsymbol{S}_{1,e}^{(y)})
        $$

        $$
        \boldsymbol{A}_e = \boldsymbol{Z}_e^{(x)} \odot \sigma\!\left(\boldsymbol{Z}_e^{(x)}\right) \odot \boldsymbol{Z}_e^{(y)}
        $$

        $$
        \hat{\boldsymbol{A}}_e,\ \boldsymbol{S}_{A,e} = \mathrm{Q}_{\text{MX}}(\boldsymbol{A}_e)
        $$

        说明：将 $\boldsymbol{W}_1$ 的前 $D$ 列 $\boldsymbol{W}_{1,e}^{(x)}$ 和后 $D$ 列 $\boldsymbol{W}_{1,e}^{(y)}$ 分别与 MX 反量化后的输入做矩阵乘法，得到 Swish 分支 $\boldsymbol{Z}_e^{(x)}$ 和门控分支 $\boldsymbol{Z}_e^{(y)}$。SwiGLU 激活对两个分支做逐元素乘积 $x \cdot \sigma(x) \cdot y$，其中 $\sigma$ 为 Sigmoid 函数，将中间维度从 $2D$ 减半为 $D$。随后对 SwiGLU 输出做 MX 量化，得到 GMM2 的量化输入 $\hat{\boldsymbol{A}}_e$。

        第三阶段对每个专家执行 GMM2 矩阵乘法，并将结果按目标 Rank 分发：

        $$
        \boldsymbol{Y}_e = \mathrm{DQ}_{\text{MX}}(\hat{\boldsymbol{A}}_e, \boldsymbol{S}_{A,e}) \cdot \mathrm{DQ}_{\text{MX}}(\boldsymbol{W}_{2,e}, \boldsymbol{S}_{2,e})
        $$

        说明：将量化后的 SwiGLU 输出与第二组权重 $\boldsymbol{W}_2$ 做 MX 反量化后的矩阵乘法，将 $D$ 维中间表示映射回 $H$ 维隐藏空间，得到每个专家的输出 $\boldsymbol{Y}_e$。计算完成后通过 RDMA peermem 将结果按目标 Rank 的专家偏移地址写入远端，实现跨 Rank 聚合。

        第四阶段对所有 Token 按路由权重加权求和，恢复为与输入相同形状的输出：

        $$
        \boldsymbol{Y}[i] = \sum_{k=0}^{K-1} G_{i,k} \cdot \boldsymbol{Y}_{e_{i,k}}[\pi(i,\, k)]
        $$

        说明：对每个 Token $i$，根据排序后的路由索引 $\pi(i,k)$ 从聚合后的专家结果中取出对应行，按 `topkWeights` 中的权重逐元素加权累加，得到最终输出 $\boldsymbol{Y}$。

        其中，$\boldsymbol{X}$ 表示参数 `x`（对应算子入参 `x`），$\boldsymbol{G}$ 表示参数 `topkWeights`（对应算子入参 `topk_weights`），$\boldsymbol{W}_1$ 表示参数 `weight1`（对应算子入参 `l1_weights`），$\boldsymbol{W}_2$ 表示参数 `weight2`（对应算子入参 `l2_weights`），$\boldsymbol{Y}$ 表示参数 `y`（对应算子入参 `y_out`），$E_{\text{local}}$ 表示属性 `moeExpertNum / epWorldSize`（每个 Rank 的专家数），$K$ 表示 `topkIds` 的第二维度（top-K 值，取值 6 或 8）。

        局部变量说明：
        - $\mathcal{T}_e$：被路由到专家 $e$ 的 Token 索引集合，由 `topkIds` 排序后确定。
        - $\hat{\boldsymbol{X}}_e,\ \boldsymbol{S}_{X,e}$：专家 $e$ 的量化输入及其 MX 缩放因子，第一阶段中间结果。
        - $\boldsymbol{W}_{1,e}^{(x)}$、$\boldsymbol{W}_{1,e}^{(y)}$：$\boldsymbol{W}_1$ 对应专家 $e$ 的前 $D$ 列和后 $D$ 列子矩阵，由 `weight1` 按 SwiGLU 拆分推导。
        - $\boldsymbol{S}_{1,e}^{(x)}$、$\boldsymbol{S}_{1,e}^{(y)}$：$\boldsymbol{W}_{1,e}^{(x)}$ 和 $\boldsymbol{W}_{1,e}^{(y)}$ 对应的 MX 缩放因子，从 `weightScales1` 按维度截取。
        - $\boldsymbol{S}_{2,e}$：$\boldsymbol{W}_{2,e}$ 对应的 MX 缩放因子，来自参数 `weightScales2`。
        - $\boldsymbol{Z}_e^{(x)},\ \boldsymbol{Z}_e^{(y)}$：GMM1 的两路矩阵乘法输出（Swish 分支和门控分支），中间结果。
        - $\boldsymbol{A}_e$：SwiGLU 激活输出，维度 $m_e \times D$，中间结果。
        - $\hat{\boldsymbol{A}}_e,\ \boldsymbol{S}_{A,e}$：量化后的 SwiGLU 输出及其 MX 缩放因子，中间结果。
        - $\boldsymbol{Y}_e$：GMM2 的专家级输出，维度 $m_e \times H$，中间结果。
        - $\pi(i, k)$：Token $i$ 的第 $k$ 个 top-k 专家在展开排序后的行索引，由路由排序确定。
        - $\mathrm{Q}_{\text{MX}}(\cdot)$：MX 逐组量化操作，block size = 32，输出 FP8 数据和 E8M0 缩放因子。
        - $\mathrm{DQ}_{\text{MX}}(\cdot)$：MX 逐组反量化操作，在 matmul 内部隐式执行。
        </details>

## mega\_moe 接口说明

```
mega_moe(x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer, *, l1_weights_sf=None, l2_weights_sf=None, l1_bias=None, l2_bias=None, x_active_mask=None, activation="swiglu", activation_clamp=None, weight1_type=None, weight2_type=None) -> (Tensor, Tensor)
```

### 参数说明

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

### 输出说明

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

### 约束说明

- **参数一致性约束**：
    - 调用算子过程中使用的 `ep_world_size`、`globalBs`、`HCCL_BUFFSIZE` 等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域使用约束**：
    - 所有卡的 `moe_expert_num`、`ep_world_size`、`ccl_buffer_size`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_dtype`、`num_max_tokens_per_rank` 参数取值需保持一致。

#### Atlas A2 训练系列产品/Atlas A3 训练系列产品/Atlas A3 推理系列产品 约束

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

#### Ascend 950PR/Ascend 950DT 约束

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

- **MXFP 量化场景约束**：
    - weight1 shape 为 (expertPerRank, N, H)，weight2 shape 为 (expertPerRank, H, N/2)。
    - weightScales1 shape 为 (expertPerRank, N, CeilDiv(H, 64), 2)，其中 CeilDiv(H, 64) = (H + 63) / 64。
    - weightScales2 shape 为 (expertPerRank, H, CeilDiv(N/2, 64), 2)，其中 CeilDiv(N/2, 64) = (N/2 + 63) / 64。
    - weightScales1 的 dim3 和 weightScales2 的 dim3 必须等于 2。
    - MXFP 场景下，dispatchQuantOutDtype=23 时 weight1 和 weight2 必须为 FLOAT8_E5M2，dispatchQuantOutDtype=24 时必须为 FLOAT8_E4M3FN。
    - xActiveMask 必须为空指针。

## get\_symm\_buffer\_for\_mega\_moe 接口说明
```
get_symm_buffer_for_mega_moe(group, num_experts: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, intermediate_hidden: int, *, max_recv_token_num: int = 0, dispatch_quant_mode: int = 0, dispatch_quant_out_dtype: int = 28, combine_quant_mode: int = 0, comm_alg: str = "") -> SymmBuffer:
```
### 参数说明

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
   <td>EP 通信域名称（专家并行通信域）。</td>
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

### 输出说明

-   **sym\_buffer** (`SymmBuffer`)：用于封装输入参数并生成 `context`、`ep_world_size` 和 `ccl_buffer_size`。

## 调用示例
-   单算子模式调用

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
        # 构造distribute_buffer
        distribute_buffer = get_symm_buffer_for_mega_moe(
            ep_group, num_experts=num_experts,
            num_max_tokens_per_rank=0, num_topk=topK,
            hidden=H, intermediate_hidden=0,
            dispatch_quant_mode=4, dispatch_quant_out_dtype=23
        )
        # 运行mega_moe
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
