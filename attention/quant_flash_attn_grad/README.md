# QuantFlashAttnGrad

## 当前 TND 实现范围

- 支持 `layout_q = layout_kv = TND`、`mask_mode = 0/3/4`、`quant_mode = 0`、`N1 = N2`、`D = Dv = 128`、BF16 out/梯度，采用确定性 swizzle 分核。
- Q、dO、out 和 dQ 为 `[Tq, N, D]`；K、V、dK 和 dV 为 `[Tkv, N, D]`；softmax_lse 为 `[N, Tq]`。
- `cu_seqlens_q` 和 `cu_seqlens_kv` 必须为 int32 的 `[B + 1]` tensor，首项为 0，严格递增，末项分别等于 Tq 和 Tkv。本阶段不支持空序列和 sinks。mask 0 不传 attn_mask；mask 3/4 传入标准 `[2048, 2048]` int8 压缩上三角 mask。
- 调用 `quant_flash_attn_metadata` 时传入相同的 cu_seqlens、seqused_q/kv（包括未传侧）、布局、head 数、mask_mode 和窗口，并启用 `is_grad_enabled=True`。metadata 在 device 侧计算各 batch 实际长度、轮次前缀和 swizzle 参数；host 不读取 cu_seqlens 内容。
- mask 3/4 按每个 batch 的局部长度右下对齐：中心列为 `i + Lkv - Lq`。mask 3 保留中心列及左侧；mask 4 使用 `win_left/win_right`，某侧为 `-1` 表示该侧不限窗；mask 0/3 的两侧窗口必须为 `-1`。
- TND metadata 反向行使用 ABI v2，最小容量为 `17 + 9 * B`；生成核数必须与主 kernel 一致。
- TND dense 使用 metadata v2 的扩展调度数组。调度行数为 `max(ceil(Sq / 512), min(AIC 核数, ceil(Skv / 512)))`，实际行列数仍独立保存，虚拟行不参与 attention 计算。
- 可传入真实最大序列长度作为 `max_seqlen_q/kv`，以减少 kernel 的安全循环上界；省略时为 -1，host 使用对应总 token 数作为上界。

- TND 可传单侧或双侧 `seqused_q/kv`：int32 `[B]`，每项满足 `0 < used[b] <= cu[b+1] - cu[b]`。cu 决定物理分段与 GM 起点，used 决定有效前缀及 mask 对齐；未传的一侧使用完整物理段。输出保持物理形状，无效后缀梯度为 0。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | ------ |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：计算量化注意力（QuantFlashAttention）的反向梯度。该算子为`aclnnQuantFlashAttention`正向算子的配套反向接口，用于计算Query、Key、Value的梯度（dq、dk、dv）以及sink梯度（dsink）。支持HIFLOAT8量化数据类型，支持BSND、BNSD、TND三种数据排布格式，支持因果mask（Causal）、滑窗mask（Band）等多种mask模式。

- 计算公式：

    量化注意力反向梯度计算分为以下几个阶段：

    阶段一：反量化输入数据，将量化输入转换到高精度浮点域

    $$
    Q_{fp} = Q_{quant} \times q\_descale
    $$

    $$
    K_{fp} = K_{quant} \times k\_descale
    $$

    $$
    V_{fp} = V_{quant} \times v\_descale
    $$

    $$
    dO_{fp} = dO_{quant} \times do\_descale
    $$

    阶段二：计算Softmax梯度（PreSfmg阶段）

    $$
    dP = dO \times V
    $$

    $$
    dS = dP \times p\_scale
    $$

    $$
    dS_{grad} = (dS - ds\_scale \times softmax(lse)) \times softmaxLse
    $$

    阶段三：计算Key/Value梯度（Cube主计算阶段）

    $$
    dV = dS_{grad}^{T} \times Q
    $$

    $$
    dK = dS_{grad}^{T} \times Q
    $$

    $$
    dQ = dS_{grad} \times K
    $$

    阶段四：后处理（Post阶段），将中间结果转换输出dtype

    $$
    dq = dQ \times softmax\_scale
    $$

    $$
    dk = dK \times softmax\_scale
    $$

    $$
    dv = dV \times softmax\_scale
    $$

    $$
    dsink = reduce(dS_{grad}, dim=S)
    $$

    其中，$q\_descale$、$k\_descale$、$v\_descale$、$do\_descale$为量化缩放因子，$p\_scale$为P矩阵的缩放因子，$ds\_scale$为反量化缩放因子，$softmax\_scale$为缩放系数（建议值为$\sqrt{D}$的倒数）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| q | 输入 | attention结构的输入Q，量化数据。 | HIFLOAT8 | ND |
| k | 输入 | attention结构的输入K，量化数据。 | HIFLOAT8 | ND |
| v | 输入 | attention结构的输入V，量化数据。 | HIFLOAT8 | ND |
| dout | 输入 | 正向输出attn_out对应的梯度，量化数据。 | HIFLOAT8 | ND |
| attn_out | 输入 | 正向计算输出的attn_out。 | BF16 | ND |
| q_descale | 输入 | Query的反量化缩放因子，每个量化块对应一个缩放因子。 | FLOAT | ND |
| k_descale | 输入 | Key的反量化缩放因子，每个量化块对应一个缩放因子。 | FLOAT | ND |
| v_descale | 输入 | Value的反量化缩放因子，shape需与k_descale一致。 | FLOAT | ND |
| do_descale | 输入 | dout的反量化缩放因子，shape需与q_descale一致。 | FLOAT | ND |
| p_scale | 输入 | P矩阵的量化缩放因子。 | FLOAT | ND |
| ds_scale | 输入 | 反量化缩放因子。 | FLOAT | ND |
| softmax_lse | 输入 | 注意力正向计算的输出softmaxLse。 | FLOAT | ND |
| cu_seqlens_q | 可选输入 | 每个Batch中Query的有效token数的累加和形式。 | INT32 | ND |
| cu_seqlens_kv | 可选输入 | 每个Batch中Key的有效token数的累加和形式。 | INT32 | ND |
| seqused_q | 可选输入 | 表示不同batch中q实际参与运算的token数。 | INT32 | ND |
| seqused_kv | 可选输入 | 表示不同batch中k实际参与运算的token数。 | INT32 | ND |
| sinks | 可选输入 | sink场景下的输入tensor，支持空tensor。 | FLOAT | ND |
| attn_mask | 可选输入 | 注意力mask，表示q和k计算的mask模式。支持空tensor。 | INT8 | ND |
| metadata | 可选输入 | 表示tiling下沉的aicpu算子输出结果。 | INT32 | ND |
| dq | 输出 | Query的梯度，dtype固定为BF16，shape与q一致。 | BF16 | ND |
| dk | 输出 | Key的梯度，dtype固定为BF16，shape与k一致。 | BF16 | ND |
| dv | 输出 | Value的梯度，dtype固定为BF16，shape与v一致。 | BF16 | ND |
| dsink | 输出 | sink的梯度，dtype固定为FLOAT，shape为(N1,)。 | FLOAT | ND |
| quant_mode | 必选属性 | 量化模式。0：UINT8量化。 | INT64 | - |
| softmax_scale | 可选属性 | 缩放系数，默认值为1.0。推荐值：sqrt(head_dim)的倒数。 | FLOAT | - |
| mask_mode | 可选属性 | 表示q和k计算的mask模式。0：No mask；3：rightDownCausal；4：band。默认值为0。 | INT64 | - |
| win_left | 可选属性 | 滑窗mask的左窗口大小，mask_mode为band模式时有效，-1表示不使用。默认值为-1。 | INT64 | - |
| win_right | 可选属性 | 滑窗mask的右窗口大小，mask_mode为band模式时有效，-1表示不使用。默认值为-1。 | INT64 | - |
| max_seqlen_q | 可选属性 | Query的最大序列长度，-1表示自动推导。默认值为-1。 | INT64 | - |
| max_seqlen_kv | 可选属性 | Key/Value的最大序列长度，-1表示自动推导。默认值为-1。 | INT64 | - |
| layout_q | 可选属性 | 表示输入q的数据排布格式，支持"BSND"、"BNSD"、"TND"，默认值为"BSND"。 | STRING | - |
| layout_kv | 可选属性 | 表示输入k/v的数据排布格式，支持"BSND"、"BNSD"、"TND"，默认值为"BSND"。 | STRING | - |

## 约束说明

- 确定性说明：aclnnQuantFlashAttnGrad默认确定性实现。
- 支持BSND、BNSD或TND layout，且layout_q与layout_kv必须保持一致。
- 关于数据shape的约束：
    - B：泛化支持。
    - S1、S2：泛化支持，支持S1、S2不等长。
    - N1：支持1~128，且num_heads_q必须能被num_heads_k整除（GQA约束，N1 = N2 × G）。
    - N2：支持1~128。
    - D：量化场景下固定为128。
- 量化场景下，q、k、v、dout、attn_out的维度必须一致，k和v的shape必须一致。
- q_descale和do_descale的shape必须一致，k_descale和v_descale的shape必须一致。
- mask_mode支持：
    - 0：不做mask操作（ALL_MASK）。
    - 3：rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。
    - 4：band模式的mask，对应滑窗场景，需配合win_left和win_right属性使用。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| PyTorch API | - | 通过[quant_flash_attn_grad](../../attention/quant_flash_attn_grad/docs/torchapi_quant_flash_attn_grad.md)接口调用算子。 |

## 参考资源（可选）

无
