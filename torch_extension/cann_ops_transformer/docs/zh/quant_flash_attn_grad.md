# quant_flash_attn_grad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
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
<!-- end id6 -->

## 功能说明

- **接口功能**:

  `quant_flash_attn_grad`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`QuantFlashAttnGrad`算子完成`quant_flash_attn`（QuantFlashAttn全量化注意力）的反向梯度计算，训练推理归一化。算子接收前向的量化输入`q`、`k`、`v`、上游输出梯度`dout`、前向输出`attn_out`及对应的反量化scale / 量化参数，输出`q`、`k`、`v`的梯度`dq`、`dk`、`dv`以及sinks的梯度`dsink`。

  输入`q`、`k`、`v`、`dout`均为量化数据（以UINT8存储，具体量化格式由`quant_mode`决定），计算时通过`q_descale`、`k_descale`、`v_descale`、`do_descale`反量化后在FP32下进行梯度计算，输出为BF16/FP32。

- **计算公式**:

  前向计算为：

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  反向计算分为三个阶段：

  阶段一：计算相关性梯度dP与Softmax梯度dS

  $$
  dP = dout \cdot V^T
  $$

  $$
  dS = P \times (dP - SoftmaxGrad(dout,\ attn\_out))
  $$

  其中$P$为前向Softmax概率矩阵，$SoftmaxGrad$为基于`softmax_lse`与`attn_out`的Softmax反向修正项。

  阶段二：计算输入梯度dQ、dK、dV

  $$
  dQ = dS \cdot K \cdot scale
  $$

  $$
  dK = dS^T \cdot Q \cdot scale
  $$

  $$
  dV = P^T \cdot dout
  $$

  阶段三：计算sinks梯度dSinks

  $$
  dSinks = ReduceSum(-P \times dP, -1)
  $$

  > [!NOTE]
  >
  > 上述公式中的$Q$、$K$、$V$、$P$均为反量化后的实值，实际计算时先按`q_descale`、`k_descale`、`v_descale`完成反量化，再执行矩阵乘与Softmax梯度计算；**scale**为`softmax_scale`，即前向的$\frac{1}{\sqrt{d}}$或显式设置的缩放因子。

## 函数原型

```python
cann_ops_transformer.quant_flash_attn_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    q_descale,
    k_descale,
    v_descale,
    do_descale,
    p_scale,
    ds_scale,
    softmax_lse,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    metadata=None,
    quant_mode=0,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND"
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 枚举说明

`quant_mode` 与 `mask_mode` 在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.quant_flash_attn_grad`：

### quant_mode 枚举（QuantMode）

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `HIF8_PER_TENSOR` | 0 | Q/K/V/dout 采用 HIFloat8 per-tensor 量化（默认值） |
| `FP4_E2M1` | 1 | Q/K/V/dout 采用 FP4（E2M1）量化 |
| `HIF4` | 2 | Q/K/V/dout 采用 HIFloat4 量化 |

### mask_mode 枚举（MaskMode）

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `ALL` | 0 | 全计算模式（默认值） |
| `CAUSAL` | 3 | Causal 模式 |
| `WINDOW` | 4 | Sliding Window 模式 |

> [!NOTE]
>
> 枚举为 `IntEnum`，可直接作为 int 传入底层算子；接口仅支持传入枚举或对应 int 值。当前不支持 `mask_mode` = 4（`WINDOW`）。

## 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch，表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N / num_heads_q        |
|    KV_N    |    输入k/v tensor的头数，对应k/v shape中的N / num_heads_kv    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    KV_S    |  输入k/v tensor的序列长度，对应k/v shape中的S  |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|     KV_T     |          输入k/v tensor所有batch序列长度的累加和          |
|     D     |          输入q/k/v tensor以及输出隐藏层最小的单元尺寸headdim         |

## 参数说明

> **说明：**
>
> 参数维度含义：b（batch Size）表示输入样本批量大小、s表示输入样本序列长度、q_n表示q的多头数、kv_n表示k/v的多头数、d（head dim）表示注意力头的维度、q_t表示q的输入样本序列长度的累加和、kv_t表示k/v的输入样本序列长度的累加和。q/k/v/dout为量化输入，以uint8存储，具体量化格式由quant_mode决定。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Q（量化输入） | uint8（量化存储，格式由quant_mode决定） | ND | layout_q为BSND时shape为(B, Q_S, Q_N, D)；为BNSD时shape为(B, Q_N, Q_S, D)；为TND时shape为(Q_T, Q_N, D) | × |
| k | Tensor | 必选 | 公式中的K（量化输入） | uint8（量化存储，格式由quant_mode决定） | ND | layout_kv为BSND时shape为(B, KV_S, KV_N, D)；为TND时shape为(KV_T, KV_N, D) | × |
| v | Tensor | 必选 | 公式中的V（量化输入） | uint8（量化存储，格式由quant_mode决定） | ND | layout_kv为BSND时shape为(B, KV_S, KV_N, D)；为TND时shape为(KV_T, KV_N, D) | × |
| dout | Tensor | 必选 | 公式中的dO，上游输出的梯度（量化输入） | uint8（量化存储，格式由quant_mode决定） | ND | 与q同形 | × |
| attn_out | Tensor | 必选 | 前向`quant_flash_attn`的输出O | bfloat16 | ND | 与q同形 | × |
| q_descale | Tensor | 必选 | q的反量化scale | float32 | ND | 由quant_mode对应量化粒度的shape决定 | × |
| k_descale | Tensor | 必选 | k的反量化scale | float32 | ND | 由quant_mode对应量化粒度的shape决定 | × |
| v_descale | Tensor | 必选 | v的反量化scale | float32 | ND | 由quant_mode对应量化粒度的shape决定 | × |
| do_descale | Tensor | 必选 | dout的反量化scale | float32 | ND | 由quant_mode对应量化粒度的shape决定 | × |
| p_scale | Tensor | 必选 | P的量化参数（与前向`quant_flash_attn`的p_scale一致） | float32 | ND | - | × |
| ds_scale | Tensor | 必选 | dS（Softmax梯度）的量化参数 | float32 | ND | - | × |
| softmax_lse | Tensor | 必选 | 前向`quant_flash_attn`返回的Softmax LSE结果 | float32 | ND | (Q_N, Q_T) | × |
| cu_seqlens_q | Tensor | 可选 | Q的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| cu_seqlens_kv | Tensor | 可选 | KV的累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | q的指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| seqused_kv | Tensor | 可选 | kv的指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| sinks | Tensor | 可选 | 指定每batch中实际使用的序列长度 | float32 | ND | (Q_N,) | × |
| metadata | Tensor | 可选 | 前向`quant_flash_attn_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | 由前向metadata接口动态计算 | × |
| quant_mode | int/QuantMode | 可选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」，默认值为0 | int32 | - | - | - |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认的1/√d计算，默认值为1.0 | float32 | - | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」，默认值为0 | int32 | - | - | - |
| win_left | int | 可选 | window左界限，默认值为-1 | int32 | - | - | - |
| win_right | int | 可选 | window右界限，默认值为-1 | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限，默认值为-1 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限，默认值为-1 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式，支持BSND/BNSD/TND，默认值为"BSND" | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式，默认值为"BSND" | string | - | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | 必选 | 输入q的梯度。 | bfloat16 | ND | 与q同形 | × |
| dk | Tensor | 必选 | 输入k的梯度。 | bfloat16 | ND | 与dq同形（与q一致，接口设计决策，不严格等于k的数学shape） | × |
| dv | Tensor | 必选 | 输入v的梯度。 | bfloat16 | ND | 与dq同形（与q一致，接口设计决策，不严格等于v的数学shape） | × |
| dsink | Tensor | 必选 | sinks的梯度，对各head聚合。 | float32 | ND | (Q_N,) | × |

> [!NOTE]
>
> dk/dv输出的shape与dq一致，即与layout_q下q的shape相同（BSND → (B, Q_S, Q_N, D)；BNSD → (B, Q_N, Q_S, D)；TND → (Q_T, Q_N, D)），而非k/v自身的shape，为接口设计决策（excel1规格）。

## 约束说明

- 参数cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、sinks及metadata属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- `quant_flash_attn_grad`的入参与前向`quant_flash_attn`的入参（quant_mode、layout、seqlen相关参数等）在调用时应该保持一致。算子无法自行校验前向/反向的一致性，正确性需要由客户自行保证。若前后向参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- 不支持非连续Tensor。
- 空Tensor场景处理：空Tensor指必选输入和输出的shape size为0，即有任意轴为0。触发空Tensor的用例将全部拦截报错。
- `mask_mode`当前仅支持0（全计算）/ 3（Causal），不支持4（Sliding Window）。
- `sinks`参数当前暂不支持。
- 输入`q`、`k`、`v`、`dout`为量化数据（uint8存储），量化格式由`quant_mode`决定：0为HIFloat8 per-tensor、1为FP4（E2M1）、2为HIFloat4。
- `layout_q`支持BSND/BNSD/TND；`layout_kv`默认BSND。
- dsink输出为float32，(Q_N,)；dq/dk/dv输出为bfloat16，shape与q一致。

## 调用示例

- 单算子模式调用（layout_q=BSND，非PA场景）：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  B = 2
  Q_S = 16
  KV_S = 16
  Q_N = 8
  KV_N = 2
  D = 128

  # 量化输入（quant_mode=0 为 HIFloat8 per-tensor，以 uint8 存储）
  q = torch.randn(B, Q_S, Q_N, D).to(torch.uint8).npu()
  k = torch.randn(B, KV_S, KV_N, D).to(torch.uint8).npu()
  v = torch.randn(B, KV_S, KV_N, D).to(torch.uint8).npu()
  dout = torch.randn(B, Q_S, Q_N, D).to(torch.uint8).npu()

  # 反量化 scale 与量化参数
  q_descale = torch.ones(1, dtype=torch.float32, device="npu")
  k_descale = torch.ones(1, dtype=torch.float32, device="npu")
  v_descale = torch.ones(1, dtype=torch.float32, device="npu")
  do_descale = torch.ones(1, dtype=torch.float32, device="npu")
  p_scale = torch.ones(1, dtype=torch.float32, device="npu")
  ds_scale = torch.ones(1, dtype=torch.float32, device="npu")

  # 前向输出与 softmax LSE（由 quant_flash_attn 前向计算获得）
  attn_out = torch.randn(B, Q_S, Q_N, D, dtype=torch.bfloat16, device="npu")
  softmax_lse = torch.zeros(Q_N, B * Q_S, dtype=torch.float32, device="npu")

  dq, dk, dv, dsink = cann_ops_transformer.quant_flash_attn_grad(
      q, k, v, dout, attn_out,
      q_descale, k_descale, v_descale, do_descale, p_scale, ds_scale,
      softmax_lse,
      quant_mode=0,
      softmax_scale=1.0 / (D ** 0.5),
      mask_mode=0,
      win_left=-1,
      win_right=-1,
      max_seqlen_q=-1,
      max_seqlen_kv=-1,
      layout_q="BSND",
      layout_kv="BSND",
  )
  torch_npu.npu.synchronize()

  assert dq.shape == (B, Q_S, Q_N, D)
  assert dk.shape == dq.shape
  assert dv.shape == dq.shape
  assert dsink.shape == (Q_N,)
  print(f"dq shape: {dq.shape}, dk shape: {dk.shape}, dv shape: {dv.shape}, dsink shape: {dsink.shape}")
  ```