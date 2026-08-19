# chunk_kda_fwd

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- 接口功能：完成不涉及CP切分的KDA分块正向计算，输出注意力结果、最终状态和反向计算所需的中间量。

- 计算公式：

  将每条序列按$C=chunkSize$划分为$M$个chunk。以第$c$个chunk为例，$i$、$j$表示chunk内token下标，$l_c$表示该chunk最后一个有效token下标。以下公式省略batch和head下标；GQA场景中，每个Value head使用对应的Query/Key head。

  令$x_{c,i,d}=g_{c,i,d}+dtBias_d$，未传入`dtBiasOptional`时令$dtBias_d=0$。激活后的gate为：

  $$
  \gamma_{c,i,d}=
  \begin{cases}
  g_{c,i,d}, & useGateInKernel=false,\\
  -\exp(A_{log})\operatorname{softplus}(x_{c,i,d}),
      & useGateInKernel=true,\ safeGate=false,\\
  lowerBound\operatorname{sigmoid}(\exp(A_{log})x_{c,i,d}),
      & useGateInKernel=true,\ safeGate=true.
  \end{cases}
  $$

  `gkOut`为chunk内以2为底的累计gate：

  $$
  gk_{c,i,d}=\frac{1}{\ln 2}\sum_{t=0}^{i}\gamma_{c,t,d}.
  $$

  `qgOut`和`kgOut`分别为gate缩放后的Query和Key：

  $$
  qg_{c,i,d}=q_{c,i,d}2^{gk_{c,i,d}},\qquad
  kg_{c,i,d}=k_{c,i,d}2^{gk_{c,l_c,d}-gk_{c,i,d}}.
  $$

  `aqkOut`为包含对角线的下三角Query-Key系数矩阵：

  $$
  Aqk_{c,i,j}=\mathbb{1}_{j\le i}\cdot scale
  \sum_d q_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}}.
  $$

  定义严格下三角矩阵$L_c$，`akkOut`为$I+L_c$的逆矩阵：

  $$
  L_{c,i,j}=\mathbb{1}_{j<i}\cdot\beta_{c,i}
  \sum_d k_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\qquad
  Akk_c=(I+L_c)^{-1}.
  $$

  定义$w^{seed}$和$u^{seed}$：

  $$
  w^{seed}_{c,i,d}=\beta_{c,i}k_{c,i,d}2^{gk_{c,i,d}},\qquad
  u^{seed}_{c,i,e}=\beta_{c,i}v_{c,i,e}.
  $$

  `wOut`和`uOut`为：

  $$
  w_c=Akk_c\,w^{seed}_c,\qquad u_c=Akk_c\,u^{seed}_c.
  $$

  令$h_c$为第$c$个chunk计算前的状态。传入`initialStateOptional`时$h_0=initialStateOptional$，否则$h_0=0$。`vNewOut`、下一个chunk状态、`hOut`和`finalStateOut`为：

  $$
  v^{new}_c=u_c-w_c\,h_c,
  $$

  $$
  h_{c+1}=2^{gk_{c,l_c}}\odot h_c+kg_c^T\,v^{new}_c,
  $$

  $$
  hOut[c]=h_c,\qquad finalStateOut=h_M.
  $$

  其中，$2^{gk_{c,l_c}}\odot h_c$表示沿$K$维缩放状态矩阵。`attnOut`为：

  $$
  attnOut_c=scale\cdot qg_c\,h_c+Aqk_c\,v^{new}_c.
  $$

- 符号说明：

  | 符号 | 含义 |
  | --- | --- |
  | B | Batch Size。 |
  | S | 四维输入的序列长度。 |
  | T | 三维输入的总token数。 |
  | H | Query和Key的head数。 |
  | HV | Value和gate的head数。 |
  | K | Query和Key的head dim。 |
  | V | Value的head dim。 |
  | N | 逻辑序列数。 |
  | C | chunk大小。 |
  | NC | 未传入`cuSeqlensOptional`时为每条序列的chunk数$\lceil S/C\rceil$；传入时为所有逻辑序列的chunk总数。 |

## 函数原型

```python
cann_ops_transformer.chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    chunk_size=64,
    *,
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    use_exp2=True,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
) -> Tuple[
    Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor,
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]
]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | 必选 | Query 输入。 | float16、bfloat16 | BSND: [B,S,H,K]；BNSD: [B,H,S,K]；TND: [T,H,K]；NTD: [H,T,K] |
| k | Tensor | 必选 | Key 输入，shape 和 dtype 与 q 相同。 | 同 q | 同 q |
| v | Tensor | 必选 | Value 输入。 | 同 q | BSND: [B,S,HV,V]；BNSD: [B,HV,S,V]；TND: [T,HV,V]；NTD: [HV,T,V] |
| g | Tensor | 必选 | 已激活 gate 或 raw gate。 | float32、bfloat16 | BSND: [B,S,HV,K]；BNSD: [B,HV,S,K]；TND: [T,HV,K]；NTD: [HV,T,K] |
| beta | Tensor | 必选 | Delta 更新系数。 | float32、bfloat16 | BSND: [B,S,HV]；BNSD: [B,HV,S]；TND: [T,HV]；NTD: [HV,T] |
| scale | float | 必选 | Query 缩放系数，通常取 K 的负二分之一次方。 | - | - |
| chunk_size | int | 可选 | Chunk 大小，仅支持 64、128，默认 64。 | - | - |
| layout | str | 可选 | 输入布局，支持 BSND、BNSD、TND、NTD，默认 BSND。 | - | - |
| initial_state | Tensor | 可选 | 初始状态；None 表示使用零状态。 | float32 | state_v_first=False: [N,HV,K,V]；为 True: [N,HV,V,K] |
| output_final_state | bool | 可选 | 是否返回最终状态，默认 False。 | - | - |
| cu_seqlens | int序列 | 可选 | 变长序列累计长度，首元素为0、末元素为S或T。 | int64 | [N+1] |
| chunk_indices | int序列 | 可选 | 展平的 (seq_id, chunk_id) 对，必须为 sequence-major canonical 顺序。 | int64 | [2*NC] |
| safe_gate | bool | 可选 | 是否使用有界 gate，默认 False。 | - | - |
| lower_bound | float | 可选 | 有界 gate 下界；None 时使用 -5.0。 | - | - |
| use_qk_l2norm_in_kernel | bool | 可选 | 预留参数，当前仅支持 False，默认 False。 | - | - |
| use_gate_in_kernel | bool | 可选 | 是否在算子内由 raw gate 计算激活值，默认 False。 | - | - |
| use_beta_sigmoid_in_kernel | bool | 可选 | 预留参数，当前仅支持 False，默认 False。 | - | - |
| use_exp2 | bool | 可选 | 预留参数，当前不影响计算，默认 True。 | - | - |
| A_log | Tensor | 可选 | gate 衰减参数；use_gate_in_kernel=True 时必选。 | float32 | [HV] |
| dt_bias | Tensor | 可选 | gate 偏置，仅在内核 gate 模式下参与计算。 | float32 | [HV*K] |
| disable_recompute | bool | 可选 | 是否导出反向所需的 w、u、qg、kg、v_new，默认 False。 | - | - |
| return_intermediate_states | bool | 可选 | 是否导出每个 chunk 的中间状态 h，默认 False。 | - | - |
| state_v_first | bool | 可选 | 状态末两维是否按 [V,K] 排列，默认 False。 | - | - |

## 返回值说明

接口依次返回以下 12 项：

1. attn_out：注意力输出。四维输入为 [B,S,HV,V]，三维输入为 [T,HV,V]，dtype 与 q 相同。
2. final_state：output_final_state=True 时返回最终状态，否则为 None；dtype 为 float32。
3. gk_out：use_gate_in_kernel=False 时返回 chunk-local 累计 gate，否则为 None；dtype 为 float32。
4. aqk：Query-Key chunk 系数矩阵。
5. akk：chunk 内下三角求逆结果。
6. w：disable_recompute=True 时返回，否则为 None。
7. u：disable_recompute=True 时返回，否则为 None。
8. qg：disable_recompute=True 时返回，否则为 None。
9. kg：disable_recompute=True 时返回，否则为 None。
10. v_new：disable_recompute=True 时返回，否则为 None。
11. h：return_intermediate_states=True 时返回，否则为 None。
12. initial_state_out：与输入 initial_state 对应；未提供初始状态时为 None。

## 约束说明

- H 和 HV 满足 0 < H <= HV <= 128，且 HV 能被 H 整除。
- K 和 V 的范围为 [16,256]，并且必须为 16 的倍数。
- 四维变长输入要求 B=1；变长输入最多支持 1024 条逻辑序列。
- chunk_indices 必须和 cu_seqlens 同时使用，并按 sequence-major canonical 顺序包含每个 chunk。
- safe_gate=True 且 use_gate_in_kernel=True 时，lower_bound 的范围为 [-5,0)。
- use_qk_l2norm_in_kernel 和 use_beta_sigmoid_in_kernel 当前均仅支持 False。
- use_exp2 为预留参数，默认 True，当前取值不影响计算结果。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import chunk_kda_fwd

    torch_npu.npu.set_device(0)

    B, S, H, HV, K, V = 1, 64, 96, 96, 128, 128
    q = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
    v = torch.randn(B, S, HV, V, device="npu", dtype=torch.bfloat16)
    g = torch.randn(B, S, HV, K, device="npu", dtype=torch.float32)
    beta = torch.randn(B, S, HV, device="npu", dtype=torch.float32)

    outputs = chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        K ** -0.5,
        chunk_size=64,
        layout="BSND",
    )
    attn_out = outputs[0]
    ```
