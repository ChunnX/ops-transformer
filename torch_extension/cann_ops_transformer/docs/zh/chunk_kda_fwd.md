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

- **接口功能**：完成不涉及CP切分的KDA分块正向计算，计算注意力输出、最终状态以及可选的反向中间量。

- **计算公式**：

  将每条序列按$C=chunk\_size$划分为$M$个chunk。以第$c$个chunk为例，$i$、$j$为chunk内token下标，$l_c$为该chunk最后一个有效token下标。公式省略batch和head下标；GQA场景中，每个Value head使用其对应的Query/Key head。

  1. 计算激活后gate。令$x_{c,i,d}=g_{c,i,d}+dt\_bias_d$，未传入`dt_bias`时令$dt\_bias_d=0$：

      $$
      \gamma_{c,i,d}=
      \begin{cases}
      g_{c,i,d}, & use\_gate\_in\_kernel=false,\\
      -\exp(A_{log})\operatorname{softplus}(x_{c,i,d}),
          & use\_gate\_in\_kernel=true,\ safe\_gate=false,\\
      lower\_bound\operatorname{sigmoid}(\exp(A_{log})x_{c,i,d}),
          & use\_gate\_in\_kernel=true,\ safe\_gate=true.
      \end{cases}
      $$

  2. 计算chunk-local的log2累计gate `gk`，以及gate缩放后的Query `qg`和Key `kg`：

      $$
      gk_{c,i,d}=\frac{1}{\ln 2}\sum_{t=0}^{i}\gamma_{c,t,d},
      $$

      $$
      qg_{c,i,d}=q_{c,i,d}2^{gk_{c,i,d}},\qquad
      kg_{c,i,d}=k_{c,i,d}2^{gk_{c,l_c,d}-gk_{c,i,d}}.
      $$

  3. 计算包含对角线的下三角Query-Key系数矩阵 `Aqk`：

      $$
      Aqk_{c,i,j}=\mathbb{1}_{j\le i}\cdot scale
      \sum_d q_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}}.
      $$

  4. 计算严格下三角矩阵$L_c$以及三角求逆结果 `Akk`：

      $$
      L_{c,i,j}=\mathbb{1}_{j<i}\cdot\beta_{c,i}
      \sum_d k_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\qquad
      Akk_c=(I+L_c)^{-1}.
      $$

  5. 计算 `w`和 `u`：

      $$
      w^{seed}_{c,i,d}=\beta_{c,i}k_{c,i,d}2^{gk_{c,i,d}},\qquad
      u^{seed}_{c,i,e}=\beta_{c,i}v_{c,i,e},
      $$

      $$
      w_c=Akk_c\,w^{seed}_c,\qquad u_c=Akk_c\,u^{seed}_c.
      $$

  6. 计算 `v_new`、`h`和 `final_state`。令$h_c$为第$c$个chunk计算前的状态；传入`initial_state`时$h_0=initial\_state$，否则$h_0=0$：

      $$
      v^{new}_c=u_c-w_c\,h_c,
      $$

      $$
      h_{c+1}=2^{gk_{c,l_c}}\odot h_c+kg_c^T\,v^{new}_c,
      $$

      $$
      h[c]=h_c,\qquad final\_state=h_M.
      $$

      其中$2^{gk_{c,l_c}}\odot h_c$表示按$K$维缩放状态矩阵。

  7. 计算注意力输出 `attn_out`：

      $$
      attn\_out_c=scale\cdot qg_c\,h_c+Aqk_c\,v^{new}_c.
      $$

  8. Python接口的第12项返回值原样透传输入对象：

      $$
      initial\_state\_out=initial\_state.
      $$

  可选输出参数只决定是否向调用方返回对应中间量，不改变上述计算语义。

## 函数原型

```python
cann_ops_transformer.ops.chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    chunk_size=64,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    layout="BSND") -> tuple
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| q | Tensor | 必选 | Query，对应公式中的$q$。不支持空Tensor，数据格式支持ND。 | bfloat16、float16 | `layout="BSND"`：[B,S,H,K]<br>`layout="BNSD"`：[B,H,S,K]<br>`layout="TND"`：[T,H,K]<br>`layout="NTD"`：[H,T,K] |
| k | Tensor | 必选 | Key，对应公式中的$k$。shape和数据类型必须与`q`一致，数据格式支持ND。 | bfloat16、float16 | 与`q`一致 |
| v | Tensor | 必选 | Value，对应公式中的$v$。数据类型必须与`q`一致，数据格式支持ND。 | bfloat16、float16 | `layout="BSND"`：[B,S,HV,V]<br>`layout="BNSD"`：[B,HV,S,V]<br>`layout="TND"`：[T,HV,V]<br>`layout="NTD"`：[HV,T,V] |
| g | Tensor | 必选 | `use_gate_in_kernel=False`时为已激活的自然对数gate；`use_gate_in_kernel=True`时为raw gate。数据格式支持ND。 | float32、bfloat16 | `layout="BSND"`：[B,S,HV,K]<br>`layout="BNSD"`：[B,HV,S,K]<br>`layout="TND"`：[T,HV,K]<br>`layout="NTD"`：[HV,T,K] |
| beta | Tensor | 必选 | Delta系数，对应公式中的$\beta$。数据格式支持ND。 | float32、bfloat16 | `layout="BSND"`：[B,S,HV]<br>`layout="BNSD"`：[B,HV,S]<br>`layout="TND"`：[T,HV]<br>`layout="NTD"`：[HV,T] |
| scale | float | 必选 | Query缩放系数，对应公式中的$scale$，通常取$K^{-0.5}$。 | float | - |
| initial_state | Tensor | 可选 | 每条逻辑序列的初始状态，默认值为None，表示零状态。数据格式支持ND。 | float32 | `state_v_first=False`：[N,HV,K,V]<br>`state_v_first=True`：[N,HV,V,K] |
| output_final_state | bool | 可选 | 是否返回`final_state`，默认值为False。 | bool | - |
| cu_seqlens | Sequence[int]、Tensor | 可选 | 变长序列累计长度，默认值为None。首元素必须为0，末元素必须为S或T，元素单调不减。Tensor输入在Python层转为host整数序列。 | 整数 | [N+1] |
| cu_seqlens_cpu | Sequence[int]、Tensor | 可选 | `cu_seqlens`的host副本，默认值为None。传入时优先使用该参数；同时传入非Tensor的`cu_seqlens`时，两者元素必须完全相同。 | 整数 | [N+1] |
| chunk_indices | Sequence[int]、Tensor | 可选 | chunk索引，按`(seq_id, chunk_id)`展平保存，默认值为None。必须为sequence-major canonical顺序，传入时必须同时传入`cu_seqlens`。 | 整数 | [2×NC] |
| safe_gate | bool | 可选 | 是否使用有界gate计算，默认值为False。 | bool | - |
| lower_bound | float | 可选 | 有界gate下界，默认值为None，计算时按-5.0处理。`safe_gate=True`且`use_gate_in_kernel=True`时取值范围为[-5,0)。 | float | - |
| use_gate_in_kernel | bool | 可选 | 是否在算子内由raw gate计算激活后gate，默认值为False。 | bool | - |
| A_log | Tensor | 条件必选 | gate衰减参数，`use_gate_in_kernel=True`时必须传入。bfloat16输入在Python层转为float32后调用aclnn接口。数据格式支持ND。 | float32、bfloat16 | [HV] |
| dt_bias | Tensor | 可选 | gate偏置，默认值为None。bfloat16输入在Python层转为float32后调用aclnn接口。数据格式支持ND。 | float32、bfloat16 | [HV×K] |
| chunk_size | int | 可选 | chunk大小，支持64、128，默认值为64。 | int | - |
| disable_recompute | bool | 可选 | 是否保留反向计算使用的`gk`、`w`、`u`、`qg`、`kg`、`v_new`和`h`，默认值为False。 | bool | - |
| return_intermediate_states | bool | 可选 | 是否额外返回每个chunk的输入状态`h`，默认值为False。 | bool | - |
| state_v_first | bool | 可选 | 状态张量是否使用(V,K)末两维顺序，默认值为False。 | bool | - |
| layout | str | 可选 | 输入布局，支持`BSND`、`BNSD`、`TND`、`NTD`，默认值为`BSND`。该参数只描述输入布局。 | str | - |

## 返回值说明

返回12项Tensor或None，顺序如下：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| attn_out | Tensor | 必选 | 注意力输出，对应公式中的$attn\_out$。rank-4输入固定输出BSND，rank-3输入固定输出TND，数据格式支持ND。 | 与`v`一致 | rank-4：[B,S,HV,V]<br>rank-3：[T,HV,V] |
| final_state | Tensor、None | 可选 | 每条逻辑序列最后一个chunk更新后的状态。`output_final_state=True`时返回，否则返回None。数据格式支持ND。 | float32 | `state_v_first=False`：[N,HV,K,V]<br>`state_v_first=True`：[N,HV,V,K] |
| gk | Tensor、None | 可选 | chunk-local的log2累计gate。`use_gate_in_kernel=False`或`disable_recompute=True`时返回，否则返回None。固定为head-major，数据格式支持ND。 | float32 | rank-4：[B,HV,S,K]<br>rank-3：[HV,T,K] |
| Aqk | Tensor | 必选 | chunk内Query-Key系数矩阵，对应公式中的$Aqk$。固定为head-major，数据格式支持ND。 | 与`q`一致 | rank-4：[B,HV,S,C]<br>rank-3：[HV,T,C] |
| Akk | Tensor | 必选 | chunk内三角求逆结果，对应公式中的$Akk$。固定为head-major，数据格式支持ND。 | 与`q`一致 | rank-4：[B,HV,S,C]<br>rank-3：[HV,T,C] |
| w | Tensor、None | 可选 | 对应公式中的$w$。`disable_recompute=True`时返回，否则返回None。固定为head-major，数据格式支持ND。 | 与`q`一致 | rank-4：[B,HV,S,K]<br>rank-3：[HV,T,K] |
| u | Tensor、None | 可选 | 对应公式中的$u$。`disable_recompute=True`时返回，否则返回None。固定为head-major，数据格式支持ND。 | 与`v`一致 | rank-4：[B,HV,S,V]<br>rank-3：[HV,T,V] |
| qg | Tensor、None | 可选 | gate缩放后的Query，对应公式中的$qg$。`disable_recompute=True`时返回，否则返回None。固定为head-major，数据格式支持ND。 | 与`q`一致 | rank-4：[B,HV,S,K]<br>rank-3：[HV,T,K] |
| kg | Tensor、None | 可选 | gate缩放后的Key，对应公式中的$kg$。`disable_recompute=True`时返回，否则返回None。固定为head-major，数据格式支持ND。 | 与`k`一致 | rank-4：[B,HV,S,K]<br>rank-3：[HV,T,K] |
| v_new | Tensor、None | 可选 | 状态校正后的Value，对应公式中的$v^{new}$。`disable_recompute=True`时返回，否则返回None。固定为head-major，数据格式支持ND。 | 与`v`一致 | rank-4：[B,HV,S,V]<br>rank-3：[HV,T,V] |
| h | Tensor、None | 可选 | 各个chunk计算前的状态$h_c$。`disable_recompute=True`或`return_intermediate_states=True`时返回，否则返回None。固定为sequence-major，数据格式支持ND。 | 与`q`一致 | rank-4且`state_v_first=False`：[B,NC,HV,K,V]<br>rank-4且`state_v_first=True`：[B,NC,HV,V,K]<br>rank-3且`state_v_first=False`：[NC,HV,K,V]<br>rank-3且`state_v_first=True`：[NC,HV,V,K] |
| initial_state | Tensor、None | 可选 | 原样返回输入`initial_state`对象，不是aclnn输出。 | 与输入`initial_state`一致 | 与输入`initial_state`一致 |

## 约束说明

- 该接口支持单算子模式调用。
- B表示batch大小，S表示rank-4输入的序列长度，T表示rank-3输入的总token数，H表示Query和Key的head数，HV表示Value和gate的head数，K表示Query和Key的head dim，V表示Value的head dim，N表示逻辑序列数，C表示chunk大小。未传入`cu_seqlens`时，NC表示每条序列的chunk数$\lceil S/C\rceil$；传入`cu_seqlens`时，NC表示所有逻辑序列的chunk总数。
- H和HV必须满足$0<H\le HV\le128$且$HV\bmod H=0$。
- K和V取值范围为[16,256]，且必须为16的倍数。
- `q`、`k`、`v`的数据类型必须相同。
- `chunk_size`仅支持64、128。
- `use_gate_in_kernel=True`时必须传入`A_log`，`dt_bias`可以不传入。
- `safe_gate=True`且`use_gate_in_kernel=True`时，`lower_bound`的有效取值范围为[-5,0)。
- rank-4变长输入要求B=1。
- 变长输入最多支持1024条逻辑序列。
- `chunk_indices`传入时必须同时传入`cu_seqlens`，并且每个chunk必须恰好存在一个按sequence-major canonical顺序排列的`(seq_id, chunk_id)`对。
- `layout`只描述输入布局，输出布局按返回值说明固定。

## 确定性计算

- 默认支持确定性计算。

## 调用说明

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import chunk_kda_fwd

    B, S, H, K, V = 1, 128, 4, 128, 128
    q = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, V, device="npu", dtype=torch.bfloat16)
    g = torch.randn(B, S, H, K, device="npu", dtype=torch.float32)
    beta = torch.rand(B, S, H, device="npu", dtype=torch.bfloat16)
    A_log = torch.randn(H, device="npu", dtype=torch.bfloat16)
    dt_bias = torch.randn(H * K, device="npu", dtype=torch.bfloat16)

    (
        attn_out,
        final_state,
        gk,
        Aqk,
        Akk,
        w,
        u,
        qg,
        kg,
        v_new,
        h,
        initial_state,
    ) = chunk_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K ** -0.5,
        output_final_state=True,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=64,
        disable_recompute=True,
        return_intermediate_states=True,
        state_v_first=False,
        layout="BSND",
    )

    assert attn_out.shape == (B, S, H, V)
    assert final_state.shape == (B, H, K, V)
    assert gk.shape == (B, H, S, K)
    assert Aqk.shape == (B, H, S, 64)
    assert Akk.shape == (B, H, S, 64)
    assert w.shape == (B, H, S, K)
    assert u.shape == (B, H, S, V)
    assert qg.shape == (B, H, S, K)
    assert kg.shape == (B, H, S, K)
    assert v_new.shape == (B, H, S, V)
    assert h.shape == (B, 2, H, K, V)
    assert initial_state is None
    ```
