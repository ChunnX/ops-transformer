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

- **接口功能**：

    执行 KDA（Kernelized Delta Attention）分块前向计算，封装 aclnnChunkKdaFwd。支持 BSND、BNSD、TND、NTD 四种输入布局、定长及变长序列，并可按需返回最终状态和反向计算所需的中间量。

- **输出布局**：

    输入 layout 仅描述 q、k、v、g 和 beta。注意力输出固定为 sequence-major；gk、aqk、akk、w、u、qg、kg 和 v_new 固定为 head-major。

## 函数原型

~~~python
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
~~~

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
