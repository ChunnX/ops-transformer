# recurrent_kda

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

    执行 recurrent KDA（Kernelized Delta Attention）前向计算，封装 aclnnRecurrentKda。接口支持固定长度 BSND 和变长 TND 布局，可按配置在算子内部完成 Q/K L2 归一化、门控预处理和 beta sigmoid，并支持状态槽索引及投机解码。

- **状态更新**：

    initial_state 是可变输入。当 output_final_state=True 且 inplace_final_state=True 时，最终状态原地写回 initial_state；当 output_final_state=True 且 inplace_final_state=False 时，最终状态通过第二个返回值返回。

## 函数原型

~~~python
cann_ops_transformer.recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    *,
    cu_seqlens=None,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
) -> Tuple[Tensor, Optional[Tensor]]
~~~

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | 必选 | Query 输入。 | bfloat16 | BSND: [B,T,H,K]；TND: [T,H,K] |
| k | Tensor | 必选 | Key 输入，shape 与 q 相同。 | bfloat16 | 同 q |
| v | Tensor | 必选 | Value 输入。 | bfloat16 | BSND: [B,T,HV,V]；TND: [T,HV,V] |
| g | Tensor | 必选 | 门控输入。 | float16、bfloat16、float32 | BSND: [B,T,HV,K]；TND: [T,HV,K] |
| beta | Tensor | 必选 | Delta 更新系数。 | float16、bfloat16、float32 | BSND: [B,T,HV]；TND: [T,HV] |
| initial_state | Tensor | 必选 | 初始递归状态；在原地模式下会被更新。 | bfloat16、float32 | state_v_first=False: [state_capacity,HV,K,V]；为 True: [state_capacity,HV,V,K] |
| cu_seqlens | Tensor | 可选 | 变长序列累计长度。 | int32、int64 | [seq_num+1] |
| ssm_state_indices | Tensor | 可选 | 每个逻辑序列对应的状态槽索引。 | int32、int64 | [seq_num] |
| A_log | Tensor | 可选 | 内核门控模式使用的 A_log。 | float32 | [HV] |
| dt_bias | Tensor | 可选 | 内核门控模式使用的 dt bias。 | float32 | [HV*K] |
| num_accepted_tokens | Tensor | 可选 | 投机解码中每个序列已接受的 token 数。 | int32、int64 | [seq_num] |
| layout | str | 可选 | 输入布局，可选 BSND 或 TND，默认 BSND。 | - | - |
| scale | float | 可选 | QK 缩放系数；None 时使用 K 的负二分之一次方。 | - | - |
| output_final_state | bool | 可选 | 是否返回有效最终状态，默认 False。 | - | - |
| inplace_final_state | bool | 可选 | 是否将最终状态原地写回 initial_state，默认 True。 | - | - |
| use_qk_l2norm_in_kernel | bool | 可选 | 是否在内核中执行 Q/K L2 归一化，默认 False。 | - | - |
| use_gate_in_kernel | bool | 可选 | 是否在内核中处理门控，默认 False。 | - | - |
| use_beta_sigmoid_in_kernel | bool | 可选 | 是否在内核中对 beta 执行 sigmoid，默认 False。 | - | - |
| allow_neg_eigval | bool | 可选 | 是否允许负特征值路径，默认 False。 | - | - |
| safe_gate | bool | 可选 | 是否启用安全门控，默认 False。 | - | - |
| lower_bound | float | 可选 | 安全门控下界；None 时使用 -5.0。 | - | - |
| state_v_first | bool | 可选 | 状态最后两维是否采用 [V,K] 顺序，默认 False。 | - | - |

## 返回值说明

- 第一个 Tensor 为注意力输出，shape 与 v 相同，dtype 为 bfloat16。
- 第二个返回值为最终状态。output_final_state=False 时返回 None；为 True 时 shape 和 dtype 与 initial_state 相同。当 inplace_final_state=True 时，该返回值与更新后的 initial_state 对应。

## 约束说明

- 仅支持 K=128，V=128 或 256。
- H 和 HV 必须为正数，且 HV 能被 H 整除。
- TND 布局下 q、k、v、g 为 3 维，beta 为 2 维；BSND 布局下 q、k、v、g 为 4 维，beta 为 3 维。
- safe_gate=True 时必须同时设置 use_gate_in_kernel=True，且 lower_bound 取值范围为 [-5, 0)。
- 未提供 ssm_state_indices 时，initial_state 的 state_capacity 必须等于逻辑序列数；提供后由索引选择状态槽。
- output_final_state 和 inplace_final_state 为独立开关；只有 output_final_state=True 时第二个返回值包含有效最终状态。
