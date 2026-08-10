# ChunkKdaFwd API

## 功能说明

`ChunkKdaFwd`完成不涉及CP切分的KDA分块正向计算。完整数学定义见[ChunkKdaFwd功能说明](../README.md#功能说明)。各输出的计算语义概括如下：

令$x_{c,i,d}=g_{c,i,d}+dt\_bias_d$，未传入`dt_bias`时令$dt\_bias_d=0$。激活后gate为：

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

$$
\begin{aligned}
gk_{c,i,d} &= \frac{1}{\ln2}\sum_{t=0}^{i}\gamma_{c,t,d},\\
qg_{c,i,d} &= q_{c,i,d}2^{gk_{c,i,d}},\\
kg_{c,i,d} &= k_{c,i,d}2^{gk_{c,l_c,d}-gk_{c,i,d}},\\
Aqk_{c,i,j} &= \mathbb{1}_{j\le i}\cdot scale
\sum_d q_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\\
L_{c,i,j} &= \mathbb{1}_{j<i}\cdot\beta_{c,i}
\sum_d k_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\\
Akk_c &= (I+L_c)^{-1},\\
w_c &= Akk_c\left(\beta_c\odot k_c\odot2^{gk_c}\right),\\
u_c &= Akk_c\left(\beta_c\odot v_c\right),\\
v^{new}_c &= u_c-w_c\,h_c,\\
h_{c+1} &= 2^{gk_{c,l_c}}\odot h_c+kg_c^T\,v^{new}_c,\\
h[c] &= h_c,\qquad final\_state=h_M,\\
attn\_out_c &= scale\cdot qg_c\,h_c+Aqk_c\,v^{new}_c,\\
initial\_state\_out &= initial\_state.
\end{aligned}
$$

传入`initial_state`时$h_0=initial\_state$，否则$h_0=0$。Python接口的`initial_state`返回项为输入对象透传，不是aclnn输出。

## Python接口

### 函数原型

```python
from cann_ops_transformer.ops import chunk_kda_fwd

outputs = chunk_kda_fwd(
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
    layout="BSND",
)
```

### 参数说明

| 参数名 | 参数类型 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| q | Tensor | 必选 | Query输入，shape由`layout`解释。 |
| k | Tensor | 必选 | Key输入，shape和数据类型必须与`q`一致。 |
| v | Tensor | 必选 | Value输入，head数为HV。 |
| g | Tensor | 必选 | raw gate或已激活的自然对数gate。 |
| beta | Tensor | 必选 | Delta系数。 |
| scale | float | 必选 | Query缩放系数，通常取$K^{-0.5}$。 |
| initial_state | Tensor | 可选 | 初始状态，默认值为None。 |
| output_final_state | bool | 可选 | 是否返回`final_state`，默认值为False。 |
| cu_seqlens | Sequence[int]、Tensor | 可选 | 变长序列累计长度，默认值为None。 |
| cu_seqlens_cpu | Sequence[int]、Tensor | 可选 | `cu_seqlens`的host副本，默认值为None。 |
| chunk_indices | Sequence[int]、Tensor | 可选 | 按`(seq_id, chunk_id)`展平的canonical chunk索引，默认值为None。 |
| safe_gate | bool | 可选 | 是否使用有界gate，默认值为False。 |
| lower_bound | float | 可选 | 有界gate下界，默认值为None，计算时按-5.0处理。 |
| use_gate_in_kernel | bool | 可选 | 是否在算子内由raw gate计算激活后gate，默认值为False。 |
| A_log | Tensor | 条件必选 | gate衰减参数，`use_gate_in_kernel=True`时必须传入。 |
| dt_bias | Tensor | 可选 | gate偏置，默认值为None。 |
| chunk_size | int | 可选 | chunk大小，支持64、128，默认值为64。 |
| disable_recompute | bool | 可选 | 是否保留反向计算使用的中间量，默认值为False。 |
| return_intermediate_states | bool | 可选 | 是否额外返回`h`，默认值为False。 |
| state_v_first | bool | 可选 | 状态张量是否使用(V,K)末两维顺序，默认值为False。 |
| layout | str | 可选 | 输入布局，支持`BSND`、`BNSD`、`TND`、`NTD`，默认值为`BSND`。 |

参数的完整dtype、shape和范围约束见[PyTorch API文档](../../../torch_extension/cann_ops_transformer/docs/zh/chunk_kda_fwd.md#参数说明)。

### 返回值说明

Python接口固定返回以下12项：

| 参数名 | 参数类型 | 可选/必选 | 返回条件 |
| --- | --- | --- | --- |
| attn_out | Tensor | 必选 | 始终返回。 |
| final_state | Tensor、None | 可选 | `output_final_state=True`时返回，否则为None。 |
| gk | Tensor、None | 可选 | `use_gate_in_kernel=False`或`disable_recompute=True`时返回，否则为None。 |
| Aqk | Tensor | 必选 | 始终返回。 |
| Akk | Tensor | 必选 | 始终返回。 |
| w | Tensor、None | 可选 | `disable_recompute=True`时返回，否则为None。 |
| u | Tensor、None | 可选 | `disable_recompute=True`时返回，否则为None。 |
| qg | Tensor、None | 可选 | `disable_recompute=True`时返回，否则为None。 |
| kg | Tensor、None | 可选 | `disable_recompute=True`时返回，否则为None。 |
| v_new | Tensor、None | 可选 | `disable_recompute=True`时返回，否则为None。 |
| h | Tensor、None | 可选 | `disable_recompute=True`或`return_intermediate_states=True`时返回，否则为None。 |
| initial_state | Tensor、None | 可选 | 原样返回Python输入`initial_state`对象。 |

`initial_state`返回项不是aclnn输出。

## aclnn接口

### 函数原型

```cpp
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### GetWorkspaceSize参数说明

| 参数名 | 输入/输出 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| q | 输入 | 必选 | Query输入。 |
| k | 输入 | 必选 | Key输入。 |
| v | 输入 | 必选 | Value输入。 |
| g | 输入 | 必选 | raw gate或已激活的自然对数gate。 |
| beta | 输入 | 必选 | Delta系数。 |
| aLogOptional | 输入 | 可选 | gate衰减参数，`useGateInKernel=true`时必须非空。aclnn张量契约仅支持FLOAT。 |
| dtBiasOptional | 输入 | 可选 | gate偏置。aclnn张量契约仅支持FLOAT。 |
| initialStateOptional | 输入 | 可选 | 每条逻辑序列的初始状态。 |
| cuSeqlensOptional | 输入 | 可选 | 变长序列累计长度。 |
| chunkIndicesOptional | 输入 | 可选 | 按`(seq_id, chunk_id)`展平的canonical chunk索引。 |
| layout | 属性 | 必选 | 输入布局，支持`BSND`、`BNSD`、`TND`、`NTD`。 |
| scale | 属性 | 必选 | Query缩放系数。 |
| chunkSize | 属性 | 必选 | chunk大小，支持64、128。 |
| safeGate | 属性 | 必选 | 是否使用有界gate。 |
| lowerBound | 属性 | 必选 | 有界gate下界。 |
| useGateInKernel | 属性 | 必选 | 是否在算子内由raw gate计算激活后gate。 |
| stateVFirst | 属性 | 必选 | 状态张量是否使用(V,K)末两维顺序。 |
| attnOut | 输出 | 必选 | 注意力输出。 |
| finalStateOut | 输出 | 可选 | 最终状态。为空时不向调用方导出该结果。 |
| gkOut | 输出 | 可选 | chunk-local的log2累计gate。为空时不向调用方导出该结果。 |
| aqkOut | 输出 | 必选 | chunk内Query-Key系数矩阵。 |
| akkOut | 输出 | 必选 | chunk内三角求逆结果。 |
| wOut | 输出 | 可选 | $w$中间量。为空时不向调用方导出该结果。 |
| uOut | 输出 | 可选 | $u$中间量。为空时不向调用方导出该结果。 |
| qgOut | 输出 | 可选 | gate缩放后的Query。为空时不向调用方导出该结果。 |
| kgOut | 输出 | 可选 | gate缩放后的Key。为空时不向调用方导出该结果。 |
| vNewOut | 输出 | 可选 | 状态校正后的Value。为空时不向调用方导出该结果。 |
| hOut | 输出 | 可选 | 各个chunk计算前的状态。为空时不向调用方导出该结果。 |
| workspaceSize | 输出 | 必选 | 返回执行算子所需workspace大小，单位为Byte。 |
| executor | 输出 | 必选 | 返回算子执行器。 |

aclnn L2只描述张量契约，不接收`output_final_state`、`disable_recompute`或`return_intermediate_states`，也不解释utograd的重计算策略。每个可选输出指针均独立表示是否向调用方导出对应结果。

### 执行接口参数说明

| 参数名 | 输入/输出 | 可选/必选 | 说明 |
| --- | --- | --- | --- |
| workspace | 输入 | 必选 | 执行算子所需workspace。`workspaceSize=0`时可传入空指针。 |
| workspaceSize | 输入 | 必选 | `aclnnChunkKdaFwdGetWorkspaceSize`返回的workspace大小。 |
| executor | 输入 | 必选 | `aclnnChunkKdaFwdGetWorkspaceSize`返回的算子执行器。 |
| stream | 输入 | 必选 | 执行算子的AscendCL Stream。 |

## 输入与输出布局

`layout`只解释`q`、`k`、`v`、`g`、`beta`输入。每个输出的固定布局如下：

| 输出名 | rank-4输入 | rank-3输入 |
| --- | --- | --- |
| attnOut | BSND：[B,S,HV,V] | TND：[T,HV,V] |
| finalStateOut | [N,HV,K,V]；`stateVFirst=true`时为[N,HV,V,K] | [N,HV,K,V]；`stateVFirst=true`时为[N,HV,V,K] |
| gkOut | BNSD：[B,HV,S,K] | NTD：[HV,T,K] |
| aqkOut | BNSD：[B,HV,S,C] | NTD：[HV,T,C] |
| akkOut | BNSD：[B,HV,S,C] | NTD：[HV,T,C] |
| wOut | BNSD：[B,HV,S,K] | NTD：[HV,T,K] |
| uOut | BNSD：[B,HV,S,V] | NTD：[HV,T,V] |
| qgOut | BNSD：[B,HV,S,K] | NTD：[HV,T,K] |
| kgOut | BNSD：[B,HV,S,K] | NTD：[HV,T,K] |
| vNewOut | BNSD：[B,HV,S,V] | NTD：[HV,T,V] |
| hOut | [B,NC,HV,K,V]；`stateVFirst=true`时为[B,NC,HV,V,K] | [NC,HV,K,V]；`stateVFirst=true`时为[NC,HV,V,K] |

## 调用路径

| 调用路径 | 入口 |
| --- | --- |
| Python稳定入口 | `cann_ops_transformer.ops.chunk_kda_fwd` |
| PyTorch dispatcher | `torch.ops.cann_ops_transformer.chunk_kda_fwd`，由Python稳定入口调用 |
| aclnn GetWorkspaceSize | `aclnnChunkKdaFwdGetWorkspaceSize` |
| aclnn执行 | `aclnnChunkKdaFwd` |

`cu_seqlens_cpu`只存在于Python包装层，aclnn接口只接收一份`aclIntArray`。Python稳定入口接受BF16或FP32的`A_log`和`dt_bias`，BF16输入在调用aclnn前转为FP32，不改变L0和aclnn张量契约。
