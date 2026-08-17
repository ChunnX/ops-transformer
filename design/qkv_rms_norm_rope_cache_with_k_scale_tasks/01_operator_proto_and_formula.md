# 算子原型与计算公式设计

## 1. 算子名

```text
QkvRmsNormRopeCacheWithKScale
```

首期平台：arch35/Ascend950。

## 2. 输入输出原型

本节给出包含 M-RoPE 能力的完整接口。input/output anchor 与属性采用固定索引；具体调用
只允许第 11 节定义的 `ROPE_FP8_QK` 和 `MROPE_BF16_Q_INT8_K` 两种完整组合。算子接口不
提供独立 scene 属性，而是根据 optional 实参 presence 识别候选场景。

### 2.1 输入 tensor

| index | 名称 | ParamType | RoPE dtype/shape | M-RoPE dtype/shape | 说明 |
|------:|------|-----------|--------------------|---------------------|------|
| 0 | `qkv` | REQUIRED | `bf16`，见 `layout_qkv` | `bf16`，见 `layout_qkv` | Q/K/V 融合输入 |
| 1 | `q_gamma` | REQUIRED | `float32[D]` | `float32[D]` | Q RMSNorm gamma |
| 2 | `k_gamma` | REQUIRED | `float32[D]` | `float32[D]` | K RMSNorm gamma |
| 3 | `cos_sin` | REQUIRED | `float32[MaxSeqLen,D]` | `float32[MaxSeqLen,D]` | 前 `D/2` 为 cos，后 `D/2` 为 sin |
| 4 | `slot_mapping` | REQUIRED | `int32[T]` | `int32[T]` | token 到 cache slot 的映射 |
| 5 | `k_cache` | REQUIRED | `float8_e4m3fn[BlockNum,Nk,BlockSize,D]` | `int8[BlockNum,Nk,BlockSize,D]` | 输入输出复用空间 |
| 6 | `v_cache` | REQUIRED | `float8_e4m3fn[BlockNum,Nv,BlockSize,D]` | 同 RoPE | 输入输出复用空间 |
| 7 | `k_scale_cache` | REQUIRED | `float32[BlockNum,Nk,BlockSize,1]` | 同 RoPE | 输入输出复用空间 |
| 8 | `query_start_loc` | OPTIONAL | `int32[Batch+1]`，必须存在 | 必须不存在 | RoPE 场景的当前调用 local 前缀和 |
| 9 | `seq_lens` | OPTIONAL | `int32[Batch]`，必须存在 | 必须不存在 | RoPE 场景的实际序列长度 |
| 10 | `rotation` | OPTIONAL | `bf16[D,D]`，当前必须存在 | 同 RoPE | Q/K 分支共享的 Cube 右矩阵 |
| 11 | `v_scale` | OPTIONAL | `float32[Nv]`，当前必须存在 | `float32[Nv,D]`，当前必须存在 | V 分支量化缩放因子；RoPE 按 head，M-RoPE 按 head/channel |
| 12 | `mrope_position` | OPTIONAL | 必须不存在 | `int32[T,3]`，必须存在 | 每个 token 占一行，列 0/1/2 为 T/H/W 三路 `cos_sin` 行下标 |

这里的 OPTIONAL 是 OpDef 接口级 `ParamType`。非空但零元素的 tensor、graph descriptor 存在但 shape 缺失，
以及 Torch undefined tensor 均按 present 处理并在合同校验中失败，不能伪装成 absent。

### 2.2 输出 tensor

| index | 名称 | ParamType | RoPE dtype/shape | M-RoPE dtype/shape | 说明 |
|------:|------|-----------|--------------------|---------------------|------|
| 0 | `q_out` | REQUIRED | `float8_e4m3fn`，见 `layout_q_out` | `bf16`，见 `layout_q_out` | Q 分支输出 |
| 1 | `q_scale` | REQUIRED | `float32`，见 `layout_q_out` | `float32`，见 `layout_q_out` | Q dynamic quant scale |
| 2 | `k_cache` | REQUIRED | 与 input 5 同 dtype/shape | 与 input 5 同 dtype/shape | 原地更新 |
| 3 | `v_cache` | REQUIRED | 与 input 6 同 dtype/shape | 与 input 6 同 dtype/shape | 原地更新 |
| 4 | `k_scale_cache` | REQUIRED | 与 input 7 同 dtype/shape | 与 input 7 同 dtype/shape | 原地更新 |

OpDef 的五个 output anchor 全部为 `REQUIRED`。Torch 的 `q_scale` 类型声明为 `Tensor`：RoPE 返回
FP32 scale，M-RoPE 运行值为 `None`。ACLNN 仍使用可选 `qScaleOptional`，M-RoPE 传 `nullptr`。

### 2.3 属性

| index | 名称 | OpDef 类型 | 默认值 | 说明 |
|------:|------|------------|--------|------|
| 0 | `head_nums` | REQUIRED `ListInt` | 无 | 长度固定为 3，依次为 `[Nq,Nk,Nv]`；两场景统一要求 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv` |
| 1 | `layout_qkv` | OPTIONAL `String` | `TND` | `qkv` 输入 layout，仅支持 `NTD`、`TND` |
| 2 | `layout_q_out` | OPTIONAL `String` | `NTD` | `q_out/q_scale` 输出 layout，仅支持 `NTD`、`TND` |
| 3 | `epsilon` | OPTIONAL `Float` | `1e-6` | RMSNorm 防止除零，由 tiling data 传入 kernel |
| 4 | `mrope_section` | OPTIONAL `ListInt` | absent | `[t,h,w]` 三元属性；只在 M-RoPE 模式存在，具体值域和语义见 11.1/11.3 |
| 5 | `q_quant_mode` | OPTIONAL `String` | `PerTokenPerHead` | 当前只支持 `PerTokenPerHead`、`NoQuant` |
| 6 | `q_out_dtype` | OPTIONAL `Int` | `ge::DT_FLOAT8_E4M3FN` | 当前只支持 `ge::DT_FLOAT8_E4M3FN`、`ge::DT_BF16` |

### 2.4 ACLNN、L0 与 Torch 完整签名

OpDef/L0 在既有 input 参数组尾部追加 `mrope_position`，在既有属性组尾部保留
`mrope_section/q_quant_mode/q_out_dtype`，其中属性 6 的 `q_out_dtype` 仍是标准 GE dtype enum 对应的
`Int`。ACLNN C ABI 在既有 input 参数组尾部追加 `mropePositionOptional`，在属性参数组尾部只追加
`mropeSectionOptional/qQuantMode`，不提供独立 `qOutDtype` 参数；
`queryStartLocOptional/seqLensOptional` 保留原位置但允许传 `nullptr`：

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qGamma,
    const aclTensor *kGamma,
    const aclTensor *cosSin,
    const aclTensor *slotMapping,
    aclTensor *kCacheRef,
    aclTensor *vCacheRef,
    aclTensor *kScaleCacheRef,
    const aclTensor *queryStartLocOptional,
    const aclTensor *seqLensOptional,
    const aclTensor *rotationOptional,
    const aclTensor *vScaleOptional,
    const aclTensor *mropePositionOptional,
    const aclIntArray *headNums,
    const char *layoutQkv,
    const char *layoutQOut,
    float epsilon,
    const aclIntArray *mropeSectionOptional,
    const char *qQuantMode,
    aclTensor *qOut,
    aclTensor *qScaleOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

执行接口不承载算子参数，本次保持不变。ACLNN 以 `qOut->GetDataType()` 为输出 dtype 的唯一来源：
RoPE 场景要求 FP8 E4M3FN，M-RoPE 场景要求 BF16；ACLNN 将该 dtype 映射为属性 6 的标准 GE enum，
不接受也不比较第二份 dtype 标量。

Torch inplace 与 functional schema 为：

```text
qkv_rms_norm_rope_cache_with_k_scale_(
    Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping,
    Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor(c!) k_scale_cache,
    Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, *,
    Tensor? rotation=None, Tensor? v_scale=None,
    str? layout_qkv="TND", str? layout_q_out="NTD", float epsilon=0.000001,
    Tensor? mrope_position=None, int[]? mrope_section=None,
    str q_quant_mode="PerTokenPerHead", ScalarType? q_out_dtype=None
) -> (Tensor, Tensor)

qkv_rms_norm_rope_cache_with_k_scale(
    Tensor qkv, Tensor q_gamma, Tensor k_gamma, Tensor cos_sin, Tensor slot_mapping,
    Tensor k_cache, Tensor v_cache, Tensor k_scale_cache,
    Tensor? query_start_loc, Tensor? seq_lens, int[] head_nums, *,
    Tensor? rotation=None, Tensor? v_scale=None,
    str? layout_qkv="TND", str? layout_q_out="NTD", float epsilon=0.000001,
    Tensor? mrope_position=None, int[]? mrope_section=None,
    str q_quant_mode="PerTokenPerHead", ScalarType? q_out_dtype=None
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

`query_start_loc/seq_lens` 使用无默认值的 `Tensor?`，因为其后仍有必选 `head_nums`；M-RoPE 调用在
固定位置显式传 `None`。新增参数都追加在 keyword-only 尾部，因此既有 Torch 调用表达式不需要改变。
Python public API 的 `q_out_dtype` 类型为 `torch.dtype`，默认 `torch.float8_e4m3fn`；注册 schema 使用
`ScalarType? q_out_dtype=None`，由 wrapper 将 `None` 归一化为该默认值。Torch C++ bridge 接收
`at::ScalarType` 并据此创建 `qOut`，不把 PyTorch ScalarType 的内部整数编码传给 ACLNN。
`q_quant_mode` 缺省为 `PerTokenPerHead`；ACLNN 的 `qQuantMode=nullptr` 或空串按默认值处理。既有 ACLNN
C 调用只需在固定签名中补 `mropePositionOptional=nullptr`、`mropeSectionOptional=nullptr` 和
`qQuantMode=PerTokenPerHead` 三个新增参数位，并传入 FP8 E4M3FN 的 `qOut`。

### 2.5 Layout 与输出 shape

`layout_qkv` 和 `layout_q_out` 对 shape 的解释如下：

| `layout_qkv` | `layout_q_out` | `qkv` shape | `q_out` shape | `q_scale` shape |
|---------------|----------------|-------------|---------------|-----------------|
| `NTD` | `NTD` | `[Nq + Nk + Nv, T, D]` | `[Nq, T, D]` | `[Nq, T]` |
| `TND` | `TND` | `[T, Nq + Nk + Nv, D]` | `[T, Nq, D]` | `[T, Nq]` |
| `TND` | `NTD` | `[T, Nq + Nk + Nv, D]` | `[Nq, T, D]` | `[Nq, T]` |

`NTD->TND` 不支持，由 checker 拒绝。RoPE 场景支持表中三种组合；M-RoPE 只支持
`TND->TND`。M-RoPE 的 Torch `q_scale` 运行值为 `None`，ACLNN `qScaleOptional` 为 `nullptr`。

## 3. Shape 符号

| 符号 | 含义 |
|------|------|
| `T` | 当前请求 token 数 |
| `Batch` | RoPE 场景的 batch 数，满足 `query_start_loc.shape[0] = Batch + 1`；M-RoPE 不使用该位置源 |
| `Nq` | Q head 数 |
| `Nk` | K head 数 |
| `Nv` | V head 数 |
| `qkvN` | qkv 输入 N 轴长度，`qkvN = Nq + Nk + Nv` |
| `D` | head dimension，固定为 128 |
| `MaxSeqLen` | `cos_sin` 第一维，需要覆盖所有 batch 的实际 RoPE position |
| `local_len(b)` | 本次调用内第 `b` 个 batch 的 token 数，等于 `query_start_loc[b + 1] - query_start_loc[b]` |
| `history_offset(b)` | 第 `b` 个 batch 的 RoPE 历史偏移，等于 `seq_lens[b] - local_len(b)` |
| `BlockNum` | cache block 数 |
| `BlockSize` | cache 单 block token 数 |

`Nq/Nk/Nv` 不由 kernel 从 tensor shape 推导，而由必选属性 `head_nums = [Nq, Nk, Nv]` 给出。host checker 需要按 `layout_qkv` 校验：

```text
layout_qkv == NTD: qkv.shape == [Nq + Nk + Nv, T, D]
layout_qkv == TND: qkv.shape == [T, Nq + Nk + Nv, D]
k_cache.shape == [BlockNum, Nk, BlockSize, D]
v_cache.shape == [BlockNum, Nv, BlockSize, D]
k_scale_cache.shape == [BlockNum, Nk, BlockSize, 1]
ROPE_FP8_QK: v_scale.shape == [Nv]
MROPE_BF16_Q_INT8_K: v_scale.shape == [Nv,D]
```

首期 arch35 实现的 RoPE 与 M-RoPE 场景统一只支持：

```text
0 < Nq <= 64
Nq == 8 * Nk
Nk == Nv
```

接口和 `head_nums` 仍保留独立的 `Nq/Nk/Nv` 字段，便于后续扩展；不满足上述任一关系时不进入正常
kernel 路径，由 checker 或 tiling 返回不支持。

## 4. Q/K/V 拆分

`nq/nk/nv` 是 Q/K/V 分支内部的局部 N 轴下标；`n_qkv` 是 `qkv` 输入的全局 N 轴下标。kernel 使用由 `layout_qkv` 决定的 GM 逻辑 stride：

```text
layout_qkv == NTD:
    qkvGmStrideT = D
    qkvGmStrideN = T * D

layout_qkv == TND:
    qkvGmStrideT = qkvN * D
    qkvGmStrideN = D
```

kernel 统一读取：

```text
qkv_value(t,n_qkv,d) = qkv_gm[n_qkv * qkvGmStrideN + t * qkvGmStrideT + d]
```

拆分公式应写成局部下标到 qkv N 轴下标的映射：

```text
n_qkv = nq
Q[t,nq,d] = qkv_value(t,n_qkv,d)

n_qkv = Nq + nk
K[t,nk,d] = qkv_value(t,n_qkv,d)

n_qkv = Nq + Nk + nv
V[t,nv,d] = qkv_value(t,n_qkv,d)
```

其中：

```text
0 <= nq < Nq
0 <= nk < Nk
0 <= nv < Nv
0 <= d < D
```

Q/K 进入 RMSNorm + RoPE + 共享 `rotation` matmul + dynamic quant。V 不参与 RMSNorm/RoPE/rotation，只做 `v * v_scale` 后写 `v_cache`。V cache 写回属于后处理语义，但不依赖 Cube 输出。当前 V-first kernel 先发起 V vector quant，再执行 Q/K RMSNorm/RoPE，随后写 `v_cache`、搬运 Q/K AL1 并发送 `A_READY`；V-first 不表示 `v_cache` 写回先于 Q/K preprocess。

## 5. RMSNorm 公式

对 Q/K 的每个 `[t,h,:]` row 独立计算。数学公式中使用 `X` 表示 Q/K 拼接后的 FP32 输入：

```text
T_set = {0, 1, ..., T - 1}
```

```text
H_qk_set = {0, 1, ..., Nq + Nk - 1}
```

```text
D_set = {0, 1, ..., D - 1}
```

```text
X[t,h,j] =
  fp32(Q[t,h,j])       if 0 <= h < Nq
  fp32(K[t,h-Nq,j])    if Nq <= h < Nq + Nk
```

上述公式的下标范围为：

```text
t in T_set, h in H_qk_set, j in D_set
```

平方和张量 `S` 的数学 shape 为：

```text
S shape: [T, Nq + Nk]
```

`S` 定义为：

```text
S[t,h] = sum over j in [0, D): X[t,h,j] * X[t,h,j]
```

工程实现中的 `sum_sq[t,h]` 对应 `S_{t,h}`；如果实现张量保留尾部 singleton 维，则 `sum_sq[t,h,0]` 对应 `S_{t,h}`。

RMS：

```text
R[t,h] = sqrt(S[t,h] / D + epsilon)
```

Gamma：

先定义两组 gamma 数学向量，它们分别由工程输入 `q_gamma` 和 `k_gamma` 承载：

```text
GammaQ shape: [D]
GammaK shape: [D]
```

再定义拼接到 Q/K head 维后的 gamma 张量 `G`：

```text
G[h,j] =
  GammaQ[j]    if 0 <= h < Nq
  GammaK[j]    if Nq <= h < Nq + Nk
```

输出：

```text
Y[t,h,j] = X[t,h,j] / R[t,h] * G[h,j]
```

工程实现中的 `y_norm[t,h,j]` 对应 `Y_{t,h,j}`。

RMSNorm 计算过程使用 FP32。

## 6. RoPE 公式

`query_start_loc` 为当前调用内 token 的 local 前缀和：

```text
query_start_loc[0] = 0
query_start_loc[Batch] = T
local_len(b) = query_start_loc[b + 1] - query_start_loc[b]
```

对全局 token `t`，找到：

```text
b(t) = unique b such that query_start_loc[b] <= t < query_start_loc[b + 1]
local_pos(t) = t - query_start_loc[b(t)]
history_offset(b) = seq_lens[b] - local_len(b)
s(t) = history_offset(b(t)) + local_pos(t)
```

RoPE 读取：

```text
cos_sin_row = cos_sin[s(t), :]
```

设：

```text
half = D / 2
cos_i = cos_sin[s(t), i]
sin_i = cos_sin[s(t), i + half]
```

原始公式：

```text
rope_out[t,h,i]        = y_norm[t,h,i]        * cos_i - y_norm[t,h,i+half] * sin_i
rope_out[t,h,i+half]   = y_norm[t,h,i+half]   * cos_i + y_norm[t,h,i]      * sin_i
```

向量化公式：

```text
y_low  = y_norm[t,h,0:half]
y_high = y_norm[t,h,half:D]
cos    = cos_sin[s(t),0:half]
sin    = cos_sin[s(t),half:D]

out_low  = y_low  * cos - y_high * sin
out_high = y_high * cos + y_low  * sin
```

RoPE 计算过程使用 FP32，进入 Cube 前 cast 到 BF16。

## 7. Rotation Matmul 公式

`rotation` 不参与 RoPE 公式，它在 RoPE 后作为 Q/K 分支共享的右矩阵参与 matmul：

```text
q_after_cube[t,nq,j] = sum_{r=0}^{D-1} q_rope_out[t,nq,r] * rotation[r,j]
k_after_cube[t,nk,j] = sum_{r=0}^{D-1} k_rope_out[t,nk,r] * rotation[r,j]
```

随后按 `[Q, K]` head 顺序拼接得到后处理输入：

```text
qk_after_cube[t,h,j] =
  q_after_cube[t,h,j]       if 0 <= h < Nq
  k_after_cube[t,h-Nq,j]    if Nq <= h < Nq + Nk
```

输入：

```text
q_rope_out:   [T, Nq, D]
k_rope_out:   [T, Nk, D]
rotation:   [D, D]
q_after_cube: [T, Nq, D]
k_after_cube: [T, Nk, D]
qk_after_cube:[T, Nq + Nk, D]
```

首期按 Q/K 两段分别发起 dense Cube matmul，但右矩阵使用同一个外部输入 `rotation`，不启用 FWHT。

## 8. V Quant 与 Cache 写回

V 分支的 scale 语义由场景决定：

```text
ROPE_FP8_QK:
  v_out[t,nv,d] = cast_to_e4m3fn_sat_rint(float32(V[t,nv,d]) * v_scale[nv])

MROPE_BF16_Q_INT8_K:
  v_out[t,nv,d] = cast_to_e4m3fn_sat_rint(float32(V[t,nv,d]) * v_scale[nv,d])
```

`v_scale` 是乘子，任一 scale 元素为 0 均合法。M-RoPE 不允许把 `[Nv]` 隐式广播为 `[Nv,D]`，
也不允许 RoPE 接受 `[Nv,D]`；checker 与 base tiling 必须按已经解析出的场景校验精确 shape。

Cache 地址：

```text
slot = slot_mapping[t]
block_id = slot / BlockSize
block_offset = slot % BlockSize
```

写回：

```text
v_cache[block_id,nv,block_offset,d] = v_out[t,nv,d]
```

## 9. Q/K Dynamic Quant 与写回

本节统一描述 Q/K 的逐 token、逐 head 动态量化，不在量化公式中区分位置编码场景。
`q_quant_mode=NoQuant` 的 BF16 Q 路径见 11.4。

令 $A\in\{Q,K\}$，$C^A\in\mathbb R^{T\times N_A\times D}$ 为对应的 FP32 Cube 输出，目标量化类型为
$\tau_A\in\{\mathrm{FP8\ E4M3FN},\mathrm{INT8}\}$。定义目标正向量化上限

$$
M_{\tau_A}=
\begin{cases}
448, & \tau_A=\mathrm{FP8\ E4M3FN},\\
127, & \tau_A=\mathrm{INT8}.
\end{cases}
$$

两种目标类型使用同一套量化公式，只有 scale 分母 $M_{\tau_A}$ 不同：

$$
\begin{aligned}
m^A_{u,n}&=\max_{0\le d<D}\left|C^A_{u,n,d}\right|,\\
\alpha^A_{u,n}&=\frac{m^A_{u,n}}{M_{\tau_A}},\\
\widehat C^A_{u,n,d}
&=\operatorname{cast}_{\tau_A}
  \!\left(\frac{C^A_{u,n,d}}{\alpha^A_{u,n}}\right).
\end{aligned}
$$

$\operatorname{cast}_{\tau_A}$ 按目标类型执行 RINT 和饱和转换；FP8 E4M3FN 的正向上限为 448，
INT8 的对称目标范围为 $[-127,127]$。

数学符号与工程张量的对应关系：

| 数学符号 | 工程张量或中间量 |
|----------|------------------|
| $C^A$ | `qk_after_cube` 的 Q 或 K 段 |
| $m^A$ | `max_abs` |
| $\alpha^A$ | `scale_out` |
| $\widehat C^A$ | `qk_quant` |

对需要量化的 Q，内部 $\widehat C^Q/\alpha^Q$ 视图保持 token-major。若 `layout_q_out=NTD`，则

$$
Q^{\mathrm{out}}_{n,u,d}=\widehat C^Q_{u,n,d},
\qquad
Q^{\mathrm{scale}}_{n,u}=\alpha^Q_{u,n}.
$$

若 `layout_q_out=TND`，则

$$
Q^{\mathrm{out}}_{u,n,d}=\widehat C^Q_{u,n,d},
\qquad
Q^{\mathrm{scale}}_{u,n}=\alpha^Q_{u,n}.
$$

令 $b_u$ 和 $o_u$ 分别为 `slot_mapping[u]` 对应的 cache block 与 block 内偏移，则 K 写回为

$$
K^{\mathrm{cache}}_{b_u,n,o_u,d}=\widehat C^K_{u,n,d},
\qquad
K^{\mathrm{scale}}_{b_u,n,o_u,0}=\alpha^K_{u,n}.
$$

## 10. 校验约束摘要

1. 两个支持场景都要求 `rotation != nullptr`。
2. 两个支持场景都要求 `v_scale != nullptr`。
3. `rotation.shape == [D,D]`。
4. RoPE 要求 `v_scale.shape == [Nv]`；M-RoPE 要求 `v_scale.shape == [Nv,D]`。
5. 仅 RoPE 场景要求 `query_start_loc.shape == [Batch + 1]`。
6. 仅 RoPE 场景要求 `seq_lens.shape == [Batch]`，dtype 为 int32。
7. 仅 RoPE 场景且 host 可读 `query_start_loc` 时，建议校验 `query_start_loc[0] == 0`、单调不降、
   `query_start_loc[Batch] == T`。
8. RoPE 场景调用方必须保证 `seq_lens[b] >= local_len(b)` 且 `seq_lens[b] <= cos_sin.shape[0]`；若
   `seq_lens[b]` 小于该 batch 本次调用 token 数，行为未定义。M-RoPE 不读取这两个 anchor。
9. `head_nums` 必须存在，长度为 3，值为正整数 `[Nq,Nk,Nv]`。
10. 两个场景统一要求 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`；不满足任一关系时返回不支持或参数错误。
11. `qkv` 按 `layout_qkv` 解释出的逻辑 N 轴长度等于 `sum(head_nums)`；cache 与 `v_scale` 第 0 维均与
    `head_nums[2]` 一致，M-RoPE `v_scale` 第 1 维还必须等于 `D`。
12. `D` 固定为 128，并满足 Cube 对齐要求。
13. `layout_q_out` 支持 `layout_q_out == layout_qkv` 和 `layout_qkv=TND,layout_q_out=NTD`；`layout_qkv=NTD,layout_q_out=TND` 当前拒绝。

## 11. `MROPE_BF16_Q_INT8_K` 组合模式

本节展开第 2 节接口中的 M-RoPE 组合模式，说明位置编码和输出类型的场景约束；第 9 节统一动态量化
公式对所有需要量化的 Q/K 生效。完整展开见 `11_mrope_bf16_int8_extension_design.md`。

### 11.1 原型增量

既有 input index 8/9 的 `query_start_loc/seq_lens` 在 OpDef 中从 REQUIRED 改为 OPTIONAL，索引不变；
输入尾部追加可选 `mrope_position: int32[T,3]`，第 0 维为 token、末维 3 列依次为 T/H/W；属性尾部追加可选
`mrope_section: ListInt[3]`、`q_quant_mode: String` 和 `q_out_dtype: Int`。`q_quant_mode` 默认
`PerTokenPerHead`，当前只支持 `PerTokenPerHead` 和 `NoQuant`。`q_out_dtype` 默认
`ge::DT_FLOAT8_E4M3FN`，当前只支持
`ge::DT_FLOAT8_E4M3FN` 和 `ge::DT_BF16`。

ACLNN 对应形参为 `queryStartLocOptional/seqLensOptional`，Torch schema 对应
无默认值的 `Tensor? query_start_loc, Tensor? seq_lens`，M-RoPE 调用在固定参数位显式传 `None`；
二者后面仍有必选 `head_nums`，不能直接增加默认值。接口不提供单独 scene 参数；由 checker/base
tiling 根据 optional 实参 presence 识别场景。

`mrope_section=[t,h,w]` 在原型中是 ListInt 属性，不是 int32 tensor。其三项具有 int32 语义值域，但 ACLNN 使用
`const aclIntArray *`，物理承载元素为 int64。令 $D_{\mathrm{half}}=D/2$，并将三项记为
$\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})$，则 H/W 容量上限为

$$
C_{\mathrm H}=\left\lfloor\frac{D_{\mathrm{half}}+1}{3}\right\rfloor,
\qquad
C_{\mathrm W}=\left\lfloor\frac{D_{\mathrm{half}}}{3}\right\rfloor,
\qquad s_{\mathrm T}\ge 0,
\qquad 0\le s_{\mathrm H}\le C_{\mathrm H},
\qquad 0\le s_{\mathrm W}\le C_{\mathrm W},
\qquad s_{\mathrm T}+s_{\mathrm H}+s_{\mathrm W}\le D_{\mathrm{half}}.
$$

D128 下 $(C_{\mathrm H},C_{\mathrm W})=(21,21)$，三项之和不超过 64。
$s_{\mathrm T}$ 没有独立 lane 容量上限，通过非负和总和校验后不参与计算。

`q_scale` 为固定 index 1 的 REQUIRED output：

```text
RoPE PerTokenPerHead + FP8 Q: Torch/ACLNN 返回 fp32 q_scale，shape 由 layout_q_out 决定
M-RoPE NoQuant + BF16 Q: Torch 运行值为 None，ACLNN qScaleOptional 传 nullptr
```

后续 `k_cache/v_cache/k_scale_cache` 的 output index 仍为 2/3/4。Torch inplace/functional 返回 schema
的 q_scale 位置均声明为 required `Tensor`，M-RoPE 运行值仍为 `None`。
OpDef 每个 tensor 的 DataType/Format/UnknownShapeFormat 都使用等长两项相关向量；缺席的 optional
anchor 也保留对应项，所有 ND format 均重复两次。

### 11.2 单一组合门禁

合法调用只有两类：

```text
ROPE_FP8_QK:
  query_start_loc = int32[Batch+1]
  seq_lens = int32[Batch]
  mrope_position absent
  mrope_section absent
  q_quant_mode = PerTokenPerHead
  q_out_dtype = FP8 E4M3FN
  q_out = FP8 E4M3FN
  q_scale = fp32 tensor
  k_cache = FP8 E4M3FN

MROPE_BF16_Q_INT8_K:
  query_start_loc absent
  seq_lens absent
  mrope_position = int32[T,3]  // row u = [T,H,W] positions of token u
  mrope_section = ListInt[3], H/W bounds [C_H,C_W] and sum <= D_half as defined in 11.1
  q_quant_mode = NoQuant
  q_out_dtype = BF16
  q_out = BF16
  q_scale = Torch runtime None; ACLNN nullptr
  k_cache = int8
  layout_qkv = TND
  layout_q_out = TND
```

任意交叉组合均拒绝，包括 `PerTokenPerHead + BF16` 和 `NoQuant + FP8`。M-RoPE 场景不生成其他
layout key；V 仍按当前 FP8 E4M3FN 路径计算。
具体判定方向是先检查 effective presence：空 `mrope_section=[]` 与未传属性等价。`query_start_loc/seq_lens`
同时存在、`mrope_position` 缺席且 `mrope_section` 为空时得到 RoPE 候选；前一组同时缺席且
`mrope_position` 与非空 `mrope_section` 同时存在时得到 M-RoPE 候选；任一 pair 部分存在、两组同时存在
或同时缺失均拒绝。候选场景随后还必须通过量化模式、dtype/output/cache/layout 相关校验，`mrope_section[0]` 不参与
场景判定。

### 11.3 交织 M-RoPE

令 $C=\mathrm{cos\_sin}$、$P=\mathrm{mrope\_position}$、$D_{\mathrm{half}}=D/2$，并定义
$\iota(\mathrm T)=0$、$\iota(\mathrm H)=1$、$\iota(\mathrm W)=2$。$P$ 的第 0 维是 token，
第 1 维是 T/H/W 列。对 token $u$、轴
$a\in\{\mathrm T,\mathrm H,\mathrm W\}$ 和列 $d$，三路完整原始矩阵为

$$
R_{a,u,d}=C_{P_{u,\iota(a)},d},
\qquad 0\le u<T,\quad 0\le d<D.
$$

每个 token 的 UB 原始向量为

$$
\operatorname{raw}_{u}
=R_{\mathrm T,u,:}\mathbin\Vert R_{\mathrm H,u,:}\mathbin\Vert R_{\mathrm W,u,:}.
$$

将 `mrope_section=[t,h,w]` 记为 $\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})$。对局部
half lane $\ell\in\{0,\ldots,D_{\mathrm{half}}-1\}$，定义

$$
r(\ell)=\left\lfloor\frac{\ell}{3}\right\rfloor,
\qquad \rho(\ell)=\ell\bmod 3,
$$

$$
\sigma(\ell)=
\begin{cases}
\mathrm H, & \rho(\ell)=1\ \land\ r(\ell)<s_{\mathrm H},\\
\mathrm W, & \rho(\ell)=2\ \land\ r(\ell)<s_{\mathrm W},\\
\mathrm T, & \text{其他情况}.
\end{cases}
$$

最终有效 cos/sin 为

$$
\widehat c_{u,\ell}=R_{\sigma(\ell),u,\ell},
\qquad
\widehat s_{u,\ell}=R_{\sigma(\ell),u,D_{\mathrm{half}}+\ell}.
$$

对应的 Gather 元素索引为

$$
g_{\ell}=\iota\!\left(\sigma(\ell)\right)D+\ell;
$$

cos 使用偏移 $g_{\ell}$，sin 使用偏移 $D_{\mathrm{half}}+g_{\ell}$。

这里 `mrope_section[0]` 完全不生效，T 路是完整基底；超出 H/W 覆盖范围的 lane 仍读取 T 路。D128 使用
64 项 uint32 id，cos 从 raw base Gather，sin 复用同一 id 并从 `raw base` 偏移 $D_{\mathrm{half}}$ 后 Gather；索引不得乘
`sizeof(float)`。之后 half-split RoPE 公式不变。

### 11.4 BF16 Q 与 INT8 K

`q_quant_mode=NoQuant` 且 `q_out_dtype=BF16` 时，Q 不执行 dynamic quant。若 Cube 输出为 $C^Q$，则

$$
Q^{\mathrm{out}}_{u,n,d}=\operatorname{RNE}_{\mathrm{BF16}}
\!\left(C^Q_{u,n,d}\right).
$$

该模式下 Torch 的 q_scale 运行值为 `None`，ACLNN 的 `qScaleOptional` 为 `nullptr`。

K 使用第 9 节统一动态量化公式，取 $\tau_K=\mathrm{INT8}$、$M_{\tau_K}=127$；量化结果
$K^{\mathrm{int8}}=\widehat C^K$ 和 $\alpha^K$ 仍按 `slot_mapping` 写 `k_cache/k_scale_cache`。

### 11.5 InferShape 边界

InferShape 不读取 `query_start_loc/seq_lens/mrope_position/mrope_section/q_quant_mode/q_out_dtype` 的内容或
presence 做模式校验。五个 required output 按固定 logical index 直接通过 `GetOutputShape()` 取得并判空，
q_scale 始终按 `layout_q_out` 推导 shape；InferDtype 始终将 output index 1 设为 FP32。InferShape 继续保留
`NTD->TND` 拒绝和逻辑 `qkvN >= Nq`，其余完整合同不下沉。
