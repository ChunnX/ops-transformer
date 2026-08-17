# Golden 设计

## 1. 目标

Golden 用于校验 host 侧 shape 逻辑、kernel 端到端数值、cache 写回地址和 FP8 quant 行为。Golden 必须直接体现完整计算链路，Q/K 两路 RoPE 输出在 Cube 阶段共用同一个外部 `rotation` 右矩阵，不能简化成特殊 Hadamard 形式。

## 2. Golden 输入

Python/Numpy golden 输入与算子一致：

```text
qkv: [Nq + Nk + Nv, T, D], bf16 simulated by fp32 cast/round policy
q_gamma: [D], fp32
k_gamma: [D], fp32
cos_sin: [MaxSeqLen, D], fp32
slot_mapping: [T], int32
k_cache: [BlockNum, Nk, BlockSize, D], ml_dtypes.float8_e4m3fn
v_cache: [BlockNum, Nv, BlockSize, D], ml_dtypes.float8_e4m3fn
k_scale_cache: [BlockNum, Nk, BlockSize, 1], fp32
query_start_loc: [Batch + 1], int32 local prefix sum for current invocation
seq_lens: [Batch], int32 actual sequence lengths after current invocation
rotation: [D, D], bf16 simulated, Q/K 分支共享的 Cube 右矩阵
v_scale: [Nv], fp32 for RoPE; [Nv,D], fp32 for M-RoPE
head_nums: [Nq, Nk, Nv], ListInt
layout_qkv: "NTD" or "TND", default "TND", case-sensitive
layout_q_out: "NTD" or "TND", default "NTD", case-sensitive; current ACLNN path supports NTD->NTD, TND->TND and TND->NTD
epsilon: fp32
```

Golden 中 `Nq/Nk/Nv` 统一来自 `head_nums`，并按 `layout_qkv` 校验输入：`NTD` 时 `qkv.shape == [Nq + Nk + Nv, T, D]`，`TND` 时 `qkv.shape == [T, Nq + Nk + Nv, D]`。`layout_q_out` 用于解释 `q_out/q_scale` 的输出形状；空值按 `layout_qkv=TND,layout_q_out=NTD` 处理。当前端到端正常路径生成 `NTD->NTD`、`TND->TND` 和 `TND->NTD` case，`NTD->TND` 用于接口拒绝类用例。

Golden 计算函数可以保留对独立 `Nv` 的泛化实现，便于后续扩展和单元验证；但首期 arch35 的 RoPE
与 M-RoPE 实现统一只支持 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`。DiscussionForum 端到端
NPU golden/TTK 的正常路径必须满足完整 head 合同；超出 Nq 上限、Q/K 比例不符或 K/V head 不等只进入
checker/tiling 拒绝类用例，不进入 kernel 数值对比。

## 3. Golden 计算顺序

### 3.1 Split Q/K/V

```python
# Nq/Nk/Nv are local head counts from head_nums.
# qkv fused-head axis is split as [Q heads, K heads, V heads].
q = np.transpose(qkv[0:Nq, :, :], (1, 0, 2))
k = np.transpose(qkv[Nq:Nq + Nk, :, :], (1, 0, 2))
v = np.transpose(qkv[Nq + Nk:Nq + Nk + Nv, :, :], (1, 0, 2))
```

### 3.2 RMSNorm

```python
q_sum_sq = np.sum(q.astype(np.float32) * q.astype(np.float32), axis=-1, keepdims=True)
k_sum_sq = np.sum(k.astype(np.float32) * k.astype(np.float32), axis=-1, keepdims=True)
q_rms = np.sqrt(q_sum_sq / D + epsilon)
k_rms = np.sqrt(k_sum_sq / D + epsilon)

q_norm = q / q_rms * q_gamma
k_norm = k / k_rms * k_gamma
```

Q/K 必须分别使用 `q_gamma`、`k_gamma`，golden 中不要先把 Q/K 拼接后再用一个模糊的 `rms` 变量表达。

### 3.3 RoPE

对每个 token `t`：

```python
b = searchsorted(query_start_loc, t, side="right") - 1
local_len = query_start_loc[b + 1] - query_start_loc[b]
local_pos = t - query_start_loc[b]
s = seq_lens[b] - local_len + local_pos
cos = cos_sin[s, 0:D//2]
sin = cos_sin[s, D//2:D]
```

对 Q/K 分支分别执行同一套 half-split RoPE，并在 golden 中显式 materialize `q_rope_out`、`k_rope_out`。注意这里不能命名成 `rotation`，因为这个名字已经表示外部传入的 Cube 右矩阵：

```python
def apply_half_split_rope(y, cos_sin, query_start_loc, seq_lens):
    low = y[..., 0:D//2]
    high = y[..., D//2:D]
    out_low = low * cos - high * sin
    out_high = high * cos + low * sin
    return concat(out_low, out_high, axis=-1)

q_rope_out = apply_half_split_rope(q_norm, cos_sin, query_start_loc, seq_lens)
k_rope_out = apply_half_split_rope(k_norm, cos_sin, query_start_loc, seq_lens)
```

### 3.4 Rotation Matmul

Golden 必须使用外部传入的单个 `rotation`。Q/K 两路先各自右乘同一个矩阵，再按 `[Q, K]` head 顺序拼接：

```python
q_after_cube[t, nq, :] = q_rope_out[t, nq, :] @ rotation
k_after_cube[t, nk, :] = k_rope_out[t, nk, :] @ rotation
qk_after_cube = concat(q_after_cube, k_after_cube, axis=1)
```

`rotation` 可随机生成；不要求是标准 Hadamard。若测试希望覆盖 Hadamard，可额外构造 Hadamard-like value，但计算仍按 dense matmul。

### 3.5 V Quant

```python
if scene == "ROPE_FP8_QK":
    v_scaled[t,nv,d] = float32(v[t,nv,d]) * v_scale[nv]
else:  # MROPE_BF16_Q_INT8_K
    v_scaled[t,nv,d] = float32(v[t,nv,d]) * v_scale[nv,d]
v_out = cast_to_e4m3fn_sat_rint(v_scaled)
```

任一 `v_scale` 元素为 0 时均按公式计算。Golden 必须校验精确 shape，不能在 M-RoPE 中利用 NumPy
broadcasting 接受 `[Nv]`，也不能在 RoPE 中接受 `[Nv,D]`。

### 3.6 Q/K Dynamic Quant

```python
max_abs = np.max(np.abs(qk_after_cube), axis=-1)
scale = max_abs / FP8_E4M3FN_MAX
qk_div = qk_after_cube / scale[..., None]
qk_quant = cast_to_e4m3fn_sat_rint(qk_div)
```

该公式对齐 arch35 `dynamic_quant` FP8 E4M3FN 路径。

### 3.7 Cache 写回

```python
slot = slot_mapping[t]
block_id = slot // BlockSize
block_offset = slot % BlockSize

v_cache_ref[block_id, nv, block_offset, d] = v_out[t,nv,d]
k_cache_ref[block_id, nk, block_offset, d] = qk_quant[t,Nq+nk,d]
k_scale_cache_ref[block_id, nk, block_offset, 0] = scale[t,Nq+nk]
```

同一次调用内若 `slot_mapping` 出现重复 slot，会对同一 cache 地址产生重复写，最终写入顺序和结果未定义。Golden 的正常数值用例应保证 slot 唯一；重复 slot 只适合放入约束说明或负向用例，不做端到端数值对比。

Q 输出：

```python
if layout_q_out == "NTD":
    q_out_ref[nq,t,d] = qk_quant[t,nq,d]
    q_scale_ref[nq,t] = scale[t,nq]
else:
    q_out_ref[t,nq,d] = qk_quant[t,nq,d]
    q_scale_ref[t,nq] = scale[t,nq]
```

`qk_quant` 和 `scale` 是 golden 内部 token-major 中间张量；`q_out_ref` 和
`q_scale_ref` 是对外普通输出，N/T 轴顺序由 `layout_q_out` 决定。当前正常路径覆盖 `layout_q_out == layout_qkv` 以及 `layout_qkv=TND,layout_q_out=NTD`。

## 4. FP8 E4M3FN Cast

Golden 需要一个统一的：

```python
cast_to_e4m3fn_sat_rint(x)
```

要求：

1. RINT round。
2. SAT 饱和。
3. 最大有限值使用 `FP8_E4M3FN_MAX`，其值为 `448`。
4. dtype 为 `ml_dtypes.float8_e4m3fn`。

Golden 直接依赖 `ml_dtypes.float8_e4m3fn` 做 FP8 dtype cast；如果需要和
kernel dump 做 bit-level 比较，再对 FP8 array 使用 `view(np.uint8)` 或统一
helper 转成 storage byte。

## 5. 推荐 Case

### 5.1 典型 Shape 空间

Golden 需要覆盖典型 shape 空间。这里的 `B` 表示 batch 数，`S_b` 表示第 `b` 个 batch 的 token 数，`S` 表示单 batch 场景的 token 数或多 batch 场景下的 `max_b(S_b)`，`D` 固定为典型 head dim：

```text
B  in {1, 2}
head_nums in {
  [8, 1, 1], [16, 2, 2], [24, 3, 3], [32, 4, 4],
  [40, 5, 5], [48, 6, 6], [56, 7, 7], [64, 8, 8]
}
S  in [1, 32k]
D  = 128
```

对应张量关系：

```text
Batch = B
T = sum_b(local_S_b)
query_start_loc = [0, local_S_0, local_S_0 + local_S_1, ..., T]
seq_lens = [actual_S_0, actual_S_1, ...]
MaxSeqLen >= max_b(actual_S_b)
qkv.shape = [Nq + Nk + Nv, T, 128] if layout_qkv == "NTD" else [T, Nq + Nk + Nv, 128]
cos_sin.shape = [MaxSeqLen, 128]
rotation.shape = [128, 128]
k_cache.shape = [BlockNum, Nk, BlockSize, 128]
v_cache.shape = [BlockNum, Nv, BlockSize, 128]
k_scale_cache.shape = [BlockNum, Nk, BlockSize, 1]
v_scale.shape = [Nv]  # RoPE
q_out.shape = [Nq, T, 128] if layout_q_out == "NTD" else [T, Nq, 128]
q_scale.shape = [Nq, T] if layout_q_out == "NTD" else [T, Nq]
```

`Nv` 在 Golden 代码中仍必须始终从 `head_nums[2]` 获取，不能由 `Nk` 隐式推导。正常 NPU golden/TTK
用例必须同时满足 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`；不满足任一关系的用例只验证
checker/tiling 拒绝行为，不做端到端数值对比。

建议将典型 shape 拆成分层用例：

| 类型 | Shape 选择 | 目的 |
|------|------------|------|
| smoke | `B=1, S=1, head_nums=(8,1,1), D=128` | 最小路径，便于 debug |
| normal | `B=1, S in {128, 1024}, head_nums in {(16,2,2),(24,3,3),(32,4,4)}, D=128` | 常规数值和 cache 写回；包含非 2 的幂 head |
| multi-batch | `B=2, local_S_0 != local_S_1, head_nums in {(16,2,2),(32,4,4)}, D=128` | `query_start_loc` local 前缀和、`seq_lens` 历史偏移与 RoPE 位置 |
| multi-batch-offset | `query_start_loc=[0,17,18]`, `seq_lens=[8209,1]` | batch0 从 8192 开始、batch1 从 0 开始的 position 跳变 |
| large-seq | `B in {1,2}, max(S_b) in {8k, 16k, 32k}, head_nums in {(32,4,4),(64,8,8)}, D=128` | 长序列位置索引和大 shape 压力 |
| head-combo | 遍历从 `(8,1,1)` 到 `(64,8,8)` 的 8 组合法 tuple | `head_nums` 解析、精确 8:1 关系和 Q/K head 切分 |

大 shape case 不要求全部进入高精度 bit-level FP8 对比；可先用于 shape、地址、scale 统计和抽样数值校验。完整 bit-level 对比优先覆盖 smoke/normal/multi-batch。

### 5.2 功能 Case

| Case | 目的 |
|------|------|
| 单 batch，`T=1`，`D=128` | 最小正常路径 |
| 单 batch，多 token，多 Q/K/V head | 基础数值 |
| 多 batch local 前缀和，如 `[0,2,9]`，且 `seq_lens` 等于 local 长度 | RoPE batch 内位置 |
| 多 batch 非零历史偏移，如 `query_start_loc=[0,17,18]`、`seq_lens=[8209,1]` | RoPE batch 级起始 position 非 0 |
| `slot_mapping` 连续 | cache 基本写回 |
| `slot_mapping` 非连续和跨 block | cache 地址 |
| `slot_mapping` 重复 | 未定义行为说明或负向用例，不进入数值对比 |
| `v_scale` 包含 0 | V quant 边界 |
| random `rotation` | Q/K 分别 dense matmul，不依赖 Hadamard 特性 |
| Hadamard-like `rotation` | 覆盖用户场景，但仍按 dense 计算 |
| `Nq > 64`，例如 `(72,9,9)` | checker/tiling 拒绝，不进入 kernel 数值路径 |
| `Nq != 8 * Nk`，例如 `(16,4,4)` | checker/tiling 拒绝，不进入 kernel 数值路径 |
| `Nk != Nv`，例如 `(16,2,1)` | checker/tiling 拒绝，不进入 kernel 数值路径 |

## 6. 对比策略

1. `q_scale`、`k_scale_cache` 使用 FP32 容差。
2. FP8 输出优先 bit 对比；若工具暂不支持 bit decode，先做 dequant 后容差对比并记录风险。
3. cache 输入按 FP8 dtype 处理；golden 应从初始 cache 直接 copy 更新局部 slot，未写位置保持原值，不对整份 cache 重新 cast。
4. `query_start_loc`、`seq_lens` 非法输入不进入 golden 数值路径，由 UT/checker 覆盖。

## 7. 验收标准

1. Golden 可以独立生成 `q_out`、`q_scale`、`k_cache`、`v_cache`、`k_scale_cache`。
2. Golden 明确使用 `query_start_loc` local 前缀和和 `seq_lens` 共同计算 RoPE 位置。
3. Golden 明确使用外部 `rotation`，且 Q/K 两路分别 matmul并共享同一个右矩阵。
4. Golden 与 arch35 FP8 dynamic quant 行为保持一致，或者显式标注无法 bit-level 模拟的风险。

## 8. M-RoPE 组合模式 Golden

本节定义 M-RoPE 组合模式的 golden 合同，完整功能合同见
`11_mrope_bf16_int8_extension_design.md`。

### 8.1 输入与输出

M-RoPE 组合模式接口与输出口径：

```text
query_start_loc: Optional[Tensor]
seq_lens: Optional[Tensor]
mrope_position: int32[T,3]  # row u = [T,H,W] positions of token u
mrope_section: ListInt[3]
v_scale: float32[Nv,D]
q_quant_mode: NoQuant
q_out_dtype: torch.bfloat16
q_out: bf16
k_cache: int8
```

golden 调用层和 Torch wrapper 不接收单独 scene 参数，而是保持两个既有参数位：RoPE 场景传真实
`query_start_loc/seq_lens`、`mrope_position=None`，且 `mrope_section` 为 `None` 或空列表；M-RoPE 在前两个
位置显式传 `None`，并传入 `mrope_position` 与非空 `mrope_section`。Torch wrapper 只按
`mrope_position` 存在或 `mrope_section` 非空来构造候选输出，不校验 exact
presence signature 或 shape 合同。任一位置源 pair 部分存在、两组同时存在或同时缺失都应由 ACLNN checker
拒绝后跳过数值 golden。零元素 tensor 仍算“存在”，由 ACLNN 按 shape 合同拒绝，不能等价为 `None`；空
`mrope_section=[]` 则与未传属性等价，并作为 RoPE 回归用例覆盖。

Torch schema 和返回注解将 `q_scale` 声明为 `Tensor`。RoPE 在固定第 2 项返回 FP32
`q_scale`；M-RoPE 在同一位置的运行值为 `None`，后续 cache output 位置不变。ACLNN 的
`qScaleOptional` 在 M-RoPE 下传 `nullptr`。

### 8.2 交织 cos/sin golden

令 $C=\mathrm{cos\_sin}$、$P=\mathrm{mrope\_position}$、$D_{\mathrm{half}}=D/2$，并定义
$\iota(\mathrm T)=0$、$\iota(\mathrm H)=1$、$\iota(\mathrm W)=2$。$P$ 的每一行对应一个 token，
列 0/1/2 依次对应 T/H/W。三路完整输入为

$$
R_{a,u,d}=C_{P_{u,\iota(a)},d},
\qquad
a\in\{\mathrm T,\mathrm H,\mathrm W\},\quad 0\le u<T,\quad 0\le d<D.
$$

将 `mrope_section=[t,h,w]` 记为 $\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})$，H/W 容量上限为

$$
C_{\mathrm H}=\left\lfloor\frac{D_{\mathrm{half}}+1}{3}\right\rfloor,
\qquad
C_{\mathrm W}=\left\lfloor\frac{D_{\mathrm{half}}}{3}\right\rfloor.
$$

合法属性还必须满足

$$
s_{\mathrm T}\ge 0,
\qquad 0\le s_{\mathrm H}\le C_{\mathrm H},
\qquad 0\le s_{\mathrm W}\le C_{\mathrm W},
\qquad s_{\mathrm T}+s_{\mathrm H}+s_{\mathrm W}\le D_{\mathrm{half}}.
$$

对 $0\le\ell<D_{\mathrm{half}}$，令

$$
r(\ell)=\left\lfloor\frac{\ell}{3}\right\rfloor,
\qquad \rho(\ell)=\ell\bmod 3,
$$

并定义分段选源函数

$$
\sigma(\ell)=
\begin{cases}
\mathrm H, & \rho(\ell)=1\ \land\ r(\ell)<s_{\mathrm H},\\
\mathrm W, & \rho(\ell)=2\ \land\ r(\ell)<s_{\mathrm W},\\
\mathrm T, & \text{其他情况}.
\end{cases}
$$

golden 的有效 cos/sin 因而定义为

$$
\widehat c_{u,\ell}=R_{\sigma(\ell),u,\ell},
\qquad
\widehat s_{u,\ell}=R_{\sigma(\ell),u,D_{\mathrm{half}}+\ell}.
$$

其中 $s_{\mathrm T}$ 不得出现在 $\sigma$ 公式或场景判定中。T/H/W 三行始终
完整采集，T 路是所有未被 H/W 覆盖 lane 的基底。M-RoPE 调用的 `query_start_loc/seq_lens` 必须为 `None`，`mrope_position` 是唯一 RoPE
索引来源。

### 8.3 Q/K 后处理

在 `q_quant_mode=NoQuant`、`q_out_dtype=torch.bfloat16` 的 golden 前提下，若 Q/K 的 Cube 输出分别为
$C^Q$ 和 $C^K$，则 Q 参考值为

$$
Q^{\mathrm{ref}}_{u,n,d}=\operatorname{RNE}_{\mathrm{BF16}}
\!\left(C^Q_{u,n,d}\right),
\qquad q\_scale^{\mathrm{ref}}=\varnothing.
$$

K 的逐 row INT8 参考值为

$$
\begin{aligned}
m^K_{u,n}&=\max_{0\le d<D}\left|C^K_{u,n,d}\right|,\\
\alpha^K_{u,n}&=\frac{m^K_{u,n}}{127},\\
K^{\mathrm{ref}}_{u,n,d}
&=\operatorname{sat}_{[-127,127]}
  \!\left(\operatorname{rint}\!\left(\frac{C^K_{u,n,d}}{\alpha^K_{u,n}}\right)\right).
\end{aligned}
$$

K 的 cache scatter 和 `k_scale_cache` 写回地址沿用 RoPE 场景。

INT8 结果按有符号整数域逐元素比较。实际值与参考值都先提升到至少 int16 再作差，验收条件为
$|K^{\mathrm{actual}}-K^{\mathrm{ref}}|\le 1$，且必须 100% 元素满足；浮点 `atol/rtol` 或坏点比例参数
不得放宽该条件，dtype 不一致直接判失败。

### 8.4 V 逐 head/通道 scale

M-RoPE 令 $S^V=\mathrm{v\_scale}\in\mathbb{R}^{N_v\times D}$，V 参考值为

$$
V^{\mathrm{ref}}_{u,n,d}
=\operatorname{cast}_{\mathrm{FP8\ E4M3FN}}^{\mathrm{RINT,SAT}}
\!\left(\operatorname{float32}(V_{u,n,d})\,S^V_{n,d}\right).
$$

`v_scale[n,d]` 对 token 维广播，但不得对 D 维广播；每个 V head 的 128 个通道可以使用不同 scale。
V cache scatter 地址与 RoPE 相同。

### 8.5 新增 case

1. `mrope_section[1] != mrope_section[2]`，分别检测 H/W 边界，避免把两项误共用。
2. 只修改 `mrope_section[0]`，固定 H/W 为 20 并遍历 `{0,1,11,24}`；最后一项超过旧上限 22 但总和仍为 64，其余输入完全相同，所有输出必须完全一致。
3. 所有 M-RoPE 用例默认按 case seed 独立生成 `[0,MaxSeqLen)` 上均匀随机的 `[T,3]` token-major
   `mrope_position`，T/H/W 三列逐元素独立采样；固定包含 `T=5` 跨每 AIV cap=4。不能只用 `T=3`
   验证布局，因为 `[3,3]` 方阵无法暴露 token/axis 维颠倒。
4. `q_out` 为 BF16；Torch `q_scale` 运行值为 `None`，ACLNN `qScaleOptional` 为 `nullptr`。
   `q_quant_mode=NoQuant` 与 `q_out_dtype=BF16` 必须同时成立。
5. K INT8 的正负对称、RINT、饱和、`/127` scale 和跨 block scatter；逐元素有符号差值绝对值不超过 1，且 100% 元素通过。
6. M-RoPE 正常路径只覆盖 TND->TND；NTD->NTD、TND->NTD 和 NTD->TND 进入接口拒绝用例。
7. RoPE 场景的三条现有 layout、odd tail、多 tile、random dense rotation、V FP8 回归。
8. RoPE 场景 FP8 Q/K 回归，证明新增模板没有改变原输出。
9. D128 覆盖 H/W 上界 `[21,21]`、总和 64 的合法边界和总和 65 的拒绝路径，并验证 cos/sin 两个 half 各自最后一个有效 lane。
10. Host/API 拒绝完整 effective presence 负矩阵，其中 `M` 表示非空 `mrope_section`：
    `query_start_loc xor seq_lens`、`mrope_position xor M`、两组位置源同时存在、两组同时缺失；只接受
    `Q/S/P/M=1100` 和 `0011`，并覆盖空 `mrope_section=[]` 仍进入 RoPE，再校验各候选场景的 dtype/output/
    cache/layout 合同。
11. M-RoPE 使用每个 head、每个通道均不同且非对称的 `[Nv,D]` scale，证明实现读取
    `v_scale[n,d]`，而不是复用 `v_scale[n]` 或第 0 通道；零值边界由独立用例覆盖，不注入常规输入。
12. shape 门禁分别覆盖：M-RoPE 拒绝 `[Nv]`、`[Nv,D-1]`、`[Nv+1,D]`，RoPE 拒绝 `[Nv,D]`；
    两个场景均保持 FP32 dtype 要求。

拒绝用例补充覆盖 `mrope_section` 任一项小于 0、H/W 分别超过 21，以及三项之和超过 `D/2=64`；
第 0 项不做独立 lane 上限校验。非法 `q_quant_mode` 字符串、`PerTokenPerHead + BF16` 和
`NoQuant + FP8` 也必须拒绝。

### 8.6 M-RoPE compact NPU golden 矩阵

目标 M-RoPE 只有一条 compact direct-Q 流水和一个 TND->TND 模板。NPU golden 按输出规模组织，
所有 case 使用同一数值算法、同步协议、默认 GM cache 行为和 K/V 写回语义。

| 分组 | 选择条件 | 必测实现特征 |
|------|----------|--------------|
| compact 常规输出 | M-RoPE、TND->TND、INT8 K、NoQuant BF16 Q、`v_scale[Nv,D]` | input/raw 各四槽且计数独立、226 KiB UB、1024-token position window、逐通道 V scale、Fixpipe F32->BF16 Q 直写、独立 K rotation/K quant、`Stage -> K_CONSUMED -> Scatter` 同控制迭代、K quant slot reusable event、`FIX_OUTPUT_READY/MROPE_K_CONSUMED`（flag 1/4） |
| compact 大输出 | 同一合同，覆盖 `T=45000/65536/131072` | 与上组完全相同的 compact 流水；重点验证长序列、多 tile 与 Q/K/V/K-scale 写回 |

资源门限不满足时由 tiling 降低 `tokenTile`；最小 `tokenTile=2` 仍不可行的输入直接拒绝，不构造
第二个 M-RoPE key。整个矩阵覆盖 `T=1/4/5/17`、tail、多 tile、可复现随机位置、section-0 不变量、
H/W 不对称、cache stride/scatter 和 empty AIV slice；大输出组额外覆盖 `T=45000/65536/131072`，metadata
window 另覆盖跨 1024 token 的 refill。
输出比较对象固定为 `q_out`、`k_cache`、`v_cache` 和 `k_scale_cache`，并记录实际 tiling key。
