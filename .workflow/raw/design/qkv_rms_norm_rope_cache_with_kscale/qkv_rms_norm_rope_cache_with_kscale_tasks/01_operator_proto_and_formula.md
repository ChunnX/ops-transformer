# 算子原型与计算公式设计

## 1. 算子名

```text
QkvRmsNormRopeCacheWithKScale
```

首期平台：arch35/Ascend950。

## 2. 输入输出原型

### 2.1 必选输入

| 名称 | dtype | shape | 说明 |
|------|-------|-------|------|
| `qkv` | `bf16` | `[T, Nq + Nk + Nv, D]` | Q/K/V 融合输入 |
| `q_gamma` | `float32` | `[D]` | Q RMSNorm gamma |
| `k_gamma` | `float32` | `[D]` | K RMSNorm gamma |
| `cos_sin` | `float32` | `[MaxSeqLen, D]` | RoPE cos/sin，前 `D/2` 为 cos，后 `D/2` 为 sin |
| `slot_mapping` | `int32` | `[T]` | token 到 cache slot 的映射 |
| `k_cache` | `float8_e4m3fn` | `[BlockNum, Nk, BlockSize, D]` | 输入输出复用空间 |
| `v_cache` | `float8_e4m3fn` | `[BlockNum, Nv, BlockSize, D]` | 输入输出复用空间 |
| `k_scale_cache` | `float32` | `[BlockNum, Nk, BlockSize, 1]` | 输入输出复用空间 |
| `cu_seq_len` | `int32` | `[Batch + 1]` | batch 前缀和，`cu_seq_len[0] = 0`，`cu_seq_len[Batch] = T` |

### 2.2 可选输入

| 名称 | dtype | shape | 说明 |
|------|-------|-------|------|
| `rotation` | `bf16` | `[D, D]` | 接口 optional，但本场景 checker 强制非空 |
| `v_scale` | `float32` | `[Nv]` | 接口 optional，但本场景 checker 强制非空 |

### 2.3 输出

| 名称 | dtype | shape | 说明 |
|------|-------|-------|------|
| `q_out` | `float8_e4m3fn` | `[T, Nq, D]` | Q 分支 FP8 输出 |
| `q_scale` | `float32` | `[T, Nq]` | Q 分支 dynamic quant scale |
| `k_cache` | `float8_e4m3fn` | `[BlockNum, Nk, BlockSize, D]` | 原地更新 |
| `v_cache` | `float8_e4m3fn` | `[BlockNum, Nv, BlockSize, D]` | 原地更新 |
| `k_scale_cache` | `float32` | `[BlockNum, Nk, BlockSize, 1]` | 原地更新 |

### 2.4 属性

| 名称 | dtype | 默认值 | 说明 |
|------|-------|--------|------|
| `head_nums` | `ListInt` | 必选 | 长度固定为 3，依次为 `[Nq, Nk, Nv]`；host tiling 写入 tiling data 并传递给 kernel |
| `epsilon` | `float32` | `1e-6` | RMSNorm 防止除零，由 tiling data 传入 kernel |

## 3. Shape 符号

| 符号 | 含义 |
|------|------|
| `T` | 当前请求 token 数 |
| `Batch` | batch 数，满足 `cu_seq_len.shape[0] = Batch + 1` |
| `Nq` | Q head 数 |
| `Nk` | K head 数 |
| `Nv` | V head 数 |
| `D` | head dimension，当前典型值 128 |
| `MaxSeqLen` | 最大 batch 内长度，语义上等于 `max(cu_seq_len[b + 1] - cu_seq_len[b])` |
| `BlockNum` | cache block 数 |
| `BlockSize` | cache 单 block token 数 |

`Nq/Nk/Nv` 不由 kernel 从 tensor shape 推导，而由必选属性 `head_nums = [Nq, Nk, Nv]` 给出。host checker 需要校验：

```text
qkv.shape[1] == Nq + Nk + Nv
k_cache.shape[1] == Nk
v_cache.shape[1] == Nv
k_scale_cache.shape[1] == Nk
v_scale.shape[0] == Nv
```

首期 arch35 fast path 只支持：

```text
Nv == Nk
```

接口和 `head_nums` 仍保留独立 `Nv` 字段，便于后续扩展；当前实现中 `Nv != Nk` 不进入正常 kernel 路径，由 checker 或 tiling 返回不支持。

## 4. Q/K/V 拆分

`nq/nk/nv` 是 Q/K/V 分支内部的局部 head 下标；`h_qkv` 是 `qkv` 第二维的融合 head 下标。拆分公式应写成局部下标到融合下标的映射：

```text
h_qkv = nq
Q[t,nq,d] = qkv[t,h_qkv,d]

h_qkv = Nq + nk
K[t,nk,d] = qkv[t,h_qkv,d]

h_qkv = Nq + Nk + nv
V[t,nv,d] = qkv[t,h_qkv,d]
```

其中：

```text
0 <= nq < Nq
0 <= nk < Nk
0 <= nv < Nv
0 <= d < D
```

Q/K 进入 RMSNorm + RoPE + rotation + dynamic quant。V 不参与 RMSNorm/RoPE/rotation，只做 `v * v_scale` 后写 `v_cache`。V cache 写回属于后处理语义，但不依赖 Cube 输出，kernel 可以在 Q/K preprocess 完成并发送 `A_READY` 后提前执行。

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

`cu_seq_len` 为前缀和：

```text
cu_seq_len[0] = 0
cu_seq_len[Batch] = T
len_b = cu_seq_len[b + 1] - cu_seq_len[b]
```

对全局 token `t`，找到：

```text
b(t) = unique b such that cu_seq_len[b] <= t < cu_seq_len[b + 1]
s(t) = t - cu_seq_len[b(t)]
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

`rotation` 不参与 RoPE 公式，它在 RoPE 后作为右矩阵参与 matmul：

```text
qk_after_cube[t,h,j] = sum_{k=0}^{D-1} rope_out[t,h,k] * rotation[k,j]
```

输入：

```text
rope_out: [T, Nq + Nk, D]
rotation: [D, D]
qk_after_cube: [T, Nq + Nk, D]
```

首期按 dense Cube matmul 实现，不启用 FWHT。

## 8. V Quant 与 Cache 写回

V 分支：

```text
v_out[t,nv,d] = cast_to_e4m3fn_sat_rint(float32(V[t,nv,d]) * v_scale[nv])
```

`v_scale` 是乘子，`v_scale == 0` 合法。

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

定义 Cube 输出对应的数学张量 `C`：

```text
C shape: [T, Nq + Nk, D]
```

下标范围为：

```text
t in T_set, h in H_qk_set, j in D_set
```

其 D 方向绝对值最大值定义为：

```text
A[t,h] = max over j in [0, D): abs(C[t,h,j])
```

当前 `fp8_max` 近似取 `448`，scale 定义为：

```text
fp8_max ~= 448
```

```text
Scale[t,h] = A[t,h] / fp8_max
```

量化前的归一化中间结果：

```text
Z[t,h,j] = C[t,h,j] / Scale[t,h]
```

最终 FP8 E4M3FN 量化结果：

```text
Q_fp8[t,h,j] = cast_e4m3fn_sat_rint(Z[t,h,j])
```

数学符号与工程张量的对应关系：

| 数学符号 | 工程张量或中间量 |
|----------|------------------|
| `C` | `qk_after_cube` |
| `A` | `max_abs` |
| `Scale` | `scale_out` |
| `Q^fp8` | `qk_quant` |

该行为对齐 arch35 `dynamic_quant` FP8 E4M3FN 路径。某个 row 的 scale 为 0 时，不额外定义跳过分支。

Q 写回：

```text
q_out[t,nq,d] = qk_quant[t,nq,d]
q_scale[t,nq] = scale_out[t,nq]
```

K 写回：

```text
k_cache[block_id,nk,block_offset,d] = qk_quant[t,Nq+nk,d]
k_scale_cache[block_id,nk,block_offset,0] = scale_out[t,Nq+nk]
```

## 10. 校验约束摘要

1. `rotation != nullptr`。
2. `v_scale != nullptr`。
3. `rotation.shape == [D,D]`。
4. `v_scale.shape == [Nv]`。
5. `cu_seq_len.shape == [Batch + 1]`。
6. 若 host 可读 `cu_seq_len`，建议校验 `cu_seq_len[0] == 0`、单调不降、`cu_seq_len[Batch] == T`。
7. `head_nums` 必须存在，长度为 3，值为正整数 `[Nq,Nk,Nv]`。
8. 首期 fast path 要求 `Nv == Nk`；`Nv != Nk` 返回不支持或参数错误。
9. `qkv.shape[1] == sum(head_nums)`，cache/v_scale head 维与 `head_nums` 一致。
10. `D` 为偶数，且应满足 Cube 对齐要求；典型 `D=128`。
