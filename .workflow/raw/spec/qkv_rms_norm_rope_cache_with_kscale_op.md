# QkvRmsNormRopeCacheWithKScale 算子原型设计说明书

## 需求背景

QkvRmsNormRopeCacheWithKScale 是面向 Transformer decode/prefill 流程的融合算子，将 Q/K/V 拆分、RMSNorm、RoPE、rotation matmul、FP8 动态量化、KV cache 写回融合在一个算子中。首期交付限定为 arch35/Ascend950 平台。

计算公式：

- 1. Q/K/V 拆分：

  $$
  Q[t,nq,d] = qkv[t, nq, d], \quad 0 \le nq < N_q
  $$

  $$
  K[t,nk,d] = qkv[t, N_q + nk, d], \quad 0 \le nk < N_k
  $$

  $$
  V[t,nv,d] = qkv[t, N_q + N_k + nv, d], \quad 0 \le nv < N_v
  $$

- 2. RMSNorm（Q/K 分别使用独立 gamma）：

  $$
  S[t,h] = \sum_{j=0}^{D-1} X[t,h,j]^2
  $$

  $$
  R[t,h] = \sqrt{S[t,h] / D + \epsilon}
  $$

  $$
  Y[t,h,j] = X[t,h,j] / R[t,h] \cdot G[h,j]
  $$

  其中 $X$ 为 Q/K 拼接后的 FP32 输入，$G[h,j]$ 在 $h < N_q$ 时使用 $q\_gamma[j]$，在 $h \ge N_q$ 时使用 $k\_gamma[j]$。

- 3. RoPE（half-split，使用 cu_seq_len 前缀和定位 batch 内位置）：

  对全局 token $t$，找到 batch 内位置 $s(t) = t - cu\_seq\_len[b(t)]$，读取 $cos\_sin[s(t), :]$。

  $$
  out_{low} = y_{low} \cdot cos - y_{high} \cdot sin
  $$

  $$
  out_{high} = y_{high} \cdot cos + y_{low} \cdot sin
  $$

  其中 $y_{low} = Y[t,h,0:D/2]$，$y_{high} = Y[t,h,D/2:D]$，$cos = cos\_sin[s(t), 0:D/2]$，$sin = cos\_sin[s(t), D/2:D]$。

- 4. Rotation Matmul（RoPE 输出右乘 rotation 矩阵）：

  $$
  qk\_after\_cube[t,h,j] = \sum_{k=0}^{D-1} rope\_out[t,h,k] \cdot rotation[k,j]
  $$

- 5. V 分支量化：

  $$
  v\_out[t,nv,d] = cast\_to\_e4m3fn\_sat\_rint(float32(V[t,nv,d]) \cdot v\_scale[nv])
  $$

- 6. Q/K 动态量化（对齐 arch35 dynamic_quant FP8 E4M3FN 路径）：

  $$
  A[t,h] = \max_{j \in [0,D)} |qk\_after\_cube[t,h,j]|
  $$

  $$
  Scale[t,h] = A[t,h] / 448
  $$

  $$
  Q_{fp8}[t,h,j] = cast\_e4m3fn\_sat\_rint(qk\_after\_cube[t,h,j] / Scale[t,h])
  $$

- 7. Cache 写回（使用 slot_mapping 地址映射）：

  $$
  block\_id = slot\_mapping[t] / BlockSize, \quad block\_offset = slot\_mapping[t] \% BlockSize
  $$

  $$
  k\_cache[block\_id, nk, block\_offset, d] = Q_{fp8}[t, N_q + nk, d]
  $$

  $$
  v\_cache[block\_id, nv, block\_offset, d] = v\_out[t, nv, d]
  $$

  $$
  k\_scale\_cache[block\_id, nk, block\_offset, 0] = Scale[t, N_q + nk]
  $$

  $$
  q\_out[t, nq, d] = Q_{fp8}[t, nq, d], \quad q\_scale[t, nq] = Scale[t, nq]
  $$

## 规格约束

### 输入规格 — Tensor 参数

| 参数名 | 必选/可选 | dtype | format | shape | 说明 |
| ------ | --------- | ----- | ------ | ----- | ---- |
| qkv | 必选 | bfloat16 | ND | [T, Nq+Nk+Nv, D] | Q/K/V 融合输入，沿 head 维拼接 |
| qGamma | 必选 | float32 | ND | [D] | Q 分支 RMSNorm gamma |
| kGamma | 必选 | float32 | ND | [D] | K 分支 RMSNorm gamma |
| cosSin | 必选 | float32 | ND | [MaxSeqLen, D] | RoPE cos/sin 数据，前 D/2 为 cos，后 D/2 为 sin |
| slotMapping | 必选 | int32 | ND | [T] | token 到 cache slot 的映射 |
| kCache | 必选 | float8_e4m3fn | ND | [BlockNum, Nk, BlockSize, D] | K cache，输入输出复用空间 |
| vCache | 必选 | float8_e4m3fn | ND | [BlockNum, Nv, BlockSize, D] | V cache，输入输出复用空间 |
| kScaleCache | 必选 | float32 | ND | [BlockNum, Nk, BlockSize, 1] | K scale cache，输入输出复用空间 |
| cuSeqLen | 必选 | int32 | ND | [Batch+1] | batch 前缀和，cuSeqLen[0]=0，cuSeqLen[Batch]=T |
| optionalRotation | 可选 | bfloat16 | ND | [D, D] | rotation 矩阵，接口 optional 但当前场景强制非空 |
| optionalVScale | 可选 | float32 | ND | [Nv] | V 分支 per-head 量化乘子，接口 optional 但当前场景强制非空 |

### 输入规格 — Scalar / 属性参数

| 参数名 | 类型 | 取值 | 说明 |
| ------ | ---- | ---- | ---- |
| headNums | int64 数组 | 长度固定为 3 | 必选，依次为 [Nq, Nk, Nv]，三个值均为正整数 |
| epsilon | float32 | 默认 1e-6 | RMSNorm 防止除零的小量 |

### 约束条件

| 约束项 | 条件 | 说明 |
| ------ | ---- | ---- |
| 平台限制 | arch35/Ascend950 | 首期仅支持 arch35，非 arch35 返回不支持 |
| headDim | D == 128 | 首期 fast path 固定 D=128 |
| D 偶数 | D % 2 == 0 | RoPE 按 D/2 拆分低半区和高半区 |
| Nv 约束 | Nv == Nk | 首期 fast path 只支持 Nv==Nk，Nv!=Nk 由 checker 拒绝 |
| head_nums 校验 | qkv.shape[1] == Nq+Nk+Nv | qkv 第二维必须等于 head_nums 之和 |
| cache head 维 | kCache.shape[1]==Nk, vCache.shape[1]==Nv, kScaleCache.shape[1]==Nk | cache head 维与 head_nums 一致 |
| rotation 非空 | optionalRotation != nullptr | 当前场景强制存在 |
| v_scale 非空 | optionalVScale != nullptr | 当前场景强制存在 |
| rotation shape | rotation.shape == [D, D] | rotation 为方阵 |
| v_scale shape | vScale.shape == [Nv] | per-head 量化乘子 |
| cos_sin 校验 | cosSin.shape[1] == D | cos_sin 第二维等于 headDim |
| gamma 校验 | qGamma.shape == [D], kGamma.shape == [D] | gamma 维度等于 headDim |
| slot_mapping 校验 | slotMapping.shape == [T] | 长度等于 token 数 |
| cu_seq_len 校验 | cuSeqLen 为一维，长度 >= 2 | cuSeqLen[0]=0, cuSeqLen[Batch]=T |
| in-place alias | kCacheOut==kCache, vCacheOut==vCache, kScaleCacheOut==kScaleCache | cache 输出与输入共享物理空间 |

### 输出规格

| 参数名 | dtype | format | shape | 说明 |
| ------ | ----- | ------ | ----- | ---- |
| qOut | float8_e4m3fn | ND | [T, Nq, D] | Q 分支 FP8 量化输出 |
| qScale | float32 | ND | [T, Nq] | Q 分支动态量化 scale |
| kCacheOut | float8_e4m3fn | ND | [BlockNum, Nk, BlockSize, D] | K cache 原地更新输出（与 kCache alias） |
| vCacheOut | float8_e4m3fn | ND | [BlockNum, Nv, BlockSize, D] | V cache 原地更新输出（与 vCache alias） |
| kScaleCacheOut | float32 | ND | [BlockNum, Nk, BlockSize, 1] | K scale cache 原地更新输出（与 kScaleCache alias） |
