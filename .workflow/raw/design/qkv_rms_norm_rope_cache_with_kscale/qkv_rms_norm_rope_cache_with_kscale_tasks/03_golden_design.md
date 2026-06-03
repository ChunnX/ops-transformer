# Golden 设计

## 1. 目标

Golden 用于校验 host 侧 shape 逻辑、kernel 端到端数值、cache 写回地址和 FP8 quant 行为。Golden 必须直接体现完整计算链路，不能把 `rotation` 简化成特殊 Hadamard 形式。

## 2. Golden 输入

Python/Numpy golden 输入与算子一致：

```text
qkv: [T, Nq + Nk + Nv, D], bf16 simulated by fp32 cast/round policy
q_gamma: [D], fp32
k_gamma: [D], fp32
cos_sin: [MaxSeqLen, D], fp32
slot_mapping: [T], int32
k_cache: [BlockNum, Nk, BlockSize, D], fp8_e4m3fn storage
v_cache: [BlockNum, Nv, BlockSize, D], fp8_e4m3fn storage
k_scale_cache: [BlockNum, Nk, BlockSize, 1], fp32
cu_seq_len: [Batch + 1], int32 prefix sum
rotation: [D, D], bf16 simulated
v_scale: [Nv], fp32
head_nums: [Nq, Nk, Nv], ListInt
epsilon: fp32
```

Golden 中 `Nq/Nk/Nv` 统一来自 `head_nums`，并校验 `qkv.shape[1] == Nq + Nk + Nv`。

Golden 计算函数可以保留对独立 `Nv` 的泛化实现，便于后续扩展和单元验证；但首期 arch35 fast path 只支持 `Nv == Nk`。端到端 ST 的正常路径应构造 `Nv = Nk`，`Nv != Nk` 只进入 checker/tiling 拒绝类用例，不进入 kernel 数值对比。

## 3. Golden 计算顺序

### 3.1 Split Q/K/V

```python
# Nq/Nk/Nv are local head counts from head_nums.
# qkv axis 1 is fused as [Q heads, K heads, V heads].
q = qkv[:, 0:Nq, :]
k = qkv[:, Nq:Nq + Nk, :]
v = qkv[:, Nq + Nk:Nq + Nk + Nv, :]
qk = concat(q, k, axis=1)
```

### 3.2 RMSNorm

```python
sum_sq = np.sum(qk.astype(np.float32) * qk.astype(np.float32), axis=-1, keepdims=True)
rms = np.sqrt(sum_sq / D + epsilon)

q_norm = q / rms_q * q_gamma
k_norm = k / rms_k * k_gamma
```

Q/K 必须分别使用 `q_gamma`、`k_gamma`。

### 3.3 RoPE

对每个 token `t`：

```python
b = searchsorted(cu_seq_len, t, side="right") - 1
s = t - cu_seq_len[b]
cos = cos_sin[s, 0:D//2]
sin = cos_sin[s, D//2:D]
```

对每个 Q/K head：

```python
low = y[..., 0:D//2]
high = y[..., D//2:D]
out_low = low * cos - high * sin
out_high = high * cos + low * sin
rope_out = concat(out_low, out_high, axis=-1)
```

### 3.4 Rotation Matmul

Golden 必须使用外部传入的 `rotation`：

```python
qk_after_cube[t, h, :] = rope_out[t, h, :] @ rotation
```

`rotation` 可随机生成；不要求是标准 Hadamard。若测试希望覆盖 Hadamard，可额外构造 Hadamard value，但计算仍按 dense matmul。

### 3.5 V Quant

```python
v_scaled[t,nv,d] = float32(v[t,nv,d]) * v_scale[nv]
v_out = cast_to_e4m3fn_sat_rint(v_scaled)
```

`v_scale == 0` 时按公式计算。

### 3.6 Q/K Dynamic Quant

```python
max_abs = np.max(np.abs(qk_after_cube), axis=-1)
scale = max_abs / 448.0
qk_div = qk_after_cube / scale[..., None]
qk_quant = cast_to_e4m3fn_sat_rint(qk_div)
```

该公式对齐 arch35 `dynamic_quant` FP8 E4M3FN 路径，不额外定义 `scale == 0` 的全 0 分支。需要在测试中单独覆盖 `max_abs == 0`，观察并锁定硬件/工具链行为。

### 3.7 Cache 写回

```python
slot = slot_mapping[t]
block_id = slot // BlockSize
block_offset = slot % BlockSize

v_cache_ref[block_id, nv, block_offset, d] = v_out[t,nv,d]
k_cache_ref[block_id, nk, block_offset, d] = qk_quant[t,Nq+nk,d]
k_scale_cache_ref[block_id, nk, block_offset, 0] = scale[t,Nq+nk]
```

Q 输出：

```python
q_out_ref[t,nq,d] = qk_quant[t,nq,d]
q_scale_ref[t,nq] = scale[t,nq]
```

## 4. FP8 E4M3FN 模拟

Golden 需要一个统一的：

```python
cast_to_e4m3fn_sat_rint(x)
```

要求：

1. RINT round。
2. SAT 饱和。
3. 最大有限值按 448。
4. dtype 为 `float8_e4m3fn`。

如果 Python 环境没有可靠 bit-level FP8 工具，建议分两级 golden：

1. 数学 golden：RMSNorm、RoPE、matmul、scale、slot 地址必须完全验证。
2. FP8 cast golden：借助工程已有 FP8 cast 工具或参考 `dynamic_quant` 输出做 bit 对齐。

## 5. 推荐 Case

### 5.1 典型 Shape 空间

Golden 需要覆盖典型 shape 空间。这里的 `B` 表示 batch 数，`S_b` 表示第 `b` 个 batch 的 token 数，`S` 表示单 batch 场景的 token 数或多 batch 场景下的 `max_b(S_b)`，`D` 固定为典型 head dim：

```text
B  in {1, 2}
Nq in {16, 32, 64}
Nk in {2, 4, 8}
S  in [1, 32k]
D  = 128
```

对应张量关系：

```text
Batch = B
MaxSeqLen = max_b(S_b)
T = sum_b(S_b)
cu_seq_len = [0, S_0, S_0 + S_1, ..., T]
qkv.shape = [T, Nq + Nk + Nv, 128]
cos_sin.shape = [MaxSeqLen, 128]
rotation.shape = [128, 128]
k_cache.shape = [BlockNum, Nk, BlockSize, 128]
v_cache.shape = [BlockNum, Nv, BlockSize, 128]
k_scale_cache.shape = [BlockNum, Nk, BlockSize, 1]
v_scale.shape = [Nv]
q_out.shape = [T, Nq, 128]
q_scale.shape = [T, Nq]
```

`Nv` 在 Golden 代码中仍必须始终从 `head_nums[2]` 获取，不能由 `Nk` 隐式推导。首期可执行 kernel/ST 只构造 `Nv = Nk` 的正常用例；`Nv != Nk` 用于验证 checker/tiling 拒绝行为，不做端到端数值对比。

建议将典型 shape 拆成分层用例：

| 类型 | Shape 选择 | 目的 |
|------|------------|------|
| smoke | `B=1, S=1, Nq=16, Nk=2, D=128` | 最小路径，便于 debug |
| normal | `B=1, S in {128, 1024}, Nq in {16,32}, Nk in {2,4}, D=128` | 常规数值和 cache 写回 |
| multi-batch | `B=2, S_0 != S_1, Nq in {16,32}, Nk in {2,4}, D=128` | `cu_seq_len` 前缀和与 RoPE 位置 |
| large-seq | `B in {1,2}, max(S_b) in {8k, 16k, 32k}, Nq in {32,64}, Nk in {4,8}, D=128` | 长序列位置索引和大 shape 压力 |
| head-combo | 遍历 `Nq in {16,32,64}` 与 `Nk in {2,4,8}` 的代表性组合 | `head_nums` 解析和 Q/K head 切分 |

大 shape case 不要求全部进入高精度 bit-level FP8 对比；可先用于 shape、地址、scale 统计和抽样数值校验。完整 bit-level 对比优先覆盖 smoke/normal/multi-batch。

### 5.2 功能 Case

| Case | 目的 |
|------|------|
| 单 batch，`T=1`，`D=128` | 最小正常路径 |
| 单 batch，多 token，多 Q/K/V head | 基础数值 |
| 多 batch 前缀和，如 `[0,2,9]` | RoPE batch 内位置 |
| `slot_mapping` 连续 | cache 基本写回 |
| `slot_mapping` 非连续和跨 block | cache 地址 |
| `v_scale` 包含 0 | V quant 边界 |
| `max_abs == 0` row | dynamic quant 边界观察 |
| random `rotation` | dense matmul，不依赖 Hadamard 特性 |
| Hadamard-like `rotation` | 覆盖用户场景，但仍按 dense 计算 |
| `Nv != Nk` | checker/tiling 拒绝，不进入 kernel 数值路径 |

## 6. 对比策略

1. `q_scale`、`k_scale_cache` 使用 FP32 容差。
2. FP8 输出优先 bit 对比；若工具暂不支持 bit decode，先做 dequant 后容差对比并记录风险。
3. cache 输入输出 alias 时，golden 应从初始 cache copy 更新局部 slot，未写位置保持原值。
4. `cu_seq_len` 非法输入不进入 golden 数值路径，由 UT/checker 覆盖。

## 7. 验收标准

1. Golden 可以独立生成 `q_out`、`q_scale`、`k_cache`、`v_cache`、`k_scale_cache`。
2. Golden 明确使用 `cu_seq_len` 前缀和计算 RoPE 位置。
3. Golden 明确使用外部 `rotation`。
4. Golden 与 arch35 FP8 dynamic quant 行为保持一致，或者显式标注无法 bit-level 模拟的风险。
