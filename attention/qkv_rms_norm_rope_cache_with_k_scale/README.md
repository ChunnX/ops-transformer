# QkvRmsNormRopeCacheWithKScale

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |

## 功能说明

QkvRmsNormRopeCacheWithKScale 是一个面向 Transformer decode/prefill 流程的融合算子，将以下操作融合在一个算子中：

- **Q/K/V 拆分**：从融合输入 `[T, Nq+Nk+Nv, D]` 中分离 Q、K、V
- **RMSNorm**：对 Q 和 K 分别应用 RMSNorm（使用独立的 gamma 参数）
- **RoPE**：应用旋转位置编码（使用 cu_seq_len 前缀和定位 batch 内位置）
- **Rotation Matmul**：与 rotation 矩阵进行矩阵乘法
- **FP8 动态量化**：对 Q/K 进行 FP8 E4M3FN 动态量化，对 V 进行简单量化
- **KV Cache 写回**：将量化后的 K/V 根据 slot_mapping 写入 cache

### 计算公式

**RMSNorm**:
```
S[t,h] = sum(X[t,h,:]^2)
R[t,h] = sqrt(S[t,h] / D + epsilon)
Y[t,h,:] = X[t,h,:] / R[t,h] * gamma[h,:]
```

**RoPE** (half-split):
```
out_low = y_low * cos - y_high * sin
out_high = y_high * cos + y_low * sin
```

**Rotation Matmul**:
```
qk_after_cube[t,h,:] = rope_out[t,h,:] @ rotation
```

**V 量化**:
```
v_out = cast_to_fp8(V * v_scale)
```

**Q/K 动态量化**:
```
max_abs = max(abs(qk_after_cube))
scale = max_abs / 448
qk_quant = cast_to_fp8(qk_after_cube / scale)
```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| qkv | 输入 | Q/K/V 融合输入 | BFLOAT16 | ND |
| qGamma | 输入 | Q 分支 RMSNorm gamma | FLOAT | ND |
| kGamma | 输入 | K 分支 RMSNorm gamma | FLOAT | ND |
| cosSin | 输入 | RoPE cos/sin 数据 | FLOAT | ND |
| slotMapping | 输入 | token 到 cache slot 的映射 | INT32 | ND |
| kCache | 输入/输出 | K cache（in-place 更新） | FLOAT8_E4M3FN | ND |
| vCache | 输入/输出 | V cache（in-place 更新） | FLOAT8_E4M3FN | ND |
| kScaleCache | 输入/输出 | K scale cache（in-place 更新） | FLOAT | ND |
| cuSeqLen | 输入 | batch 前缀和 | INT32 | ND |
| rotation | 输入 | rotation 矩阵（必须非空） | BFLOAT16 | ND |
| vScale | 输入 | V 分支量化 scale（必须非空） | FLOAT | ND |
| headNums | 属性 | 头数配置 [Nq, Nk, Nv] | INT64[] | - |
| epsilon | 属性 | RMSNorm epsilon | FLOAT | - |
| qOut | 输出 | Q 分支量化输出 | FLOAT8_E4M3FN | ND |
| qScale | 输出 | Q 分支量化 scale | FLOAT | ND |
| kCacheOut | 输出 | K cache 更新输出 | FLOAT8_E4M3FN | ND |
| vCacheOut | 输出 | V cache 更新输出 | FLOAT8_E4M3FN | ND |
| kScaleCacheOut | 输出 | K scale cache 更新输出 | FLOAT | ND |

## 约束说明

### 平台约束
- 仅支持 Ascend 950 (arch35) 平台

### Shape 约束
- `qkv.shape[1] == Nq + Nk + Nv`
- `D == 128`（首期 fast path）
- `D % 2 == 0`（RoPE 要求）
- `Nv == Nk`（首期 fast path）
- `kCache.shape[1] == Nk`
- `vCache.shape[1] == Nv`
- `kScaleCache.shape[1] == Nk`
- `cosSin.shape[1] == D`
- `rotation.shape == [D, D]`
- `vScale.shape[0] == Nv`

### 特殊约束
- `rotation` 和 `vScale` 虽然标记为可选，但当前场景必须非空
- `kCache`、`vCache`、`kScaleCache` 为输入输出复用（in-place 更新）
- `headNums` 必须包含恰好 3 个正整数

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---------|---------|------|
| aclnn 接口 | [test_aclnn_qkv_rms_norm_rope_cache_with_kscale](./examples/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp) | 通过 [aclnnQkvRmsNormRopeCacheWithKScale](./docs/aclnnQkvRmsNormRopeCacheWithKScale.md) 接口方式调用算子 |

## 注意事项

1. 当前为首期实现，仅支持 D=128 和 Nv==Nk 的 fast path
2. rotation 和 vScale 虽然接口标记为可选，但必须传入非空指针
3. kCache、vCache、kScaleCache 为 in-place 更新，输入输出共享物理内存
4. 不支持 D!=128 或 Nv!=Nk 的配置，会返回 GRAPH_FAILED
