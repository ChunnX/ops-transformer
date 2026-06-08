# QkvRmsNormRopeCacheWithKScale

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |

## 功能说明

QkvRmsNormRopeCacheWithKScale 是面向 Transformer decode/prefill 流程的融合算子，包含 Q/K/V 拆分、Q/K RMSNorm、RoPE、Q/K 分支独立 rotation matmul、FP8 动态量化和 KV cache 原地写回。

```text
q_after_cube[t,nq,:] = rope_q[t,nq,:] @ q_rotation
k_after_cube[t,nk,:] = rope_k[t,nk,:] @ k_rotation
```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| qkv | 输入 | Q/K/V 融合输入 | BFLOAT16 | ND |
| qGamma | 输入 | Q 分支 RMSNorm gamma | FLOAT | ND |
| kGamma | 输入 | K 分支 RMSNorm gamma | FLOAT | ND |
| cosSin | 输入 | RoPE cos/sin 数据 | FLOAT | ND |
| slotMapping | 输入 | token 到 cache slot 的映射 | INT32 | ND |
| kCache | 输入/输出 | K cache，in-place 更新 | FLOAT8_E4M3FN | ND |
| vCache | 输入/输出 | V cache，in-place 更新 | FLOAT8_E4M3FN | ND |
| kScaleCache | 输入/输出 | K scale cache，in-place 更新 | FLOAT | ND |
| cuSeqLen | 输入 | batch 前缀和 | INT32 | ND |
| qRotation | 输入 | Q 分支 rotation 矩阵，必须非空 | BFLOAT16 | ND |
| kRotation | 输入 | K 分支 rotation 矩阵，必须非空 | BFLOAT16 | ND |
| vScale | 输入 | V 分支量化 scale，必须非空 | FLOAT | ND |
| headNums | 属性 | 头数配置 [Nq, Nk, Nv] | INT64[] | - |
| epsilon | 属性 | RMSNorm epsilon | FLOAT | - |
| qOut | 输出 | Q 分支量化输出 | FLOAT8_E4M3FN | ND |
| qScale | 输出 | Q 分支量化 scale | FLOAT | ND |

## 约束说明

- 仅支持 Ascend 950 (arch35) 平台。
- 首期 fast path 仅支持 `D == 128`、`Nv == Nk`。
- `qRotation.shape == [D, D]`，`kRotation.shape == [D, D]`，均为非转置 ND 矩阵。
- `vScale.shape[0] == Nv`。
- `kCache`、`vCache`、`kScaleCache` 为 in-place 参数，ACLNN 接口不再额外传入 cache 输出 tensor。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---------|---------|------|
| aclnn 接口 | [test_aclnn_qkv_rms_norm_rope_cache_with_kscale](./examples/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp) | 通过 [aclnnQkvRmsNormRopeCacheWithKScale](./docs/aclnnQkvRmsNormRopeCacheWithKScale.md) 接口方式调用算子 |
