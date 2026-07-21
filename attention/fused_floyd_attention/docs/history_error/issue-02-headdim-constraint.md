# headDim 不符合算子约束

## 现象

```
Headdim only supports 32/64/128.
[FUNC:CheckBaseInput][FILE:fused_floyd_attention_tiling_common.cpp][LINE:145]
aclnnFusedFloydAttentionGetWorkspaceSize failed! error code: 561002
```

修完注册名后，所有用例在 pyaclnn（NPU 算子执行）阶段因 headDim 非法失败。

## 根因

`generator_aclnnFusedFloydAttention.py:33`：

```python
self.D = random.randint(1, 128)   # ❌ 未约束为 32/64/128
```

生成的 220 个用例 headDim 全不合法（7、1、33、42…）。yaml 里的 `dim_values: [32]`
被 generator 的 `input_case.shape = [B, N, S1, S2, self.D]` 直接覆盖，未生效。

## 修复

**文件**：`generator_aclnnFusedFloydAttention.py:33`

```python
# 改前
self.D = random.randint(1, 128)
# 改后
self.D = random.choice([32, 64, 128])
```

改完后需重新生成用例（重跑 `atk case`）。
