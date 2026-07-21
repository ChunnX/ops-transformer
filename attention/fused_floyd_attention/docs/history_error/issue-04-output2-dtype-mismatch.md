# output[2] dtype 不匹配

## 现象

`single_bm` 对 output[2]（attention_out）报 dtype 不匹配，
`check_output_size`（base_compare.py:175）直接返回 `False`，**到不了阈值比较**。

| 输出 | dtype (golden/npu) | mean_rel_err | max_rel_err |
|------|-------------------|-------------|-------------|
| output[0] softmax_max | fp32/fp32 | 1.33e-8 | 2.42e-7 |
| output[1] softmax_sum | fp32/fp32 | 1.50e-7 | 3.70e-6 |
| output[2] attention_out | **fp32/bf16** | 2.51e-3 | 250 |

## 根因

ATK 对输入做了升精度：
- `dataset_config.py:84-85` 的 `_get_upper_type` 把 bf16 → fp32。
- golden 收到的 Q 因此是 fp32，`floyd_attn_forward` 内 `dtype = Q.dtype` 取到 fp32。
- `executor_...new_golden.py:74`：`attention_out = attention_out.to(Q.dtype)` → 输出 fp32。
- NPU 输出仍是 bf16 → 两边 dtype 不一致。

## 修复

**文件**：`executor_aclnnFusedFloydAttention_new_golden.py`

1. 第 16 行新增 import：
   ```python
   from atk.case_generator.utils.enums import TorchDtype
   ```

2. 第 75-76 行，从 case_config 取原始 dtype（而非被升精度的 Q.dtype）：
   ```python
   ori_dtype = TorchDtype.get(self.case_config.inputs[0].dtype)
   attention_out = attention_out.to(ori_dtype)
   ```

## 备注

- output[0]/output[1] 本身返回 fp32，与 NPU 一致，无需改。
- 修完 dtype 后，single_bm 的 max_rel_err=250 仍需处理 near-zero 导致的相对误差爆炸问题（调阈值）。
