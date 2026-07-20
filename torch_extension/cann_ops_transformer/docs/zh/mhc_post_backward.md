# cann_ops_transformer.mhc_post_backward

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

该接口实现MHC Post组件的反向计算：

$$grad\_x = H_l^{res} \times grad\_output$$

$$grad\_h\_res = x_l \times grad\_output^T$$

$$grad\_h\_out = (grad\_output \times H_l^{post}.unsqueeze(-1)).sum(dim=-2)$$

$$grad\_h\_post = (grad\_output \times h_l^{out}.unsqueeze(-2)).sum(dim=-1)$$

## 函数原型

```python
cann_ops_transformer.mhc_post_backward(grad_output, x, h_res, h_out, h_post) \
    -> (grad_x, grad_h_res, grad_h_out, grad_h_post)
```

## 参数与返回值

| 名称 | 输入/输出 | 数据类型 | 数据格式 | BSND shape | TND shape | 非连续Tensor |
|---|---|---|---|---|---|---|
| grad_output | 输入 | bfloat16、float16 | ND | (B, S, N, D) | (T, N, D) | 支持 |
| x | 输入 | 与grad_output一致 | ND | (B, S, N, D) | (T, N, D) | 支持 |
| h_res | 输入 | float32 | ND | (B, S, N, N) | (T, N, N) | 支持 |
| h_out | 输入 | 与grad_output一致 | ND | (B, S, D) | (T, D) | 支持 |
| h_post | 输入 | float32 | ND | (B, S, N) | (T, N) | 支持 |
| grad_x | 输出 | 与x一致 | ND | (B, S, N, D) | (T, N, D) | 支持 |
| grad_h_res | 输出 | 与h_res一致 | ND | (B, S, N, N) | (T, N, N) | 支持 |
| grad_h_out | 输出 | 与h_out一致 | ND | (B, S, D) | (T, D) | 支持 |
| grad_h_post | 输出 | 与h_post一致 | ND | (B, S, N) | (T, N) | 支持 |

所有输入必须位于同一NPU设备且不能为空。`x`仅支持3维TND或4维BSND排布；各输入的B、S、T、N、D维度必须按上表保持一致。

## 约束说明

- Atlas A2训练/推理系列产品、Atlas A3训练/推理系列产品：`N`仅支持4；`D`需按128对齐，取值范围为[1, 100000]。
- Ascend 950PR/Ascend 950DT：`N`支持4、6、8。

## 单算子调用

```python
import torch
import torch_npu
from cann_ops_transformer.ops import mhc_post_backward

B, S, N, D = 2, 8, 4, 128
grad_output = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
x = torch.randn_like(grad_output)
h_res = torch.randn(B, S, N, N, device="npu", dtype=torch.float32)
h_out = torch.randn(B, S, D, device="npu", dtype=torch.float16)
h_post = torch.randn(B, S, N, device="npu", dtype=torch.float32)

grads = mhc_post_backward(grad_output, x, h_res, h_out, h_post)
```

## 图模式调用

```python
import torch
import torchair
from cann_ops_transformer.ops import mhc_post_backward

compiled = torch.compile(mhc_post_backward, backend=torchair.get_npu_backend(), dynamic=False)
graph_grads = compiled(grad_output, x, h_res, h_out, h_post)
```
