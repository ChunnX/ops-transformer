# torch\.ops.costom.npu\_ai\_infra\_mhc\_post\_grad<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：mhc_post基于一系列计算对mHC架构中上一层输出$h_{t}^{out}$进行Post Mapping，对上一层的输入$x_j$进行ResMapping，然后对二者进行残差连接，得到下一层的输入$x_{l+1}$。该算子实现前述过程的反向功能。
  
- 计算公式：
  $$
  grad\_x = H_{l}^{res} \times grad\_output\\
  grad\_h\_res = x_{l} \times {grad\_output}^{T}
  $$
  $$
  grad\_h\_out=({grad\_output} * (H_{l}^{post}.unsqueeze(-1))).sum(dim=-2)\\
  grad\_h\_post=({grad\_output} * (h_{l}^{out}.unsqueeze(-2))).sum(dim=-1)
  $$

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
torch.ops.custom.npu_ai_infra_mhc_post_grad(grad_output, x, h_res, h_out, h_post) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

> **说明：**<br>
> 
> - 各参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、n（Head Num）表示多头数、D（Head Dim）表示隐藏层最小的单元尺寸; $T=B*S$ 表示所有Batch输入样本序列长度的累加和。

- **grad_output**（`Tensor`）：必选参数，待计算的数据，表示网络中mHC层的输入数据。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n,D]，或者为[T,n,D]。
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。

- **x**（`Tensor`）：必选参数，待计算的数据，表示网络中mHC层的输入数据。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n,D]，或者为[T,n,D]。
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
- **h_res**（`Tensor`）：必选参数，mHC的h_res变换矩阵，是做完sinkhorn变换后的双随机矩阵。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n,n]，或者为[T,n,n]。
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`。

- **h_out**（`Tensor`）：必选参数，Atten/MLP层的输出。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,D]，或者为[T,D]。
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。

- **h_post**（`Tensor`）：必选参数，mHC的h_post变换矩阵。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n]，或者为[T,n]。
  
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`。


## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

- **grad_x**（`Tensor`）：网络中mHC层的输入数据x的梯度。数据类型支持`float16`、`bfloat16`。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n,D]，或者为[T,n,D]。

- **grad_h_res**（`Tensor`）：必选参数，mHC的h_res变换矩阵，是做完sinkhorn变换后的双随机矩阵。数据类型支持`float32`。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n,n]，或者为[T,n,n]。

- **grad_h_out**（`Tensor`）：必选参数，Atten/MLP层的输出。数据类型支持`float16`、`bfloat16`。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,D]，或者为[T,D]。


- **grad_h_post**（`Tensor`）：必选参数，mHC的h_post变换矩阵。数据类型支持`float32`。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n]，或者为[T,n]。
  

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

### 确定性计算

- AiInfraMhcPostGrad 算子默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束
- 输入约束：
   - 输入Tensor `grad_output`、`x`、`h_res`、`h_out`、`h_post` 不能为空，且必须为Device侧Tensor。
   - h_res 分布满足双随机矩阵

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~512k | B*S 或T支持1~512k范围以内。|
| n | 4、6、8 | n值目前支持4, 6, 8。|
| D | 384~24k | D支持384~24k范围以内。|

### 典型值

| 规格项 | 典型值 |
|:--- |:--- |
| T或 B*S | 1024/2048/4096 |
| n | 4（推荐） |
| D | 2560/5120（推荐） |

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>


```python
import torch
import torch_npu
import omni_training_custom_ops

# 参数设置
B, S, n, D = 1, 1024, 4, 2560

# 生成随机数据, 并发送到npu
grad_output = torch.randn(B, S, n, D, dtype=torch.float16).npu()
x = torch.randn(B, S, n, D, dtype=torch.float16).npu()
h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
h_out = torch.randn(B, S, D, dtype=torch.float16).npu()
h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

# 调用 post_grad 算子
grad_x, grad_h_res, grad_h_out, grad_h_post = torch.ops.custom.npu_ai_infra_mhc_post_grad(
    grad_output, x, h_res, h_out, h_post
)
```
