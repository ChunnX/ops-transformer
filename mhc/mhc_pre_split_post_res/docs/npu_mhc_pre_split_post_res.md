# torch\.ops.custom.npu\_mhc\_pre\_split\_post\_res<a name="ZH-CN_TOPIC_0000001979260730"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √    |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：mhc_pre_split_post_res算子是mhc_pre算子的一部分，负责计算mHC（Manifold-Constrained Hyper Connection）架构中x2输入对应的post分支和comb/Res分支，输出h_post和h_res两个变换矩阵。具体流程为：将输入x展平并转FP32后计算RMSNorm的InvRMS因子，执行线性投影得到x_hat，然后切分为post分支和comb分支分别计算h_post和h_res。

- 计算公式：
  $$
  X_{flat} = \mathrm{reshape}(x, [B,S,N \cdot D])
  $$
  $$
  \mathrm{inv\_rms} = \frac{1}{\sqrt{\frac{1}{N \cdot D}\sum_{k} X_{flat,k}^2 + \mathrm{norm\_eps}}}
  $$
  $$
  X_{hat} = (X_{flat} \cdot \phi^T) \odot \mathrm{inv\_rms}
  $$
  $$
  [X_{post}, X_{comb}] = \mathrm{split}(X_{hat}, [N, N^2])
  $$
  $$
  h_{post} = 2 \cdot \sigma(\alpha_0 \cdot X_{post} + b_{post})
  $$
  $$
  h_{res} = \alpha_1 \cdot \mathrm{reshape}(X_{comb}, [B,S,N,N]) + \mathrm{reshape}(b_{comb}, [N,N])
  $$

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
torch.ops.custom.npu_mhc_pre_split_post_res(x, phi, alpha, bias, *, norm_eps=1e-6) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

> **说明：**<br>
>
> - 各参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、n（Stream Num）表示mHC超连接路数、D（Head Dim）表示hidden layer最小的单元尺寸；$T=B*S$ 表示所有Batch输入样本序列长度的累加和。

- **x**（`Tensor`）：必选参数，待计算的数据，表示网络中mHC层的输入数据x。支持非连续的Tensor，数据格式支持ND。维度shape支持[B,S,n,D]，或者为[T,n,D]。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。

- **phi**（`Tensor`）：必选参数，mHC的变换矩阵phi。支持非连续的Tensor，数据格式支持ND。维度shape支持[n^2+n, n\*D]。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`。

- **alpha**（`Tensor`）：必选参数，mHC的缩放参数。支持非连续的Tensor，数据格式支持ND。维度shape支持[2]。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`。

- **bias**（`Tensor`）：必选参数，mHC的bias参数。支持非连续的Tensor，数据格式支持ND。维度shape支持[n^2+n]。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`。

- **norm_eps**（`float`）：可选参数，RMSNorm防除0参数，默认值为1e-6。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

- **h_post**（`Tensor`）：mHC的h_post变换矩阵。数据类型支持`float32`。数据格式支持ND。维度shape支持[B,S,n]，或者为[T,n]。

- **h_res**（`Tensor`）：mHC的comb分支输出矩阵。数据类型支持`float32`。数据格式支持ND。维度shape支持[B,S,n,n]，或者为[T,n,n]。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

### 确定性计算

- MhcPreSplitPostRes 算子默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束
- 输入约束：
   - 输入Tensor `x`、`phi`、`alpha`、`bias` 不能为空，且必须为Device侧Tensor。
   - 所有输入/输出Tensor的数据格式仅支持ND。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~512K | B*S 或T支持1~512K范围以内。|
| n | 4 | n为固定值4。|
| D | 2560/5120 | D支持2560或5120。|

### 典型值

| 规格项 | 典型值 |
|:--- |:--- |
| T或B*S | 4 |
| n | 4 |
| D | 2560 |

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

```python
import torch
import torch_npu
import omni_custom_ops

# 参数设置
B, S, n, D = 1, 4, 4, 2560
phi_rows = n * n + n  # 20

# 生成随机数据，并发送到npu
x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
phi = torch.randn(phi_rows, n * D, dtype=torch.float32).npu()
alpha = torch.tensor([1.0, 1.0], dtype=torch.float32).npu()
bias = torch.zeros(phi_rows, dtype=torch.float32).npu()

# 调用 mhc_pre_split_post_res 算子
h_post, h_res = torch.ops.custom.npu_mhc_pre_split_post_res(
    x, phi, alpha, bias, norm_eps=1e-6
)

# h_post: [B, S, n] = [1, 4, 4], float32
# h_res:  [B, S, n, n] = [1, 4, 4, 4], float32
print("h_post shape:", h_post.shape)
print("h_res shape:", h_res.shape)
```
