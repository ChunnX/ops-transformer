# MaskedCausalConv1dBackward

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：对hidden层的token之间进行一维分组卷积操作的反向梯度计算。

- 计算公式：
  假定卷积输入input、卷积输出的梯度grad_output和卷积输入的梯度grad_input的shape是[S, B, H]，weight的shape是[W, H]，i和j分别表示S/B轴的索引，k为卷积窗口W内的索引，那么计算将被表示为：

  $$
  grad\_output\_masked[i,j] = mask[j,i] * grad\_output[i,j]
  $$

  $$
  grad\_input[i,j] = \sum_{k=0}^{W-1} grad\_output\_masked[i+k,j] * weight[W-1-k]
  $$

  $$
  grad\_weight[k] = \sum_{j=0}^{B-1}\sum_{i=0}^{S-1} grad\_output\_masked[i+W-1-k,j] * input[i,j]
  $$

  其中，无效位置的padding为0填充；当前W仅支持3。


## 参数说明

**说明：**<br> 

- grad_output、input、weight参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、W（Widndow Size）表示窗口大小。

-   **grad_output**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`。shape为[S, B, H]。
    
-   **input**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`，shape和数据类型与grad_output一致。
    
-   **weight**（`Tensor`）：必选参数，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`，shape为[W, H]，W目前只支持3，数据类型需与grad_output一致。
    
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **mask**（`Tensor`）：可选参数，表示卷积操作的输出掩码，数据类型支持`bool`，shape为[B, S]。。如果不指定可传入None，表示无掩码操作。

## 返回值说明

-   **grad\_input**（`Tensor`）：表示分组卷积输入input的梯度，对应公式中的grad_input。shape和数据类型与`grad_output`一致。

-   **grad\_weight**（`Tensor`）：表示分组卷积输入weight的梯度，对应公式中的grad_weight，shape为[W, H]，W目前只支持3，数据类型需与grad_output一致。

## 约束说明

-   该接口不支持图模式。
-   grad_output、input、weight、grad_input和grad_weight的数据类型必须一致。
-   输入输出的shape数据范围约束如下：
    - B（Batchsize）：取值范围为1\~32。
    - B * S（Batchsize * SeqLength）：取值范围为1\~512K。
    - H（hiddenSize）：取值范围192 * 2\~192 * 128，且是64的倍数。
    - W：W当前只支持3。

## 调用示例

| 调用方式  | 说明                                                         |
| --------- | ------------------------------------------------------------|
| aclnn接口 | 通过[aclnnMaskedCausalConv1dBackward](./docs/aclnnMaskedCausalConv1dBackward.md)调用MaskedCausalConv1d算子 |