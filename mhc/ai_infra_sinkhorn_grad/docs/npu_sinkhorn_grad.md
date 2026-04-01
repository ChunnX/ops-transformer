# SinkhornGrad

## 产品支持情况
|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200/300/500 推理产品</term>|      ×     |

## 功能说明

- 接口功能：对mHC架构中双随机矩阵Hres的梯度进行sinkhorn变换的反向计算得到输入H`res的梯度。

## 计算公式：

设当前梯度为 $\mathbf{grad}_{\text{curr}} = \mathbf{grad}_{\text{output}}$，共进行 $\mathbf{num\_iters}$ 次迭代。

#### 前 $\mathbf{num\_iters} - 1$ 次迭代  
对迭代编号 $i = \mathbf{num\_iters} - 1, \mathbf{num\_iters} - 2, \dots, 1$，依次执行：

$$
\begin{aligned}
    \mathbf{dot\_prod}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{norm\_out}_{2i+1} \right) \\
    \\
    \mathbf{grad}_{\text{curr}} &\gets \frac{{ \mathbf{grad}_{\text{curr}}} - \mathbf{dot\_prod}_{2i+1}} { \mathbf{sum\_out}_{2i+1} }, \\
    \\
    \mathbf{dot\_prod}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{num\_out}_{2i} \right), \\
    \\
    \mathbf{grad}_{\text{curr}}  &\gets  \frac{{ \mathbf{grad}_{\text{curr}}} - \mathbf{dot\_prod}_{2i}} { \mathbf{sum\_out}_{2i} }, \\
\end{aligned}
$$

#### 最后一次迭代  

$$
\begin{aligned}
    \mathbf{dot\_prod}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{norm\_out}_{1} \right), \\
    \\
    \mathbf{grad}_{\text{curr}} &\gets \frac{{ \mathbf{grad}_{\text{curr}}} - \mathbf{dot\_prod}_{1}} { \mathbf{sum\_out}_{1} }, \\
    \\
    \mathbf{dot\_prod}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{num\_out}_{0} \right), \\
    \\
    \mathbf{grad}_{\text{input}}  &\gets  \left( \mathbf{grad}_{\text{curr}} -  \mathbf{dot\_prod}_{0} \right) \cdot { \mathbf{norm\_out}_{0} }, \\
\end{aligned}
$$


## 函数原型

```
torch.ops.custom.npu_sinkhorn_grad(Tensor grad_output, Tensor norm_out, Tensor sum_out) -> (Tensor)"
```

- **参数说明:**
  
    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">  
            <col style="width: 400px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 190px">
            <col style="width: 145px">
            </colgroup>
        <thead>
        <tr>
            <th>参数名</th>
            <th>输入/输出</th>
            <th>描述</th>
            <th>使用说明</th>
            <th>数据类型</th>
            <th>数据格式</th>
            <th>维度(shape)</th>
            <th>非连续Tensor</th>
        </tr></thead>
        <tbody>
        <tr>
            <td>grad_output</td>
            <td>输入</td>
            <td>Sinkhorn变换输出的H_res的梯度</td>
            <td>数据类型与key/queryIndex/keyIndex保持一致。</td>
            <td>FLOAT32 </td>
            <td>ND</td>
            <td>(B,S,n,n)、(T,n,n)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>norm_out</td>
            <td>输入</td>
            <td>Sinkhorn变换正向计算保存的中间norm结果</td>
            <td>数据类型与query/queryIndex/keyIndex保持一致。</td>
            <td>FLOAT32 </td>
            <td>ND</td>
            <td>[2*num_iters,n,n,B,S]、[2*num_iters,n,n,T]</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sum_out</td>
            <td>输入</td>
            <td>Sinkhorn变换正向计算保存的中间sum结果</td>
            <td>数据类型与query/key/keyIndex保持一致。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>[2*num_iters,n,B,S]、[2*num_iters,n,T]</td>
            <td>√</td>
        </tr>
        </tbody>
    </table>
- **返回值参数:**
  
    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">  
            <col style="width: 400px">  
            <col style="width: 212px">  
            <col style="width: 100px">
            <col style="width: 190px">
            <col style="width: 145px">
            </colgroup>
        <thead>
        <tr>
            <th>参数名</th>
            <th>输入/输出</th>
            <th>描述</th>
            <th>使用说明</th>
            <th>数据类型</th>
            <th>数据格式</th>
            <th>维度(shape)</th>
            <th>非连续Tensor</th>
        </tr></thead>
        <tbody>
        <tr>
            <td>out</td>
            <td>输出</td>
            <td>Sinkhorn变换的输入的H_res的梯度</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,n,n)、(T,n,n)</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

## 约束说明

- 确定性计算：
  - aclnnSinkhornGrad默认确定性实现。
- 公共约束
    - 入参为空的场景处理：
        - gradOutput等输入参数不能为空Tensor，否则返回失败。
- 规格约束
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>num_iters</td>
            <td>1~100</td>
            <td>num_iters支持1~100范围以内</td>
        </tr>
        <tr>
            <td>n</td>
            <td>4， 6 ，8</td>
            <td>n值目前支持4， 6， 8</td>
        </tr>
        </tbody>
    </table>
- 典型值
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>典型值</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>T或 B*S</td>
            <td>1024/2048/4096</td>
        </tr>
        <tr>
            <td>n</td>
            <td>n = 4</td>
        </tr>
        <tr>
            <td>num_iters</td>
            <td>num_iters = 20</td>
        </tr>
        </tbody>
    </table>
-   该接口支持仅支持PTA直调。


## 调用示例

-   单算子模式调用

```python
import torch
import torch_npu
import omni_training_custom_ops
import numpy as np

torch.npu.set_device(2)
torch.manual_seed(1234)
torch.set_printoptions(threshold=float('inf'))  # 阈值设为无穷大，永不截断
torch.set_printoptions(sci_mode=False, precision=5) 

def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

def sinkhorn_forward_simple(x, num_iters, eps):
    norm_list = []
    sum_list = []

    # Step 0: Softmax
    prob = softmax(x, dim=-1)
    curr = prob + eps
    norm_list.append(prob.clone())
    sum_list.append(None)

    # Step 1: Initial Col Norm
    col_sum = curr.sum(dim=-2, keepdim=True) + eps
    curr = curr / col_sum
    norm_list.append(curr.clone())
    sum_list.append(col_sum.clone())

    # Step 2: Loop
    for i in range(num_iters - 1):
        # Row Norm
        row_sum = curr.sum(dim=-1, keepdim=True) + eps
        curr = curr / row_sum
        norm_list.append(curr.clone())
        sum_list.append(row_sum.clone())

        # Col Norm
        col_sum = curr.sum(dim=-2, keepdim=True) + eps
        curr = curr / col_sum
        norm_list.append(curr.clone())
        sum_list.append(col_sum.clone())

    return curr, norm_list, sum_list


def convert_to_npu_format(norm_list, sum_list, num_iters):
    T, n, _ = norm_list[0].shape

    # 构建norm_out: [2*num_iters, n, n, T]
    norm_out = torch.zeros(2 * num_iters, n, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, norm_tensor in enumerate(norm_list):
        # [T, n, n] -> [n, n, T]
        norm_out[idx] = norm_tensor.permute(1, 2, 0)

    # 构建sum_out: [2*num_iters, n, T]
    sum_out = torch.zeros(2 * num_iters, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, sum_tensor in enumerate(sum_list):
        if sum_tensor is None:
            continue
        # [T, 1, n] or [T, n, 1] -> [n, T]
        if sum_tensor.shape[-2] == 1:  # col_sum: [T, 1, n]
            sum_out[idx] = sum_tensor.squeeze(-2).T
        else:  # row_sum: [T, n, 1]
            sum_out[idx] = sum_tensor.squeeze(-1).T
    return norm_out, sum_out


for i in range(1):
    t = 2048
    n = 4
    num_iters = 20

    eps = 1e-6
    print(f"第{i}次比较，t={t}， n={n}，num_iters={num_iters}")
    x = torch.randn(t, n, n, dtype=torch.float32)

    # 前向传播
    y, norm_list, sum_list = sinkhorn_forward_simple(x, num_iters, eps)
    # 构造输出梯度（假设loss = sum(y^2)）
    grad_output = 2 * y
    norm_out, sum_out = convert_to_npu_format(norm_list, sum_list, num_iters)
    grad_input = torch.ops.custom.npu_sinkhorn_grad(grad_output.npu(), norm_out.npu(), sum_out.npu())

```