# ApplyRotaryPosEmb

## 功能描述

### 算子功能

对 Query、Key 张量在最后一维上应用旋转位置编码（RoPE，**half** 模式，与 aclnn 默认 `rotary_mode="half"` 一致）：将最后一维分为前后两半，按 `cos`/`sin` 做旋转变换，结果写回 query、key 缓冲区（就地）。

### 计算公式

记最后一维长度为 \(D\)，半长 \(H=D/2\)。对张量最后一维按半段处理：

- 将 \(\sin\) 的前 \(H\) 个元素取负得到 \(\sin'\)；
- 记 \(\text{swap}(x) = \text{concat}(x_{[:,H:]},\, x_{[:,:H]})\)（后半与前半拼接）；
- 对每个输入 \(x \in \{q,\,k\}\)：

\[
x_{\text{out}} = x \odot \cos + \text{swap}(x) \odot \sin'
\]

（实现中与广播后的 \(\cos,\sin\) 逐元素运算。）

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.apply_rotary_pos_emb()` 形式提供：

```python
def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    layout: int,
):
    """
    对 query、key 就地应用旋转位置编码。

    参数:
        query (Tensor): Query，与 key 同 shape、同 dtype，ND 格式。
        key (Tensor): Key，与 query 同 shape、同 dtype。
        cos (Tensor): 与 sin 同 shape；最后一维为 head_dim（64 或 128）；中间 head 维须为 1（与 tiling 约束一致）。
        sin (Tensor): 与 cos 同 shape、同 dtype。
        layout (int): 数据排布，本任务验证使用 BSND=1 或 TND=4。

    返回:
        List[Tensor]: [query, key]，即就地改写后的两路张量。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

layout = 1  # BSND
b, s, n, d = 2, 8, 4, 128
q = torch.randn(b, s, n, d, device="npu", dtype=torch.float16)
k = torch.randn(b, s, n, d, device="npu", dtype=torch.float16)
cos = torch.randn(b, s, 1, d, device="npu", dtype=torch.float16)
sin = torch.randn(b, s, 1, d, device="npu", dtype=torch.float16)
out_q, out_k = kernel_gen_ops.apply_rotary_pos_emb(q, k, cos, sin, layout)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含 conda、CANN、shell、`run_multi_test`、设备错误或任何**运行环境 / 排障**说明。
- 支持 dtype：float16、bfloat16、float32；四路输入 dtype 须一致。
- layout 仅支持 **BSND(1)** 与 **TND(4)**：BSND 时 rank 为 4；TND 时 rank 为 3。
- `cos`/`sin` 最后一维（head_dim）须为 **64 或 128**；query/key 最后一维须为相同值且与 `cos`/`sin` 的 head_dim 一致（本 validation 用例满足该条件）。
- `cos`/`sin` 在 head 维上须为 1（与 tiling 中 `coscNum==1` 一致）。
