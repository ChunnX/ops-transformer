# aclnnRopeWithSinCosCache

## 功能描述

### 算子功能
对输入的 query 和 key 张量施加 Rotary Position Embedding (RoPE) 旋转位置编码，使用预计算的 sin/cos 缓存表。

### 计算公式
对于每个 token，根据其位置从 cos/sin 缓存表中获取对应的旋转角度，然后对 query 和 key 的 rotary_dim 维度部分施加旋转变换：

**NeoxStyle (旋转切分模式)：**
$$x_1 = x[:, :, :\frac{d}{2}], \quad x_2 = x[:, :, \frac{d}{2}:]$$
$$\text{out} = [x_1 \cos\theta - x_2 \sin\theta, \; x_2 \cos\theta + x_1 \sin\theta]$$

**非 NeoxStyle (交错模式)：**
$$x_{\text{even}} = x[:, :, 0::2], \quad x_{\text{odd}} = x[:, :, 1::2]$$
$$\text{out}_{\text{even}} = x_{\text{even}} \cos\theta - x_{\text{odd}} \sin\theta$$
$$\text{out}_{\text{odd}} = x_{\text{odd}} \cos\theta + x_{\text{even}} \sin\theta$$

其中 $\theta$ 为位置对应的旋转角度，从 `cosSinCache` 中查找。

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.rope_with_sin_cos_cache()` 函数形式提供：

```python
def rope_with_sin_cos_cache(
    positions: torch.Tensor,       # 位置索引 [num_tokens]，dtype=int64
    query_in: torch.Tensor,        # 查询张量 [num_tokens, num_q_heads * head_size]
    key_in: torch.Tensor,          # 键张量 [num_tokens, num_kv_heads * head_size]
    cos_sin_cache: torch.Tensor,   # 缓存表 [max_positions, rotary_dim]
    num_q_heads: int,              # 查询头数
    num_k_heads: int,              # 键头数
    head_size: int,                # 每个头的维度
    mrope_section: List[int],      # 多维 RoPE 分段 (默认 [0,0,0])
    q_stride: int,                 # 查询步长 (默认 num_q_heads * head_size)
    k_stride: int,                 # 键步长 (默认 num_kv_heads * head_size)
    is_neox_style: bool,           # 是否使用 Neox 风格 (默认 True)
    cache_mode: int                # 缓存模式 (默认 0)
) -> Tuple[torch.Tensor, torch.Tensor]  # (query_out, key_out)
```

## 使用案例
```python
import torch
import kernel_gen_ops

num_tokens = 4
num_q_heads = 8
num_kv_heads = 2
head_size = 128
rotary_dim = 128
max_pos = 128

positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64).npu()
query = torch.randn(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.randn(max_pos, rotary_dim, dtype=torch.float32).npu()

query_out, key_out = kernel_gen_ops.rope_with_sin_cos_cache(
    positions, query, key, cos_sin_cache,
    num_q_heads, num_kv_heads, head_size,
    [0, 0, 0], num_q_heads * head_size, num_kv_heads * head_size,
    True, 0)
```

## 约束与限制
- 支持的数据类型：FLOAT32、FLOAT16、BF16
- head_size 和 rotary_dim 必须满足 64 字节对齐（即对于 float32 是 16 的倍数，对于 float16/bf16 是 32 的倍数）
- head_size >= rotary_dim
- positions 值必须在 cos_sin_cache 的第一维范围内
- 支持 NeoxStyle 和非 NeoxStyle 两种旋转模式
- 支持多维 RoPE (mRoPE) 模式（通过 mrope_section 参数配置）
