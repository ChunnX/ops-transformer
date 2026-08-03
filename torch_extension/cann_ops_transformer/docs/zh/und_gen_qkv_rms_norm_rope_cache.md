# und_gen_qkv_rms_norm_rope_cache

## 功能说明

und/gen 两段 QKV 的融合前处理算子：按 `cat_indices` 间接寻址拼出输出序列，逐 token 做
Split + RMSNorm(Q/K) + MRoPE(Q/K)，Q 作为返回值输出，K/V 经 Cast float32→bf16 后按 `slot_mapping`
写入分页 KV Cache（`k_cache`/`v_cache` 原地更新）。

## 函数原型

```python
import torch
import cann_ops_transformer

q = torch.ops.cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
    und_qkv, und_weights_q, und_weights_k, cos_sin_cache,
    k_cache, v_cache, slot_mapping, positions,
    gen_qkv=None, gen_weights_q=None, gen_weights_k=None, cat_indices=None,
    num_heads_q=8, num_heads_k=1, num_heads_v=1,
    norm_eps=1e-6, mrope_section=[16, 16, 16])
```

## 参数说明

| 参数名 | 类型 | 说明 |
| :----- | :--- | :--- |
| und_qkv | Tensor(bfloat16) | [und_len, N, D]，N = Hq + Hk + Hv，D = 128 |
| und_weights_q / und_weights_k | Tensor(bfloat16) | [D]，und 段 Q/K 的 RMSNorm 权重 |
| cos_sin_cache | Tensor(float32) | [max_pos, D]，前半 cos、后半 sin |
| k_cache | Tensor(bfloat16) | [Bn, Bs, Hk, D]，连续 BBND，原地写入 |
| v_cache | Tensor(bfloat16) | [Bn, Bs, Hv, D]，连续 BBND，原地写入 |
| slot_mapping | Tensor(int64) | [T]，slot = block_num * Bs + row_idx |
| positions | Tensor(int64) | [3, T]，MRoPE 三轴位置 |
| gen_qkv | Tensor(bfloat16)，可选 | [gen_len, N, D]，纯 prefill 时传 None |
| gen_weights_q / gen_weights_k | Tensor(bfloat16)，可选 | [D]，与 gen_qkv 同进同出 |
| cat_indices | Tensor(int64)，可选 | [T]，out_t → src_t；None 时恒等映射 |
| num_heads_q / num_heads_k / num_heads_v | int | Q/K/V 头数 |
| norm_eps | float | RMSNorm epsilon，默认 1e-6 |
| mrope_section | list[int] | 三轴分段，空列表等价 [D/2, 0, 0]（标准 RoPE） |

## 返回值

`q`：Tensor(bfloat16)，shape 为 [T, Hq, D]，T = und_len + gen_len。
`k_cache` / `v_cache` 为原地更新，调用后直接读取即可。

## 约束说明

- 仅支持 Ascend 950PR / Ascend 950DT。
- head_dim 固定 128；(Hq, Hk, Hv) 支持 (8, 1, 1) 与 (16, 2, 2)；block_size = 128。
- T = und_len + gen_len 不设上限，只要求为正；真实上限是 KV Cache 容量，要求 `Bn * Bs >= T`。上板已验证到 T = 64K。
- `slot_mapping` 取值必须唯一；`k_cache`/`v_cache` 必须内存连续。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer

D, Hq, Hk, Hv = 128, 8, 1, 1
N = Hq + Hk + Hv
und_len, gen_len = 5, 3
total = und_len + gen_len
block_num, block_size = 2, 128

und_qkv = torch.randn(und_len, N, D, dtype=torch.bfloat16).npu()
gen_qkv = torch.randn(gen_len, N, D, dtype=torch.bfloat16).npu()
w = [torch.randn(D, dtype=torch.bfloat16).npu() for _ in range(4)]
cos_sin_cache = torch.randn(32, D, dtype=torch.float32).npu()
k_cache = torch.zeros(block_num, block_size, Hk, D, dtype=torch.bfloat16).npu()
v_cache = torch.zeros(block_num, block_size, Hv, D, dtype=torch.bfloat16).npu()
slot_mapping = torch.arange(total, dtype=torch.int64).npu()
positions = torch.randint(0, 32, (3, total), dtype=torch.int64).npu()
cat_indices = torch.tensor([0, 5, 1, 6, 2, 7, 3, 4], dtype=torch.int64).npu()

q = torch.ops.cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
    und_qkv, w[0], w[1], cos_sin_cache, k_cache, v_cache, slot_mapping, positions,
    gen_qkv, w[2], w[3], cat_indices,
    num_heads_q=Hq, num_heads_k=Hk, num_heads_v=Hv,
    norm_eps=1e-6, mrope_section=[16, 16, 16])
print(q.shape, k_cache.shape)
```
