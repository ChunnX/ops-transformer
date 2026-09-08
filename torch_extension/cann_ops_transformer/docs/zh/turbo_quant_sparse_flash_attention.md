# cann_ops_transformer.turbo_quant_sparse_flash_attention

## 功能

面向 DeepSeek Sparse Attention（DSA）推理场景，对 TQ4 分页 KV cache 执行融合反量化与稀疏 Attention 计算，支持 Atlas A2/A3 前向推理。query 和输出支持 `torch.float16`、`torch.bfloat16`，内嵌 rope 必须与 query 同型。

## 接口

```python
cann_ops_transformer.turbo_quant_sparse_flash_attention(
    query, key, value, sparse_indices, *,
    key_dequant_scale=None, value_dequant_scale=None,
    block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
    scale_value=1.0, key_quant_mode=3, value_quant_mode=3,
    sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
    sparse_mode=3, pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807, attention_mode=2,
    quant_scale_repo_mode=1, tile_size=128, rope_head_dim=64,
    return_softmax_lse=False,
) -> (attention_out, softmax_max, softmax_sum)
```

## 输入与输出

| 参数 | 说明 |
| --- | --- |
| query | `[T,N,576]`，FP16/BF16；N 为 `[1,128]` 内的 2 的幂 |
| key、value | INT8，PA_BSND `[Bn,Bs,1,386]` 或 PA_BNBD `[Bn,1,Bs,386]`；value 请传入与 key 相同的张量 |
| sparse_indices | INT32 `[T,1,K]`，batch 内稀疏块索引，有效前缀后用 -1 填充 |
| key_dequant_scale、value_dequant_scale | 可选 FLOAT32 张量；当前 COMBINE 模式不读取，可传 None |
| block_table | INT32 `[B,maxBlocks]`，物理页映射 |
| actual_seq_lengths_query | INT32 `[B]`，query 长度累加和；允许重复值，末项为 T |
| actual_seq_lengths_kv | INT32 `[B]`，各 batch KV 长度，允许 0 |
| attention_out | `[T,N,512]`，与 query 同型 |
| softmax_max、softmax_sum | FLOAT32，请求时 `[1,T,N]`，否则 `[0]` |

其余参数取值、量化字节结构和约束与 [ACLNN 接口](../../../../attention/turbo_quant_sparse_flash_attention/docs/aclnnTurboQuantSparseFlashAttention.md)一致。`sparse_block_size` 支持 1/2/4/8/16，`sparse_mode` 支持 0/3；其余模式与维度属性需保持声明的固定值。

全部输入须在同一 NPU。支持两种布局下 cache 首轴非连续、页间 padding 和 storage offset，页内必须连续。`B`、物理 cache 页数和页表列数须大于零，`Bs` 是 16 至 1024 内的 16 的倍数。

空 query 返回空张量；零 KV batch 返回零 attention、max=-2e38、sum=0。索引、长度和页表的设备数值由调用方保证合法。

仅提供前向；Meta/FakeTensor 可推导输出并拒绝不支持的元数据。ACL graph 捕获前需预热完成 JIT 编译，回放只更新固定 shape 张量的内容，不改变地址。当前未提供 GE Converter。

## 示例

```python
import torch
import torch_npu
import cann_ops_transformer

# 单个 KV token：latent 每个 nibble 为 code 8，rope 为零，FLOAT16 scale=1。
query = torch.zeros((1, 8, 576), dtype=torch.float16, device="npu:0")
cache_cpu = torch.zeros((1, 16, 1, 386), dtype=torch.uint8)
cache_cpu[..., :256] = 0x88
cache_cpu[..., 385] = 0x3c
cache = cache_cpu.view(torch.int8).npu()
out, maximum, total = cann_ops_transformer.turbo_quant_sparse_flash_attention(
    query, cache, cache,
    torch.zeros((1, 1, 1), dtype=torch.int32, device=query.device),
    block_table=torch.zeros((1, 1), dtype=torch.int32, device=query.device),
    actual_seq_lengths_query=torch.ones(1, dtype=torch.int32, device=query.device),
    actual_seq_lengths_kv=torch.ones(1, dtype=torch.int32, device=query.device),
    scale_value=1.0 / (512 ** 0.5), return_softmax_lse=True,
)
torch.npu.synchronize()
print(out.shape, maximum.shape, total.shape)
```

旧 Python 调用入口见[兼容说明](../../../../attention/turbo_quant_sparse_flash_attention/torch_ops_extension/README.md)。
