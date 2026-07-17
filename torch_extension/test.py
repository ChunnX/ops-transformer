import subprocess
import torch
import torch_npu
import cann_ops_transformer

# 初始化 NPU
torch_npu.npu.set_device(0)

# 创建输入 tensor（可选参数可以传 None）
batch_size = 1
qs = 1
kvs = 1024
num_heads_q = 1
num_heads_kv = 1
head_dim = 128
block_size = 128
num_blocks = 8
block_table = torch.arange(8, dtype=torch.int32, device="npu").unsqueeze(0)


# mask_mode = torch.ops.npu_ops_transformer.npu_mixed_quant_flash_attn._MaskMode.ALL
win_left = -1
win_right = -1
layout_q = "BSND"
layout_kv = "PA_BBND"
layout_out = "BSND"
# quant_mode = torch.ops.npu_ops_transformer.npu_mixed_quant_flash_attn._QuantMode.A16C4_MXFP4
# 创建实际序列长度 tensor（可选）
q = torch.randn(batch_size, qs, num_heads_q, head_dim).to(dtype=torch.float16).npu()
k = torch.randn(num_blocks, block_size, num_heads_kv, head_dim // 2).to(dtype=torch.uint8).npu()
v = torch.randn(num_blocks, block_size, num_heads_kv, head_dim // 2).to(dtype=torch.uint8).npu()
k_descale = torch.randn(num_blocks, block_size, num_heads_kv, int(head_dim / 32)).to(dtype=torch.float8_e8m0fnu).npu()
v_descale = torch.randn(num_blocks, block_size, num_heads_kv, int(head_dim / 32)).to(dtype=torch.float8_e8m0fnu).npu()
seqused_kv = torch.tensor([1024], dtype=torch.int32, device="npu")

metadata = cann_ops_transformer.ops.mixed_quant_flash_attn_metadata(
    num_heads_q = num_heads_q,
    num_heads_kv = num_heads_kv,
    head_dim = head_dim,
    quant_mode = 1,

    batch_size = batch_size,
    max_seqlen_q = 1,
    max_seqlen_kv = 1024,
    mask_mode = 0,
    win_left = -1,
    win_right = -1,
    layout_q = layout_q,
    layout_kv = layout_kv,
    layout_attn_out = layout_out
)


# 调用算子
result = cann_ops_transformer.ops.mixed_quant_flash_attn(
    q = q,
    k = k,
    v = v,
    k_descale = k_descale,
    v_descale = v_descale,
    block_table=block_table,
    cu_seqlens_q=None,
    seqused_q = None,
    seqused_kv = seqused_kv,
    sinks = None,
    attn_mask=None,
    metadata=metadata,
    quant_mode=1,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1, win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_attn_out="BSND",
    return_softmax_lse=0
)