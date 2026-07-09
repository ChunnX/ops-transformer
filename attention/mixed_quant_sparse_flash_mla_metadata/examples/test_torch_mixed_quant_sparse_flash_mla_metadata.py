import torch
import torch_npu
import torchair
import cann_ops_transformer
import numpy as np
import torch.nn as nn

metadata = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
    cu_seqlens_q = torch.tensor([0, 10], dtype=torch.int32).npu(),
    cu_seqlens_ori_kv = None,
    cu_seqlens_cmp_kv = None,
    seqused_q = None,
    seqused_ori_kv = torch.tensor([8192], dtype=torch.int32).npu(),
    seqused_cmp_kv = torch.tensor([64], dtype=torch.int32).npu(),
    cmp_residual_kv = torch.tensor([1], dtype=torch.int32).npu(),
    ori_topk_length = None,
    cmp_topk_length = None,
    num_heads_q = 128,
    num_heads_kv = 1,
    head_dim = 512,
    quant_mode = 1,
    batch_size = 1,
    max_seqlen_q = 1,
    max_seqlen_ori_kv = 512,
    max_seqlen_cmp_kv = 32,
    ori_topk = 0,
    cmp_topk = 512,
    rope_head_dim = 64,
    cmp_ratio = 4,
    ori_mask_mode = 4,
    cmp_mask_mode = 3,
    ori_win_left = 127,
    ori_win_right = 0,
    layout_q = "TND",
    layout_kv = "PA_BBND",
    has_ori_kv = True,
    has_cmp_kv = True
)