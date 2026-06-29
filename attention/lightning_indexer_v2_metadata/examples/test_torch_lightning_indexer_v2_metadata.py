import torch
import torch_npu
import torchair
import numpy as np
import torch.nn as nn
import cann_ops_transformer

metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
    cu_seqlens_q = torch.tensor([0, 123, 230, 234, 511], dtype=torch.int32).npu(),
    cu_seqlens_k = torch.tensor([0, 3048, 4098*2, 4364*3, 4098*4], dtype=torch.int32).npu(),
    seqused_q = torch.tensor([12, 106, 3, 200], dtype=torch.int32).npu(),
    seqused_k = torch.tensor([2047, 4097, 24, 456], dtype=torch.int32).npu(),
    cmp_residual_k = torch.tensor([0, 0, 0, 0], dtype=torch.int32).npu(),
    batch_size = 4,
    max_seqlen_q = 180,
    max_seqlen_k = 5,
    num_heads_q = 64,
    num_heads_k = 1,
    head_dim = 128,
    topk = 200,
    mask_mode = 0,
    layout_q = "TND",
    layout_k = "TND",
    cmp_ratio = 13
)