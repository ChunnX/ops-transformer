import torch
import torch_npu
import torchair
import numpy as np
import torch.nn as nn
import cann_ops_transformer

metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
    cu_seqlens_q = None,
    cu_seqlens_k = None,
    seqused_q = None,
    seqused_k = None,
    cmp_residual_k = torch.tensor([3,3,3,3,3,3,3,3], dtype=torch.int32).npu(),
    batch_size = 8,
    max_seqlen_q = 10,
    max_seqlen_k = 10,
    num_heads_q = 64,
    num_heads_k = 1,
    head_dim = 128,
    topk = 2048,
    quant_mode = 1,
    mask_mode = 3,
    layout_q = "BSND",
    layout_k = "BSND",
    cmp_ratio = 128
)