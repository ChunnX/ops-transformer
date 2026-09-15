"""Generate MLA metadata inside the torch graph, as in FlashAttn assets."""

import torch
from .metadata import build_metadata


class FlashMlaWithKvcacheAclGraph(torch.nn.Module):
    def __init__(self, *, head_dim_v=512, softmax_scale=1.0, mask_mode=0,
                 max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="BSND",
                 layout_kv="PA_BBND", layout_out="BSND", return_softmax_lse=False):
        super().__init__()
        self.attrs = dict(head_dim_v=head_dim_v, softmax_scale=softmax_scale,
                          mask_mode=mask_mode, max_seqlen_q=max_seqlen_q,
                          max_seqlen_kv=max_seqlen_kv, layout_q=layout_q,
                          layout_kv=layout_kv, layout_out=layout_out,
                          return_softmax_lse=return_softmax_lse)

    def forward(self, q, k_cache, *, block_table=None, cache_seqlens=None,
                cu_seqlens_q=None, seqused_q=None, attn_mask=None, metadata=None):
        tensors = dict(block_table=block_table, cache_seqlens=cache_seqlens,
                       cu_seqlens_q=cu_seqlens_q, seqused_q=seqused_q,
                       attn_mask=attn_mask)
        metadata = build_metadata(q, k_cache, **tensors, **self.attrs)
        return torch.ops.cann_ops_transformer.flash_mla_with_kvcache(
            q, k_cache, **tensors, metadata=metadata, **self.attrs)
