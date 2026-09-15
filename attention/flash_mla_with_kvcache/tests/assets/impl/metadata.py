"""Build metadata using the MLA companion operator's own schema."""

import torch


def build_metadata(q, k_cache, *, cache_seqlens, cu_seqlens_q=None,
                   seqused_q=None, head_dim_v=512, mask_mode=0,
                   max_seqlen_q=-1, max_seqlen_kv=-1,
                   layout_q="BSND", layout_kv="PA_BBND", **unused):
    if cache_seqlens is None:
        raise ValueError("cache_seqlens is required for MLA metadata")
    return torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata(
        cache_seqlens,
        int(q.shape[1 if layout_q in ("TND", "BNSD") else 2]),
        int(k_cache.shape[2 if layout_kv == "PA_BBND" else 1]),
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=-1 if max_seqlen_q is None else int(max_seqlen_q),
        max_seqlen_kv=-1 if max_seqlen_kv is None else int(max_seqlen_kv),
        head_dim_qk=int(q.shape[-1]),
        head_dim_v=int(head_dim_v),
        mask_mode=int(mask_mode),
        layout_q=layout_q,
    ).to(device=q.device)
