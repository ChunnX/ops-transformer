# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GBSA torch 调用示例：直接从已安装 cann 包 import（API 对齐 ATK 封装签名）。

不设置/不依赖 GBSA_TORCH_EXTENSION_ROOT、ASCEND_CUSTOM_OPP_PATH、本地 PYTHONPATH。
import 不到或模块不在 site-packages/dist-packages → 直接失败。
"""

import math
import os
import sys
from math import ceil
from typing import Optional

import torch
import torch_npu

try:
    import cann_ops_transformer
    from cann_ops_transformer.ops import (
        generic_block_sparse_attention,
        generic_block_sparse_attention_metadata,
    )
except ImportError as e:
    raise SystemExit(
        f"FAIL: cannot import GBSA from cann package (cann_ops_transformer): {e}"
    ) from e


def _must_be_installed(label: str, path: Optional[str]) -> str:
    if not path:
        raise SystemExit(f"FAIL: {label} has no __file__ (not a real installed module)")
    norm = os.path.normpath(os.path.realpath(path)).replace("\\", "/")
    print(f"[cann] {label} = {norm}")
    if "site-packages" not in norm and "dist-packages" not in norm:
        raise SystemExit(
            f"FAIL: {label} is not from installed cann package "
            f"(expect site-packages/dist-packages): {norm}"
        )
    return norm


_must_be_installed("cann_ops_transformer", getattr(cann_ops_transformer, "__file__", None))

_gbsa_mod = sys.modules.get(
    "cann_ops_transformer.ops.attention.generic_block_sparse_attention"
) or sys.modules.get("cann_ops_transformer.ops.generic_block_sparse_attention")
if _gbsa_mod is not None:
    _must_be_installed("gbsa module", getattr(_gbsa_mod, "__file__", None))
else:
    # ops 聚合导出时可能没有独立子模块文件，至少确认 ops 包路径
    _ops = sys.modules.get("cann_ops_transformer.ops")
    _must_be_installed("cann_ops_transformer.ops", getattr(_ops, "__file__", None))

print("[cann] OK: loaded from installed package")

torch_npu.npu.set_device(0)
device = torch.device("npu:0")
print(f"[env] device={torch_npu.npu.get_device_name(0)}")

# ---- case（对齐 ATK: TND + PA_BBND）----
q_seqlen = 4
kv_seqlen = 256
num_heads_q = 8
num_heads_kv = 2
head_dim = 128
block_size = 128
topk = 4
quant_mode = 0
softmax_precision = 1  # A2/A3: fp16 0|1；bf16 须 0；Ascend950 须 1
dtype = torch.float16
block_shape = [1, block_size]
softmax_scale = 1.0 / math.sqrt(head_dim)

batch = 1
total_blocks = ceil(kv_seqlen / block_size)
n_phys = batch * total_blocks

torch.manual_seed(42)

query = (torch.rand(q_seqlen, num_heads_q, head_dim) * 2 - 1).to(dtype).to(device)
key = (torch.rand(n_phys, block_size, num_heads_kv, head_dim) * 2 - 1).to(dtype).to(device)
value = (torch.rand(n_phys, block_size, num_heads_kv, head_dim) * 2 - 1).to(dtype).to(device)

his = kv_seqlen - q_seqlen
sparse_block_idx = torch.full((num_heads_kv, q_seqlen, topk), -1, dtype=torch.int32)
sparse_block_count = torch.zeros((num_heads_kv, q_seqlen), dtype=torch.int32)
for h in range(num_heads_kv):
    for t in range(q_seqlen):
        q_block = (his + t) // block_size
        visible = min(total_blocks, q_block + 1)
        valid = min(topk, visible)
        sparse_block_count[h, t] = valid
        start = max(0, visible - valid)
        for i, bid in enumerate(range(start, start + valid)):
            sparse_block_idx[h, t, i] = bid
sparse_block_idx = sparse_block_idx.to(device)
sparse_block_count = sparse_block_count.to(device)

block_table = torch.arange(n_phys, dtype=torch.int32).view(batch, total_blocks).to(device)
cu_seqlens_q = torch.tensor([0, q_seqlen], dtype=torch.int64, device=device)
seqused_q = torch.tensor([q_seqlen], dtype=torch.int32, device=device)
seqused_kv = torch.tensor([kv_seqlen], dtype=torch.int32, device=device)

metadata = generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    num_heads_q,
    num_heads_kv,
    head_dim,
    block_shape,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=None,
    seqused_q=seqused_q,
    seqused_kv=seqused_kv,
    max_seqlen_q=q_seqlen,
    max_seqlen_kv=kv_seqlen,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="PA_BBND",
    mask_mode=1,
    quant_mode=quant_mode,
    softmax_precision=softmax_precision,
    win_left=-1,
    win_right=-1,
)

attention_out, _ = generic_block_sparse_attention(
    query,
    key,
    value,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    metadata=metadata,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=None,
    seqused_q=seqused_q,
    seqused_kv=seqused_kv,
    block_table=block_table,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="PA_BBND",
    softmax_scale=softmax_scale,
    mask_mode=1,
    quant_mode=quant_mode,
    dst_type_max=0.0,
    softmax_precision=softmax_precision,
    win_left=-1,
    win_right=-1,
    return_softmax_lse=False,
)

torch.npu.synchronize()
print(f"[ok] metadata shape: {tuple(metadata.shape)}")
print(f"[ok] attention_out shape: {tuple(attention_out.shape)} dtype: {attention_out.dtype}")
print(f"[ok] attention_out abs max: {attention_out.float().abs().max().item():.6g}")
