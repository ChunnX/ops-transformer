#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# run_flash_mla_op.py - flash_mla_with_kvcache 单算子调用脚本（Ascend 950）
#
# 前置（见同目录 compile_and_call_mla.md）：
#   1. 编译并安装 custom 算子包：
#        bash build_mla.sh                         # 全量（增量不可信，见 md 1.1 注）
#        ./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=<dir>
#        source <dir>/vendors/custom_transformer/bin/set_env.bash
#   2. 编译并安装 torch extension（整包，包名 cann_ops_transformer）：
#        bash build.sh --torch_extension
#        python3 -m pip install build_out/cann_ops_transformer-1.0.0-py3-none-any.whl --force-reinstall --no-deps
#
# 运行：python3 run_flash_mla_op.py
# 若内核报错，可用 ASCEND_LAUNCH_BLOCKING=1 python3 run_flash_mla_op.py 拿同步堆栈。
#
# 重要：mla 只支持分页 KV（layout_kv ∈ PA_NZ / PA_BBND / PA_BNBD），连续
# KV 不被支持（aclnn checker EZ0026）。本脚本用 test.py 同款已验证形态：
# decode 形态（max_seqlen_q=1）、PA_BBND 块池 + block_table。
# =====================================================================
import math

import torch
import torch_npu  # noqa: F401  必须最先 import，注册 PrivateUse1 后端

# 整包导入：cann_ops_transformer（不带 --ops/--vendor_name 编译时包名）
import cann_ops_transformer  # noqa: F401
from cann_ops_transformer.ops import flash_mla_with_kvcache, flash_mla_with_kvcache_metadata


def main():
    print("=== flash_mla_with_kvcache 单算子冒烟（decode + PA_BBND 分页 KV）===")
    num_heads_q = 8
    num_heads_kv = 1  # MLA 约束：kv head 数必须为 1
    head_dim_qk = 576  # nope 512 + rope 64 合并
    head_dim_v = 512
    block_size = 16  # 分页块大小（>=16 且 %16==0）
    batch = 2
    seqlen_q = 1  # decode：每个 batch 一个 query token
    cache_lens = [1024, 4096]

    cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32, device="npu")

    # ---- 1) metadata（AICPU 伴生算子，生成主算子 workspace metadata）----
    # 注意：layout_q 非 TND 时不得传 cu_seqlens_q（aclnn checker 拒绝）
    meta = flash_mla_with_kvcache_metadata(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        seqused_q=None,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=max(cache_lens),
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=0,
        layout_q="BSND",
    )
    print(f"[metadata] shape={tuple(meta.shape)} dtype={meta.dtype}")

    # ---- 2) PA_BBND 分页 KV 块池 + block_table ----
    # 块池: [num_blocks, block_size, kv_head=1, 576]
    nb0, nb1 = math.ceil(cache_lens[0] / block_size), math.ceil(cache_lens[1] / block_size)
    nb_tot = nb0 + nb1
    k_cache = torch.randn(nb_tot, block_size, num_heads_kv, head_dim_qk, dtype=torch.float16).npu()
    # block_table: [batch, max_blocks_per_batch]，每行是本 batch 的块索引
    bt = torch.zeros(batch, max(nb0, nb1), dtype=torch.int32, device="npu")
    bt[0, :nb0] = torch.arange(nb0, dtype=torch.int32)
    bt[1, :nb1] = torch.arange(nb0, nb0 + nb1, dtype=torch.int32)

    # ---- 3) 主算子（cube）----
    # q: BSND [batch, seqlen_q, num_heads_q, 576]（BSND = B,S,N,D）
    q = torch.randn(batch, seqlen_q, num_heads_q, head_dim_qk, dtype=torch.float16).npu()

    attn_out, _lse = flash_mla_with_kvcache(
        q,
        k_cache,
        block_table=bt,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=None,
        seqused_q=None,
        metadata=meta,
        head_dim_v=head_dim_v,
        softmax_scale=1.0 / math.sqrt(head_dim_qk),
        mask_mode=0,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=max(cache_lens),
        layout_q="BSND",
        layout_kv="PA_BBND",
        layout_out="BSND",
    )
    print("[main op] out shape:", tuple(attn_out.shape))
    print("[main op] out finite:", torch.isfinite(attn_out.float()).all().item())
    print("=== OK ===")


if __name__ == "__main__":
    main()