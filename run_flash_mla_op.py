#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
# run_flash_mla_op.py - flash_mla_with_kvcache 单算子调用脚本（Ascend 950）
#
# 前置（见同目录 compile_and_call_mla.md）：
#   1. 编译并安装 custom 算子包：
#        bash build_mla.sh
#        ./build_out/cann-ops-transformer-custom_linux-x86_64.run [--install-path=<dir>]
#        source <vendor_install>/vendors/custom_transformer/bin/set_env.bash  # 指定目录安装时
#   2. 编译并安装 torch extension（整包，包名 cann_ops_transformer）：
#        bash build.sh --torch_extension
#        python3 -m pip install build_out/cann_ops_transformer-1.0.0-py3-none-any.whl --force-reinstall --no-deps
#
# 运行：python3 run_flash_mla_op.py
# 若内核报错，可用 ASCEND_LAUNCH_BLOCKING=1 python3 run_flash_mla_op.py 拿同步堆栈。
# =====================================================================
import torch
import torch_npu  # noqa: F401  必须最先 import，注册 PrivateUse1 后端

# 整包导入：cann_ops_transformer（不带 --ops/--vendor_name 编译时包名）
import cann_ops_transformer
from cann_ops_transformer.ops import flash_mla_with_kvcache

# metadata 伴生算子：cann_ops_transformer.ops 顶层只 re-export 主算子，需从子模块导入
from cann_ops_transformer.ops.attention.flash_mla_with_kvcache import (
    flash_mla_with_kvcache_metadata,
)


def main():
    print("=== flash_mla_with_kvcache 单算子冒烟 ===")
    num_heads_q = 8
    num_heads_kv = 1  # MLA 约束：kv head 数必须为 1
    head_dim_qk = 576  # nope 512 + rope 64 合并
    head_dim_v = 512
    batch = 2
    seqlen_q = 2
    seqlen_kv = 4

    cache_seqlens = torch.tensor(
        [seqlen_kv, seqlen_kv - 1], dtype=torch.int32, device="npu"
    )
    seqused_q = torch.tensor([seqlen_q, seqlen_q], dtype=torch.int32, device="npu")

    # ---- 1) metadata（AICPU 伴生算子，生成主算子 workspace metadata）----
    # 注意：layout_q 非 TND 时不得传 cu_seqlens_q（aclnn checker 拒绝）
    meta = flash_mla_with_kvcache_metadata(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        seqused_q=seqused_q,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=seqlen_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=0,
        layout_q="BSND",
    )
    print(f"[metadata] shape={tuple(meta.shape)} dtype={meta.dtype}")
    print(f"[metadata] values={meta.tolist()}")

    # ---- 2) 主算子（cub）----
    # BNSD: [batch, n2(=kv_head), seq, 576]
    q = torch.randn(
        batch, num_heads_q, seqlen_q, head_dim_qk, dtype=torch.float16
    ).npu()
    k_cache = torch.randn(
        batch, num_heads_kv, seqlen_kv, head_dim_qk, dtype=torch.float16
    ).npu()

    out, lse = flash_mla_with_kvcache(
        q,
        k_cache,
        cache_seqlens=cache_seqlens,
        seqused_q=seqused_q,
        metadata=meta,
        head_dim_v=head_dim_v,
        softmax_scale=1.0,
        mask_mode=0,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=seqlen_kv,
        layout_q="BNSD",
        layout_kv="BNSD",
    )
    print("[main op] out shape:", tuple(out.shape), "lse shape:", tuple(lse.shape))
    print("[main op] out sample:", out.flatten()[:4].cpu().tolist())
    print("=== OK ===")


if __name__ == "__main__":
    main()
