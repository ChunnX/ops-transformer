#!/usr/bin/env python3
"""flash_mla_with_kvcache 单算子调用脚本(完全按 python 接口契约)。

python 接口权威: cann_ops_transformer.ops.flash_mla_with_kvcache[_metadata]
(torch.ops.cann_ops_transformer.*, 见 site-packages/cann_ops_transformer/ops/flash_mla_with_kvcache.py)
+ 官方文档 docs/zh/flash_mla_with_kvcache.md。

运行前先 source 两个环境脚本:
    source /home/developer/Ascend/ascend-toolkit/latest/bin/setenv.bash
    source /home/developer/Ascend/ascend-toolkit/latest/vendors/custom_transformer/bin/set_env.bash

用法:
    python scripts/flash_mla_with_kvcache_single.py                       # TND/PA_NZ/mask0 默认
    python scripts/flash_mla_with_kvcache_single.py --mask 3              # RIGHT_DOWN, 自动构造 attn_mask
    python scripts/flash_mla_with_kvcache_single.py --dtype fp16 --q-tokens 4096

接口契约要点(本脚本已按此构造):
  - q / k_cache 末维 = 576 (head_dim_v 512 + rope 64), 不允许 512
  - num_heads_kv = 1 (MLA 单 latent KV 头)
  - head_dim_v 只支持 512
  - layout_kv 仅 "PA_NZ" / "PA_BBND" / "PA_BNBD"
  - block_size: 1024 >= bs >= 16 且 bs % 16 == 0 (实测 R1 用 16)
  - mask_mode 仅 0 / 3; 3 时必传 2048x2048 int8 下三角 attn_mask (对角线 0), 0 时禁止传
  - layout_out 必须 == layout_q (不支持转置); TND 时必传 cu_seqlens_q
  - 流程: 先 flash_mla_with_kvcache_metadata 生成 metadata, 再传入主算子
"""

import argparse
import math

import torch
import torch_npu  # noqa: F401  确保 torch_npu 载入

from cann_ops_transformer_custom import (  # noqa: E402
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_metadata,
)

HEAD_DIM_V = 512  # nope/value 宽度 (算子只支持 512)
ROPE_DIM = 64  # rope 已合并进 q/k_cache 末 64 维
HEAD_DIM = HEAD_DIM_V + ROPE_DIM  # 576


def build_mask(mask_mode: int, device) -> torch.Tensor:
    """mask_mode=3 (RIGHT_DOWN) 时构造 2048x2048 int8 下三角 attn_mask, 对角线全 0。"""
    if mask_mode == 0:
        return None
    m = torch.tril(torch.ones(2048, 2048, dtype=torch.int8, device=device))
    return m.fill_diagonal_(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", type=int, default=0, choices=[0, 3])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--q-tokens", type=int, default=8, help="TND 总 query tokens (Q_T)")
    p.add_argument("--q-heads", type=int, default=8, help="Q 头数 (Q_N)")
    p.add_argument(
        "--cache-len", type=int, default=16, help="单 batch KV cache 长度 (S)"
    )
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument(
        "--lse", action="store_true", default=True, help="return_softmax_lse"
    )
    args = p.parse_args()

    torch.manual_seed(42)
    device = "npu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    b, qt, qn = 1, args.q_tokens, args.q_heads
    kv_n, bs, nb = 1, args.block_size, args.num_blocks
    s = args.cache_len
    assert qn > 0 and b * s >= 1

    # ---- 输入构造 (全部在 NPU) ----
    q = torch.randn(qt, qn, HEAD_DIM, dtype=dtype, device=device)
    # k_cache: PA_NZ 5 维 (num_blocks, KV_N, DSUB=576/16, block_size, 16)
    k_cache = torch.randn(nb, kv_n, HEAD_DIM // 16, bs, 16, dtype=dtype, device=device)
    cu_seqlens_q = torch.tensor([0, qt], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([s] * b, dtype=torch.int32, device=device)
    # block_table: (B, max_blocks_per_seq), 值 = k_cache 中的块索引 (正整数)
    block_table = torch.arange(nb, dtype=torch.int32, device=device).view(b, -1)
    attn_mask = build_mask(args.mask, device)

    print(
        f"[单算子调用] layout=TND/PA_NZ(mask={args.mask}) "
        f"q=({qt},{qn},576) k_cache=({nb},{kv_n},{HEAD_DIM // 16},{bs},16) "
        f"cache_seqlens={s} dtype={args.dtype}"
    )

    # ---- 1) metadata ----
    print("[1] flash_mla_with_kvcache_metadata ...", flush=True)
    try:
        metadata = flash_mla_with_kvcache_metadata(
            cache_seqlens=cache_seqlens,
            num_heads_q=qn,
            num_heads_kv=kv_n,
            head_dim_qk=HEAD_DIM,
            head_dim_v=HEAD_DIM_V,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            max_seqlen_q=s,
            max_seqlen_kv=s,
            mask_mode=args.mask,
            layout_q="TND",
        )
        torch.npu.synchronize()
        print(f"    +OK metadata shape={tuple(metadata.shape)} dtype={metadata.dtype}")
    except Exception as e:  # noqa: BLE001  打印完整异常便于定位
        print(f"    -FAIL: {e}")
        return

    # ---- 2) 主算子 ----
    print("[2] flash_mla_with_kvcache ...", flush=True)
    try:
        attn_out, softmax_lse = flash_mla_with_kvcache(
            q,
            k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            attn_mask=attn_mask,
            metadata=metadata,
            head_dim_v=HEAD_DIM_V,
            softmax_scale=1.0 / math.sqrt(HEAD_DIM_V),
            mask_mode=args.mask,
            max_seqlen_q=s,
            max_seqlen_kv=s,
            layout_q="TND",
            layout_kv="PA_NZ",
            layout_out="TND",
            return_softmax_lse=args.lse,
        )
        torch.npu.synchronize()
        print(
            f"    +OK attn_out={tuple(attn_out.shape)} dtype={attn_out.dtype} "
            f"lse={tuple(softmax_lse.shape) if args.lse else None}"
        )
        print(
            f"    finite(attn_out)={torch.isfinite(attn_out.float()).all().item()}"
            + (f" lse@0={softmax_lse[0, 0].item():.4f}" if args.lse else "")
        )
    except Exception as e:  # noqa: BLE001
        print(f"    -FAIL: {e}")
        raise


if __name__ == "__main__":
    main()
