# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""FlashKDA V2 Entry, explicit publication domains and frozen Provider graph."""

import torch
import cannbotdsl as cb

from .kernels import flash_kda_kernel

HEAD_DIM = 128
CHUNK_SIZE = 64
GROUP_CHUNKS = 8


@cb.kernel
def identity_kernel(identity):
    for row in range(16):
        for col in range(16):
            value = cb.dtypes.float32(0.0)
            if row == col:
                value = cb.dtypes.float32(1.0)
            identity[row, col] = cb.dtypes.float16(value)


@cb.jit
def initialize_constants(identity):
    identity_kernel[1](identity)


class FlashKDAProvider:
    """Static device strategy; tensor metadata remains runtime Provider ABI."""

    def __init__(self, blocks):
        self.blocks = blocks

    @cb.jit
    def run(self, k, v, q, beta, g, exchange, a_log, dt_bias,
            restored_k, gamma, decayed_q, w, mqk, u, out,
            initial_state, final_state, identity, scale, lower_bound):
        flash_kda_kernel(HEAD_DIM, GROUP_CHUNKS, 16)[self.blocks](
            k, v, q, beta, g, exchange, a_log, dt_bias,
            restored_k, gamma, decayed_q, w, mqk, u, out,
            initial_state, final_state, identity,
            v.shape[0] * v.shape[1], v.shape[2], scale, lower_bound,
        )


@cb.entry(name="FlashKda")
def flash_kda(q, k, v, g, beta, initial_state, a_log, dt_bias,
              scale, lower_bound):
    """Contiguous BNSD, BF16 tensors, FP32 state, D128, full 64-token chunks.

    beta is BNS and state is B,Nv,Dv,Dk. Returns (output, final_state).
    Publication signatures are the authoritative supported input domains.
    """
    batch = v.shape[0]
    heads = v.shape[1]
    sequence = v.shape[2]
    blocks = cb.target_capabilities().aic_count
    active_chunks = min(sequence // CHUNK_SIZE, GROUP_CHUNKS)
    tokens = batch * heads * active_chunks * CHUNK_SIZE
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    exchange = torch.empty((blocks, 2, CHUNK_SIZE, HEAD_DIM // 2), dtype=torch.float32, device=q.device)
    restored_k = torch.empty((tokens, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    gamma = torch.empty((batch * heads * active_chunks * HEAD_DIM, 1), dtype=torch.float32, device=q.device)
    decayed_q = torch.empty((tokens, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    w = torch.empty((tokens, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    mqk = torch.empty((tokens, CHUNK_SIZE), dtype=torch.bfloat16, device=q.device)
    u = torch.empty((tokens, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    identity = torch.empty((16, 16), dtype=torch.float16, device=q.device)
    initialize_constants(identity)
    FlashKDAProvider(blocks).run(
        k, v, q, beta.unsqueeze(-1), g, exchange, a_log, dt_bias,
        restored_k, gamma, decayed_q, w, mqk, u, out,
        initial_state, final_state, identity, scale, lower_bound,
    )
    return out, final_state


def signatures():
    """Finite dtype/rank/heads variants and a dynamic, bounded sequence domain."""
    batch = cb.Dim("B", min=1, max=2)
    sequence = cb.Dim("S", min=64, max=1024, multiple_of=64)
    f16 = cb.dtypes.bfloat16
    f32 = cb.dtypes.float32
    tensor = cb.TensorSpec
    result = []
    for key_heads, value_heads in ((1, 1), (1, 2), (2, 2)):
        result.append({
            "q": tensor((batch, key_heads, sequence, HEAD_DIM), f16),
            "k": tensor((batch, key_heads, sequence, HEAD_DIM), f16),
            "v": tensor((batch, value_heads, sequence, HEAD_DIM), f16),
            "g": tensor((batch, value_heads, sequence, HEAD_DIM), f16),
            "beta": tensor((batch, value_heads, sequence), f16),
            "initial_state": tensor((batch, value_heads, HEAD_DIM, HEAD_DIM), f32),
            "a_log": tensor((value_heads,), f32),
            "dt_bias": tensor((value_heads, HEAD_DIM), f32),
            "scale": cb.FloatRange(min=0.0, max=1.0),
            "lower_bound": cb.FloatRange(min=-5.0, max=-1.0),
        })
    return tuple(result)


def build(target):
    if not 1 <= target.aic_count <= 28 or target.aiv_count < 2 * target.aic_count:
        raise ValueError("FlashKDA requires explicit 1..28 resident AICs and two AIVs per AIC")
    return cb.build_operator(flash_kda, signatures=signatures(), target=target)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--npu-arch", choices=("dav-3510",), required=True)
    parser.add_argument("--aic-count", type=int, required=True)
    args = parser.parse_args()
    print(build(cb.Target(args.npu_arch, aic_count=args.aic_count,
                          aiv_count=2 * args.aic_count)).save(args.output))
