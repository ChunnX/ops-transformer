"""Model-shape smoke/stress test for the before-aligned sync-only operator."""

import argparse
import ctypes
import math

import torch
import torch_npu

from cann_ops_transformer.ops import mlp_sparse_lightning_indexer_grad_kl_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--op-timeout-s", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.runs < 1 or not 1 <= args.op_timeout_s <= 1000:
        parser.error("runs must be positive and op-timeout-s must be in [1, 1000]")

    if args.device < 0:
        parser.error("device must be nonnegative")
    device = f"npu:{args.device}"
    dtype = torch.bfloat16
    t1 = t2 = 1024
    topk = 2048
    query_heads = 64
    index_heads = 32
    torch_npu.npu.set_device(device)

    def zeros(shape, tensor_dtype):
        return torch.zeros(shape, dtype=tensor_dtype, device=device).contiguous()

    row_indices = torch.arange(topk, dtype=torch.int32).unsqueeze(0)
    valid_by_row = torch.arange(1, t1 + 1, dtype=torch.int32).unsqueeze(1)
    sparse_indices = torch.where(row_indices < valid_by_row, row_indices, -1)
    mask = torch.ones(t1, dtype=torch.int8)
    mask[: t1 // 3] = 0

    inputs = (
        zeros((t1, query_heads, 512), dtype),
        zeros((t2, 1, 512), dtype),
        zeros((t1, index_heads, 128), dtype),
        zeros((t2, 1, 128), dtype),
        zeros((t1, index_heads), torch.float32),
        sparse_indices.unsqueeze(1).contiguous().to(device),
        zeros((t1, query_heads, 8), torch.float32),
        zeros((t1, query_heads, 8), torch.float32),
        zeros((t1, query_heads, 64), dtype),
        zeros((t2, 1, 64), dtype),
        mask.to(device).contiguous(),
    )

    runtime = ctypes.CDLL("libascendcl.so")
    set_timeout = runtime.aclrtSetOpExecuteTimeOut
    set_timeout.argtypes = [ctypes.c_uint32]
    set_timeout.restype = ctypes.c_int
    status = set_timeout(args.op_timeout_s)
    if status:
        raise RuntimeError(f"aclrtSetOpExecuteTimeOut failed: {status}")

    print(f"[SLIG] starting runs={args.runs}, op_timeout_s={args.op_timeout_s}, "
          f"device={device}, TND B=1 Q=KV=1024 BF16 topk=2048", flush=True)
    for run_idx in range(args.runs):
        outputs = mlp_sparse_lightning_indexer_grad_kl_loss(
            *inputs,
            cur_seq_lengths_query=[0, t1],
            cur_seq_lengths_key=[0, t2],
            scale_value=1.0 / math.sqrt(192.0),
            layout="TND",
            sparse_mode=3,
            pre_tokens=512 * 1024,
            next_tokens=512 * 1024,
            block_size=1,
            deterministic=True,
        )
        torch.npu.synchronize()
        if run_idx == 0 or (run_idx + 1) % 100 == 0:
            print(f"[SLIG] completed={run_idx + 1}/{args.runs}", flush=True)

    expected_shapes = (
        (t1, index_heads, 128),
        (t2, 1, 128),
        (t1, index_heads),
        (1,),
    )
    assert tuple(tuple(tensor.shape) for tensor in outputs) == expected_shapes
    assert all(tensor.device.type == "npu" for tensor in outputs)
    assert all(torch.count_nonzero(tensor).cpu().item() == 0 for tensor in outputs)
    print(f"custom mlp_sparse_lightning_indexer_grad_kl_loss passed: "
          f"runs={args.runs}, topk={topk}, Q_heads={query_heads}, index_heads={index_heads}")


if __name__ == "__main__":
    main()
