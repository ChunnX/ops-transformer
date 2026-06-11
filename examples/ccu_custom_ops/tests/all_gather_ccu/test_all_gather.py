#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Multi-rank (single-machine 4-card) functional test for the custom CCU AllGather.
# Golden = torch.distributed.all_gather_into_tensor on the same data.

import os
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 4
N = 1024
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "50001"


def run(rank, world_size):
    torch.npu.set_device(rank)
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ.setdefault("HCCL_BUFFSIZE", "256")
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # force HCCL comm creation before updateContext queries the handle
    warm = torch.ones(1, dtype=torch.float32).npu()
    dist.all_reduce(warm)
    torch.npu.synchronize()

    import ascend_ops
    buf = ascend_ops.AllGatherCcu(group)

    # each rank holds a distinct chunk: value (rank+1)
    x = torch.full((N,), float(rank + 1), dtype=torch.float32).npu()

    out = buf.all_gather(x)  # [WORLD_SIZE, N]

    # golden 1: known pattern — out[r] should be all (r+1)
    expected = torch.stack([torch.full((N,), float(r + 1)) for r in range(world_size)]).npu()
    ok_pattern = torch.allclose(out, expected)

    # golden 2: torch native all_gather_into_tensor
    ref = torch.empty(world_size * N, dtype=torch.float32).npu()
    dist.all_gather_into_tensor(ref, x)
    ok_native = torch.allclose(out.reshape(-1), ref)

    maxdiff = float((out.reshape(-1) - ref).abs().max().cpu())
    print(f"[rank {rank}] ok_pattern={ok_pattern} ok_native={ok_native} "
          f"maxdiff={maxdiff} out[:,0]={out[:, 0].cpu().tolist()}", flush=True)

    assert ok_pattern, f"rank {rank}: pattern mismatch out[:,0]={out[:,0].cpu().tolist()}"
    assert ok_native, f"rank {rank}: native all_gather mismatch maxdiff={maxdiff}"
    dist.barrier(group)
    if rank == 0:
        print("ALLGATHER_CCU_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    procs = []
    for r in range(WORLD_SIZE):
        p = mp.Process(target=run, args=(r, WORLD_SIZE))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    rc = max(p.exitcode for p in procs)
    if rc != 0:
        raise SystemExit(f"some rank failed, exitcodes={[p.exitcode for p in procs]}")
