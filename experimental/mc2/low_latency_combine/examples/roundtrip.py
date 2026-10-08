# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Identity-expert Dispatch/Combine roundtrip; torchrun, 8 ranks per host."""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from ops_transformer_mc2 import ElasticBuffer


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (16, 32, 64, 128):
        raise ValueError("This example needs EP16/32/64/128")
    b, h, k, experts = 8, 7168, 8, 128
    indices = (torch.arange(b * k, device="npu").reshape(b, k) % experts).to(
        torch.int64
    )
    weights = torch.full((b, k), 1 / k, dtype=torch.float32, device="npu")
    x = torch.full((b, h), rank + 1, dtype=torch.bfloat16, device="npu")
    buffer = ElasticBuffer(group=dist.group.WORLD, explicitly_destroy=True)
    try:
        expanded, _, _, handle, event = buffer.dispatch(
            x,
            topk_idx=indices,
            topk_weights=weights,
            num_experts=experts,
            do_cpu_sync=True,
        )
        event.current_stream_wait()
        # Repeat combine to exercise the persistent bank/magic rotation.
        for _ in range(3):
            y, _, event = buffer.combine(expanded, handle)
            event.current_stream_wait()
            torch.testing.assert_close(y, x, rtol=0, atol=0)
        torch.npu.synchronize()
        print(f"rank={rank}: dispatch/combine roundtrip passed", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
