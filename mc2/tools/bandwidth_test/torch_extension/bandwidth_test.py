# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from typing import Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ...mc2.common import CommContextManager


# Mc2MoeContext: two uint32 fields, one uint64, and two uint64[1024] arrays.
_CONTEXT_INT32_SIZE = (2 * 4 + 8 + 2 * 1024 * 8) // 4
_CONTEXT_SEQUENCE = {}


class BandwidthTestOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("bandwidth_test", category="tools")

    def sources(self):
        return ["csrc/tools/bandwidth_test.cpp"]

    def schema(self):
        # Avoid the mode parameter used by PyTorch auto-functionalization handlers.
        # Communication changes context-owned memory, even when outputs are unused.
        return (
            "bandwidth_test(Tensor(a!) context, Tensor x, Tensor dstrank_id, str group, "
            "int world_size, int max_bs, int test_mode, int ccl_buffer_size, *, "
            'str comm_alg="mte", int aiv_num=-1) -> (Tensor, Tensor)'
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def bandwidth_test_meta(context, x, dstrank_id, group, world_size,
                                max_bs, test_mode, ccl_buffer_size, comm_alg="mte", aiv_num=-1):
            _check_inputs(context, x, dstrank_id, group, world_size, max_bs,
                          test_mode, ccl_buffer_size, comm_alg, aiv_num)
            return (x.new_empty((max_bs * world_size, x.size(1))),
                    x.new_empty((world_size,), dtype=torch.int32))


def _check_inputs(context, x, dstrank_id, group, world_size, max_bs,
                  test_mode, ccl_buffer_size, comm_alg, aiv_num):
    torch._check(x.dim() == 2, lambda: "x must have shape [bs, hidden]")
    torch._check(x.dtype == torch.float16, lambda: "BandwidthTest supports only float16")
    torch._check(dstrank_id.dim() == 1 and dstrank_id.dtype == torch.int32,
                 lambda: "dstrank_id must be a one-dimensional int32 tensor")
    torch._check(0 < world_size <= 1024, lambda: "world_size must be in [1, 1024]")
    torch._check(0 < max_bs <= (2**31 - 1) // world_size,
                 lambda: "max_bs * world_size must fit int32")
    torch._check(0 < x.size(0), lambda: "bs must be positive")
    # The mainline kernel uses max_bs as its input row count.
    torch._check(x.size(0) == max_bs, lambda: "bs must equal max_bs")
    torch._check(dstrank_id.size(0) == x.size(0), lambda: "dstrank_id length must equal bs")
    torch._check(0 < x.size(1) < 32768, lambda: "hidden must be in (0, 32768)")
    torch._check(x.size(1) % 16 == 0, lambda: "each float16 row must be 32-byte aligned")
    torch._check(context.dim() == 1 and context.dtype == torch.int32,
                 lambda: "context must be a one-dimensional int32 tensor")
    torch._check(context.numel() >= _CONTEXT_INT32_SIZE,
                 lambda: "context is smaller than Mc2MoeContext")
    torch._check(context.device == x.device and dstrank_id.device == x.device,
                 lambda: "all inputs must be on the same device")
    torch._check(0 < len(group.encode()) < 128 and "\0" not in group,
                 lambda: "group must be a valid HCCL group name")
    torch._check(test_mode in (0, 1), lambda: "test_mode must be 0 or 1")
    torch._check(comm_alg in ("", "mte", "urma"), lambda: "comm_alg must be mte or urma")
    torch._check(aiv_num == -1 or aiv_num > 0, lambda: "aiv_num must be -1 or positive")
    # Kernel divides the data window in two and reserves one max_bs block per half.
    row_bytes = ((x.size(1) * 2 + 511) // 512) * 512
    required_bytes = 1024 * 1024 + 2 * (world_size + 1) * max_bs * row_bytes
    torch._check(ccl_buffer_size >= required_bytes,
                 lambda: "HCCL buffer too small; increase HCCL_BUFFSIZE before creating the group")


_bandwidth_test_op_builder = BandwidthTestOpBuilder()
_bandwidth_test_op_builder._ensure_initialized()


@impl(get_as_library(), "bandwidth_test", "PrivateUse1")
def _bandwidth_test(context, x, dstrank_id, group, world_size, max_bs, test_mode,
                    ccl_buffer_size, comm_alg="mte", aiv_num=-1):
    _check_inputs(context, x, dstrank_id, group, world_size, max_bs,
                  test_mode, ccl_buffer_size, comm_alg, aiv_num)
    return _bandwidth_test_op_builder.load().bandwidth_test(
        context, x, dstrank_id, group, world_size, max_bs, test_mode,
        ccl_buffer_size, comm_alg, aiv_num)


class BandwidthTestBuffer:
    """Own the Torch context for an initialized HCCL ProcessGroup.

    All group ranks must construct this object on their current Ascend950 NPU,
    outside torch.compile. MTE requires one UB memory domain; URMA requires
    UBC_CTP links provided by the shared context manager. Set
    HCCL_BUFFSIZE before group creation and retain this object and the group
    until all operations and graph replays have completed.
    """

    def __init__(self, group, comm_alg="mte"):
        if comm_alg not in ("", "mte", "urma"):
            raise ValueError("comm_alg must be mte or urma")
        self.comm_alg = comm_alg or "mte"
        self.group = group
        self.world_size = torch.distributed.get_world_size(group)
        if not 0 < self.world_size <= 1024:
            raise ValueError("world_size must be in [1, 1024]")
        rank = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        # Match the mainline A5 tiling: twice HCCL_BUFFSIZE (MiB), default 200.
        buffer_config = os.environ.get("HCCL_BUFFSIZE", "200")
        if not buffer_config.isascii() or not buffer_config.isdecimal():
            raise ValueError("HCCL_BUFFSIZE must contain ASCII decimal digits")
        buffer_mb = int(buffer_config)
        if not 0 < buffer_mb <= 65535:
            raise ValueError("HCCL_BUFFSIZE must be in [1, 65535] MiB")
        # The shared manager cannot reconstruct cached custom-buffer contexts.
        # Use a fresh collective tag for each owner, including after its release.
        sequence = _CONTEXT_SEQUENCE.get(self.group_name, 0)
        _CONTEXT_SEQUENCE[self.group_name] = sequence + 1
        self._ctx_manager = CommContextManager(
            self.group_name, self.world_size, backend={"Ascend950": "channel"},
            commAlg="urma" if self.comm_alg == "urma" else "ub-mem",
            opName=f"bandwidth_test_torch_{self.comm_alg}_{sequence}",
            customCclBufferSize=2 * buffer_mb * 1024 * 1024)
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size
        if self.comm_alg == "mte" and self._ctx_manager.rank_num_per_server != self.world_size:
            raise ValueError("BandwidthTest requires all ranks in one UB memory domain")
        # Compile the wrapper before ACL graph capture starts.
        _bandwidth_test_op_builder.load()


def bandwidth_test(sym_buffer: BandwidthTestBuffer, x: torch.Tensor,
                   dstrank_id: torch.Tensor, world_size: int, max_bs: int,
                   mode: int = 0, comm_alg: str = "mte", aiv_num: int = -1
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exchange rows using a BandwidthTestBuffer created before graph capture.

    Args:
        sym_buffer: Context owner for the participating HCCL group.
        x: NPU float16 tensor [max_bs, hidden], hidden divisible by 16 and < 32768.
        dstrank_id: INT32 tensor [max_bs] on the same NPU, with destination IDs
            in [0, world_size). Device values are not checked during capture.
        world_size: Group size in [1, 1024], matching sym_buffer.
        max_bs: Positive row count shared by all ranks.
        mode: 0 exchanges and receives rows; 1 sends and waits, without copying outputs.
        comm_alg: "mte", "" or "urma"; must match the context owner.
        aiv_num: -1 selects platform defaults; otherwise a positive AIV count.

    Returns:
        FP16 y [max_bs * world_size, hidden] and INT32 receive_cnt [world_size].
        Only y[:receive_cnt.sum()] is valid in mode=0; mode=1 outputs are
        unspecified. Calls sharing a context must be submitted serially.
    """
    if world_size != sym_buffer.world_size:
        raise ValueError("world_size must match the context communication group")
    if (comm_alg or "mte") != sym_buffer.comm_alg:
        raise ValueError("comm_alg must match BandwidthTestBuffer")
    return torch.ops.cann_ops_transformer.bandwidth_test(
        sym_buffer.context, x, dstrank_id, sym_buffer.group_name,
        world_size, max_bs, mode, sym_buffer.ccl_buffer_size,
        comm_alg=comm_alg, aiv_num=aiv_num)
