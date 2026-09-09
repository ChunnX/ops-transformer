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
            'str comm_alg="", int aiv_num=-1, int batch_tokens=0, '
            'bool shared_jetty=False, int shared_jetty_num=1) -> (Tensor, Tensor)'
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def bandwidth_test_meta(context, x, dstrank_id, group, world_size,
                                max_bs, test_mode, ccl_buffer_size, comm_alg="", aiv_num=-1,
                                batch_tokens=0, shared_jetty=False, shared_jetty_num=1):
            _check_inputs(context, x, dstrank_id, group, world_size, max_bs,
                          test_mode, ccl_buffer_size, comm_alg, aiv_num, batch_tokens, shared_jetty, shared_jetty_num)
            return (x.new_empty((max_bs * world_size, x.size(1))),
                    x.new_empty((world_size,), dtype=torch.int32))


def _check_inputs(context, x, dstrank_id, group, world_size, max_bs,
                  test_mode, ccl_buffer_size, comm_alg, aiv_num, batch_tokens, shared_jetty=False, shared_jetty_num=1):
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
    torch._check(test_mode in (0, 1, 2, 3, 4), lambda: "test_mode must be in [0, 4]")
    torch._check(comm_alg in ("", "mte", "urma"), lambda: "comm_alg must be mte or urma")
    torch._check(test_mode not in (2, 3, 4) or comm_alg == "urma",
                 lambda: "mode 2/3/4 requires comm_alg=urma")
    torch._check(aiv_num == -1 or aiv_num > 0, lambda: "aiv_num must be -1 or positive")
    torch._check(batch_tokens >= 0, lambda: "batch_tokens must be 0 (auto) or positive")
    torch._check(comm_alg == "urma" or batch_tokens == 0,
                 lambda: "batch_tokens is only supported for urma")
    torch._check(not shared_jetty or comm_alg == "urma", lambda: "shared_jetty requires urma")
    torch._check(test_mode not in (3, 4) or shared_jetty,
                 lambda: "mode 3/4 requires shared_jetty=True")
    torch._check(1 <= shared_jetty_num <= 56, lambda: "shared_jetty_num must be in [1, 56]")
    torch._check(not shared_jetty or aiv_num == -1 or shared_jetty_num <= aiv_num,
                 lambda: "shared_jetty_num must not exceed aiv_num")
    torch._check(test_mode != 3 or (world_size > 1 and max_bs >= world_size),
                 lambda: "mode=3 requires world_size > 1 and max_bs >= world_size")
    torch._check(test_mode != 4 or (world_size > 1 and shared_jetty_num == world_size - 1),
                 lambda: "mode=4 requires world_size > 1 and shared_jetty_num=world_size-1")
    # Kernel divides the data window in two and reserves one max_bs block per half.
    row_bytes = ((x.size(1) * 2 + 511) // 512) * 512
    required_bytes = 1024 * 1024 + 2 * (world_size + 1) * max_bs * row_bytes
    if test_mode == 2:
        required_bytes = 1024 * 1024 + 2 * max_bs * x.size(1) * 2
    elif test_mode == 3:
        source_bytes = max_bs * x.size(1) * 2
        peer_bytes = (max_bs // world_size) * x.size(1) * 2
        required_bytes = 1024 * 1024 + source_bytes + world_size * shared_jetty_num * peer_bytes
    elif test_mode == 4:
        required_bytes = 1024 * 1024 + (world_size + 1) * max_bs * x.size(1) * 2
    torch._check(ccl_buffer_size >= required_bytes,
                 lambda: "HCCL buffer too small; increase HCCL_BUFFSIZE before creating the group")


_bandwidth_test_op_builder = BandwidthTestOpBuilder()
_bandwidth_test_op_builder._ensure_initialized()


@impl(get_as_library(), "bandwidth_test", "PrivateUse1")
def _bandwidth_test(context, x, dstrank_id, group, world_size, max_bs, test_mode,
                    ccl_buffer_size, comm_alg="", aiv_num=-1, batch_tokens=0, shared_jetty=False, shared_jetty_num=1):
    _check_inputs(context, x, dstrank_id, group, world_size, max_bs,
                  test_mode, ccl_buffer_size, comm_alg, aiv_num, batch_tokens, shared_jetty, shared_jetty_num)
    return _bandwidth_test_op_builder.load().bandwidth_test(
        context, x, dstrank_id, group, world_size, max_bs, test_mode,
        ccl_buffer_size, comm_alg, aiv_num, batch_tokens, shared_jetty, shared_jetty_num)


class BandwidthTestBuffer:
    """Own the Torch context for an initialized HCCL ProcessGroup.

    All group ranks must construct this object on their current Ascend950 NPU,
    outside torch.compile. MTE requires one UB memory domain; URMA requires
    UB_CTP links in the ordinary context. The independent shared jetty builder
    supports UB_CTP and cross-supernode UB_RTP (UBG) with the integration package.
    Set HCCL_BUFFSIZE before group creation and retain this object and the group
    until all operations and graph replays have completed.
    """

    def __init__(self, group, comm_alg="", shared_jetty=False, shared_jetty_num=1,
                 shared_jetty_comm_level=0):
        if comm_alg not in ("", "mte", "urma"):
            raise ValueError("comm_alg must be mte or urma")
        self.comm_alg = comm_alg or "mte"
        if shared_jetty and self.comm_alg != "urma":
            raise ValueError("shared_jetty requires comm_alg=urma")
        if not 1 <= shared_jetty_num <= 56:
            raise ValueError("shared_jetty_num must be in [1, 56]")
        if shared_jetty and shared_jetty_comm_level not in (0, 1, 2):
            raise ValueError("shared_jetty_comm_level must be 0, 1 or 2")
        self.shared_jetty = shared_jetty
        self.shared_jetty_num = shared_jetty_num
        self.shared_jetty_comm_level = shared_jetty_comm_level
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
        # Use a fresh HCCL resource tag for each owner, including after its release.
        sequence = _CONTEXT_SEQUENCE.get(self.group_name, 0)
        _CONTEXT_SEQUENCE[self.group_name] = sequence + 1
        if self.shared_jetty:
            from .shared_jetty_context import SharedJettyContextManager
            # Load the operator before registering any communicator-owned shared resources.
            _bandwidth_test_op_builder.load()
            self._ctx_manager = SharedJettyContextManager(
                group, self.group_name, self.world_size, shared_jetty_num, shared_jetty_comm_level,
                2 * buffer_mb * 1024 * 1024, sequence)
            self.context = self._ctx_manager.create_context()
            self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size
            return
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
                   mode: int = 0, comm_alg: str = "", aiv_num: int = -1, batch_tokens: int = 0,
                   shared_jetty: bool = False, shared_jetty_num: int = 1
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exchange rows using a BandwidthTestBuffer created before graph capture.

    Args:
        sym_buffer: Context owner for the participating HCCL group.
        x: NPU float16 tensor [max_bs, hidden], hidden divisible by 16 and < 32768.
        dstrank_id: INT32 tensor [max_bs] on the same NPU, with destination IDs
            in [0, world_size). Device values are not checked during capture.
        world_size: Group size in [1, 1024], matching sym_buffer.
        max_bs: Positive row count shared by all ranks.
        mode: 0 exchanges and receives rows; 1 sends and waits; 2 keeps the original Win stream;
            shared mode 3 makes every Jetty send floor(BS/world_size) rows to every remote rank;
            shared mode 4 creates one Jetty per remote rank and sends all BS rows.
        comm_alg: "mte", "" or "urma"; must match the context owner.
        aiv_num: -1 selects platform defaults; otherwise a positive AIV count.
        batch_tokens: URMA tokens per commit; mode 3 ignores it and uses the largest feasible batch.
            In mode 4, 0 selects BS and the result is capped by BS and the tiling maximum.
        shared_jetty: Enable the URMA shared jetty kernel and context.
        shared_jetty_num: Number of shared jettys and sending AIVs, in [1, min(aiv_num, 56)].

    Returns:
        FP16 y [max_bs * world_size, hidden] and INT32 receive_cnt [world_size].
        Only y[:receive_cnt.sum()] is valid in mode=0; mode=1 and mode=2/3/4 outputs are
        unspecified. Calls sharing a context must be submitted serially.
    """
    if world_size != sym_buffer.world_size:
        raise ValueError("world_size must match the context communication group")
    if (comm_alg or "mte") != sym_buffer.comm_alg:
        raise ValueError("comm_alg must match BandwidthTestBuffer")
    if shared_jetty != sym_buffer.shared_jetty or shared_jetty_num != sym_buffer.shared_jetty_num:
        raise ValueError("shared jetty settings must match BandwidthTestBuffer")
    return torch.ops.cann_ops_transformer.bandwidth_test(
        sym_buffer.context, x, dstrank_id, sym_buffer.group_name,
        world_size, max_bs, mode, sym_buffer.ccl_buffer_size,
        comm_alg=comm_alg, aiv_num=aiv_num, batch_tokens=batch_tokens,
        shared_jetty=shared_jetty, shared_jetty_num=shared_jetty_num)
