# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Callable
import os

import torch
import torch_npu
import torch.distributed as dist

from cann_ops_transformer.op_builder.builder import OpBuilder


class ElasticBufferOpBuilder(OpBuilder):
    """OpBuilder for ElasticBuffer operations"""

    def __init__(self):
        super(ElasticBufferOpBuilder, self).__init__("npu_elastic_buffer")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/elastic_buffer.cpp']

    def schema(self):
        """PyTorch operator signature."""
        return None

    def register_meta(self):
        """Meta implementation (optional for JIT compiled ops)"""
        pass

    def extra_ldflags(self):
        """Extra link flags for HCCL and ACL libraries."""
        flags = super().extra_ldflags()
        flags.append('-L' + os.path.join(self._cann_path, 'lib64'))
        flags.append('-lhcomm')
        flags.append('-lascendcl')
        return flags

    def include_paths(self):
        """Override include paths to ensure CANN headers are prioritized."""
        return [
            os.path.join(self._cann_path, 'include'),
            os.path.join(self._torch_npu_path, 'include'),
            os.path.join(self._torch_npu_path, 'include/third_party/hccl/inc'),
            os.path.join(self._torch_npu_path, 'include/third_party/acl/inc'),
            os.path.join(self._package_path, 'common/inc')
        ]


_elastic_buffer_ops = ElasticBufferOpBuilder().load()


class ElasticBuffer:
    """
    ElasticBuffer for distributed Engram storage management.
    """

    def __init__(self,
                 group: torch.distributed.ProcessGroup,
                 num_cpu_bytes: int):
        """
        Initialize the ElasticBuffer.

        Arguments:
            group: the distributed process group.
            num_cpu_bytes: the CPU buffer size in bytes (must be 2MB-aligned).
"""
        buffer_alignment = 2 * 1024 * 1024
        torch._check((group is not None),
                     lambda: (f"group must not be None."))
        torch._check((num_cpu_bytes >= 0),
                     lambda: (f"num_cpu_bytes must be non-negative, got {num_cpu_bytes=}."))
        torch._check((num_cpu_bytes % buffer_alignment == 0),
                     lambda: (f"num_cpu_bytes must be 2MB-aligned, got {num_cpu_bytes=}, "
                              f"which is not divisible by {buffer_alignment=}."))
        
        self.group = group
        rank_id = dist.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=True)

        # Create ElasticBuffer
        self.runtime = _elastic_buffer_ops.ElasticBuffer(self.group_name, num_cpu_bytes)

    @staticmethod
    def get_engram_storage_size_hint(num_entries: int, hidden_size: int,
                                     dtype: torch.dtype = torch.bfloat16) -> int:
        """
        Get the minimum CPU buffer size required for Engram storage.
        The returned value is aligned to 2 MB.

        Arguments:
            num_entries: the number of entries in the Engram storage (must be non-negative).
            hidden_size: the hidden dimension of each entry (must be 128-aligned and non-negative).
            dtype: the data type, defaults to `torch.bfloat16`.

        Returns:
            num_cpu_bytes: the recommended CPU buffer size in bytes (2 MB-aligned).
        """
        torch._check(num_entries >= 0,
                     lambda: f"num_entries must be non-negative, got {num_entries}")
        torch._check(hidden_size >= 0,
                     lambda: f"hidden_size must be non-negative, got {hidden_size}")
        torch._check(hidden_size % 128 == 0,
                     lambda: f"hidden_size must be 128-aligned, got {hidden_size}")
        torch._check(dtype in (torch.bfloat16, torch.float16, torch.float32),
                     lambda: f"dtype must be bfloat16/float16/float32, got {dtype}")
        
        return _elastic_buffer_ops.ElasticBuffer.get_engram_storage_size_hint(num_entries, hidden_size, dtype)

    def engram_write(self, storage: torch.Tensor) -> None:
        """
        Write data to the host pinned memory of ElasticBuffer.

        Arguments:
            storage: the CPU tensor to write (must be 2D, contiguous, dtype=bf16/fp16/fp32).

        Returns:
            None

        Note: barrier(with_device_sync=True) is called before and after write internally.
        """
        torch._check(storage.is_cpu, 
                     lambda: f"storage must be on CPU, got device: {storage.device}")
        torch._check(storage.dim() == 2,
                     lambda: f"storage must be 2D, got dimensions: {storage.dim()}")
        torch._check(storage.is_contiguous(),
                     lambda: "storage must be contiguous")
        torch._check(storage.dtype in (torch.bfloat16, torch.float16, torch.float32),
                     lambda: f"storage dtype must be bfloat16/float16/float32, got: {storage.dtype}")
        torch._check(storage.size(1) % 128 == 0,
                     lambda: f"storage second dimension must be 128-aligned, got: {storage.size(1)}")
        
        self.runtime.engram_write(storage)

    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Fetch Engram data from remote ranks via RDMA.

        Arguments:
            indices: the indices of entries to fetch (must be 1D NPU tensor with dtype=int32).

        Returns:
            wait_callable: a callable that returns the fetched tensor when invoked.
        """
        torch._check(indices.device.type == torch.device('npu').type,
                     lambda: f"indices must be on NPU, got device: {indices.device}")
        torch._check(indices.dim() == 1,
                     lambda: f"indices must be 1D, got dimensions: {indices.dim()}")
        torch._check(indices.dtype == torch.int32,
                     lambda: f"indices dtype must be int32, got: {indices.dtype}")
        
        return self.runtime.engram_fetch(indices)

    def destroy(self) -> None:
        """
        Destroy the ElasticBuffer and free host pinned memory.

        Returns:
            None
        """
        self.runtime.destroy()