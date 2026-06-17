# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import torch.distributed as dist
from typing import Callable

from .elastic_buffer_op import _elastic_buffer_ops
from cann_ops_transformer.ops.engram_embedding_utils import EngramEmbeddingUtils

BUFFER_ALIGNMENT = 2 * 1024 * 1024

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


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
            num_cpu_bytes: the CPU buffer size in bytes.
        """
        self.group = group
        rank_id = dist.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=True)
        
        torch._check(num_cpu_bytes >= 0,
                     lambda: f"num_cpu_bytes must be non-negative, got {num_cpu_bytes=}.")
        
        torch._check(num_cpu_bytes % BUFFER_ALIGNMENT == 0,
                     lambda: (f"num_cpu_bytes must be 2MB-aligned, "
                              f"got {num_cpu_bytes=} bytes which is not divisible by {BUFFER_ALIGNMENT} bytes."))
        
        self.num_cpu_bytes = num_cpu_bytes
        
        self.comm_context_tensor = None
        self.host_pinned_addr = 0
        self.hidden_size = 0
        self.num_engram_entries = 0
        self.engram_dtype = None
        self._write_flag = False
        
        utils = EngramEmbeddingUtils(self.group_name, self.num_cpu_bytes)
        self.comm_context_tensor = utils.create_context()
        self.host_pinned_addr = utils.host_ptr()
    
    def engram_write(self, storage: torch.Tensor) -> None:
        """
        Write data to the host pinned memory of ElasticBuffer.

        Arguments:
            storage: the CPU tensor to write, must be contiguous.

        Returns:
            None
        """
        # Check storage size does not exceed buffer capacity (most critical check)
        torch._check(storage.nbytes <= self.num_cpu_bytes,
                     lambda: f"Storage size {storage.nbytes} bytes exceeds buffer capacity {self.num_cpu_bytes} bytes.")
        
        # Check tensor device (must be CPU)
        torch._check(storage.is_cpu,
                     lambda: f"Storage tensor must be on CPU, got {storage.device=}.")
        
        # Check tensor contiguity
        torch._check(storage.is_contiguous(),
                     lambda: f"Storage tensor must be contiguous.")
        
        # Check dtype (only bf16/fp16/fp32 supported for Engram storage)
        supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        torch._check(storage.dtype in supported_dtypes,
                     lambda: (f"Engram storage dtype must be one of [bfloat16, float16, float32], "
                              f"but got {storage.dtype=}. "))
        
        # Check storage shape
        torch._check(len(storage.shape) == 2,
                     lambda: f"Storage tensor must be 2-dimensional, got {len(storage.shape)} dimensions.")
        
        torch._check(storage.shape[1] % 128 == 0,
                     lambda: f"Storage tensor second dimension must be 128-aligned, got {storage.shape[1]=}.")
        
        # Derive parameters from storage shape
        self.num_engram_entries = storage.shape[0]
        self.hidden_size = storage.shape[1]
        self.engram_dtype = storage.dtype
        
        self.__barrier(with_device_sync=True)
        
        if self.num_engram_entries > 0:
            _elastic_buffer_ops.npu_engram_write(self.host_pinned_addr, storage)
        
        self.__barrier(with_device_sync=True)
        
        self._write_flag = True
    
    def __barrier(self, with_device_sync: bool = False) -> None:
        """
        Internal method for cross-rank synchronization barrier.

        Arguments:
            with_device_sync: whether to synchronize device, defaults to False.

        Returns:
            None
        """
        _elastic_buffer_ops.npu_engram_barrier(self.group_name, with_device_sync)
    
    def destroy(self) -> None:
        """
        Destroy the ElasticBuffer and free host pinned memory.

        Returns:
            None
        """
        self.__barrier(with_device_sync=True)
        if self.host_pinned_addr != 0:
            _elastic_buffer_ops.npu_engram_destroy(self.host_pinned_addr)
            self.host_pinned_addr = 0
            self.comm_context_tensor = None
    
    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Fetch Engram data from remote ranks via RDMA.

        Arguments:
            indices: the indices of entries to fetch, must be on NPU with dtype=int32.

        Returns:
            wait_callable: a callable that returns the fetched tensor when invoked.
        """
        torch._check(self._write_flag,
                     lambda: "Must call engram_write at least once before engram_fetch")
        
        # Check tensor device (must be NPU)
        torch._check(indices.is_npu,
                     lambda: f"indices tensor must be on NPU, got {indices.device=}.")
        
        # Check tensor dimensions (must be 1-dimensional)
        torch._check(len(indices.shape) == 1,
                     lambda: f"indices tensor must be 1-dimensional, got {len(indices.shape)} dimensions.")
        
        # Check dtype (must be int32)
        torch._check(indices.dtype == torch.int32,
                     lambda: f"indices dtype must be torch.int32, but got {indices.dtype=}.")
        
        wait_callable = _elastic_buffer_ops.npu_engram_fetch(
            self.comm_context_tensor,
            indices,
            self.hidden_size,
            self.num_engram_entries,
            self.engram_dtype,
        )
        return wait_callable
    
    @staticmethod
    def get_engram_storage_size_hint(num_entries: int, hidden_size: int,
                                     dtype: torch.dtype = torch.bfloat16) -> int:
        """
        Get the minimum CPU buffer size required for Engram storage.
        The returned value is aligned to 2 MB.

        Arguments:
            num_entries: the number of entries in the Engram storage.
            hidden_size: the hidden dimension of each entry, must be 128-aligned.
            dtype: the data type, defaults to `torch.bfloat16`.

        Returns:
            num_cpu_bytes: the recommended CPU buffer size in bytes (2 MB-aligned).
        """
        torch._check(num_entries >= 0,
                     lambda: f"num_entries must be non-negative, got {num_entries=}.")
        
        torch._check(hidden_size >= 0,
                     lambda: f"hidden_size must be non-negative, got {hidden_size=}.")
        
        torch._check(hidden_size % 128 == 0,
                     lambda: f"hidden_size must be 128-aligned, got {hidden_size=}.")
        
        num_sf_packs = ceil_div(hidden_size, 32) if dtype.itemsize <= 1 else 0
        num_bytes_per_entry = align(hidden_size * dtype.itemsize + num_sf_packs * 4, 32)
        num_cpu_bytes = align(num_bytes_per_entry * num_entries, BUFFER_ALIGNMENT)
        return num_cpu_bytes