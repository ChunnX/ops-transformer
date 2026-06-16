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
        self.rank_id = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=True)
        self.num_cpu_bytes = num_cpu_bytes
        
        self.comm_context_tensor = None
        self.host_pinned_addr = 0
        self.hidden_size = 0
        self.num_engram_entries = 0
        self.engram_dtype = None  # Engram storage dtype, set in engram_write
        
        result = _elastic_buffer_ops.npu_elastic_buffer_init(
            self.group_name,
            self.rank_id,
            self.world_size,
            self.num_cpu_bytes
        )
        self.comm_context_tensor = result['commContext']
        self.host_pinned_addr = result['hostPinnedAddr']
    
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
                     lambda: (f"Storage size exceeds CPU buffer capacity, "
                              f"got {storage.nbytes=} bytes ({storage.nbytes / (1024**2):.2f} MB) "
                              f"but buffer capacity is {self.num_cpu_bytes=} bytes ({self.num_cpu_bytes / (1024**2):.2f} MB). "
                                ))
        
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
        
        # Derive parameters from storage shape
        self.num_engram_entries = storage.shape[0]
        self.hidden_size = storage.shape[1]
        self.engram_dtype = storage.dtype
        
        self.__barrier(with_device_sync=True)
        
        _elastic_buffer_ops.npu_elastic_buffer_write(self.host_pinned_addr, storage)
        
        self.__barrier(with_device_sync=True)
    
    def __barrier(self, with_device_sync: bool = False) -> None:
        """
        Internal method for cross-rank synchronization barrier.

        Arguments:
            with_device_sync: whether to synchronize device, defaults to False.

        Returns:
            None
        """
        _elastic_buffer_ops.npu_elastic_buffer_barrier(self.group_name, with_device_sync)
    
    def destroy(self) -> None:
        """
        Destroy the ElasticBuffer and free host pinned memory.

        Returns:
            None
        """
        self.__barrier(with_device_sync=True)
        if self.host_pinned_addr != 0:
            _elastic_buffer_ops.npu_elastic_buffer_destroy(self.host_pinned_addr)
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
        if self.hidden_size == 0 or self.num_engram_entries == 0 or self.engram_dtype is None:
            raise RuntimeError("Must call engram_write at least once before engram_fetch")
        
        # Check dtype (must be int32)
        torch._check(indices.dtype == torch.int32,
                     lambda: (f"indices dtype must be torch.int32, but got {indices.dtype=}. "))
        
        wait_callable = _elastic_buffer_ops.npu_engram_fetch(
            self.comm_context_tensor,
            indices,
            self.hidden_size,
            self.num_engram_entries,
            self.engram_dtype  # 直接传递torch.dtype，无需数字映射
        )
        return wait_callable
    
    @staticmethod
    def get_engram_storage_size_hint(num_entries: int, hidden: int,
                                     dtype: torch.dtype = torch.bfloat16) -> int:
        """
        Get the minimum CPU buffer size required for Engram storage.
        The returned value is aligned to 2 MB.

        Arguments:
            num_entries: the number of entries in the Engram storage.
            hidden: the hidden dimension of each entry.
            dtype: the data type, defaults to `torch.bfloat16`.

        Returns:
            num_cpu_bytes: the recommended CPU buffer size in bytes (2 MB-aligned).
        """
        buffer_alignment = 2 * 1024 * 1024
        num_sf_packs = ceil_div(hidden, 32) if dtype.itemsize <= 1 else 0
        num_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)
        num_cpu_bytes = align(num_bytes_per_entry * num_entries, buffer_alignment)
        return num_cpu_bytes