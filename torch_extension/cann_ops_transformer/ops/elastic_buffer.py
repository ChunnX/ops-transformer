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
        self.group = group
        rank_id = dist.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=True)
        
        # Create C++ ElasticBuffer runtime (C++ handles all validation)
        self.runtime = _elastic_buffer_ops.ElasticBuffer(self.group_name, num_cpu_bytes)
    
    def engram_write(self, storage: torch.Tensor) -> None:
        """
        Write data to the host pinned memory of ElasticBuffer.
        
        Arguments:
            storage: the CPU tensor to write (must be 2D, contiguous, dtype=bf16/fp16/fp32).
        
        Returns:
            None
        
        Note: barrier(with_device_sync=True) is called before and after write internally.
        """
        self.runtime.engram_write(storage)
    
    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Fetch Engram data from remote ranks via RDMA.

        Arguments:
            indices: the indices of entries to fetch (must be 1D NPU tensor with dtype=int32).

        Returns:
            wait_callable: a callable that returns the fetched tensor when invoked.
        """
        return self.runtime.engram_fetch(indices)
    
    def destroy(self) -> None:
        """
        Destroy the ElasticBuffer and free host pinned memory.

        Returns:
            None
        """
        self.runtime.destroy()
    
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
        return _elastic_buffer_ops.ElasticBuffer.get_engram_storage_size_hint(num_entries, hidden_size, dtype)