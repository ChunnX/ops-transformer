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


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


class ElasticBuffer:
    """
    ElasticBuffer - 分布式 Engram 存储管理类
    
    成员变量：
        comm_context_tensor: CommContext tensor（用于 ACLNN API）
        host_pinned_addr: host pinned memory 地址（整数）
        hidden_size: 隐藏层维度（从 engram_write 自动推导）
        num_engram_entries: Engram 条目数（从 engram_write 自动推导）
    """
    
    def __init__(self,
                 group: torch.distributed.ProcessGroup,
                 num_cpu_bytes: int):
        """
        初始化 ElasticBuffer。
        
        参数：
            group: torch.distributed.ProcessGroup - 分布式通信组
            num_cpu_bytes: int - CPU buffer 大小（字节）
        """
        self.group = group
        self.rank_id = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=True)
        self.num_cpu_bytes = num_cpu_bytes
        
        # 成员变量（从 C++ Init 返回）
        self.comm_context_tensor = None
        self.host_pinned_addr = 0
        
        # 成员变量（从 engram_write 自动推导）
        self.hidden_size = 0
        self.num_engram_entries = 0
        
        # 调用 C++ 初始化函数
        from .elastic_buffer_op import op_module
        
        result = op_module.npu_elastic_buffer_init(
            self.group_name,
            self.rank_id,
            self.world_size,
            self.num_cpu_bytes
        )
        
        # 从 dict 中提取返回值
        self.comm_context_tensor = result['commContext']
        self.host_pinned_addr = result['hostPinnedAddr']
    
    def engram_write(self, storage: torch.Tensor) -> None:
        """
        将数据写入 ElasticBuffer 的 host pinned memory。
        
        参数：
            storage: torch.Tensor - 要写入的数据（必须是连续的 CPU tensor）
            
        返回：
            None
            
        注意：
            - storage 必须是连续的 CPU tensor
            - 从 storage 自动推导 hiddenSize 和 numEngramEntries
        """
        # 验证 storage
        if not storage.is_cpu:
            raise ValueError("Storage tensor must be on CPU")
        if not storage.is_contiguous():
            raise ValueError("Storage tensor must be contiguous")
        
        # 自动推导参数
        self.num_engram_entries = storage.shape[0]
        self.hidden_size = storage.shape[1]
        
        # 前后 barrier
        self.__barrier(with_device_sync=True)
        
        # 调用 C++ write（只传地址）
        from .elastic_buffer_op import op_module
        op_module.npu_elastic_buffer_write(self.host_pinned_addr, storage)
        
        self.__barrier(with_device_sync=True)
    
    def __barrier(self, with_device_sync: bool = False) -> None:
        """
        内部方法：执行跨 rank 的同步屏障。
        
        参数：
            with_device_sync: bool = False - 是否进行设备同步
            
        返回：
            None
            
        注意：
            此方法仅供内部使用，请勿在类外部调用。
        """
        from .elastic_buffer_op import op_module
        op_module.npu_elastic_buffer_barrier(self.group_name, with_device_sync)
    
    def destroy(self) -> None:
        """
        显式销毁 ElasticBuffer 资源。
        
        返回：
            None
        """
        if self.host_pinned_addr != 0:
            from .elastic_buffer_op import op_module
            op_module.npu_elastic_buffer_destroy(self.host_pinned_addr)
            self.host_pinned_addr = 0
            self.comm_context_tensor = None
    
    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        从远程 rank 通过 RDMA 获取 Engram 数据。
        
        参数：
            indices: torch.Tensor - 要获取的条目索引
        
        返回：
            Callable[[], torch.Tensor] - callable，调用后返回 fetched tensor
        
        注意：
            - 必须先调用 engram_write 推导参数
            - indices 必须在 NPU 上，dtype=int32
        """
        if self.hidden_size == 0 or self.num_engram_entries == 0:
            raise RuntimeError("hidden_size and num_engram_entries not set. Call engram_write first.")
        
        from .elastic_buffer_op import op_module
        
        dtype_val = 11  # bfloat16
        
        wait_callable = op_module.npu_engram_fetch(
            self.comm_context_tensor,
            indices,
            self.hidden_size,
            self.num_engram_entries,
            dtype_val
        )
        
        return wait_callable
    
    @staticmethod
    def get_engram_storage_size_hint(num_entries: int,
                                      hidden: int,
                                      dtype: torch.dtype = torch.bfloat16) -> int:
        """
        计算 Engram 存储所需的 CPU buffer 大小建议。
        
        参数：
            num_entries: int - Engram 条目数量
            hidden: int - 隐藏层维度
            dtype: torch.dtype = torch.bfloat16 - 数据类型
            
        返回：
            int - CPU buffer 大小（字节），2 MB 对齐
        """
        buffer_alignment = 2 * 1024 * 1024  # 2 MB
        
        # FP8 需要额外的 scale factor packs
        if dtype.itemsize <= 1:  # FP8
            num_sf_packs = ceil_div(hidden, 32)
        else:  # BF16, FP16, FP32
            num_sf_packs = 0
        num_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)
        num_cpu_bytes = align(num_bytes_per_entry * num_entries, buffer_alignment)
        
        return num_cpu_bytes