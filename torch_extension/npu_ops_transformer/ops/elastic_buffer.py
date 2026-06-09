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
from typing import Callable, Tuple
# from .elastic_buffer_handle import op_module


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


class ElasticBuffer:
    def __init__(self,
                 group: torch.distributed.ProcessGroup,
                 num_bytes: int,
                 num_cpu_bytes: int):
        """
        初始化 ElasticBuffer。
        
        参数：
            group: torch.distributed.ProcessGroup - 分布式通信组
            num_bytes: int - 总 buffer 大小（字节）
            num_cpu_bytes: int - CPU buffer 大小（字节）
            num_allocated_qps: int - RDMA QP 数量
            allow_hybrid_mode: bool = True - 是否允许 hybrid mode
            explicitly_destory: bool = True - 是否需要显式调用 destroy()
        """
        self.group = group
        self.rank_id = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=True)
        self.num_bytes = num_bytes
        self.num_cpu_bytes = num_cpu_bytes
        
        # 导入 op_module
        from .elastic_buffer_op import op_module
        
        # 调用 C++ 初始化函数，分配 host pinned memory 并填充 handle
        self.handle = op_module.npu_elastic_buffer_init(
            self.group_name,
            self.rank_id,
            self.world_size,
            self.num_bytes,
            self.num_cpu_bytes
        )
    
    def engram_write(self, storage: torch.Tensor) -> None:
        """
        将数据写入 ElasticBuffer 的 host pinned memory。
        
        参数：
            storage: torch.Tensor - 要写入的数据（必须是连续的 CPU tensor）
            offset: int = 0 - 写入的起始偏移量（字节）
            
        返回：
            None
            
        注意：
            storage 必须是连续的 tensor，数据将被拷贝到 hostPinnedAddr + offset
        """
        self.barrier(use_comm_stream=False, with_device_sync=True)
        # 导入 op_module
        from .elastic_buffer_op import op_module
        # 调用 C++ write 函数
        op_module.npu_elastic_buffer_write(self.handle, storage)
        self.barrier(use_comm_stream=False, with_device_sync=True)
    
    def barrier(self, use_comm_stream: bool = True, with_device_sync: bool = False) -> None:
        """
        执行跨 rank 的同步屏障。
        
        参数：
            use_comm_stream: bool = True - 是否使用通信 stream（当前未使用）
            with_device_sync: bool = False - 是否进行设备同步（底层调用 aclrtSynchronizeDevice）
            
        返回：
            None
        """
        # 导入 op_module
        from .elastic_buffer_op import op_module
        
        # 调用封装的 HCCL HcclBarrier（底层会调用 aclrtSynchronizeDevice）
        op_module.npu_elastic_buffer_barrier(self.group_name, with_device_sync)
    
    def destroy(self) -> None:
        """
        显式销毁 ElasticBuffer 资源。
        
        参数：
            无
            
        返回：
            None
            
        注意：
            只有在初始化时 explicitly_destory=True 才需要手动调用此方法
        """
        if hasattr(self, 'handle') and self.handle is not None:
            from .elastic_buffer_op import op_module
            op_module.npu_elastic_buffer_destroy(self.handle)
            self.handle = None
    
    def engram_fetch(self, indices: torch.Tensor, hidden_size: int,
                     dtype: torch.dtype = torch.bfloat16) -> Callable[[], torch.Tensor]:
        """
        从远程 rank 通过 RDMA 获取 Engram 数据。
        
        参数：
            indices: torch.Tensor - 要获取的条目索引
            hidden_size: int - 隐藏层维度大小
            dtype: torch.dtype = torch.bfloat16 - 输出数据类型
            
        返回：
            torch.Tensor - 获取的 tensor，形状为 [num_tokens, hidden_size]
        """
        # 导入 op_module (已经通过 elastic_buffer_op 加载)
        from .elastic_buffer_op import op_module
        
        # 使用真实的 handle（包含 hostPinnedAddr 等）
        dtype_val = 11  # bfloat16 enum value
        wait_callable = op_module.npu_engram_fetch(
            self.handle, indices, hidden_size, dtype_val)
        
        # 直接返回 C++ 构造的 callable
        # 用户调用 wait_callable() 触发 wait 阶段
        return wait_callable 
    
    @staticmethod
    def get_engram_storage_size_hint(num_entries: int,
                                      hidden: int,
                                      num_max_tokens: int,
                                      dtype: torch.dtype = torch.bfloat16) -> Tuple[int, int]:
        """
        计算 Engram 存储所需的 buffer 大小建议。
        
        参数：
            num_entries: int - Engram 条目数量
            hidden: int - 隐藏层维度
            num_max_tokens: int - 每卡最大 token 数
            dtype: torch.dtype = torch.bfloat16 - 数据类型
            
        返回：
            Tuple[int, int] - (GPU buffer 大小, CPU buffer 大小)，均为 2 MB 对齐
        """
        buffer_alignment = 2 * 1024 * 1024
        num_sf_packs = ceil_div(hidden, 32) if dtype.itemsize <= 1 else 0
        num_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)
        num_npu_bytes = align(num_bytes_per_entry * num_max_tokens, buffer_alignment)
        num_cpu_bytes = align(num_bytes_per_entry * num_entries, buffer_alignment)
        return num_npu_bytes, num_cpu_bytes