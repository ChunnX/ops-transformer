import torch
import torch.distributed as dist
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_engram")
class Engram(BaseApi):
    """
    Engram算子 - 包含Write和Fetch两个阶段
    
    Write阶段：将本rank的表分片写入host pinned共享段
    Fetch阶段：通过RDMA从远程rank抓取数据到本rank
    
    参考：https://github.com/deepseek-ai/DeepEP/blob/main/tests/elastic/test_engram.py
    """
    
    def __init__(self, task_result):
        super().__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.buffer = None
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        
        phase = input_data.kwargs.get('phase', 'write_fetch')
        num_entries = input_data.kwargs['num_entries']
        hidden = input_data.kwargs['hidden']
        num_tokens = input_data.kwargs['num_tokens']
        num_qps = input_data.kwargs['num_qps']
        hybrid_mode = input_data.kwargs['hybrid_mode']
        
        if self.name == "cpu":
            # CPU标杆：基础数据验证
            torch.manual_seed(42 + rank)
            
            if phase == 'write':
                # Write阶段：验证storage写入
                storage = torch.randn((num_entries, hidden), dtype=torch.bfloat16, device='cpu')
                return storage
            
            elif phase == 'fetch':
                # Fetch阶段：模拟fetch结果
                global_storage = torch.randn(
                    (world_size * num_entries, hidden),
                    dtype=torch.bfloat16, device='cpu'
                )
                indices = torch.randint(0, world_size * num_entries, (num_tokens,), dtype=torch.int, device='cpu')
                return global_storage[indices]
            
            else:  # write_fetch
                # 完整流程
                global_storage = torch.randn(
                    (world_size * num_entries, hidden),
                    dtype=torch.bfloat16, device='cpu'
                )
                indices = torch.randint(0, world_size * num_entries, (num_tokens,), dtype=torch.int, device='cpu')
                return global_storage[indices]
        
        if self.name == "npu":
            # NPU测试：真实API调用
            device = f'npu:{rank}'
            
            # 1. 准备数据
            torch.manual_seed(42 + rank)
            local_storage = torch.randn(
                (num_entries, hidden), dtype=torch.bfloat16, device=device
            )
            
            global_storage = torch.empty(
                (world_size * num_entries, hidden),
                dtype=torch.bfloat16, device=device
            )
            dist.all_gather_into_tensor(global_storage, local_storage)
            
            # 2. 构造buffer
            self.buffer = self._create_buffer(
                group=input_data.kwargs['group'],
                num_entries=num_entries,
                hidden=hidden,
                num_tokens=num_tokens,
                num_qps=num_qps,
                hybrid_mode=hybrid_mode
            )
            
            # 3. 根据phase执行不同阶段
            if phase == 'write':
                # 仅Write阶段
                self.buffer.engram_write(local_storage)
                return local_storage
            
            elif phase == 'fetch':
                # 仅Fetch阶段（需要先write）
                self.buffer.engram_write(local_storage)
                indices = input_data.kwargs.get('indices')
                if indices is None:
                    indices = torch.randint(0, world_size * num_entries, (num_tokens,), dtype=torch.int, device=device)
                
                hook = self.buffer.engram_fetch(indices)
                fetched = hook()
                return fetched
            
            else:  # write_fetch
                # 完整流程
                self.buffer.engram_write(local_storage)
                
                indices = torch.randint(0, world_size * num_entries, (num_tokens,), dtype=torch.int, device=device)
                hook = self.buffer.engram_fetch(indices)
                fetched = hook()
                
                # 验证正确性
                ref_fetched = global_storage[indices]
                assert torch.equal(ref_fetched, fetched), \
                    f"Data mismatch: max diff {(ref_fetched - fetched).abs().max().item()}"
                
                return fetched
    
    def _create_buffer(self, group, num_entries, hidden, num_tokens, num_qps, hybrid_mode):
        """构造ElasticBuffer"""
        try:
            from torch_npu.contrib import ElasticBuffer
            
            num_gpu_bytes, num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
                num_entries, hidden, num_tokens, torch.bfloat16
            )
            
            buffer = ElasticBuffer(
                group=group,
                num_bytes=num_gpu_bytes + num_cpu_bytes,
                num_cpu_bytes=num_cpu_bytes,
                explicitly_destroy=True,
                num_allocated_qps=num_qps,
                allow_hybrid_mode=hybrid_mode,
                allow_multiple_reduction=False
            )
            return buffer
            
        except ImportError:
            raise NotImplementedError(
                "ElasticBuffer not available. Waiting for torch_npu implementation."
            )
    
    def init_by_input_data(self, input_data: InputDataset):
        rank = self.dist_task_info.rank
        device = f'npu:{rank}'
        
        # 移动tensor到device
        if 'indices' in input_data.kwargs and input_data.kwargs['indices'] is not None:
            input_data.kwargs['indices'] = input_data.kwargs['indices'].to(device)
        
        # 设置分布式group
        if self.device == 'npu' and dist.is_initialized():
            from torch.distributed.distributed_c10d import _get_default_group
            input_data.kwargs['group'] = _get_default_group()
    
    def cleanup(self):
        """清理资源"""
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None