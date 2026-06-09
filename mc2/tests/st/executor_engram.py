import torch
import torch.distributed as dist
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_engram")
class Engram(BaseApi):
    """
    Engram算子测试：EngramWrite + EngramFetch
    参考：https://github.com/deepseek-ai/DeepEP/blob/main/tests/elastic/test_engram.py
    
    核心流程：
    1. 每个rank生成本地storage并写入buffer (engram_write)
    2. 使用all_gather构建全局storage用于验证
    3. 生成随机indices并fetch数据 (engram_fetch)
    4. 验证fetch结果与全局storage一致性
    
    注意：
    - EngramWrite是host runtime操作，无独立ACLNN接口
    - EngramFetch分为issue和wait两个阶段
    - 无小算子级联标杆，直接使用ElasticBuffer API
    """
    
    def __init__(self, task_result):
        super().__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.buffer = None
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        
        num_entries = input_data.kwargs['num_entries']
        hidden = input_data.kwargs['hidden']
        num_tokens = input_data.kwargs['num_tokens']
        num_qps = input_data.kwargs['num_qps']
        hybrid_mode = input_data.kwargs['hybrid_mode']
        
        if self.name == "cpu":
            # CPU标杆：模拟基础数据验证
            torch.manual_seed(42 + rank)
            global_storage = torch.randn(
                (world_size * num_entries, hidden),
                dtype=torch.bfloat16, device='cpu'
            )
            
            indices = torch.randint(
                0, world_size * num_entries,
                (num_tokens,), dtype=torch.int, device='cpu'
            )
            
            return global_storage[indices].clone()
        
        if self.name == "npu":
            # NPU测试：真实API调用
            device = f'npu:{rank}'
            
            # 1. 准备本地和全局storage
            torch.manual_seed(42 + rank)
            local_storage = torch.randn(
                (num_entries, hidden), dtype=torch.bfloat16, device=device
            )
            
            global_storage = torch.empty(
                (world_size * num_entries, hidden),
                dtype=torch.bfloat16, device=device
            )
            dist.all_gather_into_tensor(global_storage, local_storage)
            
            # 2. 构造ElasticBuffer
            self.buffer = self._create_buffer(
                group=input_data.kwargs['group'],
                num_entries=num_entries,
                hidden=hidden,
                num_tokens=num_tokens,
                num_qps=num_qps,
                hybrid_mode=hybrid_mode
            )
            
            # 3. engram_write
            self.buffer.engram_write(local_storage)
            
            # 4. engram_fetch (issue + wait)
            indices = torch.randint(
                0, world_size * num_entries,
                (num_tokens,), dtype=torch.int, device=device
            )
            hook = self.buffer.engram_fetch(indices)  # issue
            fetched = hook()                           # wait
            
            # 5. 验证正确性
            ref_fetched = global_storage[indices]
            assert torch.equal(ref_fetched, fetched), \
                f"Data mismatch: max diff {(ref_fetched - fetched).abs().max().item()}"
            
            return fetched
    
    def _create_buffer(self, group, num_entries, hidden, num_tokens, num_qps, hybrid_mode):
        """构造ElasticBuffer"""
        try:
            from torch_npu.contrib import ElasticBuffer
            
            # 计算内存需求
            num_gpu_bytes, num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
                num_entries, hidden, num_tokens, torch.bfloat16
            )
            
            # 构造buffer
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
        
        # 设置分布式group
        if self.device == 'npu' and dist.is_initialized():
            from torch.distributed.distributed_c10d import _get_default_group
            input_data.kwargs['group'] = _get_default_group()
    
    def cleanup(self):
        """清理资源"""
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None


@register("function_engram_write")
class EngramWrite(BaseApi):
    """
    仅测试EngramWrite操作
    验证host pinned段写入的正确性
    """
    
    def __init__(self, task_result):
        super().__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.buffer = None
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        
        storage = input_data.kwargs['storage']
        num_entries = input_data.kwargs['num_entries']
        hidden = input_data.kwargs['hidden']
        hybrid_mode = input_data.kwargs['hybrid_mode']
        copy_flag = input_data.kwargs.get('copy_flag', 1)
        
        if self.name == "cpu":
            return storage.cpu()
        
        if self.name == "npu":
            device = f'npu:{rank}'
            storage_npu = storage.to(device)
            
            # 构造buffer
            self.buffer = self._create_buffer(
                group=input_data.kwargs['group'],
                num_entries=num_entries,
                hidden=hidden,
                num_tokens=num_entries,
                num_qps=8,
                hybrid_mode=hybrid_mode
            )
            
            # 执行write
            if copy_flag:
                self.buffer.engram_write(storage_npu)
            
            return storage_npu
    
    def _create_buffer(self, group, num_entries, hidden, num_tokens, num_qps, hybrid_mode):
        try:
            from torch_npu.contrib import ElasticBuffer
            num_gpu_bytes, num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
                num_entries, hidden, num_tokens, torch.bfloat16
            )
            return ElasticBuffer(
                group=group,
                num_bytes=num_gpu_bytes + num_cpu_bytes,
                num_cpu_bytes=num_cpu_bytes,
                explicitly_destroy=True,
                num_allocated_qps=num_qps,
                allow_hybrid_mode=hybrid_mode
            )
        except ImportError:
            raise NotImplementedError("ElasticBuffer not available")
    
    def init_by_input_data(self, input_data: InputDataset):
        rank = self.dist_task_info.rank
        device = f'npu:{rank}'
        if 'storage' in input_data.kwargs:
            input_data.kwargs['storage'] = input_data.kwargs['storage'].to(device)
        if self.device == 'npu' and dist.is_initialized():
            from torch.distributed.distributed_c10d import _get_default_group
            input_data.kwargs['group'] = _get_default_group()
    
    def cleanup(self):
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None


@register("function_engram_fetch")
class EngramFetch(BaseApi):
    """
    仅测试EngramFetch操作
    前置条件：需要先执行engram_write
    """
    
    def __init__(self, task_result):
        super().__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.buffer = None
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        
        indices = input_data.kwargs['indices']
        num_entries = input_data.kwargs['num_entries']
        hidden = input_data.kwargs['hidden']
        num_tokens = input_data.kwargs['num_tokens']
        hybrid_mode = input_data.kwargs['hybrid_mode']
        
        if self.name == "cpu":
            torch.manual_seed(42)
            global_storage = torch.randn(
                (world_size * num_entries, hidden),
                dtype=torch.bfloat16, device='cpu'
            )
            return global_storage[indices.cpu()].clone()
        
        if self.name == "npu":
            device = f'npu:{rank}'
            
            # 准备数据
            torch.manual_seed(42 + rank)
            local_storage = torch.randn(
                (num_entries, hidden), dtype=torch.bfloat16, device=device
            )
            
            global_storage = torch.empty(
                (world_size * num_entries, hidden),
                dtype=torch.bfloat16, device=device
            )
            dist.all_gather_into_tensor(global_storage, local_storage)
            
            # 构造buffer并write
            self.buffer = self._create_buffer(
                group=input_data.kwargs['group'],
                num_entries=num_entries,
                hidden=hidden,
                num_tokens=num_tokens,
                num_qps=8,
                hybrid_mode=hybrid_mode
            )
            self.buffer.engram_write(local_storage)
            
            # fetch
            indices_npu = indices.to(device)
            hook = self.buffer.engram_fetch(indices_npu)
            fetched = hook()
            
            # 验证
            ref_fetched = global_storage[indices_npu]
            assert torch.equal(ref_fetched, fetched)
            
            return fetched
    
    def _create_buffer(self, group, num_entries, hidden, num_tokens, num_qps, hybrid_mode):
        try:
            from torch_npu.contrib import ElasticBuffer
            num_gpu_bytes, num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
                num_entries, hidden, num_tokens, torch.bfloat16
            )
            return ElasticBuffer(
                group=group,
                num_bytes=num_gpu_bytes + num_cpu_bytes,
                num_cpu_bytes=num_cpu_bytes,
                explicitly_destroy=True,
                num_allocated_qps=num_qps,
                allow_hybrid_mode=hybrid_mode
            )
        except ImportError:
            raise NotImplementedError("ElasticBuffer not available")
    
    def init_by_input_data(self, input_data: InputDataset):
        rank = self.dist_task_info.rank
        device = f'npu:{rank}'
        if 'indices' in input_data.kwargs:
            input_data.kwargs['indices'] = input_data.kwargs['indices'].to(device)
        if self.device == 'npu' and dist.is_initialized():
            from torch.distributed.distributed_c10d import _get_default_group
            input_data.kwargs['group'] = _get_default_group()
    
    def cleanup(self):
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None