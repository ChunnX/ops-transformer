import torch
import torch.distributed as dist
try:
   import torch_npu
except ImportError:
   pass

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
{{ADDITIONAL_IMPORTS}}


@register("function_{{ACLNN_API_NAME}}")
class {{OP_CLASS_NAME}}(BaseApi):
    def __init__(self, task_result):
        super({{OP_CLASS_NAME}}, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.rank = self.dist_task_info.rank
        self.world_size = self.dist_task_info.world_size

    def get_cpp_func_signature_type(self):
        # 通过文档寻找aclnnxxxgetworkspacesize的函数定义
        return "xxx"
    def manual_all_to_all(self, x_splits_from_ranks, x_splits_to_ranks):
        for target_rank in range(self.world_size):
            if target_rank == self.rank:
                x_splits_from_ranks[self.rank].copy_(x_splits_to_ranks[target_rank])
                for src_rank in range(self.world_size):
                    if src_rank != self.rank:
                        dist.recv(
                            tensor=x_splits_from_ranks[src_rank],
                            src=src_rank,
                            tag=src_rank * 1000 + self.rank
                        )
            else:
                dist.send(
                    tensor=x_splits_to_ranks[target_rank],
                    dst=target_rank,
                    tag=self.rank * 1000 + target_rank
                )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        world_size = self.dist_task_info.world_size
        {{INPUT_UNPACK_FROM_KWARGS}}

        if self.name == "cpu":
            '''
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ CPU  真值 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            '''
            {{CPU_TRUTH_LOGIC}}
            '''
            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ CPU  真值 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            '''

        if self.dist_task_info.is_bm:
            '''
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ NPU 小算子级联标杆 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            '''
            {{NPU_BM_LOGIC}}
            '''
            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ NPU 小算子级联标杆 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            '''
        else:
            pass

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == 'pyaclnn' and dist.is_available():
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            input_data.kwargs['group'] = default_pg


'''
    =================================================================================
    ==============================ACLNN输入特殊处理===================================
    =================================================================================
'''

@register("pyaclnn_{{ACLNN_API_NAME}}")
class Aclnn{{OP_CLASS_NAME}}(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super(Aclnn{{OP_CLASS_NAME}}, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def init_by_input_data(self, input_data: InputDataset):
        rank_id = self.dist_task_info.rank
        self.group = input_data.kwargs['group']
        input_data.kwargs['group'] = self.group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)

        {{ACLNN_INPUT_PREPROCESSING}}

        input_args, output_packages = super().init_by_input_data(input_data)

        {{ACLNN_ARGS_POSTPROCESSING}}

        {{ACLNN_NULL_ARRAY_OPTIONAL}}

        return input_args, output_packages