import torch
import torch.distributed as dist
try:
   import torch_npu
except ImportError:
   pass

import ctypes
import ast
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
{{ADDITIONAL_IMPORTS}}


DTYPE_MAP = {
    'fp16': torch.float16,
    'float16': torch.float16,
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
    'fp32': torch.float32,
    'float32': torch.float32,
    'int8': torch.int8,
    'int32': torch.int32,
}


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
    PARAM_CONFIG = {
        {{PARAM_CONFIG_ENTRIES}}
    }

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

        fmt_indices = input_data.kwargs.pop('isFmt', [])
        if not isinstance(fmt_indices, list):
            fmt_indices = [fmt_indices]
        not_contiguous_indices = input_data.kwargs.pop('isNotContiguous', [])
        if not isinstance(not_contiguous_indices, list):
            not_contiguous_indices = [not_contiguous_indices]
        nullptr_indices = input_data.kwargs.pop('isNullptr', [])
        if not isinstance(nullptr_indices, list):
            nullptr_indices = [nullptr_indices]
        output_ops_str = input_data.kwargs.pop('outputOps', '')

        idx_to_name = {v['index']: k for k, v in self.PARAM_CONFIG.items()}

        for idx in not_contiguous_indices:
            name = idx_to_name.get(idx)
            if name is not None and not self.PARAM_CONFIG[name].get('is_output', False):
                self._handle_not_contiguous(input_data, name)

        {{ACLNN_INPUT_PREPROCESSING}}

        input_args, output_packages = super().init_by_input_data(input_data)

        for idx in fmt_indices:
            name = idx_to_name.get(idx)
            if name is not None:
                self._handle_fmt(input_args, input_data, name, output_packages)

        for idx in nullptr_indices:
            name = idx_to_name.get(idx)
            if name is not None:
                self._handle_nullptr(input_args, name)

        if output_ops_str:
            self._handle_output_modification(input_args, output_packages, output_ops_str)

        {{ACLNN_ARGS_POSTPROCESSING}}

        {{ACLNN_NULL_ARRAY_OPTIONAL}}

        return input_args, output_packages

    # 以下四个函数不可变动，直接拷贝过去
    def _handle_fmt(self, input_args, input_data, tensor_name, output_packages=None, fmt_value=29):
        """
        将指定tensor的input_args条目替换为指定format的aclTensor
        应在 super().init_by_input_data() 之后调用
        :param input_args: super().init_by_input_data() 返回的 input_args 列表
        :param input_data: InputDataset 对象
        :param tensor_name: 要修改的tensor在input_data.kwargs中的key
        :param output_packages: super().init_by_input_data() 返回的 output_packages 列表，output参数需同步更新
        :param fmt_value: ACL format值，默认29(NZ格式)
        """
        from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase
        import torch_npu

        idx = self.PARAM_CONFIG[tensor_name]['index']
        is_output = self.PARAM_CONFIG[tensor_name].get('is_output', False)
        tensor = input_data.kwargs.get(tensor_name)
        if isinstance(tensor, torch.Tensor):
            npu_tensor = tensor.npu()
            new_acl_tensor = nnopbase.create_acl_tensor(npu_tensor, fmt=fmt_value, storage_shape=None)
            torch.npu.synchronize()
            input_args[idx] = new_acl_tensor
            if is_output and output_packages is not None:
                output_names = [n for n, c in self.PARAM_CONFIG.items() if c.get('is_output', False)]
                first_output_idx = self.PARAM_CONFIG[output_names[0]]['index']
                out_idx = idx - first_output_idx
                if out_idx < len(output_packages):
                    output_packages[out_idx] = new_acl_tensor

    def _handle_not_contiguous(self, input_data, tensor_name):
        """
        将指定tensor变为非连续（通过transpose实现）
        应在 super().init_by_input_data() 之前调用
        :param input_data: InputDataset 对象
        :param tensor_name: 要修改的tensor在input_data.kwargs中的key
        """
        tensor = input_data.kwargs.get(tensor_name)
        if isinstance(tensor, torch.Tensor):
            input_data.kwargs[tensor_name] = tensor.T.contiguous().T

    def _handle_nullptr(self, input_args, param_name):
        """
        将指定参数的input_args条目置为空指针，支持tensor/attr/attrs三种类型
        应在 super().init_by_input_data() 之后调用
        :param input_args: super().init_by_input_data() 返回的 input_args 列表
        :param param_name: PARAM_CONFIG 中的参数名
        """
        from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray, TensorPtr

        cfg = self.PARAM_CONFIG[param_name]
        idx = cfg['index']
        ptype = cfg['type']

        if ptype == 'tensor':
            input_args[idx] = TensorPtr()
        elif ptype == 'attrs':
            AclIntArrayPtr = ctypes.POINTER(AclIntArray)
            null_void_ptr = ctypes.c_void_p(None)
            null_array_ptr = ctypes.cast(null_void_ptr, AclIntArrayPtr)
            input_args[idx] = null_array_ptr
        elif ptype == 'attr':
            input_args[idx] = ctypes.c_void_p(None)

    def _handle_output_modification(self, input_args, output_packages, output_ops_str):
        """
        解析outputOps字符串，修改输出参数的shape/dtype/fmt
        应在 super().init_by_input_data() 之后调用
        :param input_args: super().init_by_input_data() 返回的 input_args 列表
        :param output_packages: super().init_by_input_data() 返回的 output_packages 列表
        :param output_ops_str: outputOps字符串，格式为 "输出参数名1: shape:??, dtype:??, fmt:??;输出参数名2: ..."
        """
        from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase

        output_names = [name for name, cfg in self.PARAM_CONFIG.items() if cfg.get('is_output', False)]
        if not output_names:
            return

        first_output_idx = self.PARAM_CONFIG[output_names[0]]['index']

        segments = [s.strip() for s in output_ops_str.split(';') if s.strip()]
        for segment in segments:
            if ':' not in segment:
                continue
            param_name, ops_part = segment.split(':', 1)
            param_name = param_name.strip()
            if param_name not in self.PARAM_CONFIG or not self.PARAM_CONFIG[param_name].get('is_output', False):
                continue

            cfg = self.PARAM_CONFIG[param_name]
            idx = cfg['index']
            out_idx = idx - first_output_idx

            if out_idx >= len(self.task_result.output_info_list):
                continue

            output_data = self.task_result.output_info_list[out_idx]
            dtype = output_data.dtype
            torch_dtype = getattr(torch, dtype.replace("torch.", "")) if isinstance(dtype, str) else dtype

            new_shape = list(output_data.shape)
            new_dtype = torch_dtype
            new_fmt = AclFormat.ACL_FORMAT_ND

            sub_ops = [s.strip() for s in ops_part.split(',') if s.strip()]
            for sub_op in sub_ops:
                if sub_op.startswith('shape:'):
                    new_shape = ast.literal_eval(sub_op[len('shape:'):])
                elif sub_op.startswith('dtype:'):
                    new_dtype = DTYPE_MAP.get(sub_op[len('dtype:'):], torch_dtype)
                elif sub_op.startswith('fmt:'):
                    new_fmt = int(sub_op[len('fmt:'):])

            empty_tensor = torch.zeros(new_shape, dtype=new_dtype, device='npu')
            output = nnopbase.create_acl_tensor(empty_tensor, fmt=new_fmt, storage_shape=None)
            torch.npu.synchronize()

            input_args[idx] = output
            if out_idx < len(output_packages):
                output_packages[out_idx] = output