import torch
import torch.distributed as dist
import torch_npu

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


@register("function_aclnn_quant_matmul_allto_all")
class QuantMatmulAlltoAll(BaseApi):
    """
    Golden计算: NPU 小算子拼接
    """

    def __init__(self, task_result):
        super(QuantMatmulAlltoAll, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def manual_all_to_all(self, x_splits_from_ranks, x_splits_to_ranks):
        # (world_size, m, n/world_size)
        for target_rank in range(self.dist_task_info.world_size):
            if target_rank == self.dist_task_info.rank:
                # 发给自己
                x_splits_from_ranks[self.dist_task_info.rank].copy_(
                    x_splits_to_ranks[target_rank]
                )
                # 从别的进程获取我要的张量
                for src_rank in range(self.dist_task_info.world_size):
                    if src_rank != self.dist_task_info.rank:
                        dist.recv(
                            tensor=x_splits_from_ranks[
                                src_rank
                            ],  # 接收数据的空张量（对应第src_rank个rank）
                            src=src_rank,  # 源rank
                            tag=src_rank * 1000
                            + self.dist_task_info.rank,  # 匹配发送方的标签（源rank*1000+当前rank）
                        )
            else:
                # 发给其他rank
                dist.send(
                    tensor=x_splits_to_ranks[
                        target_rank
                    ],  # 要发送的张量（给第target_rank个rank）
                    dst=target_rank,  # 目标rank
                    tag=self.dist_task_info.rank * 1000
                    + target_rank,  # 通信标签（避免数据混淆）
                )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        x = input_data.kwargs["x1"]  # (M, K)
        weight = input_data.kwargs["x2"]  # (K, N)
        bias = input_data.kwargs["biasOptional"]  # (N, )
        pertoken_scale = input_data.kwargs["x1Scale"]  # (M, )
        scale = input_data.kwargs["x2Scale"]  # (N, )
        transB = input_data.kwargs["transposeX2"]
        is_dequant = x.dtype == torch.int8
        output_type = x.dtype

        if is_dequant:
            output_type = (
                torch.float16 if (bias.dtype == torch.float16) else torch.bfloat16
            )

        if self.name == "cpu":
            """
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ CPU  真值 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            """
            x = x.cpu().to(torch.float32)
            weight = weight.cpu().to(torch.float32)
            bias = bias.cpu().to(torch.float32)
            if transB:
                weight = weight.T.contiguous()
            if is_dequant:
                output = torch.matmul(x, weight)
                scale_todim = torch.unsqueeze(scale.cpu(), 0)
                bias_todim = torch.unsqueeze(bias, 0)
                perTokenScale_todim = torch.unsqueeze(pertoken_scale.cpu(), 1)
                output = (output * scale_todim * perTokenScale_todim + bias_todim).to(
                    output_type
                )
            else:
                output = (torch.matmul(x, weight) + bias).to(output_type)

            # 1. split to world_size times chunk, with each (m, n/world_size)
            send_chunks = list(
                torch.chunk(output, world_size, dim=1)
            )  # (world_size, m, n/world_size)
            send_chunks = [c.contiguous() for c in send_chunks]
            # 2. recv_chunks shape: (world_size, m, n/world_size)
            recv_chunks = [
                torch.empty_like(
                    send_chunks[0], device=output.device, dtype=output.dtype
                )
                for _ in range(world_size)
            ]
            # 3. alltoall
            self.manual_all_to_all(recv_chunks, send_chunks)
            # 4. output shape: (m*world_size, n/world_size)
            y = torch.cat(recv_chunks, dim=0)

            return y

        if self.name == "npu_bm":
            """
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ NPU 小算子级联标杆 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            """
            if transB:
                weight = weight.T.contiguous()
            if is_dequant:  # 量化部分
                output = torch_npu.npu_quant_matmul(
                    x1=x,
                    x2=weight,
                    scale=scale,
                    pertoken_scale=pertoken_scale,
                    bias=bias,
                    output_dtype=output_type,
                )
            else:
                output = (torch.matmul(x, weight) + bias).to(output_type)

            # 1. split to world_size times chunk, with each (m, n/world_size)
            send_chunks = list(torch.chunk(output, world_size, dim=1))
            send_chunks = [c.contiguous() for c in send_chunks]
            # 2. output shape: (m*world_size, n/world_size)
            recv_chunks = [
                torch.empty_like(send_chunks[0], device=x.device, dtype=output_type)
                for _ in range(world_size)
            ]
            # 3. alltoall
            dist.all_to_all(recv_chunks, send_chunks)
            # 4. output
            y = torch.cat(recv_chunks, dim=0)

            return y

    @staticmethod
    def get_hcomm_info(rank_id):
        from torch.distributed.distributed_c10d import _get_default_group

        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            backend_obj = getattr(default_pg, "_get_backend", None)
            hcomm_info = backend_obj(torch.device("npu")).get_hccl_comm_name(rank_id)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank_id)
        return hcomm_info

    def init_by_input_data(self, input_data: InputDataset):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        device = "npu:" + str(rank_id)

        input_data.kwargs["x1"] = input_data.kwargs["x1"].to(device)
        input_data.kwargs["x2"] = input_data.kwargs["x2"].to(device)
        input_data.kwargs["biasOptional"] = input_data.kwargs["biasOptional"].to(device)

        # 量化处理
        is_dequant = input_data.kwargs["x1"].dtype == torch.int8

        if is_dequant:
            input_data.kwargs["x2Scale"] = input_data.kwargs["x2Scale"].to(device)
            input_data.kwargs["x1Scale"] = input_data.kwargs["x1Scale"].to(device)

        # 通信域配置
        if self.device == "pyaclnn" and dist.is_available():
            from torch.distributed.distributed_c10d import _get_default_group

            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcomm_info = default_pg._get_backend(
                    torch.device("npu")
                ).get_hccl_comm_name(rank_id)
            else:
                hcomm_info = default_pg.get_hccl_comm_name(rank_id)
            input_data.kwargs["group"] = default_pg
            input_data.kwargs["world_size"] = world_size


"""
    =================================================================================
    ==============================ACLNN输入特殊处理===================================
    =================================================================================
"""


@register("pyaclnn_aclnn_quant_matmul_allto_all")
class AclnnQuantMatmulAlltoAll(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super(AclnnQuantMatmulAlltoAll, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def init_by_input_data(self, input_data: InputDataset):
        # 处理group
        rank_id = self.dist_task_info.rank
        self.group = input_data.kwargs["group"]
        input_data.kwargs["group"] = (
            input_data.kwargs["group"]
            ._get_backend(torch.device("npu"))
            .get_hccl_comm_name(rank_id)
        )

        # 确认是否量化
        is_dequant = input_data.kwargs["x1"].dtype == torch.int8

        # 抛弃 world_size
        input_data.kwargs.pop("world_size")
        # 转化为 AscendC
        input_args, output_packages = super().init_by_input_data(input_data)

        # offset, antiquant_scale, antiquant_offset, bias
        input_args[5] = self.get_null_tensor_pte()
        input_args[6] = self.get_null_tensor_pte()
        input_args[7] = self.get_null_tensor_pte()

        # scale, pertoken_scale
        if not is_dequant:
            input_args[3] = self.get_null_tensor_pte()
            input_args[4] = self.get_null_tensor_pte()

        return input_args, output_packages

    def get_null_tensor_pte(self):
        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型
        return null_tensor_ptr

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
