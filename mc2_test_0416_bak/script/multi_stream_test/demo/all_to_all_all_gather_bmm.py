# import os
# import pytest
# import torch
# import torch.distributed as dist
# from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
# import torch_npu
# from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
#
# import torch.multiprocessing as mp
#
# import torchair as tng
# from torchair.configs.compiler_config import CompilerConfig
# from torchair.core.utils import logger
# import logging
# # logger.setLevel(logging.DEBUG)
#
# import numpy as np
#
# world_size = 8
# ep_size = 4
# tp_size = 2
#
# def setup_ep_tp(rank, tp_size, ep_size, backend_type):
#     # 初始化EP域
#     print("device %d initialize ep group" % rank, flush=True)
#     for i in range(tp_size):
#         ep_ranks = [x + ep_size * i for x in range(ep_size)]
#         ep_group = dist.new_group(backend=backend_type, ranks=ep_ranks)
#         if rank in ep_ranks:
#             ep_group_tmp = ep_group
#     print("device %d initialize tp group" % rank, flush=True)
#     for i in range(ep_size):
#         tp_ranks = [x * ep_size + i for x in range(tp_size)]
#         tp_group = dist.new_group(backend=backend_type, ranks=tp_ranks)
#         if rank in tp_ranks:
#             tp_group_tmp = tp_group
#     return ep_group_tmp, tp_group_tmp
#
# def get_ep_tp_hcomm_info(rank, ep_size, tp_size):
#     ep_group, tp_group = setup_ep_tp(rank, tp_size, ep_size, "hccl")
#     if torch.__version__ > '2.0.1':
#         ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
#         tp_hcomm_info = tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
#     else:
#         ep_hcomm_info = ep_group.get_hccl_comm_name(rank)
#         tp_hcomm_info = tp_group.get_hccl_comm_name(rank)
#     return ep_hcomm_info, tp_hcomm_info
#
# class ALLTOALL_ALLGATHER_BMM_GRAPH_Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x_npu=None, weight_npu=None, ep_group=None, ep_size=None, tp_group=None, tp_size=None,
#                 bias_npu=None, x_shard_type=None, act_type=None, out_y2_flag=None, out_y3_flag=None):
#
#         e, c, h, m, k, n = [32, 1024, 1024, 1024, 16384, 16384]
#         device = torch.device("npu")
#         np_input = np.random.uniform(-1, 1, [m, k]).astype(np.float16)
#         np_weight = np.random.uniform(-1, 1, [k, n]).astype(np.float16)
#         input_ = torch.from_numpy(np_input).to(device)
#         weight = torch.from_numpy(np_weight).to(device)
#
#         with tng.scope.npu_stream_switch('1'):
#             torch.matmul(input_, weight)
#         with tng.scope.npu_stream_switch('2'):
#             output = npu_alltoall_allgather_bmm(x_npu,
#                                        weight_npu,
#                                        ep_group,
#                                        ep_size,
#                                        tp_group,
#                                        tp_size,
#                                        bias=bias_npu,
#                                        shard_type=x_shard_type,
#                                        act_type=act_type,
#                                        need_allgather_out=out_y2_flag,
#                                        need_activation_feature=out_y3_flag)
#
#         return output
#
# def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
#     torch_npu.npu.set_device(rank)
#     init_method = 'tcp://' + master_ip + ':' + master_port
#     dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
#     from torch.distributed.distributed_c10d import _get_default_group
#     default_pg = _get_default_group()
#     if torch.__version__ > '2.0.1':
#         hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
#     else:
#         hcom_info = default_pg.get_hccl_comm_name(rank)
#
#     input = torch.randn(gmm_x, dtype=dtype).npu()
#     weight = torch.randn(gmm_w, dtype=dtype).npu()
#
#     model = ALLTOALL_ALLGATHER_BMM_GRAPH_Model()
#     npu_backend = torchair.get_npu_backend(compiler_config=None)
#     # 静态图：dynamic=False；动态图：dynamic=True
#     model = torch.compile(ALLTOALL_ALLGATHER_BMM_GRAPH_Model(), backend=npu_backend, dynamic=False)
#     print(model(gmm_x=input,
#                     gmm_weight=weight,
#                     send_counts_tensor=None,
#                     recv_counts_tensor=None,
#                     mm_x=None,
#                     mm_weight=None,
#                     hcom=hcom_info,
#                     ep_world_size=ep_world_size,
#                     send_counts=send_counts,
#                     recv_counts=recv_counts,
#                     trans_gmm_weight=False,
#                     trans_mm_weight=False,
#                     permute_out_flag=True))
#
# if __name__ == "__main__":
#     dtype = torch.float16
#     x_shard_type = 1
#     out_y2_flag = True
#     out_y3_flag = False
#     act_type = "None"
#     transpose_weight = False
#     rank = int(os.environ["LOCAL_RANK"])
#     torch_npu.npu.set_device(rank)
#     dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
#     ep_group, tp_group = get_ep_tp_hcomm_info(rank, ep_size, tp_size)
#     print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {ep_group}, {tp_group}')
#     E, C, H, M = 4, 1024, 1024, 8192
#     if x_shard_type == 0:
#         x_shape = (E, C, H / tp_size)
#     elif x_shard_type == 1:
#         x_shape = (E, C / tp_size, H)
#     else:
#         x_shape = (E / ep_size, tp_size * ep_size * C, M / tp_size)
#     weight_shape = (E / ep_size, H, M / tp_size)
#     if transpose_weight == True:
#         weight_shape = (E / ep_size, M / tp_size, H)
#     bias_shape = (E / ep_size, 1, M / tp_size)
#
#     x_shape = tuple(int(item) for item in x_shape)
#     weight_shape = tuple(int(item) for item in weight_shape)
#     bias_shape = tuple(int(item) for item in bias_shape)
#     x = torch.rand(x_shape)
#     weight = torch.rand(weight_shape)
#     bias = torch.rand(bias_shape)
#     x_npu = x.npu().to(dtype)
#     weight_npu = weight.npu().to(dtype)
#     if transpose_weight == True:
#         print(f'!!!!before transpose, weight_npu.size()={weight_npu.size()}')
#         weight_npu = weight_npu.transpose(1, 2)
#         print(f'!!!!after transpose, weight_npu.size()={weight_npu.size()}')
#         print(f'!!!!after transpose, weight_npu.is_contiguous()={weight_npu.is_contiguous()}')
#     bias_npu = bias.npu().to(dtype)
#     # 赋值None可以验证bias为空的场景
#     bias_npu = None
#     epWorkSize = 8
#
#     mp.spawn(run_npu_alltoallv_gmm, args=(x_npu, weight_npu, ep_group, ep_size, tp_group, tp_size, bias_npu, x_shard_type, act_type, out_y2_flag, out_y3_flag), nprocs=epWorkSize)
#
#
#
#


import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm

import torchair as tng
import numpy as np

world_size = 8
ep_size = 4
tp_size = 2


class ALLTOALL_ALLGATHER_BMM_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_npu=None, weight_npu=None, ep_group=None, ep_size=None, tp_group=None, tp_size=None,
                bias_npu=None, x_shard_type=None, act_type=None, out_y2_flag=None, out_y3_flag=None):

        e, c, h, m, k, n = [32, 1024, 1024, 1024, 16384, 16384]
        device = torch.device("npu")
        np_input = np.random.uniform(-1, 1, [m, k]).astype(np.float16)
        np_weight = np.random.uniform(-1, 1, [k, n]).astype(np.float16)
        input_ = torch.from_numpy(np_input).to(device)
        weight = torch.from_numpy(np_weight).to(device)

        with tng.scope.npu_stream_switch('1'):
            torch.matmul(input_, weight)
        with tng.scope.npu_stream_switch('2'):
            output = npu_alltoall_allgather_bmm(x_npu,
                                       weight_npu,
                                       ep_group,
                                       ep_size,
                                       tp_group,
                                       tp_size,
                                       bias=bias_npu,
                                       shard_type=x_shard_type,
                                       act_type=act_type,
                                       need_allgather_out=out_y2_flag,
                                       need_activation_feature=out_y3_flag)

        return output

def setup_ep_tp(rank, tp_size, ep_size, backend_type):


    return ep_group_tmp, tp_group_tmp


def get_ep_tp_hcomm_info(rank, ep_size, tp_size):
    ep_group, tp_group = setup_ep_tp(rank, tp_size, ep_size, "hccl")

    return ep_hcomm_info, tp_hcomm_info


if __name__ == '__main__':
    dtype = torch.float16
    x_shard_type = 1
    out_y2_flag = True
    out_y3_flag = False
    act_type = "None"
    transpose_weight = False
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    ep_group, tp_group = get_ep_tp_hcomm_info(rank, ep_size, tp_size)
    print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {ep_group}, {tp_group}')
    E, C, H, M = 4, 1024, 1024, 8192
    if x_shard_type == 0:
        x_shape = (E, C, H / tp_size)
    elif x_shard_type == 1:
        x_shape = (E, C / tp_size, H)
    else:
        x_shape = (E / ep_size, tp_size * ep_size * C, M / tp_size)
    weight_shape = (E / ep_size, H, M / tp_size)
    if transpose_weight == True:
        weight_shape = (E / ep_size, M / tp_size, H)
    bias_shape = (E / ep_size, 1, M / tp_size)

    x_shape = tuple(int(item) for item in x_shape)
    weight_shape = tuple(int(item) for item in weight_shape)
    bias_shape = tuple(int(item) for item in bias_shape)
    x = torch.rand(x_shape)
    weight = torch.rand(weight_shape)
    bias = torch.rand(bias_shape)
    x_npu = x.npu().to(dtype)
    weight_npu = weight.npu().to(dtype)
    if transpose_weight == True:
        print(f'!!!!before transpose, weight_npu.size()={weight_npu.size()}')
        weight_npu = weight_npu.transpose(1, 2)
        print(f'!!!!after transpose, weight_npu.size()={weight_npu.size()}')
        print(f'!!!!after transpose, weight_npu.is_contiguous()={weight_npu.is_contiguous()}')
    bias_npu = bias.npu().to(dtype)
    # 赋值None可以验证bias为空的场景
    bias_npu = None

    npu_backend = tng.get_npu_backend(compiler_config=None)
    model = torch.compile(ALLTOALL_ALLGATHER_BMM_GRAPH_Model(), backend=npu_backend, dynamic=False)

    y_npu = model(x_npu,
                  weight_npu,
                  ep_group,
                  ep_size,
                  tp_group,
                  tp_size,
                  bias=bias_npu,
                  shard_type=x_shard_type,
                  act_type=act_type,
                  need_allgather_out=out_y2_flag,
                  need_activation_feature=out_y3_flag)

    # y_npu = npu_alltoall_allgather_bmm(x_npu,
    #                                     weight_npu,
    #                                     ep_group,
    #                                     ep_size,
    #                                     tp_group,
    #                                     tp_size,
    #                                     bias=bias_npu,
    #                                     shard_type=x_shard_type,
    #                                     act_type=act_type,
    #                                     need_allgather_out=out_y2_flag,
    #                                     need_activation_feature=out_y3_flag)
    print(y_npu)
    # if rank == 0:
    #     for i, y in enumerate(y_npu[0]):
    #         y.cpu().numpy().tofile(f"./y_{i}.bin")
