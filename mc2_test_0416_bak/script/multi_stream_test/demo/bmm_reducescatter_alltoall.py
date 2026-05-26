import os
import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group, ReduceOp
import torch_npu
from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall

world_size = 8
ep_size = 4
tp_size = 2


def get_hcomm_info(n, i):
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(i)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(i)
    return hcomm_info


def setup_ep_tp(rank, tp_size, ep_size, backend_type):

    return ep_group_tmp, tp_group_tmp


def get_ep_tp_hcomm_info(rank, ep_size, tp_size):
    ep_group, tp_group = setup_ep_tp(rank, tp_size, ep_size, "hccl")

    return ep_hcomm_info, tp_hcomm_info


def test_npu_bmm_reducescatter_alltoall(dtype, y_shard_type, transpose_weight):
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    ep_group, tp_group = get_ep_tp_hcomm_info(rank, ep_size, tp_size)
    hcomm_info = get_hcomm_info(world_size, rank)
    print(f'current device: {torch_npu.npu.current_device()}, local rank = {rank}, hcomm_info = {ep_group}, {tp_group}')
    E, C, H, M = 4, 1024, 1024, 8192
    if y_shard_type == 0:
        x_shape = (E / ep_size, ep_size * C, M / tp_size)
        bias_shape = (E / ep_size, 1, H / tp_size)
    else:
        x_shape = (E / ep_size, tp_size * ep_size * C, M / tp_size)
        bias_shape = (E / ep_size, 1, H)
    weight_shape = (E / ep_size, M / tp_size, H)
    if transpose_weight == True:
        weight_shape = (E / ep_size, H, M / tp_size)

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

    y = npu_bmm_reducescatter_alltoall(x_npu,
                                       weight_npu,
                                       ep_group,
                                       ep_size,
                                       tp_group,
                                       tp_size,
                                       bias=bias_npu,
                                       shard_type=y_shard_type)
    print(f'y_shape = {y.size()}')
    if y_shard_type == 0:
        assert y.size() == (E, C, int(H / tp_size))
    else:
        assert y.size() == (E, C, H)
    return y


if __name__ == '__main__':
    dtype = torch.float16
    shard_type = 1
    transpose_weight = False
    y_npu = test_npu_bmm_reducescatter_alltoall(dtype, shard_type, transpose_weight)
    rank = int(os.environ["LOCAL_RANK"])
    # if rank == 0:
    #     for i, y in enumerate(y_npu):
    #         y.cpu().numpy().tofile(f"./y_{i}.bin")
