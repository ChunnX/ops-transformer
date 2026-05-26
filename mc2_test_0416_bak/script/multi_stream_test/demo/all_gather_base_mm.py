import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

import torch, os
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import logging
# logger.setLevel(logging.DEBUG)

import numpy as np


class ALLGATHER_MM_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, weight, hcomm_info, world_size, gather_output):
        # output1, output2 = None, None
        with tng.scope.npu_stream_switch('1'):
            output1 = torch_npu.npu_all_gather_base_mm(input, weight, hcomm_info, world_size)
        with tng.scope.npu_stream_switch('2'):

            output2 = torch_npu.npu_mm_reduce_scatter_base(input, weight, hcomm_info, world_size)
        return output1, output2
def define_model(model, graph_type):
    import torchair
    if graph_type == 1:  # 传统入图模式，静态shape+在线编译场景
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:  # ACLNN入图模式，动态shape+二进制
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        model = torch.compile(model, backend=npu_backend, dynamic=True)
    else:
        print("Error type")
    return model
def get_graph(input, weight, hcomm_info, world_size, gather_output):
    model = ALLGATHER_MM_GRAPH_Model()
    model = define_model(model, 1)
    model_output = model(input, weight, hcomm_info, world_size, gather_output=gather_output)
    output_npu = model_output[0]
    gather_output_npu = model_output[1]
    return output_npu, gather_output_npu
def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)
    single_shape = [x1_shape[0] // world_size, x1_shape[1]]
    input = torch.randn(single_shape, dtype=dtype).npu()
    weight = torch.randn(x2_shape, dtype=dtype).npu()
    is_gather_out = True
    output1, output2 = get_graph(input, weight, hcomm_info, world_size, is_gather_out)
    print("output1:", output1)
    print("output2:", output2)
if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16
    mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)