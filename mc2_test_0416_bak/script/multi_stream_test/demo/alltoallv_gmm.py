import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import torchair

import torch, os
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import logging
# logger.setLevel(logging.DEBUG)

import numpy as np

class ALLTOALLV_GMM_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,gmm_x, gmm_weight,
                hcom, ep_world_size,
                send_counts, recv_counts, send_counts_tensor, recv_counts_tensor,
                mm_x, mm_weight,
                trans_gmm_weight, trans_mm_weight, permute_out_flag):

        e, c, h, m, k, n = [32, 1024, 1024, 1024, 16384, 16384]
        device = torch.device("npu")
        np_input = np.random.uniform(-1, 1, [m, k]).astype(np.float16)
        np_weight = np.random.uniform(-1, 1, [k, n]).astype(np.float16)
        input_ = torch.from_numpy(np_input).to(device)
        weight = torch.from_numpy(np_weight).to(device)

        with tng.scope.npu_stream_switch('1'):
            torch.matmul(input_, weight)
        with tng.scope.npu_stream_switch('2'):
            output = torch_npu.npu_alltoallv_gmm(gmm_x=gmm_x,
                                        gmm_weight=gmm_weight,
                                        hcom=hcom,
                                        ep_world_size=ep_world_size,
                                        send_counts=list(send_counts),
                                        recv_counts=list(recv_counts),
                                        send_counts_tensor=None,  # send_counts_tensor,
                                        recv_counts_tensor=None,  # recv_counts_tensor,
                                        mm_x=mm_x,
                                        mm_weight=mm_weight,
                                        trans_gmm_weight=trans_gmm_weight,
                                        trans_mm_weight=trans_mm_weight,
                                        permute_out_flag=permute_out_flag)

        return output

def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcom_info = default_pg.get_hccl_comm_name(rank)

    input = torch.randn(gmm_x, dtype=dtype).npu()
    weight = torch.randn(gmm_w, dtype=dtype).npu()

    model = ALLTOALLV_GMM_GRAPH_Model()
    npu_backend = torchair.get_npu_backend(compiler_config=None)
    # 静态图：dynamic=False；动态图：dynamic=True
    model = torch.compile(ALLTOALLV_GMM_GRAPH_Model(), backend=npu_backend, dynamic=False)
    print(model(gmm_x=input,
                    gmm_weight=weight,
                    send_counts_tensor=None,
                    recv_counts_tensor=None,
                    mm_x=None,
                    mm_weight=None,
                    hcom=hcom_info,
                    ep_world_size=ep_world_size,
                    send_counts=send_counts,
                    recv_counts=recv_counts,
                    trans_gmm_weight=False,
                    trans_mm_weight=False,
                    permute_out_flag=True))

if __name__ == "__main__":
    epWorkSize = 8
    e = 4
    master_ip = '127.0.0.1'
    master_port = '50001'
    BS = 512
    K = 8
    gmm_x_shape = [BS*K, 2048]
    gmm_weight_shape = [e, 2048, 2048]
    send_counts = [128] * (e * epWorkSize)
    recv_counts = [128] * (e * epWorkSize)
    dtype = torch.float16
    mp.spawn(run_npu_alltoallv_gmm, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
