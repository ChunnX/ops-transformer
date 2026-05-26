import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse


def raise_data_type_list(dtype):
    data_type_list = []
    if dtype == 'int8' or dtype == 'all':
        data_type_list.append('torch.int8')
    if dtype == 'int32' or dtype == 'all':
        data_type_list.append('torch.int32')
    if dtype == 'float32' or dtype == 'all':
        data_type_list.append('torch.float32')        
    if dtype == 'float16' or dtype == 'all':
        data_type_list.append('torch.float16')  
    if dtype == 'int64' or dtype == 'all':
        data_type_list.append('torch.int64')  
    return data_type_list    


def run_hccl(rank, world_size,master_ip,master_port,data,dtype):
    torch_npu.npu.set_device(rank)
    print("current device:", torch_npu.npu.current_device())
    init_method = 'tcp://'
    master_ip = master_ip
    master_port = master_port
    init_method += master_ip + ':' + master_port
    print(init_method)
    print('rank : %s' % rank)
    print('pid : %s' % os.getpid())
    # create default process group
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)

    try:
        #allgather
        data_type_list = raise_data_type_list(dtype)
        for data_type in data_type_list:
            print("data_type: ", data_type)
            exec("tensor_allgather_list = [torch.zeros([1, {}],dtype={}).npu() for _ in range(world_size)]".format(data, data_type))
            exec("tensor_allgather = torch.ones([1, {}],dtype={}).npu()".format(data, data_type))
            exec("dist.all_gather(tensor_allgather_list, tensor_allgather)")
            exec("print('tensor_allgather_list : ' + str(tensor_allgather_list))")
            exec("allgather_check_list = [torch.ones([1, {}],dtype={}).npu() for _ in range(world_size)]".format(data, data_type))
            exec("print('allgather_check_list : ' + str(allgather_check_list))")
            for i in range(world_size):
                exec("assert (tensor_allgather_list[i] == allgather_check_list[i]).all()")

    except Exception as e:
        print('ERROR : %s' % e)
        print('device:%s pytorch hccl test fail' % rank)
    else:
        print('device:%s pytorch hccl test success' % rank )


def main():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--worksize', default=8,
                         help="""worksize""")
    cmdline.add_argument('--master_ip', default='127.0.0.1',
                         help="""master_ip""")
    cmdline.add_argument('--master_port', default='50001',
                         help="""master_port""")
    cmdline.add_argument('--data_list', default='1024',
                         help="""data_list, ep:1024,2048,3072,10240,102400...""")
    cmdline.add_argument('--dtype', default='all',
                         help="""dtype""")
    cmdline.add_argument('--mul_num', default='1',
                         help="""mul_num""")

    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    config = {}
    worksize= int(FLAGS.worksize)
    master_ip= FLAGS.master_ip
    master_port= FLAGS.master_port
    data_list = FLAGS.data_list.split(',')
    dtype = FLAGS.dtype
    mul_num = int(FLAGS.mul_num)

    print("worksize          :%s" % worksize)
    print("master_ip       :%s" % master_ip)
    print("master_port          :%s" % master_port)
    print("data_list         :%s" % data_list)
    print("dtype         :%s" % dtype)
    print("pid           :%s" % os.getpid())
    print("mul_num           :%s" % mul_num)

    for i in range(mul_num):
        data = int(data_list[i % len(data_list)])
        print("torch stability test {}, used data size: {}".format(i, data))
        mp.spawn(run_hccl,
            args=(worksize,master_ip,master_port,data,dtype),
            nprocs=worksize)


if __name__=="__main__":
    main()
