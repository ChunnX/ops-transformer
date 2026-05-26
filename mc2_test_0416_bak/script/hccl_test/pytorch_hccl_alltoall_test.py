import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import math


def raise_data_type_list(dtype):
    data_type_list = []
    if dtype == 'int8' or dtype == 'all':
        data_type_list.append(torch.int8)
    if dtype == 'int32' or dtype == 'all':
        data_type_list.append(torch.int32)
    if dtype == 'float32' or dtype == 'all':
        data_type_list.append(torch.float32)
    if dtype == 'float16' or dtype == 'all':
        data_type_list.append(torch.float16)
    if dtype == 'int64' or dtype == 'all':
        data_type_list.append(torch.int64)
    if dtype == 'bfloat16' or dtype == 'all':
        data_type_list.append(torch.bfloat16)
    return data_type_list


def check_tensor_list(tensor_list, rank_id,world_size,data,data_type,input_list):
    alltoall_check_tensor_list = [torch.full([input_list[rank_id]],i,dtype=data_type).npu() for i in range(world_size)]
    print('alltoall_check_tensor_list : ' + str(alltoall_check_tensor_list))
    for i in range(world_size):
        assert (alltoall_check_tensor_list[i].cpu() == tensor_list[i].cpu()).all()

def run_hccl(rank, test_config):
    torch_npu.npu.set_device(rank)
    print("current device:", torch_npu.npu.current_device())
    init_method = 'tcp://' + test_config['master_ip'] + ':' + test_config['master_port']
    local_rank = rank
    rank += test_config['dev_num'] * test_config['server_index']
    world_size = test_config['world_size']
    print('init_method  : %s' % init_method)
    print('rank         : %s' % rank)
    print('pid          : %s' % os.getpid())
    # create default process group
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)

    try:
        #alltoall
        data_type_list = raise_data_type_list(test_config['dtype'])
        for data_type in data_type_list:
            for data in test_config['data_list']:
                for test_loop in range(test_config['mul_num']):
                    print("rank {} data {} data_type {} loop {}".format(rank, data, data_type, test_loop+1))
                    alltoall_input = torch.full([data], rank, dtype=data_type).npu()
                    alltoall_input_list = list(torch.tensor_split(alltoall_input, world_size))
                    print('alltoall_input_list : ' + str(alltoall_input_list))
                    input_list = [math.ceil(data / world_size) for _ in range(math.ceil(data % world_size))]
                    for _ in range(world_size - (data % world_size)):
                        input_list.append(data // world_size)
                    alltoall_output_list = [torch.empty([input_list[rank]],dtype=data_type).npu() for _ in range(world_size)]
                    print('alltoall_output_list_org : ' + str(alltoall_output_list))
                    dist.all_to_all(alltoall_output_list,alltoall_input_list)
                    print('alltoall_output_list : ' + str(alltoall_output_list))
                    check_tensor_list(alltoall_output_list, rank, world_size, data, data_type, input_list)
                
    
    except Exception as e:
        print('ERROR : %s' % e)
        print('device:%s pytorch hccl test fail' % local_rank)
    else:
        print('device:%s pytorch hccl test success' % local_rank)


def main():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--master_ip', default=None,
                         help="""master_ip""")
    cmdline.add_argument('--master_port', default='50001',
                         help="""master_port""")
    cmdline.add_argument('--dev_num', default=None,
                         help="""dev_num""")
    cmdline.add_argument('--server_num', default=None,
                         help="""server_num""")
    cmdline.add_argument('--server_index', default='0',
                         help="""server_index""")
    cmdline.add_argument('--data_list', default='1024',
                         help="""data_list, ep:1024,2048,3072,10240,102400...""")
    cmdline.add_argument('--dtype', default='all',
                         help="""dtype""")
    cmdline.add_argument('--mul_num', default='1',
                         help="""mul_num""")
    cmdline.add_argument('--used_devices', default=None,
                         help="""4,5,6,7""")
    cmdline.add_argument('--used_profiling', default='false', choices=['true', 'false'],
                         help="""true or false""")
    cmdline.add_argument('--profiling_path', default='/var/log/npu/profiling/',
                         help="""save profiling path""")
    cmdline.add_argument('--used_algo', default='auto_algo', choices=['auto_algo', 'ring', 'H-D_R', 'NHR', 'NB','pipeline', 'pairwise', 'fullmesh'],
                         help="""auto_algo ring H-D_R NHR NB or pipeline""")
    cmdline.add_argument('--expansion_mode', default='false', choices=['false', 'AI_CPU', 'AIV'],
                         help="""true or false""")
    cmdline.add_argument('--used_deterministic', default='false', choices=['true', 'false'],
                         help="""true or false""")

    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    config = {}
    config['master_ip'] = FLAGS.master_ip
    config['master_port'] = FLAGS.master_port
    config['dev_num'] = int(FLAGS.dev_num)
    config['server_num'] = int(FLAGS.server_num)
    config['server_index'] = int(FLAGS.server_index)
    config['dtype'] = FLAGS.dtype
    config['data_list'] = [int(i) for i in FLAGS.data_list.split(',')]
    config['mul_num'] = int(FLAGS.mul_num)
    config['world_size'] = config['server_num'] * config['dev_num']

    config['used_devices'] = FLAGS.used_devices
    if config['used_devices'] != None:
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = config['used_devices']

    config['used_profiling'] = FLAGS.used_profiling
    config['profiling_path'] = FLAGS.profiling_path
    if config['used_profiling'] == 'true':
        config['used_profiling'] = True
    elif config['used_profiling'] == 'false':
        config['used_profiling'] = False
    else:
        raise ValueError("Invalid command line arg(s)")

    # set HCCL_ALGO
    config['used_algo'] = FLAGS.used_algo
    if config['used_algo'] == 'auto_algo':
        os.popen('unset HCCL_ALGO')
    elif config['used_algo'] == 'ring':
        os.environ['HCCL_ALGO'] = "level0:NA;level1:ring"
    elif config['used_algo'] == 'H-D_R':
        os.environ['HCCL_ALGO'] = "level0:NA;level1:H-D_R"
    elif config['used_algo'] == 'NHR':
        os.environ['HCCL_ALGO'] = "level0:NA;level1:NHR"
    elif config['used_algo'] == 'NB':
        os.environ['HCCL_ALGO'] = "level0:NA;level1:NB"
    elif config['used_algo'] == 'pipeline':
        os.environ['HCCL_ALGO'] = "level0:NA;level1:pipeline"
    elif config['used_algo'] == 'pairwise':
        os.environ['HCCL_ALGO'] = "level0:NA;level1:pairwise"
    elif config['used_algo'] == 'fullmesh':
        os.environ['HCCL_ALGO'] = "level0:NA;level1:fullmesh"
    else:
        raise ValueError("Invalid command line arg(s)")

    config['expansion_mode'] = FLAGS.expansion_mode
    if config['expansion_mode'] == 'false':
        os.popen('unset HCCL_OP_EXPANSION_MODE')
    elif config['expansion_mode'] == 'AI_CPU':
        os.environ['HCCL_OP_EXPANSION_MODE'] = "AI_CPU"
    elif config['expansion_mode'] == 'AIV':
        os.environ['HCCL_OP_EXPANSION_MODE'] = "AIV"
    else:
        raise ValueError("Invalid command line arg(s)")

    print("master_ip        :%s" % config['master_ip'])
    print("master_port      :%s" % config['master_port'])
    print("dev_num          :%s" % config['dev_num'])
    print("server_num       :%s" % config['server_num'])
    print("server_index     :%s" % config['server_index'])
    print("worksize         :%s" % config['world_size'])
    print("data_list        :%s" % config['data_list'])
    print("dtype            :%s" % config['dtype'])
    print("mul_num          :%s" % config['mul_num'])
    print("used_devices     :%s" % config['used_devices'])
    print("used_algo        :%s" % config['used_algo'])
    print("expansion_mode   :%s" % config['expansion_mode'])
    print("main_pid         :%s" % os.getpid())

    mp.spawn(run_hccl,
        args=(config,),
        nprocs=config['dev_num'])


if __name__=="__main__":
    main()
