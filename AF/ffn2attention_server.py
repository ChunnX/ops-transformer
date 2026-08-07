import torch
import random
import torch.distributed as dist
from torch.multiprocessing import Process, Manager
import torch.multiprocessing as mp
import os
import time
import csv
import sys
import numpy as np
import pandas as pd
import copy
import torch_npu
import toolss_attn
import cann_ops_transformer as cot
import ctypes

##################### 测试开关 #####################
current_case_name = 'Test_FFNtoAttention_16die_ID001'

is_attn2ffn = 0
is_ffn2attn = 1

# region 性能检测相关
is_performance = 0
loop = 100              # 性能检测时的性能采集轮数
attn2ffn_profiling_path = './Profiling_Result_attn2ffn'
ffn2attn_profiling_path = './Profiling_Result_ffn2attn'
# endregion

# 量化相关
quant_mode = 2
is_quant = (quant_mode > 0)
is_attn2ffn_scales = 0

# 自定义windows区
enable_window = True  # 是否开启自定义window区功能
window_size = 1024 * 1024 * 200

# 
is_full_core = 1
core_num = 24
aic_core_num = core_num                                         # 非全核模式使用的aic核数
aiv_core_num = core_num * 2                                     # 非全核模式使用的aiv核数

##################### 模型相关设置 #####################
bs = 8
h = 7168
scale_pad = 128                     # 用于存放量化参数
hs = h + scale_pad                  # 量化时h的长度
k = 1
micro_batch_num = 1
default_micro_batch_id = 0          
layer_num = 1                       # 模型层数
default_layer_id = 0                # 默认模型层id

server_num = 1
server_index = 0
rank_per_dev = 2                   # 每个设备（如 GPU）上运行的 rank 数量（这里可能指逻辑 rank，用于模拟多进程）                       
world_size = 2
shared_expert_rank_num_ffn = 0
moe_rank_ffn_num = 1
experts_per_rank = 1

                
default_window_addr = 0x12c101800000 # 默认win区地址
compare_per_rank = 1            # 每个 rank 会单独计算并比较自己的结果

moe_expert_num = moe_rank_ffn_num * experts_per_rank  # 8个moe专家
shared_expert_num = 0
ffn_worker_num = shared_expert_rank_num_ffn + moe_rank_ffn_num # ffn使用5个rank
attention_worker_num = world_size - ffn_worker_num  # attention使用11个rank
expert_num_per_token = k + shared_expert_num  # 9个专家


##################### 数据格式相关 #####################
out_dtype = 0
input_dtype = torch.bfloat16 if out_dtype == 0 else torch.float16

dtype_mapping = {
    "float": torch.float,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}
set_dtype = os.getenv("DTYPE", "bfloat16")
if out_dtype:
    set_dtype = os.getenv("DTYPE", "float16")
if is_quant and is_attn2ffn:
    set_dtype = "int8"
model_dtype = dtype_mapping.get(set_dtype, torch.bfloat16)


##################### ffn_to_attention 算子参数相关 #####################
actual_token_num=bs * micro_batch_num * attention_worker_num * expert_num_per_token  # token数量

# 算子参数
token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]
token_data_table_shape = [micro_batch_num, bs, expert_num_per_token, hs]
attn_rank_table = torch.arange(attention_worker_num, dtype=torch.int32)

##################### 通信参数相关 #####################
master_ip = '71.20.45.104' if server_num > 1 else '127.0.0.1'
# [[0,1,2,...,15]]
ep_ranks_list = list(range(world_size))


def print_setting():
    ''' 打印相关参数
    '''
    print(f"{ffn_worker_num=}")
    print(f"{attention_worker_num=}")
    print(f"{expert_num_per_token=}")
    print(f"{is_performance=}")
    print(f"{enable_window=}")
    print(f"{is_full_core=}")


def ffn2attn_gen_inputs():
    def gen_x():
        '''ffn生成的token, 形状是[Y, h]
        '''
        return torch.empty(size=[actual_token_num, h], dtype=input_dtype).uniform_(-1024, 1024)

    def gen_session_ids():
        '''ffn发送的每个token对应的attention_id, token分为attention_worker_num份, 发送给对应的attention
        '''
        # 如果 actual_token_num 不能被 attention_worker_num 整除??
        session_repeat=int (actual_token_num / attention_worker_num)
        return torch.repeat_interleave(torch.arange(attention_worker_num, dtype=torch.int32), repeats=session_repeat)

    def gen_micro_batch_ids():
        """token对应的micro_batch_id
        """
        return torch.repeat_interleave(torch.arange(micro_batch_num, dtype=torch.int32), repeats=int(bs * expert_num_per_token)).repeat(attention_worker_num)

    def gen_batch_ids():
        '''这里指每个token对应的batch_id, 先按micro_batch_num分割, 再按bs
        '''
        return torch.repeat_interleave(torch.arange(bs, dtype=torch.int32), repeats=expert_num_per_token).repeat(micro_batch_num * attention_worker_num)

    def gen_expert_offsets():
        '''这里指每个token对应的专家的偏移，定义了8个专家，构造类似这样的数组循环96次[0,1,..,7]
        '''
        expert_repeats= int (actual_token_num / expert_num_per_token)
        return torch.arange(expert_num_per_token, dtype=torch.int32).repeat(expert_repeats)

    x = gen_x()
    session_ids = gen_session_ids()
    micro_batch_ids = gen_micro_batch_ids()
    batch_ids = gen_batch_ids()
    expert_offsets = gen_expert_offsets()

    return x, session_ids, micro_batch_ids, batch_ids, expert_offsets

def ffn2attn_golden(x, attention_ranks, micro_batch_ids, batch_ids, expert_offsets):
    token_info = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num_per_token)
    token_data = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num_per_token, hs)
    token_data_mask = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num_per_token, hs, dtype=torch.bool)

    for token_id in range(actual_token_num):
        # 获取当前token对应的rank、micro_batch、batch_id、expert_offset
        current_attention_rank = attention_ranks[token_id]
        current_micro_batch_id = micro_batch_ids[token_id]
        current_batch_id = batch_ids[token_id]
        current_expert_offset = expert_offsets[token_id]
        token_i = x[token_id]
        token_info[current_attention_rank, current_micro_batch_id, current_batch_id, current_expert_offset] = 1
        token_data[current_attention_rank, current_micro_batch_id, current_batch_id, current_expert_offset, :h] = token_i
        token_data_mask[current_attention_rank, current_micro_batch_id, current_batch_id, current_expert_offset, :h] = torch.ones(h, dtype=torch.bool)

    return token_info, token_data, token_data_mask


def write_row_to_csv(csv_name: str, row_content: list):
    if not os.path.exists("csv_result"):
        os.makedirs("csv_result")
    with open(f'csv_result/{csv_name}', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)


def get_performance(profiling_path):
    # op_types = path_to_op_types[profiling_path]
    op_types = ['FFNToAttention']
    rank_types = ['ffn']
    performance_info = {(op_type, rank_type): [] for op_type in op_types for rank_type in rank_types}

    long_wait_ranks = set()
    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                local_rank_id = op_summary['Device_id'].iloc[0]
                ep_rank_id = local_rank_id
                rank_type = 'ffn'
                for op_type in op_types:
                    op_info = op_summary[op_summary['OP Type'] == op_type]

                    op_execute_time_series = op_info['Task Duration(us)'].tail(50)
                    performance_info[(op_type, rank_type)].append(op_execute_time_series)

                    mean_op_wait_time = op_info['Task Wait Time(us)'].tail(50).mean()
                    if mean_op_wait_time > 5:
                        long_wait_ranks.add(local_rank_id)

    print(f"Task Wait > 5 us local rank id is {sorted(long_wait_ranks)}")

    mean_performances = []
    min_performances = []
    for execute_time_series_list in performance_info.values():
        # print(f'execute_time_series_list: {execute_time_series_list}')
        whole_execute_time_series = (
            pd.concat(execute_time_series_list) 
            if execute_time_series_list
            else pd.Series(dtype=float)
        )
        # print(f'whole_execute_time_series: {whole_execute_time_series}')
        mean_performances.append(whole_execute_time_series.mean())
        min_performances.append(whole_execute_time_series.min())
    return mean_performances + min_performances


def print_ffn2attn_performance(ffn_performance):
    print(f'ffn_performance: {ffn_performance}')
    mean_ffn_performance, min_ffn_performance = ffn_performance

    print(f'[INFO] ffn2attn融合算子Average性能为: {mean_ffn_performance} us\n')
    print(f'[INFO] ffn2attn融合算子Min性能为: {min_ffn_performance} us\n')

def save_ffn_performance_result(test_mode, case_name, op_types, op_performances):
    csv_name = f'{test_mode}_performance_result.csv'
    tille = [
        'op_type', 'case_name',
        'mc2_ffn_average', 'mc2_ffn_average',
        'recordtime'
    ]
    for op_type, op_performance in zip(op_types, op_performances):
        row = [op_type, case_name] + op_performance + [time.strftime("%Y-%m-%d %H:%M:%S")]    
        if not os.path.exists(f"csv_result/{csv_name}"):
            write_row_to_csv(csv_name, tille)
        write_row_to_csv(csv_name, row)

def get_ffn2attention_performance():
    ffn2attention_performance = get_performance(ffn2attn_profiling_path)
    print(f'ffn2attention_performance: {ffn2attention_performance}')

    print_ffn2attn_performance(ffn2attention_performance)

    op_types = ['FFNToAttention']
    save_ffn_performance_result('FFNToAttention', current_case_name, op_types, [ffn2attention_performance])

    return ffn2attention_performance



def get_profiling(loop, path):
    repeat = 1
    warmup = 1
    active = loop // repeat - warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, 
        l2_cache=False, 
        data_simplification=False
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=repeat, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config, )

    return prof


def set_device(rank):
    print()
    torch_npu.npu.set_device(rank % rank_per_dev)
    # if is_full_core:
    #     torch.npu.set_device_limit(rank % rank_per_dev, cube_num=aic_core_num, vector_num=aiv_core_num)
    # else:
    #     torch.npu.set_device_limit(rank % rank_per_dev, cube_num=aic_core_num, vector_num=aiv_core_num)
    #     print(f"当前仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")
        
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank):
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
    print(f"device_{rank} init_process_group success")
    
    print(f"device {rank} 初始化EP域")
    tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
    if rank in ep_ranks_list:
        ep_group = tmp_group
    
    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    print('ep_hcomm_info: '+ep_hcomm_info)
    
    return ep_hcomm_info, ep_group

def get_attn_num_in_server():
    # if attention_worker_num > rank_per_dev:
    #     if server_index > 0:
    #         num = attention_worker_num - rank_per_dev
    #     else:
    #         num = rank_per_dev
    # else:
    #     if server_index > 0:
    #         num = 0
    #     else:
    #         num = attention_worker_num
    attn_server = attention_worker_num // rank_per_dev
    if server_index < attn_server:
        num = rank_per_dev
    elif server_index == attn_server:
        num = attention_worker_num % rank_per_dev
    else:
        num = 0

    return num

def ffn2attn_get_kwargs(
    context, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num,
    attn_rank_table, group, world_size, token_info_table_shape, token_data_shape
):
    x = x.to(input_dtype).npu()
    session_ids = session_ids.to(torch.int32).npu()
    micro_batch_ids = micro_batch_ids.to(torch.int32).npu()
    token_ids = token_ids.to(torch.int32).npu()
    expert_offsets = expert_offsets.to(torch.int32).npu()
    actual_token_num = actual_token_num.to(torch.int64).npu()
    attn_rank_table = attn_rank_table.to(torch.int32).npu()

    return {
        'context': context,
        'x': x,
        'session_ids': session_ids,
        'micro_batch_ids': micro_batch_ids,
        'token_ids': token_ids,
        'expert_offsets': expert_offsets,
        'actual_token_num': actual_token_num,
        'attn_rank_table': attn_rank_table,
        'group': group,
        'world_size': world_size,
        'token_info_table_shape': token_info_table_shape,
        'token_data_shape': token_data_shape,
    }

_ACL_LIB = ctypes.CDLL("libascendcl.so")
_ACL_LIB.aclrtMemcpy.restype = ctypes.c_int32
_ACL_LIB.aclrtMemcpy.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
]
_ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def create_npu_tensor_from_window(shape, dtype, window_addr):
    """从 window 区 device 地址创建 NPU tensor (替代 torchair.llm_datadist.create_npu_tensors)。"""
    numel = 1
    for s in shape:
        numel *= s
    itemsize = torch.empty((), dtype=dtype).element_size()
    nbytes = numel * itemsize
    tensor = torch.empty(shape, dtype=dtype, device='npu')
    ret = _ACL_LIB.aclrtMemcpy(
        tensor.data_ptr(), nbytes, window_addr, nbytes, _ACL_MEMCPY_DEVICE_TO_DEVICE
    )
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpy from window failed, ret={ret}")
    return tensor


def ffn2attn_get_npu(ctx_buffer):
    window_addr = default_window_addr
    if enable_window:
        window_addr = ctx_buffer.get_window_addr()
        if window_addr == 0:
            print('[WARN] get_window_addr returned 0, using default_window_addr')
            window_addr = default_window_addr

    attn_window_states = create_npu_tensor_from_window(
        [micro_batch_num, bs, expert_num_per_token], torch.int32, window_addr
    )
    attn_token_size = attn_window_states.element_size() * attn_window_states.numel()
    attn_token_area_size = (attn_token_size + 511) // 512 * 512

    token_size = hs
    tensor_dtype = model_dtype

    attn_data_tensor = create_npu_tensor_from_window(
        [micro_batch_num, bs, expert_num_per_token, token_size], tensor_dtype,
        window_addr + attn_token_area_size
    )
    return attn_window_states, attn_data_tensor

def ffn2attn_gen_npu(target_func, x, token_ids, actual_token_num, session_ids, micro_batch_ids, expert_offsets):
    print("single_server scene!!!!!")
    rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)) if server_num > 1 else list(range(world_size))
    print(f"rank list is: {rank_list}")

    # 多进程，每个进程调用一张卡执行目标函数
    proc_list = []
    manager = Manager()
    result_dqueue = manager.Queue()
    mp.set_start_method("forkserver", force=True)
    for rank in rank_list:
        # print(rank)
        server_kwargs = {
            "queue": result_dqueue,
            "rank": rank,
            'x': x,
            'token_ids': token_ids,
            'actual_token_num': actual_token_num,
            'session_ids': session_ids,
            'micro_batch_ids': micro_batch_ids,
            'expert_offsets': expert_offsets
        }
        proc = Process(target=target_func, kwargs=server_kwargs)
        # print(rank)
        proc.start()
        proc_list.append(proc)
        # print(rank)

    # print("+++++")

    for proc in proc_list:
        # print("+++++")
        proc.join()

    token_states_cpu_list = [None] * attention_worker_num
    token_data_cpu_list = [None] * attention_worker_num
    if server_num > 1:
        range_num = get_attn_num_in_server()
    else:
        range_num = attention_worker_num

    for attn_rank_id in range(range_num):
        rank_id, rank_output = result_dqueue.get()
        token_states_cpu_list[rank_id] = rank_output[0]
        token_data_cpu_list[rank_id] = rank_output[1]

    # print(rank)

    return token_states_cpu_list, token_data_cpu_list


def ffn2attn_run_npu(queue, rank, x, token_ids, actual_token_num, session_ids, micro_batch_ids, expert_offsets):
    print('================开始设置卡==============')
    set_device(rank)
    print('==============开始建立通信域===================')
    ep_hcomm_info, ep_group= init_hccl_comm(rank)
    print('===================通信域建立完成======================')

    # print(rank)
    # 创建 CommContextManager context (Ascend 950 Channel backend)
    ctx_buffer = cot.FFNToAttentionV2Buffer(
        ep_group, world_size, token_info_table_shape, token_data_table_shape
    )
    print(f'[INFO] rank {rank} context created, ccl_buffer_size={ctx_buffer.ccl_buffer_size}')

    
    if attention_worker_num <= rank < world_size:
        New_rankid = rank - attention_worker_num
        # 计算每个 rank 应该处理的 token 数量
        tokens_per_rank = actual_token_num // ffn_worker_num
        remainder = actual_token_num % ffn_worker_num  # 多余的 token 分配给前几个 rank
        # 根据当前 rank 确定起始和结束位置
        if New_rankid < remainder:
            start_idx = (tokens_per_rank + 1) * New_rankid
            end_idx = start_idx + tokens_per_rank + 1
            # print(f"+++[start_idx {start_idx}] " f"[end_idx {end_idx}] ")
        else:
            start_idx = (tokens_per_rank + 1) * remainder + tokens_per_rank * (New_rankid - remainder)
            end_idx = start_idx + tokens_per_rank
            # print(f"---[start_idx {start_idx}] " f"[end_idx {end_idx}] ")

        start_idx = int(start_idx)
        end_idx = int(end_idx)
        x_local = x[start_idx:end_idx]
        session_ids_local = session_ids[start_idx:end_idx]
        micro_batch_ids_local = micro_batch_ids[start_idx:end_idx]
        token_ids_local = token_ids[start_idx:end_idx]
        expert_offsets_local = expert_offsets[start_idx:end_idx]
        local_token_num = end_idx - start_idx

        print(f"[Rank {rank}] "
                f"[remainder {remainder}] "
                f"[actual_token_num {actual_token_num}] "
                f"[tokens_per_rank {tokens_per_rank}] "
                f"[NewRank {New_rankid}] "
                f"local_token_num = {local_token_num}, "
                f"start_idx = {start_idx}, "
                f"end_idx = {end_idx}")
                
        print(f'[INFO] device_{rank} 构造ffn2attn算子输入数据')
        ffn2attn_kwargs = ffn2attn_get_kwargs(
            context = ctx_buffer.context,
            x = x_local,
            session_ids = session_ids_local,
            micro_batch_ids = micro_batch_ids_local,
            token_ids = token_ids_local,
            expert_offsets = expert_offsets_local,
            actual_token_num = torch.tensor([local_token_num], dtype=torch.int64),
            attn_rank_table = attn_rank_table,
            group = ep_hcomm_info,
            world_size = world_size,
            token_info_table_shape = token_info_table_shape,
            token_data_shape= token_data_table_shape,
        )

        if is_full_core:
            torch.npu.set_stream_limit(torch.npu.current_stream(), aic_core_num, aiv_core_num)
        else:
            # 使用一半的核
            torch.npu.set_stream_limit(torch.npu.current_stream(), aic_core_num, aiv_core_num)
            print(f"当前单算子模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")

        if not is_performance:
            cot.ffn_to_attention_v2(**ffn2attn_kwargs)
        else:
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(loop, ffn2attn_profiling_path) as prof:
                for i in range(loop):
                    _ = cot.ffn_to_attention_v2(**ffn2attn_kwargs)
                    if i == 1:
                        for j in range(100):
                            torch.exp(tensor)
                    prof.step()

        print(f'****************begin cann_ops_transformer.ffn_to_attention_v2{rank}***********')
    
    if rank < attention_worker_num:
        print(f'****************begin cann_ops_transformer.ffn_to_attention_v2 rank:{rank}***********')
        if not is_performance:
            time.sleep(30)
        else:
            time.sleep(50)
        attn_window_states, attn_data_tensor = ffn2attn_get_npu(ctx_buffer)
        # 将张量从 NPU 移到 CPU
        token_states_cpu = attn_window_states.cpu()
        token_data_cpu = attn_data_tensor.cpu()
        print(f'****************rank:{rank}, token_states_cpu : {token_states_cpu}')
        print(f'****************rank:{rank}, token_data_cpu: {token_data_cpu}')
        queue.put([
            rank,
            [token_states_cpu, token_data_cpu]
        ])

    print(f'rank {rank} epid {rank // 1} tpid {rank % 1} npu finished! \n')


def compare_output(str_name, golden_per_card, npu_per_card, compare_dtype, tensor_name, mask_per_card=None):
    def compare(golden, npu, mask, dtype, tensor_name, diffThd=0.001, pctThd=0.001):       
        np_output_npu = np.array(golden.cpu())
        np_output_golden = np.array(npu.cpu())

        if mask is not None:
            np_mask = np.array(mask.cpu())
            np_output_npu = np_output_npu[np_mask]
            np_output_golden = np_output_golden[np_mask]

        print("dtype: ", dtype)
        max_diff_hd = 0.0
        if dtype == torch.int8:
            max_diff_hd = 1.0
        res_output = toolss_attn.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)

        print(f'[INFO] tensor {tensor_name} output测试的精度结果为: {res_output}')
        toolss_attn.write_row_to_csv('attention.csv', [current_case_name, tensor_name, res_output])
        return res_output[0] == 'PASS'
    
    result = {}

    if not compare_per_rank:
        print(f"##################Compare {str_name}##################")
        mask_arg = torch.cat(mask_per_card) if mask_per_card else None
        res = compare(
            torch.cat(golden_per_card).to(compare_dtype), 
            torch.cat(npu_per_card).to(compare_dtype), 
            mask_arg,
            compare_dtype,
            None
        )
        result[tensor_name] = res
    else:
        for i in range(len(golden_per_card)):
            print(f"##################Compare {str_name} in rank",i,"##################")
            mask_arg = mask_per_card[i] if mask_per_card else None
            local_rank_id = i

            res = compare(
                golden_per_card[local_rank_id].to(compare_dtype), 
                npu_per_card[local_rank_id].to(compare_dtype), 
                mask_arg,
                compare_dtype, 
                f"{tensor_name}_{i}"
            )
            result[f"{tensor_name}_{i}"] = res

    return result



if __name__ == '__main__':
    print_setting()

    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')
    os.system(f'rm -rf /root/atc_data/*')
    torch.manual_seed(0)
    random.seed(0)

    if is_ffn2attn:
        x, session_ids, micro_batch_ids, batch_ids, expert_offsets = ffn2attn_gen_inputs()
        token_info_golden, token_data_golden, token_data_mask = ffn2attn_golden(x, session_ids, micro_batch_ids, batch_ids, expert_offsets)
        
        token_states_cpu_list, token_data_cpu_list = ffn2attn_gen_npu(ffn2attn_run_npu, x, batch_ids, actual_token_num, session_ids, micro_batch_ids, expert_offsets)
        
        # print(rank)

        if is_performance:
            ffn2attention_performance = get_ffn2attention_performance()
        
        if server_num > 1:
            compare_num = get_attn_num_in_server()
            golden_start = server_index * rank_per_dev
        else:
            compare_num = attention_worker_num
            golden_start = 0

        if compare_num > 0:
            result_ffn2attn = {}
            res = compare_output(
                'state',
                [token_info_golden[i] for i in range(golden_start, golden_start + compare_num)],
                [token_states_cpu_list[i] for i in range(golden_start, golden_start + compare_num)],
                torch.float32,
                'ffn_token_state',
                None
            )
            result_ffn2attn.update(res)
            res = compare_output(
                'data',
                [token_data_golden[i] for i in range(golden_start, golden_start + compare_num)],
                [token_data_cpu_list[i] for i in range(golden_start, golden_start + compare_num)],
                torch.float32,
                'ffn_token_data',
                [token_data_mask[i] for i in range(golden_start, golden_start + compare_num)]
            )
            result_ffn2attn.update(res)

            print("\n##################result_ffn2attn##################")
            print("result_ffn2attn: ", result_ffn2attn)
            all_result = all(result_ffn2attn.values())
            if all_result:
                print("ffn2attn test Success.")
            else:
                print("ffn2ffn test Failed.")
        else:
            time.sleep(30)
            print("ffn2attn test Success.")

        if is_performance:
            print_ffn2attn_performance(ffn2attention_performance)

        print('跑完了')