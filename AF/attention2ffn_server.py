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
import torchair
import toolss_attn

##################### 测试开关 #####################
current_case_name = 'Test_AttentionToFFN_16die_ID030'

is_attn2ffn = 1
is_ffn_batching = 0         # 是否调用算子清理FFN状态位

# 性能检测相关
is_performance = 0
loop = 105              # 性能检测时的性能采集轮数
attn2ffn_profiling_path = './Profiling_Result_attn2ffn'

# 量化相关
quant_mode = 0
is_quant = (quant_mode > 0)
is_attn2ffn_scales = 0

# 图模式相关
graph_type = 1

# 参数设置
is_mask = 0
sync_flag = 0

# 自定义windows区conda
enable_window = True                        # 是否开启自定义window区功能
# enable_window = False
default_window_size = 1024 * 1024 * 200             # 200M 空间
default_window_addr = 0x12c101800000        # 默认win区地址

# 分核配置
is_full_core = 1
core_num = 24
aic_core_num = core_num                                         # 非全核模式使用的aic核数
aiv_core_num = core_num * 2                                     # 非全核模式使用的aiv核数
is_topk_balance = True                                          # 是否生成均衡的topK

##################### 模型相关设置 #####################
# shape
bs = 512
h = 8192
scale_pad = 128                             # 用于存放量化参数
hs = h + scale_pad                          # 量化时h的长度
k = 8
X = 1
L = 1
micro_batch_num = 1                         # micro batch 数量
default_micro_batch_id = 0                  # 默认micro batch id
layer_num = 1                               # 模型层数
default_layer_id = 0                        # 默认模型层id
ffn_start_rank_id = 0                       # ffn worker 起始id

server_num = 1
server_index = 1
rank_per_dev = 16                                               # 每个设备（如 GPU）上运行的 rank 数量（这里可能指逻辑 rank，用于模拟多进程）                       
world_size = rank_per_dev * server_num
ep_world_size = world_size                                       # Expert Parallel 的 world size（专家并行规模）

shared_expert_num = 2
rank_num_per_shared_expert = 1
shared_expert_rank_num_ffn = shared_expert_num * rank_num_per_shared_expert     # 共享专家（shared expert）使用的 rank 数量

moe_rank_ffn_num = 10
moe_expert_per_rank = 64
moe_expert_num = moe_rank_ffn_num * moe_expert_per_rank                     # moe专家 总数

ffn_worker_num = shared_expert_rank_num_ffn + moe_rank_ffn_num              # ffn rank数
attention_worker_num = world_size - shared_expert_rank_num_ffn - moe_rank_ffn_num
expert_num_per_token = k + shared_expert_num                                # topK + shared 个专家

##################### 数据格式相关 #####################
out_dtype = 0
input_dtype = torch.bfloat16 if out_dtype == 0 else torch.float16

dtype_mapping = {
    "float": torch.float,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}
dtype_str = "bfloat16" if out_dtype == 0 else "float16"
set_dtype = os.getenv("DTYPE", dtype_str)
if is_quant and is_attn2ffn:
    set_dtype = "int8"
model_dtype = dtype_mapping.get(set_dtype, torch.bfloat16)

##################### 算子参数相关 #####################
# Attention → FFN 的 token 路由信息表（每个 attention worker 发送）
# 最后一维：[batch_size, valid_token_num, token_ids..., expert_ids...]
ffn_token_info_table_shape = [attention_worker_num, micro_batch_num, 1+1+bs*expert_num_per_token]

# 实际 token 数据（embedding）
# [11, 1, 16, 9, 7168] → 每个 token 被复制到 k+1=9 个位置, 量化时[11, 1, 16, 9, 7168 + 128], 用于存放量化参数
if quant_mode == 0:
    token_data_size = (h * 2 + 512 - 1) // 512 * 512
else:
    token_data_size = (h + 128 + 512 - 1) // 512 * 512
ffn_token_data_shape = [attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_data_size if is_quant else token_data_size // 2]

# Attention 侧的 token 信息
attn_token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]

# win 区需要的大小
window_size_state = attention_worker_num * micro_batch_num * (1+1+bs*expert_num_per_token) * 4
window_size_data = attention_worker_num * micro_batch_num * bs * expert_num_per_token * token_data_size

valid_window_size = (window_size_state + 512 - 1) // 512 * 512 + window_size_data

window_size = valid_window_size
# window_size = default_window_size
# print("window_size=", window_size/1024/1024, "MB")

if (valid_window_size >= default_window_size):
    os.environ['HCCL_BUFFSIZE'] = '4000'

# window_size = default_window_size

##################### ffn_to_attention 算子参数相关 #####################
actual_token_num = bs * micro_batch_num * attention_worker_num * expert_num_per_token  # token数量

# 算子参数
token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]
token_data_table_shape = [micro_batch_num, bs, expert_num_per_token, h]
attn_rank_table = torch.arange(attention_worker_num, dtype=torch.int32)

##################### 通信参数相关 #####################
master_ip = '71.20.45.104' if server_num > 1 else '127.0.0.1'
# [[0,1,2,...,15]]
ep_ranks_list = list(range(world_size))

##################### 测试脚本执行相关 #####################
compare_per_rank = 1                 # 每个 rank 会单独计算并比较自己的结果
is_check_setting = False              # 检查参数设置

def print_setting():
    ''' 打印相关参数
    '''
    print(f"{bs=}")
    print(f"{k=}")
    print(f"{h=}")
    print(f"{is_performance=}")
    print(f"{is_mask=}")
    print(f"{sync_flag=}")
    print(f"{quant_mode=}")
    print(f"{is_quant=}")
    print(f"{is_attn2ffn_scales=}")
    print(f"{graph_type=}")
    print(f"{enable_window=}")
    print(f"{moe_expert_num=}")
    print(f"{moe_rank_ffn_num=}")
    print(f"{shared_expert_num=}")
    print(f"{shared_expert_rank_num_ffn=}")
    print(f"{is_full_core=}")
    print(f"{ffn_token_info_table_shape=}: ")
    print(f"{ffn_token_data_shape=}")

def check_setting():
    def error_log_and_exit(log_info, exit_code=1):
        print(f"ERROR: {log_info}")
        exit(exit_code)

    if bs < 1:
        error_log_and_exit("bs 应大于等于 1")

    if h < 1:
        error_log_and_exit("h 应大于等于 1")

    if k < 1:
        error_log_and_exit("k 应大于等于 1")

    if moe_expert_num < 1:
        error_log_and_exit("moe 专家数应大于等于 1")

    if k > moe_expert_num:
        error_log_and_exit("k 不能大于 moe_expert_num")

    if moe_expert_num % moe_rank_ffn_num != 0:
        error_log_and_exit("moe_expert_num 必须是 moe_rank_ffn_num 的整数倍")

    if quant_mode == 1:
        error_log_and_exit("当前不支持静态量化, quant_mode 合法值仅有 0 和 2")



class ATTENTION_TO_FFN_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self, x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size, ffn_token_info_table_shape,
        ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, scales, active_mask, quant_mode, sync_flag, ffn_start_rank_id
    ):
        output = torch_npu.npu_attention_to_ffn(
            x=x,
            session_id=session_id,
            micro_batch_id=micro_batch_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
            expert_rank_table=expert_rank_table,
            group=group,
            world_size=world_size,
            ffn_token_info_table_shape=ffn_token_info_table_shape,
            ffn_token_data_shape=ffn_token_data_shape,
            attn_token_info_table_shape=attn_token_info_table_shape,
            moe_expert_num=moe_expert_num,
            scales = scales,
            active_mask = active_mask,
            quant_mode = quant_mode,
            sync_flag = sync_flag,
            ffn_start_rank_id = ffn_start_rank_id
        )
        return output

class FFN_WORKER_BATCHING_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self, schedule_context, expert_num, max_out_shape, token_dtype, need_schedule, layer_num
    ):
        output = torch_npu.npu_ffn_worker_batching(
            schedule_context = schedule_context,
            expert_num = expert_num,
            max_out_shape = max_out_shape,
            token_dtype = token_dtype,
            need_schedule = need_schedule,
            layer_num = layer_num
        )
        return output


def compile_model(model):
    ''' 图模式编译模型
    '''
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        exit(1)

    print(f"------------------graph_type: {graph_type}------------------",)

    from torchair.configs.compiler_config import CompilerConfig
    if is_full_core:
        compiler_config = None
    else:
        # 使用一半的核
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
        print(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)

    if graph_type == 1:  
        # 传统入图模式，静态shape+在线编译场景
        print('graph_type=1, 测试场景: 传统入图模式, 静态shape+在线编译场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:  
        # ACLNN入图模式，动态shape+二进制
        print('graph_type=2, 测试场景: ACLNN入图模式, 动态shape+二进制场景')
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)

    return compiled_model


def attn2ffn_gen_inputs():
    '''生成attention_to_ffn算子需要的参数
    '''
    def gen_x():
        '''生成 Attention 输出的激活值，形状 [11, 1, 16, 7168]，值在 [-1024, 1024] 之间（模拟真实分布）
        '''
        return torch.empty(size=[attention_worker_num, X, bs, h], dtype=input_dtype).uniform_(-1024, 1024)

    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]

    def gen_balanced_expert_ids_element(round_index, moe_expert_num_re):
        return torch.arange(round_index, round_index + moe_expert_num_re, dtype=torch.int32) % (moe_expert_num_re)

    def gen_expert_ids():
        '''[11, 1, 16, 8], 为所有 Attention worker（11个） 的每个 token 生成 expert IDs, 用于路由
        '''
        if is_topk_balance:
            ep_expert_ids = [gen_balanced_expert_ids_element(round_index, moe_expert_num) \
                                for round_index in range((X * bs * attention_worker_num * k) // moe_expert_num + 1)]
        else:
            ep_expert_ids = [gen_random_token_expert_ids(k, 0, moe_expert_num) for _ in range(X * bs * attention_worker_num)]

        return torch.cat(ep_expert_ids)[:X * bs * attention_worker_num * k].view([attention_worker_num, X, bs, k])

    def gen_scales():
        '''[k + 1, h] = [9, 7168], 8个MoE专家 + 1个共享专家
        '''
        if is_quant and is_attn2ffn_scales:
            scales = torch.empty(size=[L, shared_expert_num + moe_expert_num, h], dtype=torch.float32).uniform_(-1, 1)
        else:
            # 使用空张量torch.tensor([])会被dispatch算子的tiling检查拦截，只能用None，无法保证数据类型一致
            scales = None
        return scales
    
    def gen_expert_rank_table():
        '''[1, 9, max], 最后一维[k,rank_0,eid_0,rank_2,eid_2,...,pad...], 用于表示每层的专家部署在那几个卡上
        '''
        data = []
        max_len = 0
        valid_rank_id = [i for i in range(moe_rank_ffn_num)] * moe_expert_per_rank
        valid_exp_id = list(range(moe_expert_num))
        cards = {}
        for i in range(moe_rank_ffn_num):
            start = i * moe_expert_per_rank
            end = start + moe_expert_per_rank
            cards[i] = valid_exp_id[start:end]

        for l in range(layer_num):
            layer_e1 = []

            for e in range(moe_expert_num):
                cards_h = copy.deepcopy(cards)
                valid_rank_id_h = copy.deepcopy(valid_rank_id)

                i = random.randint(1, moe_rank_ffn_num)
                item = [i]
                for index_i in range(i):
                    num_rank_id = random.choice(valid_rank_id_h)
                    item.append(num_rank_id)
                    valid_rank_id_h.remove(num_rank_id)

                    num_exp_id = random.choice(cards_h[num_rank_id])
                    item.append(num_exp_id)
                    cards_h[num_rank_id].remove(num_exp_id)

                # 分配的卡数量、分配的卡id，分配到该卡的专家id
                layer_e1.append(item)  # item对应  这个 moe专家部署在多少张上、分别部署在哪些卡、及卡上的哪个专家id？？？
                max_len = max(max_len, len(item))
            # 共享专家
            for i in range(shared_expert_num):
                item = [rank_num_per_shared_expert]
                for j in range(rank_num_per_shared_expert):
                    item.append(moe_rank_ffn_num + i * rank_num_per_shared_expert + j)
                    item.append(moe_expert_num + i)
                layer_e1.append(item)
                max_len = max(max_len, len(item))
                # layer_e1.append([shared_expert_rank_num_ffn, ffn_worker_num - 1, moe_expert_num + i])  # 共享专家
            data.append(layer_e1)

        # Padding
        padded_data = []
        for layer in data:
            padded_layer = []
            for item in layer:
                item_padded = item + [0]*(max_len - len(item))
                padded_layer.append(item_padded)
            padded_data.append(padded_layer)

        tensor_data = torch.tensor(padded_data)  # shape: (L, E1, max_len)  层数,每层专家数，最大项目长度
        # print("-------- expert_rank_table:", tensor_data.size(), tensor_data)
        return tensor_data
    
    def gen_active_mask():
        '''生成 mask，形状 [attention_worker_num, X, bs]        [11, 1, 16]?
        '''
        min_real_token_num = 0 if is_mask else 1
        active_mask_list = []
        for _ in range(attention_worker_num):
            real_token_num = torch.randint(min_real_token_num, bs + 1, (1,)).item() if is_mask else bs
            active_mask = torch.cat((torch.ones(real_token_num), torch.zeros(bs - real_token_num))).to(torch.bool)
            active_mask = active_mask.unsqueeze(0)  # 形状变为 [1, bs]
            active_mask = active_mask.repeat(X, 1)  # 形状变为 [X, bs]
            active_mask_list.append(active_mask)
        
        # 将列表转换为张量并堆叠
        active_mask_list = torch.stack(active_mask_list)
        # print("-------- active_mask:", active_mask_list.size(), active_mask_list)
        return active_mask_list

    x = gen_x()
    expert_ids = gen_expert_ids()
    expert_rank_table = gen_expert_rank_table()
    scales = gen_scales()
    active_mask_list = gen_active_mask()
    # print("-------- expert_ids:", expert_ids.size(), expert_ids)

    return x, expert_ids, expert_rank_table, scales, active_mask_list

def attn2ffn_golden(x_golden, expert_ids_golden, expert_rank_table, scales, active_mask_list):
    # 初始化
    # token_size = hs if is_quant else h
    token_size = token_data_size if is_quant else token_data_size // 2

    ffn_token_state = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, 1 + 1 + bs * expert_num_per_token)  # FFN层的状态信息
    ffn_token_data = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size)  # FFN层的输入数据
    ffn_quant_scales = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, expert_num_per_token)  # FFN层的量化参数
    token_data_mask = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size, dtype=torch.bool)   # 用于结果比较时，比较有效位置上的数据
    attn_token_state = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num_per_token)  # FFN层的状态信息

    def load_banance(session_id, layer_id, expert_id):
        count = expert_rank_table[layer_id, expert_id, 0]
        idx = session_id % count
        dst_rank_id = expert_rank_table[layer_id, expert_id, 1 + idx * 2]
        dst_exp_id = expert_rank_table[layer_id, expert_id, 2 + idx * 2]
        return (dst_rank_id, dst_exp_id)
    
    def quant_process(y, scales):
        y = y.to(torch.float32) * scales           # 首先把需要量化的值乘上一个scales

        y_max, _ = y.abs().max(dim=-1)              # 量化公式第一步求绝对值的最大值
        y_max[y_max == 0] = 1.0                   # 防止除0
        dynamic_scales = torch.tensor(127.0, dtype=torch.float32) / y_max
        y = (y * dynamic_scales.view(-1, 1)).round().to(torch.int8)
        scale = 1.0 / dynamic_scales
        return y, scale.flatten()

    # 算子部分
    def att2ffn(row_idx, x_i, expert_id, attn_id):
        nonlocal ffn_token_data, ffn_token_state, token_data_mask
        scales_i = None
        hAlign = (h + 32 -1) // 32 * 32
        # 处理moe专家
        # topk_id 为索引
        for topk_id, expert_i in enumerate(expert_id):
            rank_id, exp_id = load_banance(attn_id, default_layer_id, expert_i)

            token_data_mask[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = torch.ones(h, dtype=torch.bool)
            if is_quant:
                scales_i = scales[default_layer_id, expert_i] if is_attn2ffn_scales else 1
                quant_data, dynamic_scales = quant_process(x_i, scales_i)
                ffn_token_data[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = quant_data
                ffn_quant_scales[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id] = dynamic_scales
            else:
                ffn_token_data[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = x_i

            # 标志位
            ffn_token_state[rank_id, attn_id, default_micro_batch_id, 0] = 1
            ffn_token_state[rank_id, attn_id, default_micro_batch_id, 1] = default_layer_id
            ffn_token_state[rank_id, attn_id, default_micro_batch_id, (row_idx * expert_num_per_token + topk_id + 2)] = exp_id.float()

        # 处理共享专家
        for i in range(shared_expert_num):
            shared_rank_id, shared_exp_id = load_banance(attn_id, default_layer_id, moe_expert_num + i)
            ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id, 0] = 1
            ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id, 1] = default_layer_id
            ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id, (row_idx * expert_num_per_token + k + i + 2)] = shared_exp_id.float()

            token_data_mask[shared_rank_id, attn_id, default_micro_batch_id, row_idx, k + i, :h] = torch.ones(h, dtype=torch.bool)
            if is_quant:
                scales_i = scales[default_layer_id, moe_expert_num + i] if is_attn2ffn_scales else 1
                quant_data, dynamic_scales = quant_process(x_i, scales_i)
                ffn_token_data[shared_rank_id, attn_id, default_micro_batch_id, row_idx, k + i, :h] = quant_data
                ffn_quant_scales[shared_rank_id, attn_id, default_micro_batch_id, row_idx, k + i] = dynamic_scales
            else:
                ffn_token_data[shared_rank_id, attn_id, default_micro_batch_id, row_idx, k + i, :h] = x_i

    for attn_id in range(attention_worker_num):
        for token_id in range(bs):
            if(active_mask_list[attn_id, X - 1, token_id]):
                att2ffn(token_id, x_golden[attn_id, X - 1, token_id], expert_ids_golden[attn_id, X - 1, token_id], attn_id)
            else:
                attn_token_state[attn_id, X - 1, token_id, : ] = torch.ones(expert_num_per_token)

    if (sync_flag == 0):
        ffn_token_state[:, :, :, 0] = 1     # 同步即所有ffn节点flag位均置1
    
    # for i in range(ffn_worker_num):
        # print(f"------rank {i} golden ffn_token_state:", ffn_token_state[i].size(), ffn_token_state[i])
        # print(f"------rank {i} golden ffn_token_data:", ffn_token_data[i].size(), ffn_token_data[i])

    return ffn_token_data, token_data_mask, ffn_quant_scales, ffn_token_state, attn_token_state

def write_row_to_csv(csv_name: str, row_content: list):
    if not os.path.exists("csv_result"):
        os.makedirs("csv_result")
    with open(f'csv_result/{csv_name}', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)


def get_performance(profiling_path):
    # op_types = path_to_op_types[profiling_path]
    op_types = ['AttentionToFFN']
    rank_types = ['attention']
    performance_info = {(op_type, rank_type): [] for op_type in op_types for rank_type in rank_types}

    long_wait_ranks = set()
    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                local_rank_id = op_summary['Device_id'].iloc[0]
                ep_rank_id = local_rank_id
                rank_type = 'attention'
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
        whole_execute_time_series = (
            pd.concat(execute_time_series_list) 
            if execute_time_series_list
            else pd.Series(dtype=float)
        )
        print(f'whole_execute_time_series: {whole_execute_time_series}')
        mean_performances.append(whole_execute_time_series.mean())
        min_performances.append(whole_execute_time_series.min())
    return mean_performances + min_performances


def print_attn2ffn_performance(attention_performance):
    print(f'attention_performance: {attention_performance}')
    mean_attention_performance, min_attention_performance = attention_performance

    print(f'[INFO] attn2ffn融合算子Average性能为: {mean_attention_performance} us\n')
    print(f'[INFO] attn2ffn融合算子Min性能为: {min_attention_performance} us\n')


def save_performance_result(test_mode, case_name, op_types, op_performances):
    csv_name = f'{test_mode}_performance_result.csv'
    tille = [
        'op_type', 'case_name',
        'mc2_attention_average', 'mc2_attention_min',
        'recordtime'
    ]

    for op_type, op_performance in zip(op_types, op_performances):
        row = [op_type, case_name] + op_performance + [time.strftime("%Y-%m-%d %H:%M:%S")]    
        if not os.path.exists(f"csv_result/{csv_name}"):
            write_row_to_csv(csv_name, tille)
        write_row_to_csv(csv_name, row)


def get_attn2ffn_performance():
    attn2ffn_performance = get_performance(attn2ffn_profiling_path)
    print(f'attn2ffn_performance: {attn2ffn_performance}')

    print_attn2ffn_performance(attn2ffn_performance)

    op_types = ['AttentionToFFN']
    save_performance_result('AttentionToFFN', current_case_name, op_types, [attn2ffn_performance])

    return attn2ffn_performance


def get_profiling(loop, path):
    repeat = 1
    warmup = 5
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
    # print()
    torch_npu.npu.set_device(rank % rank_per_dev)
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


def dispatch_gen_expert_ids():
    ''' 生成dispatch的expert_ids \n
        每个token选择的专家, [bs, k] \n
        moe_expert_num = k, 选择全部专家 \n
    '''
    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]                        

    
    ep_expert_ids = [gen_random_token_expert_ids(k, 0, moe_expert_num) for _ in range(bs)] 
    
    return torch.cat(ep_expert_ids)[:bs * k].view([bs, k])


def align_up(num, align):
    '''向上取整
    '''
    return ((num + align - 1) // align) * align


def dispatch_get_kwargs(
    rank, group_ep
):
    ''' dispatch算子的入参
    '''
    x_cpu = torch.zeros(size=[bs, h], dtype=input_dtype)
    x = x_cpu.to(input_dtype).npu()
    expert_dis_ids = dispatch_gen_expert_ids().to(torch.int32).npu()
    
    return {
        'x': x,
        'expert_ids': expert_dis_ids,
        'group_ep': group_ep,
        'group_tp': group_ep,
        'ep_rank_id': rank,
        'tp_rank_id': 0,
        'ep_world_size': 16,
        'tp_world_size': 1,
        'shared_expert_num': 1,
        'shared_expert_rank_num': 1,
        'moe_expert_num': 15,
        'scales': None,
        'quant_mode': 0,
        'global_bs': 0,
    }


def attn2ffn_get_kwargs(
    x, session_id, micro_batch_id, layer_id, expert_ids,expert_rank_table, group, world_size, ffn_token_info_table_shape, 
    ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num,scales, active_mask, quant_mode, sync_flag, ffn_start_rank_id
):
    '''attn2ffn算子的入参
    '''
    x = x.to(input_dtype).npu()
    session_id = session_id.to(torch.int32).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    micro_batch_id = micro_batch_id.to(torch.int32).npu()
    layer_id = layer_id.to(torch.int32).npu()
    expert_rank_table = expert_rank_table.to(torch.int32).npu()
    if is_quant and is_attn2ffn_scales:
        scales = scales.to(torch.float32).npu()
    active_mask = active_mask.to(torch.bool).npu() if is_mask else None

    return {
        'x': x,
        'session_id': session_id,
        'micro_batch_id': micro_batch_id,
        'layer_id': layer_id,
        'expert_ids':expert_ids,
        'expert_rank_table': expert_rank_table,
        'group': group,
        'world_size': world_size,
        'ffn_token_info_table_shape': ffn_token_info_table_shape,
        'ffn_token_data_shape': ffn_token_data_shape,
        'attn_token_info_table_shape': attn_token_info_table_shape,
        'moe_expert_num': moe_expert_num,
        'scales': scales,
        'active_mask': active_mask,
        'quant_mode': quant_mode,
        "sync_flag":sync_flag,
        "ffn_start_rank_id":ffn_start_rank_id,
    }


def ffn_batching_get_kwargs(
    rank, window_addr, window_size
):
    '''ffn_worker_batching算子的入参
    '''
    # token_size = hs if is_quant else h
    token_size = token_data_size if is_quant else token_data_size // 2
    max_out_shape = [attention_worker_num, bs, (k + shared_expert_num), token_size] #{Y, H} Y=A*BS*Local_E
    if quant_mode == 2:
        attn_to_ffn_token_size = align_up(hs, 512)  # 512:为了对齐 4: 量化scale fp32
        token_dtype = 2
    else:
        attn_to_ffn_token_size = align_up(h * 2, 512)
        token_dtype = 1 if input_dtype == torch.bfloat16 else 0
    ffn_to_attn_token_size = align_up(h * 2, 512)
    expert_num = moe_expert_num + shared_expert_rank_num_ffn

    context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode=0, session_num=attention_worker_num,
        micro_batch_num=micro_batch_num, micro_batch_size=bs, selected_expert_num=k+shared_expert_num,
        expert_num=expert_num, attn_to_ffn_token_size=attn_to_ffn_token_size, ffn_to_attn_token_size=ffn_to_attn_token_size, 
        ffn_window=window_addr, ffn_window_size=window_size)
    schedule_context = context_holder.get_schedule_context_tensor()
    
    batching_kwargs = {
        "schedule_context": schedule_context,
        "expert_num": moe_expert_per_rank if rank < moe_rank_ffn_num else 1,
        "max_out_shape": max_out_shape,
        "token_dtype": token_dtype,
        "need_schedule": 1,
        "layer_num": layer_num
    }

    return batching_kwargs


def attn2ffn_get_npu(ep_group):
    ''' 拿回 FFN 计算结果 \n
        ffn_window_states: [11, 1, 1+1+16*9], win区状态 \n
        ffn_data_tensor: [11, 1, 16, 9, 7168或者7168+128(量化)], win区数据 \n
    '''
    window_addr = default_window_addr
    if enable_window:
        window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()

    # 状态, [11, 1, 1+1+16*9]
    npu_state_tensor = torchair.llm_datadist.create_npu_tensors(
        [attention_worker_num, micro_batch_num, 1 + 1 + bs * expert_num_per_token], torch.int32, [window_addr]
    )
    ffn_window_states = npu_state_tensor[0]

    ffn_token_size = ffn_window_states.element_size() * ffn_window_states.numel()
    ffn_token_area_size = (ffn_token_size + 511) // 512 * 512     # +511使得需要对齐的数不是512的倍数的话，让结果推动到下一个倍数，例如513应该对齐到1024，513+511=1024,1024//512=2, 2*512 = 1024,结果正确
    # token_size = hs if is_quant else h
    token_size = token_data_size if is_quant else token_data_size // 2

    # 数据, [11, 1, 16, 9, 7168]
    ffn_data_tensors = torchair.llm_datadist.create_npu_tensors(
        [attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size], model_dtype, [window_addr + ffn_token_area_size]
    )
    ffn_data_tensor = ffn_data_tensors[0]

    # 量化系数scales
    if is_quant:
        ffn_data_tensors_float = torchair.llm_datadist.create_npu_tensors(
            [attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size//4], torch.float32, [window_addr + ffn_token_area_size]
        )
        ffn_data_tensor_float = ffn_data_tensors_float[0]
        index = align_up(h, 32) // 4
        ffn_scales_tensor = ffn_data_tensor_float[:,:,:,:,index]

    else:
        ffn_scales_tensor = torch.ones([attention_worker_num, micro_batch_num, bs, expert_num_per_token], dtype=torch.float32)

    return ffn_window_states, ffn_data_tensor, ffn_scales_tensor


def set_windows(rank, ep_group, ep_hcomm_info):
    # 是否开启window自定义功能
    if enable_window:
        if rank >= ffn_worker_num:
            # 当前 rank 属于后半部分：attention 卡 需要与前 ffn_worker_num 个 rank （ffn卡）通信
            target_ranks = list(range(ffn_worker_num))
        else:
            # 当前 rank 属于前 ffn_worker_num 个：FFN卡，需要与后面的 rank （attention卡）通信
            target_ranks = list(range(ffn_worker_num, world_size))

        print(f'target_ranks: {target_ranks}')
        ep_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size, target_ranks)
    else:
        # all2all通信, 握手
        print(f'[INFO] device_{rank} 构造dispatch算子输入数据')
        dispatch_kwargs = dispatch_get_kwargs(
            rank=rank, 
            group_ep=ep_hcomm_info,
        )
        output_dispatch = torch_npu.npu_moe_distribute_dispatch(**dispatch_kwargs)
        out1, _, _, _, _, _, _ = output_dispatch


def attn2ffn_run_npu(queue, rank, x, expert_ids, expert_rank_table, scales, active_mask_list):
    set_device(rank)
    ep_hcomm_info, ep_group= init_hccl_comm(rank)
    set_windows(rank, ep_group, ep_hcomm_info)

    # attention 算子 [5, ..., 15]
    if rank >= ffn_worker_num:
        attn2ffn_kwargs = attn2ffn_get_kwargs(
        x=x[rank-ffn_worker_num],
        session_id=torch.tensor([rank-ffn_worker_num], dtype=torch.int32),
        micro_batch_id=torch.tensor([default_micro_batch_id], dtype=torch.int32),
        layer_id=torch.tensor([default_layer_id], dtype=torch.int32),
        expert_ids=expert_ids[rank-ffn_worker_num],
        expert_rank_table=expert_rank_table,
        group=ep_hcomm_info,
        world_size=world_size,
        ffn_token_info_table_shape=ffn_token_info_table_shape,
        ffn_token_data_shape=ffn_token_data_shape,
        attn_token_info_table_shape=attn_token_info_table_shape,
        moe_expert_num=moe_expert_num,
        scales = scales,
        active_mask = active_mask_list[rank-ffn_worker_num],
        quant_mode = quant_mode,
        sync_flag = sync_flag,
        ffn_start_rank_id = ffn_start_rank_id,
        )   # 获取Attention节点入参

        if graph_type == 0 and not is_full_core:
            torch.npu.set_stream_limit(torch.npu.current_stream(), aic_core_num, aiv_core_num)
            print(f"当前单算子模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")

        if is_performance and not is_ffn_batching:
            if graph_type == 0:
                tensor = torch.ones(10000, 10000).to(torch.float16).npu()
                prof = get_profiling(loop, attn2ffn_profiling_path)
                prof.start()
                for i in range(loop):
                    torch_npu.npu_attention_to_ffn(**attn2ffn_kwargs)
                    if i == 1:
                        for j in range(100):
                            torch.exp(tensor)
                    prof.step()
                prof.stop()
            else:
                model = compile_model(ATTENTION_TO_FFN_GRAPH_Model())
                prof = get_profiling(loop, attn2ffn_profiling_path)
                prof.start()
                for i in range(loop):
                    model(**attn2ffn_kwargs)
                    prof.step()
                prof.stop()
        else:
            if graph_type == 0:
                torch_npu.npu_attention_to_ffn(**attn2ffn_kwargs)
            else:
                model = compile_model(ATTENTION_TO_FFN_GRAPH_Model())
                model(**attn2ffn_kwargs)
            if is_performance:
                time.sleep(50)
        # attn状态
        if is_mask:
            time.sleep(5)
            window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()
            npu_state_tensor = torchair.llm_datadist.create_npu_tensors(
                [micro_batch_num, bs, expert_num_per_token], torch.int32, [window_addr]
            )
            attn_window_states = npu_state_tensor[0]
            attn_state_cpu = attn_window_states.cpu()
            queue.put([
                rank, 
                [attn_state_cpu]
            ])
        print(f'rank {rank} attention finished! \n')

    # ffn的rank, [0, ..., 4]
    if rank < ffn_worker_num:
        time.sleep(50)
        ffn_window_states, ffn_data_tensor, ffn_scales_tensor = attn2ffn_get_npu(ep_group)
        # 将张量从 NPU 移到 CPU
        token_states_cpu = ffn_window_states.cpu()
        token_data_cpu = ffn_data_tensor.cpu()
        ffn_scales_cpu = ffn_scales_tensor.cpu()
        # print(f"-----rank {rank} token_state:", token_states_cpu.size(), token_states_cpu)
        # print(f"-----rank {rank} token_data:", token_data_cpu.size(), token_data_cpu)
        queue.put([
            rank, 
            [token_states_cpu, token_data_cpu, ffn_scales_cpu]
        ])
        print(f'FFN rank {rank} finished! \n')
        if is_ffn_batching and is_performance:
            # time.sleep(1)
            # 调用FFNWorkerBatching 参数配置
            window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()
            batching_kwargs = ffn_batching_get_kwargs(rank, window_addr, window_size)
            print(f'rank {rank} get ffn batching complete! \n')

    if is_performance and is_ffn_batching:
        print(f'start rank {rank} get performance! \n')

        if graph_type == 0:
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            prof = get_profiling(loop, attn2ffn_profiling_path)
            prof.start()
            for i in range(loop):
                if rank >= ffn_worker_num:
                    torch_npu.npu_attention_to_ffn(**attn2ffn_kwargs)
                    # time.sleep(1)
                if rank < ffn_worker_num:
                    batching_output = torch_npu.npu_ffn_worker_batching(**batching_kwargs)
                if i == 1:
                    for j in range(100):
                        torch.exp(tensor)

                prof.step()
            prof.stop()
        else:
            model = compile_model(ATTENTION_TO_FFN_GRAPH_Model())
            prof = get_profiling(loop, attn2ffn_profiling_path)
            prof.start()
            for i in range(loop):
                if rank >= ffn_worker_num:
                    model(**attn2ffn_kwargs)
                    # time.sleep(1)
                if rank < ffn_worker_num:
                    batching_output = torch_npu.npu_ffn_worker_batching(**batching_kwargs)
                prof.step()
            prof.stop()

    # torch.npu.synchronize()
    print(f'rank {rank} npu finished! \n')


def attn2ffn_gen_npu(target_func, x, expert_ids, expert_rank_table, scales, active_mask_list):
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
            'expert_ids': expert_ids, 
            'expert_rank_table': expert_rank_table,
            'active_mask_list': active_mask_list,
            'scales': scales,
        }
        proc = Process(target=target_func, kwargs=server_kwargs)
        proc.start()
        proc_list.append(proc)
    
    for proc in proc_list:
        proc.join()

    token_states_cpu_list = [None] * ffn_worker_num
    token_data_cpu_list = [None] * ffn_worker_num
    quant_scales_cpu_list = [None] * ffn_worker_num
    attn_token_states_cpu_list = [None] * attention_worker_num

    if server_num > 1:
        ffn_num = get_ffn_num_in_server()
    else:
        ffn_num = ffn_worker_num

    if is_mask:
        for _ in range(rank_per_dev):
            rank_id, rank_output = result_dqueue.get()
            if rank_id < ffn_num:
                token_states_cpu_list[rank_id] = rank_output[0]
                token_data_cpu_list[rank_id] = rank_output[1]
                quant_scales_cpu_list[rank_id] = rank_output[2]
            else:
                attn_token_states_cpu_list[rank_id - ffn_worker_num] = rank_output[0]
    else:
        for _ in range(ffn_num):
            rank_id, rank_output = result_dqueue.get()
            token_states_cpu_list[rank_id] = rank_output[0]
            token_data_cpu_list[rank_id] = rank_output[1]
            quant_scales_cpu_list[rank_id] = rank_output[2]

    return token_states_cpu_list, token_data_cpu_list, quant_scales_cpu_list, attn_token_states_cpu_list


def get_ffn_num_in_server():
    ffn_server = ffn_worker_num // rank_per_dev
    if server_index < ffn_server:
        num = rank_per_dev
    elif server_index == ffn_server:
        num = ffn_worker_num % rank_per_dev
    else:
        num = 0

    return num


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

            # print(f"-----rank {i} diff", sum(npu_per_card[local_rank_id] - golden_per_card[local_rank_id]))

    return result



if __name__ == '__main__':
    # 测试前准备
    print_setting()

    if is_check_setting:
        check_setting()

    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')
    os.system(f'rm -rf /root/atc_data/*')
    torch.manual_seed(0)
    random.seed(0)
    
    if is_attn2ffn:
        (x, expert_ids, expert_rank_table, scales, active_mask_list) = attn2ffn_gen_inputs()
        token_data_golden, token_data_mask, quant_scales_golden, token_info_golden, attn_info_golden = attn2ffn_golden(x, expert_ids, expert_rank_table, scales, active_mask_list)
        
        token_states_cpu_list, token_data_cpu_list, quant_scales_cpu_list, attn_token_states_cpu_list = attn2ffn_gen_npu(attn2ffn_run_npu, x, expert_ids, expert_rank_table, scales, active_mask_list)

        if server_num > 1:
            compare_num = get_ffn_num_in_server()
            compare_start = server_index * rank_per_dev
        else:
            compare_num = ffn_worker_num
            compare_start = 0
        
        if is_performance:
            attn2ffn_performance = get_attn2ffn_performance()

        if compare_num > 0:
            result_attn2ffn = {}
            # 比较win区状态tensor
            res = compare_output(
                'state',
                [token_info_golden[i] for i in range(compare_start, compare_start + compare_num)],
                [token_states_cpu_list[i] for i in range(compare_start, compare_start + compare_num)],
                torch.float32,
                'ffn_token_state',
                None
            )
            result_attn2ffn.update(res)
            # 比较win区token数据tensor
            res = compare_output(
                'data',
                [token_data_golden[i] for i in range(compare_start, compare_start + compare_num)],
                [token_data_cpu_list[i] for i in range(compare_start, compare_start + compare_num)],
                torch.float32,
                'ffn_token_data',
                [token_data_mask[i] for i in range(compare_start, compare_start + compare_num)]
            )
            result_attn2ffn.update(res)
            # 比较win区量化场景scale
            if is_quant:
                res = compare_output(
                'quant_scales',
                [quant_scales_golden[i] for i in range(compare_start, compare_start + compare_num)],
                [quant_scales_cpu_list[i] for i in range(compare_start, compare_start + compare_num)],
                torch.float32,
                'ffn_quant_scales',
                None
            )
            result_attn2ffn.update(res)
            # 比较Attention节点win区状态
            if is_mask:
                res = compare_output(
                    'state',
                    [attn_info_golden[i] for i in range(compare_start, compare_start + rank_per_dev - compare_num)],
                    [attn_token_states_cpu_list[i] for i in range(compare_start, compare_start + rank_per_dev - compare_num)],
                    torch.float32,
                    'attention_token_state',
                    None
                )
                result_attn2ffn.update(res)

            print("\n##################result_attn2ffn##################")
            print("result_attn2ffn: ", result_attn2ffn)
            all_result = all(result_attn2ffn.values())
            if all_result:
                print("attn2ffn test Success.")
            else:
                print("attn2ffn test Failed.")
        
        else:
            time.sleep(30)
            if is_mask:
                result_attn2ffn = {}
                res = compare_output(
                    'state',
                    [attn_info_golden[i] for i in range(compare_start, compare_start + rank_per_dev)],
                    [attn_token_states_cpu_list[i] for i in range(compare_start, compare_start +rank_per_dev)],
                    torch.float32,
                    'attention_token_state',
                    None
                )
                result_attn2ffn.update(res)
                print("\n##################result_attn2ffn##################")
                print("result_attn2ffn: ", result_attn2ffn)
                all_result = all(result_attn2ffn.values())
                if all_result:
                    print("attn2ffn test Success.")
                else:
                    print("attn2ffn test Failed.")
            else:
                # time.sleep(5)
                print("results are not in this server.")
                print("attn2ffn test Success.")

        if is_performance:
            print_attn2ffn_performance(attn2ffn_performance)