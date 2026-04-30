"""
AttentionToFFN算子端到端测试脚本

本脚本用于测试AttentionToFFN融合算子的功能正确性和性能，测试流程如下：
  1. 生成测试输入数据（token激活值、expertIds、expertRankTable、scales、activeMask）
  2. 计算CPU侧Golden结果（模拟Attention→FFN的token分发逻辑）
  3. 多进程启动NPU侧算子执行（Attention卡发送token，FFN卡接收token）
  4. 对比NPU输出与Golden结果的精度
  5. 可选：采集Profiling数据并输出性能指标

测试架构：
  - world_size=16个rank，分为FFN Worker（前ffn_worker_num个）和Attention Worker（后面的）
  - 每个rank独占一个NPU设备，通过HCCL通信域和peermem窗口进行跨卡数据传输
  - 支持单算子模式（graph_type=0）和图模式（graph_type=1/2）
  - 支持量化模式（quant_mode=2为动态量化）和activeMask模式
"""

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
current_case_name = 'Test_AttentionToFFN_16die_ID030'  # 当前测试用例名称，用于CSV结果记录

is_attn2ffn = 1                  # 是否执行AttentionToFFN算子测试
is_ffn_batching = 0              # 是否调用FFNWorkerBatching算子清理FFN状态位（联合性能测试时开启）

# 性能检测相关
is_performance = 0               # 是否开启性能Profiling采集
loop = 105                       # 性能检测时的总迭代轮数（前5轮warmup，后100轮采集）
attn2ffn_profiling_path = './Profiling_Result_attn2ffn'  # Profiling结果输出路径

# 量化相关
quant_mode = 0                   # 量化模式：0=不量化，2=动态量化（int8），1=静态量化（暂不支持）
is_quant = (quant_mode > 0)      # 是否启用量化的布尔标志
is_attn2ffn_scales = 0           # 是否提供smoothScales输入（per-channel平滑量化缩放因子）

# 图模式相关
graph_type = 1                   # 图模式：0=单算子模式，1=传统入图（静态shape），2=ACLNN入图（动态shape）

# 参数设置
is_mask = 0                      # 是否启用activeMask（仅发送有效token，跳过padding token）
sync_flag = 0                    # 同步模式标志：0=同步（所有FFN节点flag均置1），1=异步

# 自定义windows区conda
enable_window = True                        # 是否开启自定义window区功能（peermem通信窗口）
# enable_window = False
default_window_size = 1024 * 1024 * 200             # 200M 空间，默认peermem窗口大小
default_window_addr = 0x12c101800000        # 默认win区起始物理地址

# 分核配置
is_full_core = 1                            # 是否使用全核模式（1=全核，0=限制核数）
core_num = 24                               # 非全核模式下使用的AIC核数
aic_core_num = core_num                                         # 非全核模式使用的aic核数
aiv_core_num = core_num * 2                                     # 非全核模式使用的aiv核数（AIV核数=AIC核数×2）
is_topk_balance = True                                          # 是否生成均衡的topK（True=均匀分配专家，False=随机分配）

##################### 模型相关设置 #####################
# shape
bs = 512                                   # batch sequence size，每个Attention worker的token总数
h = 8192                                   # hidden size，每个token的隐藏维度
scale_pad = 128                            # 量化参数存放空间（int8数据末尾追加的FP32反量化scale占128字节对齐）
hs = h + scale_pad                         # 量化时token数据的总长度（h + scale_pad）
k = 8                                      # topK专家数，每个token选择k个MoE专家
X = 1                                      # token组数（通常为1，支持多组token批量处理）
L = 1                                      # 模型层数
micro_batch_num = 1                         # micro batch 数量（微批次切分，支持流水线并行）
default_micro_batch_id = 0                  # 默认micro batch id
layer_num = 1                               # 模型层数
default_layer_id = 0                        # 默认模型层id
ffn_start_rank_id = 0                       # ffn worker 起始id（FFN rank在通信域中的起始编号）

server_num = 1                              # 物理服务器数量（多机场景使用）
server_index = 1                            # 当前服务器索引（多机场景使用）
rank_per_dev = 16                           # 每个设备（如NPU）上运行的rank数量（这里模拟16个逻辑rank）
world_size = rank_per_dev * server_num      # 通信域总rank数
ep_world_size = world_size                  # Expert Parallel的world size（专家并行规模，与通信域大小一致）

shared_expert_num = 2                       # 共享专家数量（每个token都会经过的专家）
rank_num_per_shared_expert = 1              # 每个共享专家占用的rank数
shared_expert_rank_num_ffn = shared_expert_num * rank_num_per_shared_expert     # 共享专家（shared expert）使用的FFN rank数量

moe_rank_ffn_num = 10                       # MoE专家使用的FFN rank数量
moe_expert_per_rank = 64                    # 每个FFN rank上部署的MoE专家数
moe_expert_num = moe_rank_ffn_num * moe_expert_per_rank                     # MoE专家总数 = 10×64 = 640

ffn_worker_num = shared_expert_rank_num_ffn + moe_rank_ffn_num              # FFN rank总数 = 2 + 10 = 12
attention_worker_num = world_size - shared_expert_rank_num_ffn - moe_rank_ffn_num  # Attention rank数 = 16 - 12 = 4
expert_num_per_token = k + shared_expert_num                                # 每个token经过的专家数 = topK + 共享专家 = 8 + 2 = 10

##################### 数据格式相关 #####################
out_dtype = 0                               # 输出数据类型标志：0=bfloat16，1=float16
input_dtype = torch.bfloat16 if out_dtype == 0 else torch.float16  # 输入数据类型

dtype_mapping = {                            # 数据类型字符串→torch.dtype映射表
    "float": torch.float,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}
dtype_str = "bfloat16" if out_dtype == 0 else "float16"
set_dtype = os.getenv("DTYPE", dtype_str)    # 支持通过环境变量DTYPE覆盖数据类型
if is_quant and is_attn2ffn:
    set_dtype = "int8"                       # 量化模式下强制使用int8类型
model_dtype = dtype_mapping.get(set_dtype, torch.bfloat16)  # 最终模型使用的数据类型

##################### 算子参数相关 #####################
# Attention → FFN 的 token 路由信息表（每个 attention worker 发送）
# 最后一维：[flag, layer_id, token0_expert0_localId, token0_expert1_localId, ..., token_bs-1_expert_k+shared-1_localId]
#   - flag: FFN侧数据可写标志（0=可写，1=数据就绪）
#   - layer_id: 当前模型层ID
#   - 后续bs*expert_num_per_token个元素：每个token的每个专家在目标FFN卡上的本地专家ID
ffn_token_info_table_shape = [attention_worker_num, micro_batch_num, 1+1+bs*expert_num_per_token]

# 实际 token 数据（embedding）
# 非量化时：token_data_size = (h*2 + 511) // 512 * 512（FP16/BF16，512B对齐）
# 量化时：token_data_size = (h+128 + 511) // 512 * 512（int8 + 128B量化参数，512B对齐）
# shape: [attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size]
if quant_mode == 0:
    token_data_size = (h * 2 + 512 - 1) // 512 * 512    # 非量化：h个FP16元素，512B对齐
else:
    token_data_size = (h + 128 + 512 - 1) // 512 * 512  # 量化：h个int8 + 128B量化参数，512B对齐
ffn_token_data_shape = [attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_data_size if is_quant else token_data_size // 2]

# Attention 侧的 token 信息（activeMask场景下，记录padding token的flag状态）
attn_token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]

# win 区需要的大小（peermem通信窗口总大小 = 状态区 + 数据区）
window_size_state = attention_worker_num * micro_batch_num * (1+1+bs*expert_num_per_token) * 4  # 状态区：int32元素
window_size_data = attention_worker_num * micro_batch_num * bs * expert_num_per_token * token_data_size  # 数据区：token数据

valid_window_size = (window_size_state + 512 - 1) // 512 * 512 + window_size_data  # 状态区512B对齐 + 数据区

window_size = valid_window_size              # 实际使用的窗口大小
# window_size = default_window_size
# print("window_size=", window_size/1024/1024, "MB")

if (valid_window_size >= default_window_size):
    os.environ['HCCL_BUFFSIZE'] = '4000'     # 窗口超过默认200M时，扩大HCCL缓冲区到4000M

# window_size = default_window_size

##################### ffn_to_attention 算子参数相关 #####################
actual_token_num = bs * micro_batch_num * attention_worker_num * expert_num_per_token  # token总数量（所有Attention worker的所有token×专家数）

# 算子参数
token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]  # Attention侧token信息表shape
token_data_table_shape = [micro_batch_num, bs, expert_num_per_token, h]  # Attention侧token数据shape
attn_rank_table = torch.arange(attention_worker_num, dtype=torch.int32)  # Attention rank编号表

##################### 通信参数相关 #####################
master_ip = '71.20.45.104' if server_num > 1 else '127.0.0.1'  # 多机场景使用远程IP，单机使用localhost
# [[0,1,2,...,15]]
ep_ranks_list = list(range(world_size))      # Expert Parallel域的所有rank列表

##################### 测试脚本执行相关 #####################
compare_per_rank = 1                 # 是否逐rank比较结果（1=每个rank单独比较，0=所有rank拼接后比较）
is_check_setting = False              # 是否在运行前检查参数合法性

def print_setting():
    ''' 打印当前测试的所有关键参数配置
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
    ''' 检查参数设置的合法性，不合法则报错退出
    '''
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
    """AttentionToFFN算子的图模式封装模型
    将npu_attention_to_ffn算子封装为torch.nn.Module，以便通过torch.compile入图执行
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self, x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size, ffn_token_info_table_shape,
        ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, scales, active_mask, quant_mode, sync_flag, ffn_start_rank_id
    ):
        """前向传播：调用npu_attention_to_ffn算子
        Args:
            x: Attention输出的token激活值，shape=[attention_worker_num, X, bs, h]
            session_id: 当前Attention Worker的session ID
            micro_batch_id: 当前micro batch的ID
            layer_id: 当前模型层ID
            expert_ids: 每个token选择的topK专家ID，shape=[attention_worker_num, X, bs, k]
            expert_rank_table: 专家→FFN卡的映射表，shape=[L, moe_expert_num+shared_expert_num, max_len]
            group: HCCL通信域句柄
            world_size: 通信域大小
            ffn_token_info_table_shape: FFN侧tokenInfoTable的shape
            ffn_token_data_shape: FFN侧tokenData的shape
            attn_token_info_table_shape: Attention侧tokenInfoTable的shape
            moe_expert_num: MoE专家总数
            scales: smoothScales（量化模式使用），shape=[L, expert_num, h]
            active_mask: 有效token掩码（activeMask模式使用），shape=[attention_worker_num, X, bs]
            quant_mode: 量化模式（0=不量化，2=动态量化）
            sync_flag: 同步模式标志
            ffn_start_rank_id: FFN rank在通信域中的起始编号
        """
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
    """FFNWorkerBatching算子的图模式封装模型
    将npu_ffn_worker_batching算子封装为torch.nn.Module，用于FFN侧的调度和状态清理
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self, schedule_context, expert_num, max_out_shape, token_dtype, need_schedule, layer_num
    ):
        """前向传播：调用npu_ffn_worker_batching算子
        Args:
            schedule_context: FFN调度上下文张量（由create_schedule_context_holder生成）
            expert_num: 当前FFN rank上的专家数量
            max_out_shape: 最大输出shape [attention_worker_num, bs, expert_num_per_token, token_size]
            token_dtype: token数据类型标志（0=float16, 1=bfloat16, 2=int8）
            need_schedule: 是否需要调度（1=需要）
            layer_num: 模型层数
        """
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
    根据graph_type配置，将PyTorch模型编译为NPU图执行模式：
      - graph_type=1: 传统入图模式，静态shape+在线编译
      - graph_type=2: ACLNN入图模式，动态shape+二进制缓存
    支持通过is_full_core控制是否使用全核或限制核数
    '''
    if graph_type == 0:
        print('单算子模式下不应调用compile_model函数')
        exit(1)

    print(f"------------------graph_type: {graph_type}------------------",)

    from torchair.configs.compiler_config import CompilerConfig
    if is_full_core:
        compiler_config = None              # 全核模式：不限制核数，使用默认配置
    else:
        # 使用一半的核：通过CompilerConfig限制AIC/AIV核数
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
        print(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)  # 获取NPU后端

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
    '''生成AttentionToFFN算子需要的全部输入参数
    Returns:
        x: token激活值，shape=[attention_worker_num, X, bs, h]
        expert_ids: topK专家ID，shape=[attention_worker_num, X, bs, k]
        expert_rank_table: 专家→FFN卡映射表，shape=[L, expert_num, max_len]
        scales: smoothScales（量化模式），shape=[L, expert_num, h]或None
        active_mask_list: 有效token掩码，shape=[attention_worker_num, X, bs]
    '''
    def gen_x():
        '''生成 Attention 输出的激活值，shape=[attention_worker_num, X, bs, h]，值在 [-1024, 1024] 之间（模拟真实分布）
        '''
        return torch.empty(size=[attention_worker_num, X, bs, h], dtype=input_dtype).uniform_(-1024, 1024)

    def gen_random_token_expert_ids(length, from_, to_):
        '''从[from_, to_)范围内随机排列后取前length个，生成随机专家ID'''
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]

    def gen_balanced_expert_ids_element(round_index, moe_expert_num_re):
        '''生成均衡分布的专家ID：从round_index开始顺序递增，取模确保在[0, moe_expert_num)范围内'''
        return torch.arange(round_index, round_index + moe_expert_num_re, dtype=torch.int32) % (moe_expert_num_re)

    def gen_expert_ids():
        '''生成expert_ids，shape=[attention_worker_num, X, bs, k]
        is_topk_balance=True: 均衡分配，确保每个专家被选中的次数大致相同
        is_topk_balance=False: 随机分配，每个token独立随机选择k个专家
        '''
        if is_topk_balance:
            ep_expert_ids = [gen_balanced_expert_ids_element(round_index, moe_expert_num) \
                                for round_index in range((X * bs * attention_worker_num * k) // moe_expert_num + 1)]
        else:
            ep_expert_ids = [gen_random_token_expert_ids(k, 0, moe_expert_num) for _ in range(X * bs * attention_worker_num)]

        return torch.cat(ep_expert_ids)[:X * bs * attention_worker_num * k].view([attention_worker_num, X, bs, k])

    def gen_scales():
        '''生成smoothScales，shape=[L, shared_expert_num + moe_expert_num, h]
        仅在量化模式+is_attn2ffn_scales时生成，否则返回None
        scales用于per-channel平滑量化：量化前先乘以scales，反量化时除以scales
        '''
        if is_quant and is_attn2ffn_scales:
            scales = torch.empty(size=[L, shared_expert_num + moe_expert_num, h], dtype=torch.float32).uniform_(-1, 1)
        else:
            # 使用空张量torch.tensor([])会被dispatch算子的tiling检查拦截，只能用None，无法保证数据类型一致
            scales = None
        return scales
    
    def gen_expert_rank_table():
        '''生成expertRankTable，shape=[L, expert_num, max_len]
        最后一维格式：[count, rank0, expId0, rank1, expId1, ..., padding...]
          - count: 该专家部署在多少张FFN卡上
          - rank_i, expId_i: 第i个部署位置的FFN rank ID和该卡上的本地专家ID
        支持一卡多专家和负载均衡（sessionId % count选择目标rank）
        '''
        data = []
        max_len = 0
        valid_rank_id = [i for i in range(moe_rank_ffn_num)] * moe_expert_per_rank  # 可用的FFN rank列表，每个rank重复moe_expert_per_rank次
        valid_exp_id = list(range(moe_expert_num))  # 所有MoE专家ID
        cards = {}                              # 每个FFN rank上部署的专家ID列表
        for i in range(moe_rank_ffn_num):
            start = i * moe_expert_per_rank
            end = start + moe_expert_per_rank
            cards[i] = valid_exp_id[start:end]  # rank_i 上部署 [i*64, (i+1)*64) 号专家

        for l in range(layer_num):
            layer_e1 = []

            for e in range(moe_expert_num):
                # 为每个MoE专家随机分配1~moe_rank_ffn_num个FFN卡
                cards_h = copy.deepcopy(cards)
                valid_rank_id_h = copy.deepcopy(valid_rank_id)

                i = random.randint(1, moe_rank_ffn_num)  # 随机选择部署的卡数量
                item = [i]  # item[0] = 部署的卡数量
                for index_i in range(i):
                    # 随机选择一个FFN rank
                    num_rank_id = random.choice(valid_rank_id_h)
                    item.append(num_rank_id)
                    valid_rank_id_h.remove(num_rank_id)

                    # 从该rank上随机选择一个专家ID作为本地专家ID
                    num_exp_id = random.choice(cards_h[num_rank_id])
                    item.append(num_exp_id)
                    cards_h[num_rank_id].remove(num_exp_id)

                # item格式：[部署卡数量, rank0, expId0, rank1, expId1, ...]
                layer_e1.append(item)
                max_len = max(max_len, len(item))
            # 共享专家：每个共享专家固定部署在moe_rank_ffn_num之后的rank上
            for i in range(shared_expert_num):
                item = [rank_num_per_shared_expert]
                for j in range(rank_num_per_shared_expert):
                    item.append(moe_rank_ffn_num + i * rank_num_per_shared_expert + j)  # 共享专家的FFN rank ID
                    item.append(moe_expert_num + i)  # 共享专家的本地专家ID（排在MoE专家之后）
                layer_e1.append(item)
                max_len = max(max_len, len(item))
            data.append(layer_e1)

        # Padding：将所有item补齐到max_len长度（不足的填0）
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
        '''生成activeMask，shape=[attention_worker_num, X, bs]
        is_mask=True: 随机生成有效token数（0~bs），前real_token_num个为True，其余为False
        is_mask=False: 所有token均为True（全部有效）
        '''
        min_real_token_num = 0 if is_mask else 1  # is_mask时允许0个有效token，否则至少1个
        active_mask_list = []
        for _ in range(attention_worker_num):
            real_token_num = torch.randint(min_real_token_num, bs + 1, (1,)).item() if is_mask else bs
            active_mask = torch.cat((torch.ones(real_token_num), torch.zeros(bs - real_token_num))).to(torch.bool)
            active_mask = active_mask.unsqueeze(0)  # 形状变为 [1, bs]
            active_mask = active_mask.repeat(X, 1)  # 形状变为 [X, bs]
            active_mask_list.append(active_mask)
        
        # 将列表转换为张量并堆叠
        active_mask_list = torch.stack(active_mask_list)  # shape: [attention_worker_num, X, bs]
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
    """计算AttentionToFFN算子的CPU侧Golden结果
    模拟Attention→FFN的token分发逻辑：
      1. 遍历每个Attention worker的每个token
      2. 根据expertRankTable查找每个token的topK专家和共享专家对应的目标FFN rank和本地专家ID
      3. 将token数据写入目标FFN rank的ffn_token_data中
      4. 设置ffn_token_state中的flag和localExpId
      5. 量化模式下执行动态量化并记录反量化scale
    Args:
        x_golden: token激活值，shape=[attention_worker_num, X, bs, h]
        expert_ids_golden: topK专家ID，shape=[attention_worker_num, X, bs, k]
        expert_rank_table: 专家→FFN卡映射表
        scales: smoothScales（量化模式）
        active_mask_list: 有效token掩码
    Returns:
        ffn_token_data: FFN侧token数据
        token_data_mask: 有效数据位置掩码（比较时仅比较有效位置）
        ffn_quant_scales: 量化反量化scale
        ffn_token_state: FFN侧token状态表
        attn_token_state: Attention侧token状态表（activeMask场景）
    """
    # 初始化输出张量
    token_size = token_data_size if is_quant else token_data_size // 2

    ffn_token_state = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, 1 + 1 + bs * expert_num_per_token)  # FFN层的状态信息
    ffn_token_data = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size)  # FFN层的输入数据
    ffn_quant_scales = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, expert_num_per_token)  # FFN层的量化参数
    token_data_mask = torch.zeros(ffn_worker_num, attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size, dtype=torch.bool)   # 用于结果比较时，比较有效位置上的数据
    attn_token_state = torch.zeros(attention_worker_num, micro_batch_num, bs, expert_num_per_token)  # Attention侧token状态（activeMask场景下padding token的flag）

    def load_banance(session_id, layer_id, expert_id):
        """负载均衡选择：根据sessionId在专家部署的多个FFN卡中轮询选择目标rank
        与算子内核中的FindExpertRank逻辑一致：dst_rank = expertRankTable[layer][expert][1 + (sessionId % count) * 2]
        """
        count = expert_rank_table[layer_id, expert_id, 0]  # 该专家部署在多少张FFN卡上
        idx = session_id % count              # 负载均衡：sessionId % count 轮询选择
        dst_rank_id = expert_rank_table[layer_id, expert_id, 1 + idx * 2]  # 目标FFN rank ID
        dst_exp_id = expert_rank_table[layer_id, expert_id, 2 + idx * 2]  # 目标卡上的本地专家ID
        return (dst_rank_id, dst_exp_id)
    
    def quant_process(y, scales):
        """动态量化：FP16/BF16 → int8
        流程：y_fp32 = y * scales → 求绝对值最大值 → dynamic_scale = 127/max → 量化到int8
        Returns:
            y: int8量化结果
            scale: 反量化scale = 1/dynamic_scale
        """
        y = y.to(torch.float32) * scales           # 首先把需要量化的值乘上一个scales

        y_max, _ = y.abs().max(dim=-1)              # 量化公式第一步求绝对值的最大值
        y_max[y_max == 0] = 1.0                   # 防止除0
        dynamic_scales = torch.tensor(127.0, dtype=torch.float32) / y_max  # 动态缩放因子 = 127/max
        y = (y * dynamic_scales.view(-1, 1)).round().to(torch.int8)  # 缩放并四舍五入到int8
        scale = 1.0 / dynamic_scales               # 反量化scale
        return y, scale.flatten()

    # 算子部分
    def att2ffn(row_idx, x_i, expert_id, attn_id):
        """将单个token的数据分发到其对应的FFN卡
        处理topK个MoE专家 + shared_expert_num个共享专家
        """
        nonlocal ffn_token_data, ffn_token_state, token_data_mask
        scales_i = None
        hAlign = (h + 32 -1) // 32 * 32
        # 处理moe专家
        # topk_id 为索引
        for topk_id, expert_i in enumerate(expert_id):
            rank_id, exp_id = load_banance(attn_id, default_layer_id, expert_i)

            # 标记有效数据位置（比较时仅比较前h个元素）
            token_data_mask[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = torch.ones(h, dtype=torch.bool)
            if is_quant:
                # 量化模式：对token数据做动态量化后写入
                scales_i = scales[default_layer_id, expert_i] if is_attn2ffn_scales else 1
                quant_data, dynamic_scales = quant_process(x_i, scales_i)
                ffn_token_data[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = quant_data
                ffn_quant_scales[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id] = dynamic_scales
            else:
                # 非量化模式：直接写入原始FP16/BF16数据
                ffn_token_data[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :h] = x_i

            # 设置FFN侧状态表：flag=1表示数据就绪，layer_id，localExpId
            ffn_token_state[rank_id, attn_id, default_micro_batch_id, 0] = 1
            ffn_token_state[rank_id, attn_id, default_micro_batch_id, 1] = default_layer_id
            ffn_token_state[rank_id, attn_id, default_micro_batch_id, (row_idx * expert_num_per_token + topk_id + 2)] = exp_id.float()

        # 处理共享专家：共享专家的topk_id从k开始
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

    # 遍历所有Attention worker的所有token，根据activeMask决定是否分发
    for attn_id in range(attention_worker_num):
        for token_id in range(bs):
            if(active_mask_list[attn_id, X - 1, token_id]):
                # 有效token：正常分发到FFN
                att2ffn(token_id, x_golden[attn_id, X - 1, token_id], expert_ids_golden[attn_id, X - 1, token_id], attn_id)
            else:
                # padding token：在Attention侧状态表中标记为1（算子会为padding token补写flag=1）
                attn_token_state[attn_id, X - 1, token_id, : ] = torch.ones(expert_num_per_token)

    if (sync_flag == 0):
        # 同步模式：所有FFN节点的flag位均置1（同步模式下算子会对所有FFN rank设置flag）
        ffn_token_state[:, :, :, 0] = 1
    
    # for i in range(ffn_worker_num):
        # print(f"------rank {i} golden ffn_token_state:", ffn_token_state[i].size(), ffn_token_state[i])
        # print(f"------rank {i} golden ffn_token_data:", ffn_token_data[i].size(), ffn_token_data[i])

    return ffn_token_data, token_data_mask, ffn_quant_scales, ffn_token_state, attn_token_state

def write_row_to_csv(csv_name: str, row_content: list):
    """将一行数据追加写入CSV文件，文件不存在则先创建csv_result目录"""
    if not os.path.exists("csv_result"):
        os.makedirs("csv_result")
    with open(f'csv_result/{csv_name}', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row_content)


def get_performance(profiling_path):
    """从Profiling结果目录中解析算子执行性能数据
    读取op_summary CSV文件，提取AttentionToFFN算子的Task Duration和Task Wait Time
    过滤掉前5轮warmup，取后50轮的均值和最小值
    Args:
        profiling_path: Profiling结果目录路径
    Returns:
        [mean_performance, min_performance]: 平均和最小执行时间(us)
    """
    op_types = ['AttentionToFFN']
    rank_types = ['attention']
    performance_info = {(op_type, rank_type): [] for op_type in op_types for rank_type in rank_types}

    long_wait_ranks = set()
    for dirpath, _, filenames in os.walk(profiling_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))  # 读取op_summary CSV
                local_rank_id = op_summary['Device_id'].iloc[0]  # 获取当前设备的local rank ID
                ep_rank_id = local_rank_id
                rank_type = 'attention'
                for op_type in op_types:
                    op_info = op_summary[op_summary['OP Type'] == op_type]  # 筛选目标算子类型

                    op_execute_time_series = op_info['Task Duration(us)'].tail(50)  # 取后50轮执行时间
                    performance_info[(op_type, rank_type)].append(op_execute_time_series)

                    mean_op_wait_time = op_info['Task Wait Time(us)'].tail(50).mean()  # 平均等待时间
                    if mean_op_wait_time > 5:
                        long_wait_ranks.add(local_rank_id)  # 记录等待时间过长的rank

    print(f"Task Wait > 5 us local rank id is {sorted(long_wait_ranks)}")

    # 汇总所有rank的执行时间，计算均值和最小值
    mean_performances = []
    min_performances = []
    for execute_time_series_list in performance_info.values():
        whole_execute_time_series = (
            pd.concat(execute_time_series_list)   # 拼接所有rank的执行时间序列
            if execute_time_series_list
            else pd.Series(dtype=float)
        )
        print(f'whole_execute_time_series: {whole_execute_time_series}')
        mean_performances.append(whole_execute_time_series.mean())
        min_performances.append(whole_execute_time_series.min())
    return mean_performances + min_performances


def print_attn2ffn_performance(attention_performance):
    """打印AttentionToFFN算子的性能结果"""
    print(f'attention_performance: {attention_performance}')
    mean_attention_performance, min_attention_performance = attention_performance

    print(f'[INFO] attn2ffn融合算子Average性能为: {mean_attention_performance} us\n')
    print(f'[INFO] attn2ffn融合算子Min性能为: {min_attention_performance} us\n')


def save_performance_result(test_mode, case_name, op_types, op_performances):
    """将性能测试结果保存到CSV文件
    Args:
        test_mode: 测试模式名称（如'AttentionToFFN'）
        case_name: 测试用例名称
        op_types: 算子类型列表
        op_performances: 每个算子的性能指标列表 [mean, min]
    """
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
    """采集并保存AttentionToFFN算子的性能数据
    从Profiling结果中提取执行时间，打印均值和最小值，保存到CSV
    Returns:
        attn2ffn_performance: [mean_performance, min_performance]
    """
    attn2ffn_performance = get_performance(attn2ffn_profiling_path)
    print(f'attn2ffn_performance: {attn2ffn_performance}')

    print_attn2ffn_performance(attn2ffn_performance)

    op_types = ['AttentionToFFN']
    save_performance_result('AttentionToFFN', current_case_name, op_types, [attn2ffn_performance])

    return attn2ffn_performance


def get_profiling(loop, path):
    """创建NPU Profiler对象，用于采集算子执行性能数据
    Args:
        loop: 总迭代轮数
        path: Profiling结果输出路径
    Returns:
        prof: torch_npu.profiler.profile对象，调用start()/step()/stop()采集数据
    """
    repeat = 1
    warmup = 5                            # 前5轮warmup不采集
    active = loop // repeat - warmup      # 实际采集轮数 = loop - warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,  # 采集AIC流水线利用率
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
    """设置当前进程使用的NPU设备，rank % rank_per_dev映射到物理设备ID"""
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank):
    """初始化HCCL通信域和Expert Parallel通信组
    Args:
        rank: 当前进程的rank ID
    Returns:
        ep_hcomm_info: EP通信域的HCCL句柄字符串（传递给算子）
        ep_group: EP通信组对象
    """
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
    print(f"device_{rank} init_process_group success")
    
    print(f"device {rank} 初始化EP域")
    tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)  # 创建EP通信组
    if rank in ep_ranks_list:
        ep_group = tmp_group
    
    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)  # 获取HCCL通信句柄
    print('ep_hcomm_info: '+ep_hcomm_info)
    
    return ep_hcomm_info, ep_group


def dispatch_gen_expert_ids():
    ''' 生成dispatch算子的expert_ids（非window模式使用）
    每个token随机选择k个专家，shape=[bs, k]
    '''
    def gen_random_token_expert_ids(length, from_, to_):
        return (torch.randperm(to_ - from_, dtype=torch.int32) + from_)[:length]                        

    
    ep_expert_ids = [gen_random_token_expert_ids(k, 0, moe_expert_num) for _ in range(bs)] 
    
    return torch.cat(ep_expert_ids)[:bs * k].view([bs, k])


def align_up(num, align):
    '''向上取整对齐：将num对齐到align的整数倍'''
    return ((num + align - 1) // align) * align


def dispatch_get_kwargs(
    rank, group_ep
):
    ''' 生成dispatch算子（非window模式的旧接口）的输入参数
    用于不启用自定义window区时的All2All通信
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
    '''构造AttentionToFFN算子的NPU调用参数
    将CPU张量转换为NPU张量，组装算子所需的完整kwargs字典
    '''
    x = x.to(input_dtype).npu()
    session_id = session_id.to(torch.int32).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    micro_batch_id = micro_batch_id.to(torch.int32).npu()
    layer_id = layer_id.to(torch.int32).npu()
    expert_rank_table = expert_rank_table.to(torch.int32).npu()
    if is_quant and is_attn2ffn_scales:
        scales = scales.to(torch.float32).npu()
    active_mask = active_mask.to(torch.bool).npu() if is_mask else None  # 非mask模式传None

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
    '''构造FFNWorkerBatching算子的NPU调用参数
    创建调度上下文（schedule_context），配置FFN侧的token接收和调度参数
    Args:
        rank: 当前FFN rank ID
        window_addr: peermem窗口起始地址
        window_size: peermem窗口大小
    Returns:
        batching_kwargs: FFNWorkerBatching算子的完整参数字典
    '''
    token_size = token_data_size if is_quant else token_data_size // 2
    max_out_shape = [attention_worker_num, bs, (k + shared_expert_num), token_size]  # FFN输出的最大shape
    if quant_mode == 2:
        attn_to_ffn_token_size = align_up(hs, 512)  # 量化时token大小：h+128，512B对齐
        token_dtype = 2                              # int8类型标志
    else:
        attn_to_ffn_token_size = align_up(h * 2, 512)  # 非量化时token大小：h个FP16/BF16，512B对齐
        token_dtype = 1 if input_dtype == torch.bfloat16 else 0  # 1=bfloat16, 0=float16
    ffn_to_attn_token_size = align_up(h * 2, 512)   # FFN→Attention方向的token大小
    expert_num = moe_expert_num + shared_expert_rank_num_ffn

    context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode=0, session_num=attention_worker_num,
        micro_batch_num=micro_batch_num, micro_batch_size=bs, selected_expert_num=k+shared_expert_num,
        expert_num=expert_num, attn_to_ffn_token_size=attn_to_ffn_token_size, ffn_to_attn_token_size=ffn_to_attn_token_size, 
        ffn_window=window_addr, ffn_window_size=window_size)  # 创建FFN调度上下文
    schedule_context = context_holder.get_schedule_context_tensor()  # 获取调度上下文张量
    
    batching_kwargs = {
        "schedule_context": schedule_context,
        "expert_num": moe_expert_per_rank if rank < moe_rank_ffn_num else 1,  # MoE rank有64个专家，共享专家rank有1个
        "max_out_shape": max_out_shape,
        "token_dtype": token_dtype,
        "need_schedule": 1,
        "layer_num": layer_num
    }

    return batching_kwargs


def attn2ffn_get_npu(ep_group):
    ''' 从peermem窗口中读取FFN侧的计算结果
    通过torchair.llm_datadist.create_npu_tensors在peermem地址上创建NPU张量视图
    Returns:
        ffn_window_states: FFN侧token状态表，shape=[attention_worker_num, micro_batch_num, 1+1+bs*expert_num_per_token]
        ffn_data_tensor: FFN侧token数据，shape=[attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size]
        ffn_scales_tensor: 量化反量化scale（量化模式）或全1张量（非量化模式）
    '''
    window_addr = default_window_addr
    if enable_window:
        window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()  # 获取peermem窗口地址

    # 状态区：在peermem起始地址创建tokenInfoTable张量
    npu_state_tensor = torchair.llm_datadist.create_npu_tensors(
        [attention_worker_num, micro_batch_num, 1 + 1 + bs * expert_num_per_token], torch.int32, [window_addr]
    )
    ffn_window_states = npu_state_tensor[0]

    ffn_token_size = ffn_window_states.element_size() * ffn_window_states.numel()
    ffn_token_area_size = (ffn_token_size + 511) // 512 * 512     # 状态区大小512B对齐
    token_size = token_data_size if is_quant else token_data_size // 2

    # 数据区：在状态区之后创建tokenData张量
    ffn_data_tensors = torchair.llm_datadist.create_npu_tensors(
        [attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size], model_dtype, [window_addr + ffn_token_area_size]
    )
    ffn_data_tensor = ffn_data_tensors[0]

    # 量化模式：从int8数据末尾提取反量化scale（FP32格式，位于h/4位置）
    if is_quant:
        ffn_data_tensors_float = torchair.llm_datadist.create_npu_tensors(
            [attention_worker_num, micro_batch_num, bs, expert_num_per_token, token_size//4], torch.float32, [window_addr + ffn_token_area_size]
        )
        ffn_data_tensor_float = ffn_data_tensors_float[0]
        index = align_up(h, 32) // 4  # int8数据末尾的scale位置（32B对齐后转换为FP32索引）
        ffn_scales_tensor = ffn_data_tensor_float[:,:,:,:,index]  # 提取反量化scale

    else:
        ffn_scales_tensor = torch.ones([attention_worker_num, micro_batch_num, bs, expert_num_per_token], dtype=torch.float32)

    return ffn_window_states, ffn_data_tensor, ffn_scales_tensor


def set_windows(rank, ep_group, ep_hcomm_info):
    """配置peermem通信窗口：Attention卡和FFN卡之间建立双向通信通道
    Attention卡需要与前ffn_worker_num个rank通信，FFN卡需要与后面的Attention rank通信
    """
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
    """每个rank进程的NPU执行入口
    根据rank ID分为Attention Worker和FFN Worker两种角色：
      - rank >= ffn_worker_num: Attention Worker，调用npu_attention_to_ffn发送token
      - rank < ffn_worker_num: FFN Worker，从peermem窗口读取接收到的token数据
    Args:
        queue: 多进程间通信队列，用于传递NPU结果到主进程
        rank: 当前进程的rank ID
        x: token激活值
        expert_ids: topK专家ID
        expert_rank_table: 专家→FFN卡映射表
        scales: smoothScales
        active_mask_list: 有效token掩码
    """
    set_device(rank)
    ep_hcomm_info, ep_group= init_hccl_comm(rank)
    set_windows(rank, ep_group, ep_hcomm_info)

    # ============ Attention Worker：发送token到FFN ============
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

        # 单算子模式+非全核：限制AIC/AIV核数
        if graph_type == 0 and not is_full_core:
            torch.npu.set_stream_limit(torch.npu.current_stream(), aic_core_num, aiv_core_num)
            print(f"当前单算子模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")

        # 性能采集模式（仅AttentionToFFN，不含FFNWorkerBatching）
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
        # attn状态（activeMask场景：读取Attention侧窗口中的padding token状态）
        if is_mask:
            time.sleep(5)  # 等待FFN侧处理完成
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

    # ============ FFN Worker：从peermem窗口读取接收到的token数据 ============
    if rank < ffn_worker_num:
        time.sleep(50)  # 等待Attention Worker完成token发送
        ffn_window_states, ffn_data_tensor, ffn_scales_tensor = attn2ffn_get_npu(ep_group)
        # 将张量从 NPU 移到 CPU
        token_states_cpu = ffn_window_states.cpu()
        token_data_cpu = ffn_data_tensor.cpu()
        ffn_scales_cpu = ffn_scales_tensor.cpu()
        # 通过多进程队列将FFN结果传递给主进程
        queue.put([
            rank, 
            [token_states_cpu, token_data_cpu, ffn_scales_cpu]
        ])
        print(f'FFN rank {rank} finished! \n')
        if is_ffn_batching and is_performance:
            # 调用FFNWorkerBatching 参数配置
            window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()
            batching_kwargs = ffn_batching_get_kwargs(rank, window_addr, window_size)
            print(f'rank {rank} get ffn batching complete! \n')

    # AttentionToFFN + FFNWorkerBatching 联合性能采集
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
    """多进程启动NPU侧算子执行，收集各rank的结果
    为每个rank创建独立进程，通过Manager.Queue收集结果
    Args:
        target_func: 每个rank进程执行的函数（attn2ffn_run_npu）
        x, expert_ids, expert_rank_table, scales, active_mask_list: 算子输入数据
    Returns:
        token_states_cpu_list: 各FFN rank的token状态表（CPU张量列表）
        token_data_cpu_list: 各FFN rank的token数据（CPU张量列表）
        quant_scales_cpu_list: 各FFN rank的量化scale（CPU张量列表）
        attn_token_states_cpu_list: 各Attention rank的token状态（CPU张量列表）
    """
    print("single_server scene!!!!!")
    rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev)) if server_num > 1 else list(range(world_size))
    print(f"rank list is: {rank_list}")

    # 多进程，每个进程调用一张卡执行目标函数
    proc_list = []
    manager = Manager()
    result_dqueue = manager.Queue()  # 多进程间通信队列
    mp.set_start_method("forkserver", force=True)
    for rank in rank_list:
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
        proc.join()  # 等待所有进程完成

    # 从队列中收集各rank的结果
    token_states_cpu_list = [None] * ffn_worker_num
    token_data_cpu_list = [None] * ffn_worker_num
    quant_scales_cpu_list = [None] * ffn_worker_num
    attn_token_states_cpu_list = [None] * attention_worker_num

    if server_num > 1:
        ffn_num = get_ffn_num_in_server()
    else:
        ffn_num = ffn_worker_num

    if is_mask:
        # activeMask场景：FFN rank和Attention rank都会向队列发送结果
        for _ in range(rank_per_dev):
            rank_id, rank_output = result_dqueue.get()
            if rank_id < ffn_num:
                # FFN rank结果：[token_states, token_data, quant_scales]
                token_states_cpu_list[rank_id] = rank_output[0]
                token_data_cpu_list[rank_id] = rank_output[1]
                quant_scales_cpu_list[rank_id] = rank_output[2]
            else:
                # Attention rank结果：[attn_token_states]
                attn_token_states_cpu_list[rank_id - ffn_worker_num] = rank_output[0]
    else:
        # 非activeMask场景：仅FFN rank发送结果
        for _ in range(ffn_num):
            rank_id, rank_output = result_dqueue.get()
            token_states_cpu_list[rank_id] = rank_output[0]
            token_data_cpu_list[rank_id] = rank_output[1]
            quant_scales_cpu_list[rank_id] = rank_output[2]

    return token_states_cpu_list, token_data_cpu_list, quant_scales_cpu_list, attn_token_states_cpu_list


def get_ffn_num_in_server():
    """多机场景下，计算当前服务器上运行的FFN rank数量"""
    ffn_server = ffn_worker_num // rank_per_dev  # 完整占满的服务器数量
    if server_index < ffn_server:
        num = rank_per_dev                      # 当前服务器完全由FFN rank占满
    elif server_index == ffn_server:
        num = ffn_worker_num % rank_per_dev     # 当前服务器部分由FFN rank占用
    else:
        num = 0

    return num


def compare_output(str_name, golden_per_card, npu_per_card, compare_dtype, tensor_name, mask_per_card=None):
    """对比Golden结果与NPU结果的精度
    支持逐rank比较或全局比较，支持通过mask过滤无效数据位置
    Args:
        str_name: 比较项名称（用于打印）
        golden_per_card: 各rank的Golden结果列表
        npu_per_card: 各rank的NPU结果列表
        compare_dtype: 比较时的数据类型
        tensor_name: 张量名称（用于CSV记录）
        mask_per_card: 各rank的有效数据掩码列表（可选，仅比较mask=True位置的数据）
    Returns:
        result: 比较结果字典 {tensor_name: True/False}
    """
    def compare(golden, npu, mask, dtype, tensor_name, diffThd=0.001, pctThd=0.001):
        """单次精度比较：将Golden和NPU转为numpy，可选mask过滤，调用toolss_attn.data_compare_np"""
        np_output_npu = np.array(golden.cpu())
        np_output_golden = np.array(npu.cpu())

        if mask is not None:
            # 仅比较mask=True位置的数据（跳过padding填充的无效数据）
            np_mask = np.array(mask.cpu())
            np_output_npu = np_output_npu[np_mask]
            np_output_golden = np_output_golden[np_mask]

        print("dtype: ", dtype)
        max_diff_hd = 0.0
        if dtype == torch.int8:
            max_diff_hd = 1.0  # int8允许最大差异1（量化误差）
        res_output = toolss_attn.data_compare_np(np_output_golden, np_output_npu, diff_thd=diffThd, pct_thd=pctThd, max_diff_hd=max_diff_hd)

        print(f'[INFO] tensor {tensor_name} output测试的精度结果为: {res_output}')
        toolss_attn.write_row_to_csv('attention.csv', [current_case_name, tensor_name, res_output])
        return res_output[0] == 'PASS'
    
    result = {}

    if not compare_per_rank:
        # 全局比较：拼接所有rank的结果后统一比较
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
        # 逐rank比较：每个rank单独比较
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
    # ==================== 测试前准备 ====================
    print_setting()

    if is_check_setting:
        check_setting()

    # 清理旧的Profiling结果和编译缓存
    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')
    os.system(f'rm -rf /root/atc_data/*')
    torch.manual_seed(0)     # 固定随机种子，确保测试可复现
    random.seed(0)
    
    if is_attn2ffn:
        # 步骤1：生成测试输入数据
        (x, expert_ids, expert_rank_table, scales, active_mask_list) = attn2ffn_gen_inputs()
        # 步骤2：计算CPU侧Golden结果
        token_data_golden, token_data_mask, quant_scales_golden, token_info_golden, attn_info_golden = attn2ffn_golden(x, expert_ids, expert_rank_table, scales, active_mask_list)
        # 步骤3：多进程启动NPU侧算子执行
        token_states_cpu_list, token_data_cpu_list, quant_scales_cpu_list, attn_token_states_cpu_list = attn2ffn_gen_npu(attn2ffn_run_npu, x, expert_ids, expert_rank_table, scales, active_mask_list)

        # 确定当前服务器上需要比较的FFN rank范围
        if server_num > 1:
            compare_num = get_ffn_num_in_server()
            compare_start = server_index * rank_per_dev
        else:
            compare_num = ffn_worker_num
            compare_start = 0
        
        # 步骤4（可选）：采集性能数据
        if is_performance:
            attn2ffn_performance = get_attn2ffn_performance()

        # 步骤5：对比NPU结果与Golden结果的精度
        if compare_num > 0:
            result_attn2ffn = {}
            # 比较FFN侧token状态表（flag + layer_id + localExpId）
            res = compare_output(
                'state',
                [token_info_golden[i] for i in range(compare_start, compare_start + compare_num)],
                [token_states_cpu_list[i] for i in range(compare_start, compare_start + compare_num)],
                torch.float32,
                'ffn_token_state',
                None
            )
            result_attn2ffn.update(res)
            # 比较FFN侧token数据（使用mask过滤无效位置）
            res = compare_output(
                'data',
                [token_data_golden[i] for i in range(compare_start, compare_start + compare_num)],
                [token_data_cpu_list[i] for i in range(compare_start, compare_start + compare_num)],
                torch.float32,
                'ffn_token_data',
                [token_data_mask[i] for i in range(compare_start, compare_start + compare_num)]
            )
            result_attn2ffn.update(res)
            # 量化模式：比较反量化scale
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
            # activeMask模式：比较Attention侧padding token的状态
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

            # 输出最终测试结果
            print("\n##################result_attn2ffn##################")
            print("result_attn2ffn: ", result_attn2ffn)
            all_result = all(result_attn2ffn.values())
            if all_result:
                print("attn2ffn test Success.")
            else:
                print("attn2ffn test Failed.")
        
        else:
            # 多机场景：当前服务器上没有FFN rank，仅比较Attention侧状态
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
                print("results are not in this server.")
                print("attn2ffn test Success.")

        # 步骤6：打印性能结果
        if is_performance:
            print_attn2ffn_performance(attn2ffn_performance)