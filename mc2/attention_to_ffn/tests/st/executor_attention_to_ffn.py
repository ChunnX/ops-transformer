"""
=============================================================================
  ATK Executor: attention_to_ffn 算子端到端测试执行器
=============================================================================
  设计思路:
    本文件是 ATK (Ascend Test Kit) 框架下的 executor，负责将原来 server.py 中
    手动多进程管理的测试流程，封装为 ATK BaseApi 子类，由 ATK 框架自动调度多卡执行。

  核心职责:
    1. init_by_input_data(): 根据当前 rank_id 修正输入参数（世界视角 → 单卡视角），
       生成 expertRankTable、activeMask 等全局数据，为 CPU Golden 和 NPU 执行做准备
    2. attn2ffn_golden(): CPU 侧模拟 attention_to_ffn 算子的完整逻辑，计算真值
    3. __call__(): 三路分支执行 —— CPU真值 / Attention Worker NPU执行 / FFN Worker NPU读取

  与 server.py 的对应关系:
    server.py 中的多进程手动管理 → ATK 框架自动分配 rank 进程
    server.py 中的 Manager.Queue → ATK 框架自动收集返回值
    server.py 中的手动对比 → ATK 框架自动对比
    server.py 中的全局变量配置 → 从 input_data.kwargs 读取

  三路执行分支:
    __call__()
      ├─ [cpu + is_bm] CPU真值分支 → 调用 attn2ffn_golden() 计算 Golden
      │   FFN rank → 返回 (ffn_token_state, ffn_token_data)
      │   Attn rank → 返回 (attn_token_state, attn_window_states_company)
      │
      ├─ [rank >= FFNRankNum] Attention Worker 分支
      │   初始化 HCCL 通信域 + peermem 窗口
      │   调用 npu_attention_to_ffn 发送 token
      │   [activeMask] 读取 Attention 侧窗口状态
      │   返回 (attn_window_states, attn_window_states_company)
      │
      └─ [rank < FFNRankNum] FFN Worker 分支
          sleep 等待 Attention 完成
          从 peermem 窗口读取结果
          返回 (attn_states_npu, attn_data_npu)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch_npu
import torchair
import torch.distributed as dist
import time
import copy
import random
import os

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

import sys
sys.path.append('./executor')
from model import ATTENTION_TO_FFN_Model, compile_model
import logging
from atk.common.log import Logger
logging = Logger().get_logger()

@register("execute_Attention2FFN")
class FFN2Attention(BaseApi):
    """
    attention_to_ffn 算子的 ATK 执行器类。

    继承 ATK BaseApi，通过 @register 装饰器注册为 "execute_Attention2FFN"，
    ATK 框架在运行时会根据 yaml 配置自动实例化并调用此类。

    类名 FFN2Attention 是历史命名，实际执行的是 Attention → FFN 方向的 token 分发。

    属性:
        dist_task_info: ATK 分布式任务信息，包含 rank_id、world_size 等
        ep_hcomm_info: HCCL EP 域的通信句柄（字符串），传入算子作为 group 参数
        ep_group: torch.distributed 进程组对象，用于 peermem 窗口注册和通信
        model: ATTENTION_TO_FFN_Model 实例，用于 graph_type=0 的单算子模式
        compiled_model: 编译后的模型实例，用于 graph_type=1 的传统入图模式
    """

    def __init__(self, task_result: TaskResult):
        super(FFN2Attention, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.ep_hcomm_info = None
        self.ep_group = None
        # 预创建两种执行模式的模型实例，避免 __call__ 中重复创建
        self.model = ATTENTION_TO_FFN_Model()
        self.compiled_model = compile_model(ATTENTION_TO_FFN_Model(), graph_type=1)
        logging.error(f'开始执行init!')

    def init_by_input_data(self, input_data: InputDataset):
        """
        ATK 框架在执行 __call__ 之前调用的输入数据预处理方法。

        核心职责:
          1. 清理上一次编译缓存，确保每次测试环境干净
          2. 固定随机种子，保证 Golden 和 NPU 使用相同的输入数据
          3. 根据当前 rank_id，从"世界视角"的全局数据中切片出"单卡视角"的输入
          4. 生成 expertRankTable（专家-卡映射表）和 activeMask（token 活跃掩码）

        参数间约束关系:
          - world_size = FFNRankNum + AttnRankNum
          - moeRankNum = moeExpertNum / localmoeExpert（路由专家占用的 FFN 卡数）
          - sharedRankNum = FFNRankNum - moeRankNum（共享专家占用的 FFN 卡数）
          - rankNumPerSharedExpert = sharedRankNum / sharedExpertNum（每个共享专家的副本数）
          - ExpertNumPerToken = K + sharedExpertNum（每个 token 路由到的专家总数）
        """
        # 清理 atc 编译缓存，防止残留影响下一次编译
        os.system(f'rm -rf /root/atc_data/*')
        # 固定随机种子，确保 expertRankTable 生成和 expertIds 随机排列可复现
        torch.manual_seed(0)
        random.seed(0)

        def gen_expert_rank_table(moe_rank_ffn_num, moe_expert_per_rank, moe_expert_num,
                                  shared_expert_num, rank_num_per_shared_expert, layer_num, max_len):
            """
            生成 expertRankTable 张量，描述每个专家在各层中分布在哪些 FFN rank 上。

            expertRankTable 的形状为 [L, totalExpertNum, M]，其中:
              - L: 层数
              - totalExpertNum = moeExpertNum + sharedExpertNum
              - M: 最大条目长度（padding 后）

            每个条目的格式:
              [count, rank1, exp1, rank2, exp2, ..., rankN, expN, 0, 0, ...]
              - count: 该专家被分配到多少个 rank（负载均衡选择范围）
              - rank_i: 第 i 个目标 FFN rank 的 ID
              - exp_i: 在该 rank 上的本地专家 ID

            对于路由专家(moe expert):
              随机选择 1~moe_rank_ffn_num 个 rank，每个 rank 上随机选一个该 rank 拥有的专家
            对于共享专家(shared expert):
              固定分配到 rank_num_per_shared_expert 个 rank，每个 rank 上是同一个专家

            参数:
              moe_rank_ffn_num: 路由专家占用的 FFN rank 数量
              moe_expert_per_rank: 每个 FFN rank 拥有的路由专家数
              moe_expert_num: 路由专家总数
              shared_expert_num: 共享专家总数
              rank_num_per_shared_expert: 每个共享专家的副本数（跨多少个 rank）
              layer_num: 层数 L
              max_len: padding 后的条目长度 M
            """
            data = []
            # 每个路由专家被所有 rank 重复持有，用于随机选择
            valid_rank_id = [i for i in range(moe_rank_ffn_num)] * moe_expert_per_rank
            valid_exp_id = list(range(moe_expert_num))
            # cards[i] 记录 rank i 上拥有的路由专家 ID 列表
            cards = {}
            for i in range(moe_rank_ffn_num):
                start = i * moe_expert_per_rank
                end = start + moe_expert_per_rank
                cards[i] = valid_exp_id[start:end]

            for l in range(layer_num):
                layer_e1 = []
                # 为每个路由专家生成映射条目
                for e in range(moe_expert_num):
                    cards_h = copy.deepcopy(cards)
                    valid_rank_id_h = copy.deepcopy(valid_rank_id)
                    # 随机选择 1~moe_rank_ffn_num 个 rank 作为该专家的候选目标
                    i = random.randint(1, moe_rank_ffn_num)
                    item = [i]  # 第一个元素是 count
                    for index_i in range(i):
                        num_rank_id = random.choice(valid_rank_id_h)
                        item.append(num_rank_id)
                        valid_rank_id_h.remove(num_rank_id)
                        # 在该 rank 上随机选一个本地专家
                        num_exp_id = random.choice(cards_h[num_rank_id])
                        item.append(num_exp_id)
                        cards_h[num_rank_id].remove(num_exp_id)
                    layer_e1.append(item)

                # 为每个共享专家生成映射条目
                # 共享专家的 rank 从 moe_rank_ffn_num 开始编号（排在路由专家 rank 之后）
                for i in range(shared_expert_num):
                    item = [rank_num_per_shared_expert]
                    for j in range(rank_num_per_shared_expert):
                        # 共享专家 rank ID: moe_rank_ffn_num + i * rankNumPerSharedExpert + j
                        item.append(moe_rank_ffn_num + i * rank_num_per_shared_expert + j)
                        # 共享专家的本地专家 ID: moeExpertNum + i（排在路由专家之后）
                        item.append(moe_expert_num + i)
                    layer_e1.append(item)
                data.append(layer_e1)

            # 对不等长条目做 0-padding，统一到 max_len 长度
            padded_data = []
            for layer in data:
                padded_layer = []
                for item in layer:
                    item_padded = item + [0] * (max_len - len(item))
                    padded_layer.append(item_padded)
                padded_data.append(padded_layer)
            tensor_data = torch.tensor(padded_data)
            return tensor_data

        def gen_active_mask(is_mask, AttnRankNum, BS, X):
            """
            生成 activeMask 张量，标记每个 token 是否为有效 token（非 padding）。

            参数:
              is_mask: 是否启用 mask（True 时随机生成部分 token 为无效，False 时全部有效）
              AttnRankNum: Attention Worker 的 rank 数量
              BS: batch size（每个 micro_batch 的 token 数）
              X: micro_batch 数量维度（通常为 1）

            返回:
              形状为 [AttnRankNum, X, BS] 的 bool 张量
              True 表示该 token 有效，False 表示 padding token
            """
            min_real_token_num = 0 if is_mask else 1
            active_mask_list = []
            for _ in range(AttnRankNum):
                # is_mask=True 时随机选择 0~BS 个有效 token；is_mask=False 时全部有效
                real_token_num = torch.randint(min_real_token_num, BS + 1, (1,)).item() if is_mask else BS
                active_mask = torch.cat((torch.ones(real_token_num), torch.zeros(BS - real_token_num))).to(torch.bool)
                active_mask = active_mask.unsqueeze(0)  # [1, BS]
                active_mask = active_mask.repeat(X, 1)  # [X, BS]
                active_mask_list.append(active_mask)
            active_mask_list = torch.stack(active_mask_list)  # [AttnRankNum, X, BS]
            return active_mask_list

        logging.error(f'开始执行input_data_init!')
        rank_id = self.dist_task_info.rank

        # ── 从 input_data.kwargs 中提取维度参数 ──
        X = input_data.kwargs['x'].shape[0]         # micro_batch 维度
        L = input_data.kwargs['expertRankTable'].shape[0]  # 层数
        M = input_data.kwargs['expertRankTable'].shape[2]  # expertRankTable 条目 padding 长度
        K = input_data.kwargs['expertIds'].shape[2]        # 每个 token 的 topk 路由专家数
        H = input_data.kwargs['x'].shape[2]                # hidden_size
        world_size = input_data.kwargs['worldSize']
        AttnRankNum = input_data.kwargs['ffnTokenInfoTableShape'][0]  # Attention Worker 数量
        FFNRankNum = world_size - AttnRankNum                          # FFN Worker 数量
        BS = input_data.kwargs['x'].shape[1]                           # batch size
        localmoeExpert = input_data.kwargs['localmoeExpert']           # 每个 FFN rank 的路由专家数
        MircoBatchNum = input_data.kwargs['ffnTokenDataShape'][1]      # micro_batch 数
        ExpertNumPerToken = input_data.kwargs['ffnTokenDataShape'][3]  # 每个 token 路由到的专家总数 (K + sharedExpertNum)
        moeExpertNum = input_data.kwargs['moeExpertNum']               # 路由专家总数
        moeRankNum = int(moeExpertNum / localmoeExpert)                # 路由专家占用的 FFN rank 数
        sharedRankNum = FFNRankNum - moeRankNum                        # 共享专家占用的 FFN rank 数
        sharedExpertNum = input_data.kwargs['sharedExpertNum']         # 共享专家总数
        rankNumPerSharedExpert = int(sharedRankNum / sharedExpertNum)  # 每个共享专家的副本数

        # ── 修正 sessionId：Attention Worker 的 sessionId 等于其 AttnId ──
        # sessionId 用于负载均衡，决定 token 路由到哪个 FFN rank
        AttnId = 0
        if FFNRankNum <= rank_id < world_size:
            # 当前 rank 是 Attention Worker，AttnId = rank_id - FFNRankNum
            AttnId = rank_id - FFNRankNum
            input_data.kwargs['sessionId'] = torch.tensor([AttnId], dtype=torch.int32)

        # ── 修正 bsList：确保每个 Attention Worker 都有对应的 BS 值 ──
        input_data.kwargs['bsList'][0] = BS
        if input_data.kwargs['bsList'].shape[0] >= AttnRankNum:
            bs_list = input_data.kwargs['bsList'][:AttnRankNum]
        else:
            # 如果 bsList 长度不够，用第一个值填充
            bs_list = input_data.kwargs['bsList'][0].repeat(AttnRankNum)[:AttnRankNum]
        input_data.kwargs['bsList'] = bs_list

        # ── 生成全局输入数据（世界视角），用于 Golden 计算和单卡切片 ──
        x_world = input_data.kwargs['x_world']

        # 为每个 Attention Worker 的每个 token 随机生成 topk 专家 ID
        # expertIds_world 形状: [AttnRankNum, 1, BS, K]
        ep_expert_ids = [torch.randperm(moeExpertNum, dtype=torch.int32)[:K] for _ in range(BS * AttnRankNum)]
        expertIds_world = torch.cat(ep_expert_ids).view([AttnRankNum, 1, BS, K])

        # ── 处理量化参数 scalesOptional ──
        quant_mode = input_data.kwargs['quantMode']
        is_attn2ffn_scales = input_data.kwargs['is_attn2ffn_scales']
        is_quant = (quant_mode > 0)
        if is_quant and is_attn2ffn_scales:
            # 量化模式下需要 scales，形状 [L, totalExpertNum, H]
            input_data.kwargs['scalesOptional'] = torch.empty(
                size=[L, sharedExpertNum + moeExpertNum, H], dtype=torch.float32).uniform_(-1, 1)
        else:
            input_data.kwargs['scalesOptional'] = None

        # ── 生成 activeMask（全局视角） ──
        is_mask = input_data.kwargs['is_mask']
        activeMaskOptional_world = gen_active_mask(is_mask, AttnRankNum, BS, X)

        # ── 保存全局数据到 kwargs，供 Golden 计算使用 ──
        input_data.kwargs['x_world'] = x_world
        input_data.kwargs['expertIds_world'] = expertIds_world
        input_data.kwargs['activeMaskOptional_world'] = activeMaskOptional_world

        # ── 从全局数据切片出当前 rank 的输入（单卡视角） ──
        input_data.kwargs['x'] = x_world[AttnId]
        input_data.kwargs['expertIds'] = expertIds_world[AttnId]
        input_data.kwargs['activeMaskOptional'] = activeMaskOptional_world[AttnId]
        # expertRankTable 需要重新生成（因为依赖随机种子，确保与 Golden 一致）
        input_data.kwargs['expertRankTable'] = gen_expert_rank_table(
            moeRankNum, localmoeExpert, moeExpertNum, sharedExpertNum,
            rankNumPerSharedExpert, L, M)

    def init_hccl_comm(self, rank_id: int, world_size: int):
        """
        初始化 HCCL 通信域，建立 EP（Expert Parallelism）进程组。

        步骤:
          1. 设置当前 rank 对应的 NPU 设备
          2. 初始化 torch.distributed 进程组（backend=hccl）
          3. 创建 EP 域进程组（包含所有 world_size 个 rank）
          4. 获取 HCCL 通信句柄 ep_hcomm_info（字符串形式），传入算子

        参数:
          rank_id: 当前进程的 rank ID
          world_size: 总进程数

        返回:
          ep_hcomm_info: HCCL 通信句柄字符串
          ep_group: torch.distributed 进程组对象
        """
        rank_per_dev = 16  # 每个设备支持 16 个 rank
        torch_npu.npu.set_device(rank_id % rank_per_dev)
        logging.error(f'[INFO] device_{rank_id} 创建HCCL通信链路')

        if not dist.is_initialized():
            dist.init_process_group(
                backend="hccl", rank=rank_id, world_size=world_size,
                init_method=f'tcp://127.0.0.1:50001')
            logging.error(f"device_{rank_id} init_process_group success")
        else:
            logging.error(f"device_{rank_id} 进程组已存在，跳过初始化")

        # 创建 EP 域：包含所有 rank（全量通信域）
        ep_ranks_list = [list(range(0, world_size, 1))]
        logging.error(f"device {rank_id} 初始化EP域")
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank_id in ep_ranks:
                ep_group = tmp_group

        # 获取 HCCL 通信句柄，这是算子需要的 group 参数
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
        logging.error('ep_hcomm_info: ' + ep_hcomm_info)
        return ep_hcomm_info, ep_group

    def attn2ffn_get_npu(self, ep_group, AttnRankNum: int, MircoBatchNum: int,
                         BS: int, ExpertNumPerToken: int, Token_size: int,
                         H: int, model_dtype):
        """
        FFN Worker 从 peermem 窗口读取 NPU 上的计算结果。

        peermem 窗口的内存布局:
          ┌─────────────────────────────────────────────────────────┐
          │ ffn_window_states (token info table)                    │
          │ 形状: [AttnRankNum, MircoBatchNum, 1+1+BS*ExpertNum]   │
          │ 类型: int32                                            │
          ├─────────────────────────────────────────────────────────┤
          │ ffn_token_data (token data)                            │
          │ 形状: [AttnRankNum, MircoBatchNum, BS, ExpertNum, T]   │
          │ 类型: fp16/bf16/int8                                   │
          └─────────────────────────────────────────────────────────┘

        使用 torchair.llm_datadist.create_npu_tensors 直接映射到 peermem 地址，
        无需数据拷贝即可读取 Attention Worker 写入的数据。

        参数:
          ep_group: 进程组，用于获取窗口地址
          AttnRankNum: Attention Worker 数量
          MircoBatchNum: micro_batch 数
          BS: batch size
          ExpertNumPerToken: 每个 token 路由到的专家数
          Token_size: token 数据维度（量化时为 H，非量化时为 H）
          H: hidden_size
          model_dtype: 数据类型（fp16/bf16/int8）

        返回:
          ffn_window_states: token info table 张量
          ffn_data_tensor: token data 张量（截取到 H 维度）
        """
        window_addr = 0x12c101800000  # 默认 win 区地址（备用）
        # 从 ep_group 获取实际的 peermem 窗口地址
        window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()

        # 第一段：ffn_window_states（token info table）
        # 形状 [AttnRankNum, MircoBatchNum, 1 + 1 + BS * ExpertNumPerToken]
        # 其中 1+1+BS*ExpertNumPerToken = [flag, layerId, expertId_0, expertId_1, ...]
        npu_state_tensor = torchair.llm_datadist.create_npu_tensors(
            [AttnRankNum, MircoBatchNum, 1 + 1 + BS * ExpertNumPerToken], torch.int32, [window_addr]
        )
        ffn_window_states = npu_state_tensor[0]

        # 计算 token info table 占用的字节数，512 对齐后作为 data 段的起始偏移
        ffn_token_size = ffn_window_states.element_size() * ffn_window_states.numel()
        ffn_token_area_size = (ffn_token_size + 511) // 512 * 512

        # 第二段：ffn_token_data（token data）
        # 形状 [AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken, Token_size]
        ffn_data_tensors = torchair.llm_datadist.create_npu_tensors(
            [AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken, Token_size],
            model_dtype, [window_addr + ffn_token_area_size]
        )
        ffn_data_tensor = ffn_data_tensors[0]
        # 截取到实际 hidden_size H（Token_size 可能因对齐而大于 H）
        ffn_data_tensor = ffn_data_tensor[:, :, :, :, :H]
        return ffn_window_states, ffn_data_tensor

    def attn2ffn_golden(self, FFNRankNum: int, AttnRankNum: int, MircoBatchNum: int,
                        BS: int, ExpertNumPerToken: int, input_data: InputDataset):
        """
        CPU 侧模拟 attention_to_ffn 算子逻辑，计算 Golden 真值。

        核心流程:
          1. 初始化输出张量: ffn_token_state, ffn_token_data, token_data_mask, attn_token_state
          2. 遍历每个 Attention Worker 的每个 token:
             - 根据 activeMask 判断 token 是否有效
             - 有效 token: 调用 att2ffn() 将 token 数据和状态分发到目标 FFN rank
             - 无效 token: attn_token_state 对应位置置 1（标记为 padding）
          3. 根据 syncFlag 决定是否将所有 FFN rank 的 flag 位置 1

        输出张量说明:
          ffn_token_state: [FFNRankNum, AttnRankNum, MircoBatchNum, 1+1+BS*ExpertNumPerToken]
            - [0]: flag 位，1 表示有数据写入
            - [1]: layerId
            - [2:]: 每个 token 每个专家的 expertId
          ffn_token_data: [FFNRankNum, AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken, H]
            - 每个 token 每个专家的 hidden_states 数据
          token_data_mask: 与 ffn_token_data 同形状的 bool 掩码，标记有效数据位置
          attn_token_state: [AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken]
            - Attention 侧的状态信息，无效 token 对应位置置 1
        """
        # 固定随机种子，确保与 init_by_input_data 中的生成一致
        torch.manual_seed(0)
        random.seed(0)

        X = input_data.kwargs['x'].shape[0]
        K = input_data.kwargs['expertIds'].shape[2]
        H = input_data.kwargs['x'].shape[2]
        Token_size = H
        quant_mode = input_data.kwargs['quantMode']
        is_attn2ffn_scales = input_data.kwargs['is_attn2ffn_scales']
        is_quant = (quant_mode > 0)
        sharedExpertNum = input_data.kwargs['sharedExpertNum']
        moeExpertNum = input_data.kwargs['moeExpertNum']

        # ── 初始化输出张量 ──
        # FFN 侧 token info table: [flag, layerId, expertId_0, expertId_1, ...]
        ffn_token_state = torch.zeros(FFNRankNum, AttnRankNum, MircoBatchNum, 1 + 1 + BS * ExpertNumPerToken)
        # FFN 侧 token data: 每个 token 每个专家的 hidden_states
        ffn_token_data = torch.zeros(FFNRankNum, AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken, Token_size)
        # 有效数据掩码，用于精度比较时只比较有效位置
        token_data_mask = torch.zeros(FFNRankNum, AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken, Token_size, dtype=torch.bool)
        # Attention 侧状态信息
        attn_token_state = torch.zeros(AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken)

        # 从 kwargs 中获取全局输入数据
        x_world = input_data.kwargs['x_world']
        expertIds_world = input_data.kwargs['expertIds_world']
        scalesOptional = input_data.kwargs['scalesOptional']
        activeMaskOptional_world = input_data.kwargs['activeMaskOptional_world']
        expertRankTable = input_data.kwargs['expertRankTable']
        syncFlag = input_data.kwargs['syncFlag']

        def load_banance(session_id, layer_id, expert_id):
            """
            负载均衡函数：根据 sessionId 和 expertRankTable 确定目标 FFN rank 和本地专家 ID。

            逻辑:
              1. 从 expertRankTable 读取该专家的候选 rank 数量 count
              2. 用 sessionId % count 选择一个候选（实现负载均衡）
              3. 返回目标 (rank_id, exp_id)

            这与 op_kernel 中的逻辑一致:
              dstRankId = expertRankTable[layerId][expertId][1 + (sessionId % count) * 2]
              dstExpId  = expertRankTable[layerId][expertId][2 + (sessionId % count) * 2]
            """
            count = expertRankTable[layer_id, expert_id, 0]
            idx = session_id % count
            dst_rank_id = expertRankTable[layer_id, expert_id, 1 + idx * 2]
            dst_exp_id = expertRankTable[layer_id, expert_id, 2 + idx * 2]
            return (dst_rank_id, dst_exp_id)

        def quant_process(y, scale):
            """
            动态量化处理：将 FP16/BF16 数据量化为 int8。

            量化公式:
              1. 如果有外部 scale，先乘上 scale: y = y * scale
              2. 求绝对值最大值: y_max = max(|y|)
              3. 计算动态 scale: dynamic_scale = 127.0 / y_max
              4. 量化: y = round(y * dynamic_scale) → int8

            这与 op_kernel 中的动态量化逻辑一致。
            """
            if scale is not None:
                y = y.to(torch.float32) * scale

            y_max, _ = y.abs().max(dim=-1)
            y_max[y_max == 0] = 1.0  # 防止除零
            dynamic_scales = torch.tensor(127.0, dtype=torch.float32) / y_max
            y = (y * dynamic_scales.view(-1, 1)).round().to(torch.int8)
            return y

        def att2ffn(row_idx, x_i, expert_id, attn_id):
            """
            模拟单个 token 的 attention_to_ffn 分发逻辑。

            对每个 token 的每个 topk 路由专家:
              1. 通过 load_banance 确定目标 FFN rank 和本地专家 ID
              2. 将 token 数据写入 ffn_token_data 对应位置
              3. 设置 ffn_token_state 的 flag、layerId、expertId

            对每个共享专家:
              1. 同样通过 load_banance 确定目标
              2. 写入数据和状态

            参数:
              row_idx: token 在 BS 中的索引
              x_i: token 的 hidden_states，形状 [H]
              expert_id: 该 token 的 topk 专家 ID 列表，长度 K
              attn_id: 当前 Attention Worker 的 ID
            """
            nonlocal ffn_token_data, ffn_token_state, token_data_mask
            scales_i = None
            default_layer_id = 0
            default_micro_batch_id = 0

            # ── 处理路由专家 (MoE experts) ──
            for topk_id, expert_i in enumerate(expert_id):
                rank_id, exp_id = load_banance(attn_id, default_layer_id, expert_i)
                if is_quant and is_attn2ffn_scales:
                    scales_i = scalesOptional[default_layer_id, expert_i]

                # 写入 token 数据（量化时需要 quant_process）
                ffn_token_data[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :H] = \
                    x_i if not is_quant else quant_process(x_i.to('cpu'), scales_i.to('cpu'))
                # 标记有效数据位置
                token_data_mask[rank_id, attn_id, default_micro_batch_id, row_idx, topk_id, :H] = \
                    torch.ones(H, dtype=torch.bool)

                # 写入 token info table
                ffn_token_state[rank_id, attn_id, default_micro_batch_id, 0] = 1  # flag=1 表示有数据
                ffn_token_state[rank_id, attn_id, default_micro_batch_id, 1] = default_layer_id  # layerId
                # expertId 存储在 [2:] 区域，偏移 = row_idx * ExpertNumPerToken + topk_id + 2
                ffn_token_state[rank_id, attn_id, default_micro_batch_id,
                                (row_idx * ExpertNumPerToken + topk_id + 2)] = exp_id.float()

            # ── 处理共享专家 (shared experts) ──
            # 共享专家排在路由专家之后，索引从 K 开始
            for i in range(sharedExpertNum):
                shared_rank_id, shared_exp_id = load_banance(attn_id, default_layer_id, moeExpertNum + i)
                # 写入 token info table（共享专家的 expertId 位置 = row_idx * ExpertNumPerToken + K + i + 2）
                ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id, 0] = 1
                ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id, 1] = default_layer_id
                ffn_token_state[shared_rank_id, attn_id, default_micro_batch_id,
                                (row_idx * ExpertNumPerToken + K + i + 2)] = shared_exp_id.float()

                if is_quant and is_attn2ffn_scales:
                    scales_i = scalesOptional[default_layer_id, moeExpertNum + i]
                # 写入 token 数据
                ffn_token_data[shared_rank_id, attn_id, default_micro_batch_id, row_idx, K + i, :H] = \
                    x_i if not is_quant else quant_process(x_i.to('cpu'), scales_i.to('cpu'))
                token_data_mask[shared_rank_id, attn_id, default_micro_batch_id, row_idx, K + i, :H] = \
                    torch.ones(H, dtype=torch.bool)

        # ── 主循环：遍历所有 Attention Worker 的所有 token ──
        for attn_id in range(AttnRankNum):
            for token_id in range(BS):
                if activeMaskOptional_world[attn_id, X - 1, token_id]:
                    # 有效 token：调用 att2ffn 分发到目标 FFN rank
                    att2ffn(token_id, x_world[attn_id, X - 1, token_id],
                            expertIds_world[attn_id, X - 1, token_id], attn_id)
                else:
                    # 无效 token（padding）：attn_token_state 对应位置置 1
                    attn_token_state[attn_id, X - 1, token_id, :] = torch.ones(ExpertNumPerToken)

        # ── 同步模式处理 ──
        # syncFlag=0 表示同步模式，所有 FFN rank 的 flag 位均置 1
        # 这与 op_kernel 中的逻辑一致：同步模式下 FFN 侧不需要等待 flag 就可以开始处理
        if syncFlag == 0:
            ffn_token_state[:, :, :, 0] = 1

        return ffn_token_data, token_data_mask, ffn_token_state.to(torch.int32), attn_token_state.to(torch.int32)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """
        ATK 框架调用的主执行入口，根据执行模式分三路:

        1. CPU 真值分支 (self.name == "cpu" and is_bm):
           调用 attn2ffn_golden() 计算 CPU Golden 结果。
           FFN rank 返回 (ffn_token_state, ffn_token_data)
           Attn rank 返回 (attn_token_state, attn_window_states_company)

        2. Attention Worker NPU 分支 (rank_id >= FFNRankNum):
           - 初始化 HCCL 通信域和 peermem 窗口
           - 调用 npu_attention_to_ffn 将 token 发送到 FFN Worker
           - 支持 graph_type=0（单算子模式）和 graph_type=1（入图模式）
           - 如果启用了 activeMask，从窗口读取 Attention 侧状态
           - 返回 (attn_window_states, attn_window_states_company)

        3. FFN Worker NPU 分支 (rank_id < FFNRankNum):
           - sleep 等待 Attention Worker 完成写入
           - 从 peermem 窗口读取 ffn_token_state 和 ffn_token_data
           - 返回 (attn_states_npu, attn_data_npu)

        ATK 框架会自动收集所有 rank 的返回值，并与 CPU Golden 结果进行精度对比。
        """
        logging.error(f'开始执行call!')

        # ── 从 input_data.kwargs 解包算子输入参数 ──
        x = input_data.kwargs['x']
        sessionId = input_data.kwargs['sessionId']
        microBatchId = input_data.kwargs['microBatchId']
        layerId = input_data.kwargs['layerId']
        expertIds = input_data.kwargs['expertIds']
        expertRankTable = input_data.kwargs['expertRankTable']
        scalesOptional = input_data.kwargs['scalesOptional']
        activeMaskOptional = input_data.kwargs['activeMaskOptional']
        ffnTokenInfoTableShape = input_data.kwargs['ffnTokenInfoTableShape']
        ffnTokenDataShape = input_data.kwargs['ffnTokenDataShape']
        attnTokenInfoTableShape = input_data.kwargs['attnTokenInfoTableShape']
        moeExpertNum = input_data.kwargs['moeExpertNum']
        quant_mode = input_data.kwargs['quantMode']
        syncFlag = input_data.kwargs['syncFlag']
        is_mask = input_data.kwargs['is_mask']

        # ── 提取维度参数 ──
        rank_id = self.dist_task_info.rank
        X = input_data.kwargs['x'].shape[0]
        L = input_data.kwargs['expertRankTable'].shape[0]
        M = input_data.kwargs['expertRankTable'].shape[2]
        K = input_data.kwargs['expertIds'].shape[2]
        H = input_data.kwargs['x'].shape[2]
        Token_size = input_data.kwargs['ffnTokenDataShape'][4]
        world_size = input_data.kwargs['worldSize']
        AttnRankNum = input_data.kwargs['ffnTokenInfoTableShape'][0]
        FFNRankNum = world_size - AttnRankNum
        BS = input_data.kwargs['x'].shape[1]
        localmoeExpert = input_data.kwargs['localmoeExpert']
        MircoBatchNum = input_data.kwargs['ffnTokenDataShape'][1]
        ExpertNumPerToken = input_data.kwargs['ffnTokenDataShape'][3]

        moeRankNum = int(moeExpertNum / localmoeExpert)
        sharedRankNum = FFNRankNum - moeRankNum
        sharedExpertNum = input_data.kwargs['sharedExpertNum']
        rankNumPerSharedExpert = int(sharedRankNum / sharedExpertNum)
        graph_type = input_data.kwargs['graphType']
        is_quant = (quant_mode > 0)
        is_attn2ffn_scales = input_data.kwargs['is_attn2ffn_scales']
        model_dtype = input_data.kwargs['x'].dtype
        if is_quant:
            model_dtype = torch.int8

        # 初始化占位输出张量
        attn_token_state = torch.zeros(AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken)
        attn_window_states_company = torch.zeros(MircoBatchNum, BS, ExpertNumPerToken)

        # ═══════════════════════════════════════════════════════════════
        # 分支1: CPU 真值计算
        # ═══════════════════════════════════════════════════════════════
        if self.name == "cpu" and self.dist_task_info.is_bm:
            ffn_token_data, token_data_mask, ffn_token_state, attn_token_state = \
                self.attn2ffn_golden(FFNRankNum, AttnRankNum, MircoBatchNum, BS, ExpertNumPerToken, input_data)

            if rank_id < FFNRankNum:
                # FFN Worker: 返回该 rank 对应的 token_state 和 token_data
                logging.error(f'****************rank:{rank_id}, ffnRank_id:{rank_id}, ffn_token_statea_golden: {ffn_token_state[rank_id]}')
                logging.error(f'****************rank:{rank_id}, ffnRank_id:{rank_id}, ffn_token_data_golden: {ffn_token_data[rank_id]}')
                logging.error(f'****************rank:{rank_id}, ffnRank_id:{rank_id}, token_data_mask_golden: {token_data_mask[rank_id]}')
                return ffn_token_state[rank_id], ffn_token_data[rank_id]
            else:
                # Attention Worker: 返回 attn_token_state 和占位张量
                attnRank_id = rank_id - FFNRankNum
                logging.error(f'****************rank:{rank_id}, attnRank_id:{attnRank_id}, attn_token_state: {attn_token_state[attnRank_id]}')
                return attn_token_state[attnRank_id], attn_window_states_company

        # ═══════════════════════════════════════════════════════════════
        # 分支2 & 3: NPU 执行
        # ═══════════════════════════════════════════════════════════════
        else:
            # ── 初始化 HCCL 通信域（仅首次） ──
            logging.error('==============开始建立通信域===================')
            if self.ep_hcomm_info is None:
                self.ep_hcomm_info, self.ep_group = self.init_hccl_comm(rank_id, world_size)
                logging.debug(f"[rank:{rank_id}] ep_group: {self.ep_group}.")
            logging.error('===================通信域建立完成======================')
            ep_group = self.ep_group
            ep_hcomm_info = self.ep_hcomm_info

            # ── 注册 peermem 窗口 ──
            # Attention Worker 与 FFN Worker 之间通过 peermem 窗口通信
            # target_ranks: 当前 rank 需要通信的对端 rank 列表
            if FFNRankNum <= rank_id:
                # Attention 卡 → 需要与所有 FFN 卡通信
                target_ranks = list(range(FFNRankNum))
            else:
                # FFN 卡 → 需要与所有 Attention 卡通信
                target_ranks = list(range(FFNRankNum, world_size))
            logging.error(f'target_ranks: {target_ranks}')

            # 注册 200MB 的 peermem 窗口，用于 Attention ↔ FFN 数据交换
            window_size = 1024 * 1024 * 200
            backend = ep_group._get_backend(torch.device('npu'))
            if hasattr(backend, '_window_register_and_exchange'):
                backend._window_register_and_exchange(window_size, target_ranks)
            else:
                logging.error(f"Warning: _window_register_and_exchange not available, using alternative")

            # ── 分支2: Attention Worker NPU 执行 ──
            if rank_id >= FFNRankNum:
                logging.error(f'****************begin torch_npu.npu_attention_to_ffn{rank_id}***********')

                if graph_type == 1:
                    # 传统入图模式：使用编译后的模型，静态 shape + 在线编译
                    logging.error('graph_type=1, 测试场景: 传统入图模式, 静态shape+在线编译场景')
                    self.compiled_model(
                        x=x.npu(),
                        session_id=sessionId.to(torch.int32).npu(),
                        micro_batch_id=microBatchId.to(torch.int32).npu(),
                        layer_id=layerId.to(torch.int32).npu(),
                        expert_ids=expertIds.to(torch.int32).npu(),
                        expert_rank_table=expertRankTable.to(torch.int32).npu(),
                        group=ep_hcomm_info,
                        world_size=world_size,
                        ffn_token_info_table_shape=ffnTokenInfoTableShape,
                        ffn_token_data_shape=ffnTokenDataShape,
                        attn_token_info_table_shape=attnTokenInfoTableShape,
                        moe_expert_num=moeExpertNum,
                        scales=scalesOptional.to(torch.float32).npu() if is_quant and is_attn2ffn_scales else None,
                        active_mask=activeMaskOptional.to(torch.bool).npu(),
                        quant_mode=quant_mode,
                        sync_flag=syncFlag,
                    )

                elif graph_type == 0:
                    # 单算子模式：直接调用模型，不走图编译
                    logging.error('graph_type=0, 测试场景: 单算子模式')
                    self.model(
                        x=x.npu(),
                        session_id=sessionId.to(torch.int32).npu(),
                        micro_batch_id=microBatchId.to(torch.int32).npu(),
                        layer_id=layerId.to(torch.int32).npu(),
                        expert_ids=expertIds.to(torch.int32).npu(),
                        expert_rank_table=expertRankTable.to(torch.int32).npu(),
                        group=ep_hcomm_info,
                        world_size=world_size,
                        ffn_token_info_table_shape=ffnTokenInfoTableShape,
                        ffn_token_data_shape=ffnTokenDataShape,
                        attn_token_info_table_shape=attnTokenInfoTableShape,
                        moe_expert_num=moeExpertNum,
                        scales=scalesOptional.to(torch.float32).npu() if is_quant and is_attn2ffn_scales else None,
                        active_mask=activeMaskOptional.to(torch.bool).npu(),
                        quant_mode=quant_mode,
                        sync_flag=syncFlag,
                    )

                else:
                    logging.error(f"[rank:{rank_id}] Unsupport graph_type:{graph_type}.")

                # 初始化 Attention 侧窗口状态
                attn_window_states = torch.zeros(MircoBatchNum, BS, ExpertNumPerToken)

                # 如果启用了 activeMask，从 peermem 窗口读取 Attention 侧状态
                # 这对应 op_kernel 中 FFN 回写 attnTokenInfoTable 的逻辑
                if is_mask:
                    window_addr = ep_group._get_backend(torch.device('npu'))._get_window_mem().data_ptr()
                    npu_state_tensor = torchair.llm_datadist.create_npu_tensors(
                        [int(MircoBatchNum), int(BS), int(ExpertNumPerToken)], torch.int32, [window_addr]
                    )
                    attn_window_states = npu_state_tensor
                logging.error(f'rank {rank_id} npu finished!')
                logging.error(f'****************rank:{rank_id}, attn_window_states: {attn_window_states}')

                # 等待 FFN Worker 读取完成（确保 peermem 数据不被提前释放）
                time.sleep(30)
                return attn_window_states, attn_window_states_company

            # ── 分支3: FFN Worker NPU 读取 ──
            if rank_id < FFNRankNum:
                logging.error(f'****************begin torch_npu.npu_attention_to_ffn rank:{rank_id}***********')
                # FFN Worker 不主动调用算子，等待 Attention Worker 写入 peermem 后读取结果
                time.sleep(30)
                attn_states_npu, attn_data_npu = self.attn2ffn_get_npu(
                    ep_group, int(AttnRankNum), int(MircoBatchNum), int(BS),
                    int(ExpertNumPerToken), int(Token_size), int(H), model_dtype)
                # 将张量从 NPU 移到 CPU（ATK 框架会自动与 CPU Golden 对比）
                logging.error(f'****************rank:{rank_id}, token_states_npu : {attn_states_npu}')
                logging.error(f'****************rank:{rank_id}, token_data_npu: {attn_data_npu}')
                logging.error(f'rank {rank_id} npu finished!')

                return attn_states_npu, attn_data_npu
