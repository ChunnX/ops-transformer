import math
from typing import Dict, Any, List, Tuple
from src.core.base_operator import BaseOperator
from src.core.op_registry import OperatorRegistry
from pprint import pprint

@OperatorRegistry.register("LI")

class LIOperator(BaseOperator):
    def __init__(self, op_name, chipversion):
        pass
    
    def check_single_type(self, length, split_size):
        """
        单轴、单长度、单切分：判断是 0/1/2
        0=纯整块，1=纯尾块，2=混合
        """
        if split_size <= 0:
            raise ValueError("切分大小必须大于0")
        if length == 0:
            return 1

        whole = length // split_size
        rem = length % split_size

        if rem == 0:
            return 0
        else:
            return 1 if whole == 0 else 2


    def check_n_sparse3(self, S1, S2, split_n):
        """
        sparse_mode=3 专用：计算N轴块类型（45°有效区域逻辑）
        核心规则：
        1. 先拦截 S2 < S1 的非法情况（直接抛异常）
        2. 第i行（1≤i≤S1）的N有效长度 = S2 - S1 + i
        3. 综合所有行：全纯整→0、全纯尾→1、其余→2
        """
        if S2 < S1:
            raise ValueError(f"S2({S2}) 不能小于 S1({S1})（无法形成45°有效区域）")

        all_whole = True
        all_tail = True

        for i in range(1, S1 + 1):
            n_valid = S2 - S1 + i
            row_type = self.check_single_type(n_valid, split_n)

            if row_type == 0:
                all_tail = False
            elif row_type == 1:
                all_whole = False
            else:
                all_whole = False
                all_tail = False

        if all_whole:
            return 0
        elif all_tail:
            return 1
        else:
            return 2


    # 新增：PA尾块判断函数（适配sparse3下三角长度）
    def check_pa_tail_block(self, S1, S2, PA_blocksize, sparse_mode=0):
        """
        判断PA是否有尾块（适配sparse3下三角逻辑）
        :param S1: 当前batch的S1值
        :param S2: 当前batch的S2值
        :param PA_blocksize: PA切块大小
        :param sparse_mode: 稀疏模式（0=普通，3=下三角）
        :return: 0=全整块（无尾块） | 1=全尾块 | 2=混合
        """
        if PA_blocksize <= 0:
            return 0  # PA_blocksize≤0时默认无尾块（全整块）

        if sparse_mode == 3:
            # sparse3模式：逐行计算下三角有效长度，判断每行PA尾块类型后汇总
            if S2 < S1:
                raise ValueError(f"S2({S2}) 不能小于 S1({S1})（无法形成45°有效区域）")

            all_whole = True
            all_tail = True

            for i in range(1, S1 + 1):
                pa_valid_length = S2 - S1 + i  # 下三角第i行PA有效长度
                row_type = self.check_single_type(pa_valid_length, PA_blocksize)

                if row_type == 0:
                    all_tail = False
                elif row_type == 1:
                    all_whole = False
                else:
                    all_whole = False
                    all_tail = False

            if all_whole:
                return 0
            elif all_tail:
                return 1
            else:
                return 2
        else:
            # 普通模式：直接用全局S2判断PA尾块类型
            return self.check_single_type(S2, PA_blocksize)


    def global_batch_result(self, types_list):
        """
        综合所有 batch：全0→0、全1→1、其他→2
        """
        all0 = all(t == 0 for t in types_list)
        all1 = all(t == 1 for t in types_list)
        if all0:
            return 0
        elif all1:
            return 1
        else:
            return 2


    def generate_split_result(self,
        B, G, D,
        act_q, act_k,
        inter_split_M, inter_split_N,
        L1_split_M, L1_N_split, L1_K_split,
        L0_split_M, L0_N_split, L0_K_split,
        vec_split_M, vec_split_N,
        sparse_mode=0,
        PA_blocksize=0  # 新增：PA切块大小入参
    ):
        """
        输出字典格式的切分结果，键说明：
        - 'inter-core-split'：核间阶段（M+N，字符串，如"22"）
        - 'intra-core-L1'：L1阶段（M+N+K，字符串，如"000"）
        - 'intra-core-L0'：L0阶段（M+N+K，字符串，如"000"）
        - 'intra-core-Vec'：Vec阶段（M+N，字符串，如"00"）
        - 'pa-tail-block'：PA尾块判断（1位字符串，如"1"） 新增
        字符串每位含义：0=全纯整，1=全纯尾，2=混合
        """
        # ========== 输入约束校验 ==========
        if len(act_q) != B or len(act_k) != B:
            raise ValueError(f"act_q/act_k 长度必须等于 B={B}")
        if PA_blocksize < 0:
            raise ValueError(f"PA_blocksize({PA_blocksize})必须非负")

        # M轴所有阶段切分必须是G的整数倍
        m_split_list = [inter_split_M, L1_split_M, L0_split_M, vec_split_M]
        m_split_names = ["inter_split_M", "L1_split_M", "L0_split_M", "vec_split_M"]
        for split_val, split_name in zip(m_split_list, m_split_names):
            if split_val % G != 0:
                raise ValueError(f"{split_name}({split_val}) 必须是 G({G}) 的整数倍")

        # 切分粒度大小约束（inter≥L1≥L0）
        if inter_split_M < L1_split_M or L1_split_M < L0_split_M:
            raise ValueError(f"M轴切分粒度需满足 inter({inter_split_M})≥L1({L1_split_M})≥L0({L0_split_M})")
        if inter_split_N < L1_N_split or L1_N_split < L0_N_split:
            raise ValueError(f"N轴切分粒度需满足 inter({inter_split_N})≥L1({L1_N_split})≥L0({L0_N_split})")
        if L1_K_split < L0_K_split:
            raise ValueError(f"K轴切分粒度需满足 L1({L1_K_split})≥L0({L0_K_split})")

        # ========== 计算各阶段S1实际切分粒度 ==========
        inter_S1_split = inter_split_M // G
        L1_S1_split = L1_split_M // G
        L0_S1_split = L0_split_M // G
        vec_S1_split = vec_split_M // G

        # ========== 逐Batch计算各阶段各轴类型 ==========
        # 初始化各阶段结果列表（按轴存储）
        inter_M_list, inter_N_list = [], []
        L1_M_list, L1_N_list, L1_K_list = [], [], []
        L0_M_list, L0_N_list, L0_K_list = [], [], []
        vec_M_list, vec_N_list = [], []
        pa_tail_list = []  # 新增：存储每个batch的PA尾块判断结果

        for b in range(B):
            S1 = act_q[b]
            S2 = act_k[b]
            K = D

            # 1. 核间阶段（M、N）
            inter_M = self.check_single_type(S1, inter_S1_split)
            inter_N = self.check_n_sparse3(S1, S2, inter_split_N) if sparse_mode == 3 else self.check_single_type(S2, inter_split_N)
            inter_M_list.append(inter_M)
            inter_N_list.append(inter_N)

            # 2. L1阶段（M、N、K）
            L1_M = self.check_single_type(S1, L1_S1_split)
            L1_N = self.check_n_sparse3(S1, S2, L1_N_split) if sparse_mode == 3 else self.check_single_type(S2, L1_N_split)
            L1_K = self.check_single_type(K, L1_K_split)
            L1_M_list.append(L1_M)
            L1_N_list.append(L1_N)
            L1_K_list.append(L1_K)

            # 3. L0阶段（M、N、K）
            L0_M = self.check_single_type(S1, L0_S1_split)
            L0_N = self.check_n_sparse3(S1, S2, L0_N_split) if sparse_mode == 3 else self.check_single_type(S2, L0_N_split)
            L0_K = self.check_single_type(K, L0_K_split)
            L0_M_list.append(L0_M)
            L0_N_list.append(L0_N)
            L0_K_list.append(L0_K)

            # 4. Vec阶段（M、N）
            vec_M = self.check_single_type(S1, vec_S1_split)
            vec_N = self.check_n_sparse3(S1, S2, vec_split_N) if sparse_mode == 3 else self.check_single_type(S2, vec_split_N)
            vec_M_list.append(vec_M)
            vec_N_list.append(vec_N)

            # 新增：5. PA尾块判断（逐batch）
            pa_tail = self.check_pa_tail_block(S1, S2, PA_blocksize, sparse_mode)
            pa_tail_list.append(pa_tail)

        # ========== 全局综合 + 拼接为字符串 ==========
        final_result = {
            # 核间：M + N → 2位字符串
            "inter-core-split": f"{self.global_batch_result(inter_M_list)}{self.global_batch_result(inter_N_list)}",
            # L1：M + N + K → 3位字符串
            "intra-core-L1": f"{self.global_batch_result(L1_M_list)}{self.global_batch_result(L1_N_list)}{self.global_batch_result(L1_K_list)}",
            # L0：M + N + K → 3位字符串
            "intra-core-L0": f"{self.global_batch_result(L0_M_list)}{self.global_batch_result(L0_N_list)}{self.global_batch_result(L0_K_list)}",
            # Vec：M + N → 2位字符串
            "intra-core-Vec": f"{self.global_batch_result(vec_M_list)}{self.global_batch_result(vec_N_list)}",
            # 新增：PA尾块判断（1位字符串）
            "pa-tail-block": f"{self.global_batch_result(pa_tail_list)}"
        }

        return final_result
    
    def simulate_whitebox_logic(self, aic_num: int, aiv_num: int, input_params: Dict) -> Dict[str, Any]:
        """模拟C++算子内部的白盒逻辑 - 修复全局预处理架构"""
        # 第一步：提取基础参数
        print("input_params = ", input_params)
        B = input_params.get('B')
        G = input_params.get('G')
        D = input_params.get('D')
        act_q = input_params.get('actual_seq_lengths_query')
        act_k = input_params.get('actual_seq_lengths_key')
        sparse_mode = input_params.get('sparse_mode')
        PA_blocksize = input_params.get('PA_BlockSize')
        inter_split_M, inter_split_N = 512, 512
        L1_split_M, L1_N_split, L1_K_split = 256, 256, 128
        L0_split_M, L0_N_split, L0_K_split = 128, 128, 128
        vec_split_M, vec_split_N = 256, 512
        dimensions_state = self.generate_split_result(
            B, G, D, act_q, act_k,
            inter_split_M, inter_split_N,
            L1_split_M, L1_N_split, L1_K_split,
            L0_split_M, L0_N_split, L0_K_split,
            vec_split_M, vec_split_N,
            sparse_mode,
            PA_blocksize=PA_blocksize
        )
        return dimensions_state