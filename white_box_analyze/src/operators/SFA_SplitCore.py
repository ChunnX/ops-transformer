import math
from typing import Dict, Any, List, Tuple
from src.core.base_operator import BaseOperator
from src.core.op_registry import OperatorRegistry
from pprint import pprint

@OperatorRegistry.register("SFA")

class SFAOperator(BaseOperator):
    def __init__(self, op_name, chipversion):
        self.chipversion = chipversion
    
    def check_single_type(self, length, split_size):
        if split_size <= 0:
            raise ValueError("切分大小必须大于0")
        if length == 0:
            return -1

        whole = length // split_size
        rem = length % split_size

        if rem == 0:
            return 0
        else:
            return 1 if whole == 0 else 2


    def check_n_sparse3(self, S1, S2, split_n, sparse_blocksize, topk_num):
        """sparse3专用：下三角长度 + sparse_limit约束"""
        # if S2 < S1:
        #     raise ValueError(f"S2({S2}) 不能小于 S1({S1})")

        all_whole = True
        all_tail = True
        sparse_limit = sparse_blocksize * topk_num

        for i in range(1, S1 + 1):
            n_valid = S2 - S1 + i
            effective_n_valid = min(n_valid, sparse_limit)
            row_type = self.check_single_type(effective_n_valid, split_n)

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


    def check_n_sparse3_vec1_vf_mode(self, S1, S2, split_n, sparse_blocksize, topk_num):
        """sparse3专用：下三角长度 + sparse_limit约束"""
        # if S2 < S1:
        #     raise ValueError(f"S2({S2}) 不能小于 S1({S1})")

        all_whole = True
        all_tail = True
        sparse_limit = sparse_blocksize * topk_num
        vf_mode_list = []
        for i in range(1, S1 + 1):
            n_valid = S2 - S1 + i
            effective_n_valid = min(n_valid, sparse_limit)
            vf_mode_list.extend(self.check_vec1_vf_mode(effective_n_valid, split_n))
        return vf_mode_list


    def check_vec1_vf_mode(self, S2, vec1_split_N):
        """A5 vec1专用：S2在0-64/64-128/=128三种模式的哪一种"""
        vf_mode_list = []
        if S2 >= vec1_split_N:
            vf_mode_list.append(0)
        S2_tail = S2 % vec1_split_N
        if S2_tail == 0:
            return vf_mode_list
        if S2_tail > 64:
            vf_mode_list.append(1)
        else:
            vf_mode_list.append(2)
        return vf_mode_list


    def global_batch_result(self, arr):
        types_list = [x for x in arr if x != -1]
        all0 = all(t == 0 for t in types_list)
        all1 = all(t == 1 for t in types_list)
        if all0:
            return 0
        elif all1:
            return 1
        else:
            return 2


    def analyze_VF_global_batch_result(self, types_list):
        if not types_list:
            return 0  # 异常：空数组
        
        # 使用集合获取唯一值
        unique_values = set(types_list)
        
        # 检查是否包含非法元素
        if not unique_values.issubset({0, 1, 2}):
            return 0
        
        # 场景1: 全128
        if unique_values == {0}:
            return 0
        # 场景2: 只有大于64的块
        elif unique_values == {1}:
            return 1
        # 场景3: 只有小于64的块
        elif unique_values == {2}:
            return 2
        # 场景4: 只有128的块和大于64的块
        elif unique_values == {0, 1}:
            return 3
        # 场景5: 只有128的块和大于64的块
        elif unique_values == {0, 2}:
            return 4
        return 0


    # 新增：适配sparse3的PA_block类型判断函数
    def check_pa_block_sparse3(self, S1, S2, PA_blocksize, sparse_blocksize, topk_num):
        """
        sparse3模式下PA_block类型判断（逐行计算下三角有效长度）
        返回值：0=全整块 | 1=全尾块 | 2=混合
        """
        # if S2 < S1:
        #     raise ValueError(f"S2({S2}) 不能小于 S1({S1})")
        if PA_blocksize <= 0:
            return 0  # PA_blocksize≤0默认整块

        all_whole = True
        all_tail = True
        sparse_limit = sparse_blocksize * topk_num

        # 逐行计算下三角有效长度，并判断每行的PA_block类型
        for i in range(1, S1 + 1):
            # 下三角第i行的有效长度（同self.check_n_sparse3逻辑）
            n_valid = S2 - S1 + i
            effective_n_valid = min(n_valid, sparse_limit)

            # 判断当前行的PA_block类型
            row_type = self.check_single_type(effective_n_valid, PA_blocksize)

            # 更新全局判断：只要有一行不是整块，all_whole=False；只要有一行不是尾块，all_tail=False
            if row_type == 0:
                all_tail = False
            elif row_type == 1:
                all_whole = False
            else:
                all_whole = False
                all_tail = False

        # 汇总该行Batch的PA_block类型
        if all_whole:
            return 0
        elif all_tail:
            return 1
        else:
            return 2


    def generate_split_result(self,
        B, G, D_nope, D_rope,
        act_q, act_k,
        inter_split_M, inter_split_N,
        vec0_split_M, vec0_split_N,
        vec1_split_M, vec1_split_N,
        vec2_split_M, vec2_split_N,
        mm1_L1_split_M, mm1_L1_N_split, mm1_L1_K_split,
        mm1_L0_split_M, mm1_L0_N_split, mm1_L0_K_split,
        mm2_L1_split_M, mm2_L1_N_split, mm2_L1_K_split,
        mm2_L0_split_M, mm2_L0_N_split, mm2_L0_K_split,
        sparse_mode=0,
        sparse_blocksize=0,
        topk_num=0,
        PA_blocksize=0  # PA_block切块大小
    ):
        # ========== 输入约束校验 ==========
        if len(act_q) != B or len(act_k) != B:
            raise ValueError(f"act_q/act_k 长度必须等于 B={B}")
        if sparse_blocksize < 0 or topk_num < 0:
            raise ValueError(f"sparse_blocksize({sparse_blocksize})和topk_num({topk_num})必须非负")
        if PA_blocksize < 0:
            raise ValueError(f"PA_blocksize({PA_blocksize})必须非负")

        # ========== 模板类型判断（仅计算一次）==========
        # sparse_blocksize ≤4 → V模板(0)；>4 → C模板(1)
        if self.chipversion == "A3":
            template_code = "0" if sparse_blocksize <= 4 else "1"
        else:
            template_code = "0"
        template_type = "V模板" if template_code == "0" else "C模板"

        # vec0_split_M 不参与G倍数校验
        m_check = [
            inter_split_M,
            vec1_split_M, vec2_split_M,
            mm1_L1_split_M, mm1_L0_split_M,
            mm2_L1_split_M, mm2_L0_split_M
        ]
        m_names = [
            "inter_split_M",
            "vec1_split_M", "vec2_split_M",
            "mm1_L1_split_M", "mm1_L0_split_M",
            "mm2_L1_split_M", "mm2_L0_split_M"
        ]
        M_split_size = 64 if (G == 128 and self.chipversion == "A5") else G
        for v, n in zip(m_check, m_names):
            if v % M_split_size != 0:
                raise ValueError(f"{n}({v}) 必须是 G({G}) 的整数倍, 或在A5且G=128时等于64")

        D_total = D_nope + D_rope
        sparse_limit = sparse_blocksize * topk_num

        # mm1/mm2 约束
        if inter_split_M < mm1_L1_split_M or mm1_L1_split_M < mm1_L0_split_M:
            raise ValueError("mm1 M 需满足 inter ≥ L1 ≥ L0")
        if inter_split_N < mm1_L1_N_split or mm1_L1_N_split < mm1_L0_N_split:
            raise ValueError("mm1 N 需满足 inter ≥ L1 ≥ L0")
        if mm1_L1_K_split < mm1_L0_K_split:
            raise ValueError("mm1 K 需满足 L1 ≥ L0")
        if inter_split_M < mm2_L1_split_M or mm2_L1_split_M < mm2_L0_split_M:
            raise ValueError("mm2 M 需满足 inter ≥ L1 ≥ L0")
        if (self.chipversion in ['A2', 'A3'] and inter_split_N < mm2_L1_N_split) or mm2_L1_N_split < mm2_L0_N_split:
            raise ValueError("mm2 N 需满足 L1 ≥ L0, A2 A3场景下还需满足inter ≥ L1")
        if mm2_L1_K_split < mm2_L0_K_split:
            raise ValueError("mm2 K 需满足 L1 ≥ L0")

        # 计算各阶段S1切分粒度
        inter_S1 = inter_split_M // M_split_size
        vec0_S1 = vec0_split_M // M_split_size
        vec1_S1 = vec1_split_M // M_split_size
        vec2_S1 = vec2_split_M // M_split_size
        mm1_L1_S1 = mm1_L1_split_M // M_split_size
        mm1_L0_S1 = mm1_L0_split_M // M_split_size
        mm2_L1_S1 = mm2_L1_split_M // M_split_size
        mm2_L0_S1 = mm2_L0_split_M // M_split_size

        # 初始化结果列表
        inter_Ms, inter_Ns = [], []
        vec0_Ms = []  # vec0 A2A3 仅保留M轴结果
        vec0_Ns = []
        vec1_Ms, vec1_Ns, vec1_N_VFs = [], [], []
        vec2_Ms, vec2_Ns = [], []
        mm1_L1_Ms, mm1_L1_Ns, mm1_L1_Ks = [], [], []
        mm1_L0_Ms, mm1_L0_Ns, mm1_L0_Ks = [], [], []
        mm2_L1_Ms, mm2_L1_Ns, mm2_L1_Ks = [], [], []
        mm2_L0_Ms, mm2_L0_Ns, mm2_L0_Ks = [], [], []
        PA_block_types = []  # 存储每个Batch的PA_block判断结果
        # 新增：记录每个Batch的PA_block逐行详情（用于调试）
        PA_block_batch_details = []

        # 逐Batch计算
        for b in range(B):
            S1 = act_q[b]
            S2_original = act_k[b]

            # 计算有效S2（非sparse3模式下的全局有效长度）
            if sparse_mode == 0:
                S2_effective = min(S2_original, sparse_limit)
            else:
                S2_effective = S2_original

            # ========== 修正：PA_block阶段判断（适配sparse3下三角长度） ==========
            if sparse_mode == 3:
                # sparse3模式：逐行计算下三角有效长度，判断PA_block类型
                pa_block_type = self.check_pa_block_sparse3(
                    S1=S1,
                    S2=S2_effective,
                    PA_blocksize=PA_blocksize,
                    sparse_blocksize=sparse_blocksize,
                    topk_num=topk_num
                )
                # 记录该Batch的逐行详情
                batch_detail = []
                for i in range(1, S1 + 1):
                    n_valid = S2_effective - S1 + i
                    effective_n_valid = min(n_valid, sparse_limit)
                    row_type = self.check_single_type(effective_n_valid, PA_blocksize) if PA_blocksize>0 else 0
                    batch_detail.append({
                        "row": i,
                        "n_valid": n_valid,
                        "effective_n_valid": effective_n_valid,
                        "row_type": row_type,
                        "row_desc": "全整块" if row_type==0 else "全尾块" if row_type==1 else "混合"
                    })
                PA_block_batch_details.append(batch_detail)
            else:
                # 非sparse3模式：使用原逻辑（全局S2_effective）
                if PA_blocksize <= 0:
                    pa_block_type = 0  # PA_blocksize≤0默认整块
                else:
                    pa_block_type = self.check_single_type(S2_effective, PA_blocksize)
                PA_block_batch_details.append([{"mode": "non-sparse3", "S2_effective": S2_effective, "type": pa_block_type}])

            PA_block_types.append(pa_block_type)

            # 原始长度定义
            K_mm1   = D_total
            K_mm2   = S2_effective
            N_mm2   = D_nope
            N_vec2  = D_nope
            N_vec0  = D_total
            N_vec1  = S2_effective
            N_inter = S2_effective
            N_mm1   = S2_effective

            # ---------------- 核间阶段 ----------------
            inter_M = self.check_single_type(S1, inter_S1)
            if sparse_mode == 3:
                inter_N = self.check_n_sparse3(S1, S2_effective, inter_split_N, sparse_blocksize, topk_num)
            else:
                inter_N = self.check_single_type(S2_effective, inter_split_N)
            inter_Ms.append(inter_M)
            inter_Ns.append(inter_N)

            # ---------------- vec0阶段 ----------------
            # 仅判断M轴：原始长度=S2_effective，切块长度=sparse_blocksize
            if sparse_blocksize <= 0:
                vec0_M = 1  # 切块长度无效时默认尾块
            else:
                vec0_M = self.check_single_type(S2_effective, sparse_blocksize)
            vec0_Ms.append(vec0_M)
            if self.chipversion == "A5":
                if sparse_mode == 3:
                    vec0_N = self.check_n_sparse3(S1, S2_effective, vec0_split_N, sparse_blocksize, topk_num)
                else:
                    vec0_N = self.check_single_type(S2_effective, vec0_split_N)
                vec0_Ns.append(vec0_N)

            # ---------------- mm1_sparse_block阶段（前置） ----------------
            # 仅N轴：原始长度=mm1_L1_N_split，切块大小=sparse_blocksize
            if sparse_blocksize <= 0:
                mm1_sparse_block_N = 1  # 切块长度无效时默认尾块
            else:
                mm1_sparse_block_N = self.check_single_type(mm1_L1_N_split, sparse_blocksize)

            # ---------------- mm1_L1阶段 ----------------
            mm1_L1_M = self.check_single_type(S1, mm1_L1_S1)
            if sparse_mode == 3:
                mm1_L1_N = self.check_n_sparse3(S1, S2_effective, mm1_L1_N_split, sparse_blocksize, topk_num)
            else:
                mm1_L1_N = self.check_single_type(S2_effective, mm1_L1_N_split)
            mm1_L1_K = self.check_single_type(K_mm1, mm1_L1_K_split)
            mm1_L1_Ms.append(mm1_L1_M)
            mm1_L1_Ns.append(mm1_L1_N)
            mm1_L1_Ks.append(mm1_L1_K)

            # ---------------- mm1_L0阶段 ----------------
            mm1_L0_M = self.check_single_type(S1, mm1_L0_S1)
            if sparse_mode == 3:
                mm1_L0_N = self.check_n_sparse3(S1, S2_effective, mm1_L0_N_split, sparse_blocksize, topk_num)
            else:
                mm1_L0_N = self.check_single_type(S2_effective, mm1_L0_N_split)
            mm1_L0_K = self.check_single_type(K_mm1, mm1_L0_K_split)
            mm1_L0_Ms.append(mm1_L0_M)
            mm1_L0_Ns.append(mm1_L0_N)
            mm1_L0_Ks.append(mm1_L0_K)

            # ---------------- vec1阶段 ----------------
            vec1_M = self.check_single_type(S1, vec1_S1)
            if sparse_mode == 3:
                vec1_N = self.check_n_sparse3(S1, S2_effective, vec1_split_N, sparse_blocksize, topk_num)
            else:
                vec1_N = self.check_single_type(S2_effective, vec1_split_N)
            vec1_Ms.append(vec1_M)
            vec1_Ns.append(vec1_N)
            if self.chipversion == "A5":
                if sparse_mode == 3:
                    vec1_N_inner = self.check_n_sparse3_vec1_vf_mode(S1, S2_effective, vec1_split_N, sparse_blocksize, topk_num)
                else:
                    vec1_N_inner = self.check_vec1_vf_mode(S2_effective, vec1_split_N)
                vec1_N_VFs.extend(vec1_N_inner)
            # ---------------- mm2_sparse_block阶段（前置） ----------------
            # 仅K轴：原始长度=mm2_L0_K_split，切块大小=sparse_blocksize
            if sparse_blocksize <= 0:
                mm2_sparse_block_K = 1  # 切块长度无效时默认尾块
            else:
                mm2_sparse_block_K = self.check_single_type(mm2_L0_K_split, sparse_blocksize)

            # ---------------- mm2_L1阶段 ----------------
            mm2_L1_M = self.check_single_type(S1, mm2_L1_S1)
            mm2_L1_N = self.check_single_type(N_mm2, mm2_L1_N_split)
            if sparse_mode == 3:
                mm2_L1_K = self.check_n_sparse3(S1, S2_effective, mm2_L1_K_split, sparse_blocksize, topk_num)
            else:
                mm2_L1_K = self.check_single_type(S2_effective, mm2_L1_K_split)
            mm2_L1_Ms.append(mm2_L1_M)
            mm2_L1_Ns.append(mm2_L1_N)
            mm2_L1_Ks.append(mm2_L1_K)

            # ---------------- mm2_L0阶段 ----------------
            mm2_L0_M = self.check_single_type(S1, mm2_L0_S1)
            mm2_L0_N = self.check_single_type(N_mm2, mm2_L0_N_split)
            if sparse_mode == 3:
                mm2_L0_K = self.check_n_sparse3(S1, S2_effective, mm2_L0_K_split, sparse_blocksize, topk_num)
            else:
                mm2_L0_K = self.check_single_type(S2_effective, mm2_L0_K_split)
            mm2_L0_Ms.append(mm2_L0_M)
            mm2_L0_Ns.append(mm2_L0_N)
            mm2_L0_Ks.append(mm2_L0_K)

            # ---------------- vec2阶段 ----------------
            vec2_M = self.check_single_type(S1, vec2_S1)
            if sparse_mode == 3:
                vec2_N = self.check_n_sparse3(S1, S2_effective, vec2_split_N, sparse_blocksize, topk_num)
            else:
                vec2_N = self.check_single_type(S2_effective, vec2_split_N)
            vec2_Ms.append(vec2_M)
            vec2_Ns.append(vec2_N)

        # ========== 计算PA_block全局结果 ==========
        PA_block_global = self.global_batch_result(PA_block_types)
        PA_block_desc = "全整块" if PA_block_global == 0 else "全尾块" if PA_block_global == 1 else "混合"

        # ========== 生成各阶段原始编码（无模板位）==========
        stage_codes_raw = {
            "template_bit":         template_code,  # 模板位独立作为一项
            "PA_block":             f"{PA_block_global}",  # 新增：PA_block编码
            "inter-core-split":     f"{self.global_batch_result(inter_Ms)}{self.global_batch_result(inter_Ns)}",
            "intra-core-vec0":      f"{self.global_batch_result(vec0_Ms)}{self.global_batch_result(vec0_Ns)}",
            # mm1阶段：sparse_block → L1 → L0
            "mm1_sparse_block":     f"{mm1_sparse_block_N}",
            "intra-core-mm1_L1":    f"{self.global_batch_result(mm1_L1_Ms)}{self.global_batch_result(mm1_L1_Ns)}{self.global_batch_result(mm1_L1_Ks)}",
            "intra-core-mm1_L0":    f"{self.global_batch_result(mm1_L0_Ms)}{self.global_batch_result(mm1_L0_Ns)}{self.global_batch_result(mm1_L0_Ks)}",
            "intra-core-vec1":      f"{self.global_batch_result(vec1_Ms)}{self.global_batch_result(vec1_Ns)}{self.analyze_VF_global_batch_result(vec1_N_VFs)}",
            # mm2阶段：sparse_block → L1 → L0
            "mm2_sparse_block":     f"{mm2_sparse_block_K}",
            "intra-core-mm2_L1":    f"{self.global_batch_result(mm2_L1_Ms)}{self.global_batch_result(mm2_L1_Ns)}{self.global_batch_result(mm2_L1_Ks)}",
            "intra-core-mm2_L0":    f"{self.global_batch_result(mm2_L0_Ms)}{self.global_batch_result(mm2_L0_Ns)}{self.global_batch_result(mm2_L0_Ks)}",
            "intra-core-vec2":      f"{self.global_batch_result(vec2_Ms)}{self.global_batch_result(vec2_Ns)}"
        }

        # ========== 定义最终编码顺序（核心调整：PA_block在模板位后）==========
        final_stage_order = [
            "PA_block",  # 新增：PA_block放在模板位后、其他阶段前
            "inter-core-split",
            "intra-core-vec0",
            # mm1：sparse_block → L1 → L0
            "mm1_sparse_block",
            "intra-core-mm1_L1",
            "intra-core-mm1_L0",
            "intra-core-vec1",
            # mm2：sparse_block → L1 → L0
            "mm2_sparse_block",
            "intra-core-mm2_L1",
            "intra-core-mm2_L0",
            "intra-core-vec2"
        ]

        # ========== 拼接整体编码（首位模板位+按新顺序拼接阶段）==========
        full_code = stage_codes_raw["template_bit"]  # 首位模板位
        for stage in final_stage_order:
            full_code += stage_codes_raw[stage]

        # ========== 最终返回结果 ==========
        final = {
            "stage_codes": stage_codes_raw,          # 包含模板位+各阶段原始编码
            "final_stage_order": final_stage_order,  # 标注最终编码顺序
            "full_code": full_code,                  # 整体编码（首位模板位）
            "template_info": f"{template_type}(sparse_blocksize={sparse_blocksize}, 模板位={template_code})",
            # 新增：PA_block阶段信息（含sparse3下三角详情）
            "PA_block_info": {
                "axis": "S2_effective (sparse3=下三角逐行长度)",
                "sparse_mode": sparse_mode,
                "original_length_per_batch": [min(s, sparse_limit) for s in act_k] if sparse_mode==0 else act_k,
                "split_size": PA_blocksize,
                "type_code": PA_block_global,
                "type_desc": PA_block_desc,
                "default_rule": "PA_blocksize ≤0 时默认全整块（编码0）",
                "batch_details": PA_block_batch_details  # 新增：逐行详情（调试用）
            },
            # sparse_block阶段计算信息
            "sparse_block_info": {
                "mm1_sparse_block": {
                    "axis": "N轴",
                    "original_length": mm1_L1_N_split,
                    "split_size": sparse_blocksize,
                    "type_code": mm1_sparse_block_N,
                    "type_desc": "全整块" if mm1_sparse_block_N == 0 else "全尾块" if mm1_sparse_block_N == 1 else "混合"
                },
                "mm2_sparse_block": {
                    "axis": "K轴",
                    "original_length": mm2_L0_K_split,
                    "split_size": sparse_blocksize,
                    "type_code": mm2_sparse_block_K,
                    "type_desc": "全整块" if mm2_sparse_block_K == 0 else "全尾块" if mm2_sparse_block_K == 1 else "混合"
                }
            }
        }
        return final
    
    def simulate_whitebox_logic(self, aic_num: int, aiv_num: int, input_params: Dict) -> Dict[str, Any]:
        """模拟C++算子内部的白盒逻辑 - 修复全局预处理架构"""
        params = input_params
        print("params = ", params)
        B = params['B']
        G = params['G']

        D_nope = params['D_nope']
        D_rope = params['D_rope']
        D_total = D_nope + D_rope
        act_q = params['actual_seq_lengths_query']
        act_k = params['actual_seq_lengths_key']

        sparse_blocksize = params['sparse_blocksize']
        topk_num = params['topk_num']
        sparse_mode = params['sparsemode']
        PA_blocksize = params['PA_BlockSize']
        if self.chipversion in ['A2', 'A3']:
            inter_split_M, inter_split_N = 256, 512

            vec0_split_M, vec0_split_N = G, 576 # 每次处理一行s1
            vec1_split_M, vec1_split_N = 256, 512
            vec2_split_M, vec2_split_N = 256, 512

            mm1_L1_split_M, mm1_L1_N_split, mm1_L1_K_split = 128, 128, 288
            mm1_L0_split_M, mm1_L0_N_split, mm1_L0_K_split = 128, 128, 96
            mm2_L1_split_M, mm2_L1_N_split, mm2_L1_K_split = 128, 128, 256 # mm2 K轴为256，但是稀疏搬运的时候按照128搬
            mm2_L0_split_M, mm2_L0_N_split, mm2_L0_K_split = 128, 128, 128
        else: # A5
            M_split_size = 64 if G == 128 else G
            inter_split_M, inter_split_N = M_split_size, 128

            vec0_split_M, vec0_split_N = G, 16 # 每次处理一行s1，s2满16个搬出
            vec1_split_M, vec1_split_N = M_split_size, 128
            vec2_split_M, vec2_split_N = M_split_size, 512

            mm1_L1_split_M, mm1_L1_N_split, mm1_L1_K_split = M_split_size, 128, 576
            mm1_L0_split_M, mm1_L0_N_split, mm1_L0_K_split = M_split_size, 128, 128
            mm2_L1_split_M, mm2_L1_N_split, mm2_L1_K_split = M_split_size, 512, 128
            mm2_L0_split_M, mm2_L0_N_split, mm2_L0_K_split = M_split_size, 128, 128

        # 运行函数（新增PA_blocksize入参）
        try:
            res = self.generate_split_result(
                B, G, D_nope, D_rope,
                act_q, act_k,
                inter_split_M, inter_split_N,
                vec0_split_M, vec0_split_N,
                vec1_split_M, vec1_split_N,
                vec2_split_M, vec2_split_N,
                mm1_L1_split_M, mm1_L1_N_split, mm1_L1_K_split,
                mm1_L0_split_M, mm1_L0_N_split, mm1_L0_K_split,
                mm2_L1_split_M, mm2_L1_N_split, mm2_L1_K_split,
                mm2_L0_split_M, mm2_L0_N_split, mm2_L0_K_split,
                sparse_mode=sparse_mode,
                sparse_blocksize=sparse_blocksize,
                topk_num=topk_num,
                PA_blocksize=PA_blocksize  # 传入新增入参
            )
            print("=" * 80)
            print(f"版本4：新增PA_block阶段（模板位后） | sparse_mode={sparse_mode}")
            print("=" * 80)
            print(f"模板判断规则：sparse_blocksize ≤4 → V模板(0)；>4 → C模板(1)")
            print(f"当前：{res['template_info']}")
            print(f"vec0 M轴规则：原始长度=S2_effective，切块长度=sparse_blocksize；A5额外规则：原始长度=S1，切块大小={vec0_split_M}")
            print(f"sparse_limit={sparse_blocksize*topk_num}，原始S2={act_k} → 有效S2={[min(s, sparse_blocksize*topk_num) for s in act_k]}")
            print("-" * 80)
            print("【新增PA_block阶段信息】")
            print(f"PA_block：原始长度（各Batch有效S2）={res['PA_block_info']['original_length_per_batch']} | 切块大小={PA_blocksize} | 结果={res['PA_block_info']['type_desc']}（编码{res['PA_block_info']['type_code']}）")
            print("-" * 80)
            print("【新增sparse_block阶段计算信息】")
            print(f"mm1_sparse_block：N轴 | 原始长度={res['sparse_block_info']['mm1_sparse_block']['original_length']} | 切块大小={sparse_blocksize} | 结果={res['sparse_block_info']['mm1_sparse_block']['type_desc']}（编码{res['sparse_block_info']['mm1_sparse_block']['type_code']}）")
            print(f"mm2_sparse_block：K轴 | 原始长度={res['sparse_block_info']['mm2_sparse_block']['original_length']} | 切块大小={sparse_blocksize} | 结果={res['sparse_block_info']['mm2_sparse_block']['type_desc']}（编码{res['sparse_block_info']['mm2_sparse_block']['type_code']}）")
            print("-" * 80)
            print("【各编码项（按最终顺序打印，模板位独立）】")
            # 先打印模板位
            print(f"{'template_bit':<22} 编码：{res['stage_codes']['template_bit']:>6} | 0=V模板，1=C模板")
            # 按新顺序打印所有阶段编码
            stage_desc = {
                "PA_block": "S2_effective → PA_block阶段（模板位后）",
                "inter-core-split": "M+N → 核间阶段",
                "intra-core-vec0": "M+N → vec0阶段（A2A3仅M轴）",
                "mm1_sparse_block": "N → mm1_sparse_block（仅N轴，前置）",
                "intra-core-mm1_L1": "M+N+K → mm1_L1阶段",
                "intra-core-mm1_L0": "M+N+K → mm1_L0阶段",
                "intra-core-vec1": "M+N+VF → vec1阶段，其中VF仅A5",
                "mm2_sparse_block": "K → mm2_sparse_block（仅K轴，前置）",
                "intra-core-mm2_L1": "M+N+K → mm2_L1阶段",
                "intra-core-mm2_L0": "M+N+K → mm2_L0阶段",
                "intra-core-vec2": "M+N → vec2阶段"
            }
            for stage in res["final_stage_order"]:
                print(f"{stage:<22} 编码：{res['stage_codes'][stage]:>6} | {stage_desc[stage]}")
            print("-" * 80)
            print(f"【整体编码（首位=模板位 + PA_block + 其他阶段）】")
            print(f"full_code = {res['full_code']}")
            print("-" * 80)

            # 编码说明
            print("编码含义：")
            print(f"  整体编码首位：{res['stage_codes']['template_bit']}（{res['template_info'].split('(')[0]}）")
            print("  阶段编码位：0=全整块 | 1=全尾块 | 2=混合")
            print(f"  整体编码构成顺序：")
            print(f"    template_bit → PA_block → inter-core-split → vec0 → mm1_sparse_block → mm1_L1 → mm1_L0 → vec1 → mm2_sparse_block → mm2_L1 → mm2_L0 → vec2")
        except Exception as e:
            print(f"运行报错：{e}")
        return res