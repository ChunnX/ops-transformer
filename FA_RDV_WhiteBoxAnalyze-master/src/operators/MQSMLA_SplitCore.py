import math
from typing import Dict, Any, List, Tuple
from src.core.base_operator import BaseOperator
from src.core.op_registry import OperatorRegistry
from pprint import pprint

TEMPLATE_MODE_SWA = 0
TEMPLATE_MODE_HCA = 1
TEMPLATE_MODE_CSA = 2

TEMPLATE_MODE_DESC = {
    0: "SWA",
    1: "HCA",
    2: "CSA"
}

@OperatorRegistry.register("MQSMLA")

class MQSMLAOperator(BaseOperator):
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

    def check_n_causal(self, S1, S2, split_n):
        """cmp_mask_mode=3 (RightDownCausal): 下三角有效长度逐行判断"""
        all_whole = True
        all_tail = True

        for i in range(1, S1 + 1):
            n_valid = S2 - S1 + i
            if n_valid <= 0:
                continue
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

    def check_n_band(self, S1, S2, split_n, win_left, win_right):
        """ori_mask_mode=4 (Band): 滑动窗口有效长度逐行判断"""
        all_whole = True
        all_tail = True

        for i in range(1, S1 + 1):
            win_start = max(0, i - 1 - win_left)
            win_end = min(S2, i + win_right)
            n_valid = win_end - win_start
            if n_valid <= 0:
                continue
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

    def check_n_causal_vec1_vf_mode(self, S1, S2, split_n):
        """sparse3专用：下三角长度 + vf_mode"""
        vf_mode_list = []
        for i in range(1, S1 + 1):
            n_valid = S2 - S1 + i
            if n_valid <= 0:
                continue
            vf_mode_list.extend(self.check_vec1_vf_mode(n_valid, split_n))
        return vf_mode_list

    def check_n_band_vec1_vf_mode(self, S1, S2, split_n, win_left, win_right):
        """Band模式：滑动窗口长度 + vf_mode"""
        vf_mode_list = []
        for i in range(1, S1 + 1):
            win_start = max(0, i - 1 - win_left)
            win_end = min(S2, i + win_right)
            n_valid = win_end - win_start
            if n_valid <= 0:
                continue
            vf_mode_list.extend(self.check_vec1_vf_mode(n_valid, split_n))
        return vf_mode_list

    def global_batch_result(self, arr):
        types_list = [x for x in arr if x != -1]
        if not types_list:
            return 0
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
            return 0

        unique_values = set(types_list)

        if not unique_values.issubset({0, 1, 2}):
            return 0

        if unique_values == {0}:
            return 0
        elif unique_values == {1}:
            return 1
        elif unique_values == {2}:
            return 2
        elif unique_values == {0, 1}:
            return 3
        elif unique_values == {0, 2}:
            return 4
        return 0

    def check_pa_block(self, S1, S2, PA_blocksize, mask_mode, win_left=127, win_right=0):
        """PA block类型判断，适配Band/Causal mask模式"""
        if PA_blocksize <= 0:
            return 0

        if mask_mode == 3:
            return self.check_n_causal(S1, S2, PA_blocksize)
        elif mask_mode == 4:
            return self.check_n_band(S1, S2, PA_blocksize, win_left, win_right)
        else:
            return self.check_single_type(S2, PA_blocksize)

    def generate_split_result(self,
        B, G, D_nope, D_rope,
        act_q, act_ori_k, act_cmp_k,
        inter_split_M, inter_split_N,
        vec0_split_M, vec0_split_N,
        vec1_split_M, vec1_split_N,
        vec2_split_M, vec2_split_N,
        mm1_L1_split_M, mm1_L1_N_split, mm1_L1_K_split,
        mm1_L0_split_M, mm1_L0_N_split, mm1_L0_K_split,
        mm2_L1_split_M, mm2_L1_N_split, mm2_L1_K_split,
        mm2_L0_split_M, mm2_L0_N_split, mm2_L0_K_split,
        template_mode=0,
        is_split_g=0,
        quant_mode=1,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        sparse_block_count=0,
        PA_ori_blocksize=0,
        PA_cmp_blocksize=0
    ):
        if len(act_q) != B or len(act_ori_k) != B:
            raise ValueError(f"act_q/act_ori_k 长度必须等于 B={B}")
        if PA_ori_blocksize < 0 or PA_cmp_blocksize < 0:
            raise ValueError(f"PA_blocksize必须非负")

        M_split_size = 64 if (G >= 128 and is_split_g) else G

        D_total = D_nope + D_rope

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
        for v, n in zip(m_check, m_names):
            if v % M_split_size != 0:
                raise ValueError(f"{n}({v}) 必须是 M_split_size({M_split_size}) 的整数倍")

        if inter_split_M < mm1_L1_split_M or mm1_L1_split_M < mm1_L0_split_M:
            raise ValueError("mm1 M 需满足 inter >= L1 >= L0")
        if inter_split_N < mm1_L1_N_split or mm1_L1_N_split < mm1_L0_N_split:
            raise ValueError("mm1 N 需满足 inter >= L1 >= L0")
        if mm1_L1_K_split < mm1_L0_K_split:
            raise ValueError("mm1 K 需满足 L1 >= L0")
        if inter_split_M < mm2_L1_split_M or mm2_L1_split_M < mm2_L0_split_M:
            raise ValueError("mm2 M 需满足 inter >= L1 >= L0")
        if mm2_L1_N_split < mm2_L0_N_split:
            raise ValueError("mm2 N 需满足 L1 >= L0")
        if mm2_L1_K_split < mm2_L0_K_split:
            raise ValueError("mm2 K 需满足 L1 >= L0")

        inter_S1 = inter_split_M // M_split_size
        vec0_S1 = vec0_split_M // M_split_size
        vec1_S1 = vec1_split_M // M_split_size
        vec2_S1 = vec2_split_M // M_split_size
        mm1_L1_S1 = mm1_L1_split_M // M_split_size
        mm1_L0_S1 = mm1_L0_split_M // M_split_size
        mm2_L1_S1 = mm2_L1_split_M // M_split_size
        mm2_L0_S1 = mm2_L0_split_M // M_split_size

        inter_Ms, inter_Ns = [], []
        vec0_Ms, vec0_Ns = [], []
        vec1_Ms, vec1_Ns, vec1_N_VFs = [], [], []
        vec2_Ms, vec2_Ns = [], []
        mm1_L1_Ms, mm1_L1_Ns, mm1_L1_Ks = [], [], []
        mm1_L0_Ms, mm1_L0_Ns, mm1_L0_Ks = [], [], []
        mm2_L1_Ms, mm2_L1_Ns, mm2_L1_Ks = [], [], []
        mm2_L0_Ms, mm2_L0_Ns, mm2_L0_Ks = [], [], []
        PA_ori_block_types = []
        PA_cmp_block_types = []

        has_cmp = template_mode in (TEMPLATE_MODE_HCA, TEMPLATE_MODE_CSA)

        for b in range(B):
            S1 = act_q[b]
            S2_ori = act_ori_k[b]

            if template_mode == TEMPLATE_MODE_CSA:
                S2_cmp = sparse_block_count if sparse_block_count > 0 else (act_cmp_k[b] if b < len(act_cmp_k) else 0)
            elif template_mode == TEMPLATE_MODE_HCA:
                S2_cmp = act_cmp_k[b] if b < len(act_cmp_k) else 0
            else:
                S2_cmp = 0

            pa_ori_block_type = self.check_pa_block(
                S1, S2_ori, PA_ori_blocksize, ori_mask_mode, ori_win_left, ori_win_right
            )
            PA_ori_block_types.append(pa_ori_block_type)

            if has_cmp and PA_cmp_blocksize > 0:
                pa_cmp_block_type = self.check_pa_block(
                    S1, S2_cmp, PA_cmp_blocksize, cmp_mask_mode
                )
            else:
                pa_cmp_block_type = 0
            PA_cmp_block_types.append(pa_cmp_block_type)

            S2_effective = S2_ori

            K_mm1 = D_total
            K_mm2 = S2_effective
            N_mm2 = D_nope
            N_vec2 = D_nope
            N_vec0 = D_total
            N_vec1 = S2_effective
            N_inter = S2_effective
            N_mm1 = S2_effective

            inter_M = self.check_single_type(S1, inter_S1)
            if ori_mask_mode == 3:
                inter_N = self.check_n_causal(S1, S2_effective, inter_split_N)
            elif ori_mask_mode == 4:
                inter_N = self.check_n_band(S1, S2_effective, inter_split_N, ori_win_left, ori_win_right)
            else:
                inter_N = self.check_single_type(S2_effective, inter_split_N)
            inter_Ms.append(inter_M)
            inter_Ns.append(inter_N)

            if PA_ori_blocksize <= 0:
                vec0_M = 1
            else:
                vec0_M = self.check_single_type(S2_effective, PA_ori_blocksize)
            vec0_Ms.append(vec0_M)
            if ori_mask_mode == 3:
                vec0_N = self.check_n_causal(S1, S2_effective, vec0_split_N)
            elif ori_mask_mode == 4:
                vec0_N = self.check_n_band(S1, S2_effective, vec0_split_N, ori_win_left, ori_win_right)
            else:
                vec0_N = self.check_single_type(S2_effective, vec0_split_N)
            vec0_Ns.append(vec0_N)

            mm1_L1_M = self.check_single_type(S1, mm1_L1_S1)
            if ori_mask_mode == 3:
                mm1_L1_N = self.check_n_causal(S1, S2_effective, mm1_L1_N_split)
            elif ori_mask_mode == 4:
                mm1_L1_N = self.check_n_band(S1, S2_effective, mm1_L1_N_split, ori_win_left, ori_win_right)
            else:
                mm1_L1_N = self.check_single_type(S2_effective, mm1_L1_N_split)
            mm1_L1_K = self.check_single_type(K_mm1, mm1_L1_K_split)
            mm1_L1_Ms.append(mm1_L1_M)
            mm1_L1_Ns.append(mm1_L1_N)
            mm1_L1_Ks.append(mm1_L1_K)

            mm1_L0_M = self.check_single_type(S1, mm1_L0_S1)
            if ori_mask_mode == 3:
                mm1_L0_N = self.check_n_causal(S1, S2_effective, mm1_L0_N_split)
            elif ori_mask_mode == 4:
                mm1_L0_N = self.check_n_band(S1, S2_effective, mm1_L0_N_split, ori_win_left, ori_win_right)
            else:
                mm1_L0_N = self.check_single_type(S2_effective, mm1_L0_N_split)
            mm1_L0_K = self.check_single_type(K_mm1, mm1_L0_K_split)
            mm1_L0_Ms.append(mm1_L0_M)
            mm1_L0_Ns.append(mm1_L0_N)
            mm1_L0_Ks.append(mm1_L0_K)

            vec1_M = self.check_single_type(S1, vec1_S1)
            if ori_mask_mode == 3:
                vec1_N = self.check_n_causal(S1, S2_effective, vec1_split_N)
                vec1_N_inner = self.check_n_causal_vec1_vf_mode(S1, S2_effective, vec1_split_N)
            elif ori_mask_mode == 4:
                vec1_N = self.check_n_band(S1, S2_effective, vec1_split_N, ori_win_left, ori_win_right)
                vec1_N_inner = self.check_n_band_vec1_vf_mode(S1, S2_effective, vec1_split_N, ori_win_left, ori_win_right)
            else:
                vec1_N = self.check_single_type(S2_effective, vec1_split_N)
                vec1_N_inner = self.check_vec1_vf_mode(S2_effective, vec1_split_N)
            vec1_Ms.append(vec1_M)
            vec1_Ns.append(vec1_N)
            vec1_N_VFs.extend(vec1_N_inner)

            mm2_L1_M = self.check_single_type(S1, mm2_L1_S1)
            mm2_L1_N = self.check_single_type(N_mm2, mm2_L1_N_split)
            if ori_mask_mode == 3:
                mm2_L1_K = self.check_n_causal(S1, S2_effective, mm2_L1_K_split)
            elif ori_mask_mode == 4:
                mm2_L1_K = self.check_n_band(S1, S2_effective, mm2_L1_K_split, ori_win_left, ori_win_right)
            else:
                mm2_L1_K = self.check_single_type(S2_effective, mm2_L1_K_split)
            mm2_L1_Ms.append(mm2_L1_M)
            mm2_L1_Ns.append(mm2_L1_N)
            mm2_L1_Ks.append(mm2_L1_K)

            mm2_L0_M = self.check_single_type(S1, mm2_L0_S1)
            mm2_L0_N = self.check_single_type(N_mm2, mm2_L0_N_split)
            if ori_mask_mode == 3:
                mm2_L0_K = self.check_n_causal(S1, S2_effective, mm2_L0_K_split)
            elif ori_mask_mode == 4:
                mm2_L0_K = self.check_n_band(S1, S2_effective, mm2_L0_K_split, ori_win_left, ori_win_right)
            else:
                mm2_L0_K = self.check_single_type(S2_effective, mm2_L0_K_split)
            mm2_L0_Ms.append(mm2_L0_M)
            mm2_L0_Ns.append(mm2_L0_N)
            mm2_L0_Ks.append(mm2_L0_K)

            vec2_M = self.check_single_type(S1, vec2_S1)
            vec2_N = self.check_single_type(N_vec2, vec2_split_N)
            vec2_Ms.append(vec2_M)
            vec2_Ns.append(vec2_N)

        PA_ori_block_global = self.global_batch_result(PA_ori_block_types)
        PA_cmp_block_global = self.global_batch_result(PA_cmp_block_types)

        stage_codes_raw = {
            "template_mode":          f"{template_mode}",
            "split_g":                f"{is_split_g}",
            "PA_ori_block":           f"{PA_ori_block_global}",
            "PA_cmp_block":           f"{PA_cmp_block_global}",
            "inter-core-split":       f"{self.global_batch_result(inter_Ms)}{self.global_batch_result(inter_Ns)}",
            "intra-core-vec0":        f"{self.global_batch_result(vec0_Ms)}{self.global_batch_result(vec0_Ns)}",
            "intra-core-mm1_L1":      f"{self.global_batch_result(mm1_L1_Ms)}{self.global_batch_result(mm1_L1_Ns)}{self.global_batch_result(mm1_L1_Ks)}",
            "intra-core-mm1_L0":      f"{self.global_batch_result(mm1_L0_Ms)}{self.global_batch_result(mm1_L0_Ns)}{self.global_batch_result(mm1_L0_Ks)}",
            "intra-core-vec1":        f"{self.global_batch_result(vec1_Ms)}{self.global_batch_result(vec1_Ns)}{self.analyze_VF_global_batch_result(vec1_N_VFs)}",
            "intra-core-mm2_L1":      f"{self.global_batch_result(mm2_L1_Ms)}{self.global_batch_result(mm2_L1_Ns)}{self.global_batch_result(mm2_L1_Ks)}",
            "intra-core-mm2_L0":      f"{self.global_batch_result(mm2_L0_Ms)}{self.global_batch_result(mm2_L0_Ns)}{self.global_batch_result(mm2_L0_Ks)}",
            "intra-core-vec2":        f"{self.global_batch_result(vec2_Ms)}{self.global_batch_result(vec2_Ns)}"
        }

        final_stage_order = [
            "template_mode",
            "split_g",
            "PA_ori_block",
            "PA_cmp_block",
            "inter-core-split",
            "intra-core-vec0",
            "intra-core-mm1_L1",
            "intra-core-mm1_L0",
            "intra-core-vec1",
            "intra-core-mm2_L1",
            "intra-core-mm2_L0",
            "intra-core-vec2"
        ]

        full_code = ""
        for stage in final_stage_order:
            full_code += stage_codes_raw[stage]

        final = {
            "stage_codes": stage_codes_raw,
            "final_stage_order": final_stage_order,
            "full_code": full_code,
            "template_mode_info": {
                "mode_code": template_mode,
                "mode_desc": TEMPLATE_MODE_DESC.get(template_mode, "UNKNOWN"),
                "is_split_g": is_split_g,
                "quant_mode": quant_mode,
                "cmp_ratio": cmp_ratio
            },
            "PA_ori_block_info": {
                "axis": "oriS2 (Band窗口逐行长度)",
                "ori_mask_mode": ori_mask_mode,
                "original_length_per_batch": act_ori_k,
                "split_size": PA_ori_blocksize,
                "type_code": PA_ori_block_global,
                "type_desc": "全整块" if PA_ori_block_global == 0 else "全尾块" if PA_ori_block_global == 1 else "混合",
                "default_rule": "PA_ori_blocksize <=0 时默认全整块（编码0）"
            },
            "PA_cmp_block_info": {
                "axis": "cmpS2 (Causal下三角逐行长度)",
                "cmp_mask_mode": cmp_mask_mode,
                "original_length_per_batch": act_cmp_k if has_cmp else [],
                "split_size": PA_cmp_blocksize,
                "type_code": PA_cmp_block_global,
                "type_desc": "全整块" if PA_cmp_block_global == 0 else "全尾块" if PA_cmp_block_global == 1 else "混合",
                "default_rule": "PA_cmp_blocksize <=0 或非HCA/CSA模式时默认全整块（编码0）"
            }
        }
        return final

    def simulate_whitebox_logic(self, aic_num: int, aiv_num: int, input_params: Dict) -> Dict[str, Any]:
        """模拟C++算子内部的白盒逻辑"""
        params = input_params
        print("params = ", params)
        B = params['B']
        G = params['G']

        D_nope = params['D_nope']
        D_rope = params['D_rope']
        D_total = D_nope + D_rope
        act_q = params['actual_seq_lengths_query']
        act_ori_k = params['actual_seq_lengths_ori_kv']
        act_cmp_k = params.get('actual_seq_lengths_cmp_kv', [0] * B)

        template_mode = params['template_mode']
        is_split_g = params['is_split_g']
        quant_mode = params['quant_mode']
        cmp_ratio = params['cmp_ratio']
        ori_mask_mode = params['ori_mask_mode']
        cmp_mask_mode = params['cmp_mask_mode']
        ori_win_left = params['ori_win_left']
        ori_win_right = params['ori_win_right']
        sparse_block_count = params['sparse_block_count']
        PA_ori_blocksize = params['PA_ori_block_size']
        PA_cmp_blocksize = params['PA_cmp_block_size']

        M_split_size = 64 if (G >= 128 and is_split_g) else G

        inter_split_M, inter_split_N = M_split_size, 128
        vec0_split_M, vec0_split_N = G, 16
        vec1_split_M, vec1_split_N = M_split_size, 128
        vec2_split_M, vec2_split_N = M_split_size, 512

        mm1_L1_split_M, mm1_L1_N_split, mm1_L1_K_split = M_split_size, 128, D_total
        mm1_L0_split_M, mm1_L0_N_split, mm1_L0_K_split = M_split_size, 128, 128
        mm2_L1_split_M, mm2_L1_N_split, mm2_L1_K_split = M_split_size, 512, 128
        mm2_L0_split_M, mm2_L0_N_split, mm2_L0_K_split = M_split_size, 128, 128

        try:
            res = self.generate_split_result(
                B, G, D_nope, D_rope,
                act_q, act_ori_k, act_cmp_k,
                inter_split_M, inter_split_N,
                vec0_split_M, vec0_split_N,
                vec1_split_M, vec1_split_N,
                vec2_split_M, vec2_split_N,
                mm1_L1_split_M, mm1_L1_N_split, mm1_L1_K_split,
                mm1_L0_split_M, mm1_L0_N_split, mm1_L0_K_split,
                mm2_L1_split_M, mm2_L1_N_split, mm2_L1_K_split,
                mm2_L0_split_M, mm2_L0_N_split, mm2_L0_K_split,
                template_mode=template_mode,
                is_split_g=is_split_g,
                quant_mode=quant_mode,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                sparse_block_count=sparse_block_count,
                PA_ori_blocksize=PA_ori_blocksize,
                PA_cmp_blocksize=PA_cmp_blocksize
            )

            print("=" * 80)
            print(f"MQSMLA白盒分析 | template_mode={TEMPLATE_MODE_DESC.get(template_mode, 'UNKNOWN')} | is_split_g={is_split_g}")
            print("=" * 80)
            print(f"M_split_size={M_split_size}, G={G}, D_nope={D_nope}, D_rope={D_rope}, D_total={D_total}")
            print(f"quant_mode={quant_mode}, cmp_ratio={cmp_ratio}")
            print(f"ori_mask_mode={ori_mask_mode}(Band), cmp_mask_mode={cmp_mask_mode}(Causal)")
            print(f"ori_win_left={ori_win_left}, ori_win_right={ori_win_right}")
            print(f"sparse_block_count(K2)={sparse_block_count}")
            print(f"act_q={act_q}, act_ori_k={act_ori_k}, act_cmp_k={act_cmp_k}")
            print("-" * 80)
            print("【PA_block阶段信息】")
            print(f"PA_ori_block：oriS2={act_ori_k} | block_size={PA_ori_blocksize} | 结果={res['PA_ori_block_info']['type_desc']}（编码{res['PA_ori_block_info']['type_code']}）")
            if template_mode in (TEMPLATE_MODE_HCA, TEMPLATE_MODE_CSA):
                print(f"PA_cmp_block：cmpS2={act_cmp_k} | block_size={PA_cmp_blocksize} | 结果={res['PA_cmp_block_info']['type_desc']}（编码{res['PA_cmp_block_info']['type_code']}）")
            print("-" * 80)
            print("【各编码项】")
            stage_desc = {
                "template_mode": "模板模式 0=SWA 1=HCA 2=CSA",
                "split_g": "Split-G 0=不切分G 1=切分G(G>64)",
                "PA_ori_block": "oriS2 -> PA_ori_block阶段",
                "PA_cmp_block": "cmpS2 -> PA_cmp_block阶段",
                "inter-core-split": "M+N -> 核间阶段",
                "intra-core-vec0": "M+N -> vec0阶段",
                "intra-core-mm1_L1": "M+N+K -> mm1_L1阶段",
                "intra-core-mm1_L0": "M+N+K -> mm1_L0阶段",
                "intra-core-vec1": "M+N+VF -> vec1阶段",
                "intra-core-mm2_L1": "M+N+K -> mm2_L1阶段",
                "intra-core-mm2_L0": "M+N+K -> mm2_L0阶段",
                "intra-core-vec2": "M+N -> vec2阶段"
            }
            for stage in res["final_stage_order"]:
                print(f"{stage:<22} 编码：{res['stage_codes'][stage]:>6} | {stage_desc[stage]}")
            print("-" * 80)
            print(f"【整体编码】")
            print(f"full_code = {res['full_code']}")
            print("-" * 80)
            print("编码含义：")
            print("  阶段编码位：0=全整块 | 1=全尾块 | 2=混合")
            print("  整体编码构成顺序：")
            print("    template_mode -> split_g -> PA_ori_block -> PA_cmp_block -> inter-core-split -> vec0 -> mm1_L1 -> mm1_L0 -> vec1 -> mm2_L1 -> mm2_L0 -> vec2")
        except Exception as e:
            print(f"运行报错：{e}")
            import traceback
            traceback.print_exc()
            raise
        return res
