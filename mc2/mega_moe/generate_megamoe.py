#!/usr/bin/env python3
"""
generate_megamoe — ATK CaseGenerator 自定义参数约束插件 (注册名: megamoe)

用于 aclnnMegaMoe 算子的用例生成。ATK 从 YAML 范围生成初版用例后，
本插件在 after_case_config 中统一修正所有参数依赖关系。

用法:
    atk case -f megamoe.yaml -p generate_megamoe.py
"""

import random

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig


# ============================================================
# 有效参数值集合 (来自 multicase_test, 过滤掉低于 tiling 下限的值)
# tiling 约束: hidden_size ∈ [1024, 10240], N 需为偶数且 ≥ 1024
# ============================================================
VALID_BS = [1, 4, 17, 32, 128, 512, 513, 2048, 4096]
VALID_HIDDEN_SIZE = [1024, 4096, 6144, 7168, 8192]  # 512 被 tiling 拒绝
VALID_N = [1024, 4096, 6144, 8192]  # 512 过小 (SwiGLU N/2=256)
VALID_TOPK = [1, 6, 7, 8, 16]
VALID_MOE_EXPERT_NUM = [16, 32, 256, 512, 1024]
VALID_EP_WORLD_SIZE = [16, 32]

# dtype 映射: 逻辑 weight_dtype → JSON 存储 dtype
DTYPE_MAP = {
    "int4": "int32",
    "int8": "int8",
    "bf16": "bf16",
}
# 反向映射
DTYPE_REVERSE_MAP = {v: k for k, v in DTYPE_MAP.items()}


# ============================================================
# 工具函数 (从 gen_case_2.py 迁移)
# ============================================================
def _weight_packed_dim(dim_size, weight_dtype):
    """计算权重的 packed 维度大小。"""
    if weight_dtype == "int4":
        return dim_size // 8
    return dim_size


def _infer_weight_dtype(json_dtype):
    """JSON 存储 dtype → 逻辑 weight_dtype。"""
    if json_dtype not in DTYPE_REVERSE_MAP:
        raise ValueError("Unknown weight JSON dtype: %s" % json_dtype)
    return DTYPE_REVERSE_MAP[json_dtype]


def _is_quantized(weight_dtype):
    """量化权重 (int4/int8) 需要 weight_scales。"""
    return weight_dtype in ("int4", "int8")


def _needs_bias(weight_dtype):
    """仅 int4 需要 bias。"""
    return weight_dtype == "int4"


def _snap(value, valid_set):
    """将 value 吸附到最近的有效值。"""
    return min(valid_set, key=lambda v: abs(v - value))


# ============================================================
# MegaMoeGenerator
# ============================================================

@GENERATOR_REGISTRY.register("megamoe")
class MegaMoeGenerator(CaseGenerator):
    """MegaMoe 算子参数约束生成器。"""

    def after_case_config(self, case_config):
        """修正 ATK 生成的用例，确保参数间一致性。"""
        try:
            return self._do_after_case_config(case_config)
        except Exception:
            # 无效参数组合，返回原 case (ATK 可能会跳过)
            return case_config

    def _do_after_case_config(self, case_config):
        inputs = case_config.inputs

        # ---- 1. 构建 lookup ----
        lookup_idx = {}
        for i, inp in enumerate(inputs):
            if isinstance(inp, list):
                if inp and hasattr(inp[0], 'name'):
                    lookup_idx[inp[0].name] = i
            elif hasattr(inp, 'name'):
                lookup_idx[inp.name] = i

        def _get(name):
            return inputs[lookup_idx[name]]

        # ---- 2. 提取关键参数 ----
        x_inp = _get("x")
        bs = x_inp.shape[0]
        hidden_size = x_inp.shape[1]
        bs = _snap(bs, VALID_BS)
        hidden_size = _snap(hidden_size, VALID_HIDDEN_SIZE)

        moe_expert = int(_get("moe_expert_num").range_values)
        ep_world = int(_get("ep_world_size").range_values)
        moe_expert = _snap(moe_expert, VALID_MOE_EXPERT_NUM)
        ep_world = _snap(ep_world, VALID_EP_WORLD_SIZE)

        # 确保整除
        if moe_expert % ep_world != 0:
            for candidate in sorted(VALID_MOE_EXPERT_NUM, key=lambda v: abs(v - moe_expert)):
                if candidate % ep_world == 0:
                    moe_expert = candidate
                    break
        pe = moe_expert // ep_world

        # 从 weight1 推断 weight_dtype 和 N
        w1_list = _get("weight1")
        if not isinstance(w1_list, list) or len(w1_list) == 0:
            return case_config
        w1_json_dtype = w1_list[0].dtype
        try:
            weight_dtype = _infer_weight_dtype(w1_json_dtype)
        except ValueError:
            return case_config

        # 推断 N
        w1_p1 = w1_list[0].shape[1]
        if weight_dtype == "int4":
            N_approx = w1_p1 * 8
        else:
            N_approx = w1_p1
        N = _snap(N_approx, VALID_N)
        if N % 2 != 0:
            even_n = [v for v in VALID_N if v % 2 == 0]
            N = _snap(N, even_n) if even_n else N

        # 推断 topk
        topk = _snap(_get("topk_ids").shape[1], VALID_TOPK)

        # ---- 3. 验证并修正约束 ----
        if topk > moe_expert:
            topk = max([v for v in VALID_TOPK if v <= moe_expert], default=1)

        if weight_dtype == "int4":
            if hidden_size % 16 != 0:
                h_valid = [v for v in VALID_HIDDEN_SIZE if v % 16 == 0]
                if h_valid:
                    hidden_size = _snap(hidden_size, h_valid)
            if N % 64 != 0:
                n_valid = [v for v in VALID_N if v % 64 == 0]
                if n_valid:
                    N = _snap(N, n_valid)

        # 基本有效性
        if bs <= 0 or hidden_size <= 0 or N <= 0 or topk <= 0 or pe <= 0:
            return case_config

        # ---- 4. 计算派生值 (权重 shape 使用逻辑维度, tiling 按此校验) ----
        quantized = _is_quantized(weight_dtype)
        need_bias = _needs_bias(weight_dtype)
        w_json_dtype = DTYPE_MAP[weight_dtype]
        w_range = [-8, 7] if quantized else [-1, 1]
        x_dtype = "bf16"

        # ---- 5. 修正所有输入 ----
        # weight1: list of [hidden_size, N] (逻辑维度)
        # weight2: list of [N//2, hidden_size] (逻辑维度, tiling 校验 w2.dim1==hidden_size)
        # weight_scales1: list of [N] / weight_scales2: list of [hidden_size]
        # bias1: list of [N] / bias2: list of [hidden_size]
        # context
        ctx = _get("context")
        ctx.shape = [27]
        ctx.dtype = "int32"
        ctx.range_values = [0, 1]

        # x
        x_inp.shape = [bs, hidden_size]
        x_inp.dtype = x_dtype
        x_inp.range_values = [-1, 1]

        # topk_ids
        tk_ids = _get("topk_ids")
        tk_ids.shape = [bs, topk]
        tk_ids.dtype = "int32"
        tk_ids.range_values = [0, ep_world * pe - 1]

        # topk_weights
        tk_w = _get("topk_weights")
        tk_w.shape = [bs, topk]
        tk_w.dtype = "fp32"
        tk_w.range_values = [-1, 1]

        # weight1: [hidden_size, N] (逻辑维度, tiling 按此校验)
        self._fix_tensor_list(w1_list, pe, [hidden_size, N], w_json_dtype, w_range, "weight1")

        # weight2: [N//2, hidden_size] (逻辑维度, tiling 校验 w2.dim1==hidden_size)
        w2_list = _get("weight2")
        self._fix_tensor_list(w2_list, pe, [N // 2, hidden_size], w_json_dtype, w_range, "weight2")

        # weight_scales1 (仅量化时)
        ws1_idx = lookup_idx.get("weight_scales1")
        if ws1_idx is not None:
            if quantized:
                self._fix_tensor_list(inputs[ws1_idx], pe, [N], "int64", [0, 1], "weight_scales1")
            else:
                inputs[ws1_idx] = []

        # weight_scales2 (仅量化时)
        ws2_idx = lookup_idx.get("weight_scales2")
        if ws2_idx is not None:
            if quantized:
                self._fix_tensor_list(inputs[ws2_idx], pe, [hidden_size], "int64", [0, 1], "weight_scales2")
            else:
                inputs[ws2_idx] = []

        # bias1 (仅 int4 时)
        b1_idx = lookup_idx.get("bias1")
        if b1_idx is not None:
            if need_bias:
                self._fix_tensor_list(inputs[b1_idx], pe, [N], "fp32", [0, 1], "bias1")
            else:
                inputs[b1_idx] = []

        # bias2 (仅 int4 时)
        b2_idx = lookup_idx.get("bias2")
        if b2_idx is not None:
            if need_bias:
                self._fix_tensor_list(inputs[b2_idx], pe, [hidden_size], "fp32", [0, 1], "bias2")
            else:
                inputs[b2_idx] = []

        # x_active_mask
        xam = _get("x_active_mask")
        xam.shape = [bs, topk]
        xam.dtype = "int8"
        xam.range_values = [-1, -1]

        # scales
        scales = _get("scales")
        scales.shape = [pe, hidden_size]
        scales.dtype = "fp32"
        scales.range_values = [0, 1]

        # ---- attrs (默认值参考模板, 约束参考 tiling 校验与文档) ----
        _get("moe_expert_num").range_values = moe_expert
        _get("ep_world_size").range_values = ep_world
        _get("ccl_buffer_size").range_values = -1
        # 文档约束: maxRecvTokenNum ∈ [0, BS × ep × min(topK, expertPerRank)]
        _get("max_recv_token_num").range_values = bs * ep_world * min(topk, pe)
        _get("dispatch_quant_mode").range_values = 2 if quantized else 0
        _get("dispatch_quant_out_dtype").range_values = 2 if quantized else 28
        _get("combine_quant_mode").range_values = 0
        _get("comm_alg").range_values = ""
        _get("num_max_tokens_per_rank").range_values = bs
        _get("activation").range_values = "swiglu"
        _get("activation_clamp").range_values = 3.4028235e+38
        _get("activation_out_dtype").range_values = 28
        _get("transpose_weight1").range_values = False
        _get("transpose_weight2").range_values = False
        _get("weight1_interleave").range_values = 0

        # y (output placeholder)
        y_out = _get("y")
        y_out.shape = [bs, hidden_size]
        y_out.dtype = x_dtype
        y_out.range_values = [0, 1]

        # expert_token_nums (output placeholder, template default [0, 1])
        etn = _get("expert_token_nums")
        etn.shape = [pe]
        etn.dtype = "int32"
        etn.range_values = [0, 1]

        # ---- 6. 移除空的可选输入 ----
        case_config.inputs = [inp for inp in case_config.inputs
                              if not (isinstance(inp, list) and len(inp) == 0)]

        # ---- 7. 统一设置 backward 字段 ----
        for inp in case_config.inputs:
            if isinstance(inp, list):
                for elem in inp:
                    if hasattr(elem, 'backward'):
                        elem.backward = False
            elif hasattr(inp, 'backward'):
                inp.backward = False

        # ---- 8. 设置顶层字段 ----
        case_config.aclnn_api_type = "pyaclnn_aclnnMegaMoe"
        case_config.dist_api_type = "mega_moe"
        case_config.api = "pytorch"
        case_config.backward = False
        case_config.outputs = None

        return case_config

    # ---- helper ----

    @staticmethod
    def _fix_tensor_list(t_list, target_len, shape, dtype, range_vals, name):
        """调整 TensorList 长度到 target_len，统一设置每个元素的属性。"""
        if not isinstance(t_list, list):
            return
        if len(t_list) > 0:
            template = t_list[0]
        else:
            template = InputCaseConfig(
                name=name, type="tensors", dtype=dtype,
                shape=list(shape), range_values=range_vals
            )
            t_list.append(template)

        # 裁剪或复制
        while len(t_list) > target_len:
            t_list.pop()
        while len(t_list) < target_len:
            t_list.append(template.model_copy())

        for entry in t_list:
            entry.name = name
            entry.type = "tensors"
            entry.dtype = dtype
            entry.shape = list(shape)
            entry.range_values = range_vals
