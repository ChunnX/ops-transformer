import torch
import torch_npu
import torchair
import torch.distributed as dist
import inspect
from enum import IntEnum
import cann_ops_nn.ops
from cann_ops_transformer.ops import mega_moe

_MEGA_MOE_SIGNATURE = inspect.signature(mega_moe)
_MEGA_MOE_PARAMETERS = _MEGA_MOE_SIGNATURE.parameters
_MEGA_MOE_ACCEPTS_VAR_KWARGS = any(
    parameter.kind == inspect.Parameter.VAR_KEYWORD
    for parameter in _MEGA_MOE_PARAMETERS.values()
)


def _mega_moe_accepts(name):
    return _MEGA_MOE_ACCEPTS_VAR_KWARGS or name in _MEGA_MOE_PARAMETERS


def _set_mega_moe_kwarg(kwargs, name, value, *, required=False):
    if _mega_moe_accepts(name):
        kwargs[name] = value
        return True
    if required:
        raise RuntimeError(
            f"当前 cann_ops_transformer.ops.mega_moe 不支持参数 {name!r}；"
            "请安装与仓库代码匹配的 torch extension。"
        )
    return False

FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
FLOAT8_E4M3FN_DTYPE = getattr(torch_npu, "float8_e4m3fn", getattr(torch, "float8_e4m3fn", None))
FLOAT8_E5M2_DTYPE = getattr(torch_npu, "float8_e5m2", getattr(torch, "float8_e5m2", None))
FLOAT4_E2M1FN_DTYPE = getattr(torch_npu, "float4_e2m1fn", getattr(torch, "float4_e2m1fn", None))
FLOAT4_E2M1FN_X2_DTYPE = getattr(torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None))

CUSTOM_GMM_DTYPES = tuple(
    dtype for dtype in (FLOAT4_E2M1FN_X2_DTYPE,)
    if dtype is not None
)
CUSTOM_GMM_DTYPE_BY_CODE = {
    296: FLOAT4_E2M1FN_X2_DTYPE,
}


def _resolve_mega_moe_activation_clamp(clamp_limit, activation_type):
    if (
        clamp_limit is None
        and activation_type in ("swiglu", "swiglustep")
    ):
        return 3.4028234663852886e38
    return clamp_limit


def normalize_custom_gmm_dtype(dtype_or_code):
    if dtype_or_code in CUSTOM_GMM_DTYPES:
        return dtype_or_code
    return CUSTOM_GMM_DTYPE_BY_CODE.get(dtype_or_code)


def use_antiquant_scale(act_quant_dtype, weight_quant_dtype):
    return weight_quant_dtype is not None and act_quant_dtype != weight_quant_dtype


def maybe_transpose_gmm_arg(tensor, weight_dtype):
    if tensor is None:
        return tensor
    return tensor.transpose(1, 2)


def make_static_weight(weight):
    if isinstance(weight, (tuple, list, torch.nn.ParameterList)):
        return torch.nn.ParameterList(
            [torch.nn.Parameter(item, requires_grad=False) for item in weight]
        )
    return torch.nn.Parameter(weight, requires_grad=False)


def parameter_list_to_tensor_list(weight):
    if isinstance(weight, torch.nn.ParameterList):
        return [weight[index] for index in range(len(weight))]
    return weight


def dtype_matches(dtype_or_code, *candidates):
    return dtype_or_code in candidates


def resolve_mx_quant_dtype(dtype_or_code):
    if dtype_matches(dtype_or_code, 24, FLOAT8_E4M3FN_DTYPE):
        return 24
    elif dtype_matches(dtype_or_code, 23, FLOAT8_E5M2_DTYPE):
        return 23
    elif dtype_matches(
        dtype_or_code,
        296,
        FLOAT4_E2M1FN_DTYPE,
        FLOAT4_E2M1FN_X2_DTYPE,
    ):
        return 296
    raise ValueError(f"Unsupported MX quant dtype: {dtype_or_code}")


def _pad_token_rows(tensor, token_num, max_token_num):
    if token_num == max_token_num:
        return tensor.contiguous()
    padding = tensor.new_zeros((max_token_num - token_num, *tensor.shape[1:]))
    return torch.cat((tensor, padding), dim=0).contiguous()


def _all_gather_variable_tensor(tensor, token_nums_per_rank, ep_rank_id, group_ep):
    token_nums = tuple(int(token_num) for token_num in token_nums_per_rank)
    max_token_num = max(token_nums)
    local_tensor = _pad_token_rows(
        tensor, token_nums[ep_rank_id], max_token_num
    )
    gathered = tensor.new_empty(
        (max_token_num * len(token_nums), *tensor.shape[1:])
    )
    dist.all_gather_into_tensor(gathered, local_tensor, group=group_ep)
    gathered = gathered.view(
        len(token_nums), max_token_num, *tensor.shape[1:]
    )
    valid_tensors = tuple(
        gathered[rank_id].narrow(0, 0, token_num)
        for rank_id, token_num in enumerate(token_nums)
    )
    return torch.cat(valid_tensors, dim=0)


def _reduce_scatter_variable_tensor(tensor, token_nums_per_rank, ep_rank_id, group_ep):
    token_nums = tuple(int(token_num) for token_num in token_nums_per_rank)
    max_token_num = max(token_nums)
    padded_chunks = []
    token_offset = 0
    for token_num in token_nums:
        chunk = tensor.narrow(0, token_offset, token_num)
        padded_chunks.append(_pad_token_rows(chunk, token_num, max_token_num))
        token_offset += token_num
    collective_input = torch.cat(tuple(padded_chunks), dim=0).contiguous()
    local_output = tensor.new_empty(
        (max_token_num, *tensor.shape[1:])
    )
    dist.reduce_scatter_tensor(local_output, collective_input, group=group_ep)
    return local_output.narrow(0, 0, token_nums[ep_rank_id])


class EventType(IntEnum):
    MOE_SHARED = 0
    SHARED_INPUT_READY = 1
    REDUCE_SCATTER = 2


class CascadeGraphModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loop = kwargs['loop']
        self.is_performance = kwargs['is_performance']
        self.aic_core_num = kwargs['aic_core_num']
        self.aiv_core_num = kwargs['aiv_core_num']
        self.is_graph = kwargs['is_graph']
        self.performance_loop_outside_graph = bool(
            kwargs.get('performance_loop_outside_graph', False)
        )
        self.graph_type = int(kwargs.get('graph_type', 0) or 0)
        clamp_limit = kwargs['clamp_limit']
        self.clamp_limit = (
            None if clamp_limit is None or clamp_limit == -1 else clamp_limit
        )
        self.activation_type = str(kwargs.get('activation_type', 'swiglu')).lower()
        if self.activation_type in ('swiglu/oai', 'swiglu_oai'):
            self.activation_type = 'swigluoai'
        if self.activation_type == 'situglu':
            self.activation_type = 'situ'
        self.swiglu_alpha = float(kwargs.get('swiglu_alpha', 1.702))
        self.swiglu_beta = float(kwargs.get('swiglu_beta', 1.0))
        self.situ_beta = float(kwargs.get('situ_beta', 1.0))
        situ_linear_beta = kwargs.get('situ_linear_beta', None)
        self.situ_linear_beta = None if situ_linear_beta is None else float(situ_linear_beta)
        if self.activation_type not in ('swiglu', 'swiglustep', 'swigluoai', 'situ'):
            raise ValueError(
                "activation_type only supports swiglu, swiglustep, "
                "swigluoai or situ, "
                f"got {self.activation_type}"
            )
        self.topk_weights_type = bool(kwargs.get('topk_weights_type', False))
        self.activation_direct_output = bool(kwargs.get('activation_direct_output', False))
        self.activation_debug_out = None
        if self.topk_weights_type and self.clamp_limit is None:
            self.clamp_limit = 3.4028234663852886e38
        self.local_moe_expert_num = int(kwargs.get('local_moe_expert_num', 0) or 0)
        self.shared_expert_num = max(0, int(kwargs.get('shared_expert_num', 0) or 0))
        self.expert_per_rank = int(kwargs.get(
            'expert_per_rank',
            self.local_moe_expert_num + self.shared_expert_num,
        ) or 0)
        if self.local_moe_expert_num <= 0 and self.expert_per_rank > self.shared_expert_num:
            self.local_moe_expert_num = self.expert_per_rank - self.shared_expert_num
        if self.expert_per_rank <= 0:
            self.expert_per_rank = self.local_moe_expert_num + self.shared_expert_num
        self.enable_shared_expert_multistream = bool(kwargs.get('enable_shared_expert_multistream', False))
        self.shared_expert_stream = None
        self.events = None
        self.shared_expert_graph_stream_tag = "shared_expert"
        self.shared_gmm1_weight_static = None
        self.moe_gmm1_weight_static = None
        self.shared_gmm2_weight_static = None
        self.moe_gmm2_weight_static = None
        # self.input_dtype = kwargs['input_dtype']

    def set_static_weights(self, gmm1_weight, gmm2_weight):
        shared_gmm1_weight, moe_gmm1_weight = self._split_prepared_weights(gmm1_weight)
        shared_gmm2_weight, moe_gmm2_weight = self._split_prepared_weights(gmm2_weight)
        if shared_gmm1_weight is not None:
            self.shared_gmm1_weight_static = torch.nn.Parameter(
                shared_gmm1_weight, requires_grad=False
            )
        if shared_gmm2_weight is not None:
            self.shared_gmm2_weight_static = torch.nn.Parameter(
                shared_gmm2_weight, requires_grad=False
            )
        self.moe_gmm1_weight_static = torch.nn.Parameter(
            moe_gmm1_weight, requires_grad=False
        )
        self.moe_gmm2_weight_static = torch.nn.Parameter(
            moe_gmm2_weight, requires_grad=False
        )

    def _get_gmm_weights(self, gmm1_weight, gmm2_weight):
        if self.moe_gmm1_weight_static is not None:
            gmm1_weight_for_gmm = self.shared_gmm1_weight_static, self.moe_gmm1_weight_static
            gmm2_weight_for_gmm = self.shared_gmm2_weight_static, self.moe_gmm2_weight_static
        else:
            gmm1_weight_for_gmm = gmm1_weight
            gmm2_weight_for_gmm = gmm2_weight
        if gmm1_weight_for_gmm is None or gmm2_weight_for_gmm is None:
            raise RuntimeError(
                "Cascade weights are missing. Pass gmm weights or call set_static_weights before compile."
            )
        return gmm1_weight_for_gmm, gmm2_weight_for_gmm

    def _swiglu_mx_quant(self, x, dst_type, topk_weights=None):
        if topk_weights is not None:
            topk_weights = topk_weights.reshape(-1).to(torch.float32)
        y, y_scale, activation_origin = torch.ops.cann_ops_nn.swiglu_group_quant(
            x,
            weight=topk_weights,
            group_index=None,
            scale=None,
            dst_type=dst_type,
            quant_mode=1,
            block_size=32,
            round_scale=True,
            clamp_limit=self.clamp_limit,
            output_origin=self.activation_direct_output,
        )
        if self.activation_direct_output:
            self.activation_debug_out = y
        return y, y_scale

    def _situ_mx_quant(self, x, dst_type, topk_weights=None):
        gate, up = x.chunk(2, dim=-1)
        gate = gate.to(torch.float32)
        up = up.to(torch.float32)
        y = self.situ_beta * torch.tanh(gate / self.situ_beta) * torch.sigmoid(gate)
        if self.situ_linear_beta is not None:
            up = self.situ_linear_beta * torch.tanh(up / self.situ_linear_beta)
        y = (y * up).to(torch.bfloat16)
        if topk_weights is not None:
            y = y * topk_weights.reshape(-1, 1).to(y.dtype)
        y, y_scale = torch_npu.npu_dynamic_mx_quant(
            y, axis=-1, dst_type=dst_type, block_size=32
        )
        if self.activation_direct_output:
            self.activation_debug_out = y
        return y, self._normalize_dispatch_scale(y_scale)

    def _swiglustep_mx_quant(self, x, dst_type, topk_weights=None):
        gate, up = x.chunk(2, dim=-1)
        gate = torch.nn.functional.silu(gate.to(torch.float32))
        up = up.to(torch.float32)
        if self.clamp_limit is not None:
            gate = torch.clamp(gate, max=self.clamp_limit)
            up = torch.clamp(
                up, min=-self.clamp_limit, max=self.clamp_limit
            )
        y = gate * up
        if topk_weights is not None:
            y = y * topk_weights.reshape(-1, 1).to(y.dtype)
        y, y_scale = torch_npu.npu_dynamic_mx_quant(
            y.to(torch.bfloat16), axis=-1, dst_type=dst_type, block_size=32
        )
        if self.activation_direct_output:
            self.activation_debug_out = y
        return y, self._normalize_dispatch_scale(y_scale)

    def _swigluoai_mx_quant(self, x, dst_type, topk_weights=None):
        y = torch_npu.npu_clipped_swiglu(
            x.to(torch.float32),
            group_index=None,
            dim=-1,
            alpha=self.swiglu_alpha,
            limit=self.clamp_limit,
            bias=self.swiglu_beta,
            interleaved=False,
        )
        if topk_weights is not None:
            y = y * topk_weights.reshape(-1, 1).to(y.dtype)
        y, y_scale = torch_npu.npu_dynamic_mx_quant(
            y.to(torch.bfloat16), axis=-1, dst_type=dst_type, block_size=32
        )
        if self.activation_direct_output:
            self.activation_debug_out = y
        return y, self._normalize_dispatch_scale(y_scale)

    def _activation_mx_quant(self, x, dst_type, topk_weights=None):
        if self.activation_type == 'situ':
            return self._situ_mx_quant(x, dst_type, topk_weights)
        if self.activation_type == 'swiglustep':
            return self._swiglustep_mx_quant(x, dst_type, topk_weights)
        if self.activation_type == 'swigluoai':
            return self._swigluoai_mx_quant(x, dst_type, topk_weights)
        return self._swiglu_mx_quant(x, dst_type, topk_weights)

    def _use_fused_gmm1_swiglu_quant(self):
        return (
            self.activation_type == 'swiglu'
            and not self.topk_weights_type
            and self.clamp_limit is None
        )

    def _normalize_dispatch_scale(self, dynamic_scales):
        if dynamic_scales is not None and dynamic_scales.dim() == 2 and dynamic_scales.shape[-1] % 2 == 0:
            return dynamic_scales.reshape(dynamic_scales.shape[0], dynamic_scales.shape[1] // 2, 2)
        return dynamic_scales

    def _route_prequantized_tensor(
            self, tensor, expert_ids, moe_expert_num, local_expert_start, local_expert_end
    ):
        routing_tensor = tensor.contiguous().view(torch.bfloat16)
        expanded_tensor, expanded_row_idx, expert_token_nums, _ = torch_npu.npu_moe_init_routing_v2(
            routing_tensor,
            expert_ids,
            scale=None,
            active_num=int(expert_ids.numel()),
            expert_num=moe_expert_num,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[local_expert_start, local_expert_end],
        )
        return expanded_tensor.view(tensor.dtype), expanded_row_idx, expert_token_nums

    def _route_prequantized_scale(
            self, scale, expert_ids, moe_expert_num, local_expert_start, local_expert_end
    ):
        scale_shape = scale.shape[1:]
        routing_scale = scale.reshape(scale.shape[0], -1).to(torch.bfloat16)
        expanded_scale, _, _, _ = torch_npu.npu_moe_init_routing_v2(
            routing_scale,
            expert_ids,
            scale=None,
            active_num=int(expert_ids.numel()),
            expert_num=moe_expert_num,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[local_expert_start, local_expert_end],
        )
        return expanded_scale.to(scale.dtype).view(-1, *scale_shape)

    def _allgather_init_routing_dispatch(
            self, x, local_quant_x, local_dynamic_scales, expert_ids, expert_scales,
            group_ep, ep_rank_id, ep_world_size, moe_expert_num,
            token_nums_per_rank=None,
    ):
        local_expert_start = ep_rank_id * self.local_moe_expert_num
        local_expert_end = local_expert_start + self.local_moe_expert_num

        if ep_world_size == 1:
            global_quant_x = local_quant_x
            global_dynamic_scales = local_dynamic_scales
            global_expert_ids = expert_ids
            global_expert_scales = expert_scales
        elif token_nums_per_rank is not None:
            global_quant_x = _all_gather_variable_tensor(
                local_quant_x, token_nums_per_rank, ep_rank_id, group_ep
            )
            global_dynamic_scales = _all_gather_variable_tensor(
                local_dynamic_scales, token_nums_per_rank, ep_rank_id, group_ep
            )
            global_expert_ids = _all_gather_variable_tensor(
                expert_ids, token_nums_per_rank, ep_rank_id, group_ep
            )
            global_expert_scales = _all_gather_variable_tensor(
                expert_scales, token_nums_per_rank, ep_rank_id, group_ep
            )
        else:
            global_quant_x = local_quant_x.new_empty(
                (local_quant_x.shape[0] * ep_world_size, *local_quant_x.shape[1:])
            )
            dist.all_gather_into_tensor(global_quant_x, local_quant_x.contiguous(), group=group_ep)

            global_dynamic_scales = local_dynamic_scales.new_empty(
                (local_dynamic_scales.shape[0] * ep_world_size, *local_dynamic_scales.shape[1:])
            )
            dist.all_gather_into_tensor(
                global_dynamic_scales, local_dynamic_scales.contiguous(), group=group_ep
            )

            global_expert_ids = expert_ids.new_empty((expert_ids.shape[0] * ep_world_size, *expert_ids.shape[1:]))
            dist.all_gather_into_tensor(global_expert_ids, expert_ids.contiguous(), group=group_ep)

            global_expert_scales = expert_scales.new_empty(
                (expert_scales.shape[0] * ep_world_size, *expert_scales.shape[1:])
            )
            dist.all_gather_into_tensor(global_expert_scales, expert_scales.contiguous(), group=group_ep)

        global_expert_ids = global_expert_ids.to(torch.int32)

        expand_x, expanded_row_idx, expert_token_nums = self._route_prequantized_tensor(
            global_quant_x,
            global_expert_ids,
            moe_expert_num,
            local_expert_start,
            local_expert_end,
        )
        dynamic_scales = self._route_prequantized_scale(
            global_dynamic_scales,
            global_expert_ids,
            moe_expert_num,
            local_expert_start,
            local_expert_end,
        )
        dispatch_ctx = {
            "expanded_row_idx": expanded_row_idx,
            "topk_ids": global_expert_ids,
            "topk_weights": global_expert_scales,
            "global_hidden_shape": (
                (sum(token_nums_per_rank) if token_nums_per_rank is not None
                 else x.shape[0] * ep_world_size),
                *x.shape[1:],
            ),
            "local_hidden_shape": x.shape,
            "ep_world_size": ep_world_size,
            "group_ep": group_ep,
        }
        return expand_x, dynamic_scales, expert_token_nums.to(torch.int64).to(x.device), dispatch_ctx

    def _finalize_routing_output(self, y2, dispatch_ctx, scales):
        return torch_npu.npu_moe_finalize_routing(
            y2,
            skip1=None,
            skip2=None,
            bias=None,
            scales=scales,
            expanded_src_to_dst_row=dispatch_ctx["expanded_row_idx"],
            export_for_source_row=dispatch_ctx["topk_ids"],
            drop_pad_mode=2,
        ).view(dispatch_ctx["global_hidden_shape"])

    def _get_comm_quant_dtype(self, comm_quant_mode):
        if comm_quant_mode == 3:
            return FLOAT8_E5M2_DTYPE
        if comm_quant_mode == 4:
            return FLOAT8_E4M3FN_DTYPE
        return None

    def _apply_combine_comm_quant(self, y2, comm_quant_mode, output_dtype):
        comm_quant_dtype = self._get_comm_quant_dtype(comm_quant_mode)
        if comm_quant_dtype is None:
            return y2
        y2_quant, y2_scale = torch_npu.npu_dynamic_mx_quant(
            y2,
            axis=-1,
            dst_type=comm_quant_dtype,
            block_size=32,
        )
        return torch_npu.npu_anti_mx_quant(
            y2_quant,
            y2_scale,
            axis=-1,
            dst_type=output_dtype,
            src_type=comm_quant_dtype,
        )

    def _get_expanded_topk_weights(self, y2, dispatch_ctx):
        flat_row_idx = dispatch_ctx["expanded_row_idx"].reshape(-1).to(torch.long)
        flat_scales = dispatch_ctx["topk_weights"].reshape(-1).to(y2.dtype)
        dummy_idx = torch.zeros_like(flat_row_idx) + y2.shape[0]
        valid_idx = (flat_row_idx >= 0) & (flat_row_idx < y2.shape[0])
        safe_row_idx = torch.where(valid_idx, flat_row_idx, dummy_idx)
        expanded_scales = y2.new_ones((y2.shape[0] + 1,))
        expanded_scales = expanded_scales.scatter(0, safe_row_idx, flat_scales)
        return expanded_scales[:y2.shape[0]].unsqueeze(1)

    def _prepare_y2_for_combine(self, y2, dispatch_ctx, comm_quant_mode, output_dtype):
        if self.topk_weights_type:
            y2 = self._apply_combine_comm_quant(y2, comm_quant_mode, output_dtype)
            return y2, torch.ones_like(dispatch_ctx["topk_weights"])
        y2 = self._apply_combine_comm_quant(y2, comm_quant_mode, output_dtype)
        return y2, dispatch_ctx["topk_weights"]

    def _quantize_dispatch_x(self, expand_x, y_dtype):
        quant_x, dynamic_scales = torch_npu.npu_dynamic_mx_quant(
            expand_x, axis=-1, dst_type=y_dtype, block_size=32
        )
        return quant_x, self._normalize_dispatch_scale(dynamic_scales)

    def _slice_moe_weights(self, weights):
        if weights is None:
            return weights
        return weights[self.shared_expert_num:self.expert_per_rank]

    def _slice_shared_weights(self, weights):
        if weights is None or self.shared_expert_num <= 0:
            return weights
        return weights[:self.shared_expert_num]

    def _split_prepared_weights(self, weights):
        if isinstance(weights, (tuple, list)) and len(weights) == 2:
            return weights[0], weights[1]
        return self._slice_shared_weights(weights), self._slice_moe_weights(weights)

    def _run_shared_ffn(
            self, x, quant_x, dynamic_scales, quant_mode, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
            gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
            wait_mode=None, reduce_scatter_marker=None, shared_token_nums=None
    ):
        if self.shared_expert_num <= 0:
            return None

        shared_x_npu = (
            quant_x
            if self.shared_expert_num == 1
            else quant_x.repeat(self.shared_expert_num, *([1] * (quant_x.dim() - 1)))
        )
        shared_dynamic_scales = (
            dynamic_scales
            if self.shared_expert_num == 1
            else dynamic_scales.repeat(
                self.shared_expert_num, *([1] * (dynamic_scales.dim() - 1))
            )
        )
        if shared_token_nums is None:
            shared_token_nums = torch.full(
                (self.shared_expert_num,),
                x.shape[0],
                dtype=torch.int64,
                device=x.device,
            )
        if wait_mode == "native":
            self._wait_shared_input_ready()

        if self._use_fused_gmm1_swiglu_quant():
            shared_y1_quant, shared_y1_scale = self._run_gmm1_swiglu_quant(
                shared_x_npu, shared_dynamic_scales, shared_token_nums,
                gmm1_weight, self._slice_shared_weights(gmm1_weight_scale),
                weight_dtype, x_dtype, swiglu_quant_dtype,
            )
            shared_y1 = None
        else:
            shared_y1 = self._run_gmm1(
                shared_x_npu, shared_dynamic_scales, shared_token_nums,
                gmm1_weight, output_dtype,
                self._slice_shared_weights(gmm1_weight_scale),
                weight_dtype, x_dtype,
            )

        if wait_mode == "native":
            self._wait_reduce_scatter()
        elif wait_mode == "torchair_graph" and reduce_scatter_marker is not None:
            if self._use_fused_gmm1_swiglu_quant():
                shared_y1_quant = torchair.scope.npu_wait_tensor(
                    shared_y1_quant, reduce_scatter_marker
                )
            else:
                shared_y1 = torchair.scope.npu_wait_tensor(shared_y1, reduce_scatter_marker)

        if self._use_fused_gmm1_swiglu_quant():
            shared_y = self._run_quantized_gmm2(
                shared_y1_quant, shared_y1_scale, shared_token_nums, gmm2_weight,
                self._slice_shared_weights(gmm2_weight_scale),
                quant_mode, output_dtype, weight_dtype, swiglu_quant_dtype,
            )
        else:
            shared_y = self._run_swiglu_gmm2(
                shared_y1, shared_token_nums, gmm2_weight,
                self._slice_shared_weights(gmm2_weight_scale),
                quant_mode, output_dtype, weight_dtype, swiglu_quant_dtype,
            )
        if self.shared_expert_num == 1:
            return shared_y.view(x.shape[0], x.shape[1])
        return shared_y.view(self.shared_expert_num, x.shape[0], x.shape[1]).sum(dim=0)

    def _shared_expert_multistream_enabled(self):
        return (
            self.enable_shared_expert_multistream
            and self.shared_expert_num > 0
            and not self.activation_direct_output
        )

    def _use_torchair_graph_multistream(self):
        return self._shared_expert_multistream_enabled() and self.is_graph and self.graph_type in (1, 2)

    def _ensure_native_shared_expert_multistream(self):
        if not self._shared_expert_multistream_enabled() or self._use_torchair_graph_multistream():
            return False
        if self.shared_expert_stream is None:
            self.shared_expert_stream = torch.npu.Stream()
            self.events = {
                EventType.MOE_SHARED: torch.npu.Event(),
                EventType.SHARED_INPUT_READY: torch.npu.Event(),
                EventType.REDUCE_SCATTER: torch.npu.Event(),
            }
        return True

    def _record_shared_input_ready(self):
        if self._ensure_native_shared_expert_multistream():
            self.events[EventType.SHARED_INPUT_READY].record()

    def _record_reduce_scatter(self):
        if self._ensure_native_shared_expert_multistream():
            self.events[EventType.REDUCE_SCATTER].record()

    def _wait_shared_input_ready(self):
        self.events[EventType.SHARED_INPUT_READY].wait()

    def _wait_reduce_scatter(self):
        self.events[EventType.REDUCE_SCATTER].wait()

    def _record_moe_shared(self):
        self.events[EventType.MOE_SHARED].record()

    def _wait_moe_shared(self):
        self.events[EventType.MOE_SHARED].wait()

    def _record_tensor_stream(self, tensor, stream):
        if isinstance(tensor, torch.Tensor):
            tensor.record_stream(stream)

    def _start_shared_ffn(
            self, x, quant_x, dynamic_scales, quant_mode, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
            gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
            reduce_scatter_marker=None, shared_token_nums=None
    ):
        if not self._shared_expert_multistream_enabled():
            return self._run_shared_ffn(
                x, quant_x, dynamic_scales, quant_mode, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
                gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                shared_token_nums=shared_token_nums,
            ), None

        if self._use_torchair_graph_multistream():
            with torchair.scope.npu_stream_switch(self.shared_expert_graph_stream_tag, 0):
                shared_output = self._run_shared_ffn(
                    x, quant_x, dynamic_scales, quant_mode, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
                    gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                    wait_mode="torchair_graph",
                    reduce_scatter_marker=reduce_scatter_marker,
                    shared_token_nums=shared_token_nums,
                )
            return shared_output, "torchair_graph"

        if not self._ensure_native_shared_expert_multistream():
            return self._run_shared_ffn(
                x, quant_x, dynamic_scales, quant_mode, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
                gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                shared_token_nums=shared_token_nums,
            ), None

        self._record_tensor_stream(quant_x, self.shared_expert_stream)
        self._record_tensor_stream(dynamic_scales, self.shared_expert_stream)
        self._record_tensor_stream(x, self.shared_expert_stream)
        with torch.npu.stream(self.shared_expert_stream):
            shared_output = self._run_shared_ffn(
                x, quant_x, dynamic_scales, quant_mode, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
                gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                wait_mode="native",
                shared_token_nums=shared_token_nums,
            )
            self._record_moe_shared()
        return shared_output, "native"

    def _merge_shared_output(self, output_combine_npu, shared_output, shared_done_event):
        if shared_output is None:
            return output_combine_npu
        if shared_done_event is None:
            return output_combine_npu + shared_output
        if shared_done_event == "torchair_graph":
            output_combine_npu = torchair.scope.npu_wait_tensor(output_combine_npu, shared_output)
            return output_combine_npu + shared_output

        self._record_tensor_stream(shared_output, torch.npu.current_stream())
        self._wait_moe_shared()
        return output_combine_npu + shared_output

    def _run_gmm1(
            self, expand_x_npu, dynamic_scales_npu, expert_token_nums_npu, gmm1_weight, output_dtype,
            gmm1_weight_scale, weight_dtype, x_dtype
    ):
        gmm1_x_dtype = normalize_custom_gmm_dtype(x_dtype)
        gmm_weight_dtype = normalize_custom_gmm_dtype(weight_dtype)
        gmm1_weight_for_gmm = maybe_transpose_gmm_arg(gmm1_weight, weight_dtype)
        gmm1_weight_scale_for_gmm = maybe_transpose_gmm_arg(
            gmm1_weight_scale, weight_dtype
        )
        gmm1_kwargs = {
            "x": [expand_x_npu],
            "weight": [gmm1_weight_for_gmm],
            "group_list": expert_token_nums_npu,
            "group_type": 0,
            "group_list_type": 1,
            "per_token_scale": [dynamic_scales_npu],
            "split_item": 2,
            "output_dtype": output_dtype,
            "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
        }
        if use_antiquant_scale(gmm1_x_dtype, gmm_weight_dtype):
            gmm1_kwargs["antiquant_scale"] = [gmm1_weight_scale_for_gmm]
        else:
            gmm1_kwargs["scale"] = [gmm1_weight_scale_for_gmm]
        if gmm_weight_dtype is not None:
            gmm1_kwargs["weight_dtype"] = gmm_weight_dtype
        if gmm1_x_dtype is not None:
            gmm1_kwargs["x_dtype"] = gmm1_x_dtype
        return torch_npu.npu_grouped_matmul(**gmm1_kwargs)[0]

    def _run_gmm1_swiglu_quant(
            self, expand_x_npu, dynamic_scales_npu, expert_token_nums_npu,
            gmm1_weight, gmm1_weight_scale, weight_dtype, x_dtype,
            swiglu_quant_dtype
    ):
        gmm1_x_dtype = normalize_custom_gmm_dtype(x_dtype)
        gmm_weight_dtype = normalize_custom_gmm_dtype(weight_dtype)
        fused_kwargs = {
            "x": expand_x_npu,
            "weight": [maybe_transpose_gmm_arg(gmm1_weight, weight_dtype)],
            "weight_scale": [
                maybe_transpose_gmm_arg(gmm1_weight_scale, weight_dtype)
            ],
            "x_scale": dynamic_scales_npu,
            "group_list": expert_token_nums_npu,
            "dequant_mode": 2,
            "dequant_dtype": torch.float32,
            "quant_mode": 2,
            "quant_dtype": resolve_mx_quant_dtype(swiglu_quant_dtype),
            "group_list_type": 1,
            "weight_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
            "x_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
        }
        if gmm_weight_dtype is not None:
            fused_kwargs["weight_dtype"] = gmm_weight_dtype
        if gmm1_x_dtype is not None:
            fused_kwargs["x_dtype"] = gmm1_x_dtype
        return torch_npu.npu_grouped_matmul_swiglu_quant_v2(**fused_kwargs)

    def _run_quantized_gmm2(
            self, y_swiglu_quant, y_swiglu_scale, expert_token_nums_npu,
            gmm2_weight, gmm2_weight_scale, quant_mode, output_dtype,
            weight_dtype, swiglu_quant_dtype
    ):
        gmm2_x_dtype = normalize_custom_gmm_dtype(swiglu_quant_dtype)
        gmm_weight_dtype = normalize_custom_gmm_dtype(weight_dtype)
        gmm2_weight_for_gmm = maybe_transpose_gmm_arg(gmm2_weight, weight_dtype)
        gmm2_weight_scale_for_gmm = maybe_transpose_gmm_arg(
            gmm2_weight_scale, weight_dtype
        )
        gmm2_kwargs = {
            "x": [y_swiglu_quant],
            "weight": [gmm2_weight_for_gmm],
            "group_list": expert_token_nums_npu,
            "group_type": 0,
            "group_list_type": 1,
            "per_token_scale": [y_swiglu_scale],
            "split_item": 2,
            "output_dtype": output_dtype,
            "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
        }
        if use_antiquant_scale(gmm2_x_dtype, gmm_weight_dtype):
            gmm2_kwargs["antiquant_scale"] = [gmm2_weight_scale_for_gmm]
        else:
            gmm2_kwargs["scale"] = (
                [gmm2_weight_scale_for_gmm]
                if quant_mode > 0
                else None
            )
            gmm2_kwargs["scale_dtype"] = FLOAT8_E8M0FNU_DTYPE
        if gmm_weight_dtype is not None:
            gmm2_kwargs["weight_dtype"] = gmm_weight_dtype
        if gmm2_x_dtype is not None:
            gmm2_kwargs["x_dtype"] = gmm2_x_dtype
        return torch_npu.npu_grouped_matmul(**gmm2_kwargs)[0]

    def _run_swiglu_gmm2(
            self, y1, expert_token_nums_npu, gmm2_weight, gmm2_weight_scale, quant_mode, output_dtype,
            weight_dtype, swiglu_quant_dtype, topk_weights=None
    ):
        y_swiglu_quant, y_swiglu_scale = self._activation_mx_quant(
            y1, swiglu_quant_dtype, topk_weights
        )
        return self._run_quantized_gmm2(
            y_swiglu_quant, y_swiglu_scale, expert_token_nums_npu,
            gmm2_weight, gmm2_weight_scale, quant_mode, output_dtype,
            weight_dtype, swiglu_quant_dtype,
        )

    def _run_ffn(
            self, expand_x_npu, dynamic_scales_npu, expert_token_nums_npu, quant_mode, gmm1_weight, output_dtype,
            gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
            dispatch_ctx=None
    ):
        y2 = None
        if quant_mode == 4:
            if self._use_fused_gmm1_swiglu_quant():
                y_swiglu_quant, y_swiglu_scale = self._run_gmm1_swiglu_quant(
                    expand_x_npu, dynamic_scales_npu, expert_token_nums_npu,
                    gmm1_weight, gmm1_weight_scale, weight_dtype, x_dtype,
                    swiglu_quant_dtype,
                )
                y2 = self._run_quantized_gmm2(
                    y_swiglu_quant, y_swiglu_scale, expert_token_nums_npu,
                    gmm2_weight, gmm2_weight_scale, quant_mode, output_dtype,
                    weight_dtype, swiglu_quant_dtype,
                )
            else:
                y1 = self._run_gmm1(
                    expand_x_npu, dynamic_scales_npu, expert_token_nums_npu,
                    gmm1_weight, output_dtype, gmm1_weight_scale, weight_dtype, x_dtype,
                )
                topk_weights = None
                if self.topk_weights_type and dispatch_ctx is not None:
                    topk_weights = self._get_expanded_topk_weights(y1, dispatch_ctx)
                y2 = self._run_swiglu_gmm2(
                    y1, expert_token_nums_npu, gmm2_weight, gmm2_weight_scale,
                    quant_mode, output_dtype, weight_dtype, swiglu_quant_dtype, topk_weights,
                )
        else:
            raise NotImplementedError("alltoallv dispatch path currently supports quant_mode == 4 only")

        return y2

    def forward(
            self, x, expert_ids, expert_scales, x_active_mask, group_ep, group_tp, ep_rank_id, ep_world_size,
            moe_expert_num, quant_mode, global_bs, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
            gmm2_weight_scale, y_dtype, comm_quant_mode, rank, weight_dtype, swiglu_quant_dtype, x_dtype,
            shared_token_nums=None, token_nums_per_rank=None,
    ):
        gmm1_weight, gmm2_weight = self._get_gmm_weights(gmm1_weight, gmm2_weight)
        shared_gmm1_weight, moe_gmm1_weight = self._split_prepared_weights(gmm1_weight)
        shared_gmm2_weight, moe_gmm2_weight = self._split_prepared_weights(gmm2_weight)
        local_loop = (
            self.loop
            if self.is_performance
            and self.is_graph
            and not self.performance_loop_outside_graph
            else 1
        )
        original_x = x
        for loop_idx in range(local_loop):
            local_quant_x, local_dynamic_scales = self._quantize_dispatch_x(x, y_dtype)
            self._record_shared_input_ready()
            shared_output = None
            shared_done_event = None
            shared_started = False
            if self._shared_expert_multistream_enabled() and not self._use_torchair_graph_multistream():
                shared_output, shared_done_event = self._start_shared_ffn(
                    x, local_quant_x, local_dynamic_scales, quant_mode,
                    shared_gmm1_weight, output_dtype, gmm1_weight_scale, shared_gmm2_weight,
                    gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                    shared_token_nums=shared_token_nums,
                )
                shared_started = True

            moe_gmm1_weight_scale = self._slice_moe_weights(gmm1_weight_scale)
            moe_gmm2_weight_scale = self._slice_moe_weights(gmm2_weight_scale)

            expand_x_npu, dynamic_scales_npu, expert_token_nums_npu, dispatch_ctx = self._allgather_init_routing_dispatch(
                x, local_quant_x, local_dynamic_scales, expert_ids, expert_scales,
                group_ep, ep_rank_id, ep_world_size, moe_expert_num,
                token_nums_per_rank,
            )
            y2 = self._run_ffn(
                expand_x_npu, dynamic_scales_npu, expert_token_nums_npu, quant_mode, moe_gmm1_weight, output_dtype,
                moe_gmm1_weight_scale, moe_gmm2_weight, moe_gmm2_weight_scale,
                weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype, dispatch_ctx
            )
            moe_activation_debug = self.activation_debug_out

            y2, combine_scales = self._prepare_y2_for_combine(
                y2, dispatch_ctx, comm_quant_mode, output_dtype
            )
            self._record_reduce_scatter()
            output_combine_npu = self._finalize_routing_output(y2, dispatch_ctx, combine_scales)
            if not shared_started and self.shared_expert_num > 0:
                shared_output, shared_done_event = self._start_shared_ffn(
                    x, local_quant_x, local_dynamic_scales, quant_mode,
                    shared_gmm1_weight, output_dtype, gmm1_weight_scale, shared_gmm2_weight,
                    gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                    reduce_scatter_marker=output_combine_npu,
                    shared_token_nums=shared_token_nums,
                )
            if ep_world_size == 1:
                output_combine_npu = output_combine_npu.view(dispatch_ctx["local_hidden_shape"])
            elif token_nums_per_rank is not None:
                output_combine_npu = _reduce_scatter_variable_tensor(
                    output_combine_npu,
                    token_nums_per_rank,
                    ep_rank_id,
                    group_ep,
                )
            else:
                local_output_combine_npu = output_combine_npu.new_empty(dispatch_ctx["local_hidden_shape"])
                dist.reduce_scatter_tensor(local_output_combine_npu, output_combine_npu.contiguous(), group=group_ep)
                output_combine_npu = local_output_combine_npu
            if self.shared_expert_num > 0:
                output_combine_npu = self._merge_shared_output(output_combine_npu, shared_output, shared_done_event)
            output_dispatch_npu = (
                expand_x_npu,
                dynamic_scales_npu,
                dispatch_ctx["expanded_row_idx"],
                expert_token_nums_npu,
                None,
                None,
                None,
            )
            if (
                local_loop > 1
                and loop_idx + 1 < local_loop
                and output_combine_npu.shape == original_x.shape
                and output_combine_npu.dtype == original_x.dtype
            ):
                x = output_combine_npu * 0 + original_x

        if self.activation_direct_output:
            return output_combine_npu, output_dispatch_npu, moe_activation_debug
        return output_combine_npu, output_dispatch_npu

class MegaMoeGraphModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loop = kwargs['loop']
        self.is_performance = kwargs['is_performance']
        self.aic_core_num = kwargs['aic_core_num']
        self.aiv_core_num = kwargs['aiv_core_num']
        self.is_graph = kwargs['is_graph']
        self.performance_loop_outside_graph = bool(
            kwargs.get('performance_loop_outside_graph', False)
        )
        clamp_limit = kwargs['clamp_limit']
        self.clamp_limit = (
            None if clamp_limit is None or clamp_limit == -1 else clamp_limit
        )
        self.activation_type = str(kwargs.get('activation_type', 'swiglu')).lower()
        if self.activation_type in ('swiglu/oai', 'swiglu_oai'):
            self.activation_type = 'swigluoai'
        if self.activation_type == 'situglu':
            self.activation_type = 'situ'
        self.is_activation_params = bool(kwargs.get('is_activation_params', False))
        self.weight_is_tensor_list = bool(kwargs.get('weight_is_tensor_list', False))
        self.activation_direct_output = bool(kwargs.get('activation_direct_output', False))
        self.swiglu_alpha = float(kwargs.get('swiglu_alpha', 1.702))
        self.swiglu_beta = float(kwargs.get('swiglu_beta', 1.0))
        self.situ_beta = float(kwargs.get('situ_beta', 1.0))
        situ_linear_beta = kwargs.get('situ_linear_beta', None)
        self.situ_linear_beta = None if situ_linear_beta is None else float(situ_linear_beta)
        self.local_moe_expert_num = int(kwargs.get('local_moe_expert_num', 0) or 0)
        self.shared_expert_num = max(0, int(kwargs.get('shared_expert_num', 0) or 0))
        self.expert_per_rank = int(kwargs.get(
            'expert_per_rank',
            self.local_moe_expert_num + self.shared_expert_num,
        ) or 0)
        self.l1_weights_static = None
        self.l2_weights_static = None
        self.shared_l1_weights_static = None
        self.shared_l2_weights_static = None

    def set_static_weights(self, l1_weights, l2_weights, shared_l1_weights=None, shared_l2_weights=None):
        self.l1_weights_static = make_static_weight(l1_weights)
        self.l2_weights_static = make_static_weight(l2_weights)
        if shared_l1_weights is not None:
            self.shared_l1_weights_static = make_static_weight(shared_l1_weights)
        if shared_l2_weights is not None:
            self.shared_l2_weights_static = make_static_weight(shared_l2_weights)

    def _get_moe_weights(self, l1_weights, l2_weights, shared_l1_weights, shared_l2_weights):
        l1_weights_for_moe = self.l1_weights_static if self.l1_weights_static is not None else l1_weights
        l2_weights_for_moe = self.l2_weights_static if self.l2_weights_static is not None else l2_weights
        if l1_weights_for_moe is None or l2_weights_for_moe is None:
            raise RuntimeError(
                "MegaMoe weights are missing. Pass l1_weights/l2_weights or call set_static_weights before compile."
            )
        shared_l1_weights = (
            self.shared_l1_weights_static
            if self.shared_l1_weights_static is not None
            else shared_l1_weights
        )
        shared_l2_weights = (
            self.shared_l2_weights_static
            if self.shared_l2_weights_static is not None
            else shared_l2_weights
        )
        if self.shared_expert_num > 0 and (shared_l1_weights is None or shared_l2_weights is None):
            raise RuntimeError("MegaMoe shared expert weights are missing.")
        return l1_weights_for_moe, l2_weights_for_moe, shared_l1_weights, shared_l2_weights

    def _keep_moe_expert_token_nums(self, expert_token_nums):
        if self.local_moe_expert_num <= 0:
            return expert_token_nums[:0]
        return expert_token_nums[-self.local_moe_expert_num:]

    def forward(
            self, x, topk_ids, topk_weights, sym_buffer,
            rank, l1_weights_sf, l2_weights_sf, x_active_mask, weight1_type, weight2_type,
            l1_weights=None, l2_weights=None,
            shared_l1_weights=None, shared_l2_weights=None,
            shared_l1_weights_sf=None, shared_l2_weights_sf=None
    ):
        output_combine_npu = None
        combine_x = None
        l1_weights_for_moe, l2_weights_for_moe, shared_l1_weights, shared_l2_weights = self._get_moe_weights(
            l1_weights, l2_weights, shared_l1_weights, shared_l2_weights
        )
        l1_weights_for_moe = parameter_list_to_tensor_list(l1_weights_for_moe)
        l2_weights_for_moe = parameter_list_to_tensor_list(l2_weights_for_moe)
        shared_l1_weights = parameter_list_to_tensor_list(shared_l1_weights)
        shared_l2_weights = parameter_list_to_tensor_list(shared_l2_weights)
        local_loop = (
            self.loop
            if self.is_performance
            and self.is_graph
            and not self.performance_loop_outside_graph
            else 1
        )
        original_x = x
        for loop_idx in range(local_loop):
            mega_moe_kwargs = {
                "x": x,
                "topk_ids": topk_ids,
                "topk_weights": topk_weights,
                "l1_weights": l1_weights_for_moe,
                "l2_weights": l2_weights_for_moe,
                "sym_buffer": sym_buffer,
                "l1_weights_sf": l1_weights_sf,
                "l2_weights_sf": l2_weights_sf,
                "x_active_mask": None,
                "weight1_type": weight1_type,
                "weight2_type": weight2_type,
            }
            if self.activation_direct_output:
                _set_mega_moe_kwarg(
                    mega_moe_kwargs, "activation_direct_output", True, required=True
                )
            activation_name = "situglu" if self.activation_type == "situ" else self.activation_type
            _set_mega_moe_kwarg(mega_moe_kwargs, "activation", activation_name, required=True)
            activation_clamp = _resolve_mega_moe_activation_clamp(
                self.clamp_limit,
                self.activation_type,
            )
            if activation_clamp is not None:
                _set_mega_moe_kwarg(
                    mega_moe_kwargs, "activation_clamp", activation_clamp, required=True
                )

            if _mega_moe_accepts("activation_params"):
                if self.activation_type == "situ":
                    activation_params = {"beta": self.situ_beta}
                    if self.situ_linear_beta is not None:
                        activation_params["linear_beta"] = self.situ_linear_beta
                    mega_moe_kwargs["activation_params"] = activation_params
                elif self.activation_type == "swigluoai":
                    mega_moe_kwargs["activation_params"] = {
                        "alpha": self.swiglu_alpha,
                        "beta": self.swiglu_beta,
                    }
            else:
                if self.activation_type == "situ":
                    _set_mega_moe_kwarg(
                        mega_moe_kwargs, "activation_alpha", self.situ_linear_beta, required=True
                    )
                    _set_mega_moe_kwarg(
                        mega_moe_kwargs, "activation_beta", self.situ_beta, required=True
                    )
                elif self.activation_type == "swigluoai":
                    _set_mega_moe_kwarg(
                        mega_moe_kwargs, "activation_alpha", self.swiglu_alpha, required=True
                    )
                    _set_mega_moe_kwarg(
                        mega_moe_kwargs, "activation_beta", self.swiglu_beta, required=True
                    )
            if shared_l1_weights is not None:
                _set_mega_moe_kwarg(
                    mega_moe_kwargs, "shared_l1_weights", shared_l1_weights, required=True
                )
                _set_mega_moe_kwarg(
                    mega_moe_kwargs, "shared_l2_weights", shared_l2_weights, required=True
                )
                _set_mega_moe_kwarg(
                    mega_moe_kwargs, "shared_l1_weights_sf", shared_l1_weights_sf, required=True
                )
                _set_mega_moe_kwarg(
                    mega_moe_kwargs, "shared_l2_weights_sf", shared_l2_weights_sf, required=True
                )
            outputs = mega_moe(**mega_moe_kwargs)
            if self.activation_direct_output:
                output_combine_npu, expert_token_nums, activation_debug_out = outputs
            else:
                output_combine_npu, expert_token_nums = outputs
            expert_token_nums = self._keep_moe_expert_token_nums(expert_token_nums)
            if (
                local_loop > 1
                and loop_idx + 1 < local_loop
                and output_combine_npu.shape == original_x.shape
                and output_combine_npu.dtype == original_x.dtype
            ):
                x = output_combine_npu * 0 + original_x

        if self.activation_direct_output:
            return output_combine_npu, expert_token_nums, activation_debug_out
        return output_combine_npu, expert_token_nums
