import torch
import torch_npu
import torchair
import torch.distributed as dist
from enum import IntEnum
import cann_ops_nn.ops

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


class EventType(IntEnum):
    SHARED_INPUT_READY = 0
    MOE_SHARED = 1


def normalize_custom_gmm_dtype(dtype_or_code):
    if dtype_or_code in CUSTOM_GMM_DTYPES:
        return dtype_or_code
    return CUSTOM_GMM_DTYPE_BY_CODE.get(dtype_or_code)


def use_antiquant_scale(act_quant_dtype, weight_quant_dtype):
    return weight_quant_dtype is not None and act_quant_dtype != weight_quant_dtype


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
        self.oai_quant_from_fp32 = bool(kwargs.get('oai_quant_from_fp32', False))
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
        if self.topk_weights_type and self.clamp_limit is None:
            self.clamp_limit = 3.4028234663852886e38
        self.local_moe_expert_num = int(kwargs.get('local_moe_expert_num', 0) or 0)
        self.shared_expert_num = max(0, int(kwargs.get('shared_expert_num', 0) or 0))
        self.expert_per_rank = int(kwargs.get(
            'expert_per_rank',
            self.local_moe_expert_num + self.shared_expert_num,
        ) or 0)
        self.enable_shared_expert_multistream = bool(
            kwargs.get('enable_shared_expert_multistream', False)
        )
        self.shared_expert_stream = None
        self.events = None
        self.shared_expert_graph_stream_tag = "shared_expert"
        self.alltoallv_dispatch_threshold = kwargs.get('alltoallv_dispatch_threshold', 512)
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
            gmm1_weight = (
                self.shared_gmm1_weight_static,
                self.moe_gmm1_weight_static,
            )
            gmm2_weight = (
                self.shared_gmm2_weight_static,
                self.moe_gmm2_weight_static,
            )
        if gmm1_weight is None or gmm2_weight is None:
            raise RuntimeError(
                "Cascade weights are missing. Pass gmm weights or call set_static_weights before compile."
            )
        return gmm1_weight, gmm2_weight

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

    def _swiglu_mx_quant(self, x, dst_type):
        y, y_scale, _ = torch.ops.cann_ops_nn.swiglu_group_quant(
            x,
            weight=None,
            group_index=None,
            scale=None,
            dst_type=dst_type,
            quant_mode=1,
            block_size=32,
            round_scale=True,
            clamp_limit=self.clamp_limit,
            output_origin=False,
        )
        return y, y_scale

    def _situ_mx_quant(self, x, dst_type):
        gate, up = x.chunk(2, dim=-1)
        gate = gate.to(torch.float32)
        up = up.to(torch.float32)
        y = self.situ_beta * torch.tanh(gate / self.situ_beta) * torch.sigmoid(gate)
        if self.situ_linear_beta is not None:
            up = self.situ_linear_beta * torch.tanh(up / self.situ_linear_beta)
        y = (y * up).to(torch.bfloat16)
        y, y_scale = torch_npu.npu_dynamic_mx_quant(
            y, axis=-1, dst_type=dst_type, block_size=32
        )
        if y_scale.dim() == 2 and y_scale.shape[-1] % 2 == 0:
            y_scale = y_scale.reshape(y_scale.shape[0], y_scale.shape[1] // 2, 2)
        return y, y_scale

    def _swiglustep_mx_quant(self, x, dst_type):
        gate, up = x.chunk(2, dim=-1)
        gate = torch.nn.functional.silu(gate.to(torch.float32))
        up = up.to(torch.float32)
        if self.clamp_limit is not None:
            gate = torch.clamp(gate, max=self.clamp_limit)
            up = torch.clamp(
                up, min=-self.clamp_limit, max=self.clamp_limit
            )
        y, y_scale = torch_npu.npu_dynamic_mx_quant(
            (gate * up).to(torch.bfloat16),
            axis=-1,
            dst_type=dst_type,
            block_size=32,
        )
        if y_scale.dim() == 2 and y_scale.shape[-1] % 2 == 0:
            y_scale = y_scale.reshape(y_scale.shape[0], y_scale.shape[1] // 2, 2)
        return y, y_scale

    def _swigluoai_mx_quant(self, x, dst_type):
        y = torch_npu.npu_clipped_swiglu(
            x.to(torch.float32),
            group_index=None,
            dim=-1,
            alpha=self.swiglu_alpha,
            limit=self.clamp_limit,
            bias=self.swiglu_beta,
            interleaved=False,
        )
        quant_input = y if self.oai_quant_from_fp32 else y.to(torch.bfloat16)
        y, y_scale = torch_npu.npu_dynamic_mx_quant(
            quant_input,
            axis=-1,
            dst_type=dst_type,
            block_size=32,
        )
        if y_scale.dim() == 2 and y_scale.shape[-1] % 2 == 0:
            y_scale = y_scale.reshape(y_scale.shape[0], y_scale.shape[1] // 2, 2)
        return y, y_scale

    def _activation_mx_quant(self, x, dst_type):
        if self.activation_type == 'situ':
            return self._situ_mx_quant(x, dst_type)
        if self.activation_type == 'swiglustep':
            return self._swiglustep_mx_quant(x, dst_type)
        if self.activation_type == 'swigluoai':
            return self._swigluoai_mx_quant(x, dst_type)
        return self._swiglu_mx_quant(x, dst_type)

    def _use_fused_gmm1_swiglu_quant(self):
        return (
            self.activation_type == 'swiglu'
            and not self.topk_weights_type
            and self.clamp_limit is None
        )

    def _need_alltoallv_dispatch(self, x):
        return self.alltoallv_dispatch_threshold is not None and x.shape[0] > self.alltoallv_dispatch_threshold

    def _all_to_all_single(self, input_tensor, output_splits, input_splits):
        output_splits = [int(count) for count in output_splits]
        input_splits = [int(count) for count in input_splits]
        output_shape = list(input_tensor.shape)
        output_shape[0] = sum(output_splits)
        output_tensor = input_tensor.new_empty(output_shape)
        dist.all_to_all_single(
            output_tensor,
            input_tensor.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        return output_tensor

    def _alltoallv_preprocess(self, expert_ids, ep_rank_id, ep_world_size, moe_expert_num):
        if moe_expert_num % ep_world_size != 0:
            raise ValueError(f"moe_expert_num({moe_expert_num}) must be divisible by ep_world_size({ep_world_size})")

        local_expert_num = moe_expert_num // ep_world_size
        num_out_tokens = expert_ids.numel()
        num_local_tokens_per_expert = torch.histc(
            expert_ids, bins=moe_expert_num, min=0, max=moe_expert_num
        ).to(torch.int64)

        input_splits = (
            num_local_tokens_per_expert.reshape(ep_world_size, local_expert_num)
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )

        all_expert_counts = [torch.empty_like(num_local_tokens_per_expert) for _ in range(ep_world_size)]
        dist.all_gather(all_expert_counts, num_local_tokens_per_expert)
        num_global_tokens_per_expert = torch.stack(all_expert_counts, dim=0)

        local_expert_start = ep_rank_id * local_expert_num
        local_expert_end = local_expert_start + local_expert_num
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, local_expert_start:local_expert_end]
        output_splits = num_global_tokens_per_local_expert.sum(dim=1).detach().cpu().tolist()
        expert_token_nums = num_global_tokens_per_local_expert.sum(dim=0).to(torch.int64)

        local_expert_indices = None
        if local_expert_num > 1:
            expert_ids_per_ep_rank = torch.tensor(
                [i % local_expert_num for i in range(moe_expert_num)],
                dtype=torch.int32,
                device=expert_ids.device,
            )
            local_expert_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert.reshape(-1).to(torch.int64),
            )
        else:
            torch.npu.synchronize()

        return input_splits, output_splits, expert_token_nums, local_expert_indices, num_out_tokens, local_expert_num

    def _alltoallv_dispatch(self, x, expert_ids, expert_scales, ep_rank_id, ep_world_size, moe_expert_num):
        input_splits, output_splits, expert_token_nums, local_expert_indices, num_out_tokens, local_expert_num = (
            self._alltoallv_preprocess(expert_ids, ep_rank_id, ep_world_size, moe_expert_num)
        )

        hidden_shape = x.shape
        permuted_x, reversed_local_mapping = torch_npu.npu_moe_token_permute(
            tokens=x,
            indices=expert_ids,
            num_out_tokens=num_out_tokens,
        )
        global_x = self._all_to_all_single(permuted_x, output_splits, input_splits)

        reversed_global_mapping = None
        if local_expert_num > 1:
            global_x, reversed_global_mapping = torch_npu.npu_moe_token_permute(
                tokens=global_x,
                indices=local_expert_indices,
            )

        dispatch_ctx = {
            "input_splits": input_splits,
            "output_splits": output_splits,
            "topk_weights": expert_scales,
            "reversed_local_mapping": reversed_local_mapping,
            "reversed_global_mapping": reversed_global_mapping,
            "hidden_shape": hidden_shape,
            "local_expert_num": local_expert_num,
        }
        return global_x, expert_token_nums.to(x.device), dispatch_ctx

    def _quantize_dispatch_x(self, expand_x, y_dtype):
        quant_x, dynamic_scales = torch_npu.npu_dynamic_mx_quant(
            expand_x, axis=-1, dst_type=y_dtype, block_size=32
        )
        if dynamic_scales.dim() == 2 and dynamic_scales.shape[-1] % 2 == 0:
            dynamic_scales = dynamic_scales.reshape(dynamic_scales.shape[0], dynamic_scales.shape[1] // 2, 2)
        return quant_x, dynamic_scales

    def _run_gmm1_swiglu_quant(
            self, expand_x_npu, dynamic_scales_npu, expert_token_nums_npu,
            gmm1_weight, gmm1_weight_scale, weight_dtype, x_dtype,
            swiglu_quant_dtype
    ):
        gmm1_x_dtype = normalize_custom_gmm_dtype(x_dtype)
        gmm_weight_dtype = normalize_custom_gmm_dtype(weight_dtype)
        fused_kwargs = {
            "x": expand_x_npu,
            "weight": [gmm1_weight.transpose(1, 2)],
            "weight_scale": [gmm1_weight_scale.transpose(1, 2)],
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

    def _run_ffn(
            self, expand_x_npu, dynamic_scales_npu, expert_token_nums_npu, quant_mode, gmm1_weight, output_dtype,
            gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype
    ):
        y2 = None
        if quant_mode == 4:
            gmm2_x_dtype = normalize_custom_gmm_dtype(swiglu_quant_dtype)
            gmm_weight_dtype = normalize_custom_gmm_dtype(weight_dtype)

            if self._use_fused_gmm1_swiglu_quant():
                y_swiglu_quant, y_swiglu_scale = self._run_gmm1_swiglu_quant(
                    expand_x_npu, dynamic_scales_npu, expert_token_nums_npu,
                    gmm1_weight, gmm1_weight_scale, weight_dtype, x_dtype,
                    swiglu_quant_dtype,
                )
            else:
                gmm1_x_dtype = normalize_custom_gmm_dtype(x_dtype)
                gmm1_kwargs = {
                    "x": [expand_x_npu],
                    "weight": [gmm1_weight.transpose(1, 2)],
                    "group_list": expert_token_nums_npu,
                    "group_type": 0,
                    "group_list_type": 1,
                    "per_token_scale": [dynamic_scales_npu],
                    "split_item": 2,
                    "output_dtype": output_dtype,
                    "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
                }
                if use_antiquant_scale(gmm1_x_dtype, gmm_weight_dtype):
                    gmm1_kwargs["antiquant_scale"] = [gmm1_weight_scale.transpose(1, 2)]
                else:
                    gmm1_kwargs["scale"] = [gmm1_weight_scale.transpose(1, 2)]
                if gmm_weight_dtype is not None:
                    gmm1_kwargs["weight_dtype"] = gmm_weight_dtype
                if gmm1_x_dtype is not None:
                    gmm1_kwargs["x_dtype"] = gmm1_x_dtype
                y1 = torch_npu.npu_grouped_matmul(**gmm1_kwargs)[0]
                y_swiglu_quant, y_swiglu_scale = self._activation_mx_quant(
                    y1, swiglu_quant_dtype
                )

            gmm2_kwargs = {
                "x": [y_swiglu_quant],
                "weight": [gmm2_weight.transpose(1, 2)],
                "group_list": expert_token_nums_npu,
                "group_type": 0,
                "group_list_type": 1,
                "per_token_scale": [y_swiglu_scale],
                "split_item": 2,
                "output_dtype": output_dtype,
                "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
            }
            if use_antiquant_scale(gmm2_x_dtype, gmm_weight_dtype):
                gmm2_kwargs["antiquant_scale"] = [gmm2_weight_scale.transpose(1, 2)]
            else:
                gmm2_kwargs["scale"] = (
                    [gmm2_weight_scale.transpose(1, 2)]
                    if quant_mode > 0
                    else None
                )
                gmm2_kwargs["scale_dtype"] = FLOAT8_E8M0FNU_DTYPE
            if gmm_weight_dtype is not None:
                gmm2_kwargs["weight_dtype"] = gmm_weight_dtype
            if gmm2_x_dtype is not None:
                gmm2_kwargs["x_dtype"] = gmm2_x_dtype
            y2 = torch_npu.npu_grouped_matmul(**gmm2_kwargs)[0]
        else:
            raise NotImplementedError("alltoallv dispatch path currently supports quant_mode == 4 only")

        return y2

    def _run_shared_ffn(
            self, x, quant_mode, gmm1_weight, output_dtype,
            gmm1_weight_scale, gmm2_weight, gmm2_weight_scale,
            weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
            shared_token_nums=None
    ):
        if self.shared_expert_num <= 0:
            return None

        quant_x, dynamic_scales = self._quantize_dispatch_x(x, y_dtype)
        shared_x = (
            quant_x
            if self.shared_expert_num == 1
            else quant_x.repeat(
                self.shared_expert_num, *([1] * (quant_x.dim() - 1))
            )
        )
        shared_scales = (
            dynamic_scales
            if self.shared_expert_num == 1
            else dynamic_scales.repeat(
                self.shared_expert_num,
                *([1] * (dynamic_scales.dim() - 1)),
            )
        )
        if shared_token_nums is None:
            shared_token_nums = torch.full(
                (self.shared_expert_num,),
                x.shape[0],
                dtype=torch.int64,
                device=x.device,
            )
        shared_y = self._run_ffn(
            shared_x,
            shared_scales,
            shared_token_nums,
            quant_mode,
            gmm1_weight,
            output_dtype,
            self._slice_shared_weights(gmm1_weight_scale),
            gmm2_weight,
            self._slice_shared_weights(gmm2_weight_scale),
            weight_dtype,
            swiglu_quant_dtype,
            x_dtype,
            y_dtype,
        )
        if self.shared_expert_num == 1:
            return shared_y.view(x.shape[0], x.shape[1])
        return shared_y.view(
            self.shared_expert_num, x.shape[0], x.shape[1]
        ).sum(dim=0)

    def _shared_expert_multistream_enabled(self):
        return self.enable_shared_expert_multistream and self.shared_expert_num > 0

    def _use_torchair_graph_multistream(self):
        return (
            self._shared_expert_multistream_enabled()
            and self.is_graph
            and self.graph_type in (1, 2)
        )

    def _ensure_native_shared_expert_multistream(self):
        if (
            not self._shared_expert_multistream_enabled()
            or self._use_torchair_graph_multistream()
        ):
            return False
        if self.shared_expert_stream is None:
            self.shared_expert_stream = torch.npu.Stream()
            self.events = {
                EventType.SHARED_INPUT_READY: torch.npu.Event(),
                EventType.MOE_SHARED: torch.npu.Event(),
            }
        return True

    def _record_tensor_stream(self, tensor, stream):
        if isinstance(tensor, torch.Tensor):
            tensor.record_stream(stream)

    def _start_shared_ffn(
            self, x, quant_mode, gmm1_weight, output_dtype,
            gmm1_weight_scale, gmm2_weight, gmm2_weight_scale,
            weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
            shared_token_nums=None
    ):
        if self.shared_expert_num <= 0:
            return None, None
        if not self._shared_expert_multistream_enabled():
            return self._run_shared_ffn(
                x, quant_mode, gmm1_weight, output_dtype,
                gmm1_weight_scale, gmm2_weight, gmm2_weight_scale,
                weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                shared_token_nums,
            ), None

        if self._use_torchair_graph_multistream():
            with torchair.scope.npu_stream_switch(
                self.shared_expert_graph_stream_tag, 0
            ):
                shared_output = self._run_shared_ffn(
                    x, quant_mode, gmm1_weight, output_dtype,
                    gmm1_weight_scale, gmm2_weight, gmm2_weight_scale,
                    weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                    shared_token_nums,
                )
            return shared_output, "torchair_graph"

        if not self._ensure_native_shared_expert_multistream():
            return self._run_shared_ffn(
                x, quant_mode, gmm1_weight, output_dtype,
                gmm1_weight_scale, gmm2_weight, gmm2_weight_scale,
                weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                shared_token_nums,
            ), None

        for tensor in (
            x,
            gmm1_weight,
            gmm1_weight_scale,
            gmm2_weight,
            gmm2_weight_scale,
            shared_token_nums,
        ):
            self._record_tensor_stream(tensor, self.shared_expert_stream)
        self.events[EventType.SHARED_INPUT_READY].record()
        with torch.npu.stream(self.shared_expert_stream):
            self.events[EventType.SHARED_INPUT_READY].wait()
            shared_output = self._run_shared_ffn(
                x, quant_mode, gmm1_weight, output_dtype,
                gmm1_weight_scale, gmm2_weight, gmm2_weight_scale,
                weight_dtype, swiglu_quant_dtype, x_dtype, y_dtype,
                shared_token_nums,
            )
            self.events[EventType.MOE_SHARED].record()
        return shared_output, "native"

    def _merge_shared_output(
            self, output_combine_npu, shared_output, shared_done_event
    ):
        if shared_output is None:
            return output_combine_npu
        if shared_done_event is None:
            return output_combine_npu + shared_output
        if shared_done_event == "torchair_graph":
            output_combine_npu = torchair.scope.npu_wait_tensor(
                output_combine_npu, shared_output
            )
            return output_combine_npu + shared_output

        self._record_tensor_stream(shared_output, torch.npu.current_stream())
        self.events[EventType.MOE_SHARED].wait()
        return output_combine_npu + shared_output

    def _alltoallv_combine(self, y2, dispatch_ctx):
        if (
            y2.shape[0] > 0
            and dispatch_ctx["local_expert_num"] > 1
            and dispatch_ctx["reversed_global_mapping"] is not None
        ):
            y2 = torch_npu.npu_moe_token_unpermute(y2, dispatch_ctx["reversed_global_mapping"])

        local_x = self._all_to_all_single(
            y2,
            dispatch_ctx["input_splits"],
            dispatch_ctx["output_splits"],
        )
        combine_out = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=local_x,
            sorted_indices=dispatch_ctx["reversed_local_mapping"].to(torch.int32),
            probs=dispatch_ctx["topk_weights"],
            restore_shape=dispatch_ctx["hidden_shape"],
        )
        return combine_out.view(dispatch_ctx["hidden_shape"])

    def forward(
            self, x, expert_ids, expert_scales, x_active_mask, group_ep, group_tp, ep_rank_id, ep_world_size,
            moe_expert_num, quant_mode, global_bs, gmm1_weight, output_dtype, gmm1_weight_scale, gmm2_weight,
            gmm2_weight_scale, y_dtype, comm_quant_mode, rank, weight_dtype, swiglu_quant_dtype, x_dtype,
            shared_token_nums=None
    ):
        gmm1_weight, gmm2_weight = self._get_gmm_weights(gmm1_weight, gmm2_weight)
        shared_gmm1_weight, moe_gmm1_weight = self._split_prepared_weights(
            gmm1_weight
        )
        shared_gmm2_weight, moe_gmm2_weight = self._split_prepared_weights(
            gmm2_weight
        )
        moe_gmm1_weight_scale = self._slice_moe_weights(gmm1_weight_scale)
        moe_gmm2_weight_scale = self._slice_moe_weights(gmm2_weight_scale)
        local_loop = (
            self.loop
            if self.is_performance
            and self.is_graph
            and not self.performance_loop_outside_graph
            else 1
        )
        original_x = x
        for loop_idx in range(local_loop):
            shared_output, shared_done_event = self._start_shared_ffn(
                x,
                quant_mode,
                shared_gmm1_weight,
                output_dtype,
                gmm1_weight_scale,
                shared_gmm2_weight,
                gmm2_weight_scale,
                weight_dtype,
                swiglu_quant_dtype,
                x_dtype,
                y_dtype,
                shared_token_nums,
            )
            if self._need_alltoallv_dispatch(x):
                expand_x, expert_token_nums_npu, dispatch_ctx = self._alltoallv_dispatch(
                    x, expert_ids, expert_scales, ep_rank_id, ep_world_size, moe_expert_num
                )
                expand_x_npu, dynamic_scales_npu = self._quantize_dispatch_x(expand_x, y_dtype)
                y2 = self._run_ffn(
                    expand_x_npu, dynamic_scales_npu, expert_token_nums_npu,
                    quant_mode, moe_gmm1_weight, output_dtype,
                    moe_gmm1_weight_scale, moe_gmm2_weight,
                    moe_gmm2_weight_scale, weight_dtype,
                    swiglu_quant_dtype, x_dtype, y_dtype
                )
                output_combine_npu = self._alltoallv_combine(y2, dispatch_ctx)
                output_combine_npu = self._merge_shared_output(
                    output_combine_npu, shared_output, shared_done_event
                )
                output_dispatch_npu = (
                    expand_x_npu,
                    dynamic_scales_npu,
                    None,
                    expert_token_nums_npu,
                    torch.tensor(dispatch_ctx["output_splits"], dtype=torch.int64, device=x.device),
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
                continue

            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                x_active_mask=None,
                group_ep=group_ep,
                ep_world_size=ep_world_size,
                ep_rank_id=ep_rank_id,
                tp_world_size=1,
                moe_expert_num=moe_expert_num,
                expert_shard_type=0,
                shared_expert_num=1,
                shared_expert_rank_num=0,
                quant_mode=quant_mode,
                global_bs=global_bs,
                y_dtype=y_dtype,
            )

            (expand_x_npu, dynamic_scales_npu, assist_info_for_combine_npu, expert_token_nums_npu, ep_recv_counts_npu,
                tp_send_counts_npu, _) = output_dispatch_npu
            dynamic_scales_npu = dynamic_scales_npu.reshape(
                dynamic_scales_npu.shape[0], dynamic_scales_npu.shape[1] // 2, 2
            )
            y2 = self._run_ffn(
                expand_x_npu, dynamic_scales_npu, expert_token_nums_npu,
                quant_mode, moe_gmm1_weight, output_dtype,
                moe_gmm1_weight_scale, moe_gmm2_weight,
                moe_gmm2_weight_scale, weight_dtype,
                swiglu_quant_dtype, x_dtype, y_dtype
            )

            ####非量化暂不支持
            # else:
            #     y1_int32 = torch_npu.npu_grouped_matmul(
            #         x=[expand_x_npu],
            #         weight=[gmm1_weight],
            #         split_item=3,
            #         group_list_type=1,  # 默认为0，代表前缀和形式
            #         group_type=0,  # 0代表m轴分组
            #         group_list=expert_token_nums_npu,
            #         output_dtype=output_dtype)[0]
            #     y1 = torch_npu.npu_swiglu(y1_int32)
            #     y2 = torch_npu.npu_grouped_matmul(x=[y1],
            #                                         weight=[gmm2_weight],
            #                                         scale=None,
            #                                         per_token_scale=None,
            #                                         split_item=2,
            #                                         group_list_type=1,
            #                                         group_type=0,
            #                                         group_list=expert_token_nums_npu,
            #                                         output_dtype=output_dtype,
            #                                     )[0]

            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
                expand_x=y2,
                expert_ids=expert_ids,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                expert_scales=expert_scales,
                x_active_mask=None,
                group_ep=group_ep,
                ep_world_size=ep_world_size,
                ep_rank_id=ep_rank_id,
                moe_expert_num=moe_expert_num,
                tp_send_counts=tp_send_counts_npu,
                tp_world_size=1,
                expert_shard_type=0,
                shared_expert_num=1,
                shared_expert_rank_num=0,
                global_bs=global_bs,
                comm_quant_mode=comm_quant_mode,
            )
            output_combine_npu = self._merge_shared_output(
                output_combine_npu, shared_output, shared_done_event
            )
            if (
                local_loop > 1
                and loop_idx + 1 < local_loop
                and output_combine_npu.shape == original_x.shape
                and output_combine_npu.dtype == original_x.dtype
            ):
                x = output_combine_npu * 0 + original_x

        return output_combine_npu, output_dispatch_npu
