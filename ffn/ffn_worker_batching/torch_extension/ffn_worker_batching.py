

from typing import List, Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

_CONTEXT_BYTES = 1024
_MAX_SESSIONS = 1024
_MAX_EXPERTS = 8192
_MAX_SELECTED_EXPERTS = 64
_MX_BLOCK_SIZE = 32
_FP4 = 5
_TOKEN_DTYPE_NAMES = (
    "float16", "bfloat16", "int8", "float8_e5m2", "float8_e4m3fn", "float4_e2m1fn_x2"
)
_Outputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                 torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _validate(schedule_context, expert_num, max_out_shape, token_dtype, need_schedule, layer_num, sync_flag):
    # Inspect tensor metadata only: reading pointer-bearing context on CPU would
    # synchronize the stream and is invalid for Meta/FakeTensor execution.
    torch._check(schedule_context.dtype == torch.int8, lambda: "schedule_context must have dtype int8")
    torch._check(schedule_context.dim() == 1, lambda: "schedule_context must be one-dimensional")
    torch._check(schedule_context.is_contiguous(), lambda: "schedule_context must be contiguous")
    torch._check(schedule_context.numel() >= _CONTEXT_BYTES, lambda: "schedule_context needs at least 1024 bytes")
    torch._check(len(max_out_shape) == 4, lambda: "max_out_shape must be [A, BS, K, H]")
    a, bs, k, h = max_out_shape
    torch._check(0 < a <= _MAX_SESSIONS, lambda: f"A must be in [1, {_MAX_SESSIONS}], got {a}")
    torch._check(bs > 0, lambda: f"BS must be positive, got {bs}")
    torch._check(0 < k <= _MAX_SELECTED_EXPERTS, lambda: f"K must be in [1, {_MAX_SELECTED_EXPERTS}], got {k}")
    torch._check(h > 0, lambda: f"H must be positive, got {h}")
    torch._check(0 < expert_num <= _MAX_EXPERTS, lambda: f"expert_num must be in [1, {_MAX_EXPERTS}], got {expert_num}")
    torch._check(0 <= token_dtype < len(_TOKEN_DTYPE_NAMES), lambda: f"token_dtype must be in [0, 5], got {token_dtype}")
    torch._check(need_schedule in (0, 1), lambda: f"need_schedule must be 0 or 1, got {need_schedule}")
    torch._check(sync_flag in (0, 1), lambda: f"sync_flag must be 0 or 1, got {sync_flag}")
    torch._check(0 <= layer_num <= expert_num, lambda: f"layer_num must be in [0, expert_num], got {layer_num}")
    torch._check(token_dtype != _FP4 or h % 2 == 0, lambda: f"FP4 requires even H, got {h}")


def _output_specs(expert_num, max_out_shape, token_dtype, mx_output_uint8=False):
    a, bs, k, h = max_out_shape
    y = a * bs * k
    dtype = getattr(torch, _TOKEN_DTYPE_NAMES[token_dtype])
    scale_dtype = torch.float8_e8m0fnu if token_dtype >= 3 else torch.float32
    if mx_output_uint8 and token_dtype >= 3:
        scale_dtype = torch.uint8
        if token_dtype == _FP4:
            dtype = torch.uint8
    scale_shape = (y, h // _MX_BLOCK_SIZE + (h % _MX_BLOCK_SIZE != 0)) if token_dtype >= 3 else (y,)
    # PyTorch's FP4 x2 dtype counts packed bytes; ACLNN counts logical nibbles.
    return [((y, h // 2 if token_dtype == _FP4 else h), dtype),
            ((expert_num, 2), torch.int64), *[((y,), torch.int32)] * 4,
            (scale_shape, scale_dtype), ((1,), torch.int64)]


class _FfnWorkerBatchingOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("ffn_worker_batching", category="ffn")

    def sources(self):
        return ["csrc/ffn/ffn_worker_batching.cpp"]

    def schema(self) -> str:
        # RECV updates polling_index and descriptors. Mark the context mutable
        # even though the eight returned tensors are newly allocated.
        return ("ffn_worker_batching(Tensor(a!) schedule_context, int expert_num, int[] max_out_shape, "
                "*, int token_dtype=0, int need_schedule=0, int layer_num=0, int sync_flag=0, bool mx_output_uint8=False) "
                "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def ffn_worker_batching_meta(schedule_context, expert_num, max_out_shape, *,
                                    token_dtype=0, need_schedule=0, layer_num=0, sync_flag=0, mx_output_uint8=False):
            _validate(schedule_context, expert_num, max_out_shape, token_dtype, need_schedule, layer_num, sync_flag)
            outputs = [schedule_context.new_empty(shape, dtype=dtype)
                       for shape, dtype in _output_specs(expert_num, max_out_shape, token_dtype, mx_output_uint8)]
            if token_dtype == _FP4 and tuple(outputs[0].shape) == (1, 1):
                # Match the wrapper's unambiguous FP4 packing-axis stride.
                outputs[0] = outputs[0].as_strided((1, 1), (2, 1))
            return tuple(outputs)


_ffn_worker_batching_op_builder = _FfnWorkerBatchingOpBuilder()
_ffn_worker_batching_op_builder._ensure_initialized()


@impl(get_as_library(), _ffn_worker_batching_op_builder.name, "PrivateUse1")
def _ffn_worker_batching(schedule_context, expert_num, max_out_shape, *,
                         token_dtype=0, need_schedule=0, layer_num=0, sync_flag=0, mx_output_uint8=False):
    _validate(schedule_context, expert_num, max_out_shape, token_dtype, need_schedule, layer_num, sync_flag)
    return _ffn_worker_batching_op_builder.load().ffn_worker_batching(
        schedule_context, expert_num, max_out_shape, token_dtype, need_schedule, layer_num, sync_flag, mx_output_uint8)


def ffn_worker_batching(schedule_context: torch.Tensor, expert_num: int, max_out_shape: List[int], *,
                        token_dtype: int = 0, need_schedule: int = 0, layer_num: int = 0,
                        sync_flag: int = 0, mx_output_uint8: bool = False) -> _Outputs:
    """在 Ascend950 上按专家重排 token，桥接 aclnnFfnWorkerBatching。

    Args:
        schedule_context: 连续一维 NPU int8 张量，至少 1024 字节，包含真实设备指针。
            RECV 会更新 polling_index 并清理指针指向的已消费描述符；调用方须保持所有
            被引用缓冲存活至执行完成，并保证其位于同一设备及正确的流依赖。
        expert_num: 专家数，1..8192。
        max_out_shape: [A, BS, K, H]；A=1..1024、BS>0、K=1..64、H>0，K 含共享专家。
        token_dtype: 0 FP16、1 BF16、2 INT8、3 FP8 E5M2、4 FP8 E4M3FN、5 FP4 E2M1。
            FP4 要求 H 为偶数，返回 torch.float4_e2m1fn_x2 打包张量。
        mx_output_uint8: MX 图模式设为 True，FP4 y 和 E8M0 scale 以 uint8 返回原始字节，
            不做数值转换；FP8 y 保留原生 dtype。默认 False 保留 eager 原生 dtype。
        need_schedule: 0 直接 batching，1 接收并 batching，默认 0。
        layer_num: 层数，0..expert_num，默认 0。
        sync_flag: 默认 0 等待全部 session；1 消费首个非空快照，仅 need_schedule=1 生效。

    Returns:
        y、group_list、session_ids、micro_batch_ids、token_ids、expert_offsets、
        dynamic_scale、actual_token_num。Y=A*BS*K；y 为 [Y,H]，FP4 为 [Y,H/2]；
        group_list 为 [expert_num,2]；四个索引为 [Y]；MX scale 为 [Y,ceil(H/32)]，
        其它 scale 为 [Y]；actual_token_num 为 [1]。FP16/BF16 scale 不含有效数据，
        y/索引/scale 仅前 actual_token_num 行有效。支持 eager 和 Meta/FakeTensor，
        全部 token_dtype 支持固定形状推理图（MX 需 mx_output_uint8=True）：torch.compile(..., backend="npugraph_ex",
        fullgraph=True, dynamic=False)。上下文及其引用的缓冲必须在图执行期间
        保持有效，外部生产者需正确同步。
        不支持反向和 TorchAir GE Converter。
    """
    return torch.ops.cann_ops_transformer.ffn_worker_batching(
        schedule_context, expert_num, max_out_shape, token_dtype=token_dtype,
        need_schedule=need_schedule, layer_num=layer_num, sync_flag=sync_flag,
        mx_output_uint8=mx_output_uint8)
