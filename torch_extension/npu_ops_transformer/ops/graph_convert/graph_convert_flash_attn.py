try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, TensorDescriptor, TensorDef
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False, False, False],
        [False, False, False, True, True, True, True, True, True])
    def FlashAttn(q: Tensor,
        k: Tensor,
        v: Tensor,
        block_table: Optional[Tensor],
        cu_seqlens_q: Optional[Tensor],
        cu_seqlens_kv: Optional[Tensor],
        seqused_q: Optional[Tensor],
        seqused_kv: Optional[Tensor],
        sinks: Optional[Tensor],
        metadata: Optional[Tensor],
        softmax_scale: int = 1.0,
        mask_mode: int = 1,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: string = "BSND",
        layout_kv: string = "BSND",
        layout_out: string = "BSND",
        return_softmax_lse: int = 0,
        deterministic: int = 0):
        
        result = q.new_empty(q.size())
        return result
        
    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_flash_attn.default)
    def convert_npu_flash_attn(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_table: Tensor = None,
        cu_seqlens_q: Tensor = None,
        cu_seqlens_kv: Tensor = None,
        seqused_q: Tensor = None,
        seqused_kv: Tensor = None,
        sinks: Tensor = None,
        metadata: Tensor = None,
        softmax_scale: int = 1.0,
        mask_mode: int = 1,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: string = "BSND",
        layout_kv: string = "BSND",
        layout_out: string = "BSND",
        return_softmax_lse: int = 0,
        deterministic: int = 0):

        
        raise AssertionError(f"GE not supported!")
        return FlashAttn(q = q,
                        k = k,
                        v = v,
                        block_table = block_table,
                        cu_seqlens_q = cu_seqlens_q,
                        cu_seqlens_kv = cu_seqlens_kv,
                        seqused_q = seqused_q,
                        seqused_kv = seqused_kv,
                        sinks =sinks,
                        metadata = metadata,
                        softmax_scale = softmax_scale,
                        mask_mode = mask_mode,
                        win_left = win_left,
                        win_right = win_right,
                        max_seqlen_q = max_seqlen_q,
                        max_seqlen_kv = max_seqlen_kv,
                        layout_q = layout_q,
                        layout_kv = layout_kv,
                        layout_out = layout_out,
                        return_softmax_lse = return_softmax_lse,
                        deterministic = deterministic)