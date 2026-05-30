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
        [False, False, False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, False, False, False, False, False, False])
    def Compressor(x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state_cache: Tensor,
        ape: Tensor,
        norm_weight: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        state_block_table: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        seqused: Optional[Tensor],
        start_pos: Optional[Tensor],
        rope_head_dim: int = 64,
        cmp_ratio: int = 4,
        coff: int = 1,
        norm_eps: float = 1e-6,
        rotary_mode: int = 1,
        cache_mode: int = 1,
        state_cache_stride_dim0: int = 0):
        
        result = x.new_empty(x.size())
        return result
        
    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_compressor.default)
    def convert_npu_compressor(
        x: Tensor,
        wkv: Tensor,
        wgate: Tensor,
        state_cache: Tensor,
        ape: Tensor,
        norm_weight: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        state_block_table: Tensor = None,
        cu_seqlens: Tensor = None,
        seqused: Tensor = None,
        start_pos: Tensor = None,
        rope_head_dim: int = 64,
        cmp_ratio: int = 4,
        coff: int = 1,
        norm_eps: float = 1e-6,
        rotary_mode: int = 1,
        cache_mode: int = 1,
        state_cache_stride_dim0: int = 0):

        
        raise AssertionError(f"GE not supported!")
        return Compressor(x = x,
                        wkv = wkv,
                        wgate = wgate,
                        state_cache = state_cache,
                        ape = ape,
                        norm_weight = norm_weight,
                        rope_sin = rope_sin,
                        rope_cos = rope_cos,
                        state_block_table = state_block_table,
                        cu_seqlens = cu_seqlens,
                        seqused = seqused,
                        start_pos = start_pos,
                        rope_head_dim = rope_head_dim,
                        cmp_ratio = cmp_ratio,
                        coff = coff,
                        norm_eps = norm_eps,
                        rotary_mode = rotary_mode,
                        cache_mode = cache_mode,
                        state_cache_stride_dim0 = state_cache_stride_dim0)