# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

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
    @auto_convert_to_tensor([False, False], [False, False])
    def BandwidthTest(x: Tensor, dstrank_id: Tensor, *,
                      group: str, world_size: int, max_bs: int, mode: int,
                      comm_alg: str, aiv_num: int, dependencies=[], node_name=None):
        """REG_OP(BandwidthTest)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32}))\n
        .INPUT(dstrank_id, TensorType({DT_INT32}))\n
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32}))\n
        .OUTPUT(receive_cnt, TensorType({DT_INT32}))\n
        .REQUIRED_ATTR(group, String)\n
        .REQUIRED_ATTR(world_size, Int)\n
        .REQUIRED_ATTR(max_bs, Int)\n
        .REQUIRED_ATTR(mode, Int)\n
        .REQUIRED_ATTR(comm_alg, String)\n
        .REQUIRED_ATTR(aiv_num, Int)\n
        """
        inputs = {
            "x": x,
            "dstrank_id": dstrank_id,
        }

        attrs = {
            "group": attr.Str(group),
            "world_size": attr.Int(world_size),
            "max_bs": attr.Int(max_bs),
            "mode": attr.Int(mode),
            "comm_alg": attr.Str(comm_alg),
            "aiv_num": attr.Int(aiv_num),
        }

        outputs = [
            "y",
            "receive_cnt",
        ]

        return ge_op(
            op_type="BandwidthTest",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("BandwidthTest")
            .input("x", "DT_BF16, DT_FLOAT16, DT_FLOAT32")
            .input("dstrank_id", "DT_INT32")
            .required_attr("group", attr.Str)
            .required_attr("world_size", attr.Int)
            .required_attr("max_bs", attr.Int)
            .required_attr("mode", attr.Int)
            .required_attr("comm_alg", attr.Str)
            .required_attr("aiv_num", attr.Int)
            .output("y", "DT_BF16, DT_FLOAT16, DT_FLOAT32")
            .output("receive_cnt", "DT_INT32")
        )

    @declare_supported([
        Support(torch.float16, (8, 128)),
        Support(torch.bfloat16, (8, 128)),
        Support(torch.float32, (8, 128)),
    ])
    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_bandwidth_test.default)
    def convert_npu_bandwidth_test(
        x: Tensor,
        dstrank_id: Tensor,
        config,
        *,
        meta_outputs: TensorSpec = None):
        
        return BandwidthTest(
            x=x,
            dstrank_id=dstrank_id,
            group=config.group,
            world_size=config.world_size,
            max_bs=config.max_bs,
            mode=config.mode,
            comm_alg=config.comm_alg,
            aiv_num=config.aiv_num
        )