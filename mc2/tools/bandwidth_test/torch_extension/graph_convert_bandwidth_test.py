# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch

try:
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.bandwidth_test.default)
    def convert_bandwidth_test(context, x, dstrank_id, group, world_size,
                               max_bs, test_mode, ccl_buffer_size, comm_alg="",
                               aiv_num=-1, batch_tokens=0, shared_jetty=False, shared_jetty_num=1, drain_threshold=8192, meta_outputs=None):
        return ge_op(
            op_type="BandwidthTest",
            inputs={"context": context, "x": x, "dst_rank_id": dstrank_id},
            attrs={"group": attr.Str(group), "world_size": attr.Int(world_size),
                   "max_bs": attr.Int(max_bs), "mode": attr.Int(test_mode),
                   "ccl_buffer_size": attr.Int(ccl_buffer_size),
                   "comm_alg": attr.Str(comm_alg), "aiv_num": attr.Int(aiv_num),
                   "batch_tokens": attr.Int(batch_tokens),
                   "shared_jetty": attr.Bool(shared_jetty), "shared_jetty_num": attr.Int(shared_jetty_num),
                   "drain_threshold": attr.Int(drain_threshold)},
            outputs=["y", "receive_cnt"],
            ir=IrDef("BandwidthTest")
            .input("context", "DT_INT32")
            .input("x", "DT_FLOAT16")
            .input("dst_rank_id", "DT_INT32")
            .required_attr("group", attr.Str)
            .required_attr("world_size", attr.Int)
            .required_attr("max_bs", attr.Int)
            .required_attr("mode", attr.Int)
            .required_attr("ccl_buffer_size", attr.Int)
            .attr("comm_alg", attr.Str(""))
            .attr("aiv_num", attr.Int(-1))
            .attr("batch_tokens", attr.Int(0))
            .attr("shared_jetty", attr.Bool(False))
            .attr("shared_jetty_num", attr.Int(1))
            .attr("drain_threshold", attr.Int(8192))
            .output("y", "DT_FLOAT16")
            .output("receive_cnt", "DT_INT32"))
