# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

try:
    import torchair
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:
    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.lightning_indexer_kl_loss.default
    )
    def convert_lightning_indexer_kl_loss(
        target_score: Tensor,
        index_probs: Tensor,
        *,
        eps: float = 1e-9,
        meta_outputs: TensorSpec = None,
    ):
        """Convert lightning_indexer_kl_loss to GE LightningIndexerKLLoss op.

        REG_OP:
            LightningIndexerKLLoss(
                INPUT target_score : tensor,
                INPUT index_probs  : tensor,
                OUTPUT loss        : tensor,
                ATTR eps           : float
            )

        Computation:
            P = target_score / sum(target_score, dim=-1, keepdim=True)
            log_P = log(clamp_min(P, eps))
            log_Y = log(index_probs + eps)
            loss = sum((log_P - log_Y) * target_score)
        """
        return torchair.ge.custom_op(
            "LightningIndexerKLLoss",
            inputs={
                "target_score": target_score,
                "index_probs": index_probs,
            },
            attrs={
                "eps": attr.Float(eps),
            },
            outputs=["loss"],
        )
