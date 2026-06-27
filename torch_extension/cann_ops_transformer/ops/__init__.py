# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from .sparse_flash_mla_grad import sparse_flash_mla_grad, sparse_flash_mla_grad_metadata
from .moe_distribute_dispatch import npu_moe_distribute_dispatch
from .moe_distribute_combine import npu_moe_distribute_combine
from .mega_moe import (
    get_symm_buffer_for_mega_moe,
    mega_moe,
    npu_get_mega_moe_ccl_buffer_size
)
from .deep_ep import MoeDistributeBuffer
from .flash_attn import npu_flash_attn
from .flash_attn_metadata import npu_flash_attn_metadata
from .mixed_quant_sparse_flash_mla import mixed_quant_sparse_flash_mla, mixed_quant_sparse_flash_mla_metadata
from .sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from .sparse_lightning_indexer_kl_loss_grad import (
    sparse_lightning_indexer_kl_loss_grad,
    sparse_lightning_indexer_kl_loss_grad_metadata
)
from .lightning_indexer import lightning_indexer, lightning_indexer_metadata
from .quant_lightning_indexer import quant_lightning_indexer, quant_lightning_indexer_metadata
from .mhc_post import mhc_post
from .mhc_pre_sinkhorn import mhc_pre_sinkhorn
from . import graph_convert as _graph_convert
