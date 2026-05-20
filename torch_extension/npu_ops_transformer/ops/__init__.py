# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# --- EP normal MoE pipeline (dispatch_layout -> notify+dispatch -> combine) ---
# Category 1: layout only (no communication domain)
from .dispatch_layout import npu_dispatch_layout
from .graph_convert.graph_convert_dispatch_layout import convert_npu_dispatch_layout

# Category 2: dispatch (notify_dispatch + dispatch_normal; fused entry for Buffer)
from .notify_dispatch import npu_notify_dispatch
from .dispatch_normal import npu_dispatch_normal
from .moe_dispatch_prefill import npu_moe_dispatch_prefill
from .graph_convert.graph_convert_notify_dispatch import convert_npu_notify_dispatch
from .graph_convert.graph_convert_dispatch_normal import convert_npu_dispatch_normal

# Category 3: combine
from .combine_normal import npu_combine_normal
from .moe_combine_prefill import npu_moe_combine_prefill
from .graph_convert.graph_convert_combine_normal import convert_npu_combine_normal

# High-level Buffer API (layout + fused dispatch/combine)
from .buffer import Buffer

# --- Other MoE / attention ops ---
from .sparse_flash_mla import npu_sparse_flash_mla
from .moe_distribute_dispatch_v2 import npu_moe_distribute_dispatch_v2
from .moe_distribute_combine_v2 import npu_moe_distribute_combine_v2
from .moe_distribute_dispatch_v3 import npu_moe_distribute_dispatch_v3
from .moe_distribute_combine_v3 import npu_moe_distribute_combine_v3
from .mega_moe import (
    get_symm_buffer_for_mega_moe,
    mega_moe,
    npu_get_mega_moe_ccl_buffer_size,
)
from .deep_ep import MoeDistributeBuffer
from .graph_convert.graph_convert_moe_distribute_dispatch_v3 import converter_moe_distribute_dispatch_v3
from .graph_convert.graph_convert_moe_distribute_combine_v3 import convert_npu_moe_distribute_combine_v3
from .graph_convert.graph_convert_mega_moe import convert_npu_mega_moe
from .flash_attn import npu_flash_attn
from .graph_convert.graph_convert_flash_attn import convert_npu_flash_attn
from .flash_attn_metadata import npu_flash_attn_metadata
from .lightning_indexer_v2_metadata import npu_lightning_indexer_v2_metadata
from .quant_lightning_indexer_v2_metadata import npu_quant_lightning_indexer_v2_metadata
from .mixed_quant_sparse_flash_mla import npu_mixed_quant_sparse_flash_mla
from .graph_convert.graph_convert_lightning_indexer_v2_metadata import convert_npu_lightning_indexer_v2_metadata
from .graph_convert.graph_convert_quant_lightning_indexer_v2_metadata import (
    convert_npu_quant_lightning_indexer_v2_metadata,
)
