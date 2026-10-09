# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 本目录为 lightning_indexer_v2 的 torch 接口，随主干 cann_ops_transformer 包发布
# （torch_extension/setup.py 的 attention/<op>/torch_extension 两级收集自动生效）。
# 本包自包含导出全部接口：lightning_indexer / lightning_indexer_metadata（现网入口，
# lightning_indexer 旧入口内部固定以 candidate 关闭路径调用新签名 aclnn）与
# lightning_indexer_candidate（candidate 两级 TopK 扩展，仅 ascend910b/ascend910_93
# 支持开启）；注册方式与仓库其他算子一致（无条件 define/impl，不做逐算子幂等守卫）。
__all__ = [
    "lightning_indexer",
    "lightning_indexer_v2",
    "lightning_indexer_metadata",
    "lightning_indexer_candidate",
]

from .lightning_indexer import (
    lightning_indexer,
    lightning_indexer_metadata,
    lightning_indexer_candidate,
)

lightning_indexer_v2 = lightning_indexer

from . import graph_convert_lightning_indexer
