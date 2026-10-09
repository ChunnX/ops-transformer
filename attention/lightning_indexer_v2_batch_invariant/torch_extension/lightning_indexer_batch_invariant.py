# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from cann_ops_transformer.op_builder import OpBuilder
from cann_ops_transformer.op_builder.lightning_indexer import (
    lightning_indexer_schema,
    register_lightning_indexer,
)


class LightningIndexerBatchInvariantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("lightning_indexer_batch_invariant", category="attention")

    def sources(self):
        return ["csrc/attention/lightning_indexer_batch_invariant.cpp"]

    def schema(self):
        return lightning_indexer_schema(self.name)

    def register_meta(self):
        self.forward = register_lightning_indexer(self)


lightning_indexer_batch_invariant_op_builder = LightningIndexerBatchInvariantOpBuilder()
lightning_indexer_batch_invariant_op_builder._ensure_initialized()
lightning_indexer_batch_invariant = lightning_indexer_batch_invariant_op_builder.forward
