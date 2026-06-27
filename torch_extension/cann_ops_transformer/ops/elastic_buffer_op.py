# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from cann_ops_transformer.op_builder.builder import OpBuilder


class ElasticBufferOpBuilder(OpBuilder):
    """OpBuilder for ElasticBuffer operations"""
    
    def __init__(self):
        super(ElasticBufferOpBuilder, self).__init__("npu_elastic_buffer")
    
    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/elastic_buffer.cpp']
    
    def schema(self):
        """PyTorch operator signature."""
        return None
    
    def register_meta(self):
        """Meta implementation (optional for JIT compiled ops)"""
        pass
    
    def extra_ldflags(self):
        """
        Extra link flags for HCCL and ACL libraries
        """
        flags = super().extra_ldflags()
        flags.append('-L' + os.path.join(self._cann_path, 'lib64'))
        flags.append('-lhcomm')
        flags.append('-lascendcl')
        return flags
    
    def include_paths(self):
        """
        Override include paths to ensure CANN headers are prioritized
        """
        paths = [
            os.path.join(self._cann_path, 'include'),
            os.path.join(self._torch_npu_path, 'include'),
            os.path.join(self._torch_npu_path, 'include/third_party/hccl/inc'),
            os.path.join(self._torch_npu_path, 'include/third_party/acl/inc'),
            os.path.join(self._package_path, 'common/inc')
        ]
        return paths


# Instantiate the builder and load module
elastic_buffer_op_builder = ElasticBufferOpBuilder()
_elastic_buffer_ops = elastic_buffer_op_builder.load()