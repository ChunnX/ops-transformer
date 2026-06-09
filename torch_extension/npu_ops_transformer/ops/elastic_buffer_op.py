# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import os
import torch
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder, AS_LIBRARY


class ElasticBufferOpBuilder(OpBuilder):
    """OpBuilder for ElasticBuffer engram_fetch operations"""
    
    def __init__(self):
        super(ElasticBufferOpBuilder, self).__init__("npu_elastic_buffer")
    
    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/elastic_buffer.cpp']
    
    def schema(self) -> str:
        """PyTorch operator signature."""
        return None  # Not using schema registration
    
    def register_meta(self):
        """Meta implementation (optional for JIT compiled ops)"""
        pass
    
    def include_paths(self):
        """
        Override include paths to ensure CANN 9.1.0 headers are prioritized
        over torch_npu's old ACL headers
        """
        paths = [
            # CANN 9.1.0 headers FIRST (contains ACL_FLOAT8_E5M2 etc.)
            os.path.join(self._cann_path, 'include'),
            # Then torch_npu headers
            os.path.join(self._torch_npu_path, 'include'),
            os.path.join(self._torch_npu_path, 'include/third_party/hccl/inc'),
            os.path.join(self._torch_npu_path, 'include/third_party/acl/inc'),
            # Then project common headers
            os.path.join(self._package_path, 'common/inc')
        ]
        return paths


# Instantiate the builder and load module
elastic_buffer_op_builder = ElasticBufferOpBuilder()
op_module = elastic_buffer_op_builder.load()  # Compiles/loads the .so file