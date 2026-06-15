# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


__description__ = "NpuOpsTransformer"
__version__ = "1.0.0"
__package_name__ = 'npu_ops_transformer'

import os
import sys
from pathlib import Path
import setuptools


MODULE_FILE_MAP = {
    'mc2': [
        'ops/comm_context.py',
        'ops/deep_ep.py',
        'ops/lightning_indexer_v2.py',
        'ops/lightning_indexer_v2_metadata.py',
        'ops/moe_distribute_combine.py',
        'ops/moe_distribute_dispatch.py',
        'ops/quant_lightning_indexer_v2.py',
        'ops/quant_lightning_indexer_v2_metadata.py',
        'ops/npu_sparse_lightning_indexer_kl_loss_grad.py',
        'ops/npu_sparse_lightning_indexer_kl_loss_grad_metadata.py',
        'ops/csrc/comm_context.cpp',
        'ops/csrc/lightning_indexer_v2.cpp',
        'ops/csrc/lightning_indexer_v2_metadata.cpp',
        'ops/csrc/moe_distribute_combine.cpp',
        'ops/csrc/moe_distribute_dispatch.cpp',
        'ops/csrc/quant_lightning_indexer_v2.cpp',
        'ops/csrc/quant_lightning_indexer_v2_metadata.cpp',
        'ops/csrc/npu_sparse_lightning_indexer_kl_loss_grad.cpp',
        'ops/csrc/npu_sparse_lightning_indexer_kl_loss_grad_metadata.cpp',
        'ops/graph_convert/graph_convert_lightning_indexer_v2_metadata.py',
        'ops/graph_convert/graph_convert_moe_distribute_combine.py',
        'ops/graph_convert/graph_convert_moe_distribute_dispatch.py',
        'ops/graph_convert/graph_convert_quant_lightning_indexer_v2_metadata.py',
        'doc/get_low_latency_ccl_buffer_size.md',
        'doc/npu_low_latency_combine.md',
        'doc/npu_low_latency_dispatch.md',
    ],
    'attention': [
        'ops/flash_attn.py',
        'ops/flash_attn_metadata.py',
        'ops/sparse_flash_mla_grad.py',
        'ops/sparse_flash_mla_grad_metadata.py',
        'ops/mixed_quant_sparse_flash_mla.py',
        'ops/csrc/flash_attn.cpp',
        'ops/csrc/flash_attn_metadata.cpp',
        'ops/csrc/sparse_flash_mla_grad.cpp',
        'ops/csrc/sparse_flash_mla_grad_metadata.cpp',
        'ops/csrc/mixed_quant_sparse_flash_mla.cpp',
        'ops/graph_convert/graph_convert_flash_attn.py',
        'doc/npu_flash_attn.md',
    ],
    'moe': [
        'ops/mega_moe.py',
        'ops/csrc/mega_moe.cpp',
        'ops/graph_convert/graph_convert_mega_moe.py',
        'doc/mega_moe.md',
        'doc/npu_get_mega_moe_ccl_buffer_size.md',
    ],
    'mhc': [
        'ops/mhc_post.py',
        'ops/mhc_post_backward.py',
        'ops/mhc_pre_sinkhorn.py',
        'ops/mhc_pre_sinkhorn_backward.py',
        'ops/csrc/mhc_post.cpp',
        'ops/csrc/mhc_post_backward.cpp',
        'ops/csrc/mhc_pre_sinkhorn.cpp',
        'ops/csrc/mhc_pre_sinkhorn_backward.cpp',
        'doc/mhc_post.md',
        'doc/mhc_pre_sinkhorn.md',
    ],
}

COMMON_FILES = [
    '__init__.py',
    'common/inc/aclnn_common.h',
    'common/inc/hccl_common.h',
    'op_builder/builder.py',
    'ops/__init__.py',
]

MODULE_IMPORTS = {
    'mc2': [
        'from .sparse_flash_mla_grad import npu_sparse_flash_mla_grad',
        'from .sparse_flash_mla_grad_metadata import npu_sparse_flash_mla_grad_metadata',
        'from .moe_distribute_dispatch import npu_moe_distribute_dispatch',
        'from .moe_distribute_combine import npu_moe_distribute_combine',
        'from .deep_ep import MoeDistributeBuffer',
        'from .graph_convert.graph_convert_moe_distribute_dispatch import converter_moe_distribute_dispatch',
        'from .graph_convert.graph_convert_moe_distribute_combine import convert_npu_moe_distribute_combine',
        'from .lightning_indexer_v2_metadata import npu_lightning_indexer_v2_metadata',
        'from .quant_lightning_indexer_v2_metadata import npu_quant_lightning_indexer_v2_metadata',
        'from .graph_convert.graph_convert_lightning_indexer_v2_metadata import convert_npu_lightning_indexer_v2_metadata',
        'from .graph_convert.graph_convert_quant_lightning_indexer_v2_metadata import convert_npu_quant_lightning_indexer_v2_metadata',
        'from .npu_sparse_lightning_indexer_kl_loss_grad import npu_sparse_lightning_indexer_kl_loss_grad',
        'from .npu_sparse_lightning_indexer_kl_loss_grad_metadata import npu_sparse_lightning_indexer_kl_loss_grad_metadata',
        'from .lightning_indexer_v2 import npu_lightning_indexer_v2',
        'from .quant_lightning_indexer_v2 import npu_quant_lightning_indexer_v2',
    ],
    'attention': [
        'from .flash_attn import npu_flash_attn',
        'from .graph_convert.graph_convert_flash_attn import convert_npu_flash_attn',
        'from .flash_attn_metadata import npu_flash_attn_metadata',
        'from .mixed_quant_sparse_flash_mla import npu_mixed_quant_sparse_flash_mla',
    ],
    'moe': [
        'from .mega_moe import get_symm_buffer_for_mega_moe, mega_moe, npu_get_mega_moe_ccl_buffer_size',
        'from .graph_convert.graph_convert_mega_moe import convert_npu_mega_moe',
    ],
    'mhc': [
        'from .mhc_post import mhc_post',
        'from .mhc_pre_sinkhorn import mhc_pre_sinkhorn',
    ],
}


def get_module_files(module_names):
    files = COMMON_FILES.copy()
    for module in module_names:
        if module in MODULE_FILE_MAP:
            files.extend(MODULE_FILE_MAP[module])
    return files


def generate_ops_init(module_names):
    lines = [
        '# -----------------------------------------------------------------------------------------------------------',
        '# Copyright (c) 2025 Huawei Technologies Co., Ltd.',
        '# This program is free software, you can redistribute it and/or modify it under the terms and conditions of',
        '# CANN Open Software License Agreement Version 2.0 (the "License").',
        '# Please refer to the License for details. You can not use this file except in compliance with the License.',
        '# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,',
        '# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.',
        '# See LICENSE in the root of the software repository for the full text of the License.',
        '# -----------------------------------------------------------------------------------------------------------',
    ]
    for module in module_names:
        if module in MODULE_IMPORTS:
            lines.extend(MODULE_IMPORTS[module])
    return '\n'.join(lines)


def package_files(directory, module_names=None):
    paths = []
    for path, _, filenames in os.walk(directory):
        rel_path = os.path.relpath(path, directory)
        for filename in filenames:
            full_path = os.path.join(path, filename)
            rel_file = os.path.join(rel_path, filename) if rel_path != '.' else filename
            if module_names:
                if rel_file in get_module_files(module_names):
                    paths.append(full_path)
            else:
                paths.append(full_path)
    return paths


src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), __package_name__)

module_env = os.environ.get('ASCEND_MODULE_NAME', '')
module_names = []
if module_env and module_env.lower() not in ('all', ''):
    module_names = [m.strip().lower() for m in module_env.split(',')]

if module_names:
    ops_init_path = os.path.join(src_path, 'ops', '__init__.py')
    init_content = generate_ops_init(module_names)
    with open(ops_init_path, 'w', encoding='utf-8') as f:
        f.write(init_content)

setuptools.setup(
    name=__package_name__,
    version=__version__,
    author="CANN",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_package_data=True,
    package_data={'': package_files(src_path, module_names)},
    zip_safe=False
)
