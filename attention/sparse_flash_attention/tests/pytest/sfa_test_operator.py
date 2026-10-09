# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Optional repository ACLNN adapter for TorchNPU versions with old PA LSE shapes."""

import os
from functools import lru_cache
from pathlib import Path

import torch_npu


@lru_cache(maxsize=1)
def get_sfa_operator():
    extension_dir = os.environ.get("SFA_TEST_EXTENSION_DIR")
    if not extension_dir:
        return torch_npu.npu_sparse_flash_attention

    from torch.utils.cpp_extension import load

    root = Path(extension_dir)
    npu_root = Path(torch_npu.__file__).parent
    cann_root = Path(os.environ["ASCEND_HOME_PATH"])
    module = load(
        name="sfa_arch35_repo_adapter",
        sources=[
            str(root / "sparse_flash_attention.cpp"),
            str(Path(__file__).with_name("sfa_test_hash.cpp")),
        ],
        extra_include_paths=[
            str(cann_root / "include"),
            str(cann_root / "include/aclnnop"),
            str(npu_root / "include"),
            str(npu_root / "include/third_party/hccl/inc"),
            str(npu_root / "include/third_party/op-plugin"),
            str(root),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[
            f"-L{cann_root}/lib64",
            "-lascendcl",
            f"-L{npu_root}/lib",
            "-ltorch_npu",
        ],
        verbose=True,
    )

    def run(q, k, v, indices, scale, **kwargs):
        return module.sparse_flash_attention(
            q,
            k,
            v,
            indices,
            scale,
            kwargs.get("block_table"),
            kwargs.get("actual_seq_lengths_query"),
            kwargs.get("actual_seq_lengths_kv"),
            kwargs.get("query_rope"),
            kwargs.get("key_rope"),
            kwargs.get("sinks"),
            kwargs.get("sparse_block_size", 1),
            kwargs.get("layout_query", "TND"),
            kwargs.get("layout_kv", "PA_BSND"),
            kwargs.get("sparse_mode", 0),
            kwargs.get("pre_tokens", 9223372036854775807),
            kwargs.get("next_tokens", 9223372036854775807),
            kwargs.get("attention_mode", 2),
            kwargs.get("return_softmax_lse", False),
        )

    return run
