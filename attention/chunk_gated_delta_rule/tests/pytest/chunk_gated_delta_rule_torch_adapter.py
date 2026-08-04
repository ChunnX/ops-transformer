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
from pathlib import Path

import torch
import torch_npu
from torch.utils.cpp_extension import load


_EXT_MODULE = None


def _repo_root():
    return Path(__file__).resolve().parents[4]


def _ensure_custom_opp_path(ascend_home):
    custom_vendor = Path(ascend_home) / "opp" / "vendors" / "custom_transformer"
    custom_opapi_lib = custom_vendor / "op_api" / "lib"
    if custom_opapi_lib.is_dir():
        ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if str(custom_opapi_lib) not in ld_paths:
            os.environ["LD_LIBRARY_PATH"] = f"{custom_opapi_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    if custom_vendor.is_dir():
        custom_opp_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        if str(custom_vendor) not in custom_opp_paths:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{custom_vendor}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"


def _load_ext():
    global _EXT_MODULE
    if _EXT_MODULE is not None:
        return _EXT_MODULE

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
    _ensure_custom_opp_path(ascend_home)

    torch_npu_dir = Path(torch_npu.__file__).resolve().parent
    repo_root = _repo_root()
    source = Path(__file__).resolve().with_name("chunk_gated_delta_rule_torch_api.cpp")

    extra_cflags = [
        "-fstack-protector-all",
        "-Wl,-z,relro,-z,now,-z,noexecstack",
        "-fPIC",
        "-pie",
        "-s",
        "-fvisibility=hidden",
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        "-w",
        f"-D_GLIBCXX_USE_CXX11_ABI={1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0}",
    ]
    extra_include_paths = [
        str(torch_npu_dir / "include"),
        str(torch_npu_dir / "include" / "third_party" / "hccl" / "inc"),
        str(torch_npu_dir / "include" / "third_party" / "acl" / "inc"),
        str(Path(ascend_home) / "include"),
        str(repo_root / "torch_extension" / "cann_ops_transformer" / "common" / "inc"),
    ]
    extra_ldflags = [
        f"-L{Path(ascend_home) / 'lib64'}",
        "-lascendcl",
        f"-L{torch_npu_dir / 'lib'}",
        "-ltorch_npu",
    ]

    _EXT_MODULE = load(
        name="chunk_gated_delta_rule_pytest_ext",
        sources=[str(source)],
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        verbose=os.environ.get("CGDR_EXT_VERBOSE", "0").lower() in ("1", "true", "yes"),
    )
    return _EXT_MODULE


def npu_chunk_gated_delta_rule(
    query,
    key,
    value,
    *,
    beta,
    initial_state,
    actual_seq_lengths,
    scale=1.0,
    g=None,
):
    if hasattr(torch_npu, "npu_chunk_gated_delta_rule"):
        return torch_npu.npu_chunk_gated_delta_rule(
            query,
            key,
            value,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            scale=scale,
            g=g,
        )

    return _load_ext().npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        float(scale),
        g,
    )
