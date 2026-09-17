#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Preserve the installed OPP selection for both pytest and TTK."""

import os
from pathlib import Path


def configure_custom_opp_paths():
    """Use only explicitly configured vendors, in ASCEND_CUSTOM_OPP_PATH order.

    Never select CPack staging directories or another developer's installation.
    Without a custom path, CANN uses the installed ASCEND_OPP_PATH configuration.
    Call before loading the torch extension, which caches the custom library paths.
    """
    roots = [
        Path(part)
        for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if part
    ]
    library_dirs = []
    for root in roots:
        library_dir = str(root / "op_api" / "lib")
        if Path(library_dir).is_dir() and library_dir not in library_dirs:
            library_dirs.append(library_dir)
    if not library_dirs:
        return
    existing = [
        part
        for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if part and part not in library_dirs
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs + existing)
