#!/bin/bash
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Builds and installs the bsa_npu python extension.
# Pre-requisite: the block_sparse_attention kernel must already be built and installed
# via the top-level build.sh, e.g.:
#   bash build.sh --make_clean -j96 --pkg --soc=ascend910b --ops=block_sparse_attention
#   ./build/cann-ops-transformer-custom_linux-aarch64.run

if [ $# -ne 1 ]; then
    echo "Error: Exactly one argument required" >&2
    echo "Usage: $0 <PFA_LIB_value>" >&2
    echo "  where PFA_LIB_value is either 'custom' or 'ops_transformer'" >&2
    exit 1
fi

rm -rf build bsa_npu.egg-info
pip uninstall -y bsa_npu

# 'custom' = link against $ASCEND_OPP_PATH/vendors/custom_transformer (built via --ops=block_sparse_attention)
# 'ops_transformer' = link against $ASCEND_TOOLKIT_HOME/ops_transformer (full transformer pkg)
PFA_LIB=$1 pip install . --no-build-isolation
