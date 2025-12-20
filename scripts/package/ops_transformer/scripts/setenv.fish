#!/usr/bin/fish
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set REAL_SHELL_PATH (realpath (command -v $argv[0]))
set MULTI_VERSION $argv[1]
set CANN_PATH (cd (dirname $REAL_SHELL_PATH)/../../../../ && pwd)

if test -d "$CANN_PATH/opp"
    set INSATLL_PATH `cd $(dirname $REAL_SHELL_PATH)/../../../../../ && pwd`
    set _ASCEND_OPP_PATH "${CANN_PATH}/opp"
    if test "$MULTI_VERSION" = "multi_version"
        set _ASCEND_OPP_PATH "${INSATLL_PATH}/latest/opp"
    end
end

set -x ASCEND_OPP_PATH $_ASCEND_OPP_PATH

pylib_path="${_ASCEND_OPP_PATH}/python/site-packages/"
if test -d ${pylib_path}
    set -x PYTHONPATH $PYTHONPATH:$pylib_path
end

