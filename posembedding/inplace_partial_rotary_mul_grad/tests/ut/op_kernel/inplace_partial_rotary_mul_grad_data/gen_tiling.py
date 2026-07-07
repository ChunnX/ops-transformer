#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Generate tiling data binary for InplacePartialRotaryMulGrad kernel UT.

Usage: python3 gen_tiling.py <tiling_key> [slice_start] [slice_end]

The tiling data uses InplacePartialRotaryMulGradRegbaseTilingData struct
(19 int64_t fields) or InplacePartialRotaryMulGradRegbaseTilingDataAb struct
(19 int64_t fields).
"""

import sys
import numpy as np

# Number of int64_t fields in the tiling data struct (both base and AB have ~19 fields)
TILING_DATA_SIZE = 19

# Pre-calibrated tiling data for specific test cases
# To calibrate: run OpHost UT, copy tiling data from log output
tiling_params = {
    # Empty slice placeholder: kernel returns immediately via TILING_KEY_EMPTY=403
    (403, 0, 0): [0] * TILING_DATA_SIZE,
}


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 gen_tiling.py <tiling_key> [slice_start] [slice_end]",
            file=sys.stderr,
        )
        sys.exit(1)

    tiling_key = int(sys.argv[1])
    slice_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    slice_end = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    key = (tiling_key, slice_start, slice_end)

    if key in tiling_params and tiling_params[key] is not None:
        params = tiling_params[key]
    else:
        # Fallback: zero-filled tiling data (placeholder for calibration)
        print(
            f"Warning: no calibrated tiling data for (key={tiling_key}, "
            f"slice=[{slice_start},{slice_end}]), using zero-filled placeholder",
            file=sys.stderr,
        )
        params = [0] * TILING_DATA_SIZE
        # Set sliceLength=0 to trigger empty path and avoid kernel crash with all-zero tiling
        # params[16] = 0  # sliceLength, already 0

    base_params = np.array(params, dtype=np.int64)
    with open("tiling.bin", "wb") as f:
        base_params.tofile(f)

    print(
        f"Written tiling.bin: tiling_key={tiling_key}, "
        f"slice=[{slice_start},{slice_end}], size={len(params)} int64"
    )


if __name__ == "__main__":
    main()
