# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""生成自测用 rotation / qjl 矩阵 bin（非官方文件；官方有则覆盖本目录）。"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260923)
a = rng.standard_normal((128, 128), dtype=np.float64)
q, _ = np.linalg.qr(a)
rotation = q.astype(np.float32)
qjl = rng.standard_normal((128, 128), dtype=np.float32)
rotation.tofile(OUT / "rotation_matrix.bin")
qjl.tofile(OUT / "qjl_matrix.bin")
print(f"wrote {OUT / 'rotation_matrix.bin'} {rotation.nbytes} bytes")
print(f"wrote {OUT / 'qjl_matrix.bin'} {qjl.nbytes} bytes")
print("NOTE: 自生成；官方 bin 到达后请覆盖并重跑精度对拍。")
