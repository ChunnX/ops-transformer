# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import numpy as np


def golden(query, key, scale, clamp_max=4.6052, eps=1.0e-12):
    q = query.astype(np.float32)
    k = key.astype(np.float32)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + eps)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + eps)
    scores = q @ np.swapaxes(k, -1, -2)
    factor = np.exp(np.minimum(np.asarray(scale, np.float32).reshape(-1), clamp_max))
    return scores * factor.reshape(1, query.shape[1], 1, 1)
