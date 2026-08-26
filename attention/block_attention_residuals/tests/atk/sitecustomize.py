"""Allow ATK's trusted locally generated datasets under torch.load(weights_only=True)."""

import numpy as np
import torch.serialization


torch.serialization.add_safe_globals(
    [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
        type(np.dtype(np.bool_)),
    ]
)
