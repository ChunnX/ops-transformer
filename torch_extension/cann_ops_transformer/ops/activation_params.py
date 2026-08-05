# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared activation-parameter parsing for MegaMoe.

Both the eager path (``csrc/mega_moe.cpp``) and the graph path
(``graph_convert/graph_convert_mega_moe.py``) serialize activation parameters into the same
fixed-position ``ListFloat`` attribute consumed by the C++ op. Centralizing the default values,
the per-activation key whitelist and the list ordering here keeps eager and graph behavior
consistent and avoids silent fallbacks when an unknown or misspelled key is supplied.
"""

from typing import Dict, List, Optional

# float32 max (FLT_MAX). Used as the default clamp value: ClampMax(..., FLT_MAX) leaves every
# finite value unchanged, so clamping is effectively disabled when the user does not set "clamp".
CLAMP_DEFAULT = 3.4028234663852886e38
# SwiGLU-OAI sigmoid gain.
ALPHA_DEFAULT = 1.702
# SwiGLU-OAI up-branch bias / SiTUGLU gate-branch divisor.
BETA_DEFAULT = 1.0

# Keys allowed in the activation_params dict, per activation name.
_ALLOWED_ACTIVATION_KEYS = {
    "swiglu": frozenset({"clamp"}),
    "swiglustep": frozenset({"clamp"}),
    "swigluoai": frozenset({"clamp", "alpha", "beta"}),
    "situglu": frozenset({"beta", "linear_beta"}),
}


def normalize_activation_params(
    activation: str,
    activation_clamp: Optional[float] = None,
    activation_params: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Validate activation parameters and serialize them into the fixed-position list expected by
    the C++ op.

    ``activation_clamp`` and ``activation_params`` are mutually exclusive: only one may carry the
    clamp value. When ``activation_clamp`` is supplied it is wrapped as ``{"clamp": activation_clamp}``.
    Unknown keys raise ``ValueError`` instead of being silently ignored, so that a misspelled key
    (e.g. ``linear_beta`` vs ``linearbeta``) fails loudly rather than producing wrong model outputs.
    """
    if activation not in _ALLOWED_ACTIVATION_KEYS:
        raise ValueError(
            f"Unsupported activation: {activation!r}; expected one of {sorted(_ALLOWED_ACTIVATION_KEYS)}."
        )
    allowed = _ALLOWED_ACTIVATION_KEYS[activation]
    activation_params = activation_params or {}

    # activation_clamp is a shortcut for activation_params['clamp']. It is only relevant for
    # activations that use clamp (swiglu/swiglustep/swigluoai); situ ignores clamp, so the
    # shortcut is silently dropped there. When relevant, the two clamp sources are mutually
    # exclusive, and activation_clamp is merged in (coexisting with other keys like alpha/beta).
    if activation_clamp is not None and "clamp" in allowed:
        if "clamp" in activation_params:
            raise ValueError(
                "activation_clamp and activation_params['clamp'] cannot both be specified; "
                "pass the clamp via one of them only."
            )
        activation_params = {**activation_params, "clamp": float(activation_clamp)}

    for key in activation_params:
        if key not in allowed:
            raise ValueError(
                f"Unknown activation param key {key!r} for activation {activation!r}; "
                f"allowed keys: {sorted(allowed)}."
            )

    clamp = float(activation_params.get("clamp", CLAMP_DEFAULT))
    if activation in ("swiglu", "swiglustep"):
        return [clamp]
    if activation == "swigluoai":
        alpha = float(activation_params.get("alpha", ALPHA_DEFAULT))
        beta = float(activation_params.get("beta", BETA_DEFAULT))
        return [clamp, alpha, beta]
    # situglu: [beta] or [beta, linear_beta]; linear_beta defaults to disabled when absent.
    beta = float(activation_params.get("beta", BETA_DEFAULT))
    values = [beta]
    if "linear_beta" in activation_params:
        values.append(float(activation_params["linear_beta"]))
    return values
