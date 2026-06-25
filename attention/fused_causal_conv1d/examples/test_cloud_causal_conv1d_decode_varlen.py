#!/usr/bin/env python3
"""Run the 2D varlen decode smoke case through cloud_ops_trubo.

Expected dispatch in ops-transformer:
  decode-only op, x.ndim == 2 with query_start_loc -> BH 2D varlen decode path.

Shape convention used by this operator:
  x:           [total_tokens, dim]
  conv_state:  [num_cache_slots, state_len, dim]
  weight:      [kernel_width, dim]
"""

import inspect
import os

import torch
import torch_npu  # noqa: F401

import cloud_ops_trubo


def main():
    device = "npu"
    dtype = torch.bfloat16

    total_tokens = 1
    dim = 3072
    num_cache_slots = 1474
    kernel_width = 4
    state_len = kernel_width - 1

    x = torch.ones((total_tokens, dim), dtype=dtype, device=device)
    conv_state = torch.zeros((num_cache_slots, state_len, dim), dtype=dtype, device=device)
    weight = torch.ones((kernel_width, dim), dtype=dtype, device=device)

    bias = None
    # Semantically this single-token decode case does not need num_accept_tokens.
    # If the Python wrapper requires a tensor, run with USE_ACCEPT_TOKENS=1.
    use_accept_tokens = os.getenv("USE_ACCEPT_TOKENS", "0") == "1"
    num_accept_tokens = (
        torch.tensor([1], dtype=torch.int32, device=device) if use_accept_tokens else None
    )
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([0], dtype=torch.int32, device=device)
    activation_mode = 1

    try:
        print("cloud_causal_conv1d_fn signature:")
        print(inspect.signature(cloud_ops_trubo.cloud_causal_conv1d_fn))
    except (TypeError, ValueError):
        print("cloud_causal_conv1d_fn signature: <not available>")
    print("case: x=(1,3072), conv_state=(1474,3,3072), weight=(4,3072), qsl=[0,1], cache_indices=[0]")

    common_kwargs = {
        "x": x,
        "weight": weight,
        "bias": bias,
        "query_start_loc": query_start_loc,
        "cache_indices": cache_indices,
        "activation_mode": activation_mode,
    }
    try:
        y = cloud_ops_trubo.cloud_causal_conv1d_fn(
            conv_state=conv_state,
            num_accept_tokens=num_accept_tokens,
            **common_kwargs,
        )
    except TypeError as first_err:
        print("first call failed, retry with conv_states/num_accepted_tokens names:", first_err)
        y = cloud_ops_trubo.cloud_causal_conv1d_fn(
            conv_states=conv_state,
            num_accepted_tokens=num_accept_tokens,
            **common_kwargs,
        )

    if isinstance(y, tuple):
        print("return tuple shapes:", [None if t is None else tuple(t.shape) for t in y])
    else:
        print("return shape:", tuple(y.shape))

    torch_npu.npu.synchronize()
    print("done")


if __name__ == "__main__":
    main()
