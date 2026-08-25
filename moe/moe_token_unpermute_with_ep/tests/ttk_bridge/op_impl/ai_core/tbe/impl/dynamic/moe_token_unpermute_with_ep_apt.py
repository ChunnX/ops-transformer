"""Signature bridge for TTK's released AscendC-binary tiling path.

TTK uses the dynamic Python function signature to map descriptors and
attributes before calling the installed host tiling registry.  The released
AscendC package contains the host tiling library and `_apt` device binaries,
but intentionally has no TBE implementation module.  This bridge is therefore
signature-only: it is never used to compute, compile, or validate the result.
"""


def moe_token_unpermute_with_ep(
    permuted_tokens,
    sorted_indices,
    probs,
    unpermuted_tokens,
    num_topk=1,
    range=(0, 0),
    padded_mode=False,
    restore_shape=(),
    kernel_name="moe_token_unpermute_with_ep",
):
    del (
        permuted_tokens,
        sorted_indices,
        probs,
        unpermuted_tokens,
        num_topk,
        range,
        padded_mode,
        restore_shape,
        kernel_name,
    )
    raise RuntimeError("TTK signature bridge must not be invoked as a compute implementation")
