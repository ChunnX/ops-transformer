# op_kernel

This directory contains the T01 AscendC controller and basic block skeleton for
tiling key `350128` with `KERNEL_TYPE_MIX_AIC_1_2`.

`qkv_rms_norm_rope_cache_with_k_scale.cpp` is only the kernel entry and tiling
key dispatch shell. The D=128 fast-path controller, basic block, range helpers,
and tensor bundle live in `qkv_rms_norm_rope_cache_with_k_scale_d128_fast.h`.

The skeleton reads the kernel-side POD tiling data, maps AIC `GetBlockIdx()`
and AIV `GetBlockIdx() / 2` to the same cube group, computes the cube-group
tile range, and dispatches empty AIC/AIV rolling branches through `Init`,
`ComputeRange`, and `End` lifecycle hooks.

T02 and later tasks fill in AIV local slices, buffer binding, synchronization,
and real compute.
