# op_kernel

This directory contains the AscendC D=128 fast-path controller and basic block
for tiling key `350128` with `KERNEL_TYPE_MIX_AIC_1_2`.

`qkv_rms_norm_rope_cache_with_k_scale.cpp` is only the kernel entry and tiling
key dispatch shell. The D=128 fast-path controller, basic block, range helpers,
and tensor bundle live in `qkv_rms_norm_rope_cache_with_k_scale_d128_fast.h`.

The controller reads the kernel-side POD tiling data, maps AIC `GetBlockIdx()`
and AIV `GetBlockIdx() / 2` to the same cube group, computes a contiguous
balanced token range for that group, and loops over local tiles in controller
scope. It dispatches each tile through `Init`, `ComputeTile`, and `End`
lifecycle hooks.

The basic block now derives per-tile tail parameters, including `M_tile` row
size, and computes AIV0/AIV1 local token/row slices using the actual tile token
count. Buffer binding, synchronization, and real compute are left to later
tasks.
