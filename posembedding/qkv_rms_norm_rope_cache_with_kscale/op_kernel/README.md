# op_kernel

This directory contains the T01 AscendC controller and basic block skeleton for
tiling key `350128` with `KERNEL_TYPE_MIX_AIC_1_2`.

The skeleton reads the kernel-side POD tiling data, maps AIC `GetBlockIdx()` and
AIV `GetBlockIdx() / 2` to the same cube group, computes the cube-group tile
range, and dispatches empty AIC/AIV rolling branches through `Init`,
`ComputeRange`, and `End` lifecycle hooks.

T02 and later tasks fill in AIV local slices, buffer binding, synchronization,
and real compute.
