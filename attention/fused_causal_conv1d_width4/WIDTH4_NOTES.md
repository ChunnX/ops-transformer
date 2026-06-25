# fused_causal_conv1d_width4 experiment

This directory is copied from `attention/fused_causal_conv1d` and changed as a width-4 experiment.

Changed scope:

- BSH host tiling now requires `kernel_width == 4`.
- README/API docs/examples use `K=4` and `stateLen=K-1`.
- Decode state length wording is adjusted from max token + 1 to max token + 2.
- BH cache-dependent token count is changed to `min(curBatchSeq, kernelSize - 1)`.
- BH cache-dependent prefix uses the generic VF path for W4:
  - `y0 = conv([h0, h1, h2, x0], w)`
  - `y1 = conv([h1, h2, x0, x1], w)`
  - `y2 = conv([h2, x0, x1, x2], w)`
- BH pure-x tokens use a W4 wrapper over the generic NeedState VF:
  - `y[s] = conv([x[s-3], x[s-2], x[s-1], x[s]], w)`
  - the first three x rows are passed as a temporary state window and the current x row is passed as `xAddr`.
- BSH pure-x tokens use the same W4 wrapper instead of the original W3-specialized no-state VF.
- BSH tail selection for the generic W4 VF path uses 64-element VF granularity instead of the original W3 128-element no-state granularity.
- BH `convMode == 1` residual reset copies from the current batch x offset, not from the y UB offset.

Validation note:

- The original K=3-specialized no-state VF functions remain in the copied source but are no longer called from the BH/BSH W4 pure-x paths.
- The copy has been renamed to the independent op/API/kernel names `FusedCausalConv1dWidth4`, `aclnnFusedCausalConv1dWidth4`, and `fused_causal_conv1d_width4`, so it no longer registers the same op type as the original `FusedCausalConv1d` directory.
- This is a correctness-first W4 port. It still needs CANN compile and A5 hardware validation before performance conclusions.
- Keep the original `fused_causal_conv1d` directory unchanged.
