# CausalConv1dDecode Implementation Summary

This directory contains a decode-only AscendC/A5-style implementation of `CausalConv1dDecode`.

## Implemented Files

- `CMakeLists.txt`
- `README.md`
- `README.zh.md`
- `docs/aclnnCausalConv1dDecode.md`
- `docs/aclnnCausalConv1dDecode.zh.md`
- `op_graph/causal_conv1d_decode_proto.h`
- `op_host/causal_conv1d_decode_def.cpp`
- `op_host/causal_conv1d_decode_tiling.cpp`
- `op_host/causal_conv1d_decode_tiling.h`
- `op_api/causal_conv1d_decode.cpp`
- `op_api/causal_conv1d_decode.h`
- `op_api/aclnn_causal_conv1d_decode.cpp`
- `op_api/aclnn_causal_conv1d_decode.h`
- `op_kernel/causal_conv1d_decode.cpp`
- `op_kernel/causal_conv1d_decode_tiling.h`
- `examples/test_aclnn_causal_conv1d_decode.cpp`
- `tests/reference_causal_conv1d_decode.py`
- `tests/ut/op_host/test_causal_conv1d_decode_tiling.cpp`

## Notes

- The op is BF16-only in v1.
- Width is fixed at 4.
- Decode chunk `seq_len` is limited to `[1,4]`.
- The public interface intentionally omits prefill and APC fields.
- The kernel follows the repository AICore entry style with `workspace` and `tiling` parameters.
- Ascend compilation was not run in the GPU server environment.

