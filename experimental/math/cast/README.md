# Cast (ascend310p, experimental)

Elementwise Cast operator for Atlas inference SoC (ascend310p, AICORE 200 / dav_m200).

- Covered: task dtype matrix from BOOL/FP16/FP32/INT8/UINT8/INT16/INT32/INT64/BF16 input to
  BOOL/FP16/FP32/INT8/UINT8/INT16/INT32/INT64 output. BF16 is input-only per task spec.
- Bool output follows the TBE-compatible nonzero-to-true rule.
- `examples/test_aclnn_cast.cpp` contains multiple aclnn two-stage invocation cases for common 310P simulator paths.

Build: `bash build.sh --pkg --experimental --soc=ascend310p --ops=cast`
