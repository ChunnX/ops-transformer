# CausalConv1dDecode 实现总结

该目录包含一个 decode-only AscendC/A5 风格的 `CausalConv1dDecode` 实现。

## 已实现文件

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

## 说明

- v1 只支持 BF16。
- width 当前仅支持 4；存在 `num_accepted_tokens` 时启用 vLLM-style speculative state layout。
- Decode chunk `seq_len` 限制为 `[1,4]`。
- 输入支持 `[B,D]`、`[B,S,D]`、`[T,D]+query_start_loc` 三种 decode/update 形态。
- public 接口刻意删除了 prefill 和 APC 字段。
- 功能基线对齐 vLLM Ascend PR 8215 decode/update；接口和 AICore/VF 数据流按 ops-transformer A5 风格组织。
- kernel 入口采用仓内常见 AICore 风格，包含 `workspace` 和 `tiling` 参数。
- 当前 GPU 服务器环境没有 Ascend 编译环境，所以尚未跑 Ascend 编译。
