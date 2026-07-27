# 跨目录引用追踪报告

- 算子目录：`ffn`
- 项目根目录：`Desktop/code/ops-transformertorch2/ffn`
- 算子目录文件数：22
- 追踪到外部文件数：0

## 代码分析清单（完整引用列表）

模型在后续阶段应读取以下文件。外部文件已标注引用链和实际引用的函数名，
读取外部文件时只需精准读取被引用的函数体，无需全文读取。

| 文件路径 | 分析原因 |
|---------|---------|
| `ffn/op_host/op_api/aclnn_ffn.cpp` | aclnn 接口实现文件 |
| `ffn/op_host/op_api/aclnn_ffn.h` | aclnn 接口实现文件 |
| `ffn/op_host/op_api/aclnn_ffn_v2.h` | aclnn 接口实现文件 |
| `ffn/op_host/op_api/aclnn_ffn_v3.h` | aclnn 接口实现文件 |
| `ffn/op_host/ffn_def.cpp` | 算子注册定义（AddConfig） |
| `ffn/op_host/ffn_tiling.cpp` | tiling 实现 |
| `ffn/op_kernel/ffn.cpp` | kernel 实现 |
| `ffn/op_kernel/ffn_glu.cpp` | kernel 实现 |
| *(算子目录其余 14 个文件)* | 算子目录内部文件 |

## 未解析项

以下引用未能自动解析，模型分析时可跳过：

- #include "fallback/fallback_comm.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_graph/fallback_ffn.cpp) — file not found
- #include "fallback/fallback.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_graph/fallback_ffn.cpp) — file not found
- #include "log/log.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_infershape.cpp) — file not found
- #include "op_host/tiling_templates_registry.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "err/ops_err.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "log/log.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "op_host/tiling_base.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "platform/platform_info.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.cpp) — file not found
- #include "tiling/tiling_api.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/ffn_tiling.h) — file not found
- #include "kernel_tiling/kernel_tiling.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_kernel/ffn.h) — file not found
- #include "lib/matmul_intf.h" (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_kernel/ffn.h) — file not found
- OP_TYPE_REGISTER(FFN) (from m:/Users/c60107599/Desktop/code/ops-transformertorch2/ffn/ffn/op_host/op_api/ffn.cpp) — no operator directory found

