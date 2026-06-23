# 仓库结构静态分析

## 顶层目录

| 路径 | 作用 |
| --- | --- |
| `README.md` | 项目总说明，指向算子列表、安装、调用、开发和调试文档。 |
| `QUICKSTART.md` | 快速安装、编译、运行 example 和 msprof 示例。 |
| `build.sh` | 编译、UT、example 的统一 shell 入口。 |
| `CMakeLists.txt` | 顶层 CMake 入口。 |
| `version.cmake` | package 版本和 CANN 依赖声明。 |
| `cmake/` | 构建函数、opbuild、custom package、UT、symbol 等 CMake 逻辑。 |
| `scripts/` | 生成脚本、CI 脚本、opgen 模板、编译耗时分析等。 |
| `docs/` | 中文文档，包含安装、编译、调用、开发、调试、算子列表。 |
| `common/` | 公共构建和公共 C++/kernel 逻辑。 |
| `attention/` | attention 类主线算子。 |
| `ffn/` | FFN 和 Swin FFN 类主线算子。 |
| `gmm/` | grouped matmul 类主线算子。 |
| `mc2/` | 通信融合和 MC2 类主线算子。 |
| `moe/` | MoE 路由、permute、topk 等主线算子。 |
| `posembedding/` | rotary/rope/kvcache 位置编码类主线算子。 |
| `experimental/` | 实验性算子、专项 demo、benchmark。 |
| `examples/` | 独立样例算子和快速开发模板。 |
| `tests/` | 通用 UT 框架、测试配置、测试依赖。 |
| `torch_extension/` | PyTorch extension 相关目录。 |
| `agent/` | 本轮静态分析生成的 Codex 辅助文档。 |

## 主线算子目录形态

主线算子通常形如：

```text
<domain>/<op_name>/
  CMakeLists.txt
  op_host/
  op_host_aclnn/
  op_graph/
  op_kernel/
  examples/
  tests/
```

不是所有算子都拥有全部子目录。常见组合：

- 只有 `op_graph`，没有 `op_api`：偏 graph 调用路径。
- 有 `op_host_aclnn`：通常支持 eager/aclnn API。
- 有 `tests/ut` 或 `tests/utest`：具备 C++ UT 资产。
- 有 `tests/st`：具备 ST case 或 executor 资产。
- 有 `examples`：可通过 `build.sh --run_example` 查找并编译样例。

## 算子域职责

### `attention/`

覆盖 flash attention、prompt/incremental/fused infer attention、sparse/block sparse attention、MLA、NSA、lightning indexer、KV cache 操作等。源码复杂度高，后续优化通常涉及 tiling、GM/L1/L0/UB 搬运、softmax、matmul、quant、cache 更新等。

### `ffn/`

覆盖 `ffn`、worker batching 和 Swin transformer 相关融合。后续优化多关注 matmul/activation/quant/normalize 组合和 batch worker 调度。

### `gmm/`

覆盖 grouped matmul、routing、swiglu quant 等。后续优化多关注 group 分布、expert batch、tiling、workspace 和 quant/dequant 融合。

### `mc2/`

覆盖 matmul 与 all-reduce/all-gather/reduce-scatter/all-to-all 的融合，以及 MoE distribute dispatch/combine。后续优化同时受 kernel 计算、通信拓扑、HCCL、workspace 和多卡运行环境影响。

### `moe/`

覆盖 MoE routing、topk、token permute/unpermute、finalize routing 等。后续优化多关注排序/分桶、token layout、expert 路由、workspace 和小 shape 性能。

### `posembedding/`

覆盖 rotary position embedding、rope、kvcache、rms norm 融合等。后续优化多关注向量化、cache 写入、dtype/format、融合和访存连续性。

## 构建脚本和文档入口

优先查看：

- `build.sh`
- `CMakeLists.txt`
- `cmake/func.cmake`
- `cmake/opbuild.cmake`
- `cmake/custom_build.cmake`
- `cmake/ut.cmake`
- `cmake/func_utest.cmake`
- `docs/zh/context/build.md`
- `docs/zh/invocation/quick_op_invocation.md`
- `docs/zh/debug/op_debug_prof.md`
- `docs/zh/develop/aicore_develop_guide.md`

## CI 和支持矩阵入口

CI 算子列表：

- `scripts/ci/ascend910b/ops_transformer_operator_list.yaml`
- `scripts/ci/ascend910_93/ops_transformer_operator_list.yaml`
- `scripts/ci/ascend950/ops_transformer_operator_list.yaml`

测试映射：

- `tests/test_config.yaml`

这些文件有助于判断某个算子是否在特定 SoC 的日常 CI 中被覆盖，但不能替代真实编译和运行验证。
