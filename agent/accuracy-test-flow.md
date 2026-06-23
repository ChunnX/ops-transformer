# 精度测试环节静态分析

## 结论概览

本仓库没有单一的“精度测试总入口”。静态可见的验证体系由四类入口组成：

1. `build.sh --run_example`：构建并运行算子 example，常用于本地冒烟和接口调用验证。
2. `build.sh -u ...`：构建并运行 C++ UT，覆盖 `op_host`、`op_api`、`op_graph`、`op_kernel` 等层面。
3. pytest：部分 attention 算子提供 Python 精度用例，通常用 CPU 或 PyTorch 参考实现生成 golden，再调用 NPU 实现对比。
4. ST 资产：部分算子下有 `tests/st/` 的 json case 和 executor 脚本；当前新增 `scripts/st_accuracy` 局部 runner，优先覆盖 `lightning_indexer` 与 `recurrent_gated_delta_rule`。

本轮只做静态分析，没有执行任何测试命令。

## example 验证流程

入口：

```bash
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b
bash build.sh --run_example <op_name> graph cust --vendor_name=custom --soc=ascend910b
```

模块：`build.sh` 中 `build_example()`

输入：

- 算子名 `<op_name>`。
- 执行模式：`eager` 或 `graph`。
- package 模式：空值表示标准包，`cust` 表示 custom package。
- `--vendor_name`，custom package 模式下用于定位 custom OPP 路径。
- `ASCEND_OPP_PATH`、`ASCEND_CUSTOM_OPP_PATH`、CANN include/lib 路径。

处理：

- `eager` 模式查找 `*/<op_name>/examples/test_aclnn_*.cpp`。
- `graph` 模式查找 `*/<op_name>/examples/test_geir_*.cpp`。
- 非 `ascend950` SoC 从常规 `examples/` 目录查找；`ascend950` 优先查找 `examples/arch35/`。
- `eager` 标准包链接 `-lopapi_math -lopapi_transformer -lascendcl -lnnopbase -lc_sec`。
- `eager` custom package 链接 `-lopapi_math -lcust_opapi -lascendcl -lnnopbase -lc_sec`，并加入 custom lib rpath。
- MC2 example 额外链接 `-lpthread -lhccl -lhccl_fwk`。
- `graph` 模式链接 `-lgraph -lge_runner -lgraph_base -lge_compiler`。
- 编译成功后立即执行生成的 example 二进制。

输出：

- `build/test_aclnn_<op_name>` 或 `build/test_geir_<op_name>`。
- 程序 stdout/stderr。
- 如果 example 内部实现了 golden 对比，会输出精度结果；如果没有，则只代表调用链路冒烟成功。

静态判断：

- `run_example` 是最贴近用户调用方式的验证入口。
- 它不是统一精度框架。不同算子的 example 是否比较 golden，需要进入各自 `examples/` 源码确认。

## C++ UT 流程

入口：

```bash
# 全量 UT
bash build.sh -u --soc=ascend910b

# 单算子指定层级 UT
bash build.sh -u --ophost_test --ops=<op_name> --soc=ascend910b
bash build.sh -u --opapi_test --ops=<op_name> --soc=ascend910b
bash build.sh -u --opgraph_test --ops=<op_name> --soc=ascend910b
bash build.sh -u --opkernel_test --ops=<op_name> --soc=ascend910b

# 只编译不执行
bash build.sh -u --noexec --ops=<op_name> --soc=ascend910b
```

模块：

- `build.sh`
- `cmake/ut.cmake`
- `cmake/func_utest.cmake`
- `tests/ut/framework_normal/`
- 各算子 `tests/ut/` 或 `tests/utest/`

输入：

- `--ops` 传入的算子过滤条件。
- `--ophost_test`、`--opapi_test`、`--opgraph_test`、`--opkernel_test`。
- `--noexec`，控制是否只编译不执行。
- `--cov`，控制是否生成覆盖率。
- `--soc`，控制 `op_kernel` UT 使用的 SoC。

处理：

- `build.sh` 中 `set_ut_mode()` 将测试类型映射到 target：
  - `transformer_op_host_ut`
  - `transformer_op_api_ut`
  - `transformer_op_graph_ut`
  - `transformer_op_kernel_ut`
- 未指定测试类型时，默认构建全部 UT target。
- `build_ut()` 会初始化并更新 git submodule，然后进入 CMake 配置和 build。
- 新 UT 框架通过 `cmake/ut.cmake` 收集：
  - `test_*_tiling.cpp`
  - `test_*_infershape.cpp`
  - `test_aclnn_*.cpp`
- `op_kernel` UT 会生成临时 tiling so，并通过 `tests/ut/framework_normal/op_kernel/scripts/gen_tiling_head_file.sh` 生成 tiling head file。
- 老 UT 框架通过 `cmake/func_utest.cmake` 的 `OpsTest_Level*` 函数组装 op_api、op_proto、op_tiling、op_kernel、case static lib 和 launch target。

输出：

- C++ UT 可执行文件。
- GTest 风格输出，例如 `[ PASSED ]`。
- 覆盖率模式下生成覆盖率数据和报告。

## pytest 精度用例

静态发现 pytest 用例主要集中在 attention 域：

- `attention/flash_attention_score/tests/pytest`
- `attention/flash_attention_score_grad/tests/pytest`
- `attention/fused_infer_attention_score/tests/pytest`
- `attention/mla_prolog_v2/tests/pytest`
- `experimental/attention/fused_infer_attention_score/tests/pytest`

典型入口：

```bash
cd attention/fused_infer_attention_score/tests/pytest
python3 -m pytest -rA -s test.py

cd attention/flash_attention_score/tests/pytest
pytest -s
```

输入：

- pytest case 参数。
- PyTorch、`torch_npu`，部分 graph 用例还涉及 `torchair`。
- 已安装或可调用的 CANN/OPP 算子实现。

处理：

- CPU 或 PyTorch 参考实现生成 golden。
- NPU 侧直接调用 torch_npu/aclnn 或 graph 路径。
- 比较输出误差。

输出：

- pytest case 结果。
- 精度误差、失败 case、traceback。

静态判断：

- pytest 是部分复杂 attention 算子的实用精度入口。
- 根目录 `requirements.txt` 和 `tests/requirements.txt` 不完整覆盖 PyTorch、`torch_npu`、`torchair`，真实执行前要补齐环境。

## ST 资产

静态发现部分算子存在 `tests/st/`：

- `attention/attention_update/tests/st/`
- `attention/fused_infer_attention_score/tests/st/`
- `attention/incre_flash_attention/tests/st/`
- `attention/prompt_flash_attention/tests/st/`
- `mc2/all_gather_matmul/tests/st/`
- `mc2/allto_all_matmul/tests/st/`
- `mc2/matmul_all_reduce/tests/st/`
- `posembedding/apply_rotary_pos_emb/tests/st/`

典型文件形态：

- `all_<case>.json`：case 参数和输入输出描述。
- executor python：执行单个算子或 case 集合。

静态判断：

- ST 资产可用于后续构造端到端精度或功能回归。
- 当前 `scripts/st_accuracy` 是两个重点 attention 算子的局部 runner，已为 `lightning_indexer` 和 `recurrent_gated_delta_rule` 接入 native ACLNN 调用与 host golden compare；因此不能把 ST 写成“一条命令全量执行全仓库”的能力。
- 真实设备侧 `scripts/st_accuracy run` 会先对 selected case 的 op 集合强制执行 `build.sh --pkg` 并安装最新 `custom_transformer` 包，再进入 native runner；`coverage`、`taxonomy` 和 `run --no-execute` 不触发 package/install，也不能写成设备侧精度通过。
- 已在当前 910B/aarch64 环境完成两个重点算子的强制 package ST：18 passed、0 failed、0 error、0 unsupported。输出目录为 `build/st_accuracy/force_pkg_device`，`prepare_pkg_ms=104151.404`。
- `RecurrentGatedDeltaRule` 当前默认 ST case 使用唯一 `ssmStateIndices`。重复 state index 会使 `stateRef` 原地更新产生覆盖/顺序依赖，后续如需覆盖该边界，应单独确认官方语义后再建专项 case。

## CI 批量 UT 脚本

入口：`scripts/ci/ut_make.sh`

输入：

- 代码路径参数。
- `tests/test_config.yaml` 中的 changed-file 到 option/example/UT 映射。

处理：

- 遍历 `moe`、`attention`、`gmm`、`ffn`、`posembedding`、`mc2`。
- 根据配置批量执行：

```bash
bash build.sh -u --ops=<op_list> --cov --soc=ascend310p,ascend910b,ascend910_93,ascend950
```

输出：

- `log_ut/` 下日志。
- 覆盖率统计。

静态判断：

- 这是 CI 辅助脚本，适合理解批量测试策略。
- 后续单算子优化时，优先使用 `build.sh -u --ops=<op_name> --soc=ascend910b` 做局部验证。

## 精度验证模块输入输出总表

| 模块 | 输入 | 输出 | 适用场景 |
| --- | --- | --- | --- |
| `build.sh --run_example` | 算子名、模式、package、vendor、SoC | example 二进制和运行输出 | 本地冒烟、接口调用、少量 golden 对比 |
| `ophost UT` | tiling、infer shape 测试源码 | GTest 输出 | 验证 shape/tiling 逻辑 |
| `opapi UT` | aclnn API 测试源码 | GTest 输出 | 验证 API 封装和调用 |
| `opgraph UT` | graph 测试源码 | GTest 输出 | 验证 graph 接入 |
| `opkernel UT` | kernel case、tiling head、tikicpu lib | GTest 输出 | 验证 AscendC kernel 逻辑 |
| pytest | Python case、golden、torch_npu 环境 | pytest 结果和误差 | 复杂 attention 精度对比 |
| ST 资产 | json case、executor | case 运行结果 | 端到端功能和精度补充 |

## 后续优化建议

- 修改 `op_kernel/` 后至少跑对应 `opkernel` UT 和 example。
- 修改 `op_host/*tiling*` 后至少跑 `ophost` UT 和 example。
- 修改 `op_host_aclnn*` 后至少跑 `opapi` UT 和 eager example。
- 修改 `op_graph` 或注册配置后至少跑 `opgraph` UT 和 graph example。
- 对 attention 类复杂算子，优先查找 pytest 是否已有 golden 对比；有的话应纳入优化回归。
