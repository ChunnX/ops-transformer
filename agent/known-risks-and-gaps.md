# 风险和缺口

## 初始静态分析范围限制

初始仓库梳理阶段仅做静态分析，当时不能证明以下事项：

- 未执行 `bash build.sh`。
- 未安装 custom package。
- 未执行 example。
- 未执行 C++ UT、pytest 或 ST。
- 未执行 `msprof`、simulator 或 `cannsim`。

因此，本文档中的“入口”“流程”“支持线索”不能等价为“已验证可用”。

后续针对两个重点算子已补充动态验证：

- 已构建并安装 `custom_transformer` package。
- 已执行 `scripts/st_accuracy` 两个重点 attention 算子的设备侧 ST，结果为 18 passed、0 failed、0 error、0 unsupported。
- 该动态结论只覆盖 `attention/lightning_indexer` 和 `attention/recurrent_gated_delta_rule`，不能外推为全仓库 ST 已通过。

## 容易误判的位置

### 公开 op list 不等于源码全集

`docs/zh/op_list.md` 是公开算子清单，但不是所有有源码的目录都在其中。后续选择优化对象时，应以 `agent/ascendc-operators.md` 的源码判定为主，再结合公开接口判断用户可见性。

### CI list 不等于最终支持矩阵

`scripts/ci/ascend910b/ops_transformer_operator_list.yaml` 是重要的 910B 支持信号，但仍不能替代真实 build 和运行。部分有源码的目录不在 910B CI list 中，可能是未支持、未纳入 CI 或只支持其他 SoC。

### `op_host/config` 不是唯一 SoC 信号

有些算子缺少显式 `op_host/config` 或配置不完整，但仍可能通过通用路径构建。判断 SoC 支持要综合：

- `op_host/config`
- CI operator list
- CMake 条件
- `op_kernel/arch*`
- 实际 build 输出

### example 不一定做精度对比

`build.sh --run_example` 会编译并执行 example，但每个 example 是否包含 golden 比较取决于该算子的样例实现。不能把 example 成功等同于精度验证充分。

### 没有统一性能 benchmark

当前主线性能采集依赖 example 二进制加 `msprof`。不同算子 example 的输入规模不同，不能跨算子直接比较，也不能代表业务真实 shape。

### MC2 算子强依赖运行环境

`mc2/` 下很多算子涉及 HCCL、多卡、rank、通信拓扑。单卡环境、模拟环境或未正确配置 HCCL 时，example 和 profiling 结果可能不具备代表性。

### custom package 可能未被实际加载

运行 example 或 profiling 时，如果 `ASCEND_CUSTOM_OPP_PATH`、`ASCEND_OPP_PATH`、`LD_LIBRARY_PATH` 配置不正确，可能加载系统原始算子而不是当前修改后的 custom package。后续优化必须确认加载路径。

针对 `scripts/st_accuracy`，真实设备侧 `run` 已内置 selected case 对应算子的 `--pkg` build/install gate，可作为 ST 精度的 package/load 前置保障；`list`、`coverage`、`taxonomy` 和 `run --no-execute` 不触发 package/install，只能作为静态链路证据。example、profiling 或手工二进制仍需独立确认实际加载路径和产物 hash。

### 编译目录会被清理

`build.sh` 的常规 clean 会删除 `build/` 和 `output/`。不要把手动记录、profiling 对比表或临时 patch 放在这些目录。

## 后续开始真实优化前的最小核查

建议每个目标算子先完成：

1. `bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom`
2. 安装或挂载 custom package。
3. `bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b`
4. 至少执行对应层级 UT。
5. 用 `msprof op` 获取 baseline。
6. 记录环境、输入 shape、构建参数和 profiling 输出目录。

没有完成 baseline 前，不建议开始大范围重构。
