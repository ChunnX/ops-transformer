# 性能测试环节静态分析

## 结论概览

本仓库没有发现类似 `build.sh --perf` 的统一性能测试入口。主线性能采集路径是：

1. 编译并安装 custom package。
2. 用 `build.sh --run_example` 生成并运行目标算子的 example 二进制。
3. 对 example 二进制使用 `msprof op`、`msprof --application` 或 simulator profiling。
4. 从 `OPPROF_*`、ArithmeticUtilization、trace 等产物中分析 kernel 时延和硬件利用率。

因此，当前性能测试更接近“基于样例二进制的 profiling 工作流”，不是完整 benchmark 框架。

## A2/A3/910B 常规 profiling 流程

目标环境：用户当前 Ascend 910B B3 属于 910B 系列，静态上应优先使用 `--soc=ascend910b`。

典型流程：

```bash
# 1. 构建 custom package
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16

# 2. 安装 build_out 下生成的 run package
./build_out/cann-ops-transformer-custom_linux-<arch>.run

# 3. 构建并运行 eager example
bash build.sh --run_example <op_name> eager cust --vendor_name=custom --soc=ascend910b

# 4. 进入生成 example 二进制的目录后 profiling
msprof op ./test_aclnn_<op_name>
```

部分文档也给出 `msprof --application` 形式：

```bash
msprof --application="./test_aclnn_<op_name>"
```

静态判断：

- `msprof op` 更贴近 `docs/zh/debug/op_debug_prof.md` 中的算子 profiling 路径。
- `msprof --application` 在 `QUICKSTART.md` 中用于 add example 演示。
- 实际使用时要以本机 CANN/msprof 版本支持的命令为准。

## profiling 输入输出

输入：

- 已编译并安装的算子实现。
- 可独立运行的 example 二进制，例如 `test_aclnn_flash_attention_score`。
- 运行所需输入数据和环境变量。
- `ASCEND_OPP_PATH`、`ASCEND_CUSTOM_OPP_PATH`、`LD_LIBRARY_PATH` 等运行环境。

处理：

- example 触发 aclnn 或 graph 调用。
- runtime 调用 OPP 中的 host 逻辑、tiling 和 AI Core kernel。
- `msprof` 记录 kernel 执行、task、stream、block dim、频率和硬件指标。

输出：

- `OPPROF_*` 目录。
- kernel 级时延，例如 `Task Duration(us)`。
- `Block Dim`、`Current Freq` 等执行信息。
- `ArithmeticUtilization` 下的计算利用率分析数据。
- profiling 日志和可能的 trace 文件。

## simulator profiling

`docs/zh/debug/op_debug_prof.md` 记录了 simulator 路径，适合没有完整硬件采集条件或需要 pipeline 可视化时使用。

典型流程：

```bash
export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
msprof op simulator --output=$PWD/pipeline_auto --kernel-name "<KernelName>" ./test_aclnn_<op_name>
```

输出：

- `pipeline_auto/OPPROF_*/simulator/visualize_data.bin`
- pipeline 可视化数据，可用 MindStudio Insight 查看。

注意：

- simulator 结果不能直接等同真实硬件性能。
- 它更适合分析 pipeline、搬运、计算重叠和局部瓶颈。

## Ascend 950PR 特殊路径

仓库文档还记录了 `ascend950` 相关 `cannsim` 流程：

```bash
cannsim record ./test_aclnn_<op_name> -s Ascend950 --gen-report
```

输出包括：

- `cannsim_*`
- `trace_core0.json`
- 可用 Chrome tracing 查看。

这不是用户当前 B3 环境的默认路径，但后续如果跨平台比较 `ascend950`，需要单独处理。

## 编译耗时分析不是运行性能测试

脚本：`scripts/ci/analyze_ops_time.py`

作用：

- 解析 `build/binary/<soc>/gen` 下的 kernel 编译耗时和大小文本。
- 输出 `build/ops_cost_<soc>.csv`。

静态判断：

- 该脚本分析的是编译成本、生成物大小等，不是算子 runtime performance。
- 后续如果优化构建耗时或 binary size 可以参考，但不能作为 kernel 执行性能指标。

## experimental benchmark

静态发现 `experimental/` 下有若干性能脚本，例如：

- `experimental/mhc/*/test/perf_*`
- `experimental/select_attention_operators/.../benchmark_*.py`
- `experimental/attention/typhoon_mla/bench.py`

静态判断：

- 这些脚本属于 experimental 或专项 benchmark，不是主线算子统一性能入口。
- 如果后续优化对象位于 `experimental/`，应进入对应目录重新梳理 benchmark 依赖和指标。

## 性能测试模块输入输出总表

| 模块 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| custom package build | 算子源码、SoC、vendor、CANN 工具链 | run package、kernel binary | 生成待测实现 |
| package install | run package、CANN OPP 目录 | custom OPP 文件 | 让 runtime 能加载待测算子 |
| `build.sh --run_example` | 算子名、模式、vendor、SoC | example 二进制 | 生成 profiling 目标程序 |
| `msprof op` | example 二进制、运行环境 | `OPPROF_*`、kernel 时延、利用率 | 真实硬件性能采集 |
| simulator profiling | example 二进制、simulator、kernel name | pipeline 可视化数据 | pipeline 分析 |
| `analyze_ops_time.py` | build 生成的编译日志 | `ops_cost_<soc>.csv` | 编译耗时和 size 分析 |

## 后续优化测量闭环

后续单算子优化建议固定记录：

- git commit 或 patch 标识。
- `--soc`、`--ops`、`--vendor_name`、`--bisheng_flags`、`--kernel_template_input`。
- CANN 版本、driver/firmware 版本、NPU 型号、系统架构。
- example 输入规模、dtype、format、batch/seq/head 等关键 shape。
- `msprof` 命令和输出目录。
- 关键 kernel 的 `Task Duration(us)`、`Block Dim`、硬件利用率。
- 至少 3 次重复运行的均值和波动。

缺口：

- 当前仓库没有统一 benchmark schema。
- 很多 example 未必覆盖真实业务 shape。
- MC2 算子涉及 HCCL、多卡和通信拓扑，单机单卡 example 不能代表真实性能。
