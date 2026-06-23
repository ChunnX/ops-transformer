# 环境信息静态分析

## 当前用户环境基线

用户提供的当前环境：

- 硬件：Ascend 910B 系列 B3
- 系统架构：aarch64

仓库静态配置对应：

- 默认 SoC：`ascend910b`
- `ascend910b` 对应硬件族：Atlas A2
- CMake 架构目录：`arch22`
- 后续构建命令应优先使用 `--soc=ascend910b`

## 仓库版本和 CANN 依赖

`version.cmake` 记录：

- package 版本：`ops-transformer` `9.1.0`
- build 依赖最低版本：`8.5`
- run 依赖最低版本：`8.5`

主要 CANN 依赖包：

- `runtime`
- `opbase`
- `hcomm`
- `ge-executor`
- `metadef`
- `ge-compiler`
- `asc-devkit`
- `bisheng-compiler`
- `asc-tools`
- runtime 侧还依赖 `ops-nn`、`hccl`

仓库 `README.md` 和安装文档提到 WebIDE/Docker 当前默认商用 CANN 版本为 `8.5.0`。

## 支持的 SoC 和硬件族

`build.sh` 静态支持列表：

| SoC | build.sh 映射硬件 | CMake 架构目录 |
| --- | --- | --- |
| `ascend910b` | Atlas A2 | `arch22` |
| `ascend910_93` | Atlas A3 | `arch22` |
| `ascend950` | Ascend 950PR / Ascend 950DT | `arch35` |
| `ascend310p` | Atlas Inference | `arch20` |
| `kirinx90` | Kirin | `arch22` |
| `kirin9030` | Kirin | `arch22` |
| `mc62cm12a` | MC62 | `arch38` |

说明：

- 用户当前 B3 环境属于 `ascend910b` 使用范围。
- 单个算子是否实际支持某 SoC，还要结合 `op_host/config`、CI list、CMake 条件和真实 build 结果确认。

## 系统依赖

`docs/zh/context/quick_install.md` 记录的依赖：

- Python：`>=3.7.0`，文档建议 `<=3.10`
- gcc：`>=7.3.0`
- cmake：`>=3.16.0`
- make
- gawk
- dos2unix
- pigz：可选，建议 `>=2.4`
- googletest：用于 UT

根目录 `requirements.txt`：

- `pyyaml`
- `numpy`
- `decorator`
- `sympy`
- `scipy`
- `attrs`
- `protobuf`
- `psutil`

`tests/requirements.txt`：

- `tensorflow==2.20.0`
- `valgrind`

注意：

- pytest 精度用例常见依赖 `torch`、`torch_npu`，部分 graph 用例可能依赖 `torchair`。这些没有完整体现在根目录 requirements 中。
- 真实运行、profiling 和 MC2 通信类算子还需要 driver、firmware、HCCL 和可见 NPU 设备。

## CANN 路径和环境变量

关键变量：

| 变量 | 作用 |
| --- | --- |
| `ASCEND_HOME_PATH` | CANN 安装根路径，`build.sh` 会用它定位 toolkit。 |
| `ASCEND_OPP_PATH` | OPP 路径，运行和 example 构建会读取 include/lib/config。 |
| `ASCEND_CUSTOM_OPP_PATH` | custom OPP 路径，custom package 运行时常用。 |
| `LD_LIBRARY_PATH` | 运行 example、msprof、simulator 时定位 CANN/custom lib。 |
| `BISHENG_FLAGS` | 可通过 `--bisheng_flags` 影响 AscendC kernel 编译。 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 调试时可打开日志到 stdout。 |

典型初始化：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

`build.sh` 也会尝试 source：

```bash
$ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash
```

## Docker 和 WebIDE

文档给出的 910B Docker 镜像示例：

```bash
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
```

静态说明：

- 文档提到镜像支持 `arm64` 和 `amd64` 拉取。
- Docker 文档当前面向 Atlas A2 系列和 Ubuntu。
- 用户当前 aarch64 + 910B B3 与该镜像方向一致，但真实环境仍要确认 driver、firmware、container runtime 和 NPU 透传。

## 安装包和运行路径

custom package 构建后通常位于 `build_out/`，安装后进入 CANN OPP vendors 目录：

```text
${ASCEND_HOME_PATH}/opp/vendors/<vendor_name>_transformer/
```

custom package 可能通过 `ASCEND_CUSTOM_OPP_PATH` 挂载并优先于原始 CANN package。后续性能优化时，要确保 profiling 运行的确实是当前修改后的 custom package，而不是系统自带实现。

## B3 环境后续核查清单

静态分析不能确认真实设备状态。开始性能优化前建议记录：

- `uname -m` 是否为 `aarch64`。
- `npu-smi info` 中设备型号、driver、firmware。
- CANN toolkit/opp 版本。
- `which bisheng` 和 `bisheng --version`。
- `python3 --version`。
- `gcc --version`。
- `cmake --version`。
- `echo $ASCEND_HOME_PATH`、`echo $ASCEND_OPP_PATH`、`echo $ASCEND_CUSTOM_OPP_PATH`。
- `msprof --version`。
