# FlashKDA 冻结二进制接入

本目录使用当前 CANNBotDSL `package.operator` 与独立 `cannbot_framework`。
数学主体保留已有 `zyh-binary` 实现；接入层改成 `@entry` 描述分配与返回值、
两个有序 `@jit` Provider 描述常量初始化和设备计算，再由框架生成冻结
HostPlan、workspace 绑定、ACLNN 两阶段 API 和 External Task glue。

## 源码职责

```text
cannbotdsl_ops/
  CMakeLists.txt                         # 接入仓库标准 custom/builtin 主工程
  provider_only_stub.cpp                 # 主工程空传统目标的占位
  attention/flash_kda/
    flash_kda.py                        # Entry、Provider、发布域、build_operator CLI
    kernels.py                          # 设备数学与同步实现
  tests/test_flash_kda.py                # 唯一用例矩阵、CPU golden、精度判据
```

早期的 `package_flash_kda.py` 和 `install.sh` 用于独立小父 OPP 的打包探针，
正式主工程 run 打通后已删除。`stage_flash_kda.py` 也已由 framework 自带
`stage-bundle` CLI 替代。现在 CMake 必须从仓库根配置，安装与卸载均由标准
ops-transformer 工程负责。

测试只维护一个 Python 文件。它内嵌的最小 C++ fixture 只负责张量搬运、
调用生成的 ACLNN API 和回写结果，不包含第二份 golden、用例或精度阈值。
CMake 通过 `--emit-aclnn-runner` 生成该 fixture；生成过程不导入 Torch/DSL。
原未跟踪的 `attention/flash_kda` 参考目录已逐文件 SHA-256 核验并归档到
`/home/z00947297/run_toolkit/verification-20260907-ckuWoU/flash-kda-original-reference-jx4kgden`，
不再参与源码测试发现。其旧测试指向原 `samples/flash_kda`，没有作为新接口验收。

## 与旧接入的区别

| 项目 | 旧接入 | 当前接入 |
| --- | --- | --- |
| 主机编排 | source stage、手工 workspace/cache/dispatch | Entry 捕获、schema 2 bundle、冻结 HostPlan |
| ACLNN | 仓库内专用封装与 ABI 胶水 | 独立 framework 生成 public API、L0、External Task |
| 二进制调用 | 与旧 source stage 接口绑定 | 加载已发布 Provider DSO，调用时不 JIT |
| 调度策略 | 旧 host 动态 group/Dv 策略 | 8 chunk workspace 分组、Dv 分片 16，Target 显式核数 |
| 交付 | 曾用小父 OPP 探针 | 标准主工程 custom run / builtin 组件 run |

设备矩阵计算与递推主体保留；vector API 更新为当前 raw_reg 名称与关键字。
显式 28 AIC 不是新增数学算法，旧离线包也使用该核数。全网格 barrier 要求
所有 launch block 可同时驻留；本机发布 Target 为 `dav-3510`、28 AIC/56 AIV。

当前发布域也有变化：旧版本曾发布 3/96 个等头数及更大 batch/长序列；当前
选择如下有限 dtype/head 组合和有界动态 S，并增加 GQA 验收。它不是旧发布域
的简单子集，当前结果不能作为旧生产 shape 的完整回归证明。

| 参数 | 当前发布契约 |
| --- | --- |
| `q,k` / `v,g` | 连续 BNSD BF16；`[B,Nk,S,128]` / `[B,Nv,S,128]` |
| `beta` | 连续 BNS BF16；`[B,Nv,S]` |
| `initial_state` | FP32；`[B,Nv,Dv=128,Dk=128]`，调用后不变 |
| `a_log,dt_bias` | FP32；`[Nv]` / `[Nv,128]` |
| B / S | B=1..2；S=64..1024，S 为 64 的倍数 |
| `(Nk,Nv)` | `(1,1)`、`(1,2)`、`(2,2)` |
| `scale` / `lower_bound` | 有限值 `[0,1]` / `[-5,-1]` |
| 返回 | `(output,final_state)`，shape/dtype 分别与 v/initial_state 一致 |

没有声明 BSND/TND、可变长序列、零衰减或旧版的全部大 shape。额外
`lower_bound=0` 探针有 22/16384 个 state 元素超出 `atol=rtol=5e-3`
（最大绝对误差约 0.008646），因此未纳入发布域；失败记录保留。

## 冻结与主工程打包

先 source CANN，再使用 framework run 安装目录；不安装 cannbotdsl wheel。

```bash
source /home/z00947297/WORK/ENV/set_env_9.2.sh
export FRAMEWORK_PREFIX=/home/z00947297/run_toolkit/verification-20260907-ckuWoU/run-installations/framework-run-bc2a1253/cann
export PYTHONPATH="$FRAMEWORK_PREFIX/python/site-packages"
export CMAKE_PREFIX_PATH="$FRAMEWORK_PREFIX"
cd /home/z00947297/run_toolkit/ops-transformer-zyh
python -m cannbotdsl_ops.attention.flash_kda.flash_kda /absolute/fresh/bundle \
  --npu-arch dav-3510 --aic-count 28
python -m cannbot_framework.cli stage-bundle \
  --bundle /absolute/fresh/bundle --operator FlashKda --vendor custom_transformer \
  --parent-opp "$PWD" --output /absolute/fresh/stage
```

本机统一终端脚本包含完整 CMake 参数并将产物、日志和 JSON 身份写到独立输出根：

```bash
COMMANDS=/home/z00947297/run_toolkit/verification-20260907-ckuWoU/terminal-commands
export REPRO_ROOT=$(mktemp -d /home/z00947297/run_toolkit/flash-kda-builtin.XXXXXX)
bash "$COMMANDS/build-transformer-flash-kda-run.sh" --builtin
bash "$COMMANDS/verify-transformer-flash-kda-run.sh" "$REPRO_ROOT/result.env"
```

省略 `--builtin` 即构建 custom run；增加 `--rebuild-bundle` 可从当前源码重新
冻结，默认消费已验证 stage。主配置选择 `ASCEND_OP_NAME=flash_kda`、
`ENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON`、`CANNBOTDSL_PROVIDER_ONLY=ON`。
这只验证 FlashKDA，不能代表全量 transformer 算子。

## builtin 的实际适配

custom 使用 `vendors/custom_transformer/op_api/lib/libcust_opapi.so`；builtin
使用正式 `x86_64-linux/lib64/libopapi_transformer.so`，头文件进入
`x86_64-linux/include/aclnnop`，冻结资源进入
`opp/built-in/op_impl/cannbotdsl/FlashKda`。Provider-only 关闭传统 host 生成后，
仍显式创建并安装 builtin API target，链接同一 framework 生成的 OBJECT。

builtin 仅重定位 stage 的父工程安装 metadata，bundle 和生成 C++ 不改写。
当前 HostAPI 按 ACLNN 库目录的 `../../op_impl/cannbotdsl` 查找 Provider，
所以官方 XML 清单用 `install_softlink` 登记
`op_impl/cannbotdsl -> ../opp/built-in/op_impl/cannbotdsl`。实际资源仍在 builtin
OPP；相对链接随官方安装/卸载清单维护。将来 framework 支持配置 builtin
资源根后可去掉这条兼容链接，不需要改变算子数学。
当前包拥有整个 `opp/built-in/op_impl/cannbotdsl` 资源目录；尚未验证与其他
共享该目录的 builtin Provider 组件共存。多组件集成前需按算子隔离文件清单
所有权，避免一个组件卸载其他组件资源。

正式 builtin 包名中的 `9.1.0` 来自本仓 `version.cmake` 的组件版本，和运行
环境 CANN 9.2 是两个独立版本。验收必须优先加载匹配 framework run 中支持
External Task 的 `libnnopbase.so`，不能只使用 CANN 基础环境的旧 opbase。

## 统一验收

```bash
python cannbotdsl_ops/tests/test_flash_kda.py \
  --bundle /absolute/frozen/bundle --output /absolute/native-report --device 0
python cannbotdsl_ops/tests/test_flash_kda.py \
  --aclnn-executable /absolute/flash-kda-aclnn-validate \
  --output /absolute/aclnn-report --device 0
```

同一测试用独立 CPU 逐 token 递推，对完整 output/state 逐元素检查
`atol=rtol=5e-3`、finite、initial_state 不变与重复调用。8 个 case × 2 次覆盖
三种 head 组合的短 S/长 S 六条冻结 route，加 gate=-5/scale=1、scale=0 边界；
不表示穷举整个连续标量域或动态形状域。Native 模式另有四项拒绝测试，
包括保持 shape 的非连续 stride，并将 `JitFunction.compile` 替换为失败哨兵。
`--negative-only` 可只复验拒绝输入。C++ fixture 设置 20 秒 kernel timeout，
每 case 外层 90 秒超时；安装验收检查迁移目录后实际加载库与 bundle 文件。

前轮 custom 主工程精度记录保留在
`/home/z00947297/run_toolkit/repro-transformer-flash-kda.dEHcjJ/installed-validation.xvXwYe`。
那次为 6×2，通过的 output/state 最大绝对误差分别为
0.0030694007873535156 / 0.0043270885944366455，未覆盖全部六条冻结 route。
早期小父 OPP 产物只作为历史证据保留，本目录不再提供该打包入口。

2026-09-07 整理后的正式 builtin 使用 ops 提交 `2499e5ac75b884b2af44aa24e650b011a56a5500`
和 cmake-zyh 提交 `3e65fd76b88796d2cba0b748e2b3487fc102da88` 构建，NPU0
移位安装后 8×2 全量精度通过；最大 output/state 误差仍分别为
0.0030694007873535156 / 0.0043270885944366455。

| 本轮证据 | 绝对路径 |
| --- | --- |
| builtin 包与源码身份 | `/home/z00947297/run_toolkit/repro-transformer-flash-kda.JTJCiN/result.json` |
| builtin 安装后完整精度与实际加载库 | `/home/z00947297/run_toolkit/repro-transformer-flash-kda.JTJCiN/installed-validation.tMgUQ9/result.json` |
| 官方卸载与软链清理 | `/home/z00947297/run_toolkit/repro-transformer-flash-kda.JTJCiN/uninstall-probe.log` |
| custom 正式 package 构建回归 | `/home/z00947297/run_toolkit/repro-transformer-flash-kda.9lcn8s/result.json` |
| 单一测试的四项 native 负例 | `/home/z00947297/run_toolkit/repro-transformer-flash-kda.3DZJTH/single-test-negative/report.json` |

builtin run SHA-256 为 `fa7cd502ec3143a4ece06e358017e17251e6417a672bada68e30d46199d29ab1`。
当前源码重新冻结后 bundle identity 仍为
`0f62297b7093158aaf374f922fd32ba8b44eb1019cd21c3527970eacd26d03af`，证明清理未改变
冻结设备产物。本轮 custom 回归验证了打包构建，custom 数值沿用前轮证据，
没有把 builtin 数值结果记成一次新的 custom 数值运行。
