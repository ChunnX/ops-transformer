# FlashKDA V2 冻结二进制交付

本目录按照当前 CANNBotDSL `package.operator` 和独立 `cannbot_framework`
架构构建 FlashKDA。`attention/flash_kda/flash_kda.py` 声明 `@entry`、两个
`@jit` Provider 和显式发布域；`kernels.py` 保存设备计算。Entry 分配的返回张量
自动成为 output，其余活跃分配成为 workspace。常量初始化也是独立 Provider，
由冻结 HostPlan 排序执行。

交付流程为：`build_operator` → schema 2 `OperatorBundle` →
`cannbot_framework.stage` → `cannbot_framework_add_frozen_bundle` →
父 OPP 的 `libcust_opapi.so`。CMake 只消费已经生成的 stage，不导入算子 Python
源码、不执行编译。ACLNN L0、两阶段 public API、External Task 与 workspace
绑定全部由独立 framework 生成。

## 当前发布契约

| 参数 | 契约 |
| --- | --- |
| `q, k` | BF16，连续 BNSD，`[B, Nk, S, 128]` |
| `v, g` | BF16，连续 BNSD，`[B, Nv, S, 128]` |
| `beta` | BF16，连续 BNS，`[B, Nv, S]` |
| `initial_state` | FP32，`[B, Nv, Dv=128, Dk=128]`，调用后不变 |
| `a_log, dt_bias` | FP32，`[Nv]` / `[Nv, 128]` |
| `B, S` | `1 <= B <= 2`；`64 <= S <= 1024`，S 为 64 的倍数 |
| `(Nk, Nv)` | `(1,1)`、`(1,2)`、`(2,2)`，覆盖普通 attention 和 GQA |
| `scale` | 有限浮点数，`[0,1]` |
| `lower_bound` | 有限浮点数，`[-5,-1]` |
| 输出 | `(output, final_state)`，分别与 `v`、`initial_state` 同 shape/dtype |

S 是同一二进制的运行时维度，并未为测试形状逐项编译。workspace 按最多 8 个
chunk 循环复用，S=1024 验证跨 group 的状态递推。当前仅发布上述 BNSD 域，
没有声明 BSND/TND、可变长序列或零衰减域。

`lower_bound=0` 在额外探针中出现 FP32 state 误差：22/16384 个元素不满足
`atol=rtol=5e-3`，最大绝对误差约 0.008646，因此该值没有进入当前二进制契约。
相关失败记录与通过记录一并保留；验收不放宽精度阈值。

Target 必须显式提供可驻留的 AIC/AIV 数。CANNBOT-10124 的 NPU0 实测为
28 AIC / 56 AIV；全网格 barrier 要求所有 block 同时驻留，构建时使用
`--aic-count 28`。核数进入 Target 身份、Provider receiver 和 exchange workspace。
发布阶段不查询当前设备，验收运行前检查部署设备核数足够。

## 从框架 run 安装结果构建

先 source 指定 CANN 环境，然后激活框架 run 安装根的 Python 和 CMake 路径。
这里的 `FRAMEWORK_PREFIX` 必须是 run 安装结果；不需要安装 cannbotdsl wheel。

```bash
source /home/z00947297/WORK/ENV/set_env_9.2.sh
export FRAMEWORK_PREFIX=/absolute/framework/run/install/root
export PYTHONPATH="$FRAMEWORK_PREFIX/python/site-packages"
export CMAKE_PREFIX_PATH="$FRAMEWORK_PREFIX"
python -m cannbotdsl_ops.attention.flash_kda.flash_kda /absolute/fresh/bundle \
  --npu-arch dav-3510 --aic-count 28
python cannbotdsl_ops/stage_flash_kda.py \
  --bundle /absolute/fresh/bundle --output /absolute/fresh/stage \
  --parent-opp cannbotdsl_ops
```

作为独立小型父 OPP 构建：

```bash
cmake -S cannbotdsl_ops -B build-flash-kda -G Ninja \
  -DFLASH_KDA_FROZEN_STAGE=/absolute/fresh/stage \
  -DCMAKE_PREFIX_PATH="$FRAMEWORK_PREFIX" \
  -DNNOPBASE_LIBRARY="$FRAMEWORK_PREFIX/x86_64-linux/lib64/libnnopbase.so" \
  -DNNOPBASE_INCLUDE_DIR="$FRAMEWORK_PREFIX/x86_64-linux/include/aclnn"
cmake --build build-flash-kda -j 6
cmake --install build-flash-kda --prefix /absolute/fresh/payload
python cannbotdsl_ops/package_flash_kda.py \
  --payload /absolute/fresh/payload --output /absolute/flash-kda-v2.run \
  --makeself /absolute/makeself/makeself.sh
bash /absolute/flash-kda-v2.run --check
bash /absolute/flash-kda-v2.run -- /absolute/fresh/install
```

仓库主工程可使用 `ENABLE_CANNBOTDSL_PROVIDER_ACLNN=ON`、
`CANNBOTDSL_PROVIDER_ONLY=ON`、`ASCEND_OP_NAME=flash_kda` 和相同 stage。
主工程已有 `nnopbase` target 时，应另外明确设置 `NNOPBASE_LIB_DIR` 与
`NNOPBASE_OPDEV_INC_DIR` 指向该 run 的库和头文件。主工程保留传入的
`CMAKE_PREFIX_PATH` 并追加 CANN 基础依赖根，因此可直接从该前缀发现
`CANNBotFramework`。

本机正式主工程 run 的可复现入口位于
`/home/z00947297/run_toolkit/verification-20260907-ckuWoU/terminal-commands`：

```bash
COMMANDS=/home/z00947297/run_toolkit/verification-20260907-ckuWoU/terminal-commands
export REPRO_ROOT=$(mktemp -d /home/z00947297/run_toolkit/flash-kda-main.XXXXXX)
bash "$COMMANDS/build-transformer-flash-kda-run.sh"
bash "$COMMANDS/verify-transformer-flash-kda-run.sh" "$REPRO_ROOT/result.env"
```

构建脚本默认消费已验证 stage；增加 `--rebuild-bundle` 可从当前算子源码调用
run 安装的 `build_operator` 重新冻结并 stage。脚本以 `VENDOR_NAME=custom`、
`ASCEND_OP_NAME=flash_kda`、两个 Provider 开关构建仓库根 CMake 的 `package`
target，产物为 `output/cann-ops-transformer-custom_linux-x86_64.run`。
该命令只打包 FlashKDA，不代表其他 transformer 算子完成验证。
构建目录、日志、`result.env` 和带 SHA-256 的 `result.json` 均在独立
`REPRO_ROOT` 下；显式 `CMAKE_INSTALL_PREFIX` 不再被覆盖到仓内 `build_out`，
CPack 跟随所选 CMake generator，因此支持 Ninja。

验证脚本使用匹配的 framework run 中 `libnnopbase.so` 完成安装器的 DSO
检查，然后移动安装根并暂时隐藏构建目录，执行 6 个 case × 2 次 C++ ACLNN
调用。正式安装布局为 `INSTALL_ROOT/vendors/custom_transformer`，包含
`op_api/include/aclnn_FlashKda.h`、`op_api/lib/libcust_opapi.so` 和
`op_impl/cannbotdsl/FlashKda` 完整冻结 bundle。运行前须将该 vendor 的
`op_api/lib` 与匹配 framework run 的 `x86_64-linux/lib64` 依次置于
`LD_LIBRARY_PATH` 前部；不需要安装 wheel。脚本记录实际加载库路径、完整
安装文件 SHA-256 和全元素数值报告。CANN 的 makeself `--check` 会继续后续
动作，因此脚本使用 `--check --noexec --extract=<独立目录>` 检查归档。

生成的 public API 头文件为 `aclnn_FlashKda.h`；参数顺序以该头文件为准，
当前为 8 个输入、2 个输出、`lower_bound`、`scale`、workspace/executor。
调用方式为 `aclnnFlashKdaGetWorkspaceSize` / `aclnnFlashKda`。父仓拥有包布局
和安装入口，framework 拥有生成的 ACLNN 与 External Task glue。

## 完整数值验收

```bash
python cannbotdsl_ops/tests/validate_flash_kda_frozen.py \
  --bundle /absolute/fresh/bundle --output /absolute/native-report --device 0
python cannbotdsl_ops/tests/validate_flash_kda_frozen.py \
  --aclnn-executable build-flash-kda/flash_kda_aclnn_validate \
  --output /absolute/aclnn-report --device 0
```

验收用独立 CPU 逐 token 递推作为 golden，对完整 output 和 final_state 逐元素
检查 `atol=rtol=5e-3`。测试包括 S=64/128/512/1024、非零初始状态、GQA、
跨 group、多 batch、重复调用和标量边界。所有结果必须有限且 initial_state
逐位不变。Python frozen 模式将 `JitFunction.compile` 替换为失败哨兵，确认
调用阶段不会 JIT。C++ ACLNN runner 不导入 Python 或 cannbotdsl，并设置
20 秒 kernel timeout。报告包含完整元素数量及绝对误差/RMS；无采样验收。

安装后运行 ACLNN 时将安装树的 `packages/vendors/custom_transformer/op_api/lib`
放在 `LD_LIBRARY_PATH` 最前。registry 依照自身库位置相对定位冻结 bundle，
可在移动安装目录、移走构建目录后验证交付物独立性。

## 2026-09-07 CANNBOT-10124 验证记录

框架来自提交 `bc2a125375c4f1eafe68b61d571080c00ca26c29` 构建的 run，
实际安装根为 `/home/z00947297/run_toolkit/cannbot-arena/framework-run-bc2a1253/cann`。
编译器、独立 framework 和 External Task opbase 均取自该安装环境。

证据根目录：`/home/z00947297/run&toolkit/verification-20260907-ckuWoU`。

| 证据 | 相对证据根目录的路径 |
| --- | --- |
| 最终冻结 bundle / ACLNN stage | `flash-kda-bundle-bc2a1253` / `flash-kda-stage-bc2a1253` |
| 冻结 binary 全量数值 | `flash-kda-native-bc2a1253/report.json` |
| 产品 run 安装后纯 C++ ACLNN 全量数值 | `flash-kda-aclnn-installed/report.json` |
| 同 shape 非连续 stride 等负例 | `flash-kda-negative-same-shape/report.json` |
| 安装库实际加载路径 | `flash-kda-aclnn-installed-ldd.log` |
| 产品包 / 安装根 | `flash-kda-v2.run` / `flash-kda-installed` |
| 零衰减域失败探针 | `flash-kda-native.log` |

最终 bundle identity 为
`0f62297b7093158aaf374f922fd32ba8b44eb1019cd21c3527970eacd26d03af`，
包含 7 个 Provider DSO、6 条动态 route，每条 route 为两个有序 Provider。
Native 和安装后 ACLNN 均通过 6 个 case × 2 次调用的完整数值比较，
最大 output 绝对误差 0.003069401，最大 state 绝对误差 0.004327089。
纯 C++ 验收时已移走原构建目录，`ldd` 确认 `libcust_opapi.so` 和
`libnnopbase.so` 来自产品 run 的安装树，该安装树无 `.py` 或 `.whl` 文件。

产品 run SHA-256：
`618e1c89198c1280c46c2a5519aa4d0c26683debb79905cc4e54569da719ac9a`。

随后通过仓库主工程 `package` target 完成正式 FlashKDA run：
`/home/z00947297/run_toolkit/repro-transformer-flash-kda.dEHcjJ/output/cann-ops-transformer-custom_linux-x86_64.run`，
SHA-256 为 `c81434ad4c8010e9d6a25c9655601b7628cad5791117f80639efcd36e532f980`。
该输出根的 `result.json` 记录构建源码、CMake 补丁与包身份；
`installed-validation.xvXwYe/result.json`、`accuracy/report.json` 和
`installed-ldd.log` 记录 NPU0 上迁移安装后的 6 × 2 次纯 C++ ACLNN 通过。
正式安装包含同一 7 个 Provider DSO 和完整 bundle；数值最大误差与上表一致，
vendor 安装树无 `.py` 或 `.whl`。这里依赖匹配 framework run 安装的 opbase，
`libcust_opapi.so` 来自本次正式 run 的迁移安装树。

可以在仓根按以下命令复验已有安装产物：

```bash
source /home/z00947297/WORK/ENV/set_env_9.2.sh
export FRAMEWORK_PREFIX=/home/z00947297/run_toolkit/cannbot-arena/framework-run-bc2a1253/cann
export PYTHONPATH="$FRAMEWORK_PREFIX/python/site-packages"
VERIFY_DIR=/home/z00947297/run_toolkit/verification-20260907-ckuWoU
python cannbotdsl_ops/tests/validate_flash_kda_frozen.py \
  --bundle "$VERIFY_DIR/flash-kda-bundle-bc2a1253" \
  --output /tmp/flash-kda-native-recheck --device 0
export LD_LIBRARY_PATH="$VERIFY_DIR/flash-kda-installed/packages/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH"
python cannbotdsl_ops/tests/validate_flash_kda_frozen.py \
  --aclnn-executable "$VERIFY_DIR/flash-kda-aclnn-validate" \
  --output /tmp/flash-kda-aclnn-recheck --device 0
```
