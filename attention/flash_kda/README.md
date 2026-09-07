# FlashKDA：保持原实现的冻结发布适配

本目录以 `cannbot-arena/samples/flash_kda/flash_kda.py` 为原实现基线。
基线 SHA-256 为 `6dc4c436098784b314962c37a94edeb4cd16dbe9e61dbb2d02a03e8e659b996e`。
`flash_kda.py` 的前 2183 行完整保留该文件，新增适配只追加在文件末尾。
原函数、类、常量、Channel 事务、同步、计算、布局与普通 Python JIT 调用路径均保留。

## 原接口与发布接口

原 `flash_kda(q,k,v,g,beta,scale,initial_state,A_log,dt_bias,lower_bound,layout_qkv,cu_seqlens=None)`
继续保留。支持 BNSD/BSND，返回布局跟随输入；`cu_seqlens` 原来就仅接受 None。
原来的自动 group/Dv 选择和 `KDA_GROUP`/`KDA_DV_BASE` 行为保持。

新增 `@entry(name="FlashKda")` 入口用于冻结发布，不替换原 Python 函数。
当前 ACLNN 生成器不支持字符串属性，因此发布接口通过 `is_bsnd` 布尔属性表示
布局（false=BNSD，true=BSND），不接收无实际功能的 `cu_seqlens`。
这属于新增 C ABI 的映射，不能称为原 Python 签名完全不变的二进制 API。

新增适配负责：

- 用显式 Target 提供核数，替代冻结时查询当前设备；
- 在冻结计划中分配输出与工作空间；
- 用设备初始化 Provider 生成与原版等值的单位矩阵及 mask 常量；
- 复用原 `FlashKDA.run`，不复制或重写设备计算；
- 提供与原自动 group 分桶等价的可捕获表达，以及显式 group/Dv 覆盖配置；
- 提供显式发布 profile 与 bundle 构建入口。

当前框架要求冻结 profile 的维度有界。profile 描述某个二进制包实际发布的范围，
不是原算子的功能上限；改变 profile 应重新冻结，不可在运行时回退到 JIT。
`lower_bound=0` 和 `scale>1` 不再作为人为裁剪范围的无效输入。

## 构建接入

目录只包含 `flash_kda.py`、本说明、`CMakeLists.txt` 及一份
`tests/test_flash_kda.py`。没有独立 kernels.py、package_flash_kda.py 或占位 C++。
CMake 从 ops-transformer 根目录接入 frozen stage，生成真实 API 库，使用官方
custom/builtin run 打包流程。Provider-only 不生成传统空 proto/tiling 库。

验证必须使用框架 run 安装环境重新冻结当前源码。上一轮受限实现的 bundle、
run 包和精度记录仅是历史记录，不能作为本次恢复原实现后的验收证据。

## 验证原则

验证首先检查原文件字节前缀、全部原 AST 及无原符号覆盖，然后对同一输入运行
原入口与新增二进制入口，并独立对照 CPU recurrence，保留原精度阈值。
原版自身不满足 CPU 阈值的情况应单独记录为原版复现，不得裁域或放宽阈值后
声称精度全部通过。实现一致性与 CPU 数值精度是两个独立结果。

## 冻结命令

在 CANNBOT-10124 上使用已安装的 framework run，无需安装 DSL wheel：

```bash
source /home/z00947297/WORK/ENV/set_env_9.2.sh
export FRAMEWORK_PREFIX=/home/z00947297/run_toolkit/verification-20260907-ckuWoU/flash-kda-preserve-original/framework-stride-run/dsl/run-install/cann
export PYTHONPATH="$FRAMEWORK_PREFIX/python/site-packages"
export CMAKE_PREFIX_PATH="$FRAMEWORK_PREFIX"
cd /home/z00947297/run_toolkit/ops-transformer-zyh
python attention/flash_kda/flash_kda.py /absolute/fresh/bundle \
  --npu-arch dav-3510 --aic-count 28 \
  --head-pairs 1:1,1:2,2:2,2:4,4:8,3:3,96:96 \
  --batch-min 1 --batch-max 4 --sequence-min 64 --sequence-max 8192
python -m cannbot_framework.cli stage-bundle \
  --bundle /absolute/fresh/bundle --operator FlashKda --vendor custom_transformer \
  --parent-opp "$PWD" --output /absolute/fresh/stage
```

该 profile 同时发布 BNSD 和 BSND。B 的每个合法整数分别形成显式签名，S 保持动态，
以避免框架对复合 BN 条件产生冗余路由证明；没有通过采样推断未声明的输入域。
可修改 `--head-pairs`、batch/sequence 范围重新构建其他 profile。
`--group`、`--dv-base` 默认读取对应 KDA 环境覆盖值，0 表示原自动策略。
二进制调用必须使用该包声明的覆盖属性值，不会在运行时隐式读取环境变量。

## 单例 stride 的发布支持

原 API 接受大小为 1 的维度具有任意非负 stride 的连续张量。发布签名为这些维度
分别声明独立 stride 符号，保留标准构造、transpose 后 contiguous 及混合来源输入。
非单例维仍要求连续布局；没有通过重新构造测试输入来避开原 API 合法的 stride。
新增二进制接口按物理 shape 分配连续输出；大小为 1 的维度的输出 stride 可能
规范化，但 shape、dtype、布局和有效元素地址关系保持，原 Python 接口仍完全不变。

为此，framework 的通用 stage 生成器补充了仅出现在 stride 中的 Dim 绑定。
该修复位于 cannbot-dsl 的 framework/python/cannbot_framework/bundle.py，保留范围、
倍数与共享符号约束，无 FlashKDA 专用分支，相关 11 项回归通过。最终stage使用
源码重新构建并安装的 e766e2d0 framework run；没有直接修改已安装的SDK。

## 本轮原实现对照记录

最终bundle identity：`a9c7180287364fac742a0267e2452879a2c178dc9285acbc46c8c56c35e92929`。
主profile包含56个签名、86条路由和87个Provider；source与stage均有独立身份记录。

新framework run环境下，NPU2的完整native矩阵为8个case各调用2次，全部output和
final_state与原arena入口逐位相等，重复调用逐位相等、initial_state不变。
覆盖BNSD/BSND、单例维混合stride、自动Dv16/32/64/128分桶、3/96头和8192序列。
独立group4/Dv32小包的两布局对照也逐位相等，常量初始化8192/512字节逐位一致。

CPU原精度阈值仍为atol=rtol=5e-3。主8case中5个通过；3个lower_bound=0用例
原arena实现自身未通过该阈值，新入口输出与原版完全相同，报告明确标记
`equivalent_with_baseline_accuracy_failure`。本轮没有修改数学实现、放宽阈值或
排除这些输入，因此不能表述为“全部原精度用例通过”。

验证报告位于CANNBOT-10124：
`/home/z00947297/run_toolkit/verification-20260907-ckuWoU/flash-kda-preserve-original/`。
其中`main-native-validation-v2/report.json`记录完整原实现对照，
`framework-review-v2.json`记录原字节/AST、调度、常量与metadata证据。

正式 builtin run 的移位安装 ACLNN 路径也完成相同8case×2调用：output/state
与原arena全部逐位相等，重复调用逐位相等，输入状态不变。运行时隐藏原build目录，
ldd确认API来自移位后的安装目录、opbase来自新framework run验证目录。
CPU参考仍为5case通过、3case原版超差复现，未改变判据。
报告：`main-aclnn-validation-v2-retry/report.json`；库来源：
`main-aclnn-validation-v2-retry/runner-provenance.txt`。

本轮正式包：

- builtin：`/home/z00947297/run_toolkit/repro-flash-kda-preserved-builtin-v2/result.env`
- custom：`/home/z00947297/run_toolkit/repro-flash-kda-preserved-custom-v2/result.env`

两种包均完成构建、全新安装、移位及依赖/payload检查。custom本轮仅做安装与
依赖验收，没有把builtin数值结果重复记为custom数值验收。

```bash
COMMANDS=/home/z00947297/run_toolkit/verification-20260907-ckuWoU/terminal-commands
DEVICE=2 bash "$COMMANDS/verify-transformer-flash-kda-run.sh" \
  /home/z00947297/run_toolkit/repro-flash-kda-preserved-builtin-v2/result.env
bash "$COMMANDS/verify-transformer-flash-kda-run.sh" \
  /home/z00947297/run_toolkit/repro-flash-kda-preserved-custom-v2/result.env --install-only
```
