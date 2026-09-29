# KL Loss 跨 mode 同 flagId 超时：客户复现入口

这是**故意保留同步问题的诊断样例**，不是可用于训练的 KL Loss 数值实现或正式修复。
包含 before 对齐的纯同步流水，以及进一步缩减的 CV/VV 最小实验。

- [完整说明：背景、正常/卡死流水、原因、证据边界](docs/klloss_sync/README.md)
- [实测证据与 PR 验收](docs/klloss_sync/VALIDATION.md)
- [算子源码入口](attention/mlp_sparse_lightning_indexer_grad_kl_loss/README.md)
- [后续复用的 flagId 审查 skill](skills/ascend-flag-conflict-audit/SKILL.md)

## 环境

已验证：Linux aarch64、Ascend 910B3、CANN 9.1.0、Python 3.11.14、
torch 2.7.1、torch_npu 2.7.1.post5、GCC 11.4、CMake、Ninja。
准备匹配的 torch/torch_npu 环境并激活；需要安装完整 CANN 开发工具包。
`llvm-objdump` 需能解码 dav-c220；本例使用 CANN 9.2 的工具**仅做反汇编**，运行时仍为 9.1。

```bash
cd /path/to/ops-transformer-klloss
export SLIG_CANN_PATH=/usr/local/Ascend/cann-9.1.0
export SLIG_PYTHON=python3
# 改为本机实际工具路径；不会 source 这个 9.2 环境。
export SLIG_OBJDUMP=/path/to/cann-9.2.0/tools/bisheng_compiler/bin/llvm-objdump
```

所有入口均相对自身定位仓库，没有对旧 klloss 仓库、旧 OPP 或 `/home/dy/tests` 的运行依赖。
默认输出位于 `.klloss_repro/`，已加入 gitignore。源码可迁移，生成目录移动后应重新构建。

## 一键验收（推荐）

**仅在空闲、可执行故障实验的设备上运行。测试会主动触发设备超时。**
默认 NPU 0，可用 `--device` 指定。不要在生产训练 context 中调用这些诊断 kernel。

```bash
# 从源码构建 Host/ACLNN/kernel，逐个新进程执行；默认每组重复 2 次。
bash scripts/klloss/reproduce.sh --device 0 --runs 1000

# 增加单轮与 1024 轮对照。
bash scripts/klloss/reproduce.sh --device 0 --runs 1000 --extended

# 源码和产物未改时，可只重复运行。
bash scripts/klloss/reproduce.sh --device 0 --runs 1000 --skip-build
```

矩阵返回 0 表示：故障组真实报设备 507014 且 PC 匹配该 ELF 的 Wait，对照组正常通过。
扩展矩阵的单轮 same 是探索项：相同 ELF 曾通过也曾超时，只接受完整成功或上述匹配的设备超时，
如实记录，不把它设为固定应通过的安全基线。
故障组没有复现、普通运行异常、编译失败、PC 不匹配或外层 watchdog 超时均不算成功；
遇到非预期结果会停止，不重置设备或自动杀其他进程。日志和 `summary.json` 自动保留。

## 单独编译、单独执行

```bash
# before 对齐的极简同步流水，预期运行报 507014、退出非零。
bash scripts/klloss/compile.sh pipeline
bash scripts/klloss/run.sh pipeline --runs 1000 --op-timeout-s 3

# 最小协议：AIV Set2 → Set0 → Wait0；AIC 只有 Wait。
bash scripts/klloss/compile.sh same 2
bash scripts/klloss/compile.sh split 2
bash scripts/klloss/run.sh same 2 --runs 1000 --op-timeout-s 3
bash scripts/klloss/run.sh split 2 --runs 1000 --op-timeout-s 3

# 也可一次构建 pipeline + same_r2 + split_r2。
bash scripts/klloss/compile.sh all 2
```

`same` 的 CV/VV 都是 flag 0；`split` 仅将 VV 改为 15。`2` 是单个 kernel 内的轮数，
`--runs` 是 Host 调用次数。故障调用不会完成 1000 次，通常第一次就退出。
**15 只在已去掉隐式 KFC 初始化的最小样例中作为对照 ID；不是原计算 kernel 的修复建议。**

编译脚本使用已安装 SDK 的 custom-op 模板，完整重编 Host、ACLNN、FP16/BF16 kernel，
不联网下载仓库顶层构建依赖、不复用预编译种子包、不修改系统 SDK/OPP。
torch-extension 由运行脚本从本仓库源码 JIT，缓存也隔离在生成目录。
当前数值输入/硬件回归只测 BF16；编译通过不等于 FP16 已完成硬件回归。

可配置：`SLIG_WORK_DIR`（独立生成目录）、`SLIG_JOBS`（默认 8）、
`SLIG_EXTENSION_JOBS`（默认 2）、`SLIG_TIMEOUT_SECONDS`（Host watchdog，默认 300 秒）。
单独预编译适配层：`bash scripts/klloss/run.sh split 2 --build-extension`。

CPU 验证：`python3 scripts/klloss/test_tools.py -v`。图已预渲染，不需要安装绘图依赖才能阅读；
如要重绘，安装 matplotlib 和中文字体后运行 `python3 docs/klloss_sync/diagrams/draw_pipeline.py`。

本分支未改原有生产算子，未提交二进制，也未自动创建、提交或推送 PR。
