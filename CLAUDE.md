# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit Message Convention

**MANDATORY**: Before every `git commit`, read `COMMIT_CONVENTION.md` and follow the format exactly. The commit-msg hook will reject non-conforming messages. Key points:
- Format: `<type>[optional scope]: <subject>`
- Types: feat | fix | test | revert | refactor | style | docs | perf | chore
- Chinese subject, no trailing period
- Test results go in footer as `Test:` section（改动前基线 / 改动后实测 / Status），例：
  ```
  Test:
  - 改动前基线: QS128 25.10% pass
  - 改动后实测: pending A5 test
  - Status: pending A5 test
  ```
  改动无测试结果时也要带 `Test:` 段标 `Status: pending A5 test`（A5 验证在用户侧）。
- Add `Co-Authored-By: Claude <noreply@anthropic.com>` footer when AI-generated
- Claude 的提交 author 用 `yayake的claude 客户端 <yayake@claude.local>`（repo-local git config 已设 `user.name`/`user.email`）；用户自己的提交保持 `yayake <yayake@gitcode.com>`，不要重写。

## 这是什么

`ops-transformer` 是 CANN（Compute Architecture for Neural Networks）面向 transformer 类大模型计算的进阶算子库，提供 Ascend C kernel 实现、Host 侧 Tiling、op_api（aclnn）接口、图融合 pass、ONNX 插件以及 PyTorch 扩展，覆盖 attention、MoE、MC2（多芯片/通算融合）、FFN、MHC、位置编码、grouped-matmul、Mamba 等算子。

本仓跟随 CANN 版本发布 —— `version.cmake` 固定 CANN 版本（本仓为 `9.1.0`）并声明构建/运行依赖下限（`>=8.5`）。源码须与 CANN 包配套标签匹配；`master` 分支可能与已安装的 toolkit 版本不匹配。

## 构建与测试入口

所有构建都通过 `build.sh`，它是 CMake 的一层大封装。不要直接调用 `cmake` —— `build.sh` 会解析 CANN toolkit 路径、设置 SoC/arch 并生成 run 包。

### 环境

**A2 vs A5（重要）**：Claude Code 会话所在机器是 A2（ascend910b）；FIA MXFP8 arch35 代码目标是 A5（ascend950）。**A2 本地不能编/跑 arch35 kernel**——改完代码只能 `git push`，由用户在 A5 机器（`~/05-ops-transformer`）`git pull` + `build.sh --soc=ascend950` + 跑 pytest 验证。不要在本地尝试 build/run arch35（编不出可用 A5 二进制，且会浪费整轮）。

**GitCode push 凭据**：remote 是 HTTPS（`https://gitcode.com/yayake/ops-transformer.git`），push 需要凭据。GitCode access token 存在系统环境变量 `GITCODE_TOKEN` 里（不要 echo/记录其值）。push 时把 token 嵌入 URL 即可，无需交互输入：

```bash
git push "https://yayake:${GITCODE_TOKEN}@gitcode.com/yayake/ops-transformer.git" master
```

（用户名 `yayake` 与 remote owner 一致。）直接 `git push origin master` 会因非交互环境无法输入用户名密码而失败。

`build.sh` 要求已安装 CANN toolkit 并设置了 `ASCEND_HOME_PATH`（或会 source `$ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash`）。默认 toolkit 路径：非 root 为 `~/Ascend/ascend-toolkit/latest`，root 为 `/usr/local/Ascend/ascend-toolkit/latest`。

### 常用命令

```bash
# 针对指定 SoC 编译某个算子的 run 包（产物在 build_out/*.run）
bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score -j16 -O3

# 编译整个模块而非单个算子
bash build.sh --pkg --soc=ascend910b --module=attention -j16

# SoC 取值（仅这些合法）：ascend910b ascend910_93 ascend950 ascend310p kirinx90 kirin9030 mc62
#   A3 = ascend910_93，A2 训练/推理 = ascend910b，950 系列 = ascend950

# 只编译库（不生成 run 包）
bash build.sh --ophost            # ophost_transformer.so
bash build.sh --opapi             # opapi_transformer.so
bash build.sh --opgraph           # graph_plugin_transformer.so
bash build.sh --onnxplugin        # oponnx_plugin_transformer.so
bash build.sh --opkernel          # 仅 kernel 二进制

# 单元测试（编译并运行）
bash build.sh -u                  # 全部 UT
bash build.sh --ophost_test       # 仅 host 侧 UT
bash build.sh --opapi_test        # aclnn/opapi UT
bash build.sh --opgraph_test      # graph UT
bash build.sh --opkernel_test     # kernel UT
bash build.sh --ophost_test --noexec -O3   # 只编译不运行

# 端到端运行算子的 examples/test_aclnn_*.cpp 或 test_geir_*.cpp
bash build.sh --run_example flash_attention_score eager cust --vendor_name=custom
#   mode: eager | graph；pkg: cust；可选 --soc= --example_name=

# 仿真模式（无需真实 NPU，需配合 --soc）
bash build.sh --run_example add_example eager --simulator --soc=ascend910b

# experimental 算子需要该开关
bash build.sh --pkg --experimental --soc=ascend910b

# 脚手架：新建一个算子目录
bash build.sh --genop <op_name>
bash build.sh --genop_aicpu <op_name>   # AI CPU 变体

# 清理构建产物
bash build.sh --make_clean
```

`--noexec` 只编译 UT 不执行。`--cov` / `--asan` / `--valgrind` / `--oom` 分别开启覆盖率、AddressSanitizer（host 侧）、valgrind、kernel 侧 OOM。`--ops=` / `--module=` 用逗号分隔；合法 module：`mc2,attention,moe,ffn,mhc,posembedding,gmm,mamba`。

### 安装与运行构建产物

```bash
./build_out/cann-ops-transformer-*linux*.run     # 安装到 $ASCEND_HOME_PATH/opp/vendors/<vendor>
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
```

### Kernel 调试陷阱：PRINTF / DumpTensor（重要）

AscendC `PRINTF`（大写，NPU 模式）和 `DumpTensor` 在真机上把数据写进 kernel workspace 里一块预留的 debug 缓冲。**普通 `build.sh --pkg` 不会分配这块缓冲**——若 kernel 里调用了 PRINTF/DumpTensor，运行时写未映射地址 → AICore error 507015（device error type 3），原本能过的用例也会崩，且无任何输出。

- 想用 PRINTF/DumpTensor 调试：必须 `--debug` 构建（`ENABLE_DEBUG=ON`，分配 debug 缓冲），可选 `--ops-compile-options=-DDEBUG_DUMP` 配合 `#ifdef DEBUG_DUMP` 守卫 PRINTF 源码。注意 `--debug` 会改 kernel 时序（-O0），可能掩盖精度 bug。
- `-DDEBUG_DUMP` 单独用**不够**：它只是把 PRINTF 编进去，不分配缓冲，仍崩。要同时 `--debug`。
- 小写 `printf` 仅 CPU 仿真（`--simulator`）可用，真机不识别。
- 改完 kernel 务必 `rm -rf build/`（kernel 二进制缓存会让旧 bin 混进来）。
- 精度调试优先用 `ascendc-precision-debug` skill（DumpTensor 7 步法 / printf 调试法）。

## 算子目录结构

每个算子位于其模块目录下（如 `attention/flash_attention_score/`）。标准算子（生态最简 → 标准交付件）布局，参见 `docs/zh/install/dir_structure.md` 与 `CONTRIBUTING.md`：

```
${op_class}/${op_name}/
├── CMakeLists.txt
├── README.md
├── op_host/                 # Host 侧：算子定义、Tiling、infershape
│   ├── ${op_name}_def.cpp            # 算子信息库：名称、输入输出、数据类型
│   ├── ${op_name}_infershape.cpp     # 可选：输出 shape 推导
│   ├── ${op_name}_tiling.cpp/.h      # 可选：Tiling 策略（将张量切分为多个小块）
│   ├── ${op_name}_tiling_${sub}.cpp  # 可选：针对子场景的 Tiling
│   └── config/${soc}/                # 可选：二进制配置（缺省时自动生成）
├── op_kernel/              # Device 侧 Ascend C kernel
│   ├── ${op_name}.cpp               # kernel 入口：主函数与调度逻辑
│   ├── ${op_name}.h                 # kernel 实现头文件
│   ├── ${op_name}_tiling_data.h     # TilingData（块大小、并行度等）
│   ├── ${op_name}_tiling_key.h      # TilingKey（标识不同 Tiling 划分方式）
│   └── ${sub_case}/  arch22/  arch35/   # 子场景 / SoC-arch 专属 kernel 变体
├── op_api/                 # 可选：aclnn C++ 接口 + L0 实现
├── op_graph/               # 可选：图融合（proto、infer、fusion_pass/）
├── framework/             # 可选：ONNX/transformer 插件（如 NPUFlashAttention）
├── examples/               # test_aclnn_*.cpp、test_geir_*.cpp
└── tests/
    ├── ut/                 # C++ 单测，分 op_api/、op_host/、op_kernel/
    ├── pytest/             # python 参考实现 + npu 运行测试
    └── st/                 # 系统测试
```

缺省 `op_host` / `op_kernel` / `op_api` / `op_graph` 目录是合法的 —— 该算子可能复用了其他算子的实现，或暂无 Ascend C kernel。`op_host`/`op_kernel` 下的 `arch22`（910b/910_93/kirin9030）、`arch35`（950）、`arch38`（mc62）目录存放 SoC 专属变体。

## Kernel 编程模型

Device 侧 kernel 使用 **Ascend C** 编程模型（`AscendC::` 命名空间）：`LocalTensor`、`GlobalTensor`、`TPipe` / `TQue` / `TBuf`、`DataCopy`、`MMAD`/matmul、`GetTilingData()`。Host 侧 Tiling（`op_host/*_tiling.cpp`）计算如何在 AI Core 间切分任务，并写入 `TilingData` 结构供 kernel 读取；`TilingKey` 在 kernel 运行期选择不同 Tiling 策略。`common/include/op_kernel/` 中的 iterator 所面向的存储层级为：GM → L1 → L0A/L0B → L0C → UB。最近提交历史（BMM1/BMM2、s2SplitSize、splitD、SINNER/SOUTER、MXFP group）反映的是 FlashAttention 式 kernel 的 Tiling 调优 —— D/QS/KVS 维度切分与“写 UB 还是写 GM”的权衡。

## FIA MXFP8 arch35 split-core 流水（当前工作焦点）

`attention/common/op_kernel/arch35/fia_*fullquant_mx*` 是 FIA（fused_infer_attention_score）MXFP8 量化路径，A5（ascend950）专属。这是当前主要调试对象。理解它需要跨 host/kernel 多个文件，下面是架构要点。

**Split-core（AIC cube + AIV vector）**：同一份 kernel 二进制跑在两种物理核上，运行期 `g_coreType` 选分支（`AIC` vs `AIV`）。模板分发器 `fia_template_dispatcher.h` 在匹配核型上实例化 `CubBlock`/`VecFaBlock`/`VecFdBlock` 为 Normal，另一核上实例化为 **`Dummy`（空体）**。⚠️ **Dummy 类与 Normal 类的 constexpr 成员（`mBaseSize`/`dVBaseSize`/`USE_DN`…）必须逐字一致**——只在一处改 constexpr 而忘另一处，会造成 AIC/AIV 静默不匹配（历史上一度导致 D256 输出全 0）。

**单 tile 四阶段流水**（`fia_kernel_fullquant_mx.h::ExecuteTask`）：C1(BMM1=Q·Kᵀ, AIC) → V1(Softmax+cast P, AIV) → C2(BMM2=P·V, AIC) → V2(FlashUpdate 合并, AIV)。阶段间用 `SetCrossCore`/`WaitCrossCore` 同步。`PRELOAD_N=2` 软件流水：第 N 轮同时跑 tile N 的 Phase1 和 tile N−2 的 Phase2，掩盖 AIC↔AIV 等待。

**`USE_DN`（MN-layout 切换）不止换布局，还换操作数角色**：DN 路径（prefill，`quantMode==FULLQUANT_MODE_QKV_MXFP8_PREFILL`=19）把 Q 当作 B 操作数（`l1QBuffers` 当 `mm1B` 用）、K 进 `mm1A`，做 K·Qᵀ；ND 路径（decode=20）才是 Q·Kᵀ。BMM1/BMM2 都有 Nd/Dn 变体，fixpipe 的 `dualDstCtl` 也不同（Nd=1 M-split，Dn=2 M*N/2 split）。DN 还会在 `actSingleLoopS2Size > s2SplitSize` 时用 `0x7f7f` 填充一个**假的 e8m0 PScale**（BMM2 真正的 per-group scale 来自 V-scale，P 已反量化，PScale 只是占位 unity）。

**`l1PBuffers` 跨核共享**：AIV 写 P+PScale（ProcessVec1），AIC 读（IterateBmm2 的 `mm2A`）。3-buffer（`BuffersPolicy3buff`）+ `CROSS_CORE_SYNC_FORWARD`。PScale 区域在该 L1 分配的 `mBaseSize*s2BaseSize` 偏移处。

**关键 Tiling 旋钮**（`attention/common/op_host/arch35/fia_tiling_fullquant_mx.cpp`，`AdjustSinnerAndSouter`）：
- `sOuterFactor_` = `SOUTER_64`（恒 64，AIV M 单位）；`mBaseSize = sOuterFactor_ * CV_RATIO = 128`（`CV_RATIO=2`：每个 AIV 核 2 个 sub-block 各处理 M/2=64 行）。
- `sInnerFactor_` = `SINNER_512`，**但 `vHeadDim==DSIZE_256` 时改 `SINNER_256`——这就是 D256 开关**。`s2BaseSize = sInnerFactor_`，故 D128→512、D256→256；`s2SplitSize = s2BaseSize>>1`（kernel constexpr），D128→256、D256→128。
- `dBaseSize`/`dVBaseSize`：D 模板（`Aligned64/128/256/512/576`），`ComputeTilingData` 按 headDim 分桶。
- `bmm2Write2Ub`（kernel `BMM2_TOUB`）：`GetC2Position(...)` 决定 BMM2 fixpipe 到 UB 还是 GM。
- `splitD`（kernel `SPLITD`）：⚠️ **MX 路径用 `dVTemplateType > Aligned256`（D256→false），GQA 路径用 `>= Aligned256`（D256→true）**，两路径阈值不同，对比时易误读。

**D256 相对 D128 的三处编译期分叉**（`fia_block_cube_fullquant_mx.h`）：① `s2BaseSize` 512→256、`s2SplitSize` 256→128；② `s2BaseSize==256 && dBaseSize>128` 时 L0B 降为 **single-buffer**（D128 仍 double-buffer）；③ `USE_DN && dVBaseSize>128` 时 BMM2 走 `FixpipeMm2SplitM`（M 拆半，`ndNum=2`）而非 `FixpipeMm2`（D128 DN 走普通路径）。改 D256 行为时务必同时核对这三处与 Dummy 镜像。

**`runInfo.actMSizeAlign32` vs `mBaseSize`**：DN fixpipe 路径历史上用 `mBaseSize`（满 128），近期改用 `actMSizeAlign32`（尾 M-tile 可能更小）。src 用 `actMSizeAlign32`、dst 用 `mBaseSize` 的步长不对称是有意为之但脆弱，是 D256 尾块 T 相关 bug 的常见嫌疑点。

**关键文件**：`fia_block_cube_fullquant_mx.h`（BMM1/BMM2/fixpipe）、`fia_block_vec_fullquant_mx.h`（V1/V2）、`fia_kernel_fullquant_mx.h`（PRELOAD 编排）、`fia_template_dispatcher.h`（编译期配置 useDn/bmm2Write2Ub/splitD）、`fia_tiling_data_fullquant.h`、host 侧 `fia_tiling_fullquant_mx.cpp`。`examples/flash_attn_example/.../FA_IMPL.md` 是同款 noquant 流水线的最佳入门文档（内存层级 + PRELOAD 图示）。

## 仓库全局代码位置

- `common/include/` —— 共享头文件，按层划分：`op_kernel/`（iterator、mma、mem、simd、hardware）、`op_host/`、`op_api/`、`op_graph/`、`util/`、`framework/`、`tiling_sink/`、`err/`。
- `common/src/` —— `op_host`、`tiling_sink`、`fallback`、`framework` 的共享实现。
- `common/stub/` —— 供 UT 在未完整安装 CANN 时编译/链接的 stub 头文件。
- `cmake/` —— `func.cmake`（`should_add_module` 等辅助函数）、`func_utest.cmake`/`config_utest.cmake`（UT）、`package.cmake`/`makeself_built_in.cmake`（run 包生成）、`opbuild.cmake`（kernel 二进制构建）。`level2_ops_list` 枚举可构建算子。
- `tests/ut/` —— 跨算子 UT 框架，分为 `framework_normal/`（op_api、op_host、op_kernel、common）与 `framework_special/`（自定义 mock/stub 桩）。
- `experimental/` —— 社区贡献算子；由 `--experimental` 开关控制。新贡献按所属 `${op_class}`（`experimental/attention` 等）放置于此。
- `torch_extension/cann_ops_transformer/` —— PyTorch 扩展：`ops/<api_name>.py` 前端 + `csrc/` C++ + `op_builder/` JIT + `graph_convert/`。开发规范见 `docs/torch_extension_guidelines.md`。
- `examples/` —— 端到端教学样例（`add_example` 是 QUICKSTART 的参考算子）。`examples/fast_kernel_launch_example/` 是最小贡献模板。
- `scripts/`、`classify_rule.yaml`、`OAT.xml` —— license/OAT 合规工具。

## 规范与合规

- 不确定某个 flag 时，查 `bash build.sh -h` 及 `bash build.sh <topic>`（如 `bash build.sh package`）会打印该 topic 的专属帮助。
- `.clang-format`（style `file`）约束 C/C++ 格式；pre-commit 通过 clang-format v18 强制。修改 kernel 时遵循周边风格。
- pre-commit 钩子（`.pre-commit-config.yaml`）：trailing-whitespace、end-of-file-fixer、clang-format（C/C++）、ruff check+format（Python）、codespell，以及本地 OAT license 合规检查（`scripts/oat_check.sh`）。codespell 设有领域专有白名单（`CANN`、`ascend`、`EnQue`、`CopyIn`、`ArchType` 等）—— 需要时优先扩展白名单，而非禁用。
- Python 依赖（`requirements.txt`）：`numpy<2.0`、`jinja2`、`sympy`、`scipy` 等。`install_deps.sh` 安装构建工具链依赖。
- `3rdparty/catlass` 是 git 子模块（matmul 辅助库）。若 kernel 构建报缺失 `3rdparty/` 下头文件，执行 `git submodule update --init`。

## 关键文档

- `docs/QUICKSTART.md` —— 用 `add_example` 从零到运行。
- `docs/zh/develop/aicore_develop_guide.md` —— 算子开发指南（最小交付件、Tiling、kernel）。改动 `op_host`/`op_kernel` 前必读。
- `docs/zh/develop/aicpu_develop_guide.md`、`graph_develop_guide.md`、`cross_platform_migration_guide.md` —— AI CPU、图融合、跨平台迁移。
- `docs/zh/debug/cann_sim.md` —— CANN Simulator（无需 NPU 调试 kernel，`--simulator`）。
- `docs/zh/install/dir_structure.md` —— 目录结构权威参考。
- `CONTRIBUTING.md` —— PR 流程、CI 门禁（评论 `compile` 触发：编译、codecheck、UT、冒烟）、最简算子与标准算子交付件要求。
