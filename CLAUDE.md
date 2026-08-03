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

# 仿真模式（无需真实 NPU，需配合 --soc）；camodel 仿真当前仅 Ascend950 支持
bash build.sh --run_example <op_name> eager --simulator=camodel --soc=ascend950

# 只改 tiling、不重编 kernel 二进制的打包方式（JIT 包）
bash build.sh --jit --soc=ascend910b --ops=<op_name> -j16

# experimental 算子需要该开关
bash build.sh --pkg --experimental --soc=ascend910b

# 脚手架：新建一个算子目录
bash build.sh --genop <op_name>            # 或 --genop=<module>/<op_name>
bash build.sh --genop_aicpu <op_name>      # AI CPU 变体

# 清理构建产物
bash build.sh --make_clean
```

`--noexec` 只编译 UT 不执行。`--cov` / `--asan` / `--valgrind` / `--oom` 分别开启覆盖率、AddressSanitizer（host 侧）、valgrind、kernel 侧 OOM。`--ops=` / `--module=` 用逗号分隔；合法 module：`mc2,attention,moe,ffn,mhc,posembedding,gmm,mamba`。

**`--pkg` 行为**：不带 `--ops`/`--module` 时编 built-in 包；带 `--ops`/`--module` 时编 custom vendor 包。默认 vendor 名 `custom`（产物装到 `$ASCEND_HOME_PATH/opp/vendors/custom_transformer/`），用 `--vendor_name=<name>` 覆盖。`--module=mc2` 会自动并入 `gmm`。CMake `ENABLE_CCACHE` 默认 ON（自动探测 ccache）。`--ops` 用 snake_case 算子名。`torch_extension/` PyTorch 扩展随 `--pkg` 自动编译。

### 安装与运行构建产物

```bash
./build_out/cann-ops-transformer-*linux*.run     # 安装到 $ASCEND_HOME_PATH/opp/vendors/<vendor>
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
```

### Python pytest

各算子自带 `<op>/tests/pytest/`，含 `pytest.ini` 与 `conftest.py`（提供 `golden_mode`、`cache_dir` 等 fixture），测试参数在 `*_paramset_*.py` 里。CI 测试配置见 `tests/test_config.yaml`。

```bash
pytest <op>/tests/pytest/ -k "test_name"     # 直接跑某算子的 python 用例
ASCEND_DEVICE_ID=3 pytest ...                 # 指定 NPU device（默认 0）
```

**测试隔离陷阱**：kernel 崩溃会毒化同会话后续用例——失败用例单独跑（`pytest -k "specific_test_name"`），崩后先 `npu-smi set -t reset -i <device_id>` 或等设备恢复再继续。部分用例用 `ThreadPoolExecutor(max_workers=1)`，会掩盖真实异常，要看完整 traceback。

### Kernel 调试陷阱：PRINTF / DumpTensor（重要）

AscendC `PRINTF`（大写，NPU 模式）和 `DumpTensor` 在真机上把数据写进 kernel workspace 里一块预留的 debug 缓冲。**普通 `build.sh --pkg` 不会分配这块缓冲**——若 kernel 里调用了 PRINTF/DumpTensor，运行时写未映射地址 → AICore error 507015（device error type 3），原本能过的用例也会崩，且无任何输出。

- 想用 PRINTF/DumpTensor 调试：必须 `--debug` 构建（`ENABLE_DEBUG=ON`，分配 debug 缓冲），可选 `--ops-compile-options=-DDEBUG_DUMP` 配合 `#ifdef DEBUG_DUMP` 守卫 PRINTF 源码。注意 `--debug` 会改 kernel 时序（-O0），可能掩盖精度 bug。
- `-DDEBUG_DUMP` 单独用**不够**：它只是把 PRINTF 编进去，不分配缓冲，仍崩。要同时 `--debug`。
- 小写 `printf` 仅 CPU 仿真（`--simulator`）可用，真机不识别。
- 改完 kernel 务必 `rm -rf build/`（kernel 二进制缓存会让旧 bin 混进来）。
- 精度调试优先用 `ascendc-precision-debug` skill（DumpTensor 7 步法 / printf 调试法）。

### 运行时调试与排错

- **DumpTensor 运行期开关**：`export ASCENDC_DUMP=1` 开 / `=0` 关（仍需 `--debug` 构建才分配缓冲，见上）。
- **plog 定位实际加载的 kernel**：`grep "binPath.*<OpName>" $HOME/ascend/log/debug/plog/plog-*.log | head -5`。若路径是 `built-in` 而非 `vendors/custom_transformer`，说明 custom 包没装或没生效。
- **CANN built-in kernel 覆盖 vendor kernel**：同 hash 时 CANN 优先加载 built-in。custom 包改了 kernel 但行为没变时，删掉冲突的 built-in `.o`/`.json`（`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/<soc>/<op_dir>/`，**先备份**），这是每次 CANN 安装的一次性操作。
- **kernel 缓存导致旧二进制**：除 `rm -rf build/` 外，清 `$HOME/atc_data/kernel_cache/` 等缓存目录。
- **设备状态跨崩溃持久**：kernel 崩溃（507015 aicore timeout）会把设备留脏，后续用例连带失败。`npu-smi set -t reset -i <device_id>` 重置或等恢复。
- **常见错误码**：`507015` = aicore timeout（kernel 挂死/非法内存，常因同步或越界）；`507014` = aicore error（内存访问违例/非法指令）。

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

## CLAUDE.md vs AGENTS.md

`CLAUDE.md`（本文件）是权威单一源：build/test、目录结构、kernel 模型、FIA 流水架构、运行时调试，以及下文「D256 活跃调试日志」都已收录于此。`AGENTS.md` 是历史镜像，内容可能滞后；两份冲突时以 `CLAUDE.md` 为准。调试状态易变（每次实验后更新），改实验记录时更新本文件即可，不必再维护 AGENTS.md。

## FIA MXFP8 arch35 split-core 流水（当前工作焦点）

`attention/common/op_kernel/arch35/fia_*fullquant_mx*` 是 FIA（fused_infer_attention_score）MXFP8 量化路径，A5（ascend950）专属。这是当前主要调试对象。理解它需要跨 host/kernel 多个文件，下面是架构要点。**当前正在追的 D256 T=32 边界 bug 的实验记录见下文「D256 活跃调试日志」一节。**

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

## D256 活跃调试日志（MxFP8 Prefill）

> 本节状态易变，记录当前 D256 边界 bug 的调查进度。调查前先读这里，避免重做已验证无效的二分。

**状态（2026-08-03）**：✅ **D256 KVS128 已 100% pass**。V1（`x_exp_1=32`，`6dd2100ad`，25%→25.18%）+ V2（P UB→L1 DataCopy stride 翻倍，`14185b26e`，25.18%→100%）两步修复完成。下方历史调查记录保留供回溯，但「剩余 T=32 边界 bug」**已解决**。

**修复链**：
- V1（softmax 侧）：`6dd2100ad` — MxFP8 fp8 路径 `x_exp_1` 从 `ubN*4` 改回 `32`，修 P 数据 gap，25%→25.18%，T=0..31 全对。
- V2（P UB→L1 stride）：`14185b26e` — `fia_block_vec_fullquant_mx.h` P DataCopy 三处配套（`strideMul`/`subBlk1Mul`/`srcOffset`），D256 stride 翻倍让 P 落到 0x2000 grid，25.18%→100%。移植自 A5 `cann/ops-transformer` `d256-debug` 仓 commit `d3d78567`（该仓 KVS128 100%）。

**测试用例**：`TND_B1_QS128_KVS128_Nq8_Nkv8_D256_SP0`，位于 `attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test/`。A5（116 容器 `wanjian-docker`，ascend950）实测 PASSED。

**历史错误模式**（V1 修复后、V2 修复前）：
- T=0..31：全对（65536 元素，first_fail_idx=65536）
- T=32..127：全错（axis0 range=[32,127]，96/128 errors）
- 8 个 head 全受影响，256 个 D 维全受影响

### 二分法定界（DumpTensor + sentinel）

**方法**：在阶段边界用 DumpTensor（desc=10001/20001/30001）+ sentinel 填充（`Duplicate<half>(1000.0)`）探测未写入的 UB 位置。

阶段边界：
```
C1(AIC) → [BMM1结果 UB] → V1(AIV) → [P数据 UB] → C2(AIC) → [BMM2结果 UB] → V2(AIV) → [输出]
           desc=20001❌        desc=10001✅       desc=30001✅
           (float没dump出)    (fp8,8192B)       (float,256B)
```

**V1 bug（已修）**：VF softmax StoreAlign 只把 P 数据写到了一半 NZ 位置（每 128 元素行：32 非零 + 32 零交替）。根因：MxFP8 fp8 路径 `x_exp_1 = x_exp + (ubN * 4)` 应为 `x_exp + 32`。第二次 StoreAlign 写到了 512 而非 32，奇数 32-block 留零，BMM2 读到零导致输出错乱。修复：`vf_mul_sel_softmaxflashv2_cast_nz_dn.h` 行 880、1285（MxFP8 NoUpdate/Update fp8 分支），commit `6dd2100ad`。

> 注：扩展分支 `21a3ac09a` 曾有一个"revert"提交（`ef4424a03`）想把 x_exp_1 改回 32，但 diff 方向反了（重新应用了 ubN*4 而非回退到 32），修复直到现在才真正落地。

**C2 bug（已修，见 V2）**：T=32 边界错。P 数据已确认正确（fp8 roundtrip 后 5792/8192 与 golden P 匹配，其余 2400 个零是 NZ padding，DataCopy 不读）。**真因不在 `FixpipeMm2` `dualDstCtl=1`**——dump 三证合一（行错位 + 50/50 零 + axis0 range=[32,127]）指向的"dualDstCtl 第二目标落点偏 32 行"是**症状非根因**。真因是 P 从 UB 拷到 L1 时落到 0x1000 grid（应为 0x2000，LoadData 读 L1 用固定 0x2000 stride），BMM2 从错误位置读 P。修复 `14185b26e`（V2）。

#### dump 实测结论（2026-07-28，A5 @ bc86ac703，25.18% 构建）

对 16 个 AIV block 的 `desc=30001`（BMM2 fixpipe 后首 row，256 float）统计零/非零，得到干净的 50/50：

```
block= 0,2,4,6,8,10,12,14 (偶=subBlock0)  → 256/256 非零（满）
block= 1,3,5,7,9,11,13,15 (奇=subBlock1)  →   0/256 非零（全零）
```

16 block = 8 heads × 2 subBlock，偶=subBlock0（处理 Sq 0-63）、奇=subBlock1（处理 Sq 64-127）。**subBlock1 的 BMM2 结果 UB 从未被写入（全零）**，subBlock0 首 row 正确。结合测试 row 0-31 对、32-127 错：`dualDstCtl=1` 的 M-split 第二半（M/2）**落进了 subBlock0 的 UB（覆盖 row 32-63 区），而不是 subBlock1 的 UB**。两者叠加 = row 0-31 对、32-127 错，与 axis0 range=[32,127] 吻合。**T=32 的真身是 M-split 第二目标落点偏了 32 行**，与 S1 tiling / actMSize 收缩无关。

#### 行错位实锤（2026-07-28，读已保存的 npu/cpu 输出，无重新构建）

直接比对 25.18% 那次跑保存的 `golden_cache/..._cpu_output.pt`（`cpu_out` [1,8,128,256]）与 `..._npu_output.pt`（`atten_out` [128,8,256]，TND）。对每个 npu Sq 行找 L2 最匹配的 cpu 行：

```
npu[ 0..31] -> cpu[ 0..31]   dist=0.0000        ← 完全正确
npu[32..63] -> cpu[64..95]   dist≈0 (fp16 噪声)  ← 第二半 rows 64-95 错位 +32 落到 rows 32-63
npu[64..127] -> 全零          (best-match 都是 cpu[39]，dist 恒定)  ← subBlock1 饿死
```

精确诊断：cube 的 `dualDstCtl=1` M-split 第二目标写到 **subBlock0 缓冲区内 offset = 8192 元素（= 32 行 = mm2ResultSize/2）**处——覆盖 subBlock0 rows 32-63（写入第二半的 rows 64-95），第二半的 rows 96-127 溢出丢失；subBlock1 缓冲区从未被写 → rows 64-127 全零。第二目标本应跨到 subBlock1 的 UB，却落到 subBlock0 内部半偏移。**根因实锤，三方吻合**（行错位 + dump 50/50 零 + axis0 range=[32,127]）。

> 复现脚本：A5 容器内 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3`，load 两个 .pt（dict：cpu `cpu_out`/`cpu_lse`、npu `atten_out`/`lse_out`），逐行 L2 匹配。torch_npu 自动加载会炸 sqlite3，必须 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。

#### 关键 narrowing：bug 在 DN 专属路径，noquant 不能用来洗清参数

⚠️ **noquant 不是有效的对照**：noquant 的 `IterateBmm2`（`fia_block_cube_noquant_gqa.h:496`）→ `IterateBmm2l0Split`，**noquant 根本没有 DN 路径**（无 `USE_DN`）。noquant D256 是靠 **ND l0Split**（`MatmulN`、单 matmul、无 K-split）跑通的，与 MX prefill 的 `IterateBmm2Dn`（`MatmulFullMX` + K-split + `FixpipeMm2` `dualDstCtl=1`）是**不同代码路径**。虽然两者 fixpipe 参数（nSize=256/mSize=128/srcStride=128/dstStride=256/dualDstCtl=1/ndNum=1）在满 tile 下相同，但 noquant 在 ND l0Split 下用、MX 在 DN K-split 下用——不能据"noquant 能跑"排除这组参数在 DN 路径下的问题。

**D128 vs D256 都走同一条 MX DN 路径**（`BMM2_TOUB` 对 prefill 恒 true：`GetC2Position` 因 `optionalDn=useDn=true` 直接返回 VECCALC，与 D 无关），D128 能跑、D256 挂。差异仅在 nSize/dstStride（128 vs 256）、L0B 单/双缓冲、s2BaseSize/s2SplitSize。dump 症状（subBlock1 饿死 + subBlock0 row 32-63 被覆盖）指向 `dualDstCtl=1` 第二目标 offset 落在 32×256 而非 64×256——一个 2× 偏移错误，仅在 nSize≠mBaseSize（D256）时暴露。失败尝试 #6（srcStride→nSize）、#7/#8（ndNum/M-split）改的是 fixpipe 参数，**未触及 dualDstCtl 第二目标 offset**（可能是硬件固定或别处设定），故均无效。

#### 已证伪的假设（勿再试）

- ❌ **「cube `mSize=mBaseSize` vs vec `actVecMSize` 在 T=32 失配」**：核实 `CalcParams`（`fia_kernel_fullquant_mx.h:601-657`），本用例 `gS1Size=128*1=128`、单 S1 tile、`actMSize=mBaseSize=128`、`actVecMSize=64`、`dualDstCtl=1`（M/2=64+64），cube/vec 完全自洽，无失配。把 DN fixpipe `mSize/srcStride` 改成 `actMSize` 是无效改动（actMSize 本就等于 mBaseSize）。

### D128 vs D256 tiling 差异

| 参数 | D=128（正常） | D=256（异常） |
|------|--------------|----------------|
| sInnerFactor_ | SINNER_512 | SINNER_256 |
| s2BaseSize | 512 | 256 |
| s2BaseSizeCur (ubN) | 256 | 128 |
| s2SplitSize | 256 | 128 |
| blockStride (StoreAlign) | 65 | 33 |
| nSize (FixpipeMm2) | 128 | 256 |
| mBaseSize | 128 | 128 |
| srcStride (FixpipeMm2) | 128 | 128 |

D=128 能过是因为 `nSize == mBaseSize == 128`（巧合）；D=256 `nSize=256 ≠ mBaseSize=128`。

### 已尝试的修复（均回退，除非注明）

| # | 尝试 | 结果 | Commit |
|---|------|------|--------|
| 1 | splitD mode | L0C 够大，不需要 | `cf9f28019`→`27f18af48` |
| 2 | BMM2 DN N-split | 回到 K-split | `d17b29f50`→`27f18af48` |
| 3 | PScale dual copy stride | 部分保留 | `2f94fb881` |
| 4 | P 数据 UB→L1 srcOffset 硬编 65 | 无效（65=mBaseSize/2+1，M 派生，正确） | `0a1a5046c`→`f5135c65a` |
| 5 | P store 第二组 offset x_exp+(ubN/8) | 崩 | `77c9e0a28`→`93de962ae` |
| 6 | FixpipeMm2 srcStride mBaseSize→nSize | 更差（25%→1.66%） | `30ac6eee8`→`fed381273` |
| 7 | BMM2 M-half split（试过两次） | 更差（25%→1%） | `86f1754a0`→`5380aca4e`、`0a6def89e`→reverted |
| 8 | BMM2 ndNum=2 N-split | 更差（25%→1%） | `41a5fb07d`→`eec05cc07` |
| 9 | D-gate 止血（D256 禁用 DN） | host/kernel 不匹配，D256 全 0 | `8f9f9d83c`→`e8309ca24` |
| 10 | T=32 dump 探针 | PRINTF/DumpTensor 触发 AICore error | `f5999f1b4`→`c1ecf82c7` |
| 11 | x_exp_1 ubN*4→32（MxFP8 路径） | **修复 V1，25%→25.18%**（保留） | `6dd2100ad` |
| 12 | srcOffset 65→vec1BlockStride | 更差（25%→23.27%，65 是 M 派生非 N 派生） | `0522bccac`→`5f212e89e` |
| 13 | bmm2Buffers SingleBuffer→DB + 2× UB + 双 Set/WaitCrossCore（对齐 noquant） | **507015 崩**（健康设备 0 稳定复现，非探针/非瞬态；回退后回 25.18% 不崩，对照确证 DB 是崩因）。DB 乒乓与 MX PRELOAD 同步冲突或 UB 溢出。"分配/2"假设未验证 | `8fa0bbc8b`→`09b799b5d` |
| 14 | P UB→L1 DataCopy stride 翻倍（`strideMul`/`subBlk1Mul`/`srcOffset` 三处配套） | **修复 V2，25.18%→100%**（保留）。移植自 A5 `cann/ops-transformer` `d256-debug` 仓 commit `d3d78567`。与 #54 区别：#54 只改 srcOffset 65→33 致 23.27%（更差），本处三处一起改方生效 | `14185b26e` |

### DumpTensor 探针（已移除）

探针（`desc=10001/20001/30001`）已于 commit `93704dd84` 从 `fia_block_vec_fullquant_mx.h` 移除——定位使命完成（V1 x_exp_1、C2 dualDstCtl 落点），且非 `--debug` 构建会触发 507015。需要时重新加回，注意：源码无 `#ifdef DEBUG_DUMP` 守卫（无条件编进），真机运行需 `--debug` 构建分配缓冲，否则 507015；`desc=20001` 当年"float dump 不出"实为 dumpSize 过大（8192），非 float/L0C 不支持。

### 关键文件（调试）

- `attention/common/op_kernel/arch35/fia_block_cube_fullquant_mx.h` —— IterateBmm2Dn、FixpipeMm2（C2 bug 位置）
- `attention/common/op_kernel/arch35/fia_block_vec_fullquant_mx.h` —— ProcessVec1Dn（V1 已修）、ProcessVec2OnUb、**P copy UB→L1（V2 修复 `14185b26e`，`strideMul`/`subBlk1Mul`/`srcOffset` 三处）**、DumpTensor 探针
- `attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h` —— VF softmax、x_exp_1 offset（V1 修复）、StoreAlign 写入模式
- `attention/common/op_kernel/arch35/fia_template_dispatcher.h` —— 模板分发、useDn、bmm2Write2Ub、D-gate 位置
- `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp:1246` —— `sInnerFactor_ = (vHeadDim==DSIZE_256) ? SINNER_256 : SINNER_512`

### 已解决（2026-08-03）

D256 KVS128 MxFP8 prefill **100% pass**。根因 = P 从 UB 拷到 L1 时落到 0x1000 grid（应为 0x2000，LoadData 读 L1 用固定 0x2000 stride），BMM2 从错误位置读 P。`dualDstCtl=1` 第二目标落点偏移是此根因的症状。修复 `14185b26e`（V2，移植自 A5 `d3d78567`）。原「下一步」1-4（查 dualDstCtl offset / dump L0C / 咨询华为 / D-gate）随根因解决撤销。

### 残留待办（非阻塞）

1. **x_exp_1 非 MxFP8 collateral**（见记忆 `d256-x-exp-1-nonmxfp8-collateral`）——`vf_mul_sel_softmaxflashv2_cast_nz_dn.h` 7 个非 MxFP8 VF 函数 fp8 分支被 #48/#50 错改为 `ubN*4` 未恢复（原始 `32`）。D256 MxFP8 prefill 不受影响（走 MxFP8 函数已修），但非 MxFP8 量化路径（per-token/causal/…）T2 为 fp8 时会复现 V1 bug。当前测试集未覆盖故未暴露。
2. **D256 KVS>128（多子轮）/ decode / SP3 / D64 D128 回归**——V2 修在单子轮 SP0，多子轮与 decode 路径未验证；D128 数值等价（`strideMul`=32/`subBlk1Mul`=1/`srcOffset`=65<<5 与原公式相同）理论无回归但需跑确认。
3. **A5 d256-debug 仓未提交改动**——`subLoopOffset` 从 `s2BaseSizeCur*mBaseSize` 改 `s2BaseSize*mBaseSize`（D256: 16384→32768/subLoop），我们未移植；多子轮场景若 100% 失败可能需此行。

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
- `.clang-format`（style `file`）约束 C/C++ 格式：ColumnLimit 120、IndentWidth 4、PointerAlignment Right、BreakBeforeBraces Custom（AfterFunction true）；pre-commit 通过 clang-format v18 强制。修改 kernel 时遵循周边风格。本地跑：`pre-commit run --all-files`。
- pre-commit 钩子（`.pre-commit-config.yaml`）：trailing-whitespace、end-of-file-fixer、check-yaml、clang-format（C/C++）、ruff check+format（Python）、codespell，以及本地 OAT license 合规检查（`scripts/oat_check.sh`）。codespell 跳过 `.py/.cpp/.hpp/.c/.h/.md`，并设有领域专有白名单（`CANN`、`ascend`、`EnQue`、`CopyIn`、`ArchType` 等）—— 需要时优先扩展白名单，而非禁用。
- Python 依赖（`requirements.txt`）：`numpy<2.0`、`jinja2`、`sympy`、`scipy` 等。`install_deps.sh` 安装构建工具链依赖。
- `3rdparty/catlass` 是 git 子模块（matmul 辅助库）。若 kernel 构建报缺失 `3rdparty/` 下头文件，执行 `git submodule update --init`。

## 关键文档

- `docs/QUICKSTART.md` —— 用 `add_example` 从零到运行。
- `docs/zh/develop/aicore_develop_guide.md` —— 算子开发指南（最小交付件、Tiling、kernel）。改动 `op_host`/`op_kernel` 前必读。
- `docs/zh/develop/aicpu_develop_guide.md`、`graph_develop_guide.md`、`cross_platform_migration_guide.md` —— AI CPU、图融合、跨平台迁移。
- `docs/zh/debug/cann_sim.md` —— CANN Simulator（无需 NPU 调试 kernel，`--simulator`）。
- `docs/zh/install/dir_structure.md` —— 目录结构权威参考。
- `CONTRIBUTING.md` —— PR 流程、CI 门禁（评论 `compile` 触发：编译、codecheck、UT、冒烟）、最简算子与标准算子交付件要求。
