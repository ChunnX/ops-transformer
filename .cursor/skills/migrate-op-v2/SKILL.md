---
name: migrate-op-v2
description: >-
  NPUKernelBench 单算子迁移（AscendC + msopgen）：从 CANN 开源仓（ops-nn/cv/math/transformer）定位算子源码，按七阶段工作流写入 tasks/level{N}/{Category}/{OpName}/ 的 answer/0、validation、question、api_desc.md，并在 CANN 9.0.0 + torch-npu（ascend910b / ascend910_93）上完成 compile 与 precision。
  在用户说「迁移某算子」「把 Op 迁到 NPUKernelBench」「/migrate-op」「单算子迁移」「补 validation / 修精度」或给出 OpName 并要求源码定位、工程适配、编译验证、精度测试时使用；详细步骤见正文，细则见同目录 references/ 与 templates/。
disable-model-invocation: false
---
## 固定上下文

无论你的任务是什么算子，以下信息不变：

| 项目 | 值 |
|------|-----|
| **工作仓库** | `/home/xxp/NKB-Agent-Workspace/NPUKernelBench` |
| **源算子仓库** | `/home/xxp/NKB-Agent-Workspace/ops-transformer-dev` |
| **目标平台** | CANN 9.0.0 + torch-npu，芯片 `ascend910b` 和 `ascend910_93` |
| **目标目录模式** | `tasks/level{N}/{Category}/{OpName}/` |
| **编译模式** | `kernel_only_mode: True`、`compile.mode: msopgen` |
| **编译入口** | `python scripts/run_test.py -task_name <OpName> -stages compile` |
| **单用例精度验证入口** | `python scripts/run_test.py -task_name <OpName> -stages precision -n_case 1` |
| **精度验证入口** | `python scripts/run_test.py -task_name <OpName> -stages precision` |
| **容器内一键执行（非交互）** | `docker exec -u root <容器名> bash -lc 'cd /home/xxp/NKB-Agent-Workspace/NPUKernelBench && python3 scripts/run_test.py -task_name <OpName> -stages compile precision -n_case 1'`（勿依赖 `docker exec -it` 在无 TTY 环境） |

> 注意：`runs/` 目录下的路径使用缩写 `lvl{N}`（如 `runs/lvl1/...`），而 `tasks/` 目录使用 `level{N}`（如 `tasks/level1/...`）。

### 减少迭代：任务已存在时先做「体检」

若 `tasks/level{N}/.../<OpName>/` **已存在**（续跑、补编译、修精度），不要从零重复阶段 1–2，按下表**只读排查**后再改：

| 检查 | 命令 / 动作 | 目的 |
|------|-------------|------|
| arch35 / _apt | `rg -n 'arch35|_apt\\.cpp' tasks/.../<OpName>/answer/0` | 未删干净则阶段 4 一次性清掉（见下节） |
| 310 分支 | `rg -n "__CCE_AICORE__\\s*==\\s*310" tasks/.../<OpName>/answer/0/op_kernel` | 目标仅为 910B 时，可收敛为 `#else` 路径，避免残留 `#include arch35` |
| validation | 读 `prepare_inputs.get_inputs` 与 `module.ModelNew` 签名、`test_cases.csv` 列名 | 避免 CSV/占位输出与 forward 不一致 |
| 精度日志 | `runs/.../<OpName>/.../case_*/log/*precision.log` 中 REF vs TEST | 区分「真不一致」与「阈值过严」（见阶段 7） |

## 核心概念：理解你在做什么

### 为什么需要迁移

CANN 开源仓（`ops-nn` 等）中的算子是在一个大型多算子 CMake 工程中编译的，依赖公共模块、公共 tiling registry、公共日志/错误头文件。迁移的目标是将**单个算子**变成一个**自包含的独立编译单元**，放入 NPUKernelBench 的 msopgen 框架中。

### 两种框架的区别

**源框架（CANN 开源仓）：**
```
ops-nn/ops/operator_name/
├── op_host/
│   ├── *_def.cpp          # OpDef 定义
│   ├── *_tiling.cpp       # tiling 函数实现
│   ├── *_tiling.h         # TilingData 结构定义
│   └── *_infershape.cpp   # shape/type 推导
├── op_kernel/
│   ├── *.cpp / *.h        # kernel 实现（可能有多个架构分支）
│   └── arch35/            # 非目标架构代码（要删除）
├── op_api/                # aclnn API 手写实现（禁止导入）
├── tests/                 # 测试（禁止导入）
└── CMakeLists.txt         # 依赖公共 cmake 函数（要替换）
```

编译时依赖仓库根 `CMakeLists.txt` 提供的公共函数和公共 target（`nnopbase`、`exe_graph` 等），以及公共头文件（`src/common/inc/`）。

**目标框架（NPUKernelBench msopgen）：**
```
tasks/level{N}/{Category}/{OpName}/
├── api_desc.md                    # API 中文说明（与 answer 同级）
├── question/                      # LLM 输入模板（空壳）
│   ├── op_host/<kernel_name>.cpp  # 占位 tiling + 空 OpDef
│   └── op_kernel/<kernel_name>.cpp # 空 kernel 函数体
├── answer/0/                      # 参考实现（你要创建的）
│   ├── CMakeLists.txt             # 仅复杂算子需要，覆盖 msopgen 根 CMake
│   ├── op_host/
│   │   ├── CMakeLists.txt         # 多文件/子目录时需要
│   │   ├── <kernel_name>.cpp      # tiling 函数 + OpDef（简单算子合并）
│   │   ├── <kernel_name>_tiling.h # TilingData 结构
│   │   ├── <kernel_name>_def.cpp  # OpDef（复杂算子分文件）
│   │   ├── <kernel_name>_tiling.cpp # tiling 实现（复杂算子分文件）
│   │   ├── *_infershape.cpp       # shape 推导
│   │   ├── common/                # 本地化日志/工具头文件
│   │   └── tiling_base/           # 本地化 tiling registry（复杂算子）
│   └── op_kernel/
│       └── <kernel_name>.cpp      # kernel 实现 + 本地化的 include 文件
└── validation/                    # 测试验证体系
    ├── pybind.cpp                 # C++ Python 绑定
    ├── module.py                  # PyTorch 参考实现 + 自定义算子调用
    ├── prepare_inputs.py          # 测试输入生成
    └── test_cases.csv             # 测试用例（分号分隔）
```

### 编译过程（框架自动完成）

框架在编译前自动执行以下复制和渲染：

```
1. copy libs/common_include/msopgen/ → runs/.../src/      （编译脚手架）
2. copy tasks/.../validation/       → runs/.../src/validation/ （测试文件）
3. copy answer/0/                   → runs/.../src/             （算子代码）
4. render pybind.cpp：aclnnCustomOp → aclnn<OpName>，custom_pybind_api → <kernel_name>
```

这意味着：
- `answer/0/op_host/*` 覆盖到 `runs/.../src/op_host/*`
- `answer/0/op_kernel/*` 覆盖到 `runs/.../src/op_kernel/*`
- `answer/0/CMakeLists.txt`（如果存在）覆盖到 `runs/.../src/CMakeLists.txt`
- `answer/0/op_host/CMakeLists.txt`（如果存在）覆盖默认的 `op_host/CMakeLists.txt`

**你只需要维护 `tasks/.../answer/0/` 下的文件，不要直接修改 `runs/.../src/`。**

### Tiling 系统简介

Tiling 是 AscendC 算子的核心机制，在 **host（CPU）上运行**，负责：
1. 获取平台信息（UB 大小、AI 核心数量）
2. 获取输入张量信息（shape、dtype）
3. 计算分块参数（每个核心处理多少数据、分多少块）
4. 设置 block dim（启动多少 AI 核心）
5. 通过 `SaveToBuffer` 将参数打包写入 raw tiling data
6. 设置 workspace 大小

**TilingData 是 host 和 kernel 之间的 ABI（二进制接口）**：字段的类型、顺序、数组长度必须在 host（`_tiling.h`）和 device（`GET_TILING_DATA` 后读取）之间完全一致。

Kernel（device 侧）在 NPU 上运行，通过 `GET_TILING_DATA(tiling_data, tiling)` 解包 tiling 参数，使用这些参数在 UB 上分块循环处理数据。

在 msopgen 中：
- 默认 `op_host/CMakeLists.txt` 使用 `aux_source_directory` 收集当前目录下所有 `.cpp`，`opbuild()` 从 OpDef 自动生成 aclnn API，`cust_optiling` 编译 tiling 逻辑
- `op_kernel/CMakeLists.txt` 使用 `add_kernels_compile()` 编译所有 kernel 文件
- 编译产物包括 `libcust_opapi.so`、`liboptiling.so`、`libcust_opmaster_rt2.0.so` 和 `kernel_gen_ops.*.so`（完整列表见 `references/01-framework.md`）

**关键约束：aclnn 直调路径不使用 TilingParse。** NPUKernelBench 通过 pybind → aclnn 调用算子，走的是 aclnn 直调路径（而非图编译路径）。在此路径下，`TilingParse` 回调**不会被调用**，因此：
- `context->GetCompileInfo<T>()` 始终返回 nullptr，**不要使用 CompileInfo 模式**
- 平台信息（UB 大小、核心数等）直接从 runtime context 获取：`platform_ascendc::PlatformAscendC(context->GetPlatformInfo())`
- 删除源仓中的 `TilingParse` 函数和 `.TilingParse<CompileInfo>(...)` 注册，这些代码在 aclnn 路径下不会被执行

### 精度验证过程（框架自动完成）

```
1. 设置环境变量（ASCEND_CUSTOM_OPP_PATH, LD_LIBRARY_PATH 等）
2. 在独立子进程中加载编译产物（kernel_gen_ops.*.so）
3. 读取 test_cases.csv 获取测试用例
4. 对每个用例：
   a. prepare_inputs.py 生成输入张量
   b. Model (PyTorch CPU) 执行得到参考输出
   c. ModelNew (调用 kernel_gen_ops) 在 NPU 上执行
   d. 比较两个输出的逐元素误差
```

## 迁移工作流

按顺序执行以下 7 个阶段。每完成一个阶段，通过该阶段的检查点再继续。

### 阶段 1：确定算子身份

**OpName、Level、Category 必须由用户提供**。如果用户未给出完整信息，先向用户确认：

- `OpName` = PascalCase（如 `GatherV3`），这同时也是 `OP_ADD(OpName)` 中使用的名字
- `kernel_name` = `OpName` 转 snake_case（如 `gather_v3`）
- `Category` = 算子所属分类（如 `Index`、`Activation`、`MatMul`、`Moe`）
- `Level` = 任务等级（1/2/3），参考：简单逐元素计算→1，中等复杂度（含 reduce/索引/量化）→2，复杂（多策略 tiling/大 kernel）→3

如果用户未指定 Level，可先继续定位源目录并完成阶段 3 的模式识别，再根据模式向用户确认 Level（模式 1 通常→Level 1，模式 2→Level 2，模式 3/4→Level 3），确定后再创建目标目录。

定位源算子目录：

1. 根据 `OpName` 推断源目录名（可能是 PascalCase 也可能是 snake_case，需两种都尝试）
2. 在四个源仓中搜索：
   ```bash
   find /home/xxp/NKB-Agent-Workspace/ops-transformer-dev -type d -name '<OpName>' -o -type d -name '<kernel_name>' -print
   ```
3. 优先选择同时包含 `op_host/` 和 `op_kernel/` 的目录

检查源目录中是否有 910B/910_93 可用的实现。如果只有 `arch35/` 或 `*_apt.cpp` 入口，进入阶段 3 的模式 5（阻塞情况）。

**检查点：** `OpName`、`kernel_name`、`Level`、`Category` 已确认，源目录已定位。

### 阶段 2：初始化目标目录

创建目标目录结构：

```bash
mkdir -p tasks/level{Level}/{Category}/{OpName}/{question,answer/0,validation}
mkdir -p tasks/level{Level}/{Category}/{OpName}/question/{op_host,op_kernel}
mkdir -p tasks/level{Level}/{Category}/{OpName}/answer/0/{op_host,op_kernel}
# 如果是模式 3（复杂 registry），也预创建子目录：
mkdir -p tasks/level{Level}/{Category}/{OpName}/answer/0/op_host/{common,tiling_base}
```

**检查点：** 目录结构存在。

### 阶段 3：识别迁移模式

**第 1 步 — 架构可用性检查（最先执行，避免在不可迁移的算子上浪费时间）：**

```bash
# 检查源算子是否有 910B/910_93 可用实现
rg -n "ascend910b|ascend910_93|membase" op_host op_kernel | head -10
find op_kernel -name "arch35" -type d
find op_kernel -name "*_apt.cpp"
# 找非 arch35 且非 _apt 的 kernel .cpp 文件（_apt.cpp 是 910_95 入口，不可用）
find op_kernel -name "*.cpp" ! -path "*/arch35/*" ! -name "*_apt.cpp"
```

如果源算子的 `op_kernel/` 中**所有** kernel 实现都在 `arch35/` 目录下，且没有 membase/910B 实现 → **模式 5：阻塞**。不要强行迁移 arch35 代码。先去其他三个源仓寻找同名算子的 910B 版本，找不到则报告阻塞，停止后续步骤。

**第 2 步 — 模式检测（仅当第 1 步排除模式 5 后执行）：**

```bash
# 只检测注册/接口模式，不要混入 arch35/_apt 等架构标记
rg -n "TilingBaseClass|REGISTER_TILING_TEMPLATE" op_host
rg -n "IMPL_OP_OPTILING" op_host
rg -n "TeOpParas|OpRunInfo|nlohmann" op_host
# 辅助检测：提前发现需要处理的 TilingParse/CompileInfo 模式
rg -n "TilingParse|GetCompileInfo" op_host | head -10
```

模式判断的完整决策树见 `references/03-migration-patterns.md` §0（该文件是唯一的权威版本，不要依赖 SKILL.md 中的简化描述做判断）。

| 模式 | 判断特征 | 难度 |
|------|---------|------|
| **模式1: 简单 gert** | `gert::TilingContext*` + `.SetTiling()`，单文件，无 `IMPL_OP_OPTILING` | 低 |
| **模式2a: 分文件 gert** | `*_def.cpp` + `*_tiling.cpp`，用 `.SetTiling()`，无 `IMPL_OP_OPTILING` | 中 |
| **模式2b: IMPL_OP_OPTILING** | 有 `IMPL_OP_OPTILING` 但无 `TilingBaseClass`/`REGISTER_TILING_TEMPLATE` | 中 |
| **模式3: 复杂 registry** | `TilingBaseClass` + `REGISTER_TILING_TEMPLATE` + 多 tiling 策略 | 高 |
| **模式4: 旧接口** | `TeOpParas`、`OpRunInfo`、`nlohmann::json` | 高 |
| **模式5: 阻塞** | 只有 `arch35/` / `*_apt.cpp`，无 910B/membase 实现 | 阻塞 |

### 阶段 4：迁移 answer/0 代码

这是核心步骤，需要根据阶段 3 确定的模式，从源仓导入代码并做适配。

**通用规则（所有模式共享）：**

1. **导入文件：** 将源 `op_host/` 和 `op_kernel/` 内容复制到 `answer/0/` 对应目录
2. **禁止导入：** `op_api/`（aclnn API 由 `opbuild` 自动生成，永远不要手写）、`tests/`、`examples/`、`cmake/`
3. **删除非目标架构：** `arch35/` 目录、`*_apt.cpp` 文件（这是 910_95 入口）、kirin、310P、ascend950 相关代码和配置
4. **只保留芯片配置：** `ascend910b` 和 `ascend910_93`（不要保留 ascend910_95、kirinx90 等）
5. **本地化依赖：** 将 `../` 等相对路径引用的头文件复制到本地，修改 include 为本地路径
6. **TilingData ABI 不动：** 字段类型、顺序、大小不可改
7. **InferShape 规则：** 如果源仓有 `*_infershape.cpp`，保留并适配 include；如果源仓没有，需要检查输出 shape 是否简单等于输入（是则可省略，否则需要手写）

**每种模式的具体步骤、代码示例和检查清单，见：**
- `references/03-migration-patterns.md` — 五种模式的详细处理
- `references/02-tiling.md` — tiling 系统完整参考（含 Dynamic Input、ScalarList、OpAICoreConfig）
- `references/04-cmake-guide.md` — CMake 适配

**检查点：** `answer/0/` 目录下所有必要的源文件就位，无 `op_api/`、`arch35/`、`*_apt.cpp`、非目标芯片配置。

#### arch35 / AIC310 向量头：一次性清理（避免多轮编译才暴露缺文件）

目标芯片为 **ascend910b / ascend910_93** 时，`__CCE_AICORE__ == 310` 分支在产物中**通常不会走**，但头文件若仍 `#include "arch35/..."`，删目录后会直接编译失败。请**同一轮 diff**内做完：

1. `rg -n "arch35|arch35/" answer/0/op_kernel` 列出所有引用。
2. 对每个命中：若仅在 `#if __CCE_AICORE__ == 310` 内使用，**删除整个 310 分支**及对应 `#include`，只保留 `#else`（或非 310）路径；若在 host/comm 头中将某 FP8 类型在非 310 上定义为占位（如 `NoneType`），则依赖 **AIC310 VF** 的 device 函数应保持 **不可达或空实现**，避免引用已删除的 `MicroAPI`/VF 符号导致链接或编译错误。
3. **`arch32/`**（如 `rms_norm.h`）：若源码里原逻辑为「310 用 arch35 VF，否则用 arch32」，删 arch35 后改为**始终 include arch32** 并只保留非 310 实现即可；**不要**把名字里的 `arch32` 误当成「也要删」——技能禁止的是 **arch35** 与非目标芯片，不是凡带 arch 数字的目录。
4. 删除空目录 `answer/0/op_kernel/arch35/`（及子目录），再跑 `compile`。
5. **不要**手改 `runs/.../src/`；以 `tasks/.../answer/0` 为唯一真源。

### 阶段 5：生成 validation 文件

**按以下顺序生成**（文件间有强依赖关系，修改任一文件后需同步检查相关联的其他文件）：

1. **pybind.cpp** — 使用模板，将 `aclnnCustomOp` 和 `custom_pybind_api` 保持为占位符（框架自动渲染）。参数类型需与 answer 的 OpDef 属性一致
2. **module.py** — `ModelNew.forward()` 的参数签名须与 pybind.cpp 一致；推导参数在 `ModelNew.forward()` 内部计算，不出现在 `get_inputs()` 或 CSV 中
3. **prepare_inputs.py** — 返回值须与 Model/ModelNew 的 forward 签名匹配；传给内核**仅作写出缓冲**的张量用 **`torch.zeros(...)`**，勿用 `torch.empty`（未初始化内存会在精度日志里呈现 NaN/垃圾值，易误判为算子写坏）。
4. **test_cases.csv** — 分号分隔，首列 `case_id`，后续列名须与 `prepare_inputs.py` 中 `param['字段名']` 一致，覆盖多种 dtype 和 shape，与 `op_kernel` 中 **`TILING_KEY_IS`**（或等价）分支 **逐一覆盖**；**总用例数建议 10～25**。若 **tiling 分支数 ≤5**：每分支建议 **≥5 例**；若 **分支数 >5**：每分支 **2～3 例** 即可，使总行数仍落在 **10～25**。

详细规则见 `references/05-validation.md`。

**BF16 / 融合算子精度门槛（减少「改阈值—被打回」迭代）：**

- `base_config.yaml` 默认 `eval.max_abs_error` / `max_rel_error` 多为 **1e-3**，对幅值上百、上千的 **BF16 多 MatMul+RMSNorm** 等链路易出现「参考与 NPU 几乎一致仍 FAIL」。
- **流程**：先看 `runs/.../precision.log` 里 REF/TEST 是否同 shape、大误差是否集中在 BF16 量化步；再核对 `cache_index` 展平、`NZ` 权重与 pybind 中 `aclFormat` 是否与金标准一致。
- **仅当**已排除逻辑/顺序错误后，可在 **`prepare_inputs.py` 中实现 `custom_check_precision`**，对 BF16 路径使用与仓库其它 level3 算子同量级的 **atol/rtol**（并在函数注释写明原因）。**禁止**在未做上述核对前单纯放大阈值掩盖真 bug。

**检查点：** 所有四个 validation 文件存在，`pybind.cpp` 中包含占位符（不是实际算子名）。如果算子有多个 tiling key（模式 2b/3），可额外生成 `validation/tiling_cases.csv`（详见 `references/05-validation.md` 第 7 节）。

### 阶段 6：生成 question 和 API 文档

> 以下所有内容以 answer 最终状态（经过阶段 4 适配和阶段 5 可能的修改之后）为准。

1. **question/op_kernel/<kernel_name>.cpp** — 以 answer 的 kernel 函数签名为准，将 `GM_ADDR` 参数列表（含参数名和类型）完整复制到 question，但函数体留空
2. **question/op_host/<kernel_name>.cpp** — 最小的 OpDef 骨架（Input/Output 声明，仅 `ascend910b` 和 `ascend910_93`），tiling 函数返回 `ge::GRAPH_SUCCESS` 并设置 `blockDim=1`
   - 注意：如果 answer 使用 `IMPL_OP_OPTILING`（模式 2b/3），question 的 OpDef 中不写 `.SetTiling()`（模式 2b/3 的 question 是供 LLM 输入的纯文本模板，不需要可编译）
3. **api_desc.md**（与 answer/ 同级）— 中文说明：功能描述、接口定义（`kernel_gen_ops.<kernel_name>(...)`）、参数说明、约束限制、使用示例

**检查点：** question/ 目录完整，api_desc.md 位于算子根目录，描述准确。

### 阶段 7：编译和精度验证

```bash
# 完整编译
python scripts/run_test.py -task_name <OpName> -stages compile

# 如果编译成功，先进行一个用例精度验证
python scripts/run_test.py -task_name <OpName> -stages precision -n_case 1

# 如果一个用例运行成功，运行全量精度验证
python scripts/run_test.py -task_name <OpName> -stages precision

# 如果 op_host/op_kernel 没改但修改了 validation/，可加速（仅跳过算子编译，重新编译 pybind）：
# 注意：修改过 op_host/ 或 op_kernel/ 后不要使用 -skip_op_build，必须先完整重编译
python scripts/run_test.py -task_name <OpName> -stages compile precision -skip_op_build
```

**编译失败** → 查阅 `references/06-troubleshooting.md` 中的调试方法论和诊断工具。
**精度失败** → 同样查阅 `references/06-troubleshooting.md`，按系统化排障流程缩小问题范围。

**编译日志与「假失败」：**

- 日志末尾出现 **CPack / `cmake_install` / `binary/config`** 等告警时，**以 `Compiled successfully` 与 `verify_build` 是否找到 `kernel_gen_ops.*.so` 为准**；勿因次要 install 步骤报错反复清空整个 `runs` 重编，除非后续 `verify` 明确失败。
- **精度假失败**：日志中 REF/TEST 数值接近但 `acc=0`，且 `max_abs` 在个位数、`max_rel` 极大 —— 多为 **分母接近零的 rel** 或 **默认阈值过严**；按阶段 5「BF16 精度门槛」处理，而非先改 kernel。

**检查点：** 编译成功（`kernel_gen_ops.*.so` 存在），精度测试全部通过。

## 上下文管理原则

**迁移过程中严格控制上下文消耗。** 不要全文读取源文件，用精准的搜索定位需要的信息：

- **定位文件和模式**：`find` + `rg -n <pattern>` 按行号匹配，只读命中行附近的代码
- **确认关键结构**：`head -30` / `tail -20` 读文件首尾，`rg -n "^class |^struct |^extern |^OP_ADD|IMPL_OP"` 快速了解结构
- **全文读取仅限两种情况**：(1) 模板文件本身就短；(2) 编辑文件之前确认要修改的精确位置
- **不要通读 kernel 计算逻辑**：kernel 数学代码只在与 tiling ABI 交叉验证时需要看。仅读 kernel 入口的前 50 行（`GET_TILING_DATA` 字段读取、`TILING_KEY_IS` 分支、`GM_ADDR` 顺序），跳过算术循环体
- **用已有成功样例替代从头探索**：`find tasks -name "CMakeLists.txt" -path "*/answer/0/op_host/*"` 找同类算子的 CMake，直接复制适配

简单算子（模式 1）通常只需读取源 `_def.cpp`/`_tiling.cpp` 各 50-80 行 + 一个模板文件。复杂算子（模式 3）主要上下文消耗在本地化 `tiling_base/` 和 `common/` 上，直接从已迁移的同类算子复制。

## 边界

### 必须保持不变的
- kernel 数学计算路径（所有 dtype 分支、tiling key 分支）
- TilingData 字段类型、顺序、大小（host/kernel ABI）
- TilingData 注册名（`REGISTER_TILING_DATA_CLASS`）和 `OP_ADD` 名一致。如不一致，应修改 `REGISTER_TILING_DATA_CLASS` 的第一个参数以匹配 `OP_ADD`（`OP_ADD` 由目标目录名约束，不可随意改）
- 算子用户可见语义（输入/输出/属性的含义和顺序）
- 精度提升逻辑（如 int16→float32→计算的模式）

### 可以修改的
- include 路径（改为本地或 CANN 标准头）
- 日志/错误宏（替换为本地 `common/log.h` 或 `OP_LOGD`）
- 芯片配置（只保留 `ascend910b` 和 `ascend910_93`）
- CMake 结构（适配 msopgen）
- ListFloat → 拆分 Float 属性（pybind 桥接限制）

### 绝对不能做的
- 用 `npu_format_cast`（编译期不可用）
- **在未核对 GM 顺序、TilingData 字段、金标准与 NPU 输入一致性的情况下**，仅靠放宽 `custom_check_precision` 阈值、删 `test_cases` 行或跳过 case 让 precision「绿」—— 这会掩盖真实 kernel 错误
- 假设 `TilingParse` 会被调用或 `GetCompileInfo()` 返回非空（aclnn 直调路径不走 TilingParse，直接用 `PlatformAscendC` 获取平台信息）

**允许且推荐（与上条不矛盾）：** 在已确认参考路径与自定义算子语义一致后，对 BF16/融合算子使用 `custom_check_precision` 或略放宽 atol/rtol，与仓库内同类 level3 算子对齐（见阶段 5）。

## 参考文件索引

| 文件 | 内容 | 何时读取 |
|------|------|---------|
| `references/01-framework.md` | 两个框架的完整说明、opbuild 编译流程和关键限制 | 首次迁移时阅读第1-2节了解编译流程；简单模式（1/2a）可跳过其余部分 |
| `references/02-tiling.md` | Tiling 系统完整参考（含 Dynamic Input、ScalarList 等） | 处理 tiling 代码时；模式 4 必须配合第5节阅读 |
| `references/03-migration-patterns.md` | 五种迁移模式详解（含模式2a/2b区分） | 阶段 3、4 |
| `references/04-cmake-guide.md` | CMake 适配指南 | **模式 3 在阶段 4 前必须阅读**；其他模式在编译失败时阅读 |
| `references/05-validation.md` | Validation 文件生成规则 | 阶段 5 |
| `references/06-troubleshooting.md` | 系统化调试方法论和诊断工具 | 编译、运行时或精度失败时 |

### 各模式必读文件

| 模式 | 必读 | 选读（遇到问题时） |
|------|------|-------------------|
| 1: 简单 gert | SKILL.md + 03-patterns.md"模式 1" + 05-validation.md | 02-tiling.md（tiling 较复杂时） |
| 2a: 分文件 gert | SKILL.md + 03-patterns.md"模式 2a" + 05-validation.md | 02-tiling.md、04-cmake-guide.md（有子目录时） |
| 2b: IMPL_OP_OPTILING | SKILL.md + 03-patterns.md"模式 2b" + 02-tiling.md §7 + 05-validation.md | 04-cmake-guide.md（有子目录时） |
| 3: 复杂 registry | 以上全部 + 04-cmake-guide.md + 01-framework.md §1-2 | 从已迁移同类算子复制 tiling_base/ |
| 4: 旧接口 | SKILL.md + 03-patterns.md"模式 4" + 02-tiling.md §5 + 05-validation.md | 01-framework.md（理解 opbuild） |

## 模板文件索引

| 文件 | 用途 |
|------|------|
| `templates/question_op_host.cpp` | question/op_host 骨架 |
| `templates/question_op_kernel.cpp` | question/op_kernel 骨架 |
| `templates/pybind.cpp` | validation/pybind.cpp 模板 |
| `templates/module.py` | validation/module.py 模板 |
| `templates/prepare_inputs.py` | validation/prepare_inputs.py 模板 |
| `templates/test_cases.csv` | validation/test_cases.csv 模板 |
| `templates/api_desc.md` | question/api_desc.md 模板 |

## 快速排查卡片

```bash
# 查看编译产物（完整产物列表见 references/01-framework.md 第 2 节）
find runs -path "*<OpName>*" \( -name 'kernel_gen_ops*.so' -o -name 'libcust_opapi.so' -o -name 'liboptiling.so' -o -name 'libcust_opmaster_rt2.0.so' \) -print

# 查看 autogen 头文件（验证 OpDef 生成正确）
find runs -path "*<OpName>*" -name "aclnn_*.h" -print

# 查看 pybind 占位符（编译前应该是 aclnnCustomOp / custom_pybind_api）
# 将 <OpName> 替换为实际算子名
rg 'aclnn\w+|custom_pybind_api' "tasks/level<Level>/<Category>/<OpName>/validation/pybind.cpp"

# 检查链接依赖（仅在编译成功后有效）
find runs -path "*<OpName>*" \( -name "liboptiling.so" -o -name "libcust_opmaster_rt2.0.so" \) -exec ldd {} \;

# 查看编译错误日志
find runs -path "*<OpName>*" -name "*.log" -exec tail -80 {} \;

# 查看所有编译产物（.so/.o）
find runs -path "*<OpName>*" \( -name "*.so" -o -name "*.o" \) -print
```
