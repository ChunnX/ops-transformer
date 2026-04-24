---
name: migrate-op
description: 将 CANN 开源仓库（ops-nn/cv/math）中的 AscendC 算子迁移到 NPUKernelBench 单算子工程格式。输入算子名称，完成源码定位→工程适配→编译验证→精度测试全流程。
---

# AscendC 算子迁移

核心任务：将 CANN 开源仓库（ops-nn、ops-cv、ops-math）算子完成编译通过且精度验证通过的迁移到 NPUKernelBench 单算子工程格式。
**严禁修改算子计算逻辑**，仅做工程框架适配。

## 输入

`$ARGUMENTS` 传入：
1. **必填**：算子名称（如 `GatherV3`，支持 PascalCase 和 snake_case）
2. **可选**：`<level> <category>` 覆盖自动检测结果

## 适用场景

适用于 CANN 开源仓库（ops-nn、ops-cv、ops-math）中的 AscendC 算子迁移到 NPUKernelBench 单算子工程格式。
不适用于：自定义算子（非 CANN 开源仓库）、非 AscendC 算子、非 ascend910b/910_93 平台。

## 关键路径

| 路径 | 用途 |
|:---|:---|
| `scripts/migrate_helper.py` | 迁移辅助工具（find/create/extract/list_existing） |
| `tasks/<level>/<Category>/<OpName>/` | 目标工程输出 |
| `runs/msopgen/lvl{1,2,3}/<Cat>/<Op>/fixed_case_0/sample0/log` | 编译/测试日志 |
| `runs/msopgen/lvl{1,2,3}/<Cat>/<Op>/fixed_case_0/sample0/build_out/autogen/` | autogen 产出（aclnn_*.h/cpp，标量类型校验用） |
| `/mnt/workspace/ops-{nn,cv,math}` | CANN 源码仓库 |

## 目标工程结构

算子输出到 `tasks/<level>/<Category>/<OpName>/`，完整目录结构：

```
tasks/<level>/<Category>/<OpName>/
  answer/0/
    op_host/(*.cpp, *.h)        # 完整算子实现
    op_kernel/(*.cpp, *.h)
  question/
    op_host/<opname>.cpp         # 骨架：OpDef + 最小 Tiling
    op_kernel/<opname>.cpp       # 空 kernel（签名与 answer 一致）
  validation/
    module.py                    # Model（golden）+ ModelNew（kernel）
    prepare_inputs.py            # 输入生成 + 精度检查
    pybind.cpp                   # PyBind11 桥接
    test_cases.csv               # 测试用例（分号分隔）
  api_desc.md                    # API 说明（中文）
```

## 文档导航

### references/ — 知识文档（AI 读取后按规则生成代码）

按迁移阶段和算子复杂度分层，每份文档有明确边界，不重叠：

| 文档 | 适用阶段 | 适用算子 | 定位 |
|:---|:---|:---|:---|
| `conversion_rules.md` | 步骤4 | **所有算子** | op_host **通用**适配：头文件、平台清理、CompileInfo/TilingParse 决策树（替换或保留）、注册逻辑、日志宏 |
| `complex_op_host.md` | 步骤3-4 | **仅复杂算子** | 复杂算子**自包含**指南：复杂度判定→依赖收集→CompileInfo 保留（cube 硬件参数/平台特性探测）→框架 stub→继承链→CMake→调试清单 |
| `validation_guide.md` | 步骤6 | **所有算子** | 验证代码生成：pybind.cpp / module.py / prepare_inputs.py / test_cases.csv + 专项规则（ListFloat/OPTIONAL/布局） |
| `question_skeleton.md` | 步骤5 | **所有算子** | question 骨架模板（OpDef + 空 kernel 签名） |
| `api_doc_template.md` | 步骤7 | **所有算子** | api_desc.md 中文文档结构模板 |
| `iteration.md` | 步骤10 | **所有算子** | 迁移经验沉淀方法论 |

**简单/复杂分支**：步骤3由 `is_complex` 自动判定。
- **简单算子**：步骤4只读 `conversion_rules.md` 即可（CompileInfo/TilingParse 可替换也可保留，按决策树选择）
- **复杂算子**：步骤3-4以 `complex_op_host.md` 为主（自包含所有复杂场景规则），其中平台清理、头文件三件套等通用规则引用 `conversion_rules.md`

### templates/ — 文件模板（被工具脚本直接复制到目标工程）

| 文件 | 适用算子 | 用途 | 放置方式 |
|:---|:---|:---|:---|
| `root_CMakeLists.txt` | **所有算子** | 工程根目录 CMake | `create` 命令自动放置 |
| `op_host_CMakeLists.txt` | **仅复杂算子** | op_host 子目录 CMake（GLOB_RECURSE） | `extract_complex` 命令自动放置 |

简单算子的 op_host CMake 由框架 scaffold 自动生成，不需要模板文件。
references/ 中的内联代码块是模式参考（AI 读取后生成），templates/ 中的文件是实际产物（脚本直接复制），二者不重叠。
reference 文档中可能引用 `tasks/` 下的已有迁移工程作为参考示例。

## 执行流程

严格按序执行，使用 TaskCreate 追踪进度。每一步按需读取对应 `references/` 文档。

核心步骤为步骤4（op_host 代码适配），其余步骤为其准备输入或验证输出。

### 步骤1：源码定位与存量校验

**前置条件**：用户输入算子名称。
**本步骤产出**：`src_dir`、`level`、`category`、复杂度判断结果（`is_complex`）。

```bash
python .claude/skills/migrate-op/scripts/migrate_helper.py find <OpName>
python .claude/skills/migrate-op/scripts/migrate_helper.py list_existing <OpName>
```
- 从 JSON 提取 `src_dir`、`level`、`task_category`
- 若算子已存在，**覆盖前必须与用户确认**

#### 步骤1.5：复杂度判断

完成步骤1后，运行 `is_complex` 命令自动判定。判断标准详见 `references/complex_op_host.md` 第0节。

```bash
python .claude/skills/migrate-op/scripts/migrate_helper.py is_complex <src_dir>
```

若判定为复杂算子，步骤3使用 `extract_complex`，步骤4使用 `references/complex_op_host.md`。
若判定为简单算子但源码使用了 TilingParse（CompileInfo 仅为空结构体或基础参数），按 `conversion_rules.md` 第4节决策树选择方案 A（替换）或方案 B（保留）。

### 步骤2：工程目录初始化

**前置条件**：步骤1产出的 `level`、`category`。
**本步骤产出**：空的目标工程目录结构。

```bash
python .claude/skills/migrate-op/scripts/migrate_helper.py create <OpName> <level> <category>
```

### 步骤3：源码提取

**前置条件**：步骤2的目录 + 步骤1的 `src_dir`。
**本步骤产出**：`answer/0` 下的源码文件（待转换）。

**简单算子**：
```bash
python .claude/skills/migrate-op/scripts/migrate_helper.py extract <src_dir> <task_dir>/answer/0
```
工具自动：复制 kernel/辅助文件 → 标记需转换的 tiling/def → 跳过 infershape → 生成直接可用 CMakeLists.txt
- **跨算子共享头文件**：若 kernel 引用了其他算子的头文件（如 `#include "../xxx/xxx_common.h"`），复制到 `answer/0/op_kernel/` 并改为本地引用

**复杂算子**：
```bash
python .claude/skills/migrate-op/scripts/migrate_helper.py discover <src_dir>
python .claude/skills/migrate-op/scripts/migrate_helper.py extract_complex <src_dir> <task_dir>/answer/0
```
工具自动：发现跨算子依赖 → 复制所有依赖文件到子目录 → 改写 include 路径 → 生成 GLOB_RECURSE CMakeLists.txt。手动检查和补充步骤详见 `references/complex_op_host.md` 第1节。

### 步骤4：op_host 代码适配（核心）

**前置条件**：步骤3的文件 + 步骤1.5的复杂度判断结果。
**本步骤产出**：适配完成的 op_host 代码。

**简单算子**：读取 `references/conversion_rules.md`（通用规则）完成全量转换，按末尾"转换校验清单"逐项自查。

**复杂算子**：读取 `references/complex_op_host.md`（自包含指南）完成适配，按第9节"调试检查清单"逐项自查。

若 `_def.cpp` 中存在 `FORMAT_NC1HWC0`，需同步执行 `references/conversion_rules.md` 第9节的 C++ 侧格式修改（`_def.cpp` 和 question 骨架的 `FORMAT_NC1HWC0 → FORMAT_ND` 替换）。Python 侧的 reshape/permute 转换在步骤6中处理。

### 步骤5：题目骨架生成

**前置条件**：步骤3产出的 `answer/0/op_kernel/` 中的 kernel 函数签名 + 步骤4产出的 op_host 代码。
**本步骤产出**：`question/` 骨架代码。

读取 `references/question_skeleton.md` 生成：
- `question/op_host/<opname>.cpp`：OpDef + 最小 Tiling
- `question/op_kernel/<opname>.cpp`：空 kernel（签名与 answer **完全匹配**）

### 步骤6：验证体系代码生成

**前置条件**：步骤4产出的 `_def.cpp` 中的 Input/Output/Attr 声明顺序。
**本步骤产出**：`validation/` 下所有文件。

读取 `references/validation_guide.md` 生成 `pybind.cpp`、`module.py`、`prepare_inputs.py`、`test_cases.csv`。

若算子涉及 `ListFloat` 属性或 `NC1HWC0` 格式，需额外参考 `references/validation_guide.md` 第5节（ListFloat 拆分）和第8节（数据格式与布局适配）中的对应专项规则。

### 步骤7：API 文档生成

**前置条件**：步骤4的 op_host + 步骤6的 `module.py`。
**本步骤产出**：`api_desc.md`。

读取 `references/api_doc_template.md` 生成中文 `api_desc.md`。

### 步骤8：编译验证与迭代修复

**前置条件**：步骤2-7的所有文件。
**本步骤产出**：编译通过的 `.so` 文件。

循环执行以下子步骤直到编译通过：

**8.1 首次编译**：
```bash
python scripts/run_test.py -task_name <OpName> -stages compile
```

**8.2 若编译失败，读日志定位错误**：
```bash
cat runs/msopgen/lvl{level}/{cat}/{Op}/fixed_case_0/sample0/log/build.log
```

**8.3 按错误类型选择修复策略**：
- 通用编译错误：参考 `references/conversion_rules.md` 常见编译错误速查表
- 复杂算子特有错误：参考 `references/complex_op_host.md` 第9节调试清单

**8.4 修改代码后，删除 runs 目录重新编译**：
```bash
rm -rf runs/msopgen/lvl{level}/{cat}/{Op}
```
然后回到 **8.1** 重新编译。

### 步骤9：精度验证与迭代修复

**前置条件**：步骤8的编译产物。
**本步骤产出**：精度验证通过的测试报告。

循环执行以下子步骤直到精度验证通过：

**9.1 首次精度测试**：
```bash
python scripts/run_test.py -task_name <OpName> -stages precision
```

**9.2 若精度测试失败，读日志定位错误**：
查看测试输出中的误差信息，分析失败原因。

**9.3 按错误类型选择修复策略**：
参考 `references/validation_guide.md` 各专项规则（ListFloat、OPTIONAL 输入、标量类型匹配、数据布局等）。

**9.4 修改代码后，若涉及编译产物变更则删除 runs 目录**：
```bash
rm -rf runs/msopgen/lvl{level}/{cat}/{Op}
```
然后根据修改范围决定回退目标：
- 回到 **8.1**（重新编译）：修改了 `answer/0/op_host/` 或 `answer/0/op_kernel/` 下的文件，或修改了 CMakeLists.txt
- 回到 **9.1**（重新测试）：仅修改了 `validation/` 或 `api_desc.md` 下的文件

若 op_host 的 Input/Output/Attr 声明顺序发生变化，需重新执行步骤5（question 骨架）和步骤6（验证体系代码）。

**严禁中途终止，持续修复直到全部通过。**

### 步骤10：迭代本技能

**前置条件**：步骤1-9全流程。
**本步骤产出**：更新后的技能文档。

读取 `references/iteration.md`，根据本次迁移经验提炼通用性知识，按方法论更新技能文档。

## 红线（禁忌）

- 严禁修改算子计算逻辑
- 严禁修改工程基础文件（framework/、kernel_generator/ 等）
- 严禁迁移非目标平台内容（arch35、*_apt.cpp、310p tiling、kirin）
- 重新编译前删除对应 runs 目录
- 严禁覆盖已有迁移工程（除非用户明确确认）
- 复杂算子的 CompileInfo 不得替换为 PlatformAscendC（当 CompileInfo 含有 TilingParse 填充的 cube 硬件参数或平台特性探测时）
