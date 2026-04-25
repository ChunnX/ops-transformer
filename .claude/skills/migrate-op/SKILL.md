---
name: migrate-op
description: 将 CANN 开源仓库（ops-nn/cv/math/transformer）中的 AscendC 算子迁移到 NPUKernelBench 单算子工程格式。输入算子名称，完成源码定位、工程适配、编译验证、精度测试和通用经验沉淀。用户要求迁移算子，输入算子名称。例如："迁移 GatherV3 算子"、"把 BatchMatMulV3 迁移过来"、"/migrate-op ScatterElementsV2"，则需要使用该skill
---

# AscendC 算子迁移

核心任务：将 CANN 开源仓库（ops-nn、ops-cv、ops-math、ops-transformer）算子完成编译通过且精度验证通过的迁移到 NPUKernelBench 单算子工程格式。
**严禁修改算子计算逻辑**，仅做工程框架适配。

允许的工程适配包括：OpDef 注册、CMake、依赖本地化、目标芯片配置裁剪、pybind 桥接、validation 侧数据布局转换，以及因 msopgen/pybind 限制导致的接口桥接调整。禁止改变 kernel 数学路径、tiling 算法意图、TilingData ABI 和用户可见算子语义。

## 输入

`$ARGUMENTS`：算子名称（必填），可选附带 level 和 category。

## 执行流程

严格按序执行 11 个步骤。使用当前代理可用的任务计划/进度工具记录步骤状态；没有专用工具时，用简短状态更新代替。

### 步骤 1：源码定位与目标目录确定

1. 在同级目录的 `{ops-nn,ops-cv,ops-math,ops-transformer}` 中搜索算子。开源仓用下划线命名（`gather_v3`），NPUKernelBench 用大驼峰（`GatherV3`）。
2. 先扫描现有 `tasks/`、同类算子和 category 结构推断 level（1/2/3）和 category；无法合理推断时再询问用户。
3. 在 `tasks/level{N}/{category}/` 下创建 `{OpName}` 目录。

### 步骤 2：工程目录初始化

创建完整目录结构：`api_desc.md`、`question/`（op_host + op_kernel）、`answer/0/`（op_host + op_kernel + CMakeLists.txt）、`validation/`（module.py、prepare_inputs.py、pybind.cpp、test_cases.csv）。

### 步骤 3：源码提取

从开源仓提取 op_host、op_kernel 文件到 `answer/0/`。只需要tiling和kernel的核心代码，不需要config和op_api，msopgen不需要这些。

### 步骤 4：op_host 代码适配（核心）

参照 `references/tiling_migration_guide.md` 完整执行。

### 步骤 5：op_kernel 代码清理

参照 `references/kernel_migration_guide.md` 完整执行。op_kernel 只做目标架构裁剪、入口签名核对、依赖本地化和编译必要清理；不得改写计算路径。

### 步骤 6：题目骨架生成

参照 `templates/question_skeleton.md` 生成 `question/` 下的骨架代码。kernel 签名必须与 answer 完全匹配。

### 步骤 7：验证体系代码生成

参照 `references/validation_guide.md` 生成 `validation/` 下所有文件。注意 ListFloat 拆分、OPTIONAL 输入、标量类型匹配等专项规则。

### 步骤 8：API 文档生成

参照 `templates/api_doc.md` 生成 `api_desc.md`。

### 步骤 9：编译验证与迭代修复

```bash
# 编译
python scripts/run_test.py -task_name <OpName> -stages compile

# 查看编译日志
cat runs/lvl{N}/{Category}/{OpName}/fixed_case_0/sample0/log/lvl{N}_category{Category}_{OpName}_compile.log

# 编译失败修复后，删除缓存重新编译
rm -rf runs/lvl{N}/{Category}/{OpName}
```

只删除当前目标算子的 `runs/.../{OpName}` 缓存目录，不删除其他任务产物。编译错误诊断见 `references/troubleshooting.md` 第 1 节。

### 步骤 10：精度验证与迭代修复

```bash
# 精度测试
python scripts/run_test.py -task_name <OpName> -stages precision

# 查看精度日志
cat runs/lvl{N}/{Category}/{OpName}/fixed_case_0/sample0/case_{i}/log/lvl{N}_category{Category}_{OpName}_sample0_case{i}_precision.log

# 仅修改 validation 时，跳过 op_build
python scripts/run_test.py -task_name <OpName> -stages compile precision -skip_op_build

# 修改了 op_host/op_kernel 时，删除缓存
rm -rf runs/lvl{N}/{Category}/{OpName}
```

持续修复直到编译和精度全部通过。常见错误诊断见 `references/troubleshooting.md`。若遇到源码缺失、仅有非目标架构实现、缺 CANN/NPU 环境、框架级问题等阻塞，必须说明阻塞证据、已尝试步骤和最小下一步。

### 步骤 11：经验沉淀

迁移完成后，参照 `references/iteration.md` 执行迭代流程：回顾本次迁移经验，判断是否具备通用性，若值得沉淀则归入对应文档。

## 红线

- 严禁修改算子计算逻辑
- 严禁修改 `npukernelbench/` 工程基础文件
- 严禁迁移非目标平台内容（arch35、`*_apt.cpp`、310p tiling、kirin）
- 严禁为通过精度测试而放宽 golden 语义或跳过失败用例
