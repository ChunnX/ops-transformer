# 模板文件说明

本目录包含算子迁移过程中被工具脚本**直接复制**到目标工程的文件模板。

## templates/ vs references/ 边界

| 目录 | 定位 | 使用方式 |
|:---|:---|:---|
| `templates/` | **实际文件**，被 `migrate_helper.py` 脚本直接复制到目标工程 | 脚本自动放置，AI 通常不需要读取 |
| `references/` | **知识文档**，被 AI 读取后按规则生成代码 | AI 在对应步骤读取，按规则生成 |

二者不重叠：`references/` 中的内联代码块是模式参考（展示代码应该长什么样），`templates/` 中的文件是实际产物（直接成为目标工程的一部分）。

## 文件清单

- **root_CMakeLists.txt**：工程根目录 CMakeLists.txt，所有算子通用。由 `migrate_helper.py create` 命令自动放置到目标工程根目录。
- **op_host_CMakeLists.txt**：op_host 子目录 CMakeLists.txt，**仅复杂算子使用**。使用 `GLOB_RECURSE` 自动收集子目录中的所有源文件（含跨算子依赖）。简单算子由框架 scaffold 自动生成标准 CMakeLists.txt，不需要此模板。

## 适用边界

| 算子类型 | op_host CMakeLists.txt | 说明 |
|:---|:---|:---|
| 简单算子 | 框架 scaffold 自动生成 | 单文件 tiling，无子目录依赖 |
| 复杂算子 | 使用 op_host_CMakeLists.txt | 多子目录（common/、matmul_v3/ 等），需 GLOB_RECURSE |
