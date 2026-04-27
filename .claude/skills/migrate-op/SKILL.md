---
name: migrate-op
description: Migrate AscendC operators from the local CANN open-source repositories into NPUKernelBench msopgen single-operator tasks. Use when the user asks to migrate, port, adapt, or validate an operator such as "迁移 GatherV3", "把 BatchMatMulV3 迁移过来", or "/migrate-op ScatterElementsV2". The workflow is copy-first, pattern-based, and ends with compile and precision validation.
---

# AscendC 算子迁移

把本地 CANN 开源仓算子迁移到 NPUKernelBench 的独立 msopgen 编译单元，产出可用于微调数据集的 `answer/0` 参考实现，并用当前测试框架验证编译和精度。

迁移时保持原始算子的用户语义、tiling 意图、kernel 计算路径和 TilingData ABI。工程适配集中在 OpDef 注册、CMake 收录、依赖本地化、目标芯片裁剪、pybind 桥接、validation 输入和布局转换。

## 固定上下文

- 工作仓库：`/home/zhangqijia/repo/NPUKernelBench`
- 开源仓：`/home/zhangqijia/repo/{ops-nn,ops-cv,ops-math,ops-transformer}`
- 目标平台：CANN 9.0.0 + torch-npu，目标芯片配置保留 `ascend910b` 和 `ascend910_93`，其他芯片都不需要。
- 任务目录：`tasks/level{N}/{Category}/{OpName}/`
- 运行目录：`runs/lvl{N}/{Category}/{OpName}/fixed_case_{case_id}/sample{sample_id}/`
- 默认配置：`kernel_only_mode: True`，`compile.mode: msopgen`，`compile.common_dir: libs/common_include`
- 框架编译准备顺序：复制 `libs/common_include/msopgen/` 到 `runs/.../src/`，复制 `validation/`，再把 `answer/{sample_id}/` 覆盖到 `runs/.../src/`；这是框架行为，迁移时维护 `tasks/.../answer/0`，不手动编辑 `runs/.../src`
- `answer/0/CMakeLists.txt` 会覆盖 msopgen 根 CMake；只在必须改根编译规则时创建
- aclnn API 只使用 msopgen/opbuild 从 OpDef 自动生成的 `build_out/autogen/aclnn_*.cpp` 和 `aclnn_*.h`；迁移时不导入开源仓 `op_api/`，不手写 `aclnn_*` 源码
- `answer/0/op_host/CMakeLists.txt` 会覆盖默认 host CMake；只在 host 有子目录 `.cpp`、多策略 registry 或额外 include/链接规则时创建
- 默认 host CMake 只收集 `op_host` 当前目录 `.cpp`，不递归收集 `op_host/common/*.cpp`
- `validation/pybind.cpp` 中 `aclnnCustomOp` 和 `custom_pybind_api` 是占位符，编译前自动替换为 `aclnn{OpName}` 和 `{kernel_name}`
- precision 运行时会设置 `ASCEND_CUSTOM_OPP_PATH`、`CUST_OPAPI_LIB_PATH`、`CUSTOM_OPTILING_LIB_PATH` 和 `LD_LIBRARY_PATH`，加载本次编译的 `libcust_opapi.so` 与 `liboptiling.so`

## 工作方式

先把必要源码导入并本地化到 `answer/0`，再识别模式并做最小工程适配。对大代码量算子尤其如此：用文件清单和模式匹配判断结构，不全文阅读源码。

默认读取本文件即可开始迁移。按需读取：

- `references/copy-first-migration.md`：源码导入、本地化、模式识别、host/kernel/CMake 适配
- `references/validation.md`：`validation/`、`question/`、`api_desc.md` 生成规则
- `references/troubleshooting.md`：编译或精度失败后的错误定位
- `references/framework-contract.md`：需要重新判断 msopgen 复制、产物或运行加载路径时读取
- `references/iteration.md`：迁移成功后沉淀经过本次验证的规则

保持探索预算小而明确：

- 定位源码时只搜索四个开源仓的目录名、OpDef、kernel 入口和接口语义文档
- 迁移早期不查看其他已迁移算子的实现
- 编译或精度失败后，先根据当前错误日志修复；连续两轮仍无法定位时，再搜索一个最相似的已迁移算子或同类错误
- 查看相似算子时只看对应的 `CMakeLists.txt`、报错相关文件或同名接口模式

## 流程

1. 标准化算子名

将用户输入转为：

```text
OpName      = 大驼峰，例如 ScatterElementsV2
kernel_name = snake_case，例如 scatter_elements_v2
source_name = 开源仓目录名，通常同 kernel_name
```

若用户给出 level/category，直接使用。若未给出，先用轻量目录列表确认已有 category；能从开源仓一级目录或算子语义稳定映射时直接选择，不能稳定判断时询问一次。

2. 定位源算子

在四个开源仓中定位 `source_name` 目录，优先选择同时包含 `op_host/` 与 `op_kernel/` 的实现。若只有 `arch35/`、`*_apt.cpp`、Kirin 或 ascend950-only 路径，继续在同仓查找 910B/membase 实现；找不到时记录阻塞证据。

3. 初始化目标目录

创建：

```text
tasks/level{N}/{Category}/{OpName}/
  question/api_desc.md
  question/op_host/
  question/op_kernel/
  answer/0/op_host/
  answer/0/op_kernel/
  validation/module.py
  validation/prepare_inputs.py
  validation/pybind.cpp
  validation/test_cases.csv
```

4. 导入并本地化 answer

先把源算子的 `op_host/`、`op_kernel/` 和被 include 的公共依赖导入到 `answer/0/`，再在目标目录内做适配。`op_api/` 可作为接口语义参考，但不作为迁移源码导入；`aclnn_*` 始终由 opbuild 自动生成。用 `rg` 识别结构、依赖和入口；只在命中处或报错处打开局部代码。

5. 识别迁移模式

在 `answer/0` 内用模式匹配分类：

- 简单 gert：`gert::TilingContext*` + `.SetTiling`
- 分文件 gert：`*_def.cpp`、`*_tiling.cpp`、`*_infershape.cpp`
- 复杂 registry：`TilingBaseClass`、`REGISTER_TILING_TEMPLATE`、`IMPL_OP_OPTILING`
- 旧接口：`TeOpParas`、`OpRunInfo`、`ge::Operator`、`nlohmann`
- 非目标架构：`arch35`、`*_apt.cpp`、`kirin`、`310p`、`ascend950`

按 `references/copy-first-migration.md` 做对应适配。

6. 生成 question、validation 和 API 文档

`question/` 是题目骨架，不参与 answer 编译，但必须与 answer 的接口和 kernel 签名一致。`validation/` 是真实测试桥接，必须保持 PyTorch golden 语义。生成前读取 `references/validation.md`。

7. 编译验证

先只编译算子，尤其是编译时间较长的算子：

```bash
python scripts/run_test.py -task_name <OpName> -stages compile
```

`-stages compile` 不带 `-skip_op_build` 会执行完整算子编译。编译已经通过后，进入精度验证时不要再运行不带 `-skip_op_build` 的 `-stages compile precision`，否则会重复编译 OPP 产物。

查看编译日志：

```bash
runs/lvl{N}/{Category}/{OpName}/fixed_case_0/sample0/log/lvl{N}_category{Category}_{OpName}_compile.log
```

修改 `op_host/` 或 `op_kernel/` 后，删除当前算子的 `runs/lvl{N}/{Category}/{OpName}`，再重新执行完整 compile。只修改 `validation/`、`pybind.cpp`、`module.py`、`prepare_inputs.py` 或测试用例时，保留已编译 OPP 产物并使用 `-skip_op_build`。

8. 精度验证

编译通过后，先用已有 OPP 产物跑少量 case：

```bash
python scripts/run_test.py -task_name <OpName> -stages compile precision -skip_op_build -n_case 1
```

少量 case 通过后，继续复用已有 OPP 产物跑全部 case：

```bash
python scripts/run_test.py -task_name <OpName> -stages compile precision -skip_op_build
```

查看精度日志：

```bash
runs/lvl{N}/{Category}/{OpName}/fixed_case_0/sample0/case_0/log/lvl{N}_category{Category}_{OpName}_sample0_case0_precision.log
```

完成条件是编译通过且所有精度 case 通过。若受环境、源码缺失或目标架构缺失阻塞，给出具体日志路径、错误摘要、已尝试步骤和最小下一步。

## 边界

- 保持 kernel 数学计算路径、tiling 算法意图、TilingData 字段顺序和用户可见语义
- 工程适配只落在当前算子目录；框架代码 `npukernelbench/` 只有在用户明确要求修框架时修改
- 目标工程自包含，`answer/0` 不依赖开源仓根目录
- 精度验证用真实 golden 语义；不通过放宽容差、跳过失败 case 或改参考语义掩盖迁移问题
- 经验沉淀只写入已由本次编译/精度日志验证的规则
