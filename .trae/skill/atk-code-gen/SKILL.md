---
name: "atk-code-gen"
description: "Generates ATK test suite (executor/generator/yaml) for MC2 operators migrating from mc2_test to ATK framework. Invoke when user asks to create or migrate an operator's ATK test project."
---

# ATK算子测试工程生成

将MC2算子从mc2_test框架迁移到ATK框架，生成完整的ATK测试三件套。

## 输入要求

用户应提供：
- 参考代码（mc2_test工程中的算子脚本、已有ATK算子工程）
- 算子文档（aclnn接口定义、算子规格说明）
- 限制条件（如只看护某架构上的精度、特定场景排除等）
- 小算子级联实现的方法（包括接口、调用流程，如果之前的代码没有写）

## 输出物

### 1. ATK算子测试三件套

基于 `templates/` 下的模板生成：
- **executor**：`executor_aclnn<OpName>.py`，含CPU真值、NPU级联标杆、ACLNN执行三个实现
- **generator**：`generator_<variant>.py`，参数约束逻辑
- **yaml**：用例设计文件，定义输入参数规格

### 2. 场景覆盖表（implementation_details.md）

- 各参数取值表
- 取值组合对应的场景表
- 标注ATK用例覆盖了算子文档中哪些场景

### 3. 三端输入处理对照表（implementation_details.md）

面对各种场景时，CPU实现、NPU级联实现、ACLNN实现各自对输入做了什么操作。

## 约束

1. yaml中变量的声明顺序必须与generator的 `case_config.inputs` 序号一一对应，且与aclnn接口声明完全一致
2. 如有信息不全（torch_npu接口不够、遇到torch不支持的数据类型/排布方式、文档不明确），向用户提问而非猜测

## 参考资源

- `templates/executor_aclnn_template.py` — executor模板及占位符说明
- `templates/generator_template.py` — generator模板
- `templates/yaml_template.yaml` — yaml模板及各类型参数写法范例
- `docs/ATK.md` — ATK框架API参考（BaseApi/AclnnBaseApi/空指针构造/INT4处理等）
