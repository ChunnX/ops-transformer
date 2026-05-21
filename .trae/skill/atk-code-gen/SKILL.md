---
name: "atk-code-gen"
description: "Generates ATK test suite (executor/generator/yaml) for MC2 operators migrating from mc2_test to ATK framework. Invoke when user asks to create or migrate an operator's ATK test project."
---
# ATK算子测试工程生成

根据用户提供的信息生成ATK框架下的MC2算子测试脚本，生成完整的ATK测试三件套以及配套的非法场景校验。

## 输入要求

用户应提供：

- 参考代码（mc2_test工程中的算子脚本或已有ATK算子工程或小算子级联片段）
- 算子文档（aclnn接口定义、算子规格说明）
- 限制条件（如只看护某架构上的精度、特定场景排除等）

## 工作流

### 1. 先根据用户提供的信息生成executor、generator和yaml（ATK算子测试三件套）

拷贝 `templates/` 下的模板，并修改。一定要先拷贝至目标目录再修改文件！禁止完全由 Agent 生成！输出文件：

- **executor**：`executor_aclnn<OpName>.py`，含CPU真值、NPU级联标杆、ACLNN执行以及相关输入输出处理等
- **generator**：`generator_<variant>.py`，参数约束逻辑
- **yaml**：用例设计文件，定义输入参数规格

如若有许多world_size选择，可以酌情增加逻辑依赖环境变量RANK_SIZE

### 2. 生成实现细节（implementation_details.md），并让用户确认、提出意见
- 其中内容包含：场景覆盖表
    - 各参数取值表
    - 取值组合对应的场景表
    - 标注ATK用例覆盖了算子文档中哪些场景

- 其中内容包含：三端输入处理对照表：面对各种场景时，CPU实现、NPU级联实现、ACLNN实现各自对输入做了什么操作。

### 3. 生成非法场景枚举表（abnormal_cases.xlsx），并让用户确认、提出意见

根据算子文档中的**约束说明**、入参说明等描述，枚举所有非法输入场景，以表格形式列出（参考templates/abnormal_cases.xlsx）。目的是验证算子能否正确识别错误并退出，而非产生错误结果。

**枚举维度**（需覆盖以下所有维度，每个维度尽可能穷举）：

- **数据类型非法**：包括不支持的dtype、支持的数据类型但不支持的dtype组合
- **维度非法**：文档明确不支持的shape或是shape组合
- **空Tensor非法**：文档明确不支持空tensor的场景
- **非连续Tensor非法**：文档明确不支持非连续tensor的场景。需参考 `op_host/*_def.cpp` 中各参数的连续性标记：
  - `.AutoContiguous()`：算子内部会自动连续化，非连续输入不会触发校验错误，无需枚举
  - `.IgnoreContiguous()`：算子不处理连续性，非连续输入可能触发校验错误，需枚举
  - 无标记：需根据文档约束判断
- **属性值非法**：文档明确不支持的属性取值
- **空指针非法**：文档明确不能为空指针的必选参数传入空指针
- **格式非法**：文档明确不支持的format（如nz）

对于输出，只需要校验空指针、shape和dtype即可。

有时有的校验不会被实现，因为shape组合的校验也会将其包含在内。
例如，x1和x2共同决定了输出的shape，x1或x2如果是空tensor，会被校验；x1、x2不是空tensor，输出则不可能为空tensor，否则会被shape组合规则拦截。此时，空tensor校验的场景就可以被省略。

该步骤完成后，需要建议用户先运行正常用例，向用户确认脚本生成正常、异常用例表格符合要求，最后向用户咨询常规用例json路径，最后再撰写脚本生成异常用例json。

### 4. 根据异常用例表，生成异常用例（abnormal_cases.json）

根据非法场景枚举表与常规用例json的格式，生成异常用例json文件（如果用户没有提供常规用例json参考，则参考templates/cases.json）。以便于executor调用。
不同维度的非法场景有不同的构造方式：

1、对于算子的输入参数：
- json文件中直接指定即可：
  - **数据类型非法**：json用例中直接指定不支持的dtype/dtype组合
  - **维度非法**：json用例中指定对应tensor的shape
  - **空Tensor非法**：json用例中指定对应tensor的shape某维度为0
  - **属性值非法**：构造不支持的属性取值
- 需要在json文件中额外构造attrs属性，并在executor中读取、操作、并pop掉（千万不能传入aclnn接口）
  - **空指针非法**：新增attrs属性：isNullptr。它是一个int64_t列表，里面包含的内容指定了哪些tensor需要被设置为空指针
  - **格式非法**：新增attrs属性：isFmt。它是一个int64_t列表，里面包含的内容指定了哪些tensor需要被设置为指定格式（枚举变量29即NZ格式）
  - **非连续Tensor非法**：新增attrs属性：isNotContiguous。它是一个int64_t列表，里面包含的内容指定了哪些tensor需要被设置为非连续tensor
    需要额外构造attrs属性的场景的方法已在 `executor_aclnn_template.py`中写出

2、对于算子的输出参数：
- 需要在json文件中额外构造attrs属性 `outputOps`（string类型），并在executor中读取、操作、并pop掉（千万不能传入aclnn接口）
- `outputOps` 的格式为：`输出参数名1: shape:??, dtype:??, fmt:??;输出参数名2: shape:??, dtype:??, fmt:??`
  - 多个输出参数用 `;` 分隔
  - 每个输出参数的操作用 `,` 分隔
  - 支持的操作：`shape:[list]`（用Python列表语法）、`dtype:str`（如fp16/bf16/fp32/int8）、`fmt:int`
  - 示例：`"outputOps": "output: shape:[128]"` 表示将 output 的 shape 改为 [128]
  - 示例：`"outputOps": "output: shape:[2,32,64], dtype:fp32"` 表示同时修改 shape 和 dtype
- executor 中通过 `_handle_output_modification` 方法解析该字符串，创建新的 acl tensor 并同时更新 `input_args[idx]` 和 `output_packages[out_idx]`
- 输出参数的空指针非法仍通过 `isNullptr` attrs 实现，`isNullptr` 可同时引用输入和输出参数

最终需要写一份脚本，将非法场景枚举表转化为异常用例的json

## 约束

1. 脚本实际运行时yaml中变量的声明顺序与generator的 `case_config.inputs` 序号一一对应，在AclnnBaseApi子类的重载init_by_input_data后与aclnn接口声明完全一致
2. 如有信息不全（torch_npu接口不够、遇到torch不支持的数据类型/排布方式、文档不明确），向用户提问而非猜测
3. 非法场景枚举必须逐条对应文档中的约束说明，不可遗漏，不可凭空捏造文档未提及的非法场景

## 反模式

🈲 不经过用户同意，创建一堆意料之外的文件（没在交付物里面说明的文件）

🈲 过量读取模板/文档，提前读取后续任务涉及的资料

🈲 读取用户指定内容与skill自带内容以外的内容

🈲 跳过问询用户的环节

## 参考资源

- `templates/executor_aclnn_template.py` — executor模板及占位符说明
- `templates/generator_template.py` — generator模板
- `templates/yaml_template.yaml` — yaml模板及各类型参数写法范例
- `templates/abnormal_cases.xlsx` — yaml模板及各类型参数写法范例
- `templates/cases.json` — json文件模板
- `docs/ATK.md` — ATK框架API参考（BaseApi/AclnnBaseApi/空指针构造/INT4处理等）
