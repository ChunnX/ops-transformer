# torch_extension开发规范

本文档约定`cann_ops_transformer`（torch_extension）新增/修改算子api时的目录组织、命名、各层实现以及文档编写规范。开发者新增算子前请先通读本规范，并参考已有算子（如`flash_attn`）作为模板。

`cann_ops_transformer`通过PyTorch的JIT（`torch.utils.cpp_extension.load`）在首次调用时即时编译C++ kernel wrapper，把PyTorch的函数接口桥接到CANN的aclnn接口，同时通过GE Converter支持torchair图模式。一个完整的算子api通常由「Python前端、C++后端、torchair图模式Converter、文档」四部分组成。

## 1. 算子 api 目录组织规范总体

开发新增算子api涉及新增与修改以下文件（以算子`${op_api}`为例）：

```
├── torch_extension
    ├── cann_ops_transformer
        ├── __init__.py                                 # 包根入口，对外导出算子 api 接口（新增 import）
        ├── op_builder
        │   ├── builder.py                              # OpBuilder 基类，统一管理 JIT 编译、schema/meta 注册（一般无需修改）
        ├── common
        │   ├── inc
        │   │   ├── aclnn_common.h                      # ACLNN_CMD 宏、类型转换等公共能力（一般无需修改）
        │   │   ├── hccl_common.h                       # 通信类算子公共能力
        ├── ops
        │   ├── csrc
        │   │   ├── ${op_api}.cpp                       # 算子 api 的 C++ 实现，调用 aclnn 接口
        │   ├── graph_convert
        │   │   ├── graph_convert_${op_api}.py          # 算子 api 的 torchair 图模式（GE Converter）实现
        │   ├── __init__.py                             # 对外导出新增算子 api（新增 import）
        │   └── ${op_api}.py                            # 算子 api 的 Python 前端实现（OpBuilder、schema、meta、对外函数）
        ├── docs
            ├── torch_extension_guidelines.md           # 本开发规范
            ├── zh
                ├── ${op_api}.md                        # 算子 api 的中文文档
```

新增一个算子api的标准动作清单（以`flash_attn`为例）：

1. 在`ops/csrc/flash_attn.cpp`中实现C++ kernel wrapper，调用`ACLNN_CMD`拉起aclnn接口；
2. 在`ops/flash_attn.py`中编写`OpBuilder`子类（定义`sources`/`schema`/`register_meta`），注册dispatcher实现，并提供对外的Python函数；
3. 在`ops/graph_convert/graph_convert_flash_attn.py`中编写图模式Converter（若需支持图模式）；
4. 在`ops/__init__.py`中导出新增的对外接口；
5. 在包根`cann_ops_transformer/__init__.py`中import导出对外接口，使用户可直接从包根访问；
6. 在`docs/zh/flash_attn.md`中补充算子文档。

>新增文件请放在`cann_ops_transformer`包下，import路径统一以`cann_ops_transformer`为根。

## 2. 命名规范

### 2.1 API 命名

一个算子从schema注册到对外导出，涉及多个层级的命名，需保持一致且各司其职。**对外api接口及算子名一律不带`npu_`前缀**，直接采用算子语义的小写蛇形名（如`flash_attn`）：

| 层级 | 命名约定 | 示例 |
| --- | --- | --- |
| Library名（DEF域） | 固定为`cann_ops_transformer` | `AS_LIBRARY = Library("cann_ops_transformer", "DEF")` |
| schema算子名 / aten注册名 | 算子语义的小写蛇形名，不带`npu_`前缀 | `flash_attn` |
| C++ wrapper函数名 | 与schema算子名一致，置于`namespace op_api`内 | `op_api::flash_attn` |
| `PYBIND11_MODULE`导出名 | 与schema算子名一致 | `m.def("flash_attn", &flash_attn, "flash_attn");` |
| Meta实现函数名 | schema算子名 + `_meta`后缀 | `flash_attn_meta` |
| PrivateUse1 dispatcher函数名 | 下划线前缀 + schema算子名 | `_flash_attn` |
| OpBuilder子类名 | 算子名的大驼峰 + `OpBuilder`后缀；内部专用可加`_`前缀 | `FlashAttnOpBuilder`、`_FlashAttnOpBuilder` |
| 对外Python接口名 | 用户直接调用的函数名，体现使用语义，不带`npu_`前缀 | `flash_attn` |
| 图模式GE op函数名 | 与GE算子`op_type`一致的大驼峰 | `FlashAttentionScore` |
| 图模式Converter函数名 | `convert_` + schema算子名 | `convert_flash_attn` |

命名要点：

- **不带`npu_`前缀**：对外算子名与api接口统一使用算子语义名（小写蛇形），不加`npu_`等后端前缀；schema名、C++函数名、pybind导出名三者必须与该名字完全一致，否则JIT编译产物无法被正确调用。
- **接口名体现语义**：对外函数名应贴近业务语义。无论是纯透传aclnn接口的算子（如`flash_attn`），还是封装了结构体构造、参数整理等额外逻辑的接口，均采用语义化命名。
- **aclnn接口名独立**：底层aclnn接口沿用CANN命名（大驼峰，如`aclnnFlashAttentionScore`），与对外算子名解耦；C++ wrapper内通过`ACLNN_CMD(aclnnFlashAttentionScore, ...)`调用。
- **版本后缀**：同一算子的不同迭代版本以`_v2`、`_v3`等后缀区分，schema名、文件名、Converter名需同步带上版本后缀（如`flash_attn_v2`、`graph_convert_flash_attn_v2.py`）。
- **辅助/工具接口**：与主算子配套的工具函数采用动宾语义命名，如`get_flash_attn_workspace_size`。

### 2.2 文件命名

- 统一使用**小写蛇形命名**（snake_case），单词以`_`连接，禁止使用大写、驼峰或连字符。
- 同一算子的各层文件**主名保持一致**，仅靠目录和前缀区分职责：
  - Python前端：`ops/${op_api}.py`，如`flash_attn.py`；
  - C++后端：`ops/csrc/${op_api}.cpp`，主名与Python前端一致，如`flash_attn.cpp`；
  - 图模式：`ops/graph_convert/graph_convert_${op_api}.py`，统一加`graph_convert_`前缀，如`graph_convert_flash_attn.py`；
  - 文档：`docs/zh/${op_api}.md`，如`flash_attn.md`。
- 文件主名应与该文件主要导出的算子语义对应；带版本的算子文件名需带版本后缀（如`flash_attn_v2.cpp`）。
- 公共头文件放在`common/inc/`下，按能力域命名（如`aclnn_common.h`、`hccl_common.h`）。

### 2.3 标识符命名

#### Python 标识符

- **函数/变量/参数**：小写蛇形（snake_case），如`head_num`、`scale_value`、`input_layout`。
- **类名**：大驼峰（PascalCase），如`OpBuilder`、`FlashAttnOpBuilder`。
- **模块级常量**：全大写蛇形（UPPER_SNAKE_CASE），如`AS_LIBRARY`、`ASCEND_HOME_PATH`、`TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP`。
- **模块内部私有符号**：以单下划线`_`前缀标识，如`_flash_attn_op_builder`、`_op_module`、`_flash_attn`、`_TORCHAIR_AVAILABLE`。
- **类型注解**：对外接口与关键内部函数应带类型注解（`from typing import Optional, Tuple, List`），可选参数统一用`Optional[...]`，例如：
  ```python
  def flash_attn(
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      atten_mask: Optional[torch.Tensor] = None,
      scale_value: float = 1.0,
      head_num: int = 1,
      input_layout: str = "BSH",
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  ```
- **参数命名一致性**：同一算子在schema、meta、dispatcher、对外函数、Converter中的同义参数应使用相同的名字（如`head_num`、`scale_value`、`input_layout`），避免在不同层出现不一致写法。

#### C++ 标识符

- **函数/局部变量/参数**：小写蛇形，如`head_num`、`scale_value`、`input_layout_ptr`。
- **命名空间**：算子实现统一置于`namespace op_api`内。
- **常量**：`const`/`constexpr`常量使用全大写蛇形或大驼峰，如`const int DIM_THREE = 3;`、`kATenScalarTypeToAclDataTypeTable`。
- **类型别名/结构体**：大驼峰，如`TensorWrapper`、`TensorListWrapper`。
- **入参类型约定**：
  - 必选Tensor用`const at::Tensor &`；
  - 可选Tensor用`const c10::optional<at::Tensor> &`；
  - Tensor列表用`const std::vector<at::Tensor> &`，可选列表用`const c10::optional<std::vector<at::Tensor>> &`；
  - 整型属性用`int64_t`，可选整型属性用`c10::optional<int64_t>`；浮点属性用`double`；字符串属性用`std::string`。

#### Schema 标识符（算子签名）

- 参数名采用小写蛇形，与Python/C++层一致。
- 用`*`分隔位置参数与关键字参数：`*`之前为必选的位置参数，之后为可选的关键字参数（带默认值）。
- 可选参数以`?`标注并给出默认值，如`Tensor? atten_mask=None`、`int? head_num=1`；列表用`Tensor[]`，可选列表用`Tensor[]?`。
- 多输出用元组表示，如`-> (Tensor, Tensor, Tensor, Tensor)`。

  以`flash_attn`为例的schema：

  ```
  flash_attn(Tensor query, Tensor key, Tensor value, *, Tensor? atten_mask=None,
      float scale_value=1.0, int head_num=1, str input_layout="BSH")
      -> (Tensor, Tensor, Tensor, Tensor)
  ```

## 3. 各层实现规范

### 3.1 C++ 后端（`ops/csrc/${op_api}.cpp`）

负责把PyTorch张量桥接到aclnn C-API，规范要点：

1. 文件头部包含`#include <torch/extension.h>`与`#include "aclnn_common.h"`，实现置于`namespace op_api`。
2. 函数签名与schema严格对应：必选/可选参数类型按[2.3 C++入参类型约定](#c-标识符)选择。
3. **入参校验**：使用`TORCH_CHECK(cond, msg...)`校验shape、dtype、维度、取值范围等，错误信息要可读且包含实际值，例如：
   ```cpp
   TORCH_CHECK((head_num > 0), "The head_num should be greater than 0, current is: ", head_num);
   TORCH_CHECK((query.scalar_type() == key.scalar_type()), "query and key should have the same dtype.");
   ```
4. **设置DeviceGuard（关键）**：在申请输出张量之前，必须先根据输入张量设置`c10::OptionalDeviceGuard`，把当前NPU设备切到输入张量所在设备，并用`{}`作用域把「DeviceGuard +输出申请」包在一起（否则非默认卡调用时输出张量会落到错误设备，导致device不一致）：
   ```cpp
   at::Tensor attention_out{nullptr};
   {
       auto local_device = c10::Device(query.device());
       const c10::OptionalDeviceGuard device_guard(local_device);
       attention_out = at::empty(query.sizes(), query.options());
       // ... 其余输出 ...
   }
   ```
5. **输出张量手动申请**：在DeviceGuard生效的作用域内，按meta推导的shape/dtype用`at::empty(...)`申请输出（标准PyTorch实践），dtype通过`query.options().dtype(...)`指定。
6. **拉起kernel**：使用`ACLNN_CMD(aclnn接口名, 入参..., 出参...)`宏调用aclnn接口（如`ACLNN_CMD(aclnnFlashAttentionScore, ...)`），入参顺序需与aclnn接口定义一致；该宏自动完成类型转换、workspace申请、stream下发与资源释放。
7. **导出绑定**：通过`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`将C++函数绑定为与schema同名的Python接口（`m.def("flash_attn", &flash_attn, "flash_attn");`）。
8. 魔数（如维度数`3`、默认dtype枚举）应以具名常量表达，避免裸写字面量。

   C++ wrapper的典型骨架：

   ```cpp
   std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> flash_attn(
       const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
       const c10::optional<at::Tensor> &atten_mask,
       double scale_value, int64_t head_num, std::string input_layout)
   {
       // 3. 入参校验
       TORCH_CHECK((head_num > 0), "The head_num should be greater than 0, current is: ", head_num);

       at::Tensor attention_out{nullptr};
       {
           // 4. DeviceGuard：必须在申请输出之前，作用域包住输出申请
           auto local_device = c10::Device(query.device());
           const c10::OptionalDeviceGuard device_guard(local_device);
           // 5. 申请输出张量
           attention_out = at::empty(query.sizes(), query.options());
           // ... 其余输出 ...
       }

       // 6. 拉起 aclnn kernel
       ACLNN_CMD(aclnnFlashAttentionScore, query, key, value, atten_mask,
           scale_value, head_num, input_layout.data(), /* outputs */ attention_out);

       return std::make_tuple(/* ... */);
   }
   ```

### 3.2 Python 前端（`ops/${op_api}.py`）

负责JIT编译管理、schema/meta注册与对外接口封装：

1. **OpBuilder子类**：继承`OpBuilder`，在`__init__`中以`super().__init__("<schema算子名>")`传入算子名，并实现三个抽象方法：
   - `sources()`：返回相对`cann_ops_transformer`包根的C++源文件路径列表，如`['ops/csrc/flash_attn.cpp']`；
   - `schema()`：返回算子schema字符串（见[2.3 Schema标识符](#schema-标识符算子签名)）；
   - `register_meta()`：用`@impl(AS_LIBRARY, self.name, "Meta")`注册Meta实现，仅做shape/dtype推导，不触碰真实NPU计算（FakeTensor/图模式必需）。Meta中同样可用`torch._check(...)`做约束校验。
2. **实例化与编译**：模块加载时实例化builder并`load()`触发编译：
   ```python
   _flash_attn_op_builder = _FlashAttnOpBuilder()
   _op_module = _flash_attn_op_builder.load()
   ```
3. **PrivateUse1 dispatcher**：用`@impl(AS_LIBRARY, builder.name, "PrivateUse1")`注册NPU后端实现，函数体透传到编译产物`_op_module.<算子名>(...)`（如`_op_module.flash_attn(...)`）。`PrivateUse1`是PyTorch为自定义NPU后端预留的dispatch key。
4. **对外接口**：提供面向用户的函数`flash_attn(...)`，负责参数整理、默认值处理等，最终调用dispatcher实现。
5. **对外api必须书写注释（docstring）**：每个对外导出的接口都要有docstring，至少覆盖「功能说明、各参数含义/shape/dtype/取值范围、返回值说明」，必要时给出简短调用示例。docstring内容应与`docs/zh/${op_api}.md`保持一致，便于IDE提示与`help()`查看。例如：
   ```python
   def flash_attn(
       query: torch.Tensor,
       key: torch.Tensor,
       value: torch.Tensor,
       atten_mask: Optional[torch.Tensor] = None,
       scale_value: float = 1.0,
       head_num: int = 1,
       input_layout: str = "BSH",
   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
       """FlashAttention 前向计算，封装 aclnnFlashAttentionScore。

       Args:
           query (Tensor): 查询张量，shape 由 input_layout 决定（如 BSH），dtype 支持 float16/bfloat16。
           key (Tensor): 键张量，dtype 与 query 一致。
           value (Tensor): 值张量，dtype 与 query 一致。
           atten_mask (Tensor, optional): 注意力掩码，默认 None 表示不使用。
           scale_value (float): 缩放系数，默认 1.0。
           head_num (int): 单卡 head 数，即 query 的 N 轴长度，默认 1。
           input_layout (str): 输入数据排布，支持 "BSH"/"BNSD" 等，默认 "BSH"。

       Returns:
           Tuple[Tensor, Tensor, Tensor, Tensor]:
               softmax_max、softmax_sum、softmax_out、attention_out。
       """
   ```
6. Meta实现、dispatcher、对外函数三者的参数顺序与默认值必须与schema一致。

### 3.3 图模式 Converter（`ops/graph_convert/graph_convert_${op_api}.py`）

负责在torchair图模式（GE）下把aten算子转换为GE节点：

1. **可选依赖保护**：torchair相关import统一包在`try/except ImportError`中，用`_TORCHAIR_AVAILABLE`标志位控制，避免在无torchair环境下导入失败。
2. **GE op函数**：定义与`op_type`同名的大驼峰函数（如`FlashAttentionScore`），通过docstring写明`REG_OP`的IR定义（INPUT/DYNAMIC_INPUT/OPTIONAL_INPUT/OUTPUT/ATTR等），并组织`inputs`/`attrs`/`outputs`后调用`ge_op(...)`，IR通过`IrDef(...)`链式声明。
3. **Converter注册**：用`@register_fx_node_ge_converter(torch.ops.cann_ops_transformer.flash_attn.default)`装饰`convert_flash_attn`函数，其参数顺序与schema完全一致，函数体调用上面的GE op函数。
4. 在`ops/__init__.py`中导出Converter（如`convert_flash_attn`），确保注册逻辑被执行。

### 3.4 对外导出（`ops/__init__.py` 与包根 `__init__.py`）

对外导出分两级，两处都需新增import：

1. **`ops/__init__.py`（子包层）**：每个新增算子的对外接口与Converter都需在此显式import导出，导入即触发schema/meta/converter注册：
   ```python
   from .flash_attn import flash_attn
   from .graph_convert.graph_convert_flash_attn import convert_flash_attn
   ```
   同一算子若导出多个符号，使用括号分组的多行import。
2. **`cann_ops_transformer/__init__.py`（包根层）**：除已有的`from . import ops`触发注册外，还需把对外接口import到包根命名空间，使用户可直接通过`cann_ops_transformer.<接口名>`访问，而不必写完整的`cann_ops_transformer.ops.<接口名>`路径：
   ```python
   from . import ops
   from .ops import flash_attn

   __all__ = ["flash_attn"]
   ```
   建议在包根维护`__all__`，显式列出对外导出的接口名，便于管理可见接口集合。

完成两级导出后，用户既可`from cann_ops_transformer import flash_attn`，也可`cann_ops_transformer.ops.flash_attn(...)`调用；导入主包即完成schema/meta/converter注册。

## 4. 文档规范（`docs/zh/${op_api}.md`）

每个对外算子api需配套一份中文文档，建议章节顺序与已有算子文档（如`flash_attn.md`）对齐：

1. **标题**：算子名（特殊字符如`_`需转义为`\_`）。
2. **产品支持情况**：表格列出支持的产品形态（如`Ascend 950PR/Ascend 950DT`）及是否支持。
3. **功能说明**：API功能概述 +计算公式（数学表达用LaTeX），并说明各符号与参数的对应关系。
4. **函数原型**：代码块给出完整函数签名，含默认值与`*`分隔。
5. **参数说明**：逐个参数说明「必选/可选、语义、shape、dtype、数据格式（如`$ND$`）、是否支持非连续Tensor、取值范围/约束」；可选参数标注默认值与「暂不支持」说明。
6. **输出说明**：逐个输出说明shape、dtype、格式等。
7. **约束说明**：分类列出参数一致性约束、shape/取值范围约束、量化场景约束等；通信类算子还需列出通信域约束。
8. **配套接口说明**：若算子需与其他接口配套使用，补充其原型、参数与输出说明。
9. **调用示例**：给出单算子模式（必要时含多卡/通信初始化）的完整可运行示例；图模式若暂不支持需明确标注「图模式调用（暂不支持）」。

## 5. 编码通用约束

- **许可证头**：所有新增源文件（`.py`/`.cpp`/`.h`）必须包含Huawei版权与CANN Open Software License Agreement Version 2.0许可证头，年份填当年。Python/脚本用`#`注释，C++用`//`或`/* */`。
- **接口注释**：对外api接口必须书写docstring（功能、参数、返回值），见[3.2](#32-python-前端opsop_apipy)；C++ wrapper关键逻辑（校验、DeviceGuard、aclnn调用）也应有简要注释。
- **C++层DeviceGuard（关键）**：调用aclnn的C++ wrapper中，必须在申请输出张量之前用`c10::OptionalDeviceGuard`（构造自`c10::Device(输入张量.device())`）把设备切到输入张量所在设备，详见[3.1](#31-c-后端opscsrcop_apicpp)。
- **参数校验前置**：Python侧用`torch._check(cond, lambda: f"...{var=}...")`，C++侧用`TORCH_CHECK(cond, msg...)`；错误信息需包含变量实际值，便于定位。
- **错误码**：Python侧可结合`torch_npu.utils._error_code`的`ErrCode`/`ops_error`输出规范错误码，如`f"... {ops_error(ErrCode.VALUE)}."`。
- **避免魔数**：维度数、dtype枚举值等以具名常量表达，并在文档/注释中说明枚举含义（如`23 → float8_e5m2`、`24 → float8_e4m3fn`）。
- **公共能力复用**：类型转换、`ACLNN_CMD`、通信域处理等优先复用`common/inc`下的公共头，不在各算子中重复实现。
- **一致性自检**：提交前确认schema、C++ wrapper、Meta、dispatcher、对外函数、Converter、文档七处的算子名、参数名、参数顺序、默认值保持一致。
