# CANN 算子 OpDef 规范

<!-- TOC -->

- [说明](#说明)
- [适用范围](#适用范围)
- [规则总览](#规则总览)
- [规则出处](#规则出处)
- [结构规则](#结构规则)
- [Input/Output 规则](#inputoutput-规则)
- [取值校验规则](#取值校验规则)
- [AICore 配置规则](#aicore-配置规则)
- [Attr 规则](#attr-规则)
- [错误案例](#错误案例)
- [静态检查脚本](#静态检查脚本)

<!-- /TOC -->

## 说明

本规范针对算子信息库 `*_def.cpp` 文件中 `OpDef` 类的声明，约束结构、Input/Output、取值、AICore 配置、Attr 等方面的使用规则。参与 CANN 开源社区项目的开发者在修改 `_def.cpp` 时需遵循本规范。

## 适用范围

CANN 相关开源仓的 `_def.cpp` 算子信息库文件代码检视。

## 规则总览

| 规则 | 级别 | 说明 |
|------|------|------|
| S1 | ERROR | 文件必须 `#include "register/op_def_registry.h"` |
| S2 | ERROR | 必须在 `namespace ops` 中定义 |
| S3 | ERROR | 必须有 `class X : public OpDef` |
| S4 | ERROR | 必须有 `OP_ADD(ClassName)` 注册，且类名与注册名一致 |
| S5 | ERROR | 禁止 `SetInferShape`/`SetTiling`/`IMPL_OP_OPTILING`/`IMPL_OP_INFERSHAPE` |
| I1 | ERROR | 每个 Input/Output 必须含 `.ParamType(...)` |
| I2 | ERROR | 每个 Input/Output 必须含 `.DataType(...)` 或 `.DataTypeList(...)` |
| I3 | ERROR | 每个 Input/Output 必须含 `.Format(...)` 或 `.FormatList(...)`（AICPU 算子放宽） |
| I4 | ERROR | ParamType 取值必须为 `REQUIRED`/`OPTIONAL`/`DYNAMIC` |
| I5 | ERROR | DataType/Format/UnknownShapeFormat 三件套计数必须一致 |
| I6 | WARNING | DataTypeList 搭配 Format（缺少 FormatList）为可疑用法 |
| V1 | ERROR | `ge::DT_*` 必须为合法数据类型 |
| V2 | ERROR | `ge::FORMAT_*` 必须为合法格式 |
| A1 | ERROR | 必须有至少一个 `AddConfig` 调用（AICPU 算子除外） |
| A2 | WARNING | AddConfig 的 SOC 名应为已知合法值 |
| T1 | ERROR | AttrType 取值必须为 `REQUIRED`/`OPTIONAL` |
| T2 | WARNING | attr 名不应与保留字冲突 |

## 规则出处

> **在线文档来源**
> - Ascend C 算子开发指南: <https://hiascend.com/document/redirect/CannCommunityOpdevAscendC>
> - 基础数据结构和接口（ge::DataType / ge::Format 枚举）: <https://hiascend.com/document/redirect/CannCommunitybasicopapi>
>   - ge::DataType 枚举: <https://hiascend.com/document/detail/zh/CannCommunitybasicopapi/atlasopapi_07_00514.html>
>   - ge::Format 枚举: <https://hiascend.com/document/detail/zh/CannCommunitybasicopapi/atlasopapi_07_00515.html>
> - Ascend C 算子开发接口: <https://hiascend.com/document/redirect/CannCommunityAscendCApi>

| 规则 | 出处类型 | 具体出处 |
|------|---------|---------|
| S1-S4 | 官方模板 | `examples/add_example/op_host/add_example_def.cpp`（在线: Ascend C 算子开发指南 > 算子代码开发 > 算子工程目录结构） |
| S5 | 开发指南 | `docs/zh/develop/aicore_develop_guide.md` 第 810 行、第 826-828 行、第 838 行：迁移至 `_def.cpp` 时必须去掉 `SetInferShape` 和 `SetTiling`（在线: Ascend C 算子开发指南） |
| I1, I4 | INI 校验 | `cmake/scripts/util/parse_ini_to_json.py:check_op_info`（第 328-402 行）：input/output 必须含 paramType，取值必须为 `["dynamic", "optional", "required"]` |
| I2, I3 | INI 校验 | `cmake/scripts/util/parse_ini_to_json.py:check_type_format`（第 291-325 行）：校验 dtype 和 format |
| I5 | INI 校验 + 代码约定 | `parse_ini_to_json.py:check_type_format`（第 317-324 行）：dtype 数量必须等于 format 数量；C++ OpDef 格式为隐式约定 |
| I6 | 代码库约定 | 全仓库 `_def.cpp` 用法归纳：`DataTypeList` 搭配 `FormatList` 用于 tensor list |
| V1 | INI 校验 | `parse_ini_to_json.py` 第 45-83 行 `DTYPE_LIST`（37 种合法类型）；C++ 枚举见 `include/graph/types.h` 中 `ge::DataType` |
| V2 | INI 校验 | `parse_ini_to_json.py` 第 84-131 行 `FORMAT_LIST`（47 种合法格式）；C++ 枚举见 `include/graph/types.h` 中 `ge::Format` |
| A1 | CI 校验 | `scripts/ci/parse_changed_ops.py:check_soc_registered`（第 181-196 行） |
| A2 | 代码库统计 | `scripts/ci/gen_ops_soc.py:parse_foreach_config`（第 248-283 行）+ 全仓库 AddConfig SOC 统计 |
| T1 | INI 校验 | `parse_ini_to_json.py:check_attr_dict`（第 223-243 行）：`ATTR_PARAMTYPE_LIST = ["optional", "required"]`（第 34 行） |
| T2 | 解析器校验 | `cmake/scripts/util/opdesc_parser.py` 第 49-117 行 `CONFLICT_KEYWORDS`，由 `parse_attr_list`（第 265-272 行）在构建时抛异常 |

## 结构规则

### S1: 必须 include op_def_registry.h

```cpp
#include "register/op_def_registry.h"
```

### S2-S4: 必须在 namespace ops 中定义 OpDef 子类并注册

```cpp
namespace ops {
class MyOp : public OpDef {
    // ...
};
OP_ADD(MyOp);  // 类名必须与 OP_ADD 注册名一致
} // namespace ops
```

### S5: 禁止 SetInferShape / SetTiling

`_def.cpp` 只定义算子信息库，不应包含 InferShape 和 Tiling 逻辑。这些应分别放在 `_infershape.cpp` 和 `_tiling.cpp` 中。

> **出处**: `docs/zh/develop/aicore_develop_guide.md` 第 810 行原文："将原有${op_name}.cpp中算子信息库内容独立迁移至该文件，需要去掉SetInferShape和SetTiling内容。" 第 826 行标注 `// 需要去掉SetInferShape`，第 828 行标注 `// 需要去掉SetTiling`。
>
> **在线文档**: [Ascend C 算子开发指南](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)

```cpp
// 禁止出现在 _def.cpp
this->SetInferShape(...);
this->AICore().SetTiling(...);
```

## Input/Output 规则

### I1-I4: 必填项与取值

每个 Input/Output 必须含 `ParamType`、`DataType`（或 `DataTypeList`）、`Format`（或 `FormatList`）。

> **出处**: `cmake/scripts/util/parse_ini_to_json.py:check_op_info`（第 336-338 行）定义 `required_op_input_info_keys = ["paramType", "name"]`，`param_type_valid_value = ["dynamic", "optional", "required"]`（第 338 行）；`check_type_format`（第 291-325 行）校验 dtype 和 format。

```cpp
this->Input("x")
    .ParamType(REQUIRED)                              // I1, I4: 必填，取值 REQUIRED/OPTIONAL/DYNAMIC
    .DataType({ge::DT_BF16, ge::DT_BF16})             // I2: 必填
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})           // I3: 必填
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
```

**ParamType 取值说明**（对应 INI 中的小写形式）:

| C++ 值 | INI 值 | 含义 |
|--------|--------|------|
| `REQUIRED` | `required` | 必选输入/输出 |
| `OPTIONAL` | `optional` | 可选输入/输出 |
| `DYNAMIC` | `dynamic` | 动态数量输入/输出 |

### I5: 三件套计数一致

`DataType`、`Format`、`UnknownShapeFormat` 三者元素数量必须一致。

> **出处**: `cmake/scripts/util/parse_ini_to_json.py:check_type_format`（第 317-324 行）：`if op_info_dtype_num != op_info_format_num: is_valid = False`。
>
> **语义说明**: `DataType({A, B})` 声明该输入支持的 (数据类型, 格式) 组合列表，每个 (DataType[i], Format[i]) 是一个支持的类型对。计数与 `AddConfig` 的 SoC 数量**无关**——类型列表适用于所有注册的 SoC。官方模板 `add_example_def.cpp` 中 `DataType({ge::DT_FLOAT, ge::DT_INT32})` 有 2 个值但 `AddConfig` 有 3 个 SoC（ascend910b/ascend910_93/ascend950），注释为"支持数据类型"。
>
> **Format vs FormatList 计数规则差异**:
> - `Format({A, B})`：逐类型声明格式，计数**必须**等于 `DataType` 计数。
> - `FormatList({ND})`（计数为 1）：广播语义，单值适用于所有 `DataType`，不要求计数等于 `DataType` 计数。例如 `prompt_flash_attention_def.cpp` 中 `DataType` 有 32 个值，`FormatList({ge::FORMAT_ND})` 仅 1 个值。
> - `FormatList({ND, ND, ...})`（计数 > 1）：逐类型声明格式，计数**必须**等于 `DataType` 计数。
> - `UnknownShapeFormat` 与 `Format`/`FormatList` 同时存在时，计数应一致（脚本会校验）。

```cpp
// 正确: 三者均为 2
this->Input("x")
    .DataType({ge::DT_BF16, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

// 错误: DataType 为 2，Format/UnknownShapeFormat 为 1
this->Input("x")
    .DataType({ge::DT_BF16, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

### I6: DataTypeList 搭配 Format 为可疑用法

`DataType` 与 `DataTypeList` 语义不同：

| API | 语义 | 常见搭配 | 使用场景 |
|-----|------|---------|---------|
| `DataType({A, B})` | 声明该输入支持的 (数据类型, 格式) 组合列表 | `Format`（计数一致）或 `FormatList`（广播，计数不限） | 普通 tensor |
| `DataTypeList({A})` | 声明该输入为变长类型列表（tensor list） | `FormatList` | 动态数量输入 |

`DataTypeList` 搭配 `Format`（而非 `FormatList`）为可疑用法，可能是误将 `DataType` 写成了 `DataTypeList`。

> **出处**: 全仓库 `_def.cpp` 用法归纳。`DataTypeList` + `FormatList` 是 tensor list 的标准写法（如 `mc2/moe_ep_combine/op_host/moe_ep_combine_def.cpp` 中 `Input("context").DataTypeList(...).FormatList(...)`）；`DataTypeList` + `Format`（无 `FormatList`）在历史 bug 中出现过（见 [错误案例](./OpDefErrorCases.md)），需人工确认是否真的是 tensor list。

### 取值校验规则

### V1: 合法数据类型

`ge::DT_*` 必须为合法值，包括：`DT_FLOAT`, `DT_FLOAT16`, `DT_BF16`, `DT_INT8`, `DT_INT16`, `DT_INT32`, `DT_UINT8`, `DT_UINT16`, `DT_UINT32`, `DT_BOOL`, `DT_INT64`, `DT_UINT64`, `DT_QINT8`, `DT_QINT16`, `DT_QINT32`, `DT_QUINT8`, `DT_QUINT16`, `DT_DOUBLE`, `DT_COMPLEX32`, `DT_COMPLEX64`, `DT_COMPLEX128`, `DT_STRING`, `DT_RESOURCE`, `DT_DUAL`, `DT_DUAL_SUB_INT8`, `DT_DUAL_SUB_UINT8`, `DT_STRING_REF`, `DT_INT4`, `DT_UINT1`, `DT_HIFLOAT8`, `DT_FLOAT8_E4M3FN`, `DT_FLOAT8_E5M2`, `DT_FLOAT8_E8M0`, `DT_FLOAT4_E2M1`, `DT_FLOAT4_E1M2`, `DT_INT2`, `DT_UNDEFINED`。

> **出处**: `cmake/scripts/util/parse_ini_to_json.py` 第 45-83 行 `DTYPE_LIST`（INI 小写形式，如 `bfloat16` 对应 C++ `DT_BF16`）；C++ 枚举定义于 `include/graph/types.h` 中 `ge::DataType`。
>
> **在线文档**: [ge::DataType 枚举](https://hiascend.com/document/detail/zh/CannCommunitybasicopapi/atlasopapi_07_00514.html)

### V2: 合法格式

`ge::FORMAT_*` 必须为合法值，包括：`FORMAT_ND`, `FORMAT_NCHW`, `FORMAT_NHWC`, `FORMAT_NC1HWC0`, `FORMAT_FRACTAL_Z`, `FORMAT_FRACTAL_NZ` 等完整列表见 `parse_ini_to_json.py` 第 84-131 行。

> **出处**: `cmake/scripts/util/parse_ini_to_json.py` 第 84-131 行 `FORMAT_LIST`（47 种合法格式）；C++ 枚举定义于 `include/graph/types.h` 中 `ge::Format`。
>
> **在线文档**: [ge::Format 枚举](https://hiascend.com/document/detail/zh/CannCommunitybasicopapi/atlasopapi_07_00515.html)

## AICore 配置规则

### A1: 必须有 AddConfig

```cpp
this->AICore().AddConfig("ascend910b");
this->AICore().AddConfig("ascend910_93");
```

> **出处**: `scripts/ci/parse_changed_ops.py:check_soc_registered`（第 181-196 行）通过正则 `this->AICore().AddConfig("<soc>")` 检测 SoC 注册。AICPU 算子使用 `ApplyTransformerAicpuDefaultCfg` 而非 `AddConfig`，脚本对此放宽。

### A2: SOC 名应为已知合法值

已知合法 SOC：`ascend910b`, `ascend910_93`, `ascend950`, `ascend310p`, `ascend310b`, `ascend910`, `ascend610lite`, `kirinx90`, `kirin9030`, `mc62`, `ascend910_55`, `bs9sx1a`。

> **出处**: `scripts/ci/gen_ops_soc.py:parse_foreach_config`（第 248-283 行）定义了 SoC 配置映射；全仓库 `*_def.cpp` 中 `AddConfig` 调用的 SOC 统计。本仓库实际使用的 SOC 为：`ascend910_93`, `ascend910b`, `ascend950`, `ascend310p`, `kirinx90`, `kirin9030`, `mc62`, `ascend910_55`。

## Attr 规则

### T1: AttrType 取值

```cpp
this->Attr("scale").AttrType(OPTIONAL).Float(1.0);  // REQUIRED 或 OPTIONAL
```

> **出处**: `cmake/scripts/util/parse_ini_to_json.py` 第 34 行 `ATTR_PARAMTYPE_LIST = ["optional", "required"]`；`check_attr_dict`（第 237-239 行）校验 attr 的 paramType。

### T2: attr 名不应与保留字冲突

保留字列表见 `opdesc_parser.py` 第 49-117 行 `CONFLICT_KEYWORDS`，包括 Python 关键字、框架保留字等。

> **出处**: `cmake/scripts/util/opdesc_parser.py` 第 49-117 行 `CONFLICT_KEYWORDS`（含 Python 33 个保留字 + 框架内部标识符）；`parse_attr_list`（第 265-272 行）在构建时检测冲突并抛异常。

## 错误案例

详细的错误案例和避免方法见 [OpDefErrorCases.md](./OpDefErrorCases.md)。

## 静态检查脚本

覆盖以上所有规则的静态检查脚本位于 `scripts/ci/check_op_def.py`。

### 使用方法

```bash
# 检查单个文件
python scripts/ci/check_op_def.py attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_def.cpp

# 检查目录
python scripts/ci/check_op_def.py attention/

# 检查多个路径
python scripts/ci/check_op_def.py attention/ mc2/ mamba/

# 在 CI 中集成（检查变更的 _def.cpp 文件）
CHANGED_DEF_FILES=$(git diff --name-only --diff-filter=AM HEAD~1 | grep '_def\.cpp$')
if [ -n "$CHANGED_DEF_FILES" ]; then
    python scripts/ci/check_op_def.py $CHANGED_DEF_FILES
fi
```

### 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 通过（可能有 WARNING 但无 ERROR） |
| 1 | 存在 ERROR |
