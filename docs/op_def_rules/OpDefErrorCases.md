# OpDef 错误案例: chunk_gated_delta_rule FP32 state 支持

## 背景

`chunk_gated_delta_rule` 算子在支持 FP32 state 输入的过程中，连续引入了两个错误，最终由 `0b7b0a0b5fa9680d34526d71c07d0b0308c8193a` 修复。本文档记录这两个错误及其根因。

## 错误 1: 三件套计数不一致

**引入提交**: `a6faf34af`（2026-07-02）

**改动内容**: 为 `initial_state`/`final_state` 增加 FP32 支持，将 `DataType` 从 1 个值改为 2 个值，但未同步修改 `Format` 和 `UnknownShapeFormat`。

```cpp
this->Input("initial_state")
    .ParamType(REQUIRED)
    .DataType({ge::DT_BF16, ge::DT_FLOAT})     // 2 个值 ← 改了
    .Format({ge::FORMAT_ND})                    // 1 个值 ← 未同步
    .UnknownShapeFormat({ge::FORMAT_ND});       // 1 个值 ← 未同步
```

**违反规则**: I5（三件套计数一致）

**根因**: 开发者只关注了需要新增 FP32 类型，未意识到三件套是一个整体，改一必须改三。`DataType({A, B})` 声明的是支持的 (数据类型, 格式) 组合列表，每个 `DataType[i]` 需要对应的 `Format[i]` 和 `UnknownShapeFormat[i]`，与 `AddConfig` 的 SoC 数量无关。

## 错误 2: DataTypeList 误用

**引入提交**: `2d40fd26c`（2026-07-07）

**改动内容**: 遇到计数不匹配的报错后，将 `DataType` 改为 `DataTypeList`，试图"绕过"计数约束。

```cpp
this->Input("initial_state")
    .ParamType(REQUIRED)
    .DataTypeList({ge::DT_BF16, ge::DT_FLOAT})  // 错: 语义变为 tensor list
    .Format({ge::FORMAT_ND})                     // 应搭配 FormatList
    .UnknownShapeFormat({ge::FORMAT_ND});
```

**违反规则**: I6（DataTypeList 搭配 Format 为可疑用法）

**根因**: 开发者误诊了问题根因——将"计数不匹配"错当成"API 选错"，用语义完全不同的 `DataTypeList` 替代 `DataType`，导致该输入被声明为变长 tensor list，而非声明支持的类型组合。`DataType` 声明支持的 (数据类型, 格式) 组合（适用于所有 SoC），`DataTypeList` 声明该输入为 tensor list（动态数量输入），两者语义完全不同。

## 错误演变时间线

| 提交 | 日期 | 改动 | 问题 |
|------|------|------|------|
| `a6faf34af` | Jul 2 | state DataType 1→2 值，Format/UnknownShapeFormat 未同步 | I5 违规 |
| `2d40fd26c` | Jul 7 | DataType → DataTypeList | I6 违规（误诊） |
| `9e36de07b` | Jul 16 | 合入 main（MR !8738） | bug 进入主线 |
| `0b7b0a0b5` | Jul 17 | 修复（MR !8841） | 恢复 DataType 并补齐三件套 |

## 关键概念澄清: DataType 列表与 AddConfig 的关系

`DataType({A, B})` 声明的是该输入**支持的数据类型组合**，每个 `(DataType[i], Format[i])` 是一个支持的类型对，适用于所有注册的 SoC。`AddConfig` 声明算子支持哪些 SoC。两者相互独立：

- 官方模板 `add_example_def.cpp`: `DataType({ge::DT_FLOAT, ge::DT_INT32})` 有 2 个值，`AddConfig` 有 3 个 SoC（ascend910b/ascend910_93/ascend950），注释为"支持数据类型"
- 开发指南 `aicore_develop_guide.md` 第 823 行示例: `DataType({ge::DT_FLOAT16, ge::DT_FLOAT})` 有 2 个值，`AddConfig` 有 4 个 SoC

**早期版本的静态检查脚本曾包含一条 I7 规则**（"三件套计数应等于 AddConfig 调用数或为 1"），该规则基于错误的假设，在官方模板和开发指南示例上均会产生误报，已移除。

## 避免方法

### Code Review 检查清单

修改 `_def.cpp` 时，Reviewer 必须检查以下项目：

1. **三件套同步 (I5)**: `DataType` 的值数量变化时，`Format` 和 `UnknownShapeFormat` 是否同步变化
2. **API 语义 (I6)**: diff 中出现 `DataType` → `DataTypeList` 或反向切换时，必须追问该输入是否真的是 tensor list
3. **参照模板**: 不确定时参照 `examples/add_example/op_host/add_example_def.cpp` 或同系列算子的写法

### 开发者自检流程

1. 修改 `DataType`/`Format`/`UnknownShapeFormat` 中任一项时，检查另外两项的值数量是否一致
2. 遇到计数不匹配报错时，**先排查计数一致性**，而非盲目更换 API 方法
3. 提交前运行静态检查脚本：`python scripts/ci/check_op_def.py <file_or_dir>`
