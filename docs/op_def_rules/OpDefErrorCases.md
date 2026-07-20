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

**违反规则**: R1（三件套计数一致）

**根因**: 开发者只关注了需要新增 FP32 类型，未意识到三件套是一个整体，改一必须改三。

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

**违反规则**: R2（DataTypeList 搭配 Format 为可疑用法）

**根因**: 开发者误诊了问题根因——将"计数不匹配"错当成"API 选错"，用语义完全不同的 `DataTypeList` 替代 `DataType`，导致该输入被声明为变长 tensor list，而非按配置声明类型。

## 错误演变时间线

| 提交 | 日期 | 改动 | 问题 |
|------|------|------|------|
| `a6faf34af` | Jul 2 | state DataType 1→2 值，Format/UnknownShapeFormat 未同步 | R1 违规 |
| `2d40fd26c` | Jul 7 | DataType → DataTypeList | R2 违规（误诊） |
| `9e36de07b` | Jul 16 | 合入 main | bug 进入主线 |
| `0b7b0a0b5` | Jul 17 | 修复 | 恢复 DataType 并补齐三件套 |

## 避免方法

### Code Review 检查清单

修改 `_def.cpp` 时，Reviewer 必须检查以下项目：

1. **三件套同步**: `DataType` 的值数量变化时，`Format` 和 `UnknownShapeFormat` 是否同步变化
2. **API 语义**: diff 中出现 `DataType` → `DataTypeList` 或反向切换时，必须追问该输入是否真的是 tensor list
3. **配置计数**: 三件套计数是否等于 `AddConfig` 调用数（或为 1 广播）
4. **参照模板**: 不确定时参照 `examples/add_example/op_host/add_example_def.cpp` 或同系列算子的写法

### 开发者自检流程

1. 修改 `DataType`/`Format`/`UnknownShapeFormat` 中任一项时，检查另外两项的值数量是否一致
2. 遇到计数不匹配报错时，**先排查计数一致性**，而非盲目更换 API 方法
3. 提交前运行静态检查脚本：`python scripts/ci/check_op_def.py <file_or_dir>`
