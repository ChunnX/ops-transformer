---
name: "doc_valiImpl_verify"
description: "Verifies whether documented operator parameter constraints are fully validated in op_host/op_api code. Invoke when user asks to check doc-to-code validation coverage for an operator."
---

# Doc Validation Implementation Verification

本Skill用于检查算子文档中对于入参的限制描述，是否在op_host/op_tiling/op_api代码中得到了全面校验，并检查tests/ut中是否覆盖了这些校验的测试。

## 输入

用户需提供以下信息：
1. **算子交付件路径**：算子的根目录路径
2. **硬件名称**：A2、A3、A5等相关名称
3. **aclnn入口**：算子的aclnn接口名称

## 输出

{op_name}_doc_valiImpl_verify_result.md，用户如果没有指定路径，就默认在当前路径生成

## 芯片代际映射

| 芯片代际 | 产品名称 | 别名 | 代码中NpuArch枚举 |
|---------|---------|------|------------------|
| arch22/arch32 | 910B、910C | A2、A3；910C：910_93 | NpuArch::DAV_2201 |
| arch35 | 910D | A5；950 | NpuArch::DAV_3510 |

## 工作流

### 第一步：读取文档中的参数限制

1. 在交付件路径下找到 `docs/` 目录中的aclnn接口介绍文档（.md文件）
2. 解析文档中的以下关键信息：
   - **参数说明表格**：提取每个参数的数据类型、shape、format、非连续tensor支持等限制
   - **约束说明章节**：提取所有显式约束条件，包括：
     - rankSize限制
     - shape变量约束（如BS必须整除rankSize）
     - 数值范围约束（如BS/N的INT32_MAX限制、H*rankSize范围）
     - 空tensor支持度
     - 非连续tensor支持度
     - 数据类型约束（含不同芯片代际的差异）
     - 数据类型约束
     - 通信域约束
     - 其他约束
   - **返回值错误码表格**：提取文档中声明的校验错误码和对应描述
   aclnn接口校验才会返回不同的错误码，tiling校验只会返回统一校验码

3. 将每个限制条目化，形成"文档限制清单"，每条包含：
   - 参数名
   - 限制类型（dtype/shape/format/range/空指针/空tensor/非连续/属性值等）
   - 限制内容描述

### 第二步：阅读代码中的校验实现

按照以下顺序阅读代码，逐条核对文档限制清单：

#### 2.1 op_api层（aclnn入口校验）

1. 在交付件路径下找到 `op_api/` 目录
2. 读取与aclnn入口对应的 `.cpp` 和 `.h` 文件
3. 重点关注以下校验函数和模式：
   - 空指针校验
   - 空tensor校验
   - 数据类型校验
   - 格式校验
   - shape校验
   - 转置属性校验
   - 通信域长度校验
   - 非连续tensor处理逻辑
   - 芯片代际分支（`NpuArch::DAV_2201`、`NpuArch::DAV_3510`）
4. 涉及的公共校验函数

#### 2.2 op_host/op_tiling层（Tiling校验）

1. 在交付件路径下找到 `op_host/op_tiling/` 目录
2. 根据芯片代际读取对应子目录：
   - arch22/arch32 → `op_tiling/arch32/` 目录下的文件
   - arch35 → `op_tiling/arch35/` 目录下的文件
3. 重点关注以下校验：
   - `CheckAttrs`：属性校验（group长度、transposeX1、rankSize等）
   - `CheckTensor`：tensor格式校验
   - 输入组合校验
   - 输出校验
   - 其它关键内容校验
4. 同时检查共享tiling工具：
   - `mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.cpp`
   - `mc2/common/op_host/op_tiling/mc2_tiling_utils.cpp`（含CheckRankSize等）

### 第三步：检查tests/ut覆盖情况

1. 在交付件路径下找到 `tests/ut/` 目录
2. 读取以下测试文件：
   - `tests/ut/op_api/test_aclnn_*.cpp`：op_api层UT测试
   - `tests/ut/op_api/test_aclnn_*.csv`：op_api层UT测试数据
   - `tests/ut/op_host/test_*_tiling.cpp`：op_host层UT测试
   - `tests/ut/op_host/test_*_tiling.csv`：op_host层UT测试数据
   - `tests/ut/op_host/arch32/` 和 `tests/ut/op_host/arch35/`：按代际分类的测试
3. 对每条文档限制，检查是否存在对应的异常测试用例

### 第四步：输出校验结果表格

以Markdown表格形式输出，表格结构如下：

| 序号 | 文档限制内容 | 限制类型 | 代码是否实现 | 实现位置 | 实现是否合适 | UT是否覆盖 | 备注 |
|------|------------|---------|---------|------------|---------|------------|-----------|------|
| 1 | x1/x2/output不能为空指针 | 空指针 | 是 | op_api: CheckNotNull | 合适，在入口处校验 | 是 | ... |
| 2 | x1 dtype为FLOAT16/BFLOAT16 | dtype | 是 | op_api: CheckAllDtypesValid | 合适 | 部分 | 缺少BF16的异常测试 |

#### 表格列说明

- **序号**：限制条目编号
- **文档限制内容**：从文档中提取的具体限制描述
- **限制类型**：空指针/dtype/shape/format/range/空tensor/非连续/属性值/其他
- **代码是否实现**：是/否/部分实现
- **实现位置**：具体在哪个文件的哪个函数中实现，格式为 `文件:函数名`，如 `op_api/aclnn_allto_all_matmul.cpp:CheckNotNull`
- **实现是否合适**：合适/不合适（需说明原因），判断标准：
  - 校验是否在正确的层级（op_api入口应做基本校验，op_tiling做业务校验）
  - 校验是否完整覆盖文档描述的所有场景
  - 校验逻辑是否正确（如范围判断、条件分支等）
- **UT是否覆盖**：是/否/部分覆盖（需说明缺少哪些场景）
- **备注**：补充说明，如建议新增的校验、实现问题等

### 第五步：总结与建议

在表格之后，输出以下总结信息：

1. **覆盖率统计**：
   - 文档限制总条目数
   - 已实现条目数 / 未实现条目数 / 部分实现条目数
   - UT已覆盖条目数 / 未覆盖条目数 / 部分覆盖条目数

2. **关键问题清单**：
   - 文档有描述但代码未实现的校验（高优先级）
   - 代码有实现但逻辑不正确或不完整的校验（高优先级）
   - 代码有校验但UT未覆盖的场景（中优先级）
   - 实现位置不合适的校验（低优先级）

3. **改进建议**：
   - 需要新增的校验代码及建议实现位置
   - 需要新增的UT测试用例
   - 需要修正的校验逻辑

## 代码阅读指引

### 目录结构约定

```
<交付件路径>/
├── docs/                          # 算子文档
│   └── aclnn*.md                  # 算子接口文档
├── op_api/                        # aclnn入口层
│   ├── aclnn_*.h                  # 接口声明
│   └── aclnn_*.cpp                # 接口实现（含参数校验）
├── op_host/                       # Host侧
│   ├── op_tiling/                 # Tiling实现
│   │   ├── *_tiling_base.h/cpp    # Tiling基类
│   │   ├── arch32/                # 也有可能是arch22，910B/910C代际
│   │   │   └── *_tiling_910b.*    # 910B tiling实现
│   │   └── arch35/                # 950代际
│   │       └── *_tiling_base.*    # 950 tiling实现
│   └── config/                    # 配置文件
│       ├── ascend910b/            # 910B配置
│       └── ascend950/             # 950配置
├── op_kernel/                     # Kernel侧
│   ├── arch32/                    # 910B/910C kernel
│   └── arch35/                    # 950 kernel
├── op_graph/                      # 图模式
└── tests/
    ├── ut/                        # 单元测试
    │   ├── op_api/                # op_api层UT
    │   │   ├── test_aclnn_*.cpp   # 测试代码
    │   │   └── test_aclnn_*.csv   # 测试数据
    │   └── op_host/               # op_host层UT
    │       ├── test_*.cpp         # 测试代码
    │       ├── arch32/            # 910B/910C tiling UT
    │       └── arch35/            # 950 tiling UT
    └── st/                        # 系统测试
```

### 常见校验模式

| 校验类型 | op_api层常见模式 | op_tiling层常见模式 |
|---------|-----------------|-------------------|
| 空指针 | `if (ptr == nullptr)` / `CheckNotNull` | `OP_TILING_CHECK(ptr == nullptr, ...)` |
| dtype | `OP_CHECK_DTYPE_NOT_SUPPORT` / `OP_CHECK_DTYPE_NOT_SAME` | `CheckNonQuantTensorDataType` / `CheckTensorDataType` |
| shape | `CheckShapeAAMM` / `CheckShapeMMAA` | `CheckShapeInfo` / `CheckShapeDimRange` |
| format | `IsPrivateFormat` / `CheckFormat` | `CheckTensorFormat` |
| 空tensor | `CheckNotEmptyTensor` / `dim == 0` | `CheckInputShapesValid` |
| 非连续 | `IsTransposeLastTwoDims` / `TransX2Tensor` | 通常在op_api层处理 |
| 属性值 | `CheckAlltoAllAxes` / `CheckTransposeX1` / `CheckGroupLength` | `CheckAttrsInfo` |
| 范围 | `OP_API_CHECK` / 直接比较 | `OP_TILING_CHECK` / `CheckShapeDimRange` |
| 芯片分支 | `GetCurNpuArch() == NpuArch::DAV_3510` | `npuArch_ == NpuArch::DAV_3510` |

### 文档解析要点

1. **参数说明表格**：提取每个参数的dtype、format、shape、非连续tensor列
2. **约束说明**：逐条解析，注意区分不同芯片代际的差异（用`<term>`标签标注）
3. **使用说明列**：参数表格中的"使用说明"列常包含隐含约束

## 注意事项

1. 文档中的约束可能分散在多个位置（参数说明表格、约束说明章节、返回值表格），需全部收集
2. 不同芯片代际的约束可能不同，需分别标注和校验
3. op_api层和op_tiling层的校验可能有重叠，需判断是否冗余或互补
4. 某些约束可能只在文档中描述但代码未实现，需重点标注
5. UT测试可能只覆盖正常场景而缺少异常场景，需仔细区分
6. 共享工具函数（如matmul_allto_all_util.cpp）中的校验也需纳入分析
7. 检查代码时注意条件编译宏（如`__NPU_ARCH__ == 3510`）和运行时架构判断
