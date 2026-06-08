# op_api Aclnn代码实现检查和修复

> 你应当在必要时候使用subagent来完成任务，提供一个独立、聚焦的上下文来完成具体的任务，并且为其提供必要的知识或信息。

参考已有代码对qkv_rms_norm_rope_cache_with_kscale算子的aclnn代码实现进行检查，包括编码风格、校验逻辑、日志信息写法、空指针等危险编码行为检查。

## 任务说明

参考gmm/grouped_matmul_finalize_routing算子伪量化MxA8W4数据流的aclnn代码实现风格对当前算子进行编码检查，grouped_matmul_finalize_routing算子的伪量化检查由专用checker实现，请参考该代码的写法对代码进行重构，核心代码文件为gmm/grouped_matmul_finalize_routing/op_api/grouped_matmul_finalize_routing_weight_quant_950_checker.h。

在重构代码时主要关注以下内容：
- 编码风格。
- 校验逻辑严谨有序，你可以使用同样的方式使用参数结构体来存储关键参数，然后按类别逐个检查的方式来实现checker，注意空指针的校验，避免出现危险编码操作。
- 日志风格要采用 `OP_LOGE_FOR_*` 风格的错误日志格式，该格式主要用于对外参数进行统一报错格式，而内部检查错误依然使用OP_LOGE。
- 检查代码中存在的低级编码错误或可能潜在的危险编码操作并修复。

## 任务计划

- [x] 创建checker头文件（参数结构体 + 分类校验方法）
- [x] 使用OP_LOGE_FOR_*宏替换OP_LOGE进行参数校验
- [x] 重构L2文件，移除内联校验函数
- [x] 编译验证 + 57/57测试通过

## 任务完成情况

### 变更文件

1. **新增**: `op_host/op_api/qkv_rms_norm_rope_cache_with_kscale_checker.h` (497行)
   - `QkvRmsNormRopeCacheWithKScaleParams` 参数结构体（17个参数）
   - `CheckNotNull()` — 使用`OP_LOGE_FOR_INVALID_VALUE_WITH_REASON`
   - `CheckDtypeValid()` — 使用`OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON`宏
   - `CheckShape()` — 使用`OP_LOGE_FOR_INVALID_SHAPEDIM/SHAPE/SHAPES_WITH_REASON`
   - `CheckFormat()` — 使用`OP_LOGE_FOR_INVALID_FORMAT`
   - `CheckEmptyTensor()` — 空tensor检查
   - `CheckParams()` — 入口函数，保留原始错误码

2. **修改**: `op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp` (407行→152行)
   - 移除7个内联校验函数和2个常量
   - 保留`Contiguous()`函数（L2特有逻辑）
   - 使用checker进行参数校验，保留原始错误码

### 关键设计决策

- 错误码保留：`CheckParams()`直接返回各子检查函数的原始错误码（NULLPTR/INVALID）
- dtype支持列表：使用`std::initializer_list<op::DataType>`定义，与参考checker一致
- 参数结构体：包含所有17个参数（输入/输出/属性/workspace/executor），便于传递和扩展

### 二次验证改进（RECHECK）

**发现问题**：
1. Checker使用namespace内static free functions → 对齐reference应使用class-based模式（更安全，避免ODR风险）
2. L2错误传播使用`if (ret != SUCCESS) return ret` → reference使用`CHECK_RET(status == ACLNN_SUCCESS, status)`更简洁

**改进内容**：
1. 重构checker为`QkvRmsNormRopeCacheWithKScaleChecker`类：
   - 参数存储为`params_`成员变量
   - dtype列表使用`inline static const`（C++17）
   - 常量使用`static constexpr`
   - 校验方法为成员函数（CheckNotNull/CheckDtypeValid/CheckShape/CheckFormat/CheckEmptyTensor/CheckParams）
   - `CheckType`辅助函数保留在namespace级别（供宏调用）
2. L2使用`CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet)`替代`if-return`模式
3. 编译通过 + 57/57测试通过