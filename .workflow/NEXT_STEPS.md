# NEXT_STEPS — 下一步计划

> 用户任务的**索引**。记录任务概要和状态，链接到详细任务说明。

## 任务 1：项目脚手架 + op_host 基础

**任务状态**：✅ DONE

**简要说明**：创建目录结构、CMakeLists.txt 框架，实现 def.cpp（算子注册）和 infershape.cpp（shape 推导/校验）

**任务详情**：[详细任务说明](../PLAN.md#阶段一项目脚手架--op_host-基础)

**完成状态**:
- [x] 目录结构 + CMakeLists.txt
- [x] def.cpp — 算子注册
- [x] infershape.cpp — shape 推导校验

## 任务 2：op_api ACLNN 接口

**任务状态**：✅ DONE

**简要说明**：实现 ACLNN L2 公共 API（参数校验、executor 创建）和 L0 内部实现（OP_TYPE_REGISTER、INFER_SHAPE、ADD_TO_LAUNCHER_LIST）

**任务详情**：[详细任务说明](../PLAN.md#阶段二op_api-aclnn-接口)

**完成状态**:
- [x] L2 公共 API (aclnn_*.h/.cpp)
- [x] L0 内部实现 (qkv_*.h/.cpp)

## 任务 3：测试 + 文档 + 验收

**任务状态**：✅ DONE

**简要说明**：实现 op_api UT、ACLNN 接口文档、README，完成编译验证和 UT 通过

**任务详情**：[详细任务说明](../PLAN.md#阶段三测试--文档--验收)

**完成状态**:
- [x] op_api UT (4/4 测试通过)
- [x] ACLNN 接口文档 + README
- [x] 编译验证 + UT 通过

## 任务 4：op_api 接口修正

**任务状态**：✅ DONE

**简要说明**：按用户最新要求将目录/小写算子名统一为 `qkv_rms_norm_rope_cache_with_kscale`，将 `rotation` 拆分为 `q_rotation`/`k_rotation`，并将 `k_cache`/`v_cache`/`k_scale_cache` 调整为 ACLNN inplace 输入输出参数。

**任务详情**：[详细任务说明](../PLAN.md#阶段四op_api-接口修正)

**完成状态**:
- [x] `attention/qkv_rms_norm_rope_cache_with_k_scale/` 更名为 `attention/qkv_rms_norm_rope_cache_with_kscale/`
- [x] `rotation` 拆分为 `q_rotation` 和 `k_rotation`，shape 均为 `[D, D]`
- [x] cache 三参数按 `Ref`/inplace 模式接入 op_host、L0、L2
- [x] op_host + op_api 编译通过
- [x] op_api UT 编译通过

## 任务 5：op_api Aclnn接口参数检查

**任务状态**：✅ DONE（三次验证 + L0 tuple返回修正）

**简要说明**：参考已有代码对qkv_rms_norm_rope_cache_with_kscale算子的aclnn接口和实现进行检查和修复

**任务详情**：[详细任务说明](reference/next_steps/task5-aclnn_api_check.md)

**完成状态**:
- [x] inplace参数(kCache/vCache/kScaleCache)处理检查 — def.cpp同名Input/Output注册、L0同参OP_INPUT+OP_OUTPUT、L2非const aclTensor*且不调用Contiguous()，与quant_grouped_matmul_inplace_add参考一致
- [x] 多输出参数处理检查 — L0改为tuple返回所有5个输出（qOut, qScale, kCache, vCache, kScaleCache），与scatter_pa_kv_cache/gather_pa_kv_cache模式一致
- [x] L0返回模式修正：从单指针`const aclTensor*`改为`std::tuple<const aclTensor*,...>(5个)`，L2逐一检查所有返回值
- [x] 编译通过 + 57/57测试通过

## 任务 6：Aclnn代码实现检查和修复

**任务状态**：✅ DONE（二次验证 + 改进）

**简要说明**：参考已有代码对qkv_rms_norm_rope_cache_with_kscale算子的aclnn代码实现进行检查，包括编码风格、校验逻辑、日志信息写法、空指针等危险编码行为检查。

**任务详情**：[详细任务说明](reference/next_steps/task6-aclnn_impl_refactor.md)

**完成状态**:
- [x] 创建checker头文件，使用参数结构体+OP_LOGE_FOR_*宏重构校验逻辑
- [x] 重构L2文件，移除内联校验函数，使用checker
- [x] 空指针安全校验使用OP_LOGE_FOR_INVALID_VALUE_WITH_REASON格式
- [x] dtype校验使用OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON宏
- [x] 二次验证：checker重构为class-based模式（对齐grouped_matmul_finalize_routing reference），L2使用CHECK_RET宏
- [x] 编译通过 + 57/57测试通过

## 任务 7：Aclnn接口参数k_cache, v_cache, k_scale_cache的非连续性问题

**任务状态**：✅ DONE（二次验证通过）

**简要说明**：当前qkv_rms_norm_rope_cache_with_kscale算子的aclnn接口描述中对参数k_cache, v_cache, k_scale_cache的连续性没有清晰定义，需要根据任务说明进行详细检查。

**任务详情**：[详细任务说明](reference/next_steps/task7-aclnn_params_continguous.md)

**完成状态**:
- [x] 调研scatter_pa_kv_cache/gather_pa_kv_cache/quant_grouped_matmul_inplace_add的非连续tensor处理模式
- [x] 确认三个参考算子def.cpp均不使用AutoContiguous() — 当前实现一致
- [x] 移除def.cpp中k_cache/v_cache/k_scale_cache的AutoContiguous()，保留原始stride
- [x] L2中不对cache参数调用Contiguous()，直接传给L0，kernel通过原始stride写入正确位置
- [x] 二次验证确认实现符合规范
- [x] 编译通过 + 57/57测试通过

