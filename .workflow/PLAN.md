# PLAN — 实现计划

> 将 GOAL.md 的总体目标拆分为可逐步执行的阶段计划。

## 执行顺序

**第一轮**: 阶段一（脚手架 + op_host）+ 阶段二（op_api）
**第二轮**: 阶段三（测试 + 文档 + 验收）

## 阶段一：项目脚手架 + op_host 基础

### 任务 1.1: 创建目录结构 + CMakeLists.txt 框架
**目标**: 建立 `attention/qkv_rms_norm_rope_cache_with_kscale/` 完整目录树，编写各级 CMakeLists.txt
**目标文件**:
- `attention/qkv_rms_norm_rope_cache_with_kscale/CMakeLists.txt`
- `attention/qkv_rms_norm_rope_cache_with_kscale/op_host/CMakeLists.txt`
- `attention/qkv_rms_norm_rope_cache_with_kscale/op_api/CMakeLists.txt`
- `attention/qkv_rms_norm_rope_cache_with_kscale/tests/CMakeLists.txt`
**完成标准**: 目录结构完整，CMake 可解析，不破坏父目录编译
**参考**: `attention/mla_prolog_v3/CMakeLists.txt`, `attention/nsa_compress_with_cache/CMakeLists.txt`

### 任务 1.2: 实现 op_host/def.cpp
**目标**: 注册 `QkvRmsNormRopeCacheWithKScale` op type
**目标文件**: `op_host/qkv_rms_norm_rope_cache_with_kscale_def.cpp`
**完成标准**:
- 9 个必选输入 + 2 个可选输入 + 5 个输出 + 2 个属性按规格文档定义
- ascend950 平台配置正确
- k_cache/v_cache/k_scale_cache 标记 in-place/alias
- head_nums 为 REQUIRED ListInt
- epsilon 为 OPTIONAL float (默认 1e-6)
**参考**: `attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp`

### 任务 1.3: 实现 op_host/infershape.cpp
**目标**: 从 qkv 推导输出 shape，校验所有输入 shape/dtype 一致性
**目标文件**: `op_host/qkv_rms_norm_rope_cache_with_kscale_infershape.cpp`
**完成标准**:
- q_out: [T, Nq, D], q_scale: [T, Nq]
- cache 输出 shape = 对应输入 shape
- 完整 shape/dtype 校验（见规格文档约束条件表）
**参考**: `gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_infershape.cpp`

## 阶段二：op_api ACLNN 接口

### 任务 2.1: 实现 L2 公共 API
**目标**: 暴露 aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize 和 aclnnQkvRmsNormRopeCacheWithKScale
**目标文件**:
- `op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.h`
- `op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`
**完成标准**:
- 参数校验（空指针、dtype、shape）
- optionalRotation/optionalVScale 非空校验
- headNums 非空、长度 3、值正整数校验
- executor 创建和传递
- in-place 输出 alias 处理
**参考**: `attention/nsa_compress_with_cache/op_host/op_api/aclnn_nsa_compress_with_cache.cpp`

### 任务 2.2: 实现 L0 内部实现
**目标**: 完成 OP_TYPE_REGISTER、INFER_SHAPE、ADD_TO_LAUNCHER_LIST_AICORE
**目标文件**:
- `op_api/qkv_rms_norm_rope_cache_with_kscale.h`
- `op_api/qkv_rms_norm_rope_cache_with_kscale.cpp`
**完成标准**:
- 注册 op type
- 可选输入 nullptr 时创建 placeholder tensor
- aclIntArray headNums 转换为 aclTensor
- INFER_SHAPE 调用成功
- ADD_TO_LAUNCHER_LIST_AICORE 调用成功
**参考**: `attention/nsa_compress_with_cache/op_host/op_api/nsa_compress_with_cache.cpp`

## 阶段三：测试 + 文档 + 验收

### 任务 3.1: 实现 op_api UT
**目标**: ACLNN API 单元测试
**目标文件**: `tests/ut/op_api/test_aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`
**完成标准**:
- 正常调用成功
- 参数错误返回正确错误码
- dtype 错误返回正确错误码
- 空指针返回正确错误码
- 可选输入缺失返回正确错误码
**参考**: `attention/nsa_compress_attention/tests/ut/op_api/test_aclnn_nsa_compress_attention.cpp`

### 任务 3.2: 实现文档
**目标文件**:
- `docs/aclnnQkvRmsNormRopeCacheWithKScale.md`
- `README.md`

### 任务 3.3: 编译验证 + UT 通过
**完成标准**: 所有源文件编译通过，UT 全部 pass

## 当前进度

- [x] 阶段一：项目脚手架 + op_host 基础 ← **已完成**
- [x] 阶段二：op_api ACLNN 接口 ← **已完成**
- [x] 阶段三：测试 + 文档 + 验收 ← **已完成**

## 技术决策

| 决策项 | 选择 | 参考 |
|--------|------|------|
| op_api 模式 | 模式 A (L2→L0 自包含) | nsa_compress_with_cache |
| op_api 目录 | 根目录 op_api/ | mla_prolog_v3 |
| op_graph | 不交付 | 首期只 ACLNN 直调 |
| epsilon 类型 | float | 用户确认 |
| tiling | 不交付（后续阶段） | 与 kernel 紧密相关 |
| 测试范围 | 仅 op_api UT | 无 kernel 不需要 golden |

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| op_api L0 与 op_host 注册路径不匹配 | 编译失败 | 参考 nsa_compress_with_cache 完整链路 |
| FP8 dtype 在 UT 框架中不支持 | 测试失败 | 参考已有算子 FP8 测试方法 |
| in-place alias 校验逻辑 | 运行时错误 | 参考 scatter_pa_kv_cache alias 处理 |
