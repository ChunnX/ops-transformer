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

## 阶段四：op_api 接口修正

### 任务 4.1: 统一小写算子名
**目标**: 将源码目录和小写构建入口统一为 `attention/qkv_rms_norm_rope_cache_with_kscale/`
**完成标准**:
- 源码目录不再使用 `qkv_rms_norm_rope_cache_with_k_scale`
- `--ops=qkv_rms_norm_rope_cache_with_kscale` 可完成 op_host/op_api 构建

### 任务 4.2: 拆分 rotation 输入
**目标**: 将单个 `rotation` 输入拆分为 `q_rotation` 和 `k_rotation`
**完成标准**:
- op_host def 中包含 `q_rotation`/`k_rotation`
- infershape 和 L2 shape check 校验二者 shape 均为 `[D, D]`
- L0/L2 ACLNN 签名和文档示例一致

### 任务 4.3: cache 参数 inplace 化
**目标**: `k_cache`、`v_cache`、`k_scale_cache` 作为输入输出复用参数原地更新
**完成标准**:
- def 输出名使用 `k_cache`、`v_cache`、`k_scale_cache`
- L0 `OP_INPUT`/`OP_OUTPUT` 使用同一 tensor，实现 `Ref` 关系
- L2 不再暴露独立 cache out 参数，也不对 cache 输入做 Contiguous 替换
- autogen inner 代码包含 `NnopbaseSetRef` 映射

## 阶段五：ACLNN 接口检查与代码重构

### 任务 5.1: ACLNN 接口参数检查
**目标**: 检查 inplace 参数和多输出处理是否符合规范
**完成标准**:
- inplace 参数（kCache/vCache/kScaleCache）L2 签名、L0 OP_INPUT/OP_OUTPUT、def 注册均正确
- 多输出处理与 scatter_pa_kv_cache/gather_pa_kv_cache 参考一致（tuple 返回所有输出）
**结果**: ✅ 三次验证完成，L0 改为 tuple 返回所有 5 个输出

### 任务 5.2: ACLNN 代码实现检查和修复
**目标**: 参考 grouped_matmul_finalize_routing checker 模式重构校验逻辑
**目标文件**:
- `op_host/op_api/qkv_rms_norm_rope_cache_with_kscale_checker.h`（新增→二次重构为class-based）
- `op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_kscale.cpp`（重构→二次使用CHECK_RET）
**完成标准**:
- 使用参数结构体 + OP_LOGE_FOR_* 宏 + class-based checker
- 编译通过 + 57/57 测试通过
**结果**: ✅ 二次验证改进完成

### 任务 5.3: cache 参数非连续性问题
**目标**: 确保 k_cache/v_cache/k_scale_cache 的原始 stride 不被破坏
**目标文件**: `op_host/qkv_rms_norm_rope_cache_with_k_scale_def.cpp`
**完成标准**:
- 移除 cache 参数的 AutoContiguous()
- L2 不对 cache 参数调用 Contiguous()
- 编译通过 + 57/57 测试通过
**结果**: ✅ 二次验证通过，与scatter/gather/quant_grouped_matmul_inplace_add参考一致

## 当前进度

- [x] 阶段一：项目脚手架 + op_host 基础 ← **已完成**
- [x] 阶段二：op_api ACLNN 接口 ← **已完成**
- [x] 阶段三：测试 + 文档 + 验收 ← **已完成**
- [x] 阶段四：op_api 接口修正 ← **已完成**
- [x] 阶段五：ACLNN 接口检查与代码重构 ← **已完成**

## 技术决策

| 决策项 | 选择 | 参考 |
|--------|------|------|
| op_api 模式 | 模式 A (L2→L0 自包含) | nsa_compress_with_cache |
| op_api 目录 | op_host/op_api/（嵌套） | nsa_compress_with_cache |
| op_graph | 不交付 | 首期只 ACLNN 直调 |
| epsilon 类型 | float | 用户确认 |
| tiling | 不交付（后续阶段） | 与 kernel 紧密相关 |
| 测试范围 | 仅 op_api UT | 无 kernel 不需要 golden |
| 生成器命名 | def 文件名匹配 CANN autogen | CANN 将 `KScale` 生成成 `k_scale`，def 文件命名为 `qkv_rms_norm_rope_cache_with_k_scale_def.cpp` 直接匹配，无需 alias |
| cache inplace 处理 | cache 输入直接作为输出传给 L0 | 参考 quant_grouped_matmul_inplace_add 的 Ref 参数模式 |

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| op_api L0 与 op_host 注册路径不匹配 | 编译失败 | 参考 nsa_compress_with_cache 完整链路 |
| FP8 dtype 在 UT 框架中不支持 | 测试失败 | 参考已有算子 FP8 测试方法 |
| in-place alias 校验逻辑 | 运行时错误 | 参考 scatter_pa_kv_cache alias 处理 |
