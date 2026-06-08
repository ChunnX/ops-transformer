# GOAL — 总体目标

> 总体目标导向文档，牵引核心方向不偏离，保证跨会话目标一致。

## 项目背景

`QkvRmsNormRopeCacheWithKScale` 是面向 Transformer decode/prefill 流程的融合算子，将 Q/K/V 拆分、RMSNorm、RoPE、rotation matmul、FP8 动态量化、KV cache 写回融合在一个算子中。首期交付限定为 **arch35/Ascend950** 平台。

- 算子原型规格：[raw/spec/qkv_rms_norm_rope_cache_with_kscale_op.md](raw/spec/qkv_rms_norm_rope_cache_with_kscale_op.md)
- 设计文档：[raw/design/qkv_rms_norm_rope_cache_with_kscale/](raw/design/qkv_rms_norm_rope_cache_with_kscale/)

## 总体目标

完成 `QkvRmsNormRopeCacheWithKScale` 算子的 **Aclnn 通路 (opapi)** 开发，包括：

1. **op_host 层** — 算子注册 (def.cpp)、shape 推导/校验 (infershape.cpp)
2. **op_api 层** — ACLNN L2 公共 API + L0 内部实现（模式 A: L2→L0 自包含）
3. **测试** — `tests/ut/op_api/` ACLNN API 单元测试
4. **构建框架** — CMakeLists.txt 确保可编译集成
5. **文档** — ACLNN 接口文档、README

**不包含**：op_kernel、tiling、op_graph、golden.py/inputs.py

## 目标拆解

### 阶段一：项目脚手架 + op_host 基础

搭建目录结构、CMake 框架，完成 op_host 的算子注册和 shape 推导/校验。

- [ ] 任务 1.1: 创建目录结构 + CMakeLists.txt 框架
- [ ] 任务 1.2: 实现 op_host/def.cpp — 算子注册
- [ ] 任务 1.3: 实现 op_host/infershape.cpp — shape 推导与校验

### 阶段二：op_api ACLNN 接口

完成 ACLNN L2 公共 API 和 L0 内部实现。

- [ ] 任务 2.1: 实现 op_api/aclnn_*.h/.cpp — L2 公共 API
- [ ] 任务 2.2: 实现 op_api/qkv_*.h/.cpp — L0 内部实现

### 阶段三：测试 + 文档 + 验收

- [ ] 任务 3.1: 实现 tests/ut/op_api/ — ACLNN API UT
- [ ] 任务 3.2: 实现 docs/ + README.md
- [ ] 任务 3.3: 编译验证 + UT 通过

## 约束与成功标准

1. **平台约束**: 仅支持 arch35/Ascend950
2. **dtype 约束**: 固定 dtype 组合（qkv=bf16, cache=fp8_e4m3fn, scale=fp32 等）
3. **fast path 约束**: D=128, Nv==Nk, rotation/v_scale 必选非空
4. **编译通过**: 所有源文件无编译错误
5. **UT 通过**: op_api 测试全部 pass

## 关键决策记录

| 决策项 | 选择 | 原因 |
|--------|------|------|
| op_api 模式 | 模式 A (L2→L0 自包含) | 参考 nsa_compress_with_cache |
| op_api 目录 | `op_host/op_api/`（嵌套在 op_host 内） | 参考 nsa_compress_with_cache 风格 |
| op_graph | 不交付 | 首期只交付 ACLNN 直调路径 |
| epsilon 类型 | float | 用户确认 |
| tiling | 不交付（后续阶段） | 与 kernel 紧密相关 |
| 测试范围 | 仅 op_api UT | 无 kernel 不需要 golden |

## 相关链接

- [实现计划](./PLAN.md)
- [用户下一步计划](./NEXT_STEPS.md)
- [检查清单](./CHECKLIST.md)
- [变更记录](./CHANGELIST.md)
- [算子原型规格](./raw/spec/qkv_rms_norm_rope_cache_with_kscale_op.md)
- [设计文档](./raw/design/qkv_rms_norm_rope_cache_with_kscale/)
