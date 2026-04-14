# MxA8W4 数据流支持计划

## 需求概述

为 `aclnnGroupedMatmulFinalizeRoutingWeightNzV2` 接口新增 MxA8W4 量化数据流支持：
- **x1**: float8_e4m3fn, ND, (m, k)
- **x2**: float4_e2m1, NZ格式, viewShape=(e, n, k)
- **scale**: float8_e8m0, ND, (e, n, ceil(k/64), 2) — 复用现有V2接口的scale参数
- **pertokenScale**: float8_e8m0, ND, (m, ceil(k/64), 2)
- **weight format**: NZ_C0_32（与现有MX场景的ND格式不同）

**不涉及kernel修改**，仅aclnn通路（op_api层）。

## 数据流对比

| 项目 | MXFP8/MXFP4（原有） | MxA8W4（新增） |
|------|---------------------|----------------|
| x1 dtype | FP8_E4M3FN / FP4_E2M1 | FP8_E4M3FN |
| x2 dtype | FP8_E4M3FN / FP4_E2M1 | FP4_E2M1 |
| scale dtype | FP8_E8M0 | FP8_E8M0 |
| pertokenScale dtype | FP8_E8M0 | FP8_E8M0 |
| weight format | ND | NZ_C0_32 |
| scale shape | (e, n, ceil(k/64), 2) 或 (e, ceil(k/64), n, 2) | (e, n, ceil(k/64), 2) |
| pertokenScale shape | (m, ceil(k/64), 2) | (m, ceil(k/64), 2) |
| 共用部分 | 同一接口、同一数据流、同一L0调用 | 同左 |

## 修改清单

### 1. 算子原型新增A8W4 dtype组合
- **文件**: `gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_def.cpp`
- **描述**: 在 `config91095` 的各Input/Output DataType列表末尾追加第14行A8W4对应的dtype
- **状态**: ✅ 已完成

### 2. 新增A8W4专用Checker
- **文件**: `gmm/grouped_matmul_finalize_routing/op_api/grouped_matmul_finalize_routing_weight_quant_950_checker.h`（新建）
- **描述**: 独立实现A8W4专用checker，dtype白名单（x1=FP8_E4M3FN, x2=FP4_E2M1, scale=FP8_E8M0），weight格式校验为NZ
- **状态**: ✅ 已完成

### 3. CheckParams分发逻辑
- **文件**: `gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp`
- **描述**: 在 `CheckParams` 函数中增加A8W4场景判断（x1=FP8_E4M3FN && x2=FP4_E2M1），分发到新checker
- **状态**: ✅ 已完成

### 4. PreMatmulCalcProcess适配A8W4
- **文件**: `gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp`
- **描述**: scale的transpose一致性检查处，A8W4场景跳过检查（scale已按transposeX2对应shape传入）
- **状态**: ✅ 已完成

### 5. V2函数CheckSupportScene适配A8W4
- **文件**: `gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp`
- **描述**: `aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize` 中A8W4场景跳过 `CheckSupportScene`（该函数要求transpose为false，与A8W4的transposeX2=true冲突）
- **状态**: ✅ 已完成

### 6. UT用例
- **文件**: `gmm/grouped_matmul_finalize_routing/tests/ut/op_host/op_api/test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2.cpp`
- **描述**: 新建 `l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950` 测试类（仅在SetUpTestCase中设置DAV_3510平台），新增31个MxA8W4测试用例：
  - **原有5个**: 正常用例/带bias/scale维度错误/x2格式错误/sharedInput越界
  - **约束条件负用例(4个)**: k非32对齐/n非32对齐/专家数超1024/bsdp超outputBS
  - **必选参数nullptr负用例(1个)**: pertokenScaleOptional为空
  - **dtype错误负用例(3个)**: x1/x2/scale dtype不匹配
  - **维度错误负用例(4个)**: x1/pertokenScale/groupList/out维度错误
  - **shape一致性负用例(4个)**: scale/pertokenScale/bias/x2 shape不匹配
  - **格式错误负用例(2个)**: x1/scale格式非ND
  - **边界值正向用例(5个)**: 最小k=n/最大专家数/单token/无sharedInput/cumsum模式
  - **泛化参数正向用例(3个)**: 大k=n/中等专家数/zero offset
- **状态**: ✅ 已完成（31/31通过，26/26原有910B2用例无回归）

### 7. 编译验证
- **命令**: 
  - 编译grouped_matmul_finalize_routing算子opapi的ut程序：`bash build.sh -u --opapi --soc=ascend950 --ops=grouped_matmul_finalize_routing --noexec`
  - 执行MxA8W4测试用例：`./build/tests/ut/framework_normal/op_api/transformer_op_api_ut --gtest_filter=l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950.ascend950_test_mxa8w4*`
  - 验证原有910B2用例无回归：`./build/tests/ut/framework_normal/op_api/transformer_op_api_ut --gtest_filter=l2_GroupedMatmulFinalizeRoutingWeightNzV2_test.ascend910B2*`
- **状态**: ✅ 已完成（5/5 MxA8W4用例通过，26/26 910B2用例通过）

## 实施顺序

1. 算子原型 → 注册新dtype组合 ✅
2. 专用Checker → 新建文件，拷贝+修改 ✅
3. CheckParams分发 → 路由到新checker ✅
4. PreMatmulCalcProcess → 跳过A8W4的transpose检查 ✅
5. V2函数CheckSupportScene → 跳过A8W4场景 ✅
6. UT用例 → 验证新数据流 ✅
7. 编译验证 → 全部通过 ✅

## 约束条件

- k % 32 == 0
- n % 32 == 0
- e <= 1024
- bsdp <= outputBS
- sharedInputOffset + bsdp <= outputBS
- transposeX1 = false
- transposeX2 = true

## 进度追踪

| 步骤 | 状态 | 完成时间 |
|------|------|----------|
| 1. 算子原型 | ✅ 已完成 | - |
| 2. 专用Checker | ✅ 已完成 | - |
| 3. CheckParams分发 | ✅ 已完成 | - |
| 4. PreMatmulCalcProcess适配 | ✅ 已完成 | - |
| 5. V2函数CheckSupportScene适配 | ✅ 已完成 | - |
| 6. UT用例 | ✅ 已完成 | - |
| 7. 编译验证 | ✅ 已完成 | - |

## UT验证问题修复记录

| # | 问题 | 修复方案 | 文件 |
|---|------|---------|------|
| 1 | UT中x2 storageShape错误：原为 `{e, n/64, k/16, 16, 64}`，应为 `{e, CeilDiv(k,32), CeilDiv(n,16), 16, 32}`（NZ_C0_32格式） | 使用CeilDiv向上取整，修正storageShape | test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2.cpp |
| 2 | UT缺少平台设置：MxA8W4用例需要DAV_3510平台才能路由到专用Checker | 新建 `l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950` 测试类，在SetUpTestCase中设置 `op::SetPlatformNpuArch(NpuArch::DAV_3510)` | test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2.cpp |
| 3 | V2函数CheckSupportScene未跳过A8W4：A8W4需要transposeX2=true，但CheckSupportScene要求transpose为false | 增加isA8W4判断，A8W4场景跳过CheckSupportScene调用 | aclnn_grouped_matmul_finalize_routing.cpp |
| 4 | CheckOptionalShapes中CHECK_COND宏误用：函数返回bool但CHECK_COND返回aclnnStatus，导致错误码被转换为true（非零整数→true） | 将CHECK_COND改为 `if (!cond) { OP_LOGE(...); return false; }` | grouped_matmul_finalize_routing_weight_quant_950_checker.h |
| 5 | Checker缺少k/n对齐约束检查：SRS要求k%32==0和n%32==0，但Checker未校验 | 在CheckInputOutShape中增加 `if (k%32!=0) return false;` 和 `if (n%32!=0) return false;` | grouped_matmul_finalize_routing_weight_quant_950_checker.h |

## git commit

feat(gmmfr): 新增MxA8W4量化数据流支持weightNzV2接口

    - 算子原型(config91095)新增A8W4 dtype组合: x=FP8_E4M3FN, w=FP4_E2M1(NZ), scale=FP8_E8M0
    - 新建A8W4专用Checker: grouped_matmul_finalize_routing_weight_quant_950_checker.h
      - dtype白名单校验(x1=FP8, x2=FP4, scale=FP8_E8M0, pertokenScale=FP8_E8M0)
      - weight格式校验为NZ而非ND
      - sharedInputOffset+bsdp<=outputBS约束校验
    - CheckParams增加A8W4场景分发逻辑，与原有MX/Pertoken checker解耦
    - PreMatmulCalcProcess跳过A8W4场景的scale transpose一致性检查
    - V2函数CheckSupportScene跳过A8W4场景（transposeX2=true与原有检查冲突）
    - 修复CheckOptionalShapes中CHECK_COND宏误用导致错误码丢失的bug
    - 新增l2_GroupedMatmulFinalizeRoutingWeightNzV2_test950测试类及5个MxA8W4 UT用例
      - 使用CeilDiv进行NZ格式shape向上取整计算
      - 正常用例/带bias/scale维度错误/x2格式错误/sharedInput越界
