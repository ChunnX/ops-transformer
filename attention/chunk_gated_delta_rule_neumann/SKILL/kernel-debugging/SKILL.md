# Kernel Debugging & Environment Setup

## ACLNN 算子测试环境配置（关键）

### 问题：GetWorkspaceSize 返回 561103 错误

**错误码含义**：
```
ACLNN_ERR_INNER_NULLPTR (561103) = 内部指针为空
```

**根本原因**：未设置 `ASCEND_CUSTOM_OPP_PATH` 环境变量，导致算子库无法正确加载。

### 正确的环境配置

**原版算子**（chunk_gated_delta_rule）：
```bash
source /home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/vendors/custom_transformer/bin/set_env.bash
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH}
```

**Neumann 版算子**（chunk_gated_delta_rule_neumann）：
```bash
export ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

### 两个安装路径说明

| 算子版本 | 安装路径 | 库文件 | 头文件 |
|---------|---------|--------|--------|
| 原版 | `vendors/custom_transformer/vendors/custom_transformer/` | libcust_opapi.so (133KB) | aclnn_chunk_gated_delta_rule.h |
| Neumann | `vendors/custom_transformer/` | libcust_opapi.so (67KB) | aclnn_chunk_gated_delta_rule_neumann.h |

**关键**：两个路径互不干扰，但必须设置对应的 `ASCEND_CUSTOM_OPP_PATH`。

---

## TASK_RATIO 架构分析

### TASK_RATIO 对数据流的影响

**定义**：
```cpp
constexpr uint64_t TASK_RATIO = 2;  // 每个核心分为 2 个 subBlock
constexpr uint64_t AIC_AIV_1_1 = 2;  // AIC:AIV = 1:1
```

**数据切分**：
```cpp
halfChunkSize_ = chunkSize_ / TASK_RATIO;
subOffset_ = subBlockIdx_ * halfChunkSize_;

// chunkSize_=32 时：
//   subBlockIdx=0: 处理前 16 行，offset=0
//   subBlockIdx=1: 处理后 16 行，offset=16
```

### Kernel 挂起根因

**问题**：InverseCompute 输出格式不匹配

**原版正确格式**：
```cpp
// InverseCompute 输出 halfChunkSize_×chunkSize_ 子矩阵
DataCopyOutFp32(halfChunkSize_, chunkSize_, chunkSize_, attnWsGm_[subOffset_ * chunkSize_]);

// 对角线设置：只设置 halfChunkSize_ 个元素
for (uint32_t i = 0; i < halfChunkSize_; ++i) {
    inverseLocal_.SetValue((subOffset_ + i) * chunkSize_ + subOffset_ + i, 1.0f);
}
```

**错误修改**：
```cpp
// 错误：输出完整 chunkSize_×chunkSize_ 矩阵（两个 subBlock 覆盖同一位置）
DataCopyOutFp32(chunkSize_, chunkSize_, chunkSize_, attnWsGm_[0]);

// 错误：设置所有 chunkSize_ 个对角元素（subBlock 间冲突）
for (uint32_t i = 0; i < chunkSize_; ++i) {
    inverseLocal_.SetValue(i * chunkSize_ + i, 1.0f);
}
```

---

## GetSubBlockIdx() 检查规则

**TASK_RATIO=2 时**：
- `GetSubBlockIdx()` 返回 0 或 1
- Stage3 Init: `GetSubBlockIdx() == 1` 时直接 return（跳过 InitBuffer）
- Stage3 Process: `GetSubBlockIdx() == 0` 执行 CalMaskedQKT

**TASK_RATIO=1 时**（错误）：
- `GetSubBlockIdx()` 只返回 0
- `GetSubBlockIdx() == 1` 检查永远不触发 → InitBuffer 被跳过 → Kernel 挂起

**规则**：**不要修改 TASK_RATIO**，保持原版架构的 2-subBlock 设计。

---

## Chunk Size 修改影响链

**修改 chunkSize 从 64→16 的影响**：

| 参数 | 原值 | 新值 | 影响 |
|-----|------|------|------|
| chunkSize_ | 64 | 16 | 主循环长度 |
| halfChunkSize_ | 32 | 8 | 每个 subBlock 处理 8 行 |
| INVERSE_SHAPE | 32 | 16 | 求逆矩阵边长 |
| workspaceSize | × | ×3 | L_pow + tmp + INV |

**关键约束**：
- dk, dv ≤ 128（tiling 限制）
- nv 必须是 nk 的倍数（GQA 支持）
- chunkSize 必须能被 TASK_RATIO 整除

---

## 调试流程总结

1. **环境检查**：确认 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH` 设置
2. **库加载验证**：`ldd test_binary | grep cust_opapi` 检查是否找到库
3. **错误码查询**：`grep -r "561103" $ASCEND_HOME_PATH/include/` 找含义
4. **参数约束检查**：tiling.cpp 中的 `OP_CHECK_IF` 条件
5. **架构对比**：对比原版与修改版的 TASK_RATIO、halfChunkSize_、GetSubBlockIdx() 使用

---

## 常用调试命令

```bash
# 检查库是否加载
ldd test_binary | grep cust_opapi

# 检查安装路径
ls -la $ASCEND_HOME_PATH/opp/vendors/custom_transformer/*/op_api/lib/

# 检查头文件
ls -la $ASCEND_HOME_PATH/opp/vendors/custom_transformer/*/op_api/include/aclnnop/

# 查找错误码定义
grep -r "561103" $ASCEND_HOME_PATH/include/

# 查找参数约束
grep "OP_CHECK_IF" op_host/*_tiling.cpp

# 对比原版与修改版差异
diff -u attention/chunk_gated_delta_rule/op_kernel/*.h \
        attention/chunk_gated_delta_rule_neumann/op_kernel/*.h
```

---

## 相关文档

- [aclnn-single-op-test](../aclnn-single-op-test) - ACLNN 单算子测试流程
- [precision-debugging](../precision-debugging) - 精度问题调试经验
- [operator-clone](../operator-clone) - 算子克隆与重命名流程
- [docs/CHUNK16_NEUMANN_MIGRATION_PLAN.md](../../docs/CHUNK16_NEUMANN_MIGRATION_PLAN.md) - CHUNK=16 迁移计划