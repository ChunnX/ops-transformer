# GMM/GMMFR 算子错误码规范化整改 SRS

## 1. 迭代信息

| 项目 | 内容 |
| --- | --- |
| AR编号 | AR20260328158974 |
| 责任人 | 琦钧 |
| 预计工作量 | 0.7K |
| 时间节点 | SRS 5.9 → CODING 5.18 → UT 5.22 → ST 5.29 → 合入 6.1 |

---

## 2. 整改目标与范围

**目标：** 将主机层 ERROR 日志规范化为 EZ0008-EZ0024 错误码

**范围：** A5责任田 host 层（op_api + op_host）

**数据统计：**

| 错误类型 | GMMFR | GMM | 合计 |
| --- | --- | --- | --- |
| OP_LOGE | 231 | 653 | **884** |
| CHECK_COND | 0 | 397 | **397** |
| CHECK_RET | 18 | 7 | **25** |
| OPS_REPORT | 50 | 43 | **93** |
| **总计** | **299** | **1100** | **1399** |

---

## 3. 整改方案

### 3.1 直接整改类

**原则：替换为 OP_LOGE_FOR_* 宏**

**示例：dtype 错误**

```cpp
// 原代码
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 and x2 dtype should be same, actual x1 dtype is %s and x2 dtype is %s.", 
        x1_dtype, x2_dtype);

// 整改后
OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
    ACLNN_ERR_PARAM_INVALID,
    "x1, x2",
    "actual_x1_dtype, actual_x2_dtype",
    "x1 and x2 dtype should be same"
);
```

### 3.2 上报推荐新增类

**原则：上报无法匹配 EZ0008-EZ0024，推荐新增错误码**

**示例：CHECK_RET 内部错误**

```cpp
// 原代码
CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

// 整改：上报并推荐新增 EZ_INNER_NULLPTR
```

### 3.3 评审后决定类

**原则：需评审决定是否新增错误码或归类到现有类型**

**示例：空指针错误**

```cpp
// 原代码
CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "X must not be nullptr.");

// 评审决策点：
// - 方案A：新增 EZ_PARAM_NULLPTR
// - 方案B：使用 EZ0023 覆盖
```

---

## 4. 整改流程

| Phase | 任务 | 工具 |
| --- | --- | --- |
| Phase 0 | 生成整改表格 | generate_refactor_table.py |
| Phase 1 | 审阅表格 | 手动审阅 |
| Phase 2 | 代码替换 | 手动替换 |
| Phase 3 | 编译验证 | build.sh |
| Phase 4 | 生成报告 | unmatched_errors_report.md |

---

## 5. EZ错误码速查

| 错误场景 | 错误码 | 宏名称 |
| --- | --- | --- |
| Shape 错误 | EZ0008 | OP_LOGE_FOR_INVALID_SHAPE |
| Dtype 错误 | EZ0018 | OP_LOGE_FOR_INVALID_DTYPE |
| Format 错误 | EZ0016 | OP_LOGE_FOR_INVALID_FORMAT |
| Value 错误 | EZ0023 | OP_LOGE_FOR_INVALID_VALUE_WITH_REASON |

---

## 6. 待评审事项

1. **CHECK_COND(ACLNN_ERR_PARAM_NULLPTR)**：是否新增 EZ_PARAM_NULLPTR？
2. **CHECK_RET(ACLNN_ERR_INNER)**：是否新增 EZ_INNER_NULLPTR？
3. **OPS_REPORT**：需逐条分析，分类为参数验证类或内部错误类

---

## 7. 参考文档

- `ops-transformer/.opencode/skills/ascendc-error-code-refactor/SKILL.md`
- `ops-transformer/CANN_ERROR_LOG_COLLECTION_GUIDE.md`