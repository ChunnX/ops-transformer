# mask_checker 对齐 TODO 清单

## 对比依据
- 文件: `attention/flash_attn/op_host/checkers/mask_checker.cpp`
- 文档: `torch_extension/npu_ops_transformer/doc/npu_flash_attn.md` 第574-640行 Mask参数组约束

## TODO 任务列表

### 1. mask_mode 单参数校验 - 缺失 dtype 校验
**文档要求**:
- data_type支持INT
- 支持输入范围仅为 1、3、4，默认值为 1

**当前实现**:
- ✓ 已实现：只支持 1/3/4 (CheckValueSupport)
- ✗ 缺失：未校验 mask_mode 的 dtype 是否为 INT

**待实现**: 添加 mask_mode dtype 校验，确保输入类型为 INT

---

### 2. attn_mask 单参数校验 - 已完成（数值约束无法在 Tiling 校验）
**文档要求**:
- tensor_type支持INT8
- tensor_shape维度为(2048, 2048)，该矩阵固定为2048*2048的下三角矩阵对角线全0

**当前实现**:
- ✓ 已实现：dtype 校验为 INT8 (第48-51行)
- ✓ 已实现：shape 校验为 (2048, 2048) (第53-61行)
- ⚠️ 数值约束（下三角矩阵对角线全0）属于 tensor 内容校验，Tiling 阶段无法获取数值，文档声明由用户自行保证

**结论**: 无需新增校验

---

### 3. win_left 单参数校验 - 缺失 dtype 校验
**文档要求**:
- data_type支持INT
- 不传或者传入-1,代表该值为正无穷
- 值需要 >= -1

**当前实现**:
- ✓ 已实现：值 >= -1 校验 (第65-70行)
- ✗ 缺失：未校验 win_left 的 dtype 是否为 INT

**待实现**: 添加 win_left dtype 校验，确保输入类型为 INT

---

### 4. win_right 单参数校验 - 缺失 dtype 校验
**文档要求**:
- data_type支持INT
- 不传或者传入-1,代表该值为正无穷
- 值需要 >= -1

**当前实现**:
- ✓ 已实现：值 >= -1 校验 (第72-77行)
- ✗ 缺失：未校验 win_right 的 dtype 是否为 INT

**待实现**: 添加 win_right dtype 校验，确保输入类型为 INT

---

### 5. mask_mode=1 一致性校验 - 逻辑需修正
**文档要求**:
- 当mask_mode为1时，win_left和win_right必须为-1以及不支持传入attn_mask

**当前实现**:
- ✓ 已实现：不支持传入 attn_mask (第101-103行)
- ⚠️ 部分实现：检查 win_left/win_right "不应提供"，但文档要求是"必须为-1"
- 实际逻辑：hasWinLeft 判断的是 `!= -1`，当传入-1时不报错，符合文档要求
- 但错误日志写的是 "should not be provided"，建议改为更准确的描述

**待修正**: 错误日志措辞改为 "win_left and win_right must be -1 when mask_mode=1 (all mask mode)"

---

### 6. mask_mode=3 一致性校验 - 逻辑需修正
**文档要求**:
- 当mask_mode为3时，win_left和win_right必须为-1且必须传入attn_mask矩阵

**当前实现**:
- ✓ 已实现：必须传入 attn_mask (第110-113行)
- ⚠️ 逻辑冗余：第118-129行额外检查 win_left/win_right 必须为 -1
- 但第114-117行先检查 "不应提供"，与第118-129行的"必须为-1"存在逻辑冲突

**待修正**: 
- 移除第114-117行的 "不应提供" 检查，保留第118-129行的"必须为-1"检查
- 或统一为：检查 win_left/win_right 必须为 -1（无论是否提供）

---

### 7. mask_mode=4 一致性校验 - 已完成
**文档要求**:
- 当mask_mode为4时，支持传入win_left和win_right且必须传入attn_mask矩阵

**当前实现**:
- ✓ 已实现：必须传入 attn_mask (第132-136行)
- ✓ 已实现：支持 win_left/win_right 传入且校验 win_left <= win_right (第79-88行)

**结论**: 无需新增校验

---

## 总结

需要新增/修正的项目：
1. ✗ mask_mode dtype 校验 (INT)
2. ✗ win_left dtype 校验 (INT)
3. ✗ win_right dtype 校验 (INT)
4. ⚠️ mask_mode=1 错误日志措辞修正
5. ⚠️ mask_mode=3 一致性校验逻辑优化

## 执行顺序
按 TODO 1-6 顺序逐条实现