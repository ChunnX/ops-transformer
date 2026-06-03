# Neumann 算子进度与问题记录

**最后更新**: 2026-04-28 17:30

---

## 当前状态

### ✅ 已完成

| 任务 | 状态 | 说明 |
|---|---|---|
| 文档按日期分类 | ✓ 完成 | docs/2026-04-27, docs/2026-04-28 |
| AGENTS.md更新规范 | ✓ 完成 | 添加强制日志规则 |
| 方案文档创建 | ✓ 完成 | AIC_INVERSE_GENERATION_PLAN.md |
| 工作日志创建 | ✓ 完成 | WORK_LOG.md |
| chunkSize=16 | ✓ 完成 | tiling.cpp, kernel, golden |
| INVERSE_SHAPE=16 | ✓ 完成 | stage1.h |
| 原版算子验证 | ✓ 完成 | 精度测试通过 |

### 🔄 进行中

| 任务 | 状态 | 说明 |
|---|---|---|
| AIC InitInverseMatrix()设计 | 待实现 | 方案已确定，需编码 |
| 同步流程调整 | 待实现 | 需随方案一起修改 |

### ❌ 阻塞项

| 问题 | 状态 | 解决方案 |
|---|---|---|
| Kernel挂起(507015) | 待解决 | 方案3: AIC生成初始矩阵 |

---

## 核心问题分析

### 问题：InverseCompute只处理半块

**现象**：
```
chunkSize=16, TASK_RATIO=2 → halfChunkSize=8

InverseCompute(AIV):
  subBlockIdx=0: 处理 rows[0-7], 输出16×8
  subBlockIdx=1: 处理 rows[8-15], 输出16×8
  
NeumannLpowCompute(AIC):
  需要完整16×16矩阵做 L²=L×L
  
结果：workspace布局冲突，AIC读到不完整数据
```

### 方案：AIC生成初始矩阵

**详见**: `docs/2026-04-28/AIC_INVERSE_GENERATION_PLAN.md`

**核心思路**：
- 将(I-L)矩阵生成从AIV Vector操作改为AIC Cube操作
- AIC一次生成完整16×16矩阵写入INVWsGm_
- 避免subBlock分块冲突

---

## 实施步骤（待执行）

### Step 1: 验证workspace布局
```bash
# 检查 tiling 计算
grep "neumannWsSz" op_host/*_tiling.cpp

# 当前: neumannWsSz = sizeHigh * aiCoreNum * c * c * 3
# chunkSize=16: 每个core需要 3×16×16 = 768 元素
```

### Step 2: AIC InitInverseMatrix()实现
```cpp
// 位置: stage1.h AIC分支 ProcessOneChunk
// 功能:
//   1. 从kkWsGm_读取下三角L矩阵
//   2. 构造I-L写入INVWsGm_
//   3. 拷贝L到L_powWsGm_
```

### Step 3: 同步流程调整
```
原流程:
  AIV InverseCompute → flag 0x7 → AIC等待
  
新流程:
  AIC InitInverseMatrix → flag → AIV等待(或继续)
```

### Step 4: 测试验证
```bash
bash build.sh --pkg --soc=ascend910b --ops=chunk_gated_delta_rule_neumann -j16
bash output/*.run --install
cd tests/pytest && bash test_run.sh single
```

---

## 关键代码位置

| 功能 | 文件 | 行号 |
|---|---|---|
| InverseCompute(当前) | stage1.h | 484-528 |
| NeumannLpowCompute | stage1.h | 806-810 |
| NeumannMmulCompute | stage1.h | 812-816 |
| NeumannAddCompute | stage1.h | 530-546 |
| ProcessOneChunk AIC | stage1.h | 305-345 |
| ProcessOneChunk AIV | stage1.h | 346-387 |
| SetGlobalTensors | stage1.h | 80-126 |
| Workspace tiling | tiling.cpp | 136-138 |

---

## 经验教训

### 1. 文档必须及时更新
- 每次对话后更新日志
- 技术决策记录原因
- 问题状态跟踪进度

### 2. 算子修改需保持架构一致
- TASK_RATIO=2 → halfChunkSize=chunkSize/2
- 不能单独改chunkSize不改相关逻辑
- workspace布局必须匹配tiling计算

### 3. AIC/AIV分工需明确
- 矩阵操作优先用AIC/Cube
- Vector操作用于mask/element-wise
- 初始矩阵生成不应依赖subBlock分块

---

## 相关文档

- [方案文档](./AIC_INVERSE_GENERATION_PLAN.md)
- [工作日志](./WORK_LOG.md)
- [原版对比](../../chunk_gated_delta_rule/)
- [历史文档](../2026-04-27/)