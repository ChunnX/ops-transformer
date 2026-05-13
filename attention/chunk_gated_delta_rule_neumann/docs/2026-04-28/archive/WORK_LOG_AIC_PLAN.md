# 工作日志 - 2026-04-28

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


**日期**: 2026-04-28  
**主要工作**: 确定AIC生成初始矩阵方案，整理文档结构

---

## 对话记录

### 1. 问题分析（上午）

**发现的问题**：
- InverseCompute(AIV)只处理16×8半块（TASK_RATIO=2导致halfChunkSize=8）
- NeumannLpowCompute(AIC)需要完整16×16矩阵做L²=L×L
- workspace布局冲突：两个subBlock试图拼成完整矩阵

**对比数据**：
| 参数 | 值 | 影响 |
|---|---|---|
| chunkSize | 16 | ✓ 正确 |
| TASK_RATIO | 2 | halfChunkSize=8 |
| INVERSE_SHAPE | 16 | ✓ 正确 |
| InverseCompute处理范围 | 16×8半块 | ❌ 不完整 |

### 2. 方案讨论（下午）

**提出方案**：
- 方案1: TASK_RATIO=1（破坏现有架构）
- 方案2: InverseCompute整体矩阵操作（需要修改逻辑）
- 方案3: AIC生成初始矩阵（用户选择）

**用户确认**：采用方案3，将(I-L)初始矩阵生成从AIV移至AIC

**理由**：
- Neumann求逆应该是纯AIC/Cube MMA操作
- AIC可以生成完整16×16矩阵
- 避免subBlock分块冲突

### 3. 文档整理

**操作**：
1. 创建日期目录结构：`docs/2026-04-27/`, `docs/2026-04-28/`
2. 移动历史文档到对应日期目录
3. 创建方案文档：`AIC_INVERSE_GENERATION_PLAN.md`
4. 更新AGENTS.md：
   - 添加文档结构规范
   - 添加强制规则：及时修改日志
   - 添加Debug History部分

---

## 技术决策

| 决策 | 值 | 原因 |
|---|---|---|
| 初始矩阵生成方式 | AIC生成完整16×16 | 解决subBlock分块冲突 |
| TASK_RATIO | 保持2 | 不破坏现有架构 |
| 求逆精度 | FP16 MMA + FP16 Add | 与FlashKDA对齐 |

---

## 下一步工作

### 待实现

1. **AIC InitInverseMatrix()函数**
   - 位置：`stage1.h` ProcessOneChunk AIC分支
   - 功能：从kkWsGm_读取L矩阵，生成(I-L)写入INVWsGm_
   
2. **同步流程调整**
   ```
   原同步: AIC等待AIV InverseCompute(flag 0x7)
   新同步: AIC完成InitInverseMatrix后通知AIV继续
   ```

3. **验证workspace大小**
   - 确保每个core有3个完整16×16矩阵空间

### 验证步骤

1. 编译安装
2. pytest单算子测试
3. 精度验证（对比golden）

---

## 文件变更记录

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `docs/2026-04-27/` | 创建目录 | 移动历史文档 |
| `docs/2026-04-28/` | 创建目录 | 今日文档 |
| `docs/2026-04-28/AIC_INVERSE_GENERATION_PLAN.md` | 新建 | 方案详解 |
| `docs/2026-04-28/WORK_LOG.md` | 新建 | 本日志 |
| `AGENTS.md` | 更新 | 添加文档规范和强制规则 |

---

## 问题状态

| 问题 | 状态 | 解决方案 |
|---|---|---|
| Kernel挂起(507015) | 待解决 | AIC生成初始矩阵 |
| InverseCompute半块 | 已明确原因 | 方案3解决 |
| 文档结构混乱 | ✓ 已解决 | 按日期分类 |
| 日志不及时更新 | ✓ 已规范 | 强制规则已添加 |

---

**最后更新**: 2026-04-28 17:30