# moe/docs/ 算子分析文件审查报告

**审查依据**: `.claude/commands/analyze-op.md` 模板规范
**初始审查日期**: 2026-04-21
**修复完成日期**: 2026-04-22
**审查范围**: `moe/docs/` 下 31 个 `*_full_analysis.md` 文件
**报告类型判定**: 所有文件均为全量分析报告，应遵循模板 **Report Type A (Full-Operator Analysis)**

---

## 一、模板要求的章节清单 (Type A)

| 编号 | 必需章节 | 子章节 |
|------|---------|--------|
| 1 | 算子概述 | - |
| 2 | 目录结构与文件职责 | - |
| 3 | Host 侧分析 | 3.1 算子定义, 3.2 Tiling 策略, 3.3 Shape 推导, 3.4 ACLNN API |
| 4 | Kernel 侧分析 | 4.1 Kernel 入口与初始化, 4.2 核心计算流程, 4.3 数据结构与内存布局, 4.4 多核并行策略, 4.5 通信模式(如适用) |
| 5 | Host-Kernel 接口 | 5.1 Tiling 数据结构定义, 5.2 Workspace 布局 |
| 6 | 完整数据流 | 含 ASCII 数据流图 |
| 7 | 性能相关设计 | - |
| 8 | 依赖关系 | Include 依赖树 |
| 9 | 关键约束与注意事项 | - |
| 10 | 与同族变体的核心差异 | - |

---

## 二、修复状态总览

### 2.1 修复结果

**所有 31 个文件已全部按 Type A 模板规范重写完成。**

修复批次：

| 批次 | 文件数 | 评分范围 | 状态 |
|------|--------|---------|------|
| 第 1 批 | 6 | Low | 已完成 |
| 第 2 批 | 7 | Medium-Low | 已完成 |
| 第 3 批 | 18 | Medium/High | 已完成 |

### 2.2 逐文件合规状态

| # | 文件名 | 原评分 | 现状态 | 原主要缺失 |
|---|--------|-------|--------|-----------|
| 1 | moe_compute_expert_tokens | High | **已修复** | 性能相关设计 |
| 2 | moe_re_routing | High | **已修复** | 性能相关设计 |
| 3 | moe_init_routing | Medium-High | **已修复** | Host-Kernel接口, 依赖关系 |
| 4 | moe_token_permute | Medium-High | **已修复** | Host-Kernel接口, 依赖关系 |
| 5 | moe_init_routing_v2 | Medium | **已修复** | Host-Kernel接口, 性能相关, 依赖关系 |
| 6 | moe_init_routing_v2_grad | Medium | **已修复** | Host-Kernel接口, 性能相关, 依赖关系, Workspace |
| 7 | moe_init_routing_v3 | Medium | **已修复** | Host-Kernel接口, 性能相关, 依赖关系 |
| 8 | moe_init_routing_quant | Medium | **已修复** | Tiling策略, Host-Kernel接口, 性能相关, 依赖关系 |
| 9 | moe_init_routing_quant_v2 | Medium | **已修复** | Tiling策略, Host-Kernel接口, 性能相关, 依赖关系 |
| 10 | moe_finalize_routing | Medium | **已修复** | 完整数据流, 依赖关系, Host-Kernel接口 |
| 11 | moe_finalize_routing_v2 | Medium | **已修复** | 完整数据流, 依赖关系, Host-Kernel接口 |
| 12 | moe_finalize_routing_v2_grad | Medium-Low | **已修复** | 完整数据流, 依赖关系, Host-Kernel接口, Workspace, 与同族变体差异 |
| 13 | moe_gating_top_k | Medium | **已修复** | 完整数据流, 依赖关系, Host-Kernel接口, ACLNN API |
| 14 | moe_gating_top_k_softmax | Medium | **已修复** | 完整数据流, 依赖关系, Host-Kernel接口, ACLNN API |
| 15 | moe_gating_top_k_softmax_v2 | Medium | **已修复** | 依赖关系, Host-Kernel接口, ACLNN API, 性能相关 |
| 16 | moe_fused_topk | Medium | **已修复** | 依赖关系, Host-Kernel接口, 性能相关 |
| 17 | moe_token_unpermute | Medium | **已修复** | 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 18 | moe_token_unpermute_grad | Medium | **已修复** | 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 19 | 3rd_moe_gather_v2 | Medium | **已修复** | Tiling分析, Kernel分析(无源码), Host-Kernel接口, 依赖关系 |
| 20 | moe_token_permute_grad | Medium-Low | **已修复** | 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 21 | 3rd_moe_inplace_index_add | Medium-Low | **已修复** | Shape推导, ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 22 | 3rd_moe_inplace_index_add_with_sorted | Medium-Low | **已修复** | Shape推导, ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 23 | 3rd_moe_masked_scatter | Medium-Low | **已修复** | Shape推导, ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 24 | moe_token_unpermute_with_ep | Medium-Low | **已修复** | ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 25 | moe_token_unpermute_with_ep_grad | Medium-Low | **已修复** | ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 26 | moe_token_permute_with_ep | Low | **已修复** | ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系, 关键约束 |
| 27 | moe_token_permute_with_ep_grad | Low | **已修复** | ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 28 | moe_token_permute_with_routing_map | Low | **已修复** | Shape推导, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系, Workspace |
| 29 | moe_token_permute_with_routing_map_grad | Low | **已修复** | ACLNN API, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 30 | moe_token_unpermute_with_routing_map | Low | **已修复** | ACLNN API, Shape推导, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |
| 31 | moe_token_unpermute_with_routing_map_grad | Low | **已修复** | ACLNN API, Shape推导, 完整数据流, Host-Kernel接口, 性能相关, 依赖关系 |

---

## 三、原系统性问题及修复措施

### 3.1 结构不规范 — 扁平编号 → 嵌套 Type A 结构

**原问题**: 所有 31 个文件使用扁平编号（`3. 算子定义`, `4. Tiling`, `5. Shape`...），Host 侧内容拆散为同级章节。

**修复措施**: 全部重写为 Type A 嵌套结构:
```
3. Host 侧分析
   3.1 算子定义
   3.2 Tiling 策略
   3.3 Shape 推导
   3.4 ACLNN API
4. Kernel 侧分析
   4.1 Kernel 入口与初始化
   4.2 核心计算流程
   4.3 数据结构与内存布局
   4.4 多核并行策略
   4.5 通信模式
5. Host-Kernel 接口
   5.1 Tiling 数据结构定义
   5.2 Workspace 布局
```

### 3.2 原高频缺失章节 — 全部补全

| 排名 | 原缺失章节 | 原缺失文件数 | 修复状态 |
|------|-----------|-------------|---------|
| 1 | 依赖关系 | 29/31 (93.5%) | 全部补全（含 include 依赖树） |
| 2 | Host-Kernel 接口 | 29/31 (93.5%) | 全部补全（含 TilingData 定义 + Workspace 布局图） |
| 3 | 性能相关设计 | 24/31 (77.4%) | 全部补全（含硬件约束动机说明） |
| 4 | 完整数据流 | 23/31 (74.2%) | 全部补全（含 ASCII 数据流图） |
| 5 | ACLNN API | 16/31 (51.6%) | 全部补全（含两段式 API 签名） |

### 3.3 编号风格 — 统一阿拉伯数字

**原问题**: `moe_compute_expert_tokens` 使用中文序号（一~八），其余用阿拉伯数字。

**修复措施**: 全部统一为阿拉伯数字嵌套编号。

---

## 四、修复内容一致性

所有 31 个文件统一包含以下结构:

| 章节 | 内容要求 | 完成度 |
|------|---------|--------|
| 1. 算子概述 | 算子名称、类、功能描述、支持平台、数据类型 | 31/31 |
| 2. 目录结构与文件职责 | 目录树 + 各文件职责注释 | 31/31 |
| 3.1 算子定义 | 输入输出接口表格 + 属性表格 | 31/31 |
| 3.2 Tiling 策略 | TilingData 字段描述 + TilingKey 编码 + 策略选择逻辑 | 31/31 |
| 3.3 Shape 推导 | 输出 Shape 推导公式 + 校验规则 | 31/31 |
| 3.4 ACLNN API | 两段式 API 签名（有则写，无则说明） | 31/31 |
| 4.1 Kernel 入口 | TilingKey 分支列表 + 模板参数 | 31/31 |
| 4.2 核心计算流程 | 数学公式 + 分阶段流程描述 | 31/31 |
| 4.3 数据结构与内存布局 | UB 内存布局图或关键数据结构 | 31/31 |
| 4.4 多核并行策略 | 核间/核内切分策略 | 31/31 |
| 4.5 通信模式 | SyncAll/事件标志/无通信 | 31/31 |
| 5.1 Tiling 数据结构定义 | 指向 3.2 节 + CompileInfo 说明 | 31/31 |
| 5.2 Workspace 布局 | ASCII 布局图 | 31/31 |
| 6. 完整数据流 | 端到端 ASCII 数据流图 | 31/31 |
| 7. 性能相关设计 | 列表形式 + 硬件约束动机 | 31/31 |
| 8. 依赖关系 | include 依赖树 | 31/31 |
| 9. 关键约束与注意事项 | 约束表格 | 31/31 |
| 10. 与同族变体的核心差异 | 对比表格 | 31/31 |

---

## 五、特殊处理说明

| 算子类型 | 处理方式 |
|---------|---------|
| 3rd 系列算子（无自定义 kernel/tiling） | 保留 Type A 结构，无 kernel/tiling 的章节明确标注"复用 CANN 内置"或"无自定义" |
| 双路径算子（membase + regbase） | 4.1 节同时列出两条路径的入口，4.2 分别描述 |
| 量化算子 | 4.2 节详细描述量化链路（BF16→FP32→INT8 等） |
| 反向算子 | 4.2 节包含前向公式 + 反向梯度公式 |

---

## 六、结论

- **31/31 文件 (100%) 已全部修复为 Type A 模板规范**
- 原最普遍缺失的依赖关系、Host-Kernel 接口、性能相关设计、完整数据流、ACLNN API 章节均已补全
- 所有文件统一使用阿拉伯数字嵌套编号
- 所有文件包含: 概述 → 目录结构 → Host 分析(3.1-3.4) → Kernel 分析(4.1-4.5) → 接口(5.1-5.2) → 数据流 → 性能 → 依赖 → 约束 → 差异对比
