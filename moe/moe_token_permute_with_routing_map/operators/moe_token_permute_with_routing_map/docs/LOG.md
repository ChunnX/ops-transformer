# moe_token_permute_with_routing_map 开发日志

## 项目信息

| 项目 | 内容 |
|------|------|
| 算子名称 | moe_token_permute_with_routing_map |
| 算子类型 | MOE Token 重排（带路由映射） |
| 开发类型 | 功能增强（已有算子） |
| 原始代码路径 | `moe/moe_token_permute_with_routing_map/` |
| 创建时间 | 2026-05-25 |

## 需求概述

增强 `moe_token_permute_with_routing_map` 算子，支持 `dropAndPad=false` 时每个 token 选中的专家数可以少于 topk 的场景。

### 主要修改点

1. **maskselect**：新增输出总 output token 个数
2. **sort**：当总 token 个数 != topk * token 个数时，走多核排序分支；归并排序最后单核输出时用 -1 补齐每个 token 的 topk 个专家
3. **tokencopy**：判断对应位置为 -1 时不搬出

## 当前开发状态

| 阶段 | 任务 | 状态 | 完成时间 |
|------|------|------|----------|
| 1.1 | 开发准备 | ✅ | 2026-05-25 |
| 1.2 | 需求分析 | ✅ | 2026-05-25 |
| 1.3 | 方案设计 | ✅ | 2026-05-25 |
| 1.4 | 测试设计 | ⬜ | - |
| 2.1 | 迭代一：maskselect + sort 补齐 + tokencopy 跳过 | ✅ | 2026-05-25 |
| 2.2 | 迭代二 | ⬜ | - |
| 2.3 | 迭代三 | ⬜ | - |
| 3.1 | 性能验收 | ⬜ | - |
| 4.1 | 文档与示例 | ⬜ | - |
| 4.2 | 代码检视 | ⬜ | - |
| 4.3 | 开发总结 | ⬜ | - |

## 交付物清单

| 文件 | 状态 |
|------|------|
| docs/LOG.md | ✅ |
| docs/REQUIREMENTS.md | ✅ |
| docs/DESIGN.md | ✅ |
| docs/TEST_DESIGN.md | ⬜ |
| docs/DESIGN_REVIEW.md | ⬜ |
| docs/MRGSORT_OUT.md | ✅ |
| issues/ | ✅ |

## 开发记录

| 时间 | 阶段 | 摘要 |
|------|------|------|
| 2026-05-25 | 1.1 | 创建结构化开发目录，初始化开发日志 |
| 2026-05-25 | 1.2 | 完成需求分析文档，明确三大修改点（maskselect/sort/tokencopy） |
| 2026-05-25 | 1.3 | 完成详细设计文档，设计非对齐场景 tiling key 分支、-1 补齐策略、跳过 -1 搬移逻辑 |
| 2026-05-25 | 1.3 | 设计修正：tiling 侧不修改，kernel 侧运行时判断并重算参数；actualOutTokens 通过局部变量传递，不走 GM |
| 2026-05-25 | 2.1 | maskselect: Process() 新增 `int32_t& actualOutTokens` 输出参数，maskselect 计算后直接返回 |
| 2026-05-25 | 2.1 | sort: moe_mrgsort_out.h 新增 PadOutput() scalar 补齐方法，使用 -1 补齐每个 token 的 topk 专家 |
| 2026-05-25 | 2.1 | sort: moe_sort_multi_core.h Init() 从 tiling data 读取 needPad/topK/numTokens 初始化 mrgsortParam |
| 2026-05-25 | 2.1 | tokencopy: moe_index_copy.h / moe_index_copy_split_d.h 增 indicesValue == -1 跳过逻辑 |
| 2026-05-25 | 2.1 | kernel: 新增 KernelRecalcVBSMultiCoreParams kernel 侧重算 VBS 多核参数；GENERAL_OP_IMPL 判非对齐→重算+设 needPad+走多核排序 |
| 2026-05-29 | 文档 | 新增 docs/MRGSORT_OUT.md：MoeMrgsortOut SortOut 阶段实现逻辑（含 PadOutput 多轮归并与跨 chunk 补齐） |