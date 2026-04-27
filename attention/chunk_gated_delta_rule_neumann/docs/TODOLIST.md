# TODOLIST - CHUNK=16 + Neumann Migration Progress

## Overview

| 目标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| **chunkSize** | 64 | **16** (统一) | Pending |
| **求逆方法** | 前代法 (32×32) | Neumann 级数 (16×16) | Pending |
| **精度策略** | FP32 全程 | **FP16 MMA + FP16 Add** | Pending |

---

## Phase 1: Tiling Layer (op_host)

| Task | File | Line | Status | Notes |
|------|------|------|--------|-------|
| 1.1 修改 chunkSize 常量 | `*_tiling.cpp` | 113 | Pending | `c = 64 → 16` |
| 1.2 计算 Neumann workspace | `*_tiling.cpp` | 126-134 | Pending | 新增 L_pow + tmp 空间 |
| 1.3 更新 tilingData 结构 | `*_tiling_data.h` | - | Pending | 确认字段兼容性 |

---

## Phase 2: Kernel Constants (op_kernel)

| Task | File | Line | Status | Notes |
|------|------|------|--------|-------|
| 2.1 INVERSE_SHAPE 常量 | `*_stage1.h` | 32 | Pending | `32 → 16` |
| 2.2 INVERSE_COUNT 常量 | `*_stage1.h` | 33 | Pending | 重新计算 buffer 数量 |
| 2.3 UB buffer 重新分配 | `*_stage1.h` | 158-163 | Pending | 按新 chunkSize 分配 |

---

## Phase 3: Stage1 Inverse Refactor (op_kernel)

| Task | File | Status | Notes |
|------|------|--------|-------|
| 3.1 删除 AttnInverseMMCompute | `*_stage1.h` | Pending | CHUNK=16 无需块间乘法 |
| 3.2 简化 InverseAIV 调用 | `*_stage1.h` | Pending | 去除 offset 分块逻辑 |
| 3.3 新增 NeumannInverseCompute | `*_stage1.h` | Pending | 3步倍增：L²→L⁴→L⁸ |
| 3.4 新增 L_powWsGm, tmpWsGm | `*_stage1.h` | Pending | Workspace GM 地址 |
| 3.5 修改 InverseCompute | `*_stage1.h` | Pending | 调用 Neumann 函数 |

---

## Phase 4: Golden Sync (tests)

| Task | File | Status | Notes |
|------|------|--------|-------|
| 4.1 chunk_size 参数 | `*_golden.py` | Pending | `chunk_size = 64 → 16` |
| 4.2 测试数据生成 | `generate_test_inputs.py` | Pending | 验证 seqlen 与 chunk 对齐 |
| 4.3 精度阈值验证 | `test_aclnn_*.cpp` | Pending | max_abs_err < 0.01 |

---

## Phase 5: Build & Verify

| Task | Command | Status | Notes |
|------|---------|--------|-------|
| 5.1 编译算子 | `bash build.sh --pkg --soc=ascend910_93` | Pending | 验证编译通过 |
| 5.2 安装算子 | `bash build.sh --install` | Pending | 验证安装成功 |
| 5.3 运行测试 | `bash build.sh --run_example` | Pending | 精度对比 |
| 5.4 性能对比 | - | Pending | 原64 vs 新16 吞吐对比 |

---

## ✅ Confirmed Decisions

| 决策项 | 用户确认 |
|--------|----------|
| chunkSize | **统一为 16** (Stage1/2/3 全部修改) |
| 精度策略 | **FP16 MMA + FP16 Add** (与 FlashKDA 对齐) |
| 计划位置 | 项目 `docs/` 目录 (非 SKILL) |
| 进度跟踪 | TODOLIST.md 实时更新 |

---

## Progress Summary

```
Phase 1 (Tiling):     0/3  (0%)
Phase 2 (Constants):  0/3  (0%)
Phase 3 (Stage1):     0/5  (0%)
Phase 4 (Golden):     0/3  (0%)
Phase 5 (Verify):     0/4  (0%)
──────────────────────────
Total:               0/18 (0%)
```

---

## Next Actions

1. [ ] 确认 chunkSize 统一修改策略（全部 16）
2. [ ] 开始 Phase 1: Tiling 层修改
3. [ ] 编写 NeumannInverseCompute 函数框架