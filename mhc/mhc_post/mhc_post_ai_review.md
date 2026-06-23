# MhcPost 代码检视报告

## 检视概要

| 项目 | 内容 |
|------|------|
| 检视目标 | `/home/developer/ops-transformer/mhc/mhc_post/` |
| 检视日期 | 2026-06-16 |
| 检视模式 | 全量检视 (cpp-secure + ascendc-topk + cpp-general + cpp-style) |
| 代码侧别 | 混合 (Host/Tiling + Kernel/Device) |

## 统计数据

| 级别 | 数量 |
|------|------|
| 必须改 | 5 |
| 应该改 | 11 |
| 供参考 | 3 |
| 已通过 | 37 |

---

## 必须改 (5 条)

### 1. aclnnMhcPostGetWorkspaceSize 未校验 workspaceSize / executor 空指针

- **文件**: `op_host/op_api/aclnn_mhc_post.cpp:266-268`
- **规范**: cpp-secure.md 3.5 + ascendc-topk.md TOPK-1
- **后果**: 传 nullptr 直接 SIGSEGV 崩溃

```cpp
// 当前
CHECK_COND(CheckNotNull(x, hRes, hOut, hPost, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR, ...);
// 缺少
CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
```

---

### 2. l0op::MhcPost 未校验 AllocTensor 返回值

- **文件**: `op_host/op_api/mhc_post.cpp:59`
- **规范**: cpp-secure.md 3.5 + ascendc-topk.md TOPK-1
- **后果**: 显存不足时传空指针进 Kernel，设备侧崩溃

```cpp
// 当前
auto output = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
return MhcPostAICore(x, hRes, hOut, hPost, output, executor);
// 修复
auto output = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
if (output == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER, "MhcPost AllocTensor failed.");
    return nullptr;
}
return MhcPostAICore(x, hRes, hOut, hPost, output, executor);
```

---

### 3. arch35 kernel ComputeCopyOutAllX 缺少 PipeBarrier

- **文件**: `op_kernel/arch35/mhc_post.h:217-219` vs `op_kernel/mhc_post.h:222-226`
- **规范**: ascendc-topk.md TOPK-11
- **后果**: 同一 outF32 buffer 在 n×n 循环中反复读-改-写，无 PipeBarrier 时编译器可能乱序发射 Axpy，产生 RAW 冲突，**静默计算出错**

```cpp
// arch35 当前（缺少 PipeBarrier）
for (int64_t j = 0; j < tilingData_->n; j++) {
    Axpy(outF32, xF32[j * dNumAlign], hResGm_.GetValue(hResBase + j * tilingData_->n + i), dNum);
}
// 应恢复为（与原版一致）
for (int64_t j = 0; j < tilingData_->n; j++) {
    PipeBarrier<PIPE_V>();
    Axpy(outF32, xF32[j * dNumAlign], hResGm_.GetValue(hResBase + j * tilingData_->n + i), dNum);
    PipeBarrier<PIPE_V>();
}
```

---

### 4. DoMulAndAdd 用 tilingData_->n == 4 决定走 N4 分支，未考虑 nTail < 4

- **文件**: `op_kernel/arch35/mhc_post_regbase.h:124`
- **规范**: ascendc-topk.md TOPK-6
- **后果**: 若 tiling 策略使 `nTail_ < 4` 但 `n == 4`，`DoMulAndAdd_N4` 越界读取 x/hRes buffer

```cpp
// 当前
if (tilingData_->n == 4) {
    DoMulAndAdd_N4(hPostUb, hResUb, nNum, dNum);  // nNum 可能 < 4
}
// 修复
if (nNum == 4) {
    DoMulAndAdd_N4(hPostUb, hResUb, nNum, dNum);
}
```

---

### 5. arch35 CopyInX 中 DataCopyExtParams.srcStride 类型不匹配

- **文件**: `op_kernel/arch35/mhc_post.h:240`
- **规范**: cpp-secure.md 1.1
- **后果**: `static_cast<int64_t>` → `uint32_t` 缩窄转换，极端值下截断导致搬运长度错误

```cpp
// 当前
static_cast<int64_t>((tilingData_->d - dNum) * sizeof(T)),  // 应为 uint32_t
// 修复
static_cast<uint32_t>((tilingData_->d - dNum) * sizeof(T)),
```

---

## 应该改 (11 条)

| # | 文件:行 | 问题 | 规范 |
|---|---------|------|------|
| 1 | `op_host/mhc_post_tiling.cpp:424` | `3 * (...)` 魔鬼数字，应命名 `NUM_DATA_STREAMS` | cpp-general 4.2 |
| 2 | `op_host/mhc_post_tiling.cpp:448` | `(n_ + 2)` 魔鬼数字，应命名 `NON_X_BUFFER_COUNT` | cpp-general 4.2 |
| 3 | `op_host/mhc_post_tiling_regbase.cpp:133,458,471` | `8`/`4` 魔鬼数字，应替换为 `DB_BUF * sizeof(...)` | cpp-general 4.2 |
| 4 | `op_host/mhc_post_tiling_regbase.cpp:132` | `(N_MIN + 7) / 8 * 8` 中 `7`/`8` 应命名为对齐常量 | cpp-general 4.2 |
| 5 | `op_kernel/arch35/mhc_post_regbase.h:132-133,187` | `dAlign`/`nAlign` 中 `16`/`8` 魔鬼数字 | cpp-general 4.2 |
| 6 | `op_kernel/arch35/mhc_post_regbase.h:182-265` | `DoMulAndAdd_N4` 84行，与 `DoMulAndAdd` 60行 ~70% 重复 | 可维护性 |
| 7 | `op_host/mhc_post_tiling.cpp:442` | `totalCount = bsOuter_ * dOuter_` 无溢出保护 | cpp-secure 2.1 |
| 8 | `op_host/mhc_post_tiling_regbase.cpp:453,466` | 无条件 for 循环依赖 UB_SIZE break，缺上限保护 | cpp-secure 1.3 |
| 9 | `op_host/mhc_post_tiling_regbase.cpp:133 vs 458` | `IsCapable` 与 `ComputeTiling` 中 UB 公式逻辑重复 | cpp-general 1.3 |
| 10 | 多处 | 行宽超 120 字符 | cpp-style 2.1 |
| 11 | `op_host/mhc_post_infershape.cpp:120-131` | `InferDataType` 直接透传 dtype，无校验 | ascendc-topk TOPK-7 |

## 供参考 (3 条)

| # | 文件 | 问题 |
|---|------|------|
| 1 | `op_host/mhc_post_tiling.cpp:104` | `dtype_` 成员变量 set 后未使用 |
| 2 | `op_host/mhc_post_infershape.cpp:62-70` | `ToString().c_str()` 临时对象绑同一表达式，风格脆弱 |
| 3 | `op_kernel/arch35/mhc_post_regbase.h` | `__VEC_SCOPE__` 内微架构寄存器操作，需确认 MaskMergeMode::ZEROING 对尾部填充行为 |

## 已通过 (37 条)

安全编码（17条）、TOPK（10条）、通用编码（14条）、代码风格（13条）均通过，详见下方：

| 类别 | 条款 | 说明 |
|------|------|------|
| 安全 | 3.3, 4.1, 5.1-5.4 | 输入校验链完善、无内存泄漏 |
| TOPK | TOPK-2,4,5,8-10,12,13 | API 用法正确、GM 偏移用 int64、无 thread_local |
| 通用 | 1.3, 2.2-2.6, 4.1, 5.3, 8.1 | 无循环依赖、无宏常量、无 switch 裸写 |
| 风格 | 1.1-1.5, 2.4, 3.1, 3.3 | 命名规范、文件头、括号使用正确 |

## 重点区域

**`op_kernel/arch35/` (Ascend950) 是问题集中区**：5 条必须改中有 4 条在此目录。该目录代码与原版 `op_kernel/` 存在若干未对齐的差异，建议整体走查一遍。

---
*报告生成: 2026-06-16 | 规范: cpp-secure.md + ascendc-topk.md + cpp-general.md + cpp-style.md*
