# PoolKeyIndexer 算子代码检视报告

> 检视日期：2026-08-20
> 检视范围：op_host（def/infershape/tiling）、op_kernel/common、template_tiling_key、arch22/arch35 kernel
> 检视方法：参照 asc-devkit 官方规范，按 API 层级、内存管理、同步、功能 Bug、整数计算、性能、架构兼容 7 个维度系统性检视

---

## 总览

| 维度 | 数量 |
|------|------|
| 检视文件 | 10 个（op_host 4 + common 2 + template_tiling_key 1 + arch22 kernel 3 + arch35 kernel 3，实际完整检视 7 个） |
| 总计问题 | 13 个 |
| P0（阻塞） | 1 |
| P1（重要） | 3 |
| P2（改进） | 9 |
| API 层级 | SIMD C++ Basic API（Tpipe + LocalMemAllocator 混合） |
| 内存模式 | LocalMemAllocator + `__ubuf__` 数组 |

---

## P0 — 阻塞（必须修复）

### [P0] PA 场景 keyStride0 回退为 0 导致内核地址计算错误

- **文件**：`op_host/pool_key_indexer_tiling.cpp:361-365`
- **问题**：当 `GetDynamicInputStride` 返回 nullptr（`keyStridesVec_` 为空）时，`info.keyStride0` 被设为 `0`。但内核 `KeyNd2NzForPA`（`arch22/pool_key_indexer_service_cube.h:251-253`、`arch35/pool_key_indexer_service_cube.h:278-279`）使用 `blockId * constInfo_.keyStride0` 计算块地址。若 keyStride0=0，则所有 block 都映射到地址 0，导致 PA 场景读取错误数据（**静默错误，无报错**）。
- **根因**：2补非连续修复时，PA 路径的 keyStride0 从"用 shape 计算"改为"用实际 stride"，但遗漏了"实际 stride 不可用时回退到连续 stride"的兜底逻辑。
- **参考**：LIV2 `lightning_indexer_v2_tiling.cpp:1007-1010` 正确处理了此回退：当 keyStride0 为空时用 `blockSize_ * n2Size_ * headDim_` 作为连续 PA 张量的预期 stride[0]。
- **影响范围**：arch22 + arch35 的所有 PA_BBND 场景，当框架未提供 stride 信息时（某些图模式或旧框架版本）触发。
- **修复方案**：

```cpp
// 当前（错误）：
if (!keyStridesVec_.empty()) {
    info.keyStride0 = keyStridesVec_[0];
} else {
    info.keyStride0 = 0;  // ← P0: 导致 PA 地址全为 0
}

// 修复后：
if (!keyStridesVec_.empty()) {
    info.keyStride0 = keyStridesVec_[0];
} else {
    info.keyStride0 = info.blockSize * info.n2Size * info.headDim;
}
```

---

## P1 — 重要（应该修复）

### [P1-1] workspace kSeqSize 回退 8192，TND/PA 大序列可能溢出

- **文件**：`op_host/pool_key_indexer_tiling.cpp:852`
- **问题**：TND 和 PA 场景 `info.s2Size` 被设为 0（`:367`、`:393`），导致 `kSeqSize` 回退为 8192。但 TND 的实际 S2（= `keyShape->GetDim(0)`，即 T2）和 PA 的实际 S2（由 actual_seq_k 决定）都可能远超 8192（如 32768+）。scoreGm 的大小按 `kSeqSizeAlign` 计算（`:856`），若实际 S2 > 8192，workspace 不够大，内核写入越界。
- **影响**：TND/PA 大序列场景 workspace 不足，内核越界写入（可能导致段错误或数据损坏）。
- **修复方案**：TND 场景应将 `s2Size` 设为 `keyShape->GetStorageShape().GetDim(0)`；PA 场景应从 actual_seq_k 取最大值作为 s2Size 上界，用于 workspace 计算。

### [P1-2] FP16/BF16 TND+PA_BBND 在 template SEL 中缺失

- **文件**：`op_kernel/pool_key_indexer_template_tiling_key.h:124-144`
- **问题**：SEL 条目 7（FP16 PA）和 8（BF16 PA）的 `LAYOUT_Q` 仅含 `PKI_LAYOUT_BSND`，不包含 `PKI_LAYOUT_TND`。FP8 条目 9（`:151`）才同时包含 BSND 和 TND。因此 FP16/BF16 + TND + PA_BBND 组合没有匹配的模板实例，运行时会因找不到模板而失败。
- **影响**：用户使用 FP16/BF16 + TND query + PA_BBND key 组合时，算子无法运行（框架报"找不到模板"错误）。
- **修复方案**：在 SEL 条目 7、8 的 `LAYOUT_Q` 列表中增加 `PKI_LAYOUT_TND`，或新增专门的 FP16/BF16 TND/PA_BBND SEL 条目。

### [P1-3] 950 FP8 weights 仅声明 FP16，tiling 接受 BF16（死代码）

- **文件**：`op_host/pool_key_indexer_def.cpp:119-123` vs `op_host/pool_key_indexer_tiling.cpp:276-279`
- **问题**：950 配置中 FP8 组合（位置 2、3）的 weights 仅声明为 `DT_FLOAT16`。但 tiling 代码对 FP8 模式的 weights 同时接受 `DT_FLOAT16` 和 `DT_BF16`。op_def 的 DataType 列表按位置匹配，BF16 weights + FP8 query 组合在框架层就会被拒绝，tiling 中的 BF16 分支成为不可达死代码。
- **影响**：注释误导（def.cpp `:106-107` 声称"tiling still accepts BF16 weights at runtime"），存在死代码分支；若用户需要 BF16 weights + FP8，功能不可用。
- **修复方案**：要么在 def.cpp 950 配置中为 FP8 组合增加 BF16 weights 的声明（扩展组合数），要么在 tiling 中移除 BF16 分支使行为一致。

---

## P2 — 改进（建议修复）

| # | 文件:行号 | 问题 | 建议 |
|---|-----------|------|------|
| 1 | `infershape.cpp:114` | `poolSize==0` 冗余检查（死代码，`:97` 已校验 >=1） | 移除 `poolSize == 0 \|\|` 分支 |
| 2 | `tiling.cpp:164` | 同上 `poolSize==0` 冗余检查 | 同上 |
| 3 | `tiling.cpp:897` / `tiling.h:148` | `keyDequantScaleStride0` 恒为 0（死字段，步骤4范畴） | 步骤4实施时接线，或删除避免误导 |
| 4 | `tiling.h:127` | `platformInfo` 字段从未使用（死字段） | 删除 |
| 5 | `tiling.cpp:910` | `ubSize` 获取后未用于 tiling 参数约束 | 考虑用 ubSize 约束 s1BaseSize/mBaseSize |
| 6 | `common.h:20` | `PA_BSND` vs 其他文件 `PA_BBND` 命名不一致（值一致=2，功能正确） | 统一为 `PA_BBND` |
| 7 | `common.h:115` | `quantMode` 初始值 -1（host 语义）与 tiling data 的 +1 偏移（tpl 值 0/1/2）语义不一致 | 初始值改为 0（tpl 语义），或注释说明 |
| 8 | `tiling.cpp:200` | `GetStride()` 返回 `int64_t` → `vector<uint32_t>` 隐式截断（极大 stride >4G elements 时溢出） | 加 `static_cast` 并做范围检查 |
| 9 | `common.h:38-57` | RunInfo 部分字段缺少默认初始化（`loop`/`bN2Idx`/`bIdx`/`gS1Idx`/`s2Idx`/`actMBaseSize`/`actualSingleProcessSInnerSize`/`tensorQueryOffset` 等），使用前未赋值为 UB | 补充 `= 0` 默认值 |

---

## 通过项

### op_def 正确性
- [x] IgnoreContiguous 设置（pool_key + k_descale 设了，其余 AutoContiguous）
- [x] 950 dtype 4 组合基本正确（FP16/BF16 无量化 + FP8 per-token + FP8 mxFP8）
- [x] ValueDepend 设置（pool_tail_k / actual_seq_q / actual_seq_k 均为 INT64 + ValueDepend）

### infershape 正确性
- [x] BSND 输出 shape 为 3D `(B, S1, topk+poolSize-1)`
- [x] TND 输出 shape 为 2D `(T1, topk+poolSize-1)`
- [x] returnValue=false 时 sparseValuesOut 为 `(0,)`
- [x] 属性校验完整（layout/mask_mode/quant_mode/pool_size/topk/整除性）

### tiling 正确性
- [x] CheckKeyContiguous expectedStrides 计算正确（PA/BSND 4D、TND 3D）
- [x] keyStride0 填值逻辑正确（PA 用实际 stride，BSND/TND 置 0）—— 但回退值有 P0 问题
- [x] GET_TPL_TILING_KEY 参数顺序与 DECL 一致（8 参数）
- [x] quant_mode +1 映射正确（host -1/0/1 → tpl 0/1/2）
- [x] Platform API 调用正确（GetCurNpuArch/GetCoreNumAiv/GetCoreNumAic/GetCoreMemSize/CalcTschBlockDim）

### template_tiling_key.h
- [x] TPL_ 前缀消除所有同名冲突
- [x] DECL/SEL 参数与 def.cpp 属性对应正确
- [x] 末尾分号完整

### 常量与结构体
- [x] 无重复常量定义（build_fix_record.md #2/#13/#15 已修复）
- [x] ConstInfo 字段覆盖 kernel 运行时需求

---

## 修复优先级建议

| 优先级 | 问题 | 修复难度 | 建议时间 |
|--------|------|----------|----------|
| **P0** | keyStride0 回退为 0 | 1 行改动 | 立即 |
| **P1-1** | workspace kSeqSize 回退 8192 | 中等（需取实际 S2） | 本迭代 |
| **P1-2** | FP16/BF16 TND/PA SEL 缺失 | 中等（扩展 SEL 条目） | 本迭代 |
| **P1-3** | 950 FP8 weights dtype | 简单（对齐 def/tiling） | 步骤4时 |
| **P2** | 死代码/死字段/命名 | 简单 | 逐项清理 |

---

## 参考

- LIV2 回退处理：`lightning_indexer_v2_tiling.cpp:1007-1010`
- CANN 模板 Tiling Key 系统：`template_argument.h`，`ASCENDC_TPL_UINT_DECL/SEL`
- 开发规范：`torch_extension/cann_ops_transformer/docs/torch_extension_guidelines.md`
- 编译修复记录：`docs/build_fix_record.md`
