# MoeInitRoutingV3 — A3 与 A5 实现差异对比

> A3：`moe_init_routing_v3_tiling.cpp` + `moe_init_routing_v3.cpp`，适用 Atlas A2/A3 系列
> A5：`moe_init_routing_v3_tiling_arch35.cpp` + `moe_init_routing_v3_apt.cpp`，适用 Ascend 950PR/950DT（3510 RegBase）

A3 与 A5 实现的是**同一个算子接口**（`aclnnMoeInitRoutingV3`），功能与数学语义完全一致，但**实现路径**因目标芯片架构不同而显著分化。下表按维度对比。

## 1. 顶层差异一览

| 维度 | A3 | A5 |
|------|----|----|
| 目标芯片 | Atlas A2 / A3 训练/推理 | Ascend 950PR / 950DT（RegBase 3510） |
| Tiling 模板注册优先级 | `REGISTER_OPS_TILING_TEMPLATE(..., 10000)`，默认 fallback | `IsCapable()` 检测 `IsRegbaseSocVersion`，命中后优先 |
| Tiling 类名 | `MoeInitRountingV3TilingBase` | `MoeInitRoutingV3Arch35TilingClass` |
| Tiling 数据结构 | `MoeInitRoutingV3TilingData` | `MoeInitRoutingV3Arch35TilingData`（新结构） |
| Kernel 头文件目录 | `op_kernel/` 顶层 | `op_kernel/arch35/` 子目录 |
| Kernel 入口分发宏 | `GET_TILING_DATA` | `REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT` |
| Kernel 任务类型 | `KERNEL_TYPE_MIX_AIV_1_0` | 不设置（仅判 `g_coreType == AIC` return） |

## 2. 量化模式支持范围（最核心功能差异）

| quantMode | 含义 | A3 | A5 |
|-----------|------|:--:|:--:|
| -1 | 非量化 | ✓ | ✓（含 FP8/MXFP4 直转） |
| 0 | 静态量化（INT8） | ✓ | ✓ |
| 1 | 动态量化（INT8） | ✓ | ✓ |
| 2 / 3 | MXFP8（E5M2 / E4M3FN） | ✗ | ✓ |
| 4 / 5 | FP8 PerGroup RoundScale | ✗ | ✓ |
| 6 | HIF8 直转 | ✗ | ✓ |
| 7 | HIF8 PerTensor | ✗ | ✓ |
| 8 | HIF8 PerToken | ✗ | ✓ |
| 9 | MXFP4 | ✗ | ✓ |
| 11 / 12 | FP8 PerBlock | ✗ | ✓ |
| 13 | INT4 动态量化 | ✗ | ✓ |
| 14 / 15 | FP8 PerGroup + Amax clamp | ✗ | ✓ |
| 16 / 17 | MXFP8 RoundScale + Amax | ✗ | ✓ |

→ A5 比 A3 多支持 **11 种量化模式**。

## 3. 平台与硬件适配差异

| 项 | A3 | A5 |
|----|----|----|
| UB 大小使用 | 直接使用 `aicoreParams_.ubSize` | 使用 `availUbSize_ = totalUbSize_ - SIMT_DCACHE_SIZE(64KB)`，需为 SIMT 预留 DCache |
| 单核排序元素上限 | 仅受 UB 约束（`sortLoopMaxElement`） | 额外受 `AscendC::Sort` API 能力约束 `SORT_API_MAX_ELEM = 32*255` |
| Overflow 寄存器 | 不涉及 | 入口保存 `GetCtrlSpr<OVERFLOW_MODE_CTRL>`，每个分支退出前 `SetCtrlSpr` 恢复 |
| SOC 版本感知 | 仅读 `aivNum / ubSize` | 额外读取 `socVersion_` |

## 4. Tiling 实现差异

### 4.1 类结构

- A3：`MoeInitRountingV3TilingBase`，校验集中在 4 个大函数（`CheckAttr / CheckInputShape / CheckOutShape / CheckDtype`）。
- A5：`MoeInitRoutingV3Arch35TilingClass`，校验拆分为 20+ 个细粒度子函数（`ValidateExpertTokensNumType / ValidateExpertNum / ValidateDropPadMode / ValidateExpertCapacity / ValidateQuantMode / ValidateRowIdxType / CheckInputX / CheckInputExpertIdx / CheckInputScale / CheckStaticQuantScale / CheckInputOffset / CheckOutputExpandedX / CheckOutputExpandedRowIdx / ...`）。

### 4.2 GetShapeAttrsInfo 时机

- A3：标准流程，在框架调用时即获取 shape/attr。
- A5：`GetShapeAttrsInfo` 重写为空，**延后到 `DoOpTiling` 内部调用 `DoGetShapeAttrsInfo`**，目的是让 `IsCapable()` 先于 shape 读取执行。

### 4.3 GatherOut Tiling 分流

A3 仅有 `Tiling4GatherOutCompute`（按 ep/quantMode/dropPad 在内部组合）。

A5 按 quantMode 顶层分流：

```
MXFP8 / MXFP8 RS+Amax / MXFP4  → Tiling4GatherOutMxQuant
FP8 PerBlock / PerGroup         → Tiling4GatherOutFP8Quant
DropPad 且非全载                → Tiling4SrcToDstDropPadCompute + Tiling4GatherOutDropPadCompute
MXFP8 NoQuant (x is fp8)        → Tiling4GatherOutMxFP8NoQuantCompute
其它                            → Tiling4GatherOutCompute
```

### 4.4 性能模板

| 模板 | A3 | A5 |
|------|:--:|:--:|
| 低时延性能模板（shape `(1,7168)` 专化） | ✓（TilingKey `2000000`） | ✗ |
| 大 batch 性能模板（N∈[384,8192], k=8） | ✓（`1200000/1300000` 段） | ✗ |
| Counting Sort 优化 | ✓（`ComputeCountingSortTiling`） | ✗ |
| 全载模板 | ✓（非量化/静态/动态） | ✓（非量化/静态/动态，含 INT4） |

→ A5 简化为只保留全载模板 + 通用模板，去掉了 A3 的三类性能模板（low-latency / large-batch / counting-sort），但通过 14 种量化模式提供更广覆盖。

## 5. TilingKey 编码差异

两套完全独立的编码体系，**互不兼容**。

| 模板段 | A3 | A5 |
|--------|----|----|
| 空间起点 | `1 000 000` | `10 000 000` |
| 全载 | `2 100 000 / 2 200 000 / 2 300 000+ep*1e4+smoothType*1e3` | `200 000 + quantModeFactor*1e4` |
| 性能模板 | `2 000 000`（低时延）/ `1 200 000~1 301 000`（大 batch） | ✗ |
| 空tensor | `3 000 000` | `3 000 000`（巧合相同） |
| 通用量化模板 | `(quantMode+1)*1e4`（即 -1→0, 0→1, 1→2） | 多数量化模式重映射，如 MXFP8→3、FP8Group→5、FP8PerBlock→12、Group+Amax→15、MXFP8RS→17 |
| DropPad 编码位 | `dropPadMode*1e2` | `dropPadMode*1e2`（同） |

## 6. Kernel 分发差异

### 6.1 分支数量

- A3 kernel：约 20+ 个 tiling key 分支，覆盖 3 类量化 × 2 索引类型 × 2 排序核数 × DropPad/非 DropPad × 性能模板。
- A5 kernel：约 40+ 个 tiling key 分支，覆盖 14 类量化 × 2 索引类型 × 2 排序核数 × DropPad/非 DropPad。

### 6.2 Kernel 类实例化差异

A3 的量化模式用枚举模板参数（`GATHER/SCATTER/NO_SCALE/SCALE_1H/SCALE_EH`），全载动态量化分 6 个 key 区分。

A5 将 INT4 与 INT8 动态量化合并到同一 tilingkey，通过 `quantMode` 字段在 kernel 内区分：

```cpp
} else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_FULLLOAD_DYNAMIC_QUANT)) {
    if constexpr (... IsSameType<DTYPE_EXPANDED_X, int8_t>::value) {
        if (isInt8DynamicQuant) { ... MoeV3FullLoadDynamicQuant<DTYPE_X> ... }
    } else if constexpr (... IsSameType<DTYPE_EXPANDED_X, int4b_t>::value) {
        if (isInt4DynamicQuant) { ... MoeV3FullLoadDynamicQuant<DTYPE_X, int4b_t> ... }
    }
}
```

减少 key 空间膨胀，是 A5 的设计取舍。

### 6.3 空 tensor 处理

- A3：`n==0` 即进入空 tensor 分支，仅对 `expertTokensCountOrCumsumOut` 刷零。
- A5：进一步处理 `cols==0 且 n*k>0` 的子场景，调用 `InitGlobalMemory` 对 `expandedRowIdxOut / expandedScaleOut` 刷零；并按量化模式判定 `expandedScale` 是否需要初始化（MXFP8/MXFP4/FP8_PERBLOCK 在 cols==0 时 scale 元素数为 0，无需 init）。

## 7. 子模块对比

| 功能 | A3 头文件 | A5 头文件 |
|------|----------|----------|
| 排序 | `moe_v3_sort_one_core.h / moe_v3_sort_multi_core.h` | `arch35/moe_v3_sort_one_core.h / arch35/moe_v3_sort_multi_core.h` |
| 全载 | `moe_v3_full_load*.h` (5 个) | `arch35/moe_v3_full_load_*.h` |
| Gather 非量化 | `moe_v3_gather_out.h / moe_v3_gather_out_droppad.h` | `arch35/moe_v3_gather_out.h / moe_v3_gather_out_droppad.h` |
| Gather 量化 | `moe_v3_gather_static_quant.h / moe_v3_gather_dynamic_quant.h` | A3 同名 + `arch35/moe_v3_gather_mxfp8_quant.h / moe_v3_gather_fp8_perblock_quant.h / moe_v3_gather_fp8_group_quant.h / moe_v3_gather_hif8_*.h / moe_v3_gather_mxfp4_quant.h` |
| MXFP8/4 直转（非量化） | ✗ | `arch35/moe_v3_gather_out_mxfp8.h / moe_v3_gather_out_mxfp4.h` |
| Counting Sort | `moe_v3_full_load_cut_origin_t.h / moe_v3_cut_origin_t.h / moe_v3_sort_actual_expert.h / moe_v3_gather_sort_multi_core.h / moe_v3_sort_multi_core_performance.h` | ✗ |
| DropPad 索引 | `moe_v3_row_idx_gather_droppad_dynamic.h / moe_v3_row_idx_gather_droppad.h / moe_v3_gather_out_droppad.h / moe_v3_gather_droppad_static_quant.h` | `arch35/moe_v3_row_idx_gather_droppad.h / moe_v3_gather_out_droppad.h` |

## 8. 选型建议

- **Atlas A2 / A3 系列芯片**：必须用 A3，TilingKey 体系与硬件匹配；性能模板覆盖低时延/大 batch/全载/counting sort 四类特化，对常见训练场景优化充分。
- **Ascend 950PR / 950DT**：必须用 A5，TilingKey 与 RegBase 3510 硬件约束（SIMT DCache / overflow 寄存器 / Sort API 上限）严格绑定；扩展的 11 种量化模式是核心增量。
- **量化模式选择**：仅需 INT8/INT4 量化 → 两者皆可；需要 FP8/MXFP8/MXFP4/HIF8 等新格式 → 必须 A5。
- **迁移注意**：A3 ↔ A5 之间**不能直接复制 kernel 代码**，因为 (1) TilingKey 编码不同；(2) Tiling 数据结构不同；(3) 头文件路径不同；(4) RegBase 架构约束（SIMT/overflow）A3 不具备。

## 9. 一句话总结

A3 是**功能精简但性能模板丰富**的通用版本（3 种量化 × 4 类性能模板），A5 是**功能全面但模板简化**的 RegBase 特化版本（14 种量化 × 仅全载模板），两者共享相同算子语义但实现路径在 TilingKey、平台适配、模块组织、空 tensor 处理等多个层面完全分化。
