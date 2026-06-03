# QkvRmsNormRopeCacheWithKScale 拆分设计文档

本目录基于 `../qkv_rms_norm_rope_cache_with_kscale_design.md` 对首期 arch35 交付做二级设计拆分。拆分不再按代码实现任务细切，而是按设计评审和开发落地需要的几个视角组织。

## 文档列表

| 文档 | 视角 | 目的 |
|------|------|------|
| `01_operator_proto_and_formula.md` | 算子原型与计算公式 | 固化接口、shape、dtype、属性、核心数学语义 |
| `02_deliverables_overview.md` | 交付件纵览 | 说明需要新增/修改的 API、op_host、op_kernel、tests、配置文件 |
| `03_golden_design.md` | Golden 设计 | 定义 Python/Numpy golden 的计算顺序、FP8 行为和对比策略 |
| `04_tiling_and_ut_design.md` | Tiling & UT 设计 | 暂定 tiling 输入输出、待评估项、host/kernel UT 范围；具体 tiling 策略先待定 |
| `05_kernel_design.md` | Kernel 设计 | 按期望流水、代码结构、controller、basic_block、cube、vec 分层描述 |
| `06_inter_core_sync_design.md` | 核间同步设计 | 独立定义 AIC/AIV CrossCore flag、pipe、事件计数和 pingpong buffer 复用规则 |
| `07_intra_core_sync_design.md` | 核内同步设计 | 独立定义同核 pipe 间正向/反向同步、event id 分配和 buffer 复用规则 |
| `08_implementation_task_breakdown.md` | 实现子任务拆解 | 基于 05/06/07 拆解 kernel、核间同步、核内同步和验证子任务 |

## 全局约束

1. 首期只交付 arch35/Ascend950。
2. `rotation` 和 `v_scale` 保持接口 optional，但当前 checker 强制非空。
3. `cu_seq_len` 固定为 `[Batch + 1]` 前缀和输入，不再设计双模式 tiling key。
4. `head_nums` 是必选 ListInt 属性，内容为 `[Nq, Nk, Nv]`，host tiling 需要传递到 kernel。
5. 首期 fast path 只支持 `Nv == Nk`；接口仍保留独立 `Nv` 字段，`Nv != Nk` 由 checker/tiling 拒绝。
6. 所有 `float8_e4m3` 均按 `float8_e4m3fn`。
7. `rotation` 是外部输入矩阵，当前 dense Cube 计算 `rope_out @ rotation`，不启用 FWHT。
8. Q/K dynamic quant 对齐 arch35 `dynamic_quant` FP8 E4M3FN 路径：`scale = max_abs / 448`，量化按 `Div -> Cast fp8`。
9. V cache 写回属于后处理语义，但不依赖 Cube 输出；AIV 在完成 Q/K preprocess、发送 `A_READY` 后即可提前执行 V quant 与 `v_cache` 写回。

## 与完整设计文档的关系

完整设计文档保留为主文档，路径为：

```text
../qkv_rms_norm_rope_cache_with_kscale_design.md
```

本目录中的文档是面向评审和开发的拆分说明。若两者冲突，以完整设计文档的最新版本为准，并同步更新本目录。
