# QkvRmsNormRopeCacheWithKScale 拆分设计文档

本目录基于 `../qkv_rms_norm_rope_cache_with_k_scale_design.md` 对首期 arch35 交付做二级设计拆分。拆分不再按代码实现任务细切，而是按设计评审和开发落地需要的几个视角组织。

## 文档列表

| 文档 | 视角 | 目的 |
|------|------|------|
| `01_operator_proto_and_formula.md` | 算子原型与计算公式 | 固化接口、shape、dtype、属性、核心数学语义 |
| `02_deliverables_overview.md` | 交付件纵览 | 说明需要新增/修改的 API、op_host、op_kernel、tests、配置文件 |
| `03_golden_design.md` | Golden 设计 | 定义 Python/Numpy golden 的计算顺序、FP8 行为和对比策略 |
| `04_tiling_and_ut_design.md` | Tiling 与验证设计 | 定义 token-only tiling、AIV token split、资源公式、Host/Tiling/API UT 与 DiscussionForum kernel 验证范围 |
| `05_kernel_design.md` | Kernel 设计 | 按期望流水、代码结构、controller、basic_block、cube、vec 分层描述 |
| `06_inter_core_sync_design.md` | 核间同步设计 | 独立定义 AIC/AIV CrossCore flag、pipe、事件计数和 pingpong buffer 复用规则 |
| `07_intra_core_sync_design.md` | 核内同步设计 | 独立定义同核 pipe 间正向/反向同步、event id 分配和 buffer 复用规则 |
| `08_implementation_task_breakdown.md` | 实现子任务拆解 | 基于 05/06/07 拆解 kernel、核间同步、核内同步和验证子任务 |
| `09_vf_optimization_summary.md` | VF 优化总结 | 总结 Q/K 前处理 VF 展开、寻址、load/store 接口替换和验证记录 |
| `10_pack_elimination_performance.md` | pack 消除性能记录 | 记录 TND->NTD 路径 pack 指令消除前后的 msprof task duration 对比 |
| `11_mrope_bf16_int8_extension_design.md` | M-RoPE 组合模式设计 | 定义 `MROPE_BF16_Q_INT8_K` 模式及其接口、公式、资源与验证合同 |
| `12_mrope_kernel_staged_implementation_design.md` | M-RoPE Kernel 分阶段设计 | 保留初版到 compact 的演进记录；当前实现真值以本页“当前实现真值”和代码为准 |
| `13_operator_beginner_guide.md` | 算子入门讲解 | 从 Q/K/V 拆分、RMSNorm、RoPE、rotation、量化和 Cache 写回理解完整计算链路 |
| `14_pipeline_analysis.md` | 流水分析与制图记录 | 分析 Q/K/V、VEC/CUBE、跨 tile rolling、L1 和 UB 分配，并汇总配套 SVG |
| `15_pipeline_and_memory_allocation_guide.md` | 流水与内存分配综合讲解 | 结合四张 SVG 串联计算顺序、核间同步、L1/UB 多缓冲和性能分析方法 |
| `16_qkv_stc.md` | 模式级 STC | 枚举 RoPE/M-RoPE 全部合法与非法布局组合，并按模式记录 800 例中的测试场景、门限和结果 |
| `17_qkv_mindmap_nodes.md` | 脑图节点 | 以三个一级节点组织全部输入、当前支持的四种模式组合和严格对应 full800 的泛化用例生成与结果 |
| `18_mrope_math_beginner_guide.md` | M-RoPE 数学入门 | 从三轴位置输入和 lane 交织选源开始，推导 RMSNorm、M-RoPE、rotation、BF16 Q、INT8 K、FP8 V 与 Cache 写回公式 |
| `19_cache_stride_and_write_address_guide.md` | Cache 地址入门 | 说明 Cache shape/stride、slot 到 block/offset 的转换，以及 K/V/K-scale 从 UB 写回 GM 时的二维搬运 stride |
| `20_tiling_partition_beginner_guide.md` | Tiling 切分入门 | 说明全局 T 分核、核内 token tile 选择、UB/L0C 资源约束、双 AIV 尾块切分及完整算例 |

## 全局约束

1. 首期只交付 arch35/Ascend950。
2. `rotation` 和 `v_scale` 当前必须非空，checker 需要强制校验。`v_scale` 的 shape 按场景区分：RoPE
   使用 `float32[Nv]`，按 V head 广播；M-RoPE 使用 `float32[Nv,D]`，按 V head 和通道逐元素相乘，
   即 `V[t,n,d] * v_scale[n,d]`。两个 shape 不允许跨场景兼容。
3. RoPE 路径中，`query_start_loc` 固定为 `[Batch + 1]` 当前调用 local 前缀和，`seq_lens [Batch]` 提供每个 batch 的实际序列长度，用于计算 RoPE 历史偏移；M-RoPE 组合模式的接口级 optional 与场景判定覆盖规则见第 14 条和 `11_mrope_bf16_int8_extension_design.md`。
4. `head_nums` 是必选 ListInt 属性，内容为 `[Nq, Nk, Nv]`，host tiling 需要传递到 kernel。
5. RoPE 与 M-RoPE 统一要求 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`；接口仍保留独立的
   `Nq/Nk/Nv` 字段，不满足任一关系时由 checker/tiling 拒绝。
6. 所有 `float8_e4m3` 均按 `float8_e4m3fn`。
7. `rotation` 是外部输入矩阵，当前 dense Cube 计算 `rope_out @ rotation`，不启用 FWHT。
8. RoPE 场景的 `q_quant_mode=PerTokenPerHead` 时，Q/K dynamic quant 对齐 arch35 `dynamic_quant` FP8 E4M3FN 路径：`scale = max_abs / FP8_E4M3FN_MAX`，其中 `FP8_E4M3FN_MAX = 448`；量化按 `Div -> Cast fp8`。M-RoPE 使用 `q_quant_mode=NoQuant`，Q 不量化。
9. V cache 写回属于后处理语义，但不依赖 Cube 输出。当前 V-first 调度在 `Vec::ComputeTile` 内依次执行 `QuantVToFp8 -> BuildRopeBf16NzUb -> ScatterVCache -> CopyRopeBf16NzUbToL1Nz`，返回后由 BasicBlock 发送 `A_READY`。这里的 V-first 表示 V vector quant 先于 Q/K preprocess 发起，不表示 `v_cache` 写回先于 Q/K preprocess；V 写回仍早于 Q/K AL1 handoff 和 `A_READY`。
10. Host tiling 当前使用 `TOKEN_TILE_PER_AIV_CAP = 4`；`MAX_TOKEN_TILE = 16` 是资源硬上限，cap=4 使偶数主路径实际满足 `tokenTile <= 8`。
11. InferShape 保持最小路径：只做安全读取和有效输出 shape/dtype 推导所需的最小校验，其中保留 `NTD->TND` 拒绝和 `qkvN >= Nq`；其余完整 shape/dtype/layout/资源契约由 ACLNN checker 与 tiling/base tiling 负责，不在 InferShape 重复实现。
12. `ops-transformer` 内保留 Host/Tiling/API UT；kernel 数值、layout、tail、多 tile、cache stride 和端到端无死锁验证直接使用 DiscussionForum 的 NPU golden/TTK 用例，不要求新增 `tests/ut/op_kernel` 或 ST target。
13. 核内 event 通过实际 id、pipe pair、live buffer 冲突和 Init/End Set/Wait 配平审计，不以 event id 范围 `static_assert` 作为门禁。
14. M-RoPE 组合模式将 `query_start_loc/seq_lens` 声明为 OPTIONAL，并新增 token-major `mrope_position: int32[T,3]`、`mrope_section=[t,h,w]`、`q_quant_mode` 和 `q_out_dtype`。场景由 optional effective presence 判定：空 `mrope_section=[]` 与未传属性等价；RoPE 使用成对的 `query_start_loc/seq_lens`，M-RoPE 使用 `mrope_position` 与非空 section。RoPE 固定为 `PerTokenPerHead + FP8`，M-RoPE 固定为 `NoQuant + BF16`。OpDef 五个输出均为 REQUIRED；Torch `q_scale` 类型为 `Tensor`，RoPE 返回 scale，M-RoPE 运行值为 `None`；ACLNN 的 `qScaleOptional` 在 M-RoPE 下为 `nullptr`。D128 下 section 三项非负，H/W 单项上限均为 21，第 0 项没有独立 lane 上限且不参与计算，三项之和不超过 64。完整合同以 `11_mrope_bf16_int8_extension_design.md` 为准。
15. kernel 使用独立的 `ROPE_MODE/K_QUANT_MODE/Q_QUANT_MODE` 编译期参数；host 只生成 `(ROPE,FP8,PER_TOKEN_PER_HEAD)` 与 `(MROPE,INT8,NO_QUANT)` 两组相关组合。进入 kernel 后禁止通过 tiling data flag、`q_quant_mode`、dtype 或 optional GM 地址做 runtime 场景判断。RoPE 保留 3 个 layout 选择；M-RoPE 只实例化 1 个 `TND->TND` compact direct-Q 路径，总计 4 个注册选择。`K_QUANT_MODE` 在 device 侧仍是编译期量化路径，host 侧直接由 `kCache` dtype 生成对应 key，不维护额外的 host 枚举状态。

16. M-RoPE compact 是唯一可选的 M-RoPE 流水。`kRows<=14`、Q/K preprocess 不超过 40 KiB、输入行数不超过 80 是 tile 可行性约束；host 会降低 `tokenTile` 重新选择，连最小 `tokenTile=2` 都放不下时才拒绝，不切换第二条 M-RoPE 路径。当前 compact 布局使用 4-slot input/raw pool、独立 K rotation/K quant staging、4 KiB `[Nv,D]` V scale 常驻区和 1024-token position window；compact reserve 为 40 KiB，总 UB 为 `80+80+14+40+12=226 KiB`。16 MiB workspace 属于框架强制预留，不作为本算子可调资源。

## 当前实现真值（2026-08-03）

以下条目覆盖历史草案与当前代码不一致的地方，优先级高于各节中未标注“历史”的旧方案描述：

1. M-RoPE 只支持 `TND->TND` compact direct-Q。不存在可由 tiling key 选择的第二条 M-RoPE golden 路径；未满足 compact tile 资源约束时只缩小 tile，最小 tile 仍不可行则报错。
2. M-RoPE 不再按输出大小生成第二个 Fixpipe L2 cache-hint 模板；q_out 使用默认 GM cache 行为。删除前 `aaa73a6dc` 基线的 5 轮交替 A/B 中，`T=45000` 慢 `0.42%`、`T=65536` 快 `0.29%`，两个成对 95% 置信区间都跨 0，归类为性能波动。
3. 16 MiB workspace 仍是框架强制预留，不作为本算子可调资源。
4. 大输出 E2E 用例为 `n16_kv2_mrope_perf_t45000_tnd`，保持 `Nq/Nk/Nv=16/2/2`、`T=45000`。Q/K/V/K-scale 实际写入约 198.10 MiB，四路输出均需 100% 对齐；该用例只覆盖大输出精度，不再承担 L2 模板选择语义。
5. 当前验证记录：Host/Tiling UT `120/120`，ACLNN UT `128/128`；DiscussionForum NPU 全量
   `185/185`，section-0 不变量比较 `3/3`。全部 49 个 M-RoPE case 默认使用按 case seed 可复现的
   均匀随机 `mrope_position[T,3]`；随机 `[Nv,D]` scale 覆盖 `Nv=1/2/8` 和长序列，`v_cache` 与
   CPU Golden 逐元素 `100%` 一致。RoPE A/B profiler 的历史性能记录见
   `10_pack_elimination_performance.md`。
6. M-RoPE 保留一拍 AIC/AIV rolling pipeline：控制迭代处理当前 tile 的 preprocess，同时处理上一拍
   `lastTile` 的 K。`StageMropeK(lastTile)` 完成后先返还 `MROPE_K_CONSUMED` rotation credit，再在同一
   控制迭代立即调用 `ScatterMropeK(lastTile)`。K quant 仍为双缓冲，每个 slot 由 `MTE3 -> V` reusable
   event 保护，只有 K cache 与 K scale 两路写回完成后才能复用。
7. M-RoPE Q 不经过 Vec 的 FP32->BF16 后处理。Cube Fixpipe 直接执行 F32->BF16 并写 `q_out` GM；不可达
   的 Vec `QCastBf16D128TndVfImpl` 路径已经删除。buffer 数量不再为每类资源暴露 common 全局
   `BUFFER_NUM`，而是在 Controller/BasicBlock/Vec owner 内根据物理 pool/slot 容量局部确定；各资源的
   use counter 仍保持独立。
8. M-RoPE `v_scale=[Nv,D]` 已在 `ops-transformer 6ca8b3400` 落地：ACLNN 与 base tiling 按场景
   精确校验 shape，kernel 循环前搬运有效的 `Nv*D` 个 FP32 scale，VF 通过编译期模板按
   `head*D+d` 逐通道读取；RoPE 继续使用 `[Nv]` head 广播。compact reserve 为 40 KiB，连同
   12 KiB position window 的总 UB 为 226 KiB，不增加 tiling key。
9. 与 `0311b5e7d` 的 2026-07-31 单轮 msprof 基线相比，M-RoPE `T=5379,Nq/Nk/Nv=16/2/2`
   task duration 从 `40.037 us` 变为 `40.920 us`，增加 `0.883 us`（`+2.21%`）；同轮 7 个 RoPE
   对照为 `-0.93%..+0.16%`。该增量对应逐通道 scale vector load，是新语义的实际成本；RoPE
   未出现同方向回归。原始数据位于 `/home/s00620103/tmp/qkv_perf_mrope_v_channel_20260803`。
10. 当前默认 msprof 矩阵包含 M-RoPE `T=5379/65536/131072`，并新增 RoPE 128K 配对 case。
    随机 `mrope_position`、ReduceAll 邻接、单轮采集下，M-RoPE task duration 分别为
    `40.052/517.545/1102.565 us`；128K 完整场景相对 64K 为 `2.130x`，本轮作为随机 position
    合同的新性能基线。原始数据位于
    `/home/s00620103/tmp/qkv_mrope_random_position_perf_20260803`。

## 与完整设计文档的关系

完整设计文档保留为主文档，路径为：

```text
../qkv_rms_norm_rope_cache_with_k_scale_design.md
```

本目录中的文档是面向评审和开发的拆分说明。完整设计文档中的历史草案可能早于当前实现；涉及接口合同、tiling key、UB 常量和同步协议时，以 `ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale` 当前代码及本目录中标注为“当前实现”的段落为准，并同步修正文档。

## 文档使用口径

1. `01` 第 2 节给出完整接口，其余章节区分 RoPE 公式与 M-RoPE 差异；`02` 到 `07`
   固化交付、tiling、kernel 和同步协议，并在末尾标出各视角的 M-RoPE 组合模式影响；`08` 记录
   实现任务与验收门禁；`09/10` 是带验证时间点的优化记录；`11` 固化完整 M-RoPE 合同。
2. “当前实现”描述必须能在 `ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale` 中找到直接对应；建议方案或后续扩展必须显式标注，不能与已落地行为混写。
3. 构建、UT、golden 和性能数字均是对应实现版本的历史记录。只修改文档不代表当前 worktree 已重新执行这些验证。
