# BSA MXFP8 全量化设计说明

> **已取代（2026-08-17）**  
> 几何与内存决策已修订为 **kvs 基块 128 / G=8 / Dpv=12**。请改读：  
> [`2026-08-17-bsa-mxfp8-design.md`](./2026-08-17-bsa-mxfp8-design.md)  
> 画布：`docs/superpowers/canvases/BSA-mxfp8-design-report.canvas.tsx`（及 Cursor canvases 同名文件）。  
> 下文保留为 0814 历史稿（256×128 基块、G=4、双切 k、半块 P）。

日期：2026-08-14
基线分支：`mxfp8_0814`（BSA MXFP4 全量化已完成的版本）
状态：~~设计已评审通过~~ → **历史稿，已被 2026-08-17 修订取代**

## 1. 目标

在现有 BSA MXFP4 全量化实现（`block_sparse_attention_kernel_arch35_mxfp4_full_quant.h` 一整套）之上扩展出 MXFP8 全量化路径，二者共存、由 tiling key 选择。

硬性约束：

- tile 几何不变：qs 基块 128、kvs 基块 256、D = 128。
- 一组（tile group）从 MXFP4 的 16 个基块（4096×128）改为 **4 个基块（1024×128）**。
- Q/K/V/P 全部 `e4m3fn`，micro-scale 全部 `e8m0`、MX 组恒 32 个元素，只做 OCP 模式（不做 MXFP4 的 CX 模式）。

## 2. 术语

| 术语 | 含义 |
| --- | --- |
| 基块（base tile） | 一次 softmax 的单位，256(kvs)×128(qs)。由 QK 的两条 128 列 mmad 拼成 |
| 组（tile group） | L0C 累加窗口，同时是 group-max / pScale 的作用域。MXFP8 = 4 个基块，MXFP4 = 16 个 |
| loop | 主循环里的全局基块序号（`coreTileLoop`） |
| Dpv | `PV_RE_DELAY_N`，PV 落后 softmax 的基块数（本文里单独的 D 一律指 embed 维 = 128） |
| G | `TILE_GROUP_N`，一组的基块数 |

数据流与 MXFP4 完全同构：一次 QK mmad 出 128(kvs)×128(qs) 进一块 L0C，两条拼成一个基块，fixpipe 成 fp16 落进某个 AIV 的 S 槽；基块按 `loop` 轮流分给 AIV0/AIV1（`GetSPBufferIdx(loop) = loop / 2 % 2`：`/2` 是"每个 AIV 隔一个拿一块"，`%2` 是该 AIV 自己的双缓冲）；一组 4 个基块里每个 AIV 拿 2 块；组尾两个 AIV 通过 L1 交换 localGlobalMax。

## 3. 流水与延迟

### 3.1 延迟下界推导

组 g 覆盖基块 `[gG, gG+G-1]`。该组最后一个基块在 loop `gG+G-1` 出 S；再过 `DELAY_PSCALE_N = 3` 拍，gmax 的 UB→L1 搬运与 pScale 计算完成。PV 处理组 g 的首块必须严格晚于这一时刻，而 PV 在 loop `L` 处理的是基块 `L-Dpv`，因此

```
Dpv >= G + 3
```

G = 4 时下界 7，取 **Dpv = 8**：与 MXFP4 取 20（下界 19 再加 1 拍余量）同一口径，且 8 是偶数，两个 AIV 各自静态分到 4 个 P 槽。

环大小恰好等于 Dpv：槽 `loop % 8` 在 loop `loop+8` 被 PV 读取，而主循环里 PV 块排在 softmax 写入之前，因此 8 个槽刚好不冲突，不需要 9 个。

### 3.2 常量

| 常量 | MXFP4 | MXFP8 | 位置 |
| --- | --- | --- | --- |
| `TILE_GROUP_N` | 16 | **4** | `arch35/kernel_utils.hpp` |
| `PV_RE_DELAY_N` | 20 | **8** | kernel 顶部 |
| `DELAY_PSCALE_N` | 3 | 3 | kernel 顶部 |
| `PRELOAD_TASK_CACHE_SIZE` | 21 | **9** | `= PV_RE_DELAY_N + 1` |
| `L1_P_BUF_CNT` / `L1_P_SCALE_BUF_CNT` | 20 | **8** | `= PV_RE_DELAY_N` |
| `UB_LOCAL_GROUP_MAX_CNT`（1K 为单位） | 20 | **8** | `= PV_RE_DELAY_N`，即 4 个 2K 逻辑槽 |

`isUpdatePScale`（组尾）与 `isTileGoupFirstTile`（组首）的判据不变，只是 `% TILE_GROUP_N` 的模数从 16 变 4，于是组首/组尾事件——PV mmad 置 `initC`、fixpipe L0C→UB、跨核 pScale/C2_V2 同步——的频率变成 4 倍。`GetLocalGroupMaxBufIdx(loop) = loop / 2 * 2 % 8`。

### 3.3 尾部收尾

主循环的 `while (coreTaskId < coreTaskNum || delay8CoreTaskId < coreTaskNum || delay3CoreTaskId < coreTaskNum)` 结构不变，延迟从 20 降到 8 只是缩短了 drain 长度；kv tile 数小于 8 的小 case 由同一套条件覆盖。

## 4. L1 分配（512 KB，占用 438 KB）

规则：**数据槽字节数翻倍，scale 槽字节数不变**。后者的依据是 `attention/common/op_kernel/matmul.h` 里 fp8 的 `LoadData2DMxParams.yStep = (k + 63) >> 5 >> 1` 与 fp4 的 `(singleK + 63) / 32 / 2` 是同一个公式——scale 平面的分形几何与元素位宽无关，只有数据侧的 C0 从 64 元素变 32 元素。

| 区 | 偏移 | 单槽 | 槽数 | 合计 | 说明 |
| --- | --- | --- | --- | --- | --- |
| P 数据 | 0 | 32 KB = 128×256×1B | 8 | 256 KB | 槽位 = `loop % 8` |
| P-scale | 256 KB | 1280 B | 8 | 10 KB | 128×2×(256/64+1)，与 fp4 相同 |
| Q 数据 | 266 KB | 16 KB = 128×128×1B | 2 | 32 KB | 每 task 只搬一次 |
| Q-scale | 298 KB | 512 B | 2 | 1 KB（+1 KB 对齐填充） | 与 fp4 相同 |
| KV 共享池 | 300 KB | 32 KB | 4 | 128 KB | QK 存 K，PV 存 V，都是 256(kvs)×128(D)×1B |
| KV-scale | 428 KB | 2 KB | 4 | 8 KB | 与 fp4 相同 |
| localGlobalMax | 436 KB | 512 B | 4 | 2 KB | 两个 AIV 各一行 |
| 空闲 | 438 KB | — | — | 74 KB | |

KV 槽能维持 32 KB 而不是 40 KB，前提是 PV 沿 k 切段（见 §5）后 rowsum seed 只需要 18 KB，不再是 36 KB。

## 5. L0 分配（L0A/L0B 各 64 KB，L0C 256 KB）

fp8 下 L0 是真正的瓶颈，而且**操作数角色是固定的**：现有 loader 是 K→L0A（`CopyL1ToL0AMxQKA5`）、Q→L0B（`CopyL1ToL0BMxQKA5`），于是 L0C 的 m 方向是 kvs、n 方向是 qs，正好对上 UB 里 S = 256(kvs)×128(qs) 的布局。**不能把 Q 挪到 L0A**——那样 L0C 出来的是 Sᵀ，下游 softmax/rescale 的整套 UB 布局都要推翻。

因此 L0A 必须同时装 K 与 Vᵀ，L0B 同时装 Q 与 Pᵀ。照搬 MXFP4 的双缓冲：L0A 需要 K 2×16 + Vᵀ 2×36 = 104 KB，L0B 需要 Q 2×16 + Pᵀ 2×32 = 96 KB，都超 64 KB。即使只让 PV 沿 k 切段，L0A 仍是 K 2×16 + Vᵀ 2×18 = 68 KB > 64 KB。

**决策：QK 与 PV 都沿 k 切 2 段——QK 沿 k(D) 切成 2 段各 64，PV 沿 k(S2) 切成 2 段各 128；同一基块的两段累加进同一块 L0C（首段沿用原有 `initC` 语义，第二段恒 false）。**

| | 偏移 | 单槽 | 槽数 | 合计 |
| --- | --- | --- | --- | --- |
| L0A：K | 0 | 8 KB = 128×64×1B | 2 | 16 KB |
| L0A：Vᵀ | 16 KB | 18 KB = 144×128×1B | 2 | 36 KB |
| L0B：Q | 0 | 8 KB = 128×64×1B | 2（两段都是 task 内常驻） | 16 KB |
| L0B：Pᵀ | 16 KB | 16 KB = 128×128×1B | 2 | 32 KB |
| L0C：QK | 0 | 64 KB | 2 | 128 KB |
| L0C：PV | 128 KB | 72 KB = 144×128×4B | 1 | 72 KB |

L0A 用 52 KB（余 12 KB），L0B 用 48 KB（余 16 KB），L0C 用 200 KB（余 56 KB）。偏移布局与 MXFP4 完全一致（QK 区 0…16 KB、PV 区 16 KB 起），只有槽内几何变了，改动集中在 loader 的 C0 与 step 参数。

scale 平面不占 L0 预算：MXFP4 的 `L0A_PV_BUF_SIZE = 18432` 恰好等于 Vᵀ 数据 144×256×0.5B，`L0A_QK_BUF_SIZE = 8192` 恰好等于 K 子块 128×128×0.5B，说明 mx `LoadData` 的 scale 落在 cube 的专用 scale 存储里，不在 L0A/L0B 地址空间。

每个基块的 mmad 条数：QK 从 2 条（m128/n128/k128）变 4 条（k=64），PV 从 1 条（k=256）变 2 条（k=128）。切 k 的收益：

- K/Vᵀ/Pᵀ 全部保住双缓冲，MTE1 与 MMAD 可重叠；
- rowsum seed 从 36 KB 降到 18 KB。seed 是借一个 KV 槽整槽填 1.0 后搬进 L0A 的，槽必须装得下 seed 镜像，18 KB ≤ 32 KB 才让 L1 的 KV 槽维持 32 KB（否则要涨到 40 KB，多吃 32 KB L1）；
- k=64 仍是 MX scale 组（32 元素）的整数倍，scale 侧不需要特殊处理。

代价是 mmad 条数翻倍、每条更短，指令开销与短 k 效率需实测。

被否决的方案：K 或 Vᵀ 退回单缓冲（L0A 能放下，但对应的 MTE1 与 mmad 完全串行）；全单缓冲（性能最差）。

## 6. UB 分配（256 KB，占用约 236 KB）

fp8 下一个基块的 P 从 16 KB 变 32 KB。若照搬"两块整基块交错"就是 P 区 64 KB，即使 localGroupMax 从 20 KB 省到 8 KB，总量仍是 268 KB 超出 12 KB。

**决策：P 区仍是 32 KB，但两块各放半个基块（128 列 kvs × 128 行 qs × 1B = 16 KB）。VF softmax 按 128 列分两次算，两半分别写进两个 UB P 槽、再各自拷到同一个 L1 P 槽的对应半区；算后半时前半正在拷 L1，重叠不丢。** 这与 QK 本来就按 128 列出两次结果同构，且 `CopyPUBToPL1` 已有"高低半区分别搬运"的分支可以复用。

可行性依据：softmax 量化 P 时用的指数基准是 `localGlobalMax`（上一组交换得到），不是本基块自己的 max，因此按 128 列切分不改变数值；`localGroupMax` 是跑动最大值，分两次更新即可。

| 区 | 偏移 | 大小 | 说明 |
| --- | --- | --- | --- |
| S（mm1Res） | 0 | 128 KB | fp16，2 槽各 64 KB = 256×128，按 256 B 交错 |
| oTmp（mm2Res） | 128 KB | 32 KB | fp32，不变 |
| local/globalRowSum | 160 KB | 512 B | 不变 |
| P | 160 KB+512 | 32 KB | **2 槽各 16 KB，每槽半个基块** |
| oTrans（复用 P 区） | 160 KB+512 | 18 KB | 不变 |
| pScale（复用 P 区） | 178 KB+512 | 8×1280 B = 10 KB | 不变 |
| attentionOut | 192 KB+512 | 32 KB | 不变 |
| peerGlobalMax | 224 KB+512 | 4×256 B | 不变 |
| softmaxMax | 225 KB+512 | 256 B | 不变 |
| localGroupMax | 225 KB+768 | **8×1 KB = 8 KB** | 原 20×1 KB；每 AIV 在飞 = Dpv/2 = 4 个 2K 槽 |
| localGlobalMax / updateScale / index | 233 KB+768 起 | 各 1 KB / 1 KB / 256 B | 不变，整体前移 12 KB |

## 7. 数据类型与量化链路

| 位置 | 类型 |
| --- | --- |
| q / k / v（GM） | `DT_FLOAT8_E4M3FN` |
| q/k/v descale（GM） | `DT_FLOAT8_E8M0`，MX 组 32：Q/K 沿 D，V 沿 S2 |
| S（UB） | fp16 |
| P（UB→L1） | `fp8_e4m3fn_t` |
| P-scale（UB→L1） | `fp8_e8m0_t` |
| L0C 累加 | fp32 |
| 输出 | 与现状一致（bf16 / fp16） |

- Fixpipe L0C→UB 走 **NO_QUANT**：反量化已由 mmad 内的组内 scale 完成，再挂 deqScalar 会二次缩放。
- rowsum trick 不变（Vᵀ 的 m 方向 pad 16 行常量 1.0）。seed 用数据 `0x38`（e4m3 的 1.0）配 scale `0x7F`（e8m0 的 1.0），乘积正好 1.0；不需要 MXFP4 的 4.0×0.25 变通。
- L1→L0 搬运只改数据侧 C0：`FP4_C0_ELEMS = 64` → `FP8_C0_ELEMS = 32`（影响 `LoadData2DParamsV2.kStep` 等），MX 参数侧仍按 k/64。
- 精度：e4m3 有 3 位尾数（相对步长 12.5%），远好于 e2m1 的 1 位；且组从 16 基块缩到 4 基块后，固定指数基准更贴近各基块自身的 max，下溢更少。

## 8. 主机侧与 tiling key

- `QuantMode` 增加 `MXFP8_OCP_QUANT`（MXFP4 已占 2 = OCP、3 = CX），host 侧 dtype 校验放开 `DT_FLOAT8_E4M3FN`（q/k/v）与 `DT_FLOAT8_E8M0`（descale）。
- `block_sparse_attention_tilingkey.h` 增加 MXFP8 的取值；`block_sparse_attention.cpp` 与 `block_sparse_attention_kernel_interface.cpp` 增加分发与模板实例化（`BsaInferInterfaceMXFP8FullQuant`）。
- `CalcBaseTileTilingParams950()` 无需改动：D ≤ 128 时 kvBaseTile 仍取 256。
- MXFP8 不走 CX 模式，`log2Cx / log2CxCeil` 保持 OCP 缺省值。

## 9. 文件清单

新增：

- `op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h`
- `op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_mxfp8.hpp`（含沿 k(D) 切 2 段的循环）
- `op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_mxfp8.hpp`（含沿 k(S2) 切 2 段的循环）
- `op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp`
- `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_low_prec_fp16_mxfp8.hpp`
- `op_kernel/attn_infra/epilogue/block/block_epilogue_compute_pscale_arch35_mxfp8.hpp`
- `op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_arch35_reg_high_prec_mxfp8.hpp`
- `op_kernel/attn_infra/epilogue/block/mxfp8_vf/*`：128 列分段的 softmax + e4m3 cast/gather、computeScale、updateScale 等
- `tests/mxfp8_pytest/*`：golden + 用例（照 `tests/mxfp4_pytest` 结构）

修改：

- `arch35/kernel_utils.hpp`：新增 `MXFP8Kernel` 命名空间（`TILE_GROUP_N = 4`）
- `gemm/block/block_mmad_arch35_utils.hpp`：新增 `MXFP8` 命名空间（§4、§5 的表，配 `static_assert` 越界检查）
- `epilogue/block/block_epilogue_arch35_utils.hpp`：新增 `MXFP8` 命名空间（§6 的表）
- `gemm/bsa_gemm_dispatch_policy.hpp`、`gemm/block/bsa_block_mmad.hpp`、`epilogue/bsa_epilogue_dispatch_policy.hpp`、`epilogue/block/bsa_block_epilogue.hpp`：注册 MXFP8 policy
- `block_sparse_attention_tilingkey.h`、`block_sparse_attention.cpp`、`block_sparse_attention_kernel_interface.cpp`
- `op_host/block_sparse_attention_tiling.{h,cpp}`、算子 def 的 dtype 列表
- `tests/ut/op_host/test_block_sparse_attention_tiling.cpp`：新 quantMode 的 tiling 用例

## 10. 风险与未决项

1. **fp8 转置 LoadData 的硬约束**。`matmul.h` 里 fp8/hifloat8/int8 的转置分支会把 `mStep` 向上取偶（`(mStep+1)>>1<<1`），并且在 `ifTranspose && (dstStride & 1)` 时改走「`mStep = M_STEP_ALIGN_BASE` + `l0bLoop = (mStep+1)>>1` 次分段下发」的路径。本设计里 Pᵀ 的 dstStride = 128/16 = 8（偶，安全）；Vᵀ 按 `matmul.h` 的公式本应是 `ceil(D/16) = 8`，但 BSA 的 loader 为了 rowsum 的 16 行 pad 写成 `ceil(D/16) + 1 = 9`（奇）——**这是最可能踩的坑**，第一期就要用最小用例验证。

   兜底方案：把 pad 从 1 个分形改成 2 个（dstStride = 10，偶），代价是 PV 的 m 从 144 padding 到 160——L0A 的 Vᵀ 单槽 18 KB → 20 KB（L0A 合计 56 KB ≤ 64 KB）、L0C 的 PV 槽 72 KB → 80 KB（L0C 合计 208 KB ≤ 256 KB）、seed 镜像 20 KB ≤ 32 KB 不影响 L1。三处都还有余量，所以这条兜底是安全的。
2. **mmad 条数翻倍**。每个基块从 3 条（2 条 QK + 1 条 PV）变成 6 条，每条 k 更短。L0 侧因为切 k 反而留出了余量（L0A 12 KB、L0B 16 KB），但 cube 的指令发射开销与短 k 的流水效率需实测；若 QK 的 k=64 效率不可接受，退路是 K 退回单缓冲、QK 不切 k（L0A = K 1×16 + Vᵀ 2×18 = 52 KB 也放得下）。
3. **同步频率 4 倍**。组首/组尾的跨核 flag 与 fixpipe 从每 16 拍一次变成每 4 拍一次，可能取代 MTE2 成为新瓶颈，需实测确认。
4. **kvs 尾块补零**要按 1 字节/元素重推：fp4 的"两个 plane 各补一次"退化为一次连续 Fill。
5. **P-scale 槽 1280 B 的等价性**已由 `matmul.h` 的公式相等推得，但未在硬件上验证过 fp8 的 P-scale 分形，实现时以最小用例确认。

## 11. 验收标准

- 新 tiling key 分支编译通过；L1/L0/UB 三张表的 `static_assert` 全部成立。
- 与 golden 的精度对比达到与 MXFP4 用例同一口径的阈值。
- 覆盖用例：kvs 满块 256、kvs 尾块（32 对齐 / 64 对齐 / 非对齐）、qs = 128 与 64、kv tile 数小于 8（验证 drain 不死锁）。
- 性能：记录 MXFP8 与 MXFP4 在同 shape 下的耗时比，并标注瓶颈单元（用于判断 §10.3 是否成立）。

## 12. 实施分期

整体改动量偏大（cube 两个 block + 一套 VF + host + 测试），建议分三期，每期都可独立验证：

1. **cube 通路**：常量表（§4/§5）、`copy_l1_to_l0_mx_fp8_a5.hpp`、QK/PV 两个 block（都含切 k）、dispatch policy、tiling key 与分发。这一期先把 pScale 固定成 `0x7F`（=1.0）、绕开 group-max 与跨核交换，只验证 fp8 的搬运与 mmad 数值正确——正好用来提前暴露 §10.1 的 Vᵀ 转置风险。
2. **完整流水**：`TILE_GROUP_N = 4` 的 group-max / pScale / dm 全链路，UB 半基块双缓冲的 VF，延迟 8/3 的主循环与跨核同步。
3. **收尾**：kvs 尾块补零与非对齐分支、精度对比、性能实测与瓶颈标注。
