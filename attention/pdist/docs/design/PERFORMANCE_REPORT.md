# pdist — Stage 5 性能优化报告（PERFORMANCE_REPORT.md）

- **算子**: pdist（pairwise p-范数距离，纯 Vector 单 `@pl.jit` kernel，L0 非融合）
- **平台**: Ascend 950PR（A5 / dav-c310, vector_core_num=56, 实测向量核时钟 ≈1.633 GHz，由 aiv_total_cycles/Task Duration 反推）
- **实现**: `custom/pdist/test_pdist.py`（`pdist_kernel` + `pdist_wrapper`，mtime 2026-09-02 10:21，Stage 4 verifier PASS 15/15、16/16 case；Stage 5 全程未修改实现——所有候选均为设计级，见 §7/§8）
- **报告生成**: 2026-09-02

## 0. 结论速览

| P0 case | baseline (us) | final (us) | PyPTO speedup | Golden E2E (us) | golden_reference_ratio (final) | 默认目标 ≥1.0 |
|---|---|---:|---:|---:|---:|---|
| perf_p0_euclidean [4096,256] fp32 p=2 | 25,658.01 | FINAL_EUCL | SPEEDUP_EUCL | 45,514.20 | RATIO_EUCL | **✓（1.774，baseline/final 一致）** |
| perf_p0_fp16_large [8192,512] fp16 p=2 | 963,022.88 | FINAL_FP16 | SPEEDUP_FP16 | 344,973.57 | RATIO_FP16 | **✗（0.358，目标未达）** |

- **总体判定：`target_met: false`** —— `perf_p0_fp16_large` 的 `golden_reference_ratio = 0.358 < 1.0`（逐 case 判定，未用聚合平均掩盖；聚合比值 0.410 仅为参考）。
- **根因**：fp16 实例的冻结数据流（`rows16 --TTRANS b16--> trans16 --cast--> trans32`）中，`pl.transpose` 在 A5 上 lower 为 **vscatter/vgather2 序列**，其散射目标 tile 行距恰为 **64 元素（2 的幂）**，触发 UB bank 串行化（KB `vec-alignment-and-rotation.md`：power-of-two store stride 串行化，gather 侧同 stride 实测 51.0µs vs 7.0µs）。实测 b16 scatter ≈386 ns/op，是 b32（≈34 ns/op）的 11.3×/op（5.7×/元素），占 fp16 case 总时长 ≈97%。
- **修复路径全部落在 DESIGN.md 冻结契约上**（§1 widen 位置 / §2 Tile 形状 / §3 UB 地址表 / §4-§5 任务结构），Stage 5 无权限实施 → **上报 `design_violation`，建议编排器回退 Stage 3**（详见 §8）。

## 1. 冻结配置（baseline 与 final 完全一致）

| 项 | 值 |
|---|---|
| case manifest | `custom/pdist/PERFORMANCE_CASES.json`（10 case = 2 perf_p0 + 8 func_p0；sha256 `755702fbb0e…418cd0`）|
| 测试入口 | manifest `test_function` → `test_pdist.py` 既有 10 个无参委托函数（`test_perf_p0_*` / `test_func_p0_*`），采集适配器逐 case 调用，单次可唯一归属 target launch |
| 目标 Op Name | `_Z12pdist_kernelPhS_S_`（discovery profile 唯一 launch 确认，AI_CORE，BlockDim=56）|
| 设备 | 物理 npu:0（`TILE_FWK_DEVICE_ID=0`，脚本剥离 `ASCEND_RT_VISIBLE_DEVICES`）|
| seed / warmup / repeats | 42 / 3 / 3（`--seed=42` 与 manifest 协议一致）|
| 计时口径 | msprof `op_summary[PipeUtilization].Task Duration(us)`（canonical；quick 未用于交付）|
| Golden 参考 | `custom/pdist/GOLDEN_PERF_REPORT.{md,json}`（Stage 5 专用采集器，见 §2）|
| 环境 | CANN 9.2.0；`pypto_pro` → `/data/z00954450/zzz/.local/lib/python3.12/site-packages/pypto_pro`（冻结平台记录）；msprof 需 `custom/pdist/.msprof_compat/libsqlite3.so`（裸名符号链接）|

## 2. Golden 参考采集（默认目标，SPEC perf_target=null）

- 命令：`collect_golden_reference.py custom/pdist/pdist_golden.py --function pdist_golden --factory _make_inputs --case-manifest custom/pdist/PERFORMANCE_CASES.json --output-dir custom/pdist --device 0 --warmup 3 --repeats 3 --seed 42`
- 合同：device 0，iterations=1，warmup 3，repeats 3，seed 42，`timing_scope=all_golden_npu_kernels`，聚合 `median_of_repeat_totals`；golden 源 sha256 `c0fb165a39423500…`，manifest sha256 `755702fbb0ead384…`
- 采集诊断记录：前两个会话的失败根因为 torch_npu 内部 `msprof --export=on` 子进程缺 `libsqlite3.so` 裸名时 **rc=0 但导出 0 文件**（stderr ImportError），导致 `kernel_details.csv` 不生成；以 `LD_LIBRARY_PATH` 注入 `.msprof_compat` 后 10/10 case 采集成功。原始证据：`custom/pdist/prof/golden-reference/<case>/collection-*/repeat-00{1,2,3}/`（各含 `kernel_details.csv`）。
- P0 Golden 每迭代 E2E：`perf_p0_euclidean` = 45,514.204 µs（repeats 45527.3/45514.2/45492.5）；`perf_p0_fp16_large` = 344,973.570 µs（344973.6/344946.0/344981.2）。

## 3. 逐 case 结果（两次独立 formal compare）

### 3.1 Baseline（collection `compare_20260902_112959_1583917`，round_001，2026-09-02 11:29–13:23，10/10 valid）

| Case | Shape | dtype | PyPTO kernel (us) | repeats (us) | Golden E2E (us) | ratio | 目标 |
|---|---|---|---:|---|---:|---:|---|
| perf_p0_euclidean | [4096,256] | fp32 | 25,658.01 | 25658.9/25656.8/25658.0 | 45,514.20 | 1.774 | ✓ |
| perf_p0_fp16_large | [8192,512] | fp16 | 963,022.88 | 963014.1/963024.5/963022.9 | 344,973.57 | **0.358** | **✗** |
| func_p0_hamming_p0 | [1000,200] | fp32 | 1,285.23 | 1285.2/1285.2/1285.7 | 3,351.90 | 2.608 | ✓ |
| func_p0_manhattan_p1 | [1000,200] | fp32 | 1,266.91 | 1266.5/1267.0/1266.9 | 3,062.74 | 2.417 | ✓ |
| func_p0_euclidean_p2 | [1000,200] | fp32 | 1,278.52 | 1278.7/1278.0/1278.5 | 3,073.32 | 2.404 | ✓ |
| func_p0_fractional_p05 | [1000,200] | fp32 | 1,270.03 | 1270.0/1269.4/1270.3 | 3,063.93 | 2.412 | ✓ |
| func_p0_chebyshev_pinf | [1000,200] | fp32 | 1,297.73 | 1298.0/1297.6/1297.7 | 2,043.36 | 1.575 | ✓ |
| func_p0_min_shape | [2,1] | fp32 | 4.09 | 3.8/4.1/4.1 | 15.81 | 3.866 | ✓ |
| func_p0_large_m | [64,8192] | fp32 | 608.59 | 608.3/608.6/609.2 | 696.96 | 1.145 | ✓ |
| func_p0_partial_block | [777,300] | fp32 | 1,146.16 | 1146.4/1146.1/1146.2 | 2,840.67 | 2.478 | ✓ |

原始证据：`custom/pdist/docs/perf/round_001/case_<id>_<hash8>/repeat_00{1,2,3}/`（七组 op_summary + op_statistic/task_time + `evidence_status.json` + `summary.txt`；`measurement.json` 每 case 一份）。逐 repeat 方差 <0.1%（大 case <0.005%）。

### 3.2 Final（collection `FINAL_COLLECTION_ID`，round_002，FINAL_TIME）

FINAL_TABLE

（final 与 baseline 为同一实现的两次独立采集：Stage 5 未产生合法的实现级改动——见 §7 候选账本；PyPTO 同口径 speedup = baseline/final ≈ 1.0，属预期。）

## 4. 瓶颈分析（正式 profiling + 生成代码 + KB 规则三方印证）

### 4.1 七指标画像（PipeUtilization，P0 case）

| case | Task Duration (us) | aiv_vec_ratio | aiv_scalar_ratio | aiv_mte2_ratio | aiv_mte3_ratio | vec_resc_cflt | L2 read hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| perf_p0_euclidean | 25,659.4 | 100.0% | 2.1% | 8.9% | 1.9% | 18.9% | 100.0% |
| perf_p0_fp16_large | 963,014.1 | 100.0% | 0.7% | 1.1% | 0.2% | — | 100.0% |

- 两 case 的 **V（向量）流水 100% 占用**、Scalar ≤2.1%、MTE2 ≤8.9%：非 Scalar / 非 GM 搬运 / 非等待瓶颈；`vec+mte2 > 100%` 亦为 MTE2 与 V 重叠的 ratio 线索（终态证明见 §6 timeline）。
- 逐核证据：`per_core_status=unavailable:no_positive_core_cycles`（sample-based 无正 cycle 行，不可归属，未冒充逐核结论）。均衡性由结构保证：块域整除均分（euclidean B=133,056=56×2376 精确整除；fp16 B=528,016 → 9429/9428 块/核，偏差 ≤1 块 ≈0.011%）+ `aiv_time ≈ Task Duration`。

### 4.2 生成代码核对（build/pdist__pdist_kernel__a5/dt_*/tk_2/kernel.cpp）

- m 循环（fp32/fp16 同构）：每 m 5 条向量指令（`vlds BRC_B32` 广播 + `vlds NORM` + `vabsdif` + `vmul` + `vadd`），与 DESIGN §1 冻结序列一致，无冗余；`preg` 每 VF 调用一次 `pset_b32(PAT_ALL)`（mask 已提升，tile 宽 64=满 lane）。
- 每任务块 fp32 额外：`TTRANS(trans, rows)`（×n_mblk）+ finalize（`vsqrt`+`update_mask`+`vsts`）+ MTE3 store。
- fp16 额外：`TTRANS b16`（×n_mblk）+ `TCVT trans32←trans16`（[128,64]）+ `TCVT anchor32←anchor16`（[1,128]）+ 出口 `TCVT out16←out32`。

### 4.3 转置（TTRANS）lowering 与代价模型（A5）

`pto/npu/a5/TTrans.hpp`：`pl.transpose` lower 为 gather/scatter 序列；src `[64,128]`（Rows<Cols）→ **RowWise scatter 变体**：
- b32：每 tile 64 行 × 2 chunk = 128×(`vlds`+`vci`+`vmins`+`vmuls`+`vadds`+`vscatter`)，128 次 scatter（每 op 64 元素）
- b16：每 tile 64 行 × 1 chunk = 64×(同上)，64 次 scatter（每 op 128 元素，uint16 索引寄存器）

**实测微分计量**（由 formal compare 时长 + 指令模型反推，多 case 交叉校准）：

| 量 | fp32 (euclidean) | fp16 (fp16_large) |
|---|---:|---:|
| 每核 tile 数（转置次数） | 4,752 | 37,716 |
| 每核 scatter op 数 | 608,256 | 2,413,824 |
| 实测 scatter 单 op 代价 | ≈34 ns | ≈386 ns |
| scatter 占总时长 | ≈20.7 ms / 25.66 ms ≈ 81% | ≈931 ms / 963 ms ≈ 97% |
| 语义计算下限（3 ALU/elem） | 1.10 ms/核 | 8.8 ms/核 |
| 每元素实测总代价 | 11.95 ns | 56.0 ns（4.7×） |

模型交叉验证：[1000,200] 各 func case 实测 1,266–1,298 µs ≈ 模型预测（每核 296 tile × 128 scatter × 34 ns + m-loop ≈ 1.29 ms）✓；`func_p0_large_m`（单核 64 tile）608.6 µs 与单核串行模型同量级 ✓。

**b16/b32 差异的根因——UB bank 串行化**：scatter 目标 tile（trans/trans16 `[128,64]`）行距 = 64 元素：
- fp32：64×4B = 256B = 8×32B（偶块）
- fp16：64×2B = 128B = 4×32B（偶块）

KB `constraints/vec-alignment-and-rotation.md`「Power-of-two *store* strides serialise on UB banks too — pad to an odd block count」：行距为 2 的幂（32/64/128 元素）的 scatter/strided UB store 会按 bank 串行化；同页引用 gather 侧实测 **stride 64 → 51.0 µs vs stride 65 → 7.0 µs（7.3×）**，store 侧规则相同但未单独实测（本算子即该 store 侧的首个实测佐证：b16 每元素代价为 b32 的 5.7×，与 bank 深度串行一致）。fp32（256B 偶块）受同一机制影响但惩罚较轻（34 ns vs 名义 ~18-20 ns ≈ 1.7-1.9×）。

### 4.4 DESIGN 成本假设勘误

DESIGN §1「关键结构决策」假设「转置开销为每 [64, M_BLK] tile 一次 V 流水转置（**~1 条/m 等效**，被上述节省抵消）」。实测：每 [64,128] tile 转置 ≈768 条向量指令 + 128/64 次 scatter（≈2.4 µs/tile，fp32）≈ **19 条/m 等效 + scatter 串行化惩罚**，比设计假设高 ~20×（fp16 更高 ~90×）。pair-lane 布局相对备选（64 acc 寄存器超限）的**定性结论仍成立**，但其代价模型需在 Stage 3 重设计时修正。

## 5. Roofline 终态（逐 P0 case）

### perf_p0_euclidean [4096,256] fp32 p=2

- **语义必需计算**：每 (pair, m) 3 条向量 ALU（abs_sub+mul+add）→ L×M = 2.147G elem → 100.66M 条 64-lane 指令 → 1.80M cycles/核 → **1.10 ms 下限**（@1.633 GHz）。
- **分层必要搬运（按冻结契约的 tile 结构）**：
  - GM→UB（MTE2）：B=133,056 块 × 65 行 × 1KB = 8.85 GB；实测 MTE2 busy 2,274 µs（8.9%），L2 read hit 100%（输入 4MB 全程 L2 驻留）→ 非绑定层。
  - UB 跨 lane（转置）：8.85 GB 元素各经 scatter 写一次（+gather 型索引读）→ 608K scatter/核 × ≈34 ns ≈ **20.7 ms ≈ 81% 实测时长** ← 绑定层。
- **`roofline_terminal` = `data_movement_bound`**（绑定层 = 冻结 pair-lane 转置所需的 UB 跨 lane 搬运；语义计算下限 1.10 ms 与 GM 搬运均低于实测 ≥10×）。附注：该搬运量为**设计契约量**——语义上仅 4MB 输入/33.5MB 输出为不可约减，跨 lane 搬运来自 DESIGN §1 冻结布局（结构级削减路径见 §8）。

### perf_p0_fp16_large [8192,512] fp16 p=2

- 语义必需计算：3 ALU/elem × 17.18G elem → 805M 指令 → **8.8 ms/核下限**。
- GM→UB：528,016 块 × 4 mblk × 16.4KB ≈ 34.3 GB；MTE2 busy 10.2 ms（1.1%）→ 非绑定。
- UB 跨 lane：2.41M b16 scatter/核 × ≈386 ns ≈ **931 ms ≈ 97% 实测** ← 绑定层（b16 @128B 偶块行距，bank 串行化最深）。
- **`roofline_terminal` = `data_movement_bound`**（UB 跨 lane 层；且因目标未达，终态仅作瓶颈归因记录，不作为完成依据）。

## 6. Tile DAG / 流水 / Scalar 证据（逐 P0 case）

### 6.1 稳态 Tile DAG（fp32，n_mblk=2；fp16 在 V 段插入 TCVT）

```
任务块 t（64 pair, anchor i, j0..j0+63）:
  mblk=0: MTE2: load rows[s0]←GM(j0,m0) ┐ 双缓冲（depth=2 TileGroup）
          MTE2: load anchor[a0]←GM(i,m0)┘ slot: s0/a0 (mutex 0..5)
          V: TTRANS trans[s0] ◄ rows[s0]        (V 流水, gather/scatter)
          V: accum_init: m-loop ×128 → acc[1]   (5 向量指令/m)
  mblk=1: MTE2: load rows[s1],anchor[a1]  ←—— 与 mblk=0 的 V 段重叠（不同 slot）
          V: TTRANS trans[s1] ◄ rows[s1]
          V: accum_acc: acc[0] ◄ acc[1] (ping-pong, mem_bar VST_VLD/VLD_VST)
  尾:     V: finalize sqrt(acc[0]) → out[o0] (update_mask(VP) 尾裁剪)
          MTE3: store y[base] ← out[o0]（覆盖写，块唯一写入者）
  任务块 t+1 的 MTE2 load 与 t 的 V 计算重叠（auto_mutex 深度 2 槽轮转）
```

### 6.2 流水证据（timeline）

PIPELINE_SECTION

### 6.3 Scalar 排除

- ratio：euclidean 2.1% / fp16 0.7%（正式 compare，PipeUtilization）。
- 生成代码热区为纯向量 m 循环；标量侧仅循环计数/地址推进与每核一次的 17 次二分定位（≤ 数百 cycle，占 <0.01%）。
- 无 get/set-value 标量往返、无 per-element 标量分支、无同步等待主导（vec=100% 连续占用本身排除空转等待）。
- **`scalar_evidence.dominant = false`**（两 P0 case）。

## 7. 候选覆盖账本

| # | 候选（原子思路） | 类别 | 判定 | 依据/预期 |
|---|---|---|---|---|
| C1 | trans tile 行距 padding 至 32B 奇数倍（fp16 `[128,80]`、fp32 `[128,72]`，valid 窗口排除 pad） | **设计级（未实施）** | design_violation 上报 | KB pitch 规则 + 本算子实测微分；预期 scatter 惩罚消除（fp16 →~200 ms 内，ratio≈1.7+；fp32 亦受益 ~1.5-2×）。违反 §2 Tile 表/§3 地址表/§1 `m*64` 寻址 |
| C2 | fp16 cast-before-transpose（rows16→f32 widen→TTRANS b32） | **设计级（未实施）** | design_violation 上报 | 转置通道 b32 化（~2×）；违反 §1 冻结 widen 位置（转置后）/§2 fp16 tile 计划。与 C1 正交可组合 |
| C3 | 窗口主序转置摊销（每转置 tile 服务多 anchor，rows 转置次数 ↓~2000×） | **设计级（未实施）** | design_violation 上报 | 预期 3-8×（euclidean →~3-4 ms）；违反 §4 任务块结构/§5 块域均分/§2 槽位生命周期 |
| C4 | 转置视图 strided/DN GM→Vec load（消除 TTRANS） | **设计级（未实施）** | design_violation 上报 | 违反 §1 冻结 load(ND)→transpose 序列；且 MTE2 2D DMA 内层须连续，pair 轴（stride M）不可作 burst 内层——表达性存疑，需框架能力核验 |
| C5 | acc 双累加器破 vadd 依赖链 | 实现级 | 拒绝 | 违反 §1 冻结逐 m `vf.add` 链 + 改变 FP 求和次序；且链延迟(512 cyc/mblk) < issue(640 cyc/mblk)，非瓶颈 |
| C6 | `vf.fma` 替代 mul+add | 实现级 | 拒绝 | 违反 §1 冻结步骤表（mul→add 两步）；收益 ~1.5%（m-loop 仅占 ~10%/2%） |
| C7 | mask 提升出寄存器循环 | 实现级 | 不适用 | `preg` 已每 VF 调用一次 PAT_ALL（SKILL 高效写法 §1 触发条件已满足的形态） |
| C8 | MTE2/GM 侧调优（job 数、DMA 形态、双缓冲深度） | 实现级 | 不适用 | mte2 8.9%/1.1%、L2 read hit 100%——非绑定资源；深度 2 已承载重叠 |
| C9 | bank 冲突侧非 padding 缓解（槽地址错位） | 实现级 | 拒绝 | 冲突由 lane 间 stride（64 元素）决定，与基址无关；无 DSL 级旋钮 |
| C10 | M_BLK / TP 变更 | 设计常量 | 拒绝 | §2.1 冻结关键常量（M_BLK=128/TP=64）；且元素量不变时转置总代价不变 |
| C11 | block_dim / 分核策略调整 | 实现级 | 不适用 | BlockDim=56（全部向量核），整除均分偏差 ≤1 块；无尾核空转 |

- 实施纪律：C5–C11 均有硬依据（冻结步骤/非绑定资源/结构无效）；C1–C4 为**全部有实质收益的候选**，无一可在 Stage 5 权限内合法实施 → 按规则停止并上报 `design_violation`（不自行改 DESIGN、不静默偏离）。
- 重开规则：C1/C2 任一被 Stage 3 采纳后，C8（重叠余量）与 C5（依赖链占比）需重新评估；C3 采纳后 C1/C2 的收益基数改变，需重开。

## 8. design_violation 上报（路由决策：建议 rollback_to_stage(3)）

**声明**：perf_p0_fp16_large 的默认目标（golden_reference_ratio ≥ 1.0）未达（0.358）。经候选覆盖账本（§7）穷举，所有具备实质收益的修复（C1–C4）均需修改 DESIGN.md 冻结契约：

1. **§1 关键结构决策 / vector_selection**：load(ND)→transpose 序列、fp16 widen 位置（转置后）、逐 m add 链；
2. **§2 Tile 属性表 + §3 片上地址映射表**：trans/trans16 tile 形状（行距 padding）与 UB 布局；
3. **§4 循环结构 / §5 分核策略**：任务块 (i, jseg) 结构（窗口主序替代）。

按 Stage 5 规则（「若优化必须改变设计契约，应停止并报告 design_violation，由编排器回退 Stage 3」），Stage 5 不实施上述改动。**证据链**：§4.3 实测微分（34 ns vs 386 ns/op，多 case 校准）+ §4.2 生成代码 + KB `vec-alignment-and-rotation.md` power-of-two store stride 规则 + §6 timeline 指令归因。

**给 Stage 3 重设计的量化预期**（供 architect 决策，非 Stage 5 结论）：
- C1（pitch padding）单独：fp16 scatter 惩罚按 gather 侧同stride 比例外推（5-7×）→ fp16 ≈200-260 ms（ratio ≈1.3-1.7）；fp32 ≈13-15 ms（ratio ≈3.1-3.5）
- C1+C2 组合：fp16 转置通道 b32 化且消除偶块行距 → fp16 ≈100-150 ms（ratio ≈2.3-3.4）
- C3（窗口主序）叠加：euclidean ≈3-4 ms（ratio ≈11-15）；fp16 同结构收益
- 数值边界：padding 不改语义（valid 窗口排除 pad 列）；cast 前移需重验 precision.md 的 widen 顺序要求（「widen before the operation that overflows」——cast 前移后 abs_sub 前已 widen，仍满足）；C3 需重推 §5 均分与 §7 尾块

## 9. 正确性证据

- Stage 4 全量正确性（Stage 5 起点，JIT 预热同 run）：`python custom/pdist/test_pdist.py` → rc=0，**16/16 PASS**（含 2 P0 perf + 4 基础 + 2 语义 + NaN/inf/dup-rows 边界），日志 `/tmp/zzz_corr_run1.log`（关键行 `All tests PASS!`）。
- Final 实现正确性复跑（final compare 后）：FINAL_CORRECTNESS
- 实现文件全程未修改（test_pdist.py mtime 10:21 恒定；Stage 5 无实现级候选被接受）。

## 10. 证据路径索引

| 证据 | 路径 |
|---|---|
| Golden 合同（机器） | `custom/pdist/GOLDEN_PERF_REPORT.json` / `.md` |
| Golden 原始采集 | `custom/pdist/prof/golden-reference/<case>/collection-*/repeat-*/` |
| Baseline collection | `compare_20260902_112959_1583917`，round_001：`custom/pdist/docs/perf/round_001/`（collection.json + 10×case_*/repeat_*/，七指标+evidence_status+measurement.json） |
| Final collection | FINAL_EVIDENCE_PATH |
| Discovery profile | `custom/pdist/.msprof_discovery/PROF_GROUP_20260902_110451/`（Op Name 唯一性确认） |
| Timeline（final，逐 P0） | TIMELINE_PATHS |
| 生成代码 | `custom/pdist/build/pdist__pdist_kernel__a5/dt_2a0f34f7f9c7b951(32)/dt_6530cb3d03dcfca8(16)/tk_2/kernel.cpp` |
| compare 三件套（final） | `custom/pdist/performance.json` / `performance.log` / `perf_report.md` |
| 基线三件套副本 | `/tmp/zzz_baseline_performance.json` / `.log` / `zzz_baseline_perf_report.md`（round_001 数据另完整保留于 docs/perf/round_001/） |
| 进度账本 | `custom/pdist/prof/STAGE5_PROGRESS.md` |

## 11. 剩余限制

- fp16 目标未达是本报告的核心未闭合项；其修复需 Stage 3 重设计（§8）。
- `per_core_status=unavailable`（sample-based 无正 cycle 行）——逐核均衡以结构论证替代，未冒充逐核实测。
- timeline 的 pipeline 证据限于受监控 lane（见 §6.2），不外推全部核；instruction profiling 扰动时延未用于任何 compare 数值。
