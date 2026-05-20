# chunk_gated_delta_rule_C128 性能基线记录

## 记录规则

- 每完成一次可对比的 kernel 修改并跑完同一组 shape，就新增一个 Step。
- 耗时使用 `performance_compare.csv` 里的 measurement 行 `task_duration_us`。
- 比值定义：`ratio_vs_prev = current_task_duration_us / previous_task_duration_us`。
- `ratio_vs_prev < 1` 表示变快，`ratio_vs_prev > 1` 表示变慢。
- 当前精度未过的结果必须明确标注，不能当作最终有效性能。

## Step 001 - Stage3 flag 修复后首版 C128 baseline

- 日期：2026-05-14
- 状态：kernel 可 launch 并完整 synchronize；`out` 精度未过，`state` 接近 CPU golden。
- 包 SHA256：`6569b87621fb6e3f12ab691c208b511fd08ee308a32692184747a055cad71d2b`
- 隔离安装路径：`/tmp/cgdr_c128_pkg_test_stage3flag_20260514_1855`
- 性能数据来源：
  `tests/pytest/prof_out/c128_stage3flag_unverified_refshapes_20260514_1924/performance_compare.csv`
- shape 参考：
  `chunk_gated_delta_rule_neumann/tests/pytest/prof_out/packed64_phaseA122_aiv_gamma_half_broadcast_20260514/performance_compare.csv`
- 输入模式：只保留原始默认 `g=random`，没有额外 `g=None` case。

### 本步修改

- 修复 Stage3 AIV/AIC 跨核 flag 配对：两个 AIV 子块各写自己的 64 行，但共同 set `0x4`。
- AIC 只等待 `0x4`，不再等待不存在的第二个 flag。
- 恢复 `C128_DEBUG_RETURN_POINT=0`，使用完整 kernel 路径采集性能。

### 耗时与比值

首个 baseline 没有上一版，因此 `ratio_vs_prev` 记为 `N/A`。

| case | B | S | D | Hq | Hkv | task_duration_us | ratio_vs_prev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `prefill_b8_s64_d64_h16_hkv4` | 8 | 64 | 64 | 16 | 4 | 816.30 | N/A |
| `prefill_b8_s512_d64_h16_hkv4` | 8 | 512 | 64 | 16 | 4 | 1487.70 | N/A |
| `prefill_b8_s2048_d128_h16_hkv4` | 8 | 2048 | 128 | 16 | 4 | 4569.46 | N/A |
| `prefill_b8_s10240_d128_h16_hkv4` | 8 | 10240 | 128 | 16 | 4 | 21567.14 | N/A |

## Step 002 - BF16 L1 resident three-way pass 8-shape performance

- 日期：2026-05-16
- 状态：三方精度 gate 已通过，阈值 `atol=rtol=0.005`，外部输入/输出为 BF16，
  `g` 为 FP32，`actual_seq_lengths` 为 INT32。
- 当前 Stage1 SHA256：
  `765f94d287ed20d0e44f9a7fa23c130afb4fda7dd5b3294d451cf06119a41262`
- 精度通过 baseline：
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_bf16_l1resident_threeway_pass_20260516.h`
- 隔离安装路径：`/tmp/cgdr_c128_pkg_threeway_20260516_1421`
- 性能数据来源：
  `tests/pytest/prof_out/c128_bf16_l1resident_8shapes_20260516/performance_compare.csv`
- 参考 shape 来源：
  `chunk_gated_delta_rule_neumann/tests/pytest/prof_out/packed64_phaseA132_stage3_dual_aiv_20260515_8shapes_rerun1/performance_compare.csv`
- 输入模式：`g=random`。
- 采集命令：
  `python3 tests/pytest/run_chunk_gdr_C128_performance_msprof.py --output tests/pytest/prof_out/c128_bf16_l1resident_8shapes_20260516 --ops both`

### 本步修改

- 无 kernel 修改。
- 将 C128 性能 paramset 扩展为与 Neumann 参考一致的 8 个 shape：
  `S=64/65/512/513/2048/2049/10240/10241`。
- 同一轮采集 `normal` 与 `c128`，并与 Neumann 参考 CSV 做离线对比。

### C128 对比 normal 与 Neumann 参考

C128/normal ratios `S=64/65/512/513/2048/2049/10240/10241 = 1.211 / 1.042 / 0.886 / 1.085 / 0.899 / 1.099 / 0.898 / 0.899`

Neumann/normal ratios `S=64/65/512/513/2048/2049/10240/10241 = 0.811 / 0.812 / 0.793 / 0.793 / 0.836 / 0.832 / 0.840 / 0.845`

| case | normal_us | C128_us | C128/normal | Neumann_ref_us | C128/Neumann | fastest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `B8 S64 D64 Hq16 Hkv4` | 670.36 | 812.04 | 1.211 | 583.48 | 1.392 | Neumann_ref |
| `B8 S65 D64 Hq16 Hkv4` | 808.60 | 842.58 | 1.042 | 654.40 | 1.288 | Neumann_ref |
| `B8 S512 D64 Hq16 Hkv4` | 1661.66 | 1472.28 | 0.886 | 1348.46 | 1.092 | Neumann_ref |
| `B8 S513 D64 Hq16 Hkv4` | 1790.14 | 1942.50 | 1.085 | 1408.02 | 1.380 | Neumann_ref |
| `B8 S2048 D128 Hq16 Hkv4` | 5091.76 | 4577.88 | 0.899 | 4204.24 | 1.089 | Neumann_ref |
| `B8 S2049 D128 Hq16 Hkv4` | 5164.80 | 5677.64 | 1.099 | 4300.68 | 1.320 | Neumann_ref |
| `B8 S10240 D128 Hq16 Hkv4` | 24387.66 | 21893.68 | 0.898 | 20416.96 | 1.072 | Neumann_ref |
| `B8 S10241 D128 Hq16 Hkv4` | 24788.52 | 22292.80 | 0.899 | 20677.66 | 1.078 | Neumann_ref |

结论：C128 在整 chunk 长序列 `S=512/2048/10240/10241` 上相对 normal 约快
`10%`，但 `S=64/65/513/2049` 仍慢；与 Neumann 参考相比，当前 C128 8 个 shape
全部落后，差距约 `7.2%` 到 `39.2%`。

### 与 Step 001 的可比 shape

| case | Step001_C128_us | Step002_C128_us | ratio_vs_prev |
| --- | ---: | ---: | ---: |
| `B8 S64 D64 Hq16 Hkv4` | 816.30 | 812.04 | 0.995 |
| `B8 S512 D64 Hq16 Hkv4` | 1487.70 | 1472.28 | 0.990 |
| `B8 S2048 D128 Hq16 Hkv4` | 4569.46 | 4577.88 | 1.002 |
| `B8 S10240 D128 Hq16 Hkv4` | 21567.14 | 21893.68 | 1.015 |

## Step 003 - BF16 RHS prefetch/DB + MMAD scheduling after N5/N6

- 日期：2026-05-16
- 状态：N5/N6 三方精度 gate 已通过，30/30 passed，阈值 `atol=rtol=0.005`。
- 当前 Stage1 SHA256：
  `a744d3c0766bb22955d27ea9c41f39e518792a969d21eca54e1a940c0bfe535f`
- 性能数据来源：
  `tests/pytest/prof_out/c128_bf16_rhs_prefetch_n6_8shapes_20260516/performance_compare.csv`
- Neumann 同轮参考：
  `tests/pytest/prof_out/neumann_ref_repack_8shapes_20260516/performance_compare.csv`
- 输入模式：`g=random`。
- 原始数据 SHA256：
  `684e46b184136f29d4e6396a38ad68a3db1b223e3bd220b42c4860d3f6779a77`

### 本步修改

- `FusedRhsC128AICProcess()` 使用 BF16 resident inverse 和 BF16 RHS。
- RHS tile 采用 BF16 L1 stage ping-pong，循环内预取下一 tile。
- top、bottom、lower-left 三段 RHS MMAD 拆成 issue/load-ready/begin/finish 形式。
- 未删除 `RestoreStage1MatmulLocalBuffers()`；tail 仍保留原 `MatmulImpl` fallback。

### C128 对比 normal、Step002 与 Neumann 参考

C128/normal ratios `S=64/65/512/513/2048/2049/10240/10241 = 1.223 / 1.078 / 0.865 / 1.094 / 0.894 / 1.104 / 0.893 / 0.898`

Neumann/normal ratios `S=64/65/512/513/2048/2049/10240/10241 = 0.908 / 0.847 / 0.831 / 0.809 / 0.836 / 0.844 / 0.836 / 0.842`

| case | normal_us | C128_N6_us | C128/normal | Step002_C128_us | N6/Step002 | Neumann_ref_us | C128/Neumann | fastest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B8 S64 D64 Hq16 Hkv4` | 669.06 | 818.36 | 1.223 | 812.04 | 1.008 | 576.36 | 1.420 | Neumann_ref |
| `B8 S65 D64 Hq16 Hkv4` | 790.76 | 852.06 | 1.078 | 842.58 | 1.011 | 684.88 | 1.244 | Neumann_ref |
| `B8 S512 D64 Hq16 Hkv4` | 1652.42 | 1428.58 | 0.865 | 1472.28 | 0.970 | 1371.20 | 1.042 | Neumann_ref |
| `B8 S513 D64 Hq16 Hkv4` | 1749.38 | 1914.34 | 1.094 | 1942.50 | 0.986 | 1423.56 | 1.345 | Neumann_ref |
| `B8 S2048 D128 Hq16 Hkv4` | 5077.68 | 4537.52 | 0.894 | 4577.88 | 0.991 | 4238.98 | 1.070 | Neumann_ref |
| `B8 S2049 D128 Hq16 Hkv4` | 5167.58 | 5707.38 | 1.104 | 5677.64 | 1.005 | 4378.36 | 1.304 | Neumann_ref |
| `B8 S10240 D128 Hq16 Hkv4` | 24541.20 | 21904.30 | 0.893 | 21893.68 | 1.000 | 20372.02 | 1.075 | Neumann_ref |
| `B8 S10241 D128 Hq16 Hkv4` | 24985.10 | 22427.04 | 0.898 | 22292.80 | 1.006 | 20828.52 | 1.077 | Neumann_ref |

结论：N5/N6 后 C128 在 `S=512/2048/10240/10241` 仍相对 normal 约快
`10%`，`S=512/513` 相对 Step002 有小幅改善；`S=64/65/2049/10241`
没有改善。tail 慢点仍集中在 `S=65/513/2049`，下一步优先做 tail RHS raw-MMAD，
再评估 `RestoreStage1MatmulLocalBuffers()` 是否能缩小使用范围。

## Step 004 - N7 tail RHS raw-MMAD 初版

- 日期：2026-05-17
- 状态：N7 tail raw-MMAD 三方精度 gate 已通过，30/30 passed，阈值 `atol=rtol=0.005`。
- 当前 Stage1 SHA256：
  `6bac805a25f71c1b4d58c4b2ee5f73003801a5b6f53d39c5463388fa8c2a16a1`
- 性能数据来源：
  `tests/pytest/prof_out/c128_tailraw_n7_8shapes_20260517/performance_compare.csv`
- Neumann 参考：
  `tests/pytest/prof_out/neumann_ref_repack_8shapes_20260516/performance_compare.csv`
- 输入模式：`g=random`。
- 原始数据 SHA256：
  `0b3a51140fdaf8ccc1cc4be9ad63ec777f6b9c54bd1b5153da35b9ad3e1f76e5`

### 本步修改

- tail RHS 不再先 `RestoreStage1MatmulLocalBuffers()` 后走 `AICProcess()` fallback。
- `TailRhsC128AICProcess()` 直接从 BF16 `attnWsGm_` 搬 tail inverse 的 top/bottom/lower-left
  到 L1，再复用 BF16 raw-MMAD RHS tile loop。
- full chunk RHS 与 tail RHS 共用 `ComputeRhsPacked64Bf16Tiles()`。
- 仍未删除 `RestoreStage1MatmulLocalBuffers()`；本步只替换 tail RHS 计算路径。

### C128 对比 normal、N6 与 Neumann 参考

C128/normal ratios `S=64/65/512/513/2048/2049/10240/10241 = 1.293 / 1.118 / 0.919 / 1.115 / 0.919 / 1.145 / 0.887 / 0.881`

N7/N6 ratios `S=64/65/512/513/2048/2049/10240/10241 = 1.044 / 1.019 / 1.067 / 1.039 / 1.023 / 1.034 / 0.987 / 0.980`

| case | normal_us | C128_N7_us | C128/normal | C128_N6_us | N7/N6 | Neumann_ref_us | C128/Neumann | fastest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B8 S64 D64 Hq16 Hkv4` | 660.52 | 854.14 | 1.293 | 818.36 | 1.044 | 576.36 | 1.482 | Neumann_ref |
| `B8 S65 D64 Hq16 Hkv4` | 776.82 | 868.10 | 1.118 | 852.06 | 1.019 | 684.88 | 1.268 | Neumann_ref |
| `B8 S512 D64 Hq16 Hkv4` | 1659.56 | 1524.58 | 0.919 | 1428.58 | 1.067 | 1371.20 | 1.112 | Neumann_ref |
| `B8 S513 D64 Hq16 Hkv4` | 1784.92 | 1989.52 | 1.115 | 1914.34 | 1.039 | 1423.56 | 1.398 | Neumann_ref |
| `B8 S2048 D128 Hq16 Hkv4` | 5049.42 | 4641.98 | 0.919 | 4537.52 | 1.023 | 4238.98 | 1.095 | Neumann_ref |
| `B8 S2049 D128 Hq16 Hkv4` | 5156.04 | 5903.98 | 1.145 | 5707.38 | 1.034 | 4378.36 | 1.348 | Neumann_ref |
| `B8 S10240 D128 Hq16 Hkv4` | 24371.28 | 21615.12 | 0.887 | 21904.30 | 0.987 | 20372.02 | 1.061 | Neumann_ref |
| `B8 S10241 D128 Hq16 Hkv4` | 24933.68 | 21972.94 | 0.881 | 22427.04 | 0.980 | 20828.52 | 1.055 | Neumann_ref |

结论：N7 初版精度通过，但性能是 mixed。长序列 `S=10240/10241` 相对 N6 分别快
`1.3% / 2.0%`；其余 6 个 shape 回退 `1.9%` 到 `6.7%`。tail 慢点
`S=65/513/2049` 没有收敛，下一步不继续扩大改动，先缩小 tail raw 的计算范围，
让 `tail>64` 只写实际 bottom 行，再重新过精度 gate。

## Step 005 - N7.1 tail RHS raw-MMAD 16-aligned rows probe

- 日期：2026-05-17
- 状态：N7.1 16 对齐 tail rows 三方精度 gate 已通过，30/30 passed，阈值
  `atol=rtol=0.005`。
- 当前 Stage1 SHA256：
  `8d9a0fe71a28d201392ec3748f47d9b046716767af5cf7288b665b696669564d`
- build/install 包：
  `/tmp/cgdr_c128_pkg_tailrows16_n7_1_20260517_1122`
- package SHA256：
  `cb5bc2b7338f3f3bcaa85bc180193f081e065ee279b462646b7e36b9f848d60d`
- 性能数据来源：
  `tests/pytest/prof_out/c128_tailrows16_n7_1_tail4_20260517/performance_compare.csv`
- 输入模式：`g=random`。
- 原始数据 SHA256：
  `697e3a415cc5fbe3c69d61e3d5a2d493d6a4a05f9fc2ce62a612363134b95c96`

### 本步修改

- `LoadRhs64Bf16OperandsToL0()` / `IssueRhsPacked64Bf16RectToL0Db()` /
  `ComputeRhsPacked64Bf16Tiles()` 增加 tail row 参数。
- tail RHS raw-MMAD 不再固定按 64 行写回所有 tail tile，而是按 16 对齐有效行写：
  `tail<=64` 使用 `AlignUp(tailLen, 16)`，`tail>64` 的 bottom/lower-left 使用
  `AlignUp(tailLen - 64, 16)`。
- 精确 `tailLen` 行版本已验证失败：`varlen_b2_17_65_d64` 不通过。原因边界是
  downstream Stage2/Stage3 仍依赖 16 粒度 padded rows 被确定性产生，因此不能退回
  exact-row 写法。

### tail4 probe 对比 N6 / N7 初版

本步只复测 tail shapes `S=65/513/2049/10241`，不是全 8-shape baseline。

| case | C128_N7.1_us | N7.1/N7 | N7.1/N6 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `B8 S65 D64 Hq16 Hkv4` | 800.98 | 0.923 | 0.940 | 比 N7 和 N6 都快 |
| `B8 S513 D64 Hq16 Hkv4` | 1969.78 | 0.990 | 1.029 | 比 N7 略快，仍慢于 N6 |
| `B8 S2049 D128 Hq16 Hkv4` | 5764.66 | 0.976 | 1.010 | 比 N7 快，仍略慢于 N6 |
| `B8 S10241 D128 Hq16 Hkv4` | 21661.46 | 0.986 | 0.966 | 比 N7 和 N6 都快 |

结论：16 对齐 tail rows 收敛有效，四个 tail shape 相对 N7 初版全部改善；
其中 `S=65/10241` 已超过 N6，`S=513/2049` 还差 `2.9% / 1.0%`。下一步不扩大
到 unrelated 改动，优先评估 `RestoreStage1MatmulLocalBuffers()` 是否能缩小调用范围；
但禁止直接删除全路径 restore，必须以 30-case 三方 gate 为准。

## Step 006 - N7.2 rejected last-restore reinit probe

- 日期：2026-05-17
- 状态：rejected，源码已回退到 N7.1。
- 30-case 三方精度 gate：通过，`30 passed, 2 warnings in 42.79s`。
- probe 包：
  `/tmp/cgdr_c128_pkg_n7_2_last_restore_20260517_1135`
- probe package SHA256：
  `3b7d1d88d2fc06863cd0d8643ee7ce495b567417a596784c5db4a1d4e5dbc719`
- probe 性能数据来源：
  `tests/pytest/prof_out/c128_n7_2_last_restore_tail4_20260517/performance_compare.csv`
- probe CSV SHA256：
  `250b435904e5cc51a6291d95fc7423f1e6f2403859ba31c5a75639ef175cbad3`
- 回退后 active Stage1 SHA256：
  `8d9a0fe71a28d201392ec3748f47d9b046716767af5cf7288b665b696669564d`

### 本步尝试

- 给 `GDRStageOneInitParams` 增加最后一个 chunk group 标记。
- 只在全 kernel 最后一个 Stage1 chunk group 的最后一个 para chunk，把
  `RestoreStage1MatmulLocalBuffers()` 的 `mm.Init()` 跳过；中间 para chunk 仍完整
  restore。
- 该尝试正确性通过，但性能不通过，因此已撤销。

### tail4 probe 对比 N7.1 / N6

| case | C128_N7.2_us | N7.2/N7.1 | N7.2/N6 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `B8 S65 D64 Hq16 Hkv4` | 870.42 | 1.087 | 1.022 | 回退 |
| `B8 S513 D64 Hq16 Hkv4` | 1979.96 | 1.005 | 1.034 | 回退 |
| `B8 S2049 D128 Hq16 Hkv4` | 5826.48 | 1.011 | 1.021 | 回退 |
| `B8 S10241 D128 Hq16 Hkv4` | 21922.28 | 1.012 | 0.977 | 相对 N6 仍快，但慢于 N7.1 |

结论：`RestoreStage1MatmulLocalBuffers()` 中最后一次 `mm.Init()` 即使可省，也没有带来
tail4 收益，反而使四个 tail shape 相对 N7.1 全部回退。后续不要再走
“末轮跳过 `mm.Init()`”方向；若继续清理 restore，只能寻找更具体的 raw/MM buffer
生命周期重排点。

## Step 007 - N7.3 rejected short-tail MatmulImpl fallback hybrid probe

- 日期：2026-05-17
- 状态：rejected，源码已回退到 N7.1。
- 30-case 三方精度 gate：通过，`30 passed, 2 warnings in 42.44s`。
- probe 包：
  `/tmp/cgdr_c128_pkg_n7_3_hybrid_tail_20260517_1151`
- probe package SHA256：
  `f115cf9bf5401ead6bf7748d44e6e0ed470780372e6527da7c70ef2fc473e4b9`
- installed `libcust_opapi.so` SHA256：
  `e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`
- probe 性能数据来源：
  `tests/pytest/prof_out/c128_n7_3_hybrid_tail_tail4_20260517/performance_compare.csv`
- probe CSV SHA256：
  `c929d4fc63d4c22ecf54f469632b98161d863a8674d7e1ec787a8b9b9548861a`
- probe Stage1 SHA256：
  `e5f4ad7c3b05f66859cd41ad1255df124552fc97103c267375772efe201f52c2`
- 回退后 active Stage1 SHA256：
  `8d9a0fe71a28d201392ec3748f47d9b046716767af5cf7288b665b696669564d`

### 本步尝试

- 对 full chunk 后面挂 `tailLen<=16` 的短尾，跳过 tail raw-MMAD，先 restore
  Stage1 matmul local buffers，再走旧 BF16 `AICProcess()` RHS fallback。
- 目标是验证 `S=513/2049` 的单行 tail 是否应该保留旧 `MatmulImpl` tail 路径。
- 该尝试正确性通过，但 tail4 性能不满足保留条件，因此已撤销。

### tail4 probe 对比 N7.1 / N6

| case | C128_N7.3_us | N7.3/N7.1 | N7.3/N6 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `B8 S65 D64 Hq16 Hkv4` | 813.28 | 1.015 | 0.954 | 慢于 N7.1，仍快于 N6 |
| `B8 S513 D64 Hq16 Hkv4` | 2005.48 | 1.018 | 1.048 | 回退，且慢于 N6 |
| `B8 S2049 D128 Hq16 Hkv4` | 5755.48 | 0.998 | 1.008 | 相对 N7.1 微快，但仍慢于 N6 |
| `B8 S10241 D128 Hq16 Hkv4` | 22222.98 | 1.026 | 0.991 | 慢于 N7.1，仍快于 N6 |

结论：短尾 fallback hybrid 没有追回 `S=513/2049`，还损失了 N7.1 在 `S=65/10241`
的收益。后续不要再把 tail raw 和旧 `AICProcess()` fallback 混在同一轮；应继续在
BF16 raw-MMAD tail 路径内部缩小固定开销或调整调度。

## Step 008 - N7.4 rejected tail top K-rows contraction probe

- 日期：2026-05-17
- 状态：rejected，源码已回退到 N7.1。
- 30-case 三方精度 gate：通过，`30 passed, 2 warnings in 42.50s`。
- probe 包：
  `/tmp/cgdr_c128_pkg_n7_4_tail_krows_20260517_1206`
- probe `.run` SHA256：
  `98f2e05888e37db84ab777106f4a1bc897e5ea64d05eb4ee7bce97eb05160b31`
- probe C128 kernel object SHA256：
  `02bcc69630128c02cc71066d9b44e5b0c90285e45b35da0865ea3c908ecdbd7a`
- probe 性能数据来源：
  `tests/pytest/prof_out/c128_n7_4_tail_krows_tail4_20260517/performance_compare.csv`
- probe CSV SHA256：
  `b4209262679345f922f81336e5117f5aef03c6ca107cf3f30b668a494b173714`
- probe Stage1 SHA256：
  `6c1946d279dba8bd6f437338722561309bd91c42301da43bfcf5c5a7a823c1cd`
- 回退后 active Stage1 SHA256：
  `8d9a0fe71a28d201392ec3748f47d9b046716767af5cf7288b665b696669564d`

### 本步尝试

- 在 BF16 raw-MMAD tail RHS helper 内新增 `topKRows`，让 `tail<=64` 时 top RHS 的
  L1->L0 K 维和 MMAD K 维从固定 64 收缩为 `AlignUp(tailLen,16)`。
- full chunk 和 `tail>64` 仍保持 K=64，不改 resident dtype，不引入 half RHS。
- 该尝试正确性通过，但 tail4 性能相对 N7.1 全部回退，因此已撤销。

### tail4 probe 对比 N7.1 / N6

| case | C128_N7.4_us | N7.4/N7.1 | N7.4/N6 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `B8 S65 D64 Hq16 Hkv4` | 867.96 | 1.084 | 1.019 | 明显回退 |
| `B8 S513 D64 Hq16 Hkv4` | 1983.90 | 1.007 | 1.036 | 回退，且慢于 N6 |
| `B8 S2049 D128 Hq16 Hkv4` | 5790.72 | 1.005 | 1.015 | 回退，且慢于 N6 |
| `B8 S10241 D128 Hq16 Hkv4` | 22291.44 | 1.029 | 0.994 | 慢于 N7.1，仍快于 N6 |

结论：收缩 tail top K 维减少了部分 MTE/MMAD 指标，但没有降低端到端 task duration；
`S=65` 还明显退化。后续不要再在 `tail<=64` 上用可变 K 维 MMAD 作为优化方向。

## Step 009 - N7.5 rejected bottom-branch MMAD-before-Fixpipe scheduling probe

- 日期：2026-05-17
- 状态：rejected，源码已回退到 N7.1。
- 30-case 三方精度 gate：通过，`30 passed, 2 warnings in 42.80s`。
- probe 包：
  `/tmp/cgdr_c128_pkg_n7_5_bottom_sched_20260517_1224`
- probe `.run` SHA256：
  `680a2961b41cc980365774772051f8d3d7b34b274a2b2ebbbbfdd876707225e5`
- probe C128 kernel object SHA256：
  `4b2bc6ad440686da940058c3f83011c1204d2c86d2a9fcd2e1fd4e972de0dcb2`
- installed `libcust_opapi.so` SHA256：
  `e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`
- probe 性能数据来源：
  `tests/pytest/prof_out/c128_n7_5_bottom_sched_tail4_20260517/performance_compare.csv`
- probe CSV SHA256：
  `5e790502d76e452ee2f2307086f39f0646c23ca47995318bb23cf48ff0261662`
- probe Stage1 SHA256：
  `e467bafed11ce37827fba3deee7fb606df22a6933829e49415c5dfb3c00ef1a4`
- 回退后 active Stage1 SHA256：
  `8d9a0fe71a28d201392ec3748f47d9b046716767af5cf7288b665b696669564d`

### 本步尝试

- 保持 BF16 resident/BF16 RHS、N7.1 16-aligned tail rows 不变。
- 只调整 `ComputeRhsPacked64Bf16Tiles()` 的 `hasBottom` 分支顺序：先完成 top
  MMAD、bottom MMAD、lower-left accumulate MMAD，再集中做 top/bottom Fixpipe 写回。
- 目标是减少 top writeback 和 bottom accumulate 之间的 M/FIX 切换固定开销。
- 该尝试正确性通过，但 tail4 性能不满足保留条件，因此已撤销。

### tail4 probe 对比 N7.1 / N6

| case | C128_N7.5_us | N7.5/N7.1 | N7.5/N6 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `B8 S65 D64 Hq16 Hkv4` | 842.36 | 1.052 | 0.989 | 慢于 N7.1，仍快于 N6 |
| `B8 S513 D64 Hq16 Hkv4` | 1965.48 | 0.998 | 1.027 | 相对 N7.1 微快，仍慢于 N6 |
| `B8 S2049 D128 Hq16 Hkv4` | 5825.70 | 1.011 | 1.021 | 回退，且慢于 N6 |
| `B8 S10241 D128 Hq16 Hkv4` | 21940.06 | 1.013 | 0.978 | 慢于 N7.1，仍快于 N6 |

结论：bottom 分支先算后写降低了部分 MTE2 指标，但没有降低端到端 task duration；
`aic_fixpipe_time_us` 在 `S=65/2049/10241` 反而增加。后续不要再用“延后 top
Fixpipe、集中写回 top/bottom”的调度作为优化方向。

## Step 010 - N7.6 rejected top-only unused bottom L0 buffer allocation probe

- 日期：2026-05-17
- 状态：rejected，源码已回退到 N7.1。
- 30-case 三方精度 gate：通过，`30 passed, 2 warnings in 43.84s`。
- probe 包：
  `/tmp/cgdr_c128_pkg_n7_6_toponly_l0idx_20260517_1234`
- probe `.run` SHA256：
  `e3b2a27d38893e2a95e1d7ef78ebcb291a3bbd2310b0dc27520e91371b1bee9c`
- probe C128 kernel object SHA256：
  `5a9a26dc38f18bf17d526a6e31c60d138ce0d251ca3a69a9b0b92914fa22fdba`
- probe 性能数据来源：
  `tests/pytest/prof_out/c128_n7_6_toponly_l0idx_tail4_20260517/performance_compare.csv`
- probe CSV SHA256：
  `165fc0b8f158aa339fb3328f19be81b0e85d947608ef95b6092f3442c64f82dc`
- probe Stage1 SHA256：
  `5fb571879fd503efaba5565299a21b6d72968a363d4d5e1481df5d112e507298`
- 回退后 active Stage1 SHA256：
  `8d9a0fe71a28d201392ec3748f47d9b046716767af5cf7288b665b696669564d`

### 本步尝试

- 保持 BF16 resident/BF16 RHS、N7.1 16-aligned tail rows 不变。
- 只在 `hasBottom=false` 的 top-only RHS 路径中避免分配 unused
  `bottomL0BufferIdx`，试图降低 `tailLen<=64` 固定事件/cursor 开销。
- 该尝试正确性通过，但 tail4 性能相对 N7.1 全部回退，因此已撤销。

### tail4 probe 对比 N7.1 / N6

| case | C128_N7.6_us | N7.6/N7.1 | N7.6/N6 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `B8 S65 D64 Hq16 Hkv4` | 855.80 | 1.068 | 1.004 | 回退，且慢于 N6 |
| `B8 S513 D64 Hq16 Hkv4` | 1996.58 | 1.014 | 1.043 | 回退，且慢于 N6 |
| `B8 S2049 D128 Hq16 Hkv4` | 5878.76 | 1.020 | 1.030 | 回退，且慢于 N6 |
| `B8 S10241 D128 Hq16 Hkv4` | 22129.24 | 1.022 | 0.987 | 慢于 N7.1，仍快于 N6 |

结论：跳过 top-only 路径的 unused bottom L0 buffer allocation 并没有降低端到端
耗时。虽然部分 `aic_fixpipe_time_us` / `aic_mte2_time_us` 降低，但 AIV/task
duration 全部回退，说明该 raw L0 DB cursor 顺序变动对流水排布有负面影响。
后续不要再走“top-only 不取 unused bottom L0 idx”的方向。

## Step 011 - N7.7 accepted tail RHS phase split probe

- 日期：2026-05-17
- 状态：accepted checkpoint，源码已前进到 N7.7，并保存 baseline。
- 进入 probe 前 active Stage1 SHA256：
  `8d9a0fe71a28d201392ec3748f47d9b046716767af5cf7288b665b696669564d`
- N7.7 Stage1 SHA256：
  `64f1d42330c2b1437da3bf1907550d01c332d4beb6a3b8edf8cdc21093a56d6c`
- 保存 baseline：
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_tail_phase_n7_7_pass_20260517.h`
- 30-case 三方精度 gate：通过，`30 passed, 2 warnings in 42.73s`。
- probe 包：
  `/tmp/cgdr_c128_pkg_n7_7_tail_phase_20260517_1251`
- probe `.run` SHA256：
  `19c2fbc28976391ebd59203934a12d0ad5b8d967e0ba44c3866832c58b32b535`
- probe C128 kernel object SHA256：
  `6015fb19350e3fa15f35f92746ed8fa9f332449c05cae876a8f4154249944086`
- probe 性能数据来源：
  `tests/pytest/prof_out/c128_n7_7_tail_phase_tail4_20260517/performance_compare.csv`
- probe CSV SHA256：
  `375db270fdef7864e8f8580c9627f23dccb7a120a2dd1d03e8bb7f526f58fb38`

### 本步边界

- 前 3 个 tail raw probe 已说明：单独压缩 AIC MTE/Fixpipe 指标，不足以保证 task
  duration 下降；`N7.1` tail4 的 task duration 仍基本跟 `aiv_time_us` 同步。
- 当前 C128 host tiling 固定 `STAGE_ONE_PARA_NUM=1`，因此本步只验证 tail-only 组
  的 RHS phase split：tail `gBK` RHS 在 `CrossCoreWaitFlag(0x5)` 之前消费，
  tail `vBeta` RHS 仍在 `0x5` 之后消费。
- 仍保持 BF16 resident/BF16 RHS，不改数学、不改 dtype、不混回旧 `AICProcess()`
  fallback。
- 该重排不能直接泛化到多 para：tail helper 复用 resident slot0/1/2 装载 GM tail
  inverse；若以后把 `stageOneParaNum` 放大，必须先给 tail inverse 独立 BF16 scratch，
  否则会覆盖同组 full para 后续 `vBeta` 仍需读取的 resident inverse。

### tail4 probe 对比 N7.1 / N6

| case | C128_N7.7_us | N7.7/N7.1 | N7.7/N6 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `B8 S65 D64 Hq16 Hkv4` | 842.34 | 1.052 | 0.989 | 慢于 N7.1，仍快于 N6 |
| `B8 S513 D64 Hq16 Hkv4` | 1886.56 | 0.958 | 0.985 | 追回 N6，保留 |
| `B8 S2049 D128 Hq16 Hkv4` | 5520.40 | 0.958 | 0.967 | 追回 N6，保留 |
| `B8 S10241 D128 Hq16 Hkv4` | 22077.18 | 1.019 | 0.984 | 慢于 N7.1，仍快于 N6 |

结论：N7.7 解决了 N7.1 仍慢于 N6 的 `S=513/2049`，四个 tail shape 全部快于
N6；但 `S=65/10241` 相对 N7.1 回退。下一步 N7.8 只在当前 chunk group 同时包含
full chunk 和 tail 时启用 split，尝试保留 `S=513/2049` 收益并拿回 pure-tail
group 的回退。

## Step 012 - N7.8 conditional tail RHS phase split probe checkpoint

- 日期：2026-05-17
- 状态：accepted checkpoint，源码已前进到 N7.8，并保存 baseline。
- 进入 probe 前 N7.7 Stage1 SHA256：
  `64f1d42330c2b1437da3bf1907550d01c332d4beb6a3b8edf8cdc21093a56d6c`
- N7.8 active Stage1 SHA256：
  `f3447d43c9a8af403487bea7c799af8cb249d10aa68cd96a3ad0a746429276a3`
- 安装包：
  `/tmp/cgdr_c128_pkg_n7_8_cond_tail_phase_20260517_1315`
- `.run` SHA256：
  `627dfd43f707fcafcbefd6de4dad2f2324f8bad7234575474761169725690444`
- C128 object SHA256：
  `3764a30efd136b512afd59167d5dc45b00f5826ce1e1a0695133616cc7149242`
- `libcust_opapi.so` SHA256：
  `e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`

### 本步边界

- 在 N7.7 基础上把 split 条件收窄为 `curParaNum == 1 && numChunk_ > 1`。
- 预期：`S=513/2049` 的首个 chunk group 同时有 full chunk 和 tail，继续走 N7.7；
  `S=65` 以及 `S=10241` 的最后一个 pure-tail chunk group 回到 N7.1 tail RHS 时序。
- 仍保持 BF16 resident/BF16 RHS，不改数学、不改 dtype、不混回旧 `AICProcess()`
  fallback。
- 精度 gate：`30 passed, 2 warnings in 42.67s`。
- tail4 profiling 输出：
  `tests/pytest/prof_out/c128_n7_8_cond_tail_phase_tail4_20260517/performance_compare.csv`
- profile CSV SHA256：
  `d98da3f18ae95300068a9dad4ae5bfa0114560b703c4abf112b2058046ed3712`
- 保存 baseline：
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_tail_phase_n7_8_pass_20260517.h`
- 当前进入 N7.9 probe：在 N7.8 baseline 上补 `subValidRows==0` 的 AIV 零化 fast path，
  不改 BF16 resident/BF16 RHS / tail raw-MMAD 数学语义。
- N7.9 precision checkpoint：
  Stage1 SHA256 `f549d62490c31015bb8108445ee2eecf8b28142a6ce12c5522033324c081c0cd`；
  `.run` SHA256 `afea5a935d8c43e058c2aa83d31c2390ea38c844c32a5520bc924b0b3d8fc809`；
  C128 object SHA256 `347e5f7acfd6fe8c6b8960928f99842451ecba42f3bac79d7221f5b12c1abadf`。
  30-case 三方精度 gate：`30 passed, 2 warnings in 43.54s`。

### tail4 对比

| case | C128_N7.9_us | N7.9/N7.8 | N7.9/N7.7 | N7.9/N7.1 | N7.9/N6 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| S=65 | 839.40 | 1.0159 | 0.9959 | 1.0159 | 0.9841 | 回退于 N7.8/N7.1，仍快于 N6 |
| S=513 | 1877.34 | 0.9943 | 0.9949 | 0.9906 | 0.9792 | 略快于 N7.8，但不够保留 |
| S=2049 | 5571.14 | 0.9914 | 1.0092 | 0.9630 | 0.9802 | 比 N7.8 快一点，但仍慢于 N7.7 |
| S=10241 | 21968.80 | 0.9880 | 0.9949 | 1.0130 | 0.9827 | 仅这项继续快于 N7.1/N6 |

结论：N7.9 只在 `S=10241` 上有收益，其余 tail shape 没有形成稳定改进，
整体判定为 rejected probe，不更新 baseline。下一步应换到新的候选方向，
而不是继续扩大 `subValidRows==0` fast path。

## Step 013 - 长序列 aligned 8-shape profiling 口径切换

- 日期：2026-05-17
- 状态：profiling 口径已改，C128 已起跑，Neumann C64 也已起跑；本轮只测长序列 8 个 aligned shape：
  `S=1024/2048/4096/8192/10240/12288/16384/20480`。
- 当前 active performance 目标仍以 N7.8 为最优版本参考；Neumann C64 只保留作为对照。
- 本步先不改 kernel，只改性能参数集和记录口径。

### 本步修改

- `tests/pytest/compare_chunk_gdr_C128_performance_paramset.py` 的 8-shape 集合改为长序列 aligned 口径。
- `../chunk_gated_delta_rule_neumann/tests/pytest/compare_chunk_gdr_performance_paramset.py` 的 8-shape 集合也改为长序列 aligned 口径，seed 对齐到 `20260514 + index`。
- 后续只测最优 N7.8 与 Neumann C64，不再跑 normal/C128 全量对比。

### 说明

- 这一版还没有新的测量结果，下面的表和比值仍然是旧口径的历史记录。
- C128 侧 DB / prefetch / inverse 目前的结论仍是：RHS double-buffer / 预取已按 Neumann C64 的思路对齐，
  inverse / compose 仍是 C128 特化的 lower-left 128 split 形态，不是 1:1 复刻。

## Step 014 - 长序列 aligned 8-shape profiling 结果

- 日期：2026-05-17
- 状态：长序列 8-shape profiling 已完成；C128 取 N7.8 baseline，Neumann 取 C64 对照。
- C128 profiling：
  `tests/pytest/prof_out/c128_n7_8_long_20260517/performance_compare.csv`
- Neumann profiling：
  `../chunk_gated_delta_rule_neumann/tests/pytest/prof_out/neumann_c64_long_20260517/performance_compare.csv`
- C128 CSV SHA256：
  `0300b75351938fabb2daad2e63cb48001b5d94ac8cb0b3fecd1884ad192e8cac`
- Neumann CSV SHA256：
  `b785341358e6b4975ae8b7f9d620ba8be673593c41967d7c249d9218421554e6`

### 对比结果

| case | C128_us | C64_us | ratio | aicore_ratio | aic_mac_ratio | aic_mte1_ratio | aic_mte2_ratio | aic_fixpipe_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S=1024` | 2578.64 | 2325.56 | 1.109 | 1.110 | 1.165 | 1.055 | 0.862 | 0.787 |
| `S=2048` | 4599.98 | 4203.48 | 1.094 | 1.095 | 1.165 | 1.065 | 0.861 | 0.799 |
| `S=4096` | 8766.76 | 8320.70 | 1.054 | 1.054 | 1.165 | 1.060 | 0.884 | 0.835 |
| `S=8192` | 17363.90 | 16753.24 | 1.036 | 1.037 | 1.165 | 1.055 | 0.876 | 0.829 |
| `S=10240` | 21647.82 | 20188.14 | 1.072 | 1.072 | 1.165 | 1.064 | 0.906 | 0.856 |
| `S=12288` | 26059.58 | 24686.14 | 1.056 | 1.056 | 1.165 | 1.054 | 0.889 | 0.831 |
| `S=16384` | 34890.80 | 32876.38 | 1.061 | 1.061 | 1.165 | 1.065 | 0.905 | 0.851 |
| `S=20480` | 43171.74 | 41056.06 | 1.052 | 1.052 | 1.165 | 1.063 | 0.878 | 0.834 |

### 结论

C128 在这 8 个长序列上全部慢于 Neumann C64，`task_duration` 平均慢 `6.7%`，
范围 `3.6% ~ 10.9%`。  
`aic_mte2` / `aic_fixpipe` 都没有变差，真正拉高的是 `aic_mac`（稳定 `+16.5%`）
和一部分 `aicore` / `aiv_vec` 时间，说明当前瓶颈更像是 C128 特化的 inverse /
compose 额外计算和 materialize，而不是 stage23 的搬运或同步气泡。

代码侧对应关系：
- `Stage2` 基本已经和 Neumann 对齐，不是主差异来源。
- `Stage3` 的主路径也已对齐到同类的 AIV/AIC 双态同步形态，额外开销不大。
- `DB` / `prefetch` 已按 Neumann 的双缓冲思路做了，但 `inverse / compose`
  仍是 C128 特化的 128 split，`ComposeC128LowerLeftFp16()` 还保留了额外
  raw-MMAD + BF16 resident materialize，因此并不是 1:1 的 C64 复刻。

## Step 015 - full128 lower-left compose probe

- 日期：2026-05-17
- 状态：rejected，未进入性能 profiling；active Stage1 已恢复到 N7.8 baseline。
- 目标：只测试 C128 特有 lower-left compose 是否能改成更接近 `ref.cpp`
  `matrix_size=128` 的 full128 L0 容器；不改 RHS、Stage2/Stage3、tail、restore，也不改
  top/bottom 两个 64 inverse。
- 构建：C128-only 包和 normal+C128 三方包均构建通过。
- 三方包 `.run` SHA256：
  `e6f8d165982f41074ed35694a964489e9ec34fc28df7f47bee1b6d86c5bdb622`
- installed `libcust_opapi.so` SHA256：
  `e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`
- installed C128 object SHA256：
  `42affdab011449908754d4af6c913ea0c5293a3f6ce59e2ab2e8e29b9a443a89`

结果：
- tail-only cases `S=64/80/96/112` 通过三方 gate 前半段。
- 首个 full chunk case `S=128` 挂起，NPU AICore 100%，手动终止。
- plog：
  `/home/developer/ascend/log/run/plog/plog-3703839_20260517204307255.log`
- 关键日志：
  `SynchronizeExecutedTask: report three minutes timeout! stream_id=46, task_id=77, pendingNum=3`

结论：直接把现有 two-64 lower-left compose 塞进 full128 raw-MMAD 容器不可保留。
如果后续继续 ref-style full128 方向，需要做完整 128x128 inverse 容器和事件模型，
而不是只替换 `ComposeC128LowerLeftFp16()` 的两段 MMAD。

## Step 016 - N9.8 full128 L0C-resident long profiling baseline

- 日期：2026-05-18
- 状态：三方 precision gate 已通过，30/30 passed，阈值 `atol=rtol=0.005`。
- 当前 Stage1 SHA256：
  `c81849e091f35002a327aae5efef54b9cebf32e43b52f546d230281f3260af59`
- 构建包 `.run` SHA256：
  `d6ab479428c35e639d8d9ba01e3eadf93655afd0bc77be36f8683e2f28f814df`
- 隔离安装路径：
  `/tmp/cgdr_c128_pkg_revert_dual_n9_10_20260518_1822`
- 性能数据来源：
  `tests/pytest/prof_out/c128_n9_8_full128_long_20260518/performance_compare.csv`
- 输入模式：`g=random`

### 本步修改

- 保持 full128 求逆/compose 在 `cInv=aicL0C_` 中完成。
- 最终 `StorePacked128InverseToFullResident()` 只做一次
  `FP32 L0C -> BF16 L1 resident` materialize。
- RHS 统一消费单一 `aicBf16L1ResidentInv128_`。
- 这里的 `normal` 是同一性能脚本里的原始 `chunk_gated_delta_rule` 基线，不是 C64/Neumann。

### C128/normal 与 N9.8 基线

C128/normal ratios `S=1024/2048/4096/8192/10240/12288/16384/20480 = 1.0238 / 1.0018 / 0.9574 / 0.9650 / 0.9669 / 0.9720 / 0.9710 / 0.9752`

| case | normal_us | C128_us | C128/normal |
| --- | ---: | ---: | ---: |
| `S=1024` | 2839.74 | 2908.30 | 1.0238 |
| `S=2048` | 5055.50 | 5064.50 | 1.0018 |
| `S=4096` | 10067.10 | 9643.84 | 0.9574 |
| `S=8192` | 19997.16 | 19250.62 | 0.9650 |
| `S=10240` | 24391.90 | 23617.72 | 0.9669 |
| `S=12288` | 29469.14 | 28641.48 | 0.9720 |
| `S=16384` | 39465.58 | 38322.70 | 0.9710 |
| `S=20480` | 48504.58 | 47344.16 | 0.9752 |

结论：N9.8 full128 L0C-resident 版在长序列上仍是 mixed，但已经是当前 active
full128 基线。

## Step 017 - N9.10 RHS DB/load-ahead rejected

- 日期：2026-05-18
- 状态：rejected。构建和三方 precision gate 通过，但性能回退，active 源码已回退。
- 当前 Stage1 SHA256：
  `c81849e091f35002a327aae5efef54b9cebf32e43b52f546d230281f3260af59`
- 构建包 `.run` SHA256：
  `d6ab479428c35e639d8d9ba01e3eadf93655afd0bc77be36f8683e2f28f814df`
- 隔离安装路径：
  `/tmp/cgdr_c128_pkg_revert_dual_n9_10_20260518_1822`
- 性能数据来源：
  `tests/pytest/prof_out/c128_n9_10_rhsdb_long_20260518/performance_compare.csv`
- 输入模式：`g=random`

### 本步修改

- 尝试把 full128 RHS 调度对齐成 C64 风格 first-prefetch + L0AB DB。
- 保留 full128 inverse 的 L0C resident 语义不变，只改 RHS prefetch / DB 调度。
- 这里的 `N9.8` / `N9.10` 指的是 C128 源码版本对比；`normal` 仍然是同包的非-C128 baseline，不是 C64。

### 对比 N9.8

N9.10 / N9.8 ratios `S=1024/2048/4096/8192/10240/12288/16384/20480 = 1.0210 / 1.0173 / 1.0288 / 1.0252 / 1.0263 / 1.0184 / 1.0271 / 1.0223`

结论：这版比 N9.8 慢约 `1.8%~2.9%`。根因不是 DB 基础设施缺失，而是当前
`stageOneParaNum=1` 且 `C128_RESIDENT_PARA_SLOTS=1`，full128 RHS 本身只面对单个
`128` 列 tile，没有 C64 那种可吃到收益的跨 para / next tile 重叠空间；再叠加多出的
调度事件和准备步骤，反而变慢。

## Step 018 - N9.13 resident/RHS/restore final pass

- 日期：2026-05-19
- 状态：N9.13.0 到 N9.13.4 全部完成，build / 4-case / ladder / 30-case / profiling 均已完成。
- 当前 Stage1 SHA256：
  `ecec2438b0c69b112199c8e2b748587c7fac681a6cdbe830f097a8c5edeaf5e2`
- final baseline：
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_n9_13_4_final_pass_20260519.h`
- 构建包 `.run` SHA256：
  `5eaab89d74a4278295daccef83a07d6c1c4d52f5b4e17b1bd5a31f918e0e80e2`
- installed `libcust_opapi.so` SHA256：
  `552341a934dde044b5d568d1c951cdc9d3e905872d4adb5bf9d2be8c14bf772f`
- installed C128 object SHA256：
  `975ae44a2cac890ef0920c703e038c8868484f2fc201254e75d1a90833b1f5eb`
- 性能数据来源：
  `tests/pytest/prof_out/neumann_vs_c128_n9_13_final_20260519/c128_vs_neumann_compare.csv`
- C128 raw CSV SHA256：
  `5b766dc822147c2a6965d4c19a8b3b4ba73fdbf9cbf24ed5655d3aba86e04131`
- Neumann raw CSV SHA256：
  `68d2c6eccb6f018d90cf2571c56c81eadd80f5eb40b6239373469a7086c1c65d`
- 对比 CSV SHA256：
  `5ea7603e7136e22f01d2663d7b3ed1b2561bb7cd3463e03be40fa02affdde057`

### 本步修改

- full128 L1 footprint 固化为 7 个 tile：
  `invInit / diagWork / powerWork / identity / zero / packedInv / residentInv`。
- inverse epoch 完成后复用 `invInit` / `powerWork` 作为 GBK / VBeta 首块 RHS prefetch buffer。
- `ComputeRhsPacked128Bf16Tiles()` 保持 C64 式 first issue + wait current + prepare next tile / next para 的 DB 顺序。
- `ReleaseRawL0AbDbEvents()` 统一 raw event wait 与 cursor/state reset。
- `RestoreStage1MatmulLocalBuffers()` 保留 `pipe_->Reset()` 和 `mm.Init()`，并在 reset 后显式清
  `aicLocalBuffersReady_` 与 packed128 aux state，避免 reset 后复用旧 local tensor 句柄。

### 精度 gate

- 4 个 known regression case：
  `b4_s256_d64_h16_h4 / b2_s160_d64_h32_h8 / b1_s384_d128_h8_h4 / b1_s16384_d128_h4_h4` 全部 PASS。
- full / tail ladder：
  `S=64/80/96/112/128/160/256/1024` 全部 PASS。
- 30-case CPU golden：
  `30 passed, 2 warnings in 7.67s`。

### C128 vs Neumann 长序列 8-shape

| case | C128_us | Neumann_us | C128/Neumann |
| --- | ---: | ---: | ---: |
| `S=1024` | 2536.36 | 2391.70 | 1.0605 |
| `S=2048` | 4330.28 | 4282.28 | 1.0112 |
| `S=4096` | 8175.22 | 8612.04 | 0.9493 |
| `S=8192` | 16398.84 | 16793.96 | 0.9765 |
| `S=10240` | 20372.36 | 20447.50 | 0.9963 |
| `S=12288` | 24133.48 | 24590.90 | 0.9814 |
| `S=16384` | 32333.44 | 32932.06 | 0.9818 |
| `S=20480` | 40235.24 | 41141.30 | 0.9780 |

结论：N9.13 final 版在 8 个长序列上的平均 `C128/Neumann = 0.9919`。
`S=1024/2048` 仍慢，`S>=4096` 整体快于 Neumann。后续若继续优化，应聚焦
`S=1024/2048` 的 inverse / compose MAC 和 AIV vec 开销；不要再走直接删除
`ResetRawL0AbDbState()` / `pipe_->Reset()` / `mm.Init()` 的方向。

## Step 019 - N9.13 cleanup validation

- 日期：2026-05-20
- 状态：清理 active Stage1 的 dead helper / dead buffer，并删除不再保留的早期 baseline；保留
  `N9.8` 冻结基线与 `N9.13.4` final baseline。
- 当前 Stage1 SHA256：
  `ffa00a838eef0958dca49d5ca9ae1648ef2c1b04cde745dec11ea2965ce0b2f8`
- 构建包 `.run` SHA256：
  `0de1bc3c3f3fcb53984871f7c843c8cd1b8e8c6b2b6be83679c4ca6faa0f46ef`
- installed C128 object SHA256：
  `b5e5d86694885a363742da36f738c6c600e26ec30585d00cec78c29e228b38e8`

### 本步修改

- 删除 legacy AIV inverse helper、half output helper 与旧 grouped accumulation 数据流。
- 收缩 active Stage1 的无用 UB/L1/L0 buffer 与 queue 成员。
- 删除 2026-05-16/17 的早期 baseline 快照；`N9.8` 作为冻结基线保留。

### 精度 gate

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128/tests/pytest
python3 -m pytest -rA -s compare_chunk_gdr_C128_cpu_golden.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

结果：`30 passed in 9.41s`。
