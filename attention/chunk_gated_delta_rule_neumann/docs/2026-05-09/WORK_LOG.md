# 2026-05-09 工作日志

## Fused cube inverse 当前状态复核

- 当前 active selector 仍为 `fulllocalnew/C64_1616_FP16/chunkSize=64`。
- 当前 active fast inverse 配置：
  - `FAST_STRICTRI_INVERSE_C64 = true`。
  - `FAST_STRICTRI_ACTIVE_DEBUG_MODE = FAST_STRICTRI_INVERSE_MODE_FULL`。
  - `S=64` full chunk 会进入 `FastStrictTriInverse64::Compute()`，不是旧 P3/P8 fallback。
- 当前 target 文件：
  - `op_kernel/fulllocalnew/fast_strictri_inverse.h`。
  - `op_kernel/fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`。

## 当前验证结果

- build 成功：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- install 成功：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- `S=64` smoke 不再 timeout，但仍 FAIL：
  - `CPU vs ChunkGDRNeumann out max_abs=3.430384e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.301773e-01`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=3.417969e-01`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=1.298828e-01`
- 失败形态稳定：state 最坏点 actual 接近 0，更像块内容丢失、错位或后续消费到空块，不像单纯 FP16 累积误差。

## 纯对照线验证

- 已将 selector 切到 `p8a_baseline` 对照线：
  - `op_kernel/selected/stage1_selected.h`
  - `op_kernel/selected/stage1_selected_config.h`
- 这条线保持：
  - P3：单个 `16x16` Neumann 逆写回 GM。
  - P8：原始 GM compose 路径不变。
- build/install/S=64 smoke 结果：
  - build PASS。
  - install PASS。
  - `S=64` smoke PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
- 结论：
  - P3 单块求逆本身可作为安全对照。
  - 当前 fused/local 失败不是 P3 单块求逆误差主导。

## 已尝试的最小改动

- 对齐过 `op_kernel/fulllocalnew/fast_strictri_inverse.h::CopyPackedL1BlockToL0()` 的 L0 destination offset：
  - 从 `CompactMatrixNzOffset(row, col, blockSize)` 改为 compact L0 fractal offset：
    `(rowBlock * blockCount + colBlock) * FAST_STRICTRI_RAW_HALF_FRACTAL_ELEMS`。
- 该改动 build/install 成功。
- `S=64` smoke 指标完全不变，说明当前主因不在这个单点，或该点不是最终敏感消费路径。

## 调试边界修正

- 禁止使用 `dumpdebug`。
- 禁止把中间结果塞进 `out` 或 `attnWsGm_` 再从 Python 读取。
- 原因：
  - 会改变 AIC/AIV 共享 workspace 的正常消费路径。
  - 会引入跨核时序、cache clean/flush、不同 core 可见性差异。
  - 对当前 L1/L0/Fixpipe packed layout 问题没有直接可信性。
- 后续只允许使用最终正常输出形态做边界判断，不使用 dump 数据作为真值。

## 当前结论

- 当前已经是完整 fused cube inverse 目标路径在运行。
- 卡点已经不是“是否卡死”，而是 FULL 路径能返回但精度失败。
- `S=64` smoke 是当前验收边界；build 成功不能代表 fused inverse 正确。
- 后续必须继续在 target path 内定位：
  - packed L1 -> L0 source/destination block placement。
  - `ComposeTwo32From16()` 第二次 MMAD 后的两个 16x16 lower block 落点。
  - `Compose64From32()` 的 `lower @ topInv` 与 `bottomInv @ tmp` 的 source/destination block 落点。
  - `Fixpipe CFG_NZ` 写回 packed L1 的 `srcStride/src offset/dstStride`。

## 2026-05-09 递进诊断分支

- 新的递进分支已切到 `p3l1_p8a`：
  - selector: `op_kernel/selected/stage1_selected.h`
  - config: `op_kernel/selected/stage1_selected_config.h`
  - active Stage1: `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- 当前变更：
  - P3 `16x16` 对角逆结果先落到独立 packed L1 tile。
  - P8 的对角 operand 从这份 L1 tile 取。
  - 下游最终仍通过原 GM 结果链消费，保持 AIV 输出路径不变。
- build/install PASS。
- `S=64` smoke 仍 FAIL，数值形态与旧版一致：
  - `CPU vs ChunkGDRNeumann out max_abs=3.430384e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.301773e-01`
  - state worst actual 仍接近 `0`
- 结论：
  - 这条桥只是把 P3/P8 的 residency 问题单独拆出来。
  - 下一步按 `aa.txt` 的递进顺序继续往 32x32 L1-resident compose 推进。

## p3l1_p8a packed 写回边界复核

- 在 `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h` 上继续做单点隔离。
- 先尝试将 compose 结果通过 compact L1 中转写回，仍然 FAIL，且数值完全不变：
  - `CPU vs ChunkGDRNeumann out max_abs=3.430384e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.301773e-01`
- 然后将 P8 compose 回退到已验证 GM direct，只保留 P3 结果先落 packed L1，再从 packed L1 写回 GM diagonal，仍然 FAIL，且仍是同一组失败签名。
- 最后改成：
  - P3 求逆结果仍复制一份到 `aicHalfL1P3Result_`。
  - GM diagonal 写回恢复为 baseline 同款 compact tile 直接写回。
  - P8 compose 使用已验证 GM direct helper。
- build/install/S=64 smoke PASS：
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
  - `ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`
  - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
- 结论：
  - 旧失败签名已经收敛到 packed L1 -> GM diagonal 回读/写回链路。
  - P3 单块求逆、P3 compact GM 写回、P8 GM direct compose、final RHS 不是当前主因。
  - 当前 active 是通过的诊断桥，不代表 step 3 的 L1-resident compose 已完成。

## 18:41 packed L1 -> GM diagonal 读回修复

- 修改点仍限制在 `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`：
  - `CopyHalfL1CompactBlock()` 改为通过 `RawMmadPackedHalfL1AccumulateTile(src, identity)` + `Fixpipe CFG_NZ` 写入目标 packed/compact L1。
  - `CopyHalfL1PackedBlockToCompact()` 同样改为 packed source 经 identity MMAD 写入 compact scratch。
  - `C64InverseBy16x16Neumann()` 在每个 P3 diagonal inverse 后，启用 `aicHalfL1P3Result_` -> GM diagonal 写回。
  - P8 compose 仍保持 `P8Compose32From16SingleGmDirect()` / `P8Compose64From32GmDirect32()`，final RHS 未改。
- build/install PASS。
- smoke 结果：
  - `S=64` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
  - `S=256` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=1.172006e-03`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
  - `S=1024` PASS:
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=1.137391e-03`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=9.765625e-04`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- 结论：
  - packed L1 side copy -> GM diagonal 写回已成为通过路径。
  - 旧失败签名来自 scalar packed/compact L1 回读语义，不是 packed 结果本身。
  - 现在可以进入 step 3：仅替换 32x32 compose 的输入提取/计算为 L1 resident，继续保持 64x64 compose 和 final RHS 不变。

## 18:48 step 3: 32x32 L1-resident compose

- 在上一条 packed readback PASS 边界基础上，仅将两处 `32x32 from 16x16` compose 从 `P8Compose32From16SingleGmDirect()` 切到 `P8Compose32From16SingleDirect()`。
- 保持不变：
  - P3 diagonal result 继续在 `aicHalfL1P3Result_` 与 GM diagonal 双写。
  - `64x64 from 32x32` 仍使用 `P8Compose64From32GmDirect32()`。
  - Stage2/Stage3/QK/`gBK`/`vBeta`/final RHS 未改。
- build/install PASS。
- smoke 结果：
  - `S=64` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
  - `S=256` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=1.172006e-03`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=4.882812e-04`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
  - `S=1024` PASS:
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=1.137391e-03`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=9.765625e-04`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- 结论：
  - step 3 通过。
  - 当前可继续进入 step 4：仅将 `64x64 from 32x32` compose 从 GM direct 切到 L1-resident `P8Compose64From32Direct32()`。

## 18:54 step 4: complete P8fuse compose

- 在 step 3 PASS 边界基础上，仅将 `64x64 from 32x32` compose 从 `P8Compose64From32GmDirect32()` 切到 `P8Compose64From32Direct32()`。
- 保持不变：
  - P3 diagonal result 继续在 `aicHalfL1P3Result_` 与 GM diagonal 双写。
  - 两处 `32x32 from 16x16` compose 保持 L1-resident `P8Compose32From16SingleDirect()`。
  - Stage2/Stage3/QK/`gBK`/`vBeta`/final RHS 未改。
- build/install PASS。
- smoke 结果：
  - `S=64` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.169131e-02`
    - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=3.173828e-02`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=4.882812e-04`
  - `S=256` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
    - `CPU vs ChunkGDRNeumann state max_abs=1.172006e-03`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
  - `S=1024` PASS:
    - `CPU vs ChunkGDRNeumann out max_abs=6.747369e-02`
    - `CPU vs ChunkGDRNeumann state max_abs=1.137391e-03`
    - `ChunkGDR vs ChunkGDRNeumann out max_abs=6.738281e-02`
    - `ChunkGDR vs ChunkGDRNeumann state max_abs=9.765625e-04`
- 结论：
  - 完整 P8fuse compose 在 `S=64/256/1024` smoke 上通过。
  - 但相对 step 3，`out` 最大误差从约 `3.17e-03` 阶跃到 `3.17e-02~6.75e-02`，`state` 仍稳定。
  - 需要继续跑当前 64-aligned compare set；若 compare PASS，可把 step 4 记为通过但带 out 误差上升风险。

## 18:56 step 4 compare

- 执行当前 64-aligned compare set：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare`
- 结果：
  - `16 passed in 6.18s`
- 覆盖范围：
  - `S=64/128/256/512`、varlen 64-aligned、不同 batch/head、`dk/dv=32/64/128`。
  - 非 64-aligned tail cases 仍按当前阶段要求保持注释。
- 观察：
  - compare 全部 PASS。
  - 多数 case 的 `ChunkGDR vs ChunkGDRNeumann out max_abs` 在 `~2e-02` 到 `~7.5e-02`，高于 step 3，但仍通过当前精度阈值。
  - `state` 仍稳定，常见 `ChunkGDR vs ChunkGDRNeumann state max_abs` 为 `4.882812e-04` 或 `9.765625e-04`。
- 结论：
  - step 4 完整 P8fuse compose 可作为当前 active 通过边界。
  - 文档和后续汇报必须保留 out 误差阶跃上升这个风险，不应描述为与 step 3 完全等价精度。
