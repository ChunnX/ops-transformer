# 任务：tri_inverse MixInverse 全部优化

## 目标
在 `op_kernel/tri_inverse_mix.h` 内完成 MixInverse 优化，保持接口/tiling/golden/wrapper 不变，最终性能超过 `op_kernel/tri_inverse.h` 参考 baseline。

## 当前边界
- 只改 `op_kernel/tri_inverse_mix.h` 和必要文档。
- `op_kernel/tri_inverse.h` 是保护 baseline，不能优化或覆盖。
- MMAD 输入必须是 FP16；输出可用 Fixpipe 随路转 BF16。
- packed64 diagonal inverse 的语义是把 4 个独立 `16x16` diagonal block 放进一个 `64x64` packed 容器并行求 4 路 `16x16` inverse；它不是完整 `64x64` triangular inverse，不能把 inner-lower/cross-lower 混进这个 diagonal solve。
- 每次只改一个数据流；精度门禁为 `64/256/1024 seed=0/21` 后再跑 full 30 cases。
- 每个 PASS 阶段保存 `op_kernel/baseline/` 版本。
- profile 不只保留 csv，需要把典型 `Duration(us)`、cube/MAC 等原始字段值写入 md 表格。
- 单点字段改善也要保留：例如 Phase11/Phase13 的 `aic_mte2_time(us)` 改善，即使总耗时仍未超过 v0。

## 当前状态
- 当前 active baseline：Phase31a。
- 保存文件：`op_kernel/baseline/tri_inverse_mix_phase31a_zero64_identity64_once_full30_pass.h`。
- Active sha256：`866a0f6c0d1d3ec8e68f274ae524ddac678ff9f5f740db36d19e0c40b293dabf`。
- 验证记录：Phase31a 已在健康 chip1 上用 `ASCEND_RT_VISIBLE_DEVICES=1` 验证通过：
  `shape=64/256/1024 seed=0/21 PASS`，`bash test_run.sh single -> 30 passed`。
- Phase31a 语义：保留 Phase31 的 P7 packed64 resident 边界；P7 先把四个独立 `16x16` diagonal inverse 写入同一个 packed64 L0C 容器，
  `cInv=aicL0C_` 在 P7 边界语义上持有 `D0/D1/D2/D3` 四个 diagonal lane。每个非零 lane 写入后立即 Fixpipe 到 L1 `result64` shadow，off-diagonal lane 也用 zero MMAD + Fixpipe 消费，避免未消费 L0C lane 写入导致超时/挂起。Phase31 后半段仍保留 Phase30 兼容输出路径，尚未实现 P8 L0C 累加。
- Phase31a 新增：`InitRawMmadLocalBuffers()` 后一次性准备 L1 resident `zero64` 和 `identity64`，`identity64` 使用 Neumann 同款 packed64 diagonal offset/stride 从 `TRI_INV_IDENTITY16_NZ4` 搬入；当前只准备并复用常量，不改 P8 compose/accumulation。
- Phase31 profile 已补测：`tests/pytest/prof_out/tri_inverse_mixinverse_phase31_p7_4x16_packed64_diag_full30_6shape/tri_inverse_mixinverse_phase31_p7_4x16_packed64_diag_full30_6shape.csv`。典型 `Duration(us)` 为 `64=25.560, 65=28.020, 1024=28.661, 1025=28.980, 10240=152.083, 10241=167.203`，明显慢于 Phase30；Phase31 只能作为 P7 resident 正确性边界，不能作为性能方向本身。
- 下一步必须是 Phase32：基于 P7 resident 的 `cInv` 做 P8 first-layer `cTmp + cInv` L0C 累加；不要回到先做 legacy compose 的路径。
- Phase29a profile：`tests/pytest/prof_out/tri_inverse_mixinverse_phase29a_identity32_first_batch_l1_reuse_full30_6shape/tri_inverse_mixinverse_phase29a_identity32_first_batch_l1_reuse_full30_6shape.csv`。
- Phase30 profile：`tests/pytest/prof_out/tri_inverse_mixinverse_phase30_result64_topright_zero_only_full30_6shape/tri_inverse_mixinverse_phase30_result64_topright_zero_only_full30_6shape.csv`。
- Phase23/27/29a rerun 对比：Phase29a 总 `Duration(us)` 在 `64/65/10240/10241` 最好，`1024/1025` 仍是 Phase23 最好；Phase29a 比 Phase27 6 个 shape 全部更快。Phase27 不是纯 identity 方案，它还把 raw-MMAD init/restore/reset 移出 batch 循环，所以不能用来否定“identity 全局一次加载”。
- Phase30 对 Phase29a：`64` 快 2.800us、`10241` 快 2.780us；但 `65/1024/1025/10240` 变慢，尤其 `10240` 慢 0.800us。Phase30 是可用分支，但是否继续作为 active 需要看下一步能否做条件化或消除额外 scalar/同步开销。
- v0/reference：Phase30 首次让 `10241` 优于 v0（64.861 vs 66.040），但 `64/65/1024/1025` 仍慢于 v0，所以最终性能还未达成。
- 2026-05-12 设备边界：物理 chip0 仍是 Alarm，物理 chip1 OK；后续运行必须显式 `export ASCEND_RT_VISIBLE_DEVICES=1`，因为 wrapper 内部会 `set_device(0)`。
- 无效/失败分支不要复用：Phase15 删除中间 BF16 GM 写回后受并发 profiling/设备 Alarm 污染，不是有效 baseline；Phase15a 在 `shape=64 seed=0` 挂起；Phase20 AIC 直接 `SetValue` 构造 `identity32` 失败（max_abs=1.0）；Phase21 compose32 lowerWork 2D `DataCopyPad` 失败（max_abs=0.234375）。
- 2026-05-12 KDA 参考结论：`/mnt/workspace/gitCode/refresh/KDA/kernel_tri_inv_rec_unroll_annotated.cpp` 只可参考 diagonal `16x16` fractal 固定搬入思想。`N=64,F=16` 时 diagonal lane L0 偏移为 `0/1280/2560/3840`（`i * F * (N + F)`）；MBH/LX/RX 上三角 compose 公式不能搬到本 lower-triangular 路径。
- 2026-05-12 Phase31 失败边界：单次 packed64 diagonal-lane smoke（固定 L1->L0 搬入、一次 `64x64` MMAD、Fixpipe 到 L1，结果不参与输出）可通过 `shape=64 seed=0`。但把每轮 `16x16` inverse update 直接替换为 full `64x64` packed MMAD 后，`shape=64 seed=0` 超时/挂起，两次尝试均回退。不要继续走“每轮 full64 MMAD 求逆”的绕路；下一步应保留 `16x16` 为计算单元，只优化 4 路固定搬运/identity 生命周期，或者单独做 L0C 常驻累加。
- 2026-05-12 顺序修正：必须先做 P7 `4x16 -> packed64` diagonal inverse，使 P7 后 `cInv=aicL0C_` 语义上持有 `D0/D1/D2/D3` 四个 `16x16` diagonal inverse。只有这个边界成立后，P8/compose 才能采用 `cTmp + cInv` L0C 累加方案。当前输入 scratch 已经是 `lowerNeg=-L`，后续 compose 不要额外加负号。
- 2026-05-13 并发污染边界：Phase32 曾插入一个 no-Fixpipe 隔离候选（只做 `lowerNeg @ topInv`，再做 `bottomInv @ tmp` 写 scratch L0C，不参与最终输出），但同一 worktree 中另一个 Codex 会话反复启动 `chunk_gated_delta_rule*` build/profiling，并把 `op_kernel/tri_inverse_mix.h` 写回包含非法 `InvPlusPowerScratch` 的旧候选；共享 `build/` 和 installed OPP 也被覆盖。因此本次没有可信 Phase32 PASS/FAIL 结论，不能保存 baseline。已恢复 active 到 Phase31a：
  `sha256(op_kernel/tri_inverse_mix.h)=866a0f6c0d1d3ec8e68f274ae524ddac678ff9f5f740db36d19e0c40b293dabf`。

## 待办事项
- [x] 读取 `AGENTS.md` 与 `MIX_INVERSE_NEXT_WINDOW_HANDOFF.md`。
- [x] Phase 2：只改 diagonal inverse 到 FP16 raw-MMAD，full 30 PASS。
- [x] Phase 3：只改 `16 -> 32` lower-triangular compose，full 30 PASS。
- [x] Phase 4：只改 `32 -> 64` lower-triangular compose，full 30 PASS。
- [x] Phase 5-10：移除旧 AIV baseline compute、packed32 compose、diag32 cube、zero/scratch 调整，保存 PASS baseline。
- [x] Phase 11：`32x32` lowerWork 一次搬入，保留 MTE2 改善 baseline。
- [x] Phase 12：去掉 top diag reload，full 30 PASS。
- [x] Phase 13：把 Phase11 的 lowerWork 一次搬入合到 Phase12，full 30 PASS，profile 已写入 handoff。
- [x] Phase 14：L1 `result64` shadow 组装，full 30 PASS baseline 已保存。
- [x] 清理：删除本目录 pytest/python 缓存，确认无 Phase15 baseline 或 invalid profile 残留。
- [x] 环境恢复边界：使用 `ASCEND_RT_VISIBLE_DEVICES=1` 绕开 Alarm chip0；Phase14 已在 chip1 重验通过。
- [x] Phase 15b：只减少 final `result64 @ I32` replay，不动 diagonal/compose 自身 GM 写回；full 30 PASS，baseline 已保存。
- [x] Phase 15b profile：显式 6-shape profile 已采集并写入 handoff；10240 变慢，不能作为最终性能方向。
- [x] Phase 16：每 tile 只生成/加载一次 `identity32`，复用到 top/bottom diagonal32；full 30 PASS，baseline/profile 已保存。
- [x] Phase 17：只减少 diagonal32 final `INV32 @ I32` replay；full 30 PASS，baseline/profile 已保存。
- [x] Phase 18：只减少 compose32 final replay；不动 AIV scratch/identity，不做 L0C 累加；full 30 PASS，baseline/profile 已保存。
- [x] Phase 19：bottom diag scratch 由 subBlock1 写，去掉 subBlock0 对 bottom half 的二次搬入；full 30 PASS，profile 已写入 handoff。
- [x] Phase 20 失败：AIC 每 tile 在 L1 直接 `SetValue` 构造 `identity32`，保留 AIV identity scratch 写，仍在 `shape=64 seed=0` 失败（max_abs=1.0）；已回退 Phase19，不再原样重试。
- [x] Phase 21 失败：compose32 lowerWork AIV->GM scratch 写从逐行 `DataCopyPad` 改一次 2D `DataCopyPad`，`shape=64 seed=0` 失败（max_abs=0.234375）；已回退 Phase19，不再原样重试。
- [x] Phase 22：AIC 侧每 batch 首 tile 加载一次 `identity32`，top/bottom 和后续 tile 复用；full 30 PASS，baseline 已保存。
- [x] Phase 23：AIV 侧每 batch 首 tile 生成/写出一次 `identity32` scratch，和 Phase22 配套；full 30 PASS，baseline/profile 已保存。
- [x] Phase 27：`identity32` 尝试提升到 AIC 生命周期，并同时 hoist raw-MMAD init/restore/reset；full 30 PASS，baseline 已保存。rerun 中大 shape `aic_mte2_time(us)` 降低，但总 `Duration(us)` 全 shape 慢于 Phase23。
- [x] Phase 29a：只保留“第一个 batch 写/加载一次 identity32 并在 L1 持久复用”，不 hoist init/reset；full 30 PASS，baseline/profile 已保存。
- [x] Phase 30：只把 `result64` 清零从整块 `64x64` 缩到 top-right `32x32`；full 30 PASS，baseline/profile 已保存。profile 对 64/10241 有收益，对 65/1024/1025/10240 退化，后续不能无脑叠加。
- [x] Phase 31：先做 P7 `4x16 -> packed64` diagonal inverse。目标是把四个独立 `16x16` diagonal block 放进同一个 `64x64` packed 容器/lane 中求逆，P7 结束时 `cInv=aicL0C_` 保持：
  `D0 0 0 0 / 0 D1 0 0 / 0 0 D2 0 / 0 0 0 D3`。这一步不改 lower-left compose，不做 P8 L0C 累加，不合并 DataCopy，不改接口。先过 `shape=64 seed=0`。
- [x] Phase 31a 实现边界：参考 GDN/Neumann 已有 P7 packed64 diagonal 逻辑，只复用固定 lane 搬运、`identity64` 生命周期、`cInv/cPower` L0C 分工；不要复制整套 Neumann inverse，也不要把每轮 `16x16` update 变成完整 `64x64` triangular inverse。MMAD 输入仍为 FP16。
- [x] Phase 31b 验证边界：P7 packed64 diagonal inverse 先跑 `shape=64/256/1024 seed=0/21`，再 full 30；通过后保存 `op_kernel/baseline/tri_inverse_mix_phase31_p7_4x16_packed64_diag_full30_pass.h`。
- [x] Phase 31a 常量生命周期：在 `InitRawMmadLocalBuffers()` 后一次性准备 `zero64` 和 `identity64`，后续路径复用；不启动 P8 first-layer/second-layer 累加。`shape=64/256/1024 seed=0/21 PASS`，full 30 PASS，baseline 已保存。
- [ ] Phase 32 前置安全检查：确认无其它 Codex/build/profiling 会话在同一 worktree 写 `tri_inverse_mix.h` 或共享 `build/`/installed OPP；重新校验 active 源码、package header、installed header、object hash 后再采信任何结果。
- [ ] Phase 32：P7 PASS 后再做 P8 first-layer L0C 累加。公式固定为 `tmp = lowerNeg @ topInv`、`out = bottomInv @ tmp`，不额外加负号。用第二块 L0C `cTmp` 算 tmp，再 `RAW_MMAD_ACCUMULATE_C` 累加到 P7 保留下来的 `cInv`，先 full64 Fixpipe 回 L1 给现有 second-layer 用。
- [ ] Phase 33：first-layer PASS 后，再把 second-layer 改成 `tmp64 = lowerNeg32 @ B0`，`cInv += B1 @ tmp64`，最后 full64 Fixpipe 一次到 L1 `result64`/验证 GM。
- [ ] 后续优化队列：16x16 求逆仍有冗余；目标是 4 个 `16x16` 一次打包进 `64x64` diagonal lanes 并行求 4 路 `16x16` inverse，不是求完整 `64x64` inverse。当前 Phase30 仍是 top32/bottom32 两个 diagonal solve；`HalfL1InvAddBlock32()` 只做局部 L0C 累加，尚未做完整 L0C 常驻，中间 power/inv 仍 Fixpipe 回 L1。
- [ ] 后续优化队列：更大一步再单独做 L0C 常驻累加，减少每轮 `inv @ I` copy；不要和 movement/DataCopy 合并。
- [ ] 后续优化队列：梳理 DataCopy 绕路，当前仍有 AIV BF16->FP16->GM scratch -> AIC L1、compose32 lowerWork GM 往返、diagonal/compose 结果 L1+GM 双写。后续优先处理 `shape=64` 启动/固定开销和 `10241` 尾 tile 开销。
- [ ] 后续优化队列：`identity32` 已从每 tile 降到每 batch 一次，但还不是整个算子全局一次；`TRI_INV_IDENTITY16_NZ` 静态 GM 常量当前 active 路径未使用，直接 AIC/static identity 旧尝试不稳定，不能原样重试。
- [ ] 后续边界：L1 是后续 compose/RHS 内部结果，GM 是 standalone pytest/golden 的最终 `cc_out` 和 AIV->AIC scratch 通道；融合到后续算子后再删除可移除 GM 输出。
- [ ] 通过profiling继续寻找优化点，看是谁拖慢了时间，该方案必须必对比方案好能达到至少0.7x加速比。

## 命令
```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=tri_inverse -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet
cd experimental/attention/tri_inverse/tests/pytest
source ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/bin/set_env.bash >/dev/null 2>&1 || true
export ASCEND_RT_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
bash test_run.sh single
```
