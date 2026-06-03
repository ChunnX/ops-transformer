# 2026-05-07 P7/P8 Packed Fast Inverse 方案理解

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

当前 active path 已回退到 P3：`C64_1616_FP16` 的 16x16 Neumann 输出与 16->32、32->64 compose 中间态保留在 FP16 GM scratch，full-L1 resident compose 未通过。

P4 直接 full-L1 compose 的问题集中在 resident compose 本体：

- `Compose64From32FullL1ViaGm()` 诊断路径 `S=16/17/33` PASS，说明 full-L1 diagonal 输入、最终写回和 AIV half->BF16 cast 链路基本可用。
- direct resident `32->64` compose 在 `S=33` 失败，state `max_abs=9.669919e-02`。
- 因此继续沿 P4 朴素 full-L1 offset 修补，风险较高。

用户建议直接转向 P7/P8，参考 `pto-kernels` 的 fast inverse 实现：

- `CopyDiagonalFractalsL1ToL0()`
- `CopyOddOrEvenBlocksL1ToL0()`
- MXR / MBH 总体思路

## 方案理解

P7/P8 的核心不是简单把 GM compose 的输入输出换成 L1，而是改成结构化打包矩阵乘。

### L1 zero-fill

P7/P8 的 packed 矩阵必须先可靠清零。`set2d` 可以批量设置 L1 内存初始值为 0，应作为 packed tile 初始化的优先手段：

- P7 先将 packed diagonal L1 scratch 清零，再只填四个 16x16 diagonal fractal。
- P8 先将 even/odd packed L1 scratch 清零，再只填本轮参与的 even 或 odd blocks。
- 清零不能用逐元素标量循环作为默认实现；逐元素方式只适合作为调试 fallback。
- 如果最终实现需要先在 L1 构造 packed tile，再搬到 L0A/L0B，`set2d` 可以避免 off-diagonal 残留污染 MMAD。

### P7：packed diagonal 16x16 Neumann

参考 `CopyDiagonalFractalsL1ToL0()`：

- 从 full C64 half L1/TSCM 中抽取四个 16x16 对角块。
- 将它们放到 L0A/L0B 的 block-diagonal 位置，其余 off-diagonal 位置保持 0。
- 用一次 64x64 packed MMAD 同时表达四个独立的 16x16 `power@power` 或 `inv@power`。
- 输出仍是 block-diagonal 结果，再散回 full C64 L1/TSCM 的四个对角位置。

这一步的目标是减少四个独立 `Neumann16InverseBlock()` 的 LoadData、MMAD、Fixpipe、同步和 local buffer reset 固定成本。

### P8：odd/even packed MBH compose

参考 `CopyOddOrEvenBlocksL1ToL0()` 与 Unrolled-MBH：

```text
LX = even_blocks(X, block_size)
RX = odd_blocks(X, block_size)
X = (LX @ (-M) + I) @ RX + LX
block_size *= 2
```

迁移到当前 C64_1616 时，对应两层 compose：

- `block_size=16`：把 16x16 diagonal inverse 组装为两个 32x32 inverse。
- `block_size=32`：把两个 32x32 inverse 组装为完整 64x64 inverse。

PTO helper 的关键点是按奇偶块把 `X` 的 block diagonal 搬到 L0：

- L0A 只放 even blocks。
- L0B 只放 odd blocks。
- 目标 tile 先清零，未参与的 block 必须保持 0。
- block 内部仍按 fractal 16x16 粒度搬运，`block_size=32` 时需要搬 2x2 个 fractal。

这样一次 packed MMAD 只会在数学上产生本轮需要的 compose 结果，避免朴素 P4 在 full/compact NZ offset 上直接索引子块导致的布局风险。

## 与当前代码的映射

当前 P3 主流程：

- `Neumann16InverseBlock(paraIdx, base)`
- `Compose32From16(paraIdx, base)`
- `Compose64From32(paraIdx)`
- 数据主要落在 `INVInitHalfWsGm_`，compose 通过 `HalfGmComposeBlock()` 做局部 half L1 MMAD 后写回 GM。
- 初始 `L/I-L` 生成保持 Stage1 原有简单写法：AIV 从 `kkWsGm_` 生成 `L`，`I-L` 只是 `-L` 后将 diagonal 置 1；P3 没有把初始矩阵生成复杂化，只是把输出 dtype/落点改为 FP16 scratch。

P7/P8 目标流程：

- AIC 将当前 `L_powHalfWsGm_` / `INVInitHalfWsGm_` 先搬入 C64 full half L1/TSCM。
- P7 对四个 diagonal 16x16 block 做 packed diagonal inverse。
- P8 用 odd/even packed MBH 组装 16->32 和 32->64。
- 最终只把完整 inverse 写回 FP16 GM scratch 或直接写 BF16 `attnWsGm_`；final RHS 仍保持当前 BF16 GM `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_, ...)`。

## 实施顺序

1. 新增独立 P7/P8 helper，不修改 P3 干净路径。
2. 先做 P7 packed `block@I` 探针：验证四个 16x16 diagonal block 抽取、L0 block-diagonal offset、L0C 输出散回位置。
3. 再做 P7 packed Neumann：只替换四个 diagonal inverse，compose 继续走已通过 GM fallback。
4. P7 全量 compare 通过后，做 P8 `block_size=16` odd/even compose，验证 `S=17/33/64`。
5. 再做 P8 `block_size=32` compose，验证 `S=33/64/125`。
6. P7/P8 全路径精度通过后，再跑一次典型 shape 性能并暂停。

## 验收与风险

验收必须继续遵守单项 gate：

- 每个子项单独 build/install。
- smoke 覆盖 `S=16/17/33/64/125`。
- full compare `20 passed` 后才能进入下一项。
- P7/P8 未完整通过前不跑性能。

主要风险：

- AscendC 当前代码不直接使用 PTO tile API，不能逐行照搬 `TEXTRACT/TASSIGN/TMATMUL`，需要用现有 raw MMAD/local tensor 能力表达等价打包。
- L0A 与 L0B 的 row-major/col-major 内部布局不同，packed offset 必须分别处理。
- 目标 packed tile 必须清零，否则 off-diagonal 残留会污染 packed MMAD；优先用 `set2d` 批量清 L1，再填 diagonal/even/odd block。
- tail validLen 需要明确：未触发的 diagonal/odd/even block 必须保持 identity/zero 语义，不能读未初始化块。
- 未通过的 P7/P8 helper 不能进入 baseline，也不能作为后续优化基线。

## 2026-05-08 实施结果补充

当前实际落地状态：

- P8 已实现并进入 active compose 路径：
  - `P8ComposeBoth32From16Scratch64()` 用 64x64 FP16 L1 scratch 同时完成两个 `16->32` compose。
  - `P8Compose64From32Scratch64()` 用 64x64 FP16 L1 scratch 完成最终 `32->64` compose。
  - P8 helper 在填充参与 block 前调用 `ZeroHalfL1()`，该 helper 内部使用 `InitConstValue` 批量清零 L1，承接用户提醒的 `set2d`/批量清零方向，避免 packed scratch 中 off-diagonal 残留。
- P7 packed diagonal inverse 暂不作为 active 路径：
  - 原始 `P7PackedDiagonalNeumannGroup()` active 尝试在 `S=17` 出现精度失败。
  - 后续只写回 diagonal 子块的 Fixpipe 诊断在 `S=17` 超时。
  - full 32x32 block-diagonal MMAD 保守尝试仍在 `S=17` 精度失败。
  - 目前 P7 helper 只保留为诊断代码，不进入 `C64InverseBy16x16Neumann()` active path。
- 当前 active 方案是 `P3 16x16 diagonal inverse + P8 scratch64 compose`：
  - 四个 16x16 diagonal inverse 仍使用已通过的 `Neumann16InverseBlock()`。
  - `validLen > 16` 时先做 top-left `Compose32From16()`。
  - `validLen > 48` 时计算 lower-right 两个 16x16 diagonal inverse 并调用 `P8ComposeBoth32From16Scratch64()`。
  - `validLen > 32` 时调用 `P8Compose64From32Scratch64()`。
  - final RHS 仍保持 BF16 GM：`AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)` 与 `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`。

重新验证：

- 构建：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`，成功。
- 安装：`../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`，成功。
- smoke：`S=17/32/33/49/64/129` 全部 PASS。
- full compare：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.22s`。

注意：`S>=33` 的 Neumann out 误差相对 normal 明显增大，但仍在当前三方 compare 判据内通过；后续若要收紧误差阈值，需要单独处理数值/compose 累积误差。
