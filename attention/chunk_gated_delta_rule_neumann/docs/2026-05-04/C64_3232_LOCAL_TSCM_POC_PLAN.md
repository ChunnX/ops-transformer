# 2026-05-04 C64_3232 Local TSCM PoC Plan

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

当前最终三方案中，`C64_3232_NeumannCompose` 是已测性能最好的 C64 路线，但仍慢于 normal。`C64_1616` raw inv-add 探针已证明微调同步、DCCI、identity 写入无法解决 wall time，继续在该线上微调收益很低。

当前主要瓶颈仍是：

- tiny Matmul 固定调度开销。
- Neumann/compose 中间矩阵经 GM workspace 往返。
- FIX/MTE2 和 AIC/AIV 等待时间随子块调用数放大。

## 目标

在 `C64_3232_NeumannCompose` 上做一个独立 local/TSCM PoC，验证把 `INV`、`Lpow`、compose temp 尽量留在片上 buffer 是否能降低端到端耗时。

该 PoC 不改变三种最终方案命名，只作为 `C64_3232_NeumannCompose` 的实现变体。

## 实施原则

- 默认 selector 保持稳定 `C64_3232_FP32_CVInverse`，PoC 通过后再切换。
- 不再把 `C64_1616` raw inv-add 探针作为默认性能线。
- 先做小范围可验证 PoC，再跑全量性能。
- 精度门槛继续使用 `CGDR_GOLDEN_CHUNK_SIZE=64` 与 `atol=rtol=0.08`。

## 推荐实现顺序

### Phase 1：独立 selector

- 新增独立 Stage1 文件，建议命名为：
  - `chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_LocalTscm_CVInverse.h`
- 复用 `C64_3232_FP32_CVInverse` 的流程和同步。
- 不直接修改当前默认 `C64_3232_FP32_CVInverse` 主体，降低回退成本。

### Phase 2：片上 inverse PoC

- 参考 `C64_1616_FP16_CVInverse` 已有的 `TPosition::TSCM + CubeFormat::NZ + half` 本地 Matmul 写法。
- AIV 生成 half 版 `L` 与 `I-L`，AIC 复制到 TSCM。
- AIC 在 TSCM 内完成：
  - 32x32 对角块 4 轮 Neumann。
  - `validLen <= 16` 的 16x16 shortcut。
  - `Compose64From32()` 的 lower-left compose temp。

### Phase 3：final RHS

- 优先让最终 `INV` 从 TSCM 直接作为 A 输入执行：
  - `INV @ gBKWs`
  - `INV @ vBetaWs`
- 第一版可保留 RHS half workspace，使用 `half x half -> float` 输出到 GM。
- 若精度不稳，再回退 final RHS 到 FP32/GM，只保留 inverse/compose local 化做对比。

### Phase 4：验证

- 构建安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 精度：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
- 性能 probe：
  - `prefill_b8_s64_d64_h16_hkv4`
  - `prefill_b8_s512_d128_h32_hkv8`
- 只有 probe 明显优于 `C64_3232_hybrid` 时，才执行 32 shape 全量 msprof。

## 回退条件

- 20 case 精度无法在 `0.08/0.08` 内通过。
- `S=64/S=512` probe 仍接近或慢于当前 `C64_3232_hybrid`。
- local/TSCM 版本引入大量额外 AIV scalar 或同步等待，抵消 GM 往返收益。

## 预期产物

- 独立 PoC selector 文件。
- 精度记录。
- `S=64/S=512` msprof probe CSV。
- 若 probe 有收益，再生成 32 shape 全量性能表。
