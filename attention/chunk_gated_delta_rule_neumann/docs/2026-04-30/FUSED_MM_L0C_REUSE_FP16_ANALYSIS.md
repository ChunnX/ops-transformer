# 2026-04-30 Fused MM、L0C 复用与 FP16 Neumann 分析

## 背景

当前 `chunk_gated_delta_rule_neumann` 的 Stage1 Neumann 求逆仍以多次 `MatmulImpl + GM workspace + FIX/MTE2` 串联为主。即使已经把 `INV += INV @ Lpow` 从 AIV 搬到 AIC/FixPipe atomic add，`L_pow / attn / INV / tmp` 仍频繁落 GM，profiling 中 `aic_scalar / aic_mte2 / aic_fixpipe` 固定开销仍明显。

用户要求后续四个 Stage1 方案都改成 fusedMM，不允许 Neumann 级数展开过程再出 AIC 核：

1. `C16_FP32_CVInverse`
2. `C16_FP16_CVInverse`
3. `C64_1616_FP32/FP16_CVInverse`
4. `C64_3232_FP32/FP16_CVInverse`

Stage2/Stage3 暂时不动。

## 已确认接口结论

### 输入侧没有 FP32 -> FP16 随路 cast

本地 CANN 9.0.0 头文件显示，Cube `Mmad` 支持的核心输入组合为：

- `half × half -> float`
- `bfloat16_t × bfloat16_t -> float`
- `float × float -> float`
- `int8 × int8 -> int32`

证据：

- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/TMatmul.hpp:80`
- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/TMatmul.hpp:82`
- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/TMatmul.hpp:84`
- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/TMatmul.hpp:86`

因此 `float GM/L1 -> half L0A/L0B -> Cube` 不能靠 Matmul 接口自动完成。真正 FP16/BF16 inverse 必须显式生成 half/bf16 副本，然后用 half/bf16 MatmulType。

### 输出侧支持 F32 -> F16/BF16 cast

L0C/Fixpipe/TStore 支持把 FP32 accumulator 写出为 half/bf16：

- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/common.hpp:18`
- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/common.hpp:22`
- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/common.hpp:24`
- `/home/developer/Ascend/cann-9.0.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp:590`
- `/mnt/workspace/gitCode/cann/ops-transformer/common/include/kernel/l0c_to_gm_iterator.h:53`
- `/mnt/workspace/gitCode/cann/ops-transformer/common/include/kernel/l0c_to_gm_iterator.h:115`

这能用于 FP16/BF16 Neumann 中间结果落半精度，但不能解决输入侧 FP32 自动降精度。

### Matmul 支持片上串联，但不是 L0C 直接串联

PFA 已有 `mm1 -> VECCALC -> mm2` 的本地串联范式：

- `/mnt/workspace/gitCode/cann/ops-transformer/attention/prompt_flash_attention/op_kernel/arch32/prompt_flash_attention_base.h:141`
- `/mnt/workspace/gitCode/cann/ops-transformer/attention/prompt_flash_attention/op_kernel/arch32/prompt_flash_attention_base.h:144`
- `/mnt/workspace/gitCode/cann/ops-transformer/attention/prompt_flash_attention/op_kernel/arch35/service/prompt_flash_attention_mm2.h:172`
- `/mnt/workspace/gitCode/cann/ops-transformer/attention/prompt_flash_attention/op_kernel/arch35/service/prompt_flash_attention_mm2.h:178`

这说明可以把 Matmul 输出放在 `VECCALC/TSCM`，再作为下一次 Matmul 的输入，从而绕开 GM workspace。它仍然经过 `L0C -> Fixpipe -> VECCALC/TSCM -> LoadData -> L0A/L0B`，不是 CO1/L0C 直接作为下一次 A/B。

## 能否调整 MM 顺序把累加留在 L0C

### 结论

可以做一部分“累加融合”，但不能把 Neumann 链条长期直接留在 L0C。

原因：

- L0C/CO1 是 Matmul accumulator 位置，只能作为输出或同一 Matmul 的累加 C。
- 下一次 Matmul 的 A/B 必须进入 L0A/L0B，不能直接从 L0C/CO1 作为 A/B。
- 若要把上一次结果作为下一次输入，至少需要 `Fixpipe/TStore` 到 `VECCALC/TSCM/L1/GM`，再 `LoadData` 到 L0A/L0B。

所以真正可行的方向是：

1. **同一 Matmul 内部的 K 分块累加留 L0C**：可行，但只适合同一个输出 C 的 K 分片累加。
2. **把 `INV += INV @ P` 改写成一次 Matmul**：可行，收益明确。
3. **把中间结果留在 VECCALC/TSCM/L1，而不是 GM**：可行，需要新 MatmulType 和本地 buffer。
4. **跨 round 长期复用 L0C 作为 A/B**：不可行，除非写 raw micro-kernel 并做 `L0C -> L1/UB -> L0A/B` 显式搬运，本质仍不是 L0C 直接复用。

## 推荐的 MM 顺序调整

### 当前写法

每轮 Neumann 近似如下：

```text
P_next = P @ P
tmp    = INV @ P_next
INV    = INV + tmp
```

当前 AIC atomic-add 版本实际是：

```text
P_next -> GM
INV @ P_next -> Fixpipe atomic add 到 INV_GM
```

这消除了 AIV add，但仍有 GM read-modify-write/atomic 和多次 GM workspace 往返。

### 改写为因子乘法

利用：

```text
INV_new = INV_old + INV_old @ P
        = INV_old @ (I + P)
```

其中 `P = L² / L⁴ / L⁸`，严格下三角幂矩阵对角线为 0，只需把 `P` 的对角线置 1，就得到 `I + P`。

新顺序：

```text
P2  = L @ L
F2  = I + P2
INV1 = INV0 @ F2

P4  = P2 @ P2
F4  = I + P4
INV2 = INV1 @ F4

P8  = P4 @ P4
F8  = I + P8
INV3 = INV2 @ F8
```

收益：

- 删除 `INV += tmp` 的 atomic-add 语义。
- 删除 `tmpWsGm_` 作为 add 中间结果的必要性。
- `INV` 使用 ping-pong buffer，避免同一 GM tensor 同时读写。
- Matmul 次数不增加；`INV @ P` 和 `INV @ (I+P)` 都是同尺寸 dense Matmul。
- 如果 `F2/F4/F8` 与 `INV ping-pong` 留在 `VECCALC/TSCM`，可以进一步减少 GM 往返。

风险：

- 需要 AIC 内对 `P` 对角线置 1。若 high-level Matmul 输出在 GM，可用 AIC scalar 或局部 buffer 做；若输出在 VECCALC/TSCM，需要用本地 vector/scalar 写对角。
- 需要确认 Matmul output 到 input ping-pong 时的 buffer 生命周期和同步。
- 最终 `INV` 活跃 buffer 可能是 `INV_A` 或 `INV_B`，后续 `INV @ k_cumdecay`、`INV @ v_beta` 必须读正确 buffer。

## L0/L1/VECCALC 复用机会

### 机会 1：`INV` 最终矩阵不落 GM，直接喂后续两次 Matmul

Stage1 内最终 inverse 只用于：

```text
outKCumdecay = INV @ gBK
outVInner    = INV @ vBeta
```

`INV` 不需要作为 Stage1 输出暴露。因此如果 final `INV` 留在 `VECCALC/TSCM/L1`，可以连续作为 A 输入给两次 Matmul，避免：

- final `INV` 写 GM。
- `INV @ gBK` 读 GM。
- `INV @ vBeta` 再读 GM。

这是比单纯优化 Neumann round 更值得优先做的片上复用点。

### 机会 2：`P2/P4/P8` 留在 VECCALC/TSCM，而不是 GM

`P_next = P @ P` 的输出不能直接从 L0C 作为下一次 B，但可以输出到 `VECCALC/TSCM`，再作为下一次 Matmul 输入：

```text
L_GM/Local -> P2_Local
P2_Local -> F2_Local -> INV1_Local
P2_Local -> P4_Local
P4_Local -> F4_Local -> INV2_Local
P4_Local -> P8_Local
P8_Local -> F8_Local -> INV3_Local
```

这能把 `L_powWsGm_ / attnWsGm_ / tmpWsGm_` 的 GM 往返替换成片上 buffer 往返。

### 机会 3：C64 compose 过程用 local temp

C64 compose 当前类似：

```text
tmp_lower = lower @ A_inv
lower     = D_inv @ tmp_lower
```

`tmp_lower` 没有必要落 GM。可以输出到 `VECCALC/TSCM`，第二次 Matmul 直接读 local temp。

对于 `C64_3232`，compose 次数更少，局部化收益更稳定；对于 `C64_1616`，子块数更多，local temp 能减少 GM 往返，但 tiny Matmul 调度次数仍多。

### 机会 4：raw micro-kernel 下复用 L0A/L0B

如果下沉到 raw `LoadData/Mmad/Fixpipe`：

- 16×16、32×32 块可以显式放入 L1/L0A/L0B。
- 同一矩阵作为 A/B 的平方 `P @ P` 可以复用 L1 副本，减少 GM load。
- `MmadParams(initC=false)` 可用于同一个 C 的 K 分块累加。

但限制仍在：

- `P @ P` 的结果在 L0C，要作为下一轮 `P` 输入仍需搬到 L1/UB 后再 LoadData。
- L0C 不能直接变成 L0A/L0B。
- raw micro-kernel 实现复杂度明显高于 high-level MatmulImpl，建议等 VECCALC/TSCM PoC 证明收益后再做。

## FP16 Neumann 路线判断

### 为什么可能有收益

如果 `L / P / F / INV` 全部用 half/bf16 保存，并用：

```text
half × half -> float accumulator
Fixpipe float -> half
```

则理论收益包括：

- 中间矩阵 bandwidth 减半。
- local/GM workspace 减半。
- FP16/BF16 Cube 路径通常比 FP32 更适合小矩阵吞吐。
- 若配合 local fused，`VECCALC/TSCM` 可容纳更多 ping-pong buffer。

### 必须注意的边界

`Mmad` 不支持 `half × float` 或 `float × half`。因此如果 final `INV` 是 half，而 `gBKWsGm_ / vBetaWsGm_` 仍是 float，则不能直接做 final Matmul。

可选策略：

1. **只让 Neumann 内部 FP16，final INV cast 回 FP32 再做后续 Matmul**  
   风险低，但收益可能被 cast 和额外搬运抵消。

2. **Neumann + final `INV @ rhs` 都 FP16/BF16**  
   需要把 `gBK / vBeta` 也生成 half/bf16 副本，性能潜力更大，但精度风险更高。

3. **BF16 优先于 FP16 评估**  
   输入本身是 BF16，BF16 动态范围更稳；FP16 可能更快但需要重点看长序列和 gamma/beta 组合下误差。

### 实施顺序

1. 先做 `C16_FP16_CVInverse`：验证 Neumann 内部 half/bf16 是否精度可接受、性能是否真实下降。
2. 如果 C16 FP16 对 `aic_mac / mte2 / fixpipe` 有明显收益，再扩展到：
   - `C64_1616_FP16_CVInverse.h`
   - `C64_3232_FP16_CVInverse.h`
3. 若 C16 FP16 只改善很小，应优先做 `VECCALC/TSCM` local fused，而不是扩大 FP16 文件数量。

## 四个方案的落地优先级

### 第一阶段：FP32 fusedMM，先消除 atomic add

目标：所有 FP32 方案先改成因子乘法 `INV @ (I+P)` + ping-pong。

- `C16_FP32_CVInverse.h`
- `C64_1616_FP32_CVInverse.h`
- `C64_3232_FP32_CVInverse.h`

预期收益：

- 降低 `aic_fixpipe` 和 atomic-add 相关开销。
- 删除或弱化 `tmpWsGm_`。
- 为后续 local buffer fused 铺路。

### 第二阶段：C16 FP16 fusedMM

目标：在 `chunk_gated_delta_rule_neumann_stage1_C16_FP16_CVInverse.h` 实现真实 half/bf16 输入 Matmul，不再只是文件名 FP16。

验证口径：

- `CGDR_GOLDEN_CHUNK_SIZE=16 bash test_run.sh compare`
- 32 shape msprof，重点看 `aic_mac_time`、`aic_mte2_time`、`aic_fixpipe_time`。

### 第三阶段：VECCALC/TSCM local fused PoC

目标：把 `P/F/INV` ping-pong 从 GM 搬到 `VECCALC/TSCM`，最终 `INV` 直接喂 `INV @ gBK/vBeta`。

优先做：

1. C16 local fused primitive。
2. C64_3232 local compose。
3. C64_1616 local compose。

### 第四阶段：raw micro-kernel

仅在 high-level Matmul local fused 仍受限时进入。目标是专门为 16×16/32×32 inverse 写 `LoadData/Mmad/Fixpipe` 流水，显式管理 L1/L0A/L0B/L0C。

## 当前判断

- 用户提出“调整 MM 顺序，把部分累加留在 L0C/复用 L0 tensor”的方向是对的，但要落到可执行层面：**不要追求跨 round 直接 L0C 常驻**，应优先把加法折进 Matmul 因子 `I+P`，再把中间矩阵从 GM 改到 `VECCALC/TSCM`。
- 真正可立刻尝试的优化是 `INV += INV@P` -> `INV @ (I+P)`，它不增加 dense Matmul 次数，但能去掉 atomic add 和 `tmp` add 语义。
- L0C 直接复用只能用于同一 Matmul 内部累加；Neumann 的下一轮需要把结果作为 A/B，因此至少要经 `Fixpipe` 到片上 local buffer。
- FP16 方案值得做，但必须是真 half/bf16 输入链路；仅靠输出随路 cast 不能获得完整 FP16 Matmul 收益。
