# 2026-04-30 C64_1616_FP32 Raw MMAD CO1 Bias 改造计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 目标

先在 `C64_1616_FP32_CVInverse` 上验证 raw `Mmad(..., bias, ...)` 的 CO1/L0C bias 累加路径，精度通过后再推广到三种最终方案和 FP32/FP16 两类实现。

目标文件：

- `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`
- 临时 selector：`op_kernel/chunk_gated_delta_rule_neumann_stage1.h`

## 背景与修正结论

此前使用 high-level `MatmulImpl` 做 `INV += INV @ P`，路径是：

```text
A/B -> MatmulImpl -> L0C -> FixPipe -> GM(INV) atomic add
```

这能把 AIV 加法移到 AIC/FixPipe，但仍然有 high-level Matmul 调度、FixPipe 写 GM、GM atomic add 等固定开销。

用户指出 raw `Mmad` 可以指定 bias 是 L0C/CO1 矩阵。复查底层接口后确认：

- `Mmad(dst, a, b, bias, params)` 支持 bias 位于 `CO1` 或 `C2`。
- bias 位于 `CO1/L0C` 时，底层使用 C 矩阵作为初值参与 `C = A @ B + C`。
- 这适合把 `INV += INV @ P` 改成 raw MMAD 的 L0C 内累加。

## 本阶段只改什么

只优化 Stage1 的 C64 16×16 Neumann inverse 路径，不修改 Stage2/Stage3。

第一阶段只替换 16×16 diagonal inverse 中的三次累加：

```text
Round 1: INV += INV @ L2
Round 2: INV += INV @ L4
Round 3: INV += INV @ L8
```

当前路径：

```text
AICBlockGmLocalGmStrided(INVWsGm, P_local, INVWsGm, 16, enableAtomicAdd=true)
```

目标 PoC 路径：

```text
1. Load INV block to L0C bias buffer
2. Load INV block to L0A
3. Load P block to L0B
4. Mmad(outL0C, INV_L0A, P_L0B, biasL0C, params)
5. FixPipe/DataCopy CO1 -> GM INVWsGm
```

## 不在第一阶段解决的问题

- `P2=P@P` / `P4=P2@P2` 仍可能用现有 local MatmulImpl，先不强行 raw 化。
- `Compose32From16` 和 `Compose64From32` 先不改，等 diagonal inverse PoC 精度通过后再评估。
- `INV @ gBK`、`INV @ vBeta` 先不改。
- 不改 `C64_3232`、`C16`、FP16 文件，避免一次性扩大风险。

## 设计约束

1. **精度门槛**：三方精度 `atol=0.08, rtol=0.08` 全量通过。
2. **只动 Stage1**：不修改 `stage2`、`stage3`。
3. **优先可回退**：保留现有 high-level helper，新增 raw helper；失败时只需切回旧 helper。
4. **不自行推送**：本地修改和验证，不执行 `git push`。
5. **性能暂不全量**：本阶段先精度；性能等用户确认后按 msprof seq sweep/全量规则执行。

## 实现步骤

### Step 1：新增 raw MMAD L0 buffer

在 `Stage1` AIC 初始化中新增：

- `TBuf<TPosition::A1>` 或对应 L0A buffer
- `TBuf<TPosition::B1>` 或对应 L0B buffer
- `TBuf<TPosition::CO1>` for output/bias
- 可能需要两个 CO1 buffer：一个保存 bias `INV`，一个保存输出；若同址 bias/output 可行，则先用同址减少 CO1 占用，否则用 ping-pong。

需要参考已有 raw MMAD 用例，优先使用 repo 中 `TBuf<TPosition::CO1>`、`LoadData`、`Mmad`、`Fixpipe/DataCopyCO12DstParams` 的写法。

### Step 2：实现 16×16 block load/store helper

新增 helper：

```text
LoadGmBlockToL0A(gm, baseOffset, blockSize=16)
LoadLocalBlockToL0B(local, blockSize=16)
LoadGmBlockToL0CBias(gm, baseOffset, blockSize=16)
StoreL0CBlockToGm(gm, baseOffset, blockSize=16)
```

如果 `LocalTensor<float>` 不能直接 LoadData 到 L0B，则 first PoC 可退一步：把 `P_local` 先保持现有 MatmulImpl local 输出，再通过支持路径搬入 L0B；若接口不支持，第一版改为 GM power 输入路径，再做 local 优化。

### Step 3：新增 raw 累加 helper

新增：

```text
RawMmadInvAdd16(uint64_t invOffset, LocalTensor<float> powerLocal)
```

语义：

```text
INV[block] = INV[block] @ powerLocal + INV[block]
```

关键参数：

```text
MmadParams params;
params.m = 16;
params.n = 16;
params.k = 16;
params.cmatrixInitVal = false;
params.cmatrixSource = false;  // C 初值来自 CO1/L0C
```

### Step 4：替换 diagonal inverse 三轮累加

将 `Neumann16InverseBlock()` 中三处：

```text
AICBlockGmLocalGmStrided(..., enableAtomicAdd=true)
```

替换为：

```text
RawMmadInvAdd16(offset, powerLocal)
```

保留 `L2/L4/L8` 生成逻辑不变。

### Step 5：切 selector 与构建

临时切换：

```text
#include "chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h"
```

构建：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
```

### Step 6：精度验证

先 smoke：

```bash
cd attention/chunk_gated_delta_rule_neumann/tests/pytest
CGDR_GOLDEN_CHUNK_SIZE=64 python3 compare_chunk_gdr_three_way.py --batch-size 1 --seqlen 16 --nk 1 --nv 1 --dk 16 --dv 16 --seed 123 --atol 0.08 --rtol 0.08
```

再全量：

```bash
CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare
```

### Step 7：根据结果决策

- **精度 PASS**：记录为 `C64_1616_FP32 raw MMAD CO1 bias` 第一阶段完成，然后推广到 compose / C16 / C64_3232 / FP16。
- **编译失败**：优先判断 L0A/L0B/CO1 buffer position 或 LoadData 参数问题，不扩大改动。
- **精度失败**：保留 raw helper，增加 debug 路径只验证 `INV + INV@P` 单轮；必要时回退 helper 并记录原因。

## 推广计划（精度通过后）

通过后进入 3×2：

| 方案 | FP32 | FP16 |
|---|---|---|
| C16_1616_Neumann | 改 `INV += INV@P` 为 raw MMAD CO1 bias | 同步实现 half 版 raw MMAD bias |
| C64_1616_NeumannCompose | 当前 PoC，后续扩展 compose | half 版，先 diagonal 后 compose |
| C64_3232_NeumannCompose | 32×32 diagonal inverse 使用 raw MMAD bias | half 版 32×32 |

优先级：

1. C64_1616_FP32 diagonal inverse PoC。
2. C64_1616_FP32 compose 中间乘法是否也能 raw 化。
3. C16_FP32 同步迁移。
4. C64_3232_FP32 迁移。
5. FP16 三方案迁移。

## 风险点

- CO1 buffer 同址 bias/output 是否可行；若不稳定，改双 CO1 buffer。
- `LocalTensor<float>` powerLocal 进入 L0B 的路径可能需要通过 L1/TSCM 中转。
- raw MMAD 输出 layout 与现有 GM ND layout 必须一致；若不一致，需要 FixPipe/DataCopy 参数做 NZ/ND 转换。
- 16×16 raw MMAD 的 fixed overhead 未必低于 high-level Matmul；最终仍需 msprof 验证 `aic_scalar/mte2/fixpipe` 是否下降。
