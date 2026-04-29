# C64 Neumann 求逆两方案实施计划 - 2026-04-29

**目标**: 将 `chunk_gated_delta_rule_neumann` 从当前 C16 对比版本扩展到 C64，对齐原 `chunk_gated_delta_rule` 的 chunk=64，同时比较两种 FP32 CVInverse 路径。

---

## 前置约束

1. 先完成性能基线保存：`tests/pytest/prof_out/chunk_gdr_prefill_full/performance_compare.csv` 与 `performance_raw.csv` 已生成。
2. 当前 stage selector 为 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`，通过 include 切换不同 stage1 实现。
3. 两个目标文件：
   - 方案1：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`
   - 方案2：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`
4. 当前两个 C64 文件看起来仍是从 C16 文件复制而来，宏 guard、`INVERSE_SHAPE=16` 等标识需要系统改造。

---

## 总体步骤

### Step 1: 切 chunk 到 64

- 修改 host tiling：`op_host/chunk_gated_delta_rule_neumann_tiling.cpp` 中 `chunkSize`/`c` 改为 64。
- 同步检查 workspace 公式：所有 `chunkSize * chunkSize` 的 mask、stage workspace、inter workspace 需要能覆盖 64×64。
- Golden/pytest 若用于 C64 精度，需要确认 golden 的 chunk_size 与当前算子一致；CPU golden 逻辑不改，只改测试参数或已有 chunk 配置。

### Step 2: 建立 stage1 selector 切换

- `chunk_gated_delta_rule_neumann_stage1.h` 增加明确注释和三选一 include：
  - `C16_FP16_CVInverse`：当前可运行性能基线。
  - `C64_1616_FP32_CVInverse`：方案1。
  - `C64_3232_FP32_CVInverse`：方案2。
- 每次只 include 一个目标文件编译，保证“换文件编译即可实现不同算子效果”。

---

## 方案1：C64 + 16×16 Neumann 分块求 32×32，再求 FP32 逆

**目标文件**: `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`

### 数学结构

将 32×32 下三角块矩阵写成：

```text
M32 = [ A  0 ]
      [ C  D ]
```

其中：
- `A` 是左上 16×16。
- `D` 是右下 16×16。
- `C` 是左下 16×16。
- 上右块为 0，因为矩阵下三角。

逆矩阵为：

```text
M32^{-1} = [ A^{-1}                 0 ]
           [ -D^{-1} * C * A^{-1}   D^{-1} ]
```

### 实现路径

1. 保持 AIC 16×16 Neumann 子程序作为基本块：分别计算 `A^{-1}` 与 `D^{-1}`。
2. 新增 AIC matmul 计算左下块：
   - `T = C * A^{-1}`
   - `LD = -D^{-1} * T`
3. 组合得到 32×32 inverse：左上 `A^{-1}`，右下 `D^{-1}`，左下 `LD`，右上清零。
4. 对 64×64 的完整下三角，可先构造四个 32×32 块：

```text
M64 = [ P  0 ]
      [ R  Q ]
```

再用同样 block inverse 公式得到 64×64 inverse。
5. 最终继续沿用原有 FP32 inverse 后续流程：`INV @ k_cumdecay`、`INV @ v_beta`、`qPrime` 等保持一致。

### 关键改造点

- `INVERSE_SHAPE` 不能再简单等于 16；需要区分：
  - `BLOCK16 = 16`
  - `BLOCK32 = 32`
  - `CHUNK64 = 64`
- workspace 需要至少能放：
  - 16×16 子逆 `AInv/DInv`
  - 16×16/32×32 临时 matmul `tmp`
  - 32×32 中间逆
  - 64×64 最终 `INVWsGm_`
- AIV 当前半块逻辑 `halfChunkSize_=chunkSize_/TASK_RATIO=32` 可继续用于 64 chunk 的 vector 分工，但 16×16 子块求逆最好由 AIC 完成，减少 AIV 参与。

### 风险

- 分块组合需要严格处理 tail chunk：validLen < 64 时，上述块公式需要 mask 或退化处理。
- 当前 stage1 workspace 映射按 `chunkSize * chunkSize` 分配，对 64×64 增加 16 倍 C16 面积，需确认 `stageWorkspaceSz` 充足。
- 多次小矩阵 matmul 调度开销可能较大，但比 32×32 直接 Neumann 展开更可控。

---

## 方案2：C64 + 32×32 Neumann 直接展开求逆

**目标文件**: `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_CVInverse.h`

### 数学结构

32×32 严格下三角 `L` 满足 `L^32=0`，逆为：

```text
(I + L)^{-1} = Σ_{k=0}^{31} (-L)^k
```

倍增法可用 5 轮覆盖：

```text
INV0 = I - L
L2  = L  * L     ; INV1 = INV0 + INV0 * L2
L4  = L2 * L2    ; INV2 = INV1 + INV1 * L4
L8  = L4 * L4    ; INV3 = INV2 + INV2 * L8
L16 = L8 * L8    ; INV4 = INV3 + INV3 * L16
```

对 64×64 可继续用 32×32 block inverse 组合。

### 实现路径

1. 将 C16 Neumann 子程序泛化为 32×32：
   - `INVERSE_SHAPE = 32`
   - 每轮 matmul 使用 32×32。
   - 轮数从 3 轮增加到 4 轮，用 `L²/L⁴/L⁸/L¹⁶`。
2. 对 chunk=64，按 32×32 左上/右下直接 Neumann，左下按 block inverse 公式组合。
3. 或进一步尝试 64×64 直接 Neumann：需要 6 轮 `L²...L³²`，但不作为当前优先方案，调度和 workspace 风险更高。

### 关键改造点

- `INVERSE_SHAPE=32` 后，UB/GM 临时 buffer、`colBuffer_`、`inverseBufferPerBlock` 都要扩大。
- AIC matmul tiling 当前是否支持 32×32 小矩阵稳定高效，需要编译和 msprof 验证。
- `NeumannAddCompute()` 如果仍走 AIV，向 32×32 扩展会增加 vector 搬运；建议优先保持现有路径通过，再考虑 AIC atomic add 优化。

### 风险

- 32×32 直接 Neumann 需要更多小矩阵 matmul，单 chunk 内计算量和同步次数明显增加。
- 相比方案1，数值误差传播路径更长，需要三方精度 case 扩展到 chunk=64 后验证。
- 若 workspace/UB 不足，可能需要拆分计算或复用临时区。

---

## 验证计划

### 编译验证

每个方案单独切 selector 后：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=${ASCEND_HOME_PATH}/opp
```

### 精度验证

先小 case，再扩大：

```bash
cd attention/chunk_gated_delta_rule_neumann/tests/pytest
bash test_run.sh compare
```

必要时新增 C64 专用三方精度 paramset，覆盖：
- exact chunk: `S=64`
- tail: `S=65/127`
- multi chunk: `S=128/256`
- GQA: `Hq/Hkv=4`
- `D=64/128`

### 性能验证

对每个方案复用已建 msprof 框架：

```bash
cd attention/chunk_gated_delta_rule_neumann/tests/pytest
python3 run_chunk_gdr_performance_msprof.py --output ./prof_out/chunk_gdr_neumann_c64_方案名
```

对比文件：
- baseline: `prof_out/chunk_gdr_prefill_full/performance_compare.csv`
- 方案1: `prof_out/chunk_gdr_neumann_c64_1616/performance_compare.csv`
- 方案2: `prof_out/chunk_gdr_neumann_c64_3232/performance_compare.csv`

---

## 当前优先级

1. 已完成并保存当前 C16/现状性能基线。
2. 下一步先实现方案1，因为 16×16 Neumann 子块已有可运行基础，风险低于直接 32×32 展开。
3. 方案1通过编译和小 case 精度后，再实现方案2，避免同时改两套导致问题不可定位。
