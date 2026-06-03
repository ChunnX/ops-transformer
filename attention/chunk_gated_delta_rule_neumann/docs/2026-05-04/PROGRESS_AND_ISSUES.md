# 2026-05-04 进度与问题

## 已完成

- 已核对当前真实状态：`C64_1616_FP32_CVInverse` raw inv-add 探针精度可过，但性能不可接受。
- 已确认 raw inv-add 微调探针结果：
  - `S=64` Neumann/normal `31.405x`
  - `S=512` Neumann/normal `129.931x`
  - 主问题不是 DCCI、identity 写入或单次 raw inv-add 公式形态，而是固定调度与 AIV scalar/wait 放大。
- 已将默认 selector 切回 `C64_3232_FP32_CVInverse`。
- 已补齐 `C64_3232_FP32_CVInverse` 的构造兼容层，适配主 kernel 当前统一传入的 FP16 Matmul 实例。
- 构建安装通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 当前默认实现全量三方精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.17s`
- 新增下一阶段计划：
  - `docs/2026-05-04/C64_3232_LOCAL_TSCM_POC_PLAN.md`
- 继续 local/TSCM PoC：
  - 修复 `C64_3232_FP16_LocalTscm_CVInverse` 构造函数参数不匹配问题，使其兼容主 kernel 当前 10 参数 `Stage1` 构造路径。
  - PoC 构建和安装通过。
  - PoC 全量三方精度未通过：`17 failed, 3 passed in 6.30s`。
  - 已将 selector 回退到稳定 `C64_3232_FP32_CVInverse`。
  - 稳定默认线重新构建安装后全量三方精度通过：`20 passed in 6.33s`。

## 进行中

- 默认实现：`C64_3232_NeumannCompose` hybrid。
- host 主 chunk：`64`。
- 当前实现仍是 `MatmulImpl + GM workspace` 串联，不是完整片上 local fused。
- `C64_1616` raw MMAD 代码保留，但降级为探针/参考线。
- `C64_3232_FP16_LocalTscm_CVInverse` 作为独立 PoC 文件保留，但 selector 默认不启用。

## 阻塞项

- 无精度阻塞项。
- 性能方向的阻塞项是结构性问题：
  - Neumann round 之间仍有 `L0C/Fixpipe -> GM -> LoadData` 往返。
  - `C64_1616` raw inv-add 微调未能降低 wall time。
  - FP32 TSCM local Matmul 在仓库内没有直接成熟样例；half/TSCM 有先例，适合作为第一阶段 PoC。
- `C64_3232_FP16_LocalTscm_CVInverse` 当前精度阻塞：
  - 构建可过，但 20 case compare 仅 3 case 通过。
  - 主要怀疑 64x64 NZ 大矩阵中的 32x32 子块 offset 与 local Matmul compact tile 布局不匹配。
  - 未达到 msprof probe 条件。

## 风险

- `C64_3232_FP32_CVInverse` 当前只是恢复默认稳定线，不是新性能优化完成。
- local/TSCM PoC 若走 half 路线，需要继续用 `atol=rtol=0.08` 做全量精度门槛。
- 若 PoC 只减少 GM 往返但仍保留大量 tiny Matmul 调用，端到端收益可能仍有限，必须用 msprof 验证。
- 不应把已失败的 `C64_3232_FP16_LocalTscm_CVInverse` 切为默认 selector；默认线必须保持 `C64_3232_FP32_CVInverse`，直到 PoC 全量 compare 通过。

## 下一动作

- 按 `C64_3232_LOCAL_TSCM_POC_PLAN.md` 继续，但顺序调整：
  1. 先验证 TSCM/NZ 子块布局，避免 64x64 NZ 大矩阵直接按 32x32 offset 被 local Matmul 误读。
  2. 优先复用已验证过的 `C64_1616_FP16_CVInverse` 16x16 local/TSCM 路线做 compact tile 对照。
  3. 再恢复 32x32 diagonal inverse 和 `Compose64From32()` 的片上中转。
  4. 全量精度通过后才跑 `S=64/S=512` 两 case msprof probe。
  5. probe 明显优于当前 `C64_3232_hybrid` 后再跑 32 shape 全量性能。

## 最新状态确认

- 当前稳定状态：默认 selector 应保持 `C64_3232_FP32_CVInverse`。
- 最新稳定验证：构建安装后 `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 为 `20 passed in 6.33s`。
- 当前未通过项：`C64_3232_FP16_LocalTscm_CVInverse` PoC，构建可过但 compare 为 `17 failed, 3 passed`。
- 当前下一步：先做 TSCM/NZ compact 子块布局验证，再继续 local/TSCM PoC。

## local/TSCM 累加定位

- `C64_3232_FP16_LocalTscm_CVInverse` PoC 的 local Matmul 封装在 `AICBlockProcessFp16L1()`。
- 当前 `INV += INV@Lpow` 的实际累加发生在 `AddLocalMatrix()`：
  - Matmul 先把乘积写到 TSCM local `power` 或 `scratch`。
  - `AddLocalMatrix()` 用 AIC 标量循环把乘积加回 `inv`。
- 当前 PoC 未使用 local Matmul atomic add 来完成 Neumann 累加。
- 已确认这不是 Matmul 结果矩阵累加；`enableAtomicAdd` 参数当前在 Neumann round 中未启用。
- 后续若继续优化，应优先考虑 `INV@(I+P)` 或验证 local Matmul C 输出累加，而不是保留逐元素标量加法。

## 底层 MMAD 累加方向

- 最终目标仍是底层 `Mmad` 的 C 矩阵累加，不是标量循环。
- 相关参数口径：
  - `MmadParams(..., cmatrixSource, cmatrixInitVal)`
  - 当前 raw helper 中 `initC` 对应 `cmatrixInitVal`。
  - `initC=true` 初始化 L0C，`initC=false` 继续在 L0C 上累加。
  - `cmatrixSource=true` 可从额外 C/bias tensor 作为 C 初值来源。
- 当前阶段允许先用标量循环实现精度正确版本并跑性能对比，但后续需要回到 L0C accumulator 或 C 初值来源方案。

## 既有 L0C 累加未完成原因

- 既有 raw MMAD 路线只验证了局部 `old_INV + INV@P` 可用，没有完成端到端 local/TSCM fused inverse。
- 主要缺口：
  - L0C 结果仍写回 GM 后再作为下一轮输入，未实现片上持续驻留。
  - final RHS 乘法仍依赖 `MatmulImpl` fallback，raw final 曾触发 out 精度失败。
  - `C64_1616` 子块调用数多，局部累加替换不能消除整体调度/同步/搬运开销。
  - `C64_3232_FP16_LocalTscm` 当前又有 compact TSCM/NZ 布局阻塞。
- 后续应先解决 local/TSCM compact tile 正确性，再把标量 `AddLocalMatrix()` 替换成底层 C 初值/累加。

## 标量前代正确基线

### 已完成

- 已确认 L1/TSCM 承接不是硬件限制；GM 只是当前验证路径。
- 已新增方案文档：
  - `docs/2026-05-04/C64_3232_SCALAR_FORWARD_BASELINE_PLAN.md`
- 已将 selector 切到稳定 `C64_3232_FP32_CVInverse`。
- 已在 `C64_3232_FP32_CVInverse` 中增加 `DEBUG_SCALAR_FORWARD_GM_INVERSE=true` 和 `C64InverseByScalarForwardGm()`。
- 已完成构建安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 已完成精度验证：
  - Neumann-only smoke 3 case PASS。
  - 全量三方 compare：`20 passed in 6.16s`。
- 已完成最小性能对比：
  - `tests/pytest/prof_out/chunk_gdr_c64_3232_scalar_forward_gm_smoke/performance_compare.csv`
  - `B=8,S=16` 标量 Neumann/normal `2.573x`
  - `B=8,S=64` 标量 Neumann/normal `23.856x`

### 当前结论

- 标量前代递推公式和接入点正确，但只是历史正确性基线。
- 标量版本性能不可接受，主耗时是 AIC scalar time。
- 当前默认实现已回到 CMatrix/MMAD 累加，不再使用标量前代作为主线。
- 当前 `C64_3232_FP32_CVInverse` 的 Neumann inverse `INV += INV@P` 由底层 `MmadParams(..., cmatrixSource=false, cmatrixInitVal=initC)` 控制：
  - `initC=true` 初始化 L0C；
  - `INV @ I` 将 old `INV` 种到 L0C；
  - 后续 `INV @ P` 使用 `initC=false` 直接累加到同一 L0C/CMatrix；
  - row tile 完成后再 `Fixpipe` 写回 `INVWsGm_`。

### 进行中

- 当前默认 `C64_3232_FP32_CVInverse` 已完成 Neumann inverse 乘加的 CMatrix/MMAD 累加正确性验证。
- 后续继续围绕 local/TSCM resident inverse 推进，但先解决 layout 正确性。
- 若继续减少 GM 往返，下一步目标是 `L0C accumulator -> Fixpipe -> L1/TSCM -> LoadData`，或进一步验证 `cmatrixSource=true` 的 C 初值来源。

### 阻塞项

- `L0C` 结果不能直接作为下一轮 A/B，需要 `Fixpipe -> L1/TSCM/VECCALC -> LoadData`。
- Fixpipe 到 L1/TSCM 后的 compact NZ/stride/layout 尚未验证清楚。
- `C64_3232_FP16_LocalTscm_CVInverse` 仍未通过全量精度，不能进入正式性能 probe。
- 本轮没有重新跑 msprof，当前只确认 CMatrix/MMAD 累加后的精度正确。

### 下一动作

1. 保留标量 GM 前代结果作为历史公式基线，不作为默认路径。
2. 继续验证 local/TSCM resident 的数据驻留和 tile layout。
3. 将 local/TSCM PoC 中的 `AddLocalMatrix()` 替换为 `cmatrixInitVal=false` 或 `cmatrixSource=true` 的 MMAD 累加前，先保证 compact tile 读取正确。
4. 只在全量 compare 通过后运行 msprof。

## 最新验证：CMatrix/MMAD 累加默认线

### 已完成

- 当前 selector 保持默认 `C64_3232_FP32_CVInverse`。
- 已从当前默认文件移除标量前代残留：
  - `DEBUG_SCALAR_FORWARD_GM_INVERSE`
  - `C64InverseByScalarForwardGm()`
  - 未使用的 `rawInvAddScratchGm_`
  - 未使用的 `RAW_LOAD3DV2_NO_FMATRIX_CONFIG` 与 `CeilDivLocal()`
- `RawMmadInvAddOutputTile()` 现在使用同一 L0C tile 完成 `old_INV + INV@P`：
  - 第一拍 `INV @ I`，`initC=true`
  - 后续 `INV @ P`，`initC=false`
  - `StoreRawMmadInvAddRow()` 在整行两个 16x16 输出 tile 都算完后再写回，避免 in-place 覆盖 hazard。
- 构建安装通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 精度验证通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.22s`

### 当前状态

- Neumann inverse 的矩阵乘累加已经由 CMatrix/MMAD 控制，不是标量循环累加，也不是 Fixpipe atomic add 累加。
- `INV @ k_cumdecay` / `INV @ v_beta` final RHS 乘法仍走 FP32 GM `MatmulImpl`，本轮没有继续 raw final。
- 最新用户状态确认：默认实现可编译、可安装、全量 compare 通过；尚未重新跑性能 profiling。

## 最新状态：lower-triangular power 固化

### 已完成

- 复跑默认 `C64_3232_FP32_CVInverse` CMatrix/MMAD 累加线，全量 compare 通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.19s`
- 完成 lower-triangular power smoke：
  - `tests/pytest/prof_out/chunk_gdr_c64_3232_lower_tri_power_smoke/performance_compare.csv`
  - `B=8,S=16` Neumann/normal `1.323x`
  - `B=8,S=64` Neumann/normal `1.664x`
- 验证并回退两个不稳定探针：
  - 全 tile power 会导致部分 case 出现 `1e-2` 级 out 漂移。
  - `INV@P` k-range 裁剪也会导致部分 case 出现 `1e-2` 级 out 漂移。
- 最终源码保持：
  - `RawMmadLowerTriangularBlock()` 计算 power，并显式清零上三角。
  - `RawMmadInvAddBlock()` 全 tile 执行 `INV@I + INV@P`，通过 `cmatrixInitVal=false` 在 L0C/CMatrix 内累加。
  - final RHS 仍使用 `MatmulImpl`。
- 最终重建、重装、全量三方精度通过：
  - 构建：`bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - 安装：`bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
  - 精度：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 结果：`20 passed in 6.15s`

### 当前结论

- `INV += INV@P` 已按用户要求由底层 CMatrix/MMAD 控制，不依赖标量循环或 Fixpipe atomic add。
- power 的下三角约束是精度必要条件，不能为了少量性能回退到未清零上三角的全 tile power。
- `INV@P` 的 k-range 裁剪当前不可靠，后续若要优化，需先构造更严格的 tile 数值校验，而不是直接进默认线。

### 下一动作

1. 若继续优化当前稳定线，优先做单独 raw tile 数值校验，验证 `power` 上三角清零和 `INV@P` CMatrix 累加每个 tile 的数值。
2. local/TSCM resident 继续前，先解决 compact NZ/stride/layout，再考虑把 `Fixpipe -> GM` 换成 `Fixpipe -> L1/TSCM`。
3. raw final `INV @ v_beta` 仍是后续项，不混入当前 Neumann inverse 矩阵生成验收。

## 最新收口：CMatrix 参数语义命名

### 已完成

- 当前默认 `C64_3232_FP32_CVInverse` 继续保持 `RawMmadLowerTriangularBlock()` power + `RawMmadInvAddBlock()` 全 tile inv-add 策略。
- 代码中新增 CMatrix 语义常量：
  - `RAW_MMAD_CMATRIX_SOURCE_L0C=false`
  - `RAW_MMAD_INIT_C=true`
  - `RAW_MMAD_ACCUMULATE_C=false`
- `MakeRawMmadParams()` 的布尔参数从 `initC` 收口为 `initCMatrix`，明确对应 `MmadParams(..., cmatrixSource, cmatrixInitVal)` 的 `cmatrixInitVal`。
- `RawMmadInvAddOutputTile()` 注释已明确当前实现：
  - `old_INV @ I` 初始化 L0C。
  - `old_INV @ power` 用 `cmatrixInitVal=false` 累加到同一 CMatrix。
  - Fixpipe 只负责把 CMatrix 结果写回 `INVWsGm_`，不承担加法。

### 最新验证

- 构建通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 安装通过：
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 全量三方精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.14s`

### 当前状态

- `INV += INV@P` 的主线已稳定为底层 CMatrix/MMAD 累加。
- full-tile power 和 `INV@P` k-range 裁剪仍保持回退状态，原因是此前精度探针出现 `1e-2` 级 out 漂移。
- final RHS 仍是后续独立问题；当前验收范围是 Neumann inverse 矩阵生成。
