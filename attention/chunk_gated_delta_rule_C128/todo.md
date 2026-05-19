# C128 当前 TODO

更新时间：2026-05-19 12:05 CST

## 当前目标

N9.12.5c 已切到 packed128 tail，当前收尾清理新 dead 的 packed64 tail
helper；随后重跑 30-case gate，并补一轮 C128 vs C64 Neumann profiling：

- full chunk inverse 在 `cInv=aicL0C_` 完成，最后只写入 `aicBf16L1ResidentInv128_`
  给 RHS 消费。
- tail 已去掉 `attnWsGm_` BF16 inverse GM 中转；当前 full/tail 均走
  `C128InverseBy16x16Neumann()` 的 packed128 求逆路径。
- tail 数学上仍是 C128，不再进入 packed64 tail materialize/RHS 路径；invalid
  区继续由 AIV padding / valid rows 管理，不引入 `zero64/identity64` 兜底。
- 旧 per-16x16 helper 要清理；但 active `ComposeOddEvenLower16To32AccumFull128()`
  是 full128 递归的 `16 -> 32` 第一层 wrapper，不能作为 dead code 删除。
- `INVInitHalfWsGm_` / `L_powHalfWsGm_` 仍是 AIV->AIC half 输入。
- 每步继续单点修改，先过精度 gate，再看 profiling。

## 当前基线

- 工作目录：`/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128`
- active Stage1：`op_kernel/chunk_gated_delta_rule_C128_stage1.h`
- active Stage1 SHA256：
  `9192f2e0d0a0c5d801c254e5b6c30abc4201df499c4ed96e253051c77f11e065`
- debug 状态：
  - `C128_FULL128_DEBUG_STAGE = 0`
  - active kernel 中未检出 `C128_DEBUG_RETURN_POINT`
- N9.12.1 当前状态：
  - full chunk 已删除 `attnWsGm_` BF16 inverse 写回。
  - tail 仍保留 `InverseComputeTail(... attnWsGm_ ...)`，因为当前 tail top/bottom 还从 `attnWsGm_` 读 inverse。
  - `S=128` smoke PASS。
  - `S=64/80/96/112/128/256/1024` ladder PASS。
  - 30-case CPU golden PASS：`30 passed, 2 warnings in 7.91s`。
- N9.12.1 精度验证包：
  `/tmp/cgdr_c128_pkg_n9_12_1_tailfix_20260518_2050`
  - package SHA256：`392d0b3879369bf42579d390efb84dc171eb10336eb3ec4dd9e448546a9d4e28`
  - installed `libcust_opapi.so` SHA256：`e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`
  - normal object SHA256：`70c293ec3247e91eedadd93cf6a101a4d2136d6540726da1eae4320122f1d79c`
  - C128 object SHA256：`187fa2a73bf50392fda1593b7b86ff8382a3c89815791d32a35bd178e8c7031c`
- N9.12.1 perf smoke：
  - 输出目录：`tests/pytest/prof_out/c128_n9_12_1_full_attnws_drop_smoke_20260518/`
  - `S=1024` C128：`2489.8 us`，相对 N9.8 full128 `2908.3 us` 快约 `14.4%`
  - `S=2048` C128：`4327.88 us`，相对 N9.8 full128 `5064.5 us` 快约 `14.5%`
- N9.12.1 C128 vs C64 Neumann profiling：
  - 输出目录：`tests/pytest/prof_out/neumann_vs_c128_n9_12_1_20260518/`
  - 对比表：`tests/pytest/prof_out/neumann_vs_c128_n9_12_1_20260518/c128_vs_neumann_compare.csv`
  - 总结：8 个 prefill 点整体基本持平；C128 赢 `3/8`。
  - 简单平均 C128 慢约 `0.76%`。
  - 总 task duration 加权 C128 快约 `0.34%`。
- N9.12.1 profiling 包：
  `/tmp/cgdr_neumann_c128_pkg_n9_12_1_20260518_2058`
  - installed `libcust_opapi.so` SHA256：`552341a934dde044b5d568d1c951cdc9d3e905872d4adb5bf9d2be8c14bf772f`
  - C128 object SHA256：`187fa2a73bf50392fda1593b7b86ff8382a3c89815791d32a35bd178e8c7031c`
  - C64 Neumann object SHA256：`6094cf4767c9c68bec72fbb2bae9f7d219d041e87603350a8f27b6b2a7ad8ac6`
- N9.11 验证状态：
  - `S=64/80/96/112/128/256/1024` ladder PASS
  - 30-case CPU golden PASS：`30 passed, 2 warnings in 8.32s`
- N9.11 验证包：
  `/tmp/cgdr_c128_pkg_n9_11_rhsdb_20260518_1927`
- C128 vs C64 Neumann 性能对比：
  - 输出目录：`tests/pytest/prof_out/neumann_vs_c128_n9_11_20260518/`
  - 8 个 prefill 点整体基本持平
  - 简单平均 C128 慢约 `1.9%`
  - 总 task duration 加权 C128 慢约 `0.8%`
- N9.12.3 当前状态：
  - tail `attnWsGm_` BF16 inverse 中转已删除。
  - AIV 只写 `L_powHalfWsGm_` / `INVInitHalfWsGm_`。
  - `MaterializeTailC128DiagonalResident()` 把 tail top/bottom half inverse 经 AIC 写入 BF16 resident。
  - `ComposeTailC128LowerLeftResident()` 保持 lower-left resident 直写。
  - `TailRhsC128AICProcess()` 只读 resident，不再从 `attnWsGm_` 读 inverse。
  - `S=64/80/96/112/128/160/256/1024` ladder PASS。
  - 30-case CPU golden PASS：`30 passed, 2 warnings in 14.45s`。
- N9.12.3 验证包：
  `/tmp/cgdr_c128_pkg_n9_12_3_tail_resident_20260519_1052`
  - package SHA256：`ff3e686a157b44dc420ec945fcfe01fed7259a1adbc19b1553b217843b97dd3b`
  - installed `libcust_opapi.so` SHA256：`e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`
  - C128 object SHA256：`9635b3b8fc064b28647c5981074e6bbb20e585050f9598344cb554918ca2700e`
  - normal object SHA256：`70c293ec3247e91eedadd93cf6a101a4d2136d6540726da1eae4320122f1d79c`

## 当前 active 数据流

full chunk：

1. AIV 生成 half 中间量：
   - `L_powHalfWsGm_`
   - `INVInitHalfWsGm_`
2. AIC full128 求逆：
   - `LoadPacked128InputsToL1()`
   - `InvertPacked128DiagonalNeumannToResult()`
   - `ComposeOddEvenLower16To32AccumFull128()`
   - `ComposeOddEvenLower32To64AccumFull128()`
   - `ComposeOddEvenLower64To128AccumFull128()`
3. final inverse materialize：
   - `StorePacked128InverseToFullResident()`
   - `FP32 L0C -> BF16 L1 resident`
   - resident 为 `aicBf16L1ResidentInv128_`
4. N9.12.1 后不再写 full chunk `attnWsGm_` BF16 inverse。
5. RHS：
   - `FusedRhsC128AICProcess()`
   - 消费 BF16 RHS：`gBKWsGm_` / `vBetaWsGm_`
   - 消费 inverse resident：`aicBf16L1ResidentInv128_`

tail：

1. AIV 仍生成 `L_powHalfWsGm_` / `INVInitHalfWsGm_`，并在 `InverseCompute()`
   内对 invalid row/col 执行 tail padding。
2. AIC tail 与 full 共用 packed128 路径：
   - `LoadPacked128InputsToL1()`
   - `InvertPacked128DiagonalNeumannToResult()`
   - `ComposeOddEvenLower16To32AccumFull128()`
   - `ComposeOddEvenLower32To64AccumFull128()`
   - `ComposeOddEvenLower64To128AccumFull128()`
   - `StorePacked128InverseToFullResident()`
3. `TailRhsC128AICProcess()` 通过 `ComputeRhsPacked128Bf16Tiles()` 消费
   `aicBf16L1ResidentInv128_`，不再消费 packed64 resident 子视图。

## 已确认冗余

- [x] R1 full chunk 曾写 `attnWsGm_` BF16 inverse，N9.12.1 已删除。
  - 位置：`InverseCompute()`
  - 旧现状：`inverseRes_ -> bfloat16_t tmp -> DataCopyOutBf16(... dst ...)`
  - 当前：full branch 调 `InverseCompute(...)` 不再传 `attnWsGm_`，tail branch 走 `InverseComputeTail(...)` 保留旧写回。
  - 判断：full chunk 后续不读这份 BF16 GM inverse，是最明确 runtime 冗余。
- [x] R2 tail lower-left 仍有 GM roundtrip。
  - 位置：`ComposeTailC128LowerLeftGm()` / `TailRhsC128AICProcess()`
  - 现状：已在 N9.12.2 改成 resident 版本，lower-left 不再绕 GM 回读。
  - 判断：tail 有效路径，但 GM roundtrip 冗余已清掉。
- [x] R3 full128 旧 compose helper 仍在。
  - `ComposeOddEvenLowerLevelAccumFull128()`
  - `ComposeOddEvenLowerLevelCompactFull128()`
  - 当前 wrapper 调用 `ComposeOddEvenLowerLevelFull128()`。
  - 判断：维护冗余，不影响 runtime，但应清理。
  - 注意：active `ComposeOddEvenLower16To32AccumFull128()` / `32To64` / `64To128`
    wrappers 仍是 full128 递归入口，不是 dead helper。
- [x] R4 `PrepareRawMmadAuxiliaryTiles()` 无条件准备 C64 和 C128 aux。
  - full-only 路径只需要 C128 zero/identity。
  - C64 zero/identity 只是当前 tail packed64 helper / 旧 sub64 helper 的实现残留；
    tail 数学上仍是 C128，invalid 区应由 AIV padding 处理，不能把
    “tail 需要 C64 aux”写成目标。
- [x] R5 `LoadPacked128InputsToL1()` 每次整块清 128 scratch/power。
  - 当前属于安全 padding，不先删。
  - 参考 C64 Neumann：`aicHalfL1Zero_` / identity 这类常量 aux 在
    `InitRawMmadLocalBuffers()` 周期准备；`tileInv/tilePower/scratch`
    这类工作区按块/按阶段清零。C128 需要先区分“常量 aux 清零”和
    “每 chunk 工作区清零”，再动代码。
- [x] R6 full-only 仍保留 tail resident buffer footprint。
  - `aicBf16L1ResidentInv_` 三个 64 tile 主要服务 tail。
  - `aicBf16L1ResidentInv128_` 服务 full chunk。
  - buffer layout 放到 tail pure AIC 后再整理。

## 执行队列

### N9.12.0 冻结和复核

- [x] 记录修改前 SHA。
- [x] `rg -n "C128_DEBUG_RETURN_POINT|C128_FULL128_DEBUG_STAGE" -S .`
- [x] 确认修改前 active Stage1 是：
  `8674d94f90508d727b3d0190d9e372eae9f1b8f3812baac140cf877ce92316df`
- [x] 不改代码，只确认环境和入口。

### N9.12.1 删除 full chunk `attnWsGm_` BF16 inverse 写回

修改范围：

- [x] `InverseCompute()` 去掉 `dst` 参数。
- [x] full chunk 调用点不再传 `attnWsGm_[i * ccOffset_]`。
- [x] 删除 full chunk `inverseRes_ -> bfloat16_t tmp` cast。
- [x] 删除 full chunk `DataCopyOutBf16(... dst[subOffset_ * chunkSize_])`。
- [x] tail 调用点改为 `InverseComputeTail(...)`，保留当前 tail BF16 GM 写回。
- [x] 保留 half 写回：
  - `L_powHalfWsGm_`
  - `INVInitHalfWsGm_`

验证 gate：

- [x] build package
- [x] CLI smoke：`S=128`
- [x] ladder：`S=64/80/96/112/128/256/1024`
- [x] 30-case CPU golden
- [x] perf smoke：`S=1024/2048`，对比 N9.8/N9.10
- [x] 追加 C128 vs C64 Neumann 8-shape profiling，结果已落盘

禁止事项：

- 未改 tail resident/RHS 数据流。
- 未改 aux 初始化。
- 未删 dead helper。

### N9.12.2 tail lower-left GM roundtrip 改 resident

修改范围：

- [x] 将 `ComposeTailC128LowerLeftGm()` 改成 resident 版本，建议命名：
  `ComposeTailC128LowerLeftResident()`
- [x] lower-left 输出从 `cOut=aicL0C_[C64_L0C_TILE_ELEMS]` 直接写：
  `aicBf16L1ResidentInv_[2 * AIC_LOW_L1_TILE_ELEMS]`
- [x] 使用 `FixpipePacked64FullToBf16L1(...)`
- [x] `TailRhsC128AICProcess()` 不再从 `attnWsGm_` 读 lower-left。
- [x] top/bottom tail 暂时仍按现状从 `attnWsGm_` 读，避免一次跨到 pure AIC。

验证 gate：

- [x] tail no-bottom：`S=64`
- [x] tail lower-left：`S=80/96/112`
- [x] full boundary：`S=128`
- [x] full+tail mixed：至少覆盖 `S=160` 或 paramset 中等价 case
- [x] 30-case CPU golden

验证结果：

- `S=64/80/96/112/128/160` PASS
- 30-case CPU golden PASS：`30 passed in 7.76s`

### N9.12.3 tail pure AIC resident inverse

目标：

- tail top/bottom/lower-left 都由 AIC 生成 resident。
- RHS 只消费 resident。
- tail 不再依赖 `attnWsGm_` 作为 inverse GM 中转。

拆分：

- [x] `tailLen <= 64`：只生成 top resident。
- [x] `tailLen > 64`：生成 top resident、bottom resident、lower-left resident。
- [x] lower-left compose 直接 resident，不落 GM。
- [x] tail 数据流确认：
  - `INVInitHalfWsGm_` / `L_powHalfWsGm_` 仍作为 AIV->AIC half 输入保留。
  - `attnWsGm_` 只清 tail inverse 的 BF16 GM 中转，不承担数学输入职责。
- [x] 实现边界：
  - tail 数学上仍按 C128 lower-triangular 矩阵处理。
  - 不把 `zero64/identity64` 作为 tail 必需数学前提。
  - 无效 row/col 继续依赖 AIV padding / valid row 控制，不靠 C64 identity 兜底。
  - 可以临时复用 64 tile resident 作为 C128 的 top/bottom/lower-left 子视图，但不能把它写成最终目标。

验证 gate：

- [x] `S=64/80/96/112`
- [x] `S=128`
- [x] full+tail mixed shapes：`S=160/256/1024`
- [x] 30-case CPU golden
- [ ] tail 性能对比 N9.11

验证结果：

- `S=64/80/96/112/128/160/256/1024` PASS
- 30-case CPU golden PASS：`30 passed, 2 warnings in 14.45s`

### N9.12.5 C64 aux 初始化延后

- [x] 拆分 raw-MMAD aux 初始化，不把 `zero64/identity64` 视为 tail 必需项。
  - full128 热路径不预备 `zero64/identity64`。
  - `zero64/identity64` 只在仍调用 packed64 tail helper 或旧 sub64 helper 时 lazy init。
  - tail 的 invalid 区按 AIV `Duplicate(0)` / tail valid row 控制处理，不用 C64 identity 兜底。
  - 参考 C64 Neumann：`validLen < chunkSize` 的无效区由 AIV padding
    `Duplicate(0)` 处理，不靠额外 C64 identity 兜底。
- [ ] full-only perf smoke：`S=1024/2048`
- [x] tail smoke：`S=64/80/112`
- [x] correctness smoke：`S=1024/2048`
- [x] 30-case CPU golden
- [x] full-only perf smoke：`S=1024/2048`

验证中间结果：

- active Stage1 SHA256：
  `2d3490b8ff7402ab4cb2c7dd056cbbef79519e4c2222948711c441803b6a0021`
- 验证包：`/tmp/cgdr_c128_pkg_n9_12_5_aux_lazy_20260519_1102`
  - package SHA256：
    `d5a31107a771422786f22ff72080054425fe0d5f521dda1ccdf1198199db3564`
  - installed `libcust_opapi.so` SHA256：
    `e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`
  - C128 object SHA256：
    `1c20cfe586ea2c828f412d6e2abceaec9096e469800431501bdcb63c395cf063`
  - normal object SHA256：
    `70c293ec3247e91eedadd93cf6a101a4d2136d6540726da1eae4320122f1d79c`
- correctness smoke PASS：`S=64/80/112/1024/2048`
- 30-case CPU golden PASS：`30 passed, 2 warnings in 7.84s`
- perf smoke 输出目录：
  `tests/pytest/prof_out/c128_n9_12_5_aux_lazy_smoke_20260519/`
  - `S=1024` C128 measurement：`2453.06 us`
  - `S=2048` C128 measurement：`4275.82 us`

### N9.12.5b dead helper 清理

- [x] 删除旧 per-16x16 compose fallback：
  - `ComposeOddEvenLowerLevelAccumFull128()`
  - `ComposeOddEvenLowerLevelCompactFull128()`
  - 仅由上述 fallback 使用的 `IssuePacked16MmadOperands()` /
    `MmadPacked16BlocksToC()` 等 helper
- [x] 删除旧 `identity64 @ subblock -> bridge` fallback helper：
  - `LoadIdentity64Packed128Sub64OperandsToL0()`
  - `IssueIdentity64Packed128Sub64ToL0Db()`
  - `RawMmadIdentity64Packed128Sub64()`
- [x] 保留 active full128 wrappers：
  - `ComposeOddEvenLower16To32AccumFull128()`
  - `ComposeOddEvenLower32To64AccumFull128()`
  - `ComposeOddEvenLower64To128AccumFull128()`
- [x] build package
- [x] `S=64/80/112/128/1024`
- [x] 30-case CPU golden

验证结果：

- active Stage1 SHA256：
  `612d67a7b56d949f793d08bc369a467c60332ab545a5c387cb5c0539b62fb9d7`
- 旧 helper 静态检索 0 命中：
  `LoadPacked16PairToL0OffsetWithGap|RawMmadIdentity64Packed128Sub64|...`
- 验证包：`/tmp/cgdr_c128_pkg_n9_12_5b_dead_cleanup_20260519_1131`
  - package SHA256：
    `e9f436179e2a222874e07026b559dd862646f51b62fc5e884adb2ec3c1903fb9`
  - installed `libcust_opapi.so` SHA256：
    `e3c6abb7f726faea44eec76bbed613d25ffbca7893599feab3b9f9e96cf5f883`
  - C128 object SHA256：
    `1c20cfe586ea2c828f412d6e2abceaec9096e469800431501bdcb63c395cf063`
  - normal object SHA256：
    `70c293ec3247e91eedadd93cf6a101a4d2136d6540726da1eae4320122f1d79c`
- smoke PASS：`S=64/80/112/128/1024`
- 30-case CPU golden PASS：`30 passed, 2 warnings in 7.79s`

### N9.12.5c tail packed128 化

- [x] 将 tail inverse materialize 接到 full128 同一套 packed128 路径：
  `LoadPacked128InputsToL1()` -> `InvertPacked128DiagonalNeumannToResult()` ->
  `16->32->64->128 compose` -> `StorePacked128InverseToFullResident()`。
- [x] 将 tail RHS 改为 `ComputeRhsPacked128Bf16Tiles()` 消费
  `aicBf16L1ResidentInv128_`。
- [x] tail invalid 区继续由 AIV padding / valid row 控制，不引入 C64 identity 兜底。
- [ ] 清理新 dead 的 packed64 tail materialize/RHS helper：
  - `MaterializeTailC128DiagonalResident()`
  - `ComposeTailC128LowerLeftResident()`
  - `C64ValidLen()`
  - `ComputeRhsPacked64Bf16Tiles()`
  - 仅由上述 helper 使用的 packed64 RHS/materialize 局部 helper 与 64 resident buffer
- [ ] 单独 gate，不与 full128 清零周期同批。

已通过的 packed128 tail smoke：

- `S=64/80/96/112/128/160/256/1024` PASS

本轮清理后仍需重新验证：

- [ ] build package
- [ ] tail ladder：`S=64/80/96/112/128/160/256/1024`
- [ ] 30-case CPU golden
- [ ] C128 vs C64 Neumann profiling，并把结果落回本文件

### N9.12.6 full128 清零周期评估

- [ ] 暂不删除整块清零。
- [ ] 先按 C64 Neumann 的清零周期分类：
  - 常量 aux：`zero/identity` 只在 `InitRawMmadLocalBuffers()` 生命周期准备，
    不随每个 chunk 重复构造。
  - 工作区：`scratch/power/tileInv` 按块/按阶段清零，保证 padding 和旧值不串扰。
- [ ] 对 C128 `LoadPacked128InputsToL1()` 的
  `aicHalfL1Scratch128_` / `aicHalfL1TilePower128_` 整块清零做 msprof 评估：
  只有 MTE1/初始化开销明显时才进入代码修改。
- [ ] 若要改，只允许单独建 rollback 点，并先尝试把清零周期从“每次整块”
  收窄到“常量 aux 生命周期一次 + 工作区必要区域”，不直接删除安全 padding。

### N9.12.7 resident footprint 重整

- [ ] tail pure AIC 后再判断是否把三块 64 resident 复用为 128 resident 子视图。
- [ ] 不与正确性改动同批。

## 验证命令

构建：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_C128 -j16
```

单 shape CPU golden：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128
python3 tests/pytest/compare_chunk_gdr_C128_cpu_golden.py \
  --batch-size 1 --seqlen 128 --nk 4 --nv 4 --dk 128 --dv 128 \
  --atol 0.005 --rtol 0.005
```

tail ladder：

```bash
for S in 64 80 96 112 128 256 1024; do
  python3 tests/pytest/compare_chunk_gdr_C128_cpu_golden.py \
    --batch-size 1 --seqlen ${S} --nk 4 --nv 4 --dk 128 --dv 128 \
    --atol 0.005 --rtol 0.005 || exit 1
done
```

30-case CPU golden：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128
python3 -m pytest -rA -s tests/pytest/compare_chunk_gdr_C128_cpu_golden.py -v -m ci
```

性能脚本入口：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128
python3 tests/pytest/run_chunk_gdr_C128_performance_msprof.py --help
python3 ../chunk_gated_delta_rule_neumann/tests/pytest/run_chunk_gdr_performance_msprof.py --help
```

N9.12.1 perf smoke：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128
env TORCH_EXTENSIONS_DIR=/tmp/cgdr_c128_torch_ext_n9_12_1 \
  ASCEND_CUSTOM_OPP_PATH=/tmp/cgdr_c128_pkg_n9_12_1_tailfix_20260518_2050/vendors/custom_transformer \
  LD_LIBRARY_PATH=/tmp/cgdr_c128_pkg_n9_12_1_tailfix_20260518_2050/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} \
  python3 tests/pytest/run_chunk_gdr_C128_performance_msprof.py \
    --output tests/pytest/prof_out/c128_n9_12_1_full_attnws_drop_smoke_20260518 \
    --case prefill_b8_s1024_d128_h16_hkv4 \
    --case prefill_b8_s2048_d128_h16_hkv4 \
    --ops both
```

C128 vs C64 Neumann profiling：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 \
  --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann,chunk_gated_delta_rule_C128 -j16
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet \
  --install-path=/tmp/cgdr_neumann_c128_pkg_n9_12_1_20260518_2058

cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128
env TORCH_EXTENSIONS_DIR=/tmp/cgdr_c128_torch_ext_n9_12_1 \
  ASCEND_CUSTOM_OPP_PATH=/tmp/cgdr_neumann_c128_pkg_n9_12_1_20260518_2058/vendors/custom_transformer \
  LD_LIBRARY_PATH=/tmp/cgdr_neumann_c128_pkg_n9_12_1_20260518_2058/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} \
  python3 tests/pytest/run_chunk_gdr_C128_performance_msprof.py \
    --output tests/pytest/prof_out/neumann_vs_c128_n9_12_1_20260518 \
    --ops c128 --no-summary

env TORCH_EXTENSIONS_DIR=/tmp/cgdr_c128_torch_ext_n9_12_1 \
  ASCEND_CUSTOM_OPP_PATH=/tmp/cgdr_neumann_c128_pkg_n9_12_1_20260518_2058/vendors/custom_transformer \
  LD_LIBRARY_PATH=/tmp/cgdr_neumann_c128_pkg_n9_12_1_20260518_2058/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH} \
  python3 ../chunk_gated_delta_rule_neumann/tests/pytest/run_chunk_gdr_performance_msprof.py \
    --output tests/pytest/prof_out/neumann_vs_c128_n9_12_1_20260518 \
    --ops neumann --no-summary
```

## 不能重复走的旧方向

- Stage3 双 flag / mode4 同步试验：已多次 timeout。
- packed64 resident 上三角清零作为 out 主因：已验证无改善。
- 原始 RHS full chunk fallback：诊断更差或不覆盖 full chunk，不作为回退。
- `identity64 @ subblock -> cBridge` 桥接绕法：不符合最终边界。
- direct full128 Fixpipe 曾 timeout 的旧结论已经被后续 `unitFlag=0` 修正覆盖；当前 full128
  materialize 是可用路径，不要再按旧 timeout 结论回退。

## 历史资料入口

- 性能记录：`performance_baselines.md`
- CPU golden 脚本：
  - `tests/pytest/compare_chunk_gdr_C128_cpu_golden.py`
  - `tests/pytest/compare_chunk_gdr_C128_cpu_golden_paramset.py`
- N9.8 full128 性能：
  `tests/pytest/prof_out/c128_n9_8_full128_long_20260518/performance_compare.csv`
- N9.10 rejected RHS DB 性能：
  `tests/pytest/prof_out/c128_n9_10_rhsdb_long_20260518/performance_compare.csv`
- N9.11 C128 vs C64 Neumann 性能：
  `tests/pytest/prof_out/neumann_vs_c128_n9_11_20260518/`
- N9.12.1 C128 vs C64 Neumann 性能：
  `tests/pytest/prof_out/neumann_vs_c128_n9_12_1_20260518/c128_vs_neumann_compare.csv`
- 重要 baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_n9_8_full128_resident_pass_20260518.h`
  - `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_n9_7_full128_trim_pass_20260518.h`
  - `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_tail_phase_n7_8_pass_20260517.h`

## 当前进度

- [x] N9.11 full128 resident/RHS 调度对齐完成并通过 gate。
- [x] N9.11 C128 vs C64 Neumann 性能已测。
- [x] N9.12.1 full chunk `attnWsGm_` BF16 inverse 写回删除并通过 gate。
- [x] N9.12.2 tail lower-left GM roundtrip 改 resident 并通过 gate。
- [x] N9.12.3 tail top/bottom/lower-left pure resident inverse 并通过 gate。
- [x] N9.12.5 C64 aux 初始化延后 gate 收口。
- [x] N9.12.5b dead helper 清理。
- [ ] N9.12.5c tail packed128 化和 dead helper 清理后 gate。
- [x] N9.12 冗余点已静态确认。
- [x] N9.12.1 删除 full chunk `attnWsGm_` BF16 inverse 写回并通过 gate。
- [x] N9.12.1 C128 vs C64 Neumann profiling 已测并落盘。
