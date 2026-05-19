# C128 overlay 回收为 one-shot compose 计划

## 当前 baseline

- Active file: `op_kernel/chunk_gated_delta_rule_C128_stage1.h`
- Saved overlay-pass baseline:
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_fp16_direct16_direct32_overlay_pass_20260516.h`
- SHA256:
  `5e0faffeaa55c7094ff6d2557ac96ddadb61cf23e07a12e675c7a4c5285c1ae1`
- Saved one-shot no-overlay baseline:
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_fp16_oneshot_no_overlay_pass_20260516.h`
- SHA256:
  `33f18ffe7b96a76f91babd96f9b7dfb4dffed1061958d51653f8f438fca0f43a`
- Saved cleaned one-shot no-overlay baseline:
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_fp16_oneshot_no_overlay_clean_pass_20260516.h`
- SHA256:
  `2146252a91649016f50d75755c2fb5fb39e0e6121cf819fe9f565b2e02589580`
- Gate evidence:
  - build: `bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_C128 -j16`
  - CPU golden vs C128 NPU: `2 passed`
  - `S=128`: `out max_abs=2.457350e-04`; `state max_abs=9.371340e-05`
  - `S=16384`: `out max_abs=2.181083e-04`; `state max_abs=8.013844e-05`
- No-overlay package SHA256:
  `7785886073e060abc0df76070001cfc0643c99e984cab5e0183c25d61930c6bf`
- No-overlay install dir:
  `/tmp/cgdr_c128_pkg_oneshot_no_overlay_20260516_103004`
- Clean no-overlay package SHA256:
  `68b6c915ac007da21d75b6a1ce0aa91f8d2abdd9448166ff5601ab528b5d1f83`
- Clean no-overlay install dir:
  `/tmp/cgdr_c128_pkg_clean_20260516_105211`

说明：Neumann arch32 分支的 overlay 不是正确性参考；它本身是因为精度问题借鉴了 C128
overlay 诊断方式。本文中 direct16/direct32 overlay 只作为历史诊断 oracle，用于证明
full64 one-shot compose 的布局等价；clean 版 active 源码已经删除 overlay/dump/早退
诊断实现。

## 当前实现

当前 active C64 inverse 路径用 full64/packed one-shot 风格生成 resident inverse：

1. `LoadPacked64DiagonalInputsToL1()`
2. `InvertPacked64DiagonalNeumannToResult()`
3. `ComposeOddEvenLower16To32AccumFull64()`
4. `ComposeOddEvenLower32To64AccumFull64()`

clean 版 active 源码不再包含 direct16/direct32 overlay helper、O1/O2 dump helper、
`C128_DEBUG_RETURN_POINT` 早退或 `CopyRaw` 诊断路径。

## 为什么理论上应该等价

按 lower-triangular block inverse，块组合是分层的：

```text
32x32:
[ inv0        0   ]
[ lower16Inv  inv1]

64x64:
[ even32Inv   0   ]
[ lower32Inv  odd32Inv]
```

如果 `INVInit` 中的 lower-left 已经按当前代码约定包含符号，那么两层 compose 都是同一类乘法：

```text
tmp16       = lower16 @ topInv16
lower16Inv  = bottomInv16 @ tmp16

tmp32       = lower32 @ even32Inv
lower32Inv  = odd32Inv @ tmp32
```

full64 one-shot 和 overlay 的数学表达式应当相同。当前差异更可能来自布局等价没有成立，而不是公式本身不同。

## 当前怀疑点

1. 16x16 子块嵌入 full64 L0A/L0B 时，源 block 坐标或 `ifTranspose` 方向与 direct16 不一致。
2. `FixpipePacked64FullToHalfL1()` 写回 full64 时，目标 packed offset/stride 覆盖了正确子块之外的数据，或者目标 lower16 子块不是 direct16 期望的位置。
3. `ComposeOddEvenLower16To32AccumFull64()` 使用 full64 累加结果写回 packed inverse L1
   buffer（当前命名为 `aicHalfL1PackedInv64_`），但下一层读取 32x32 对角 inverse
   时实际消费的是不同布局。
4. direct32 已能改善但不能通过，说明 32->64 外层有一部分问题；direct16 后才通过，说明至少 16->32 内层仍有未闭合的布局差异。

## 推进方案

### Phase O0: 冻结 overlay pass baseline

- 已完成。不要在未通过 gate 的情况下覆盖该 baseline。
- 后续所有 one-shot 回收实验都必须能回到这个文件。

### Phase O1: 建立 full64 vs direct16 的最小对照

目标：只观察 16->32 层，不改 32->64。

步骤：

1. 在 `ComposeOddEvenLower16To32AccumFull64()` 之后、direct16 overlay 之前，增加临时 debug 早退或 dump 路径。
2. 输出 resident / packed inverse L1 buffer（当前命名为 `aicHalfL1PackedInv64_`）
   中两个 32x32 对角块的 `(16,0)` 子块。
3. 再运行 direct16 overlay，输出同一两个 `(16,0)` 子块。
4. 用同一输入比较 full64 产物和 direct16 oracle 的 block-level diff。

通过条件：

- 两个 lower16 子块都与 direct16 oracle 达到当前 `0.005` gate 下的误差水平。
- 如果不一致，先记录是 block0 `(16,0)`、block1 `(48,32)`，还是二者都错。

O1 观测结果（2026-05-16）：

- 诊断开关：`C128_O1_DUMP_LOWER16_BLOCKS=true`，构建后隔离安装到
  `/tmp/cgdr_c128_pkg_o1_dump_20260516_094423`。
- `.run` 实际 SHA256：
  `2cda29b7e1951d5fb141a214224061a6701f7353c6214e78860e5bd2435af940`。
- 输入：pytest CPU golden 默认首个 case，
  `B=1,S=128,nk=4,nv=4,dk=dv=128,seed=21`。
- 读取 `out.flatten()`：
  `full64 = [0:512].reshape(2,16,16)`，
  `direct16 = [512:1024].reshape(2,16,16)`。
- 结果：
  - block0 `(16,0)`：`max_abs=0`，两边本次均为全 0。
  - block1 `(48,32)`：`max_abs=1.27685547e-01`，
    `mean_abs=1.30023714e-03`，`rmse=9.99356434e-03`，
    `nonzero=245/256`；worst `(1,15)`，
    `full64=0`，`direct16=-1.27685547e-01`。
- 结论：O1 未通过。当前最小差异集中在第二个 32x32 对角块的
  `(48,32)` lower16；下一步进入 O2 时先只修 16->32 full64 compose 的
  block1 布局/写回，不动 RHS、Stage2、Stage3 或 32->64。

O1 复核结果（2026-05-16）：

- 修正 dump 使用独立 `aicL0C_[C64_L0C_TILE_ELEMS]` 后，两个 full64 lower16
  dump 均为 0，而 direct16 oracle 两个块均非 0：
  - block0 `(16,0)`: `direct16 max_abs=4.21630859e-01`。
  - block1 `(48,32)`: `direct16 max_abs=1.27685547e-01`。
- 结论：问题在 16->32 第一层 full64 L0 目的布局，不是 direct16 oracle dump 污染。

### Phase O2: 修 full64 的 16->32 布局

只允许改 16->32 full64 compose 的布局/写回，不动 RHS、Stage2、Stage3、tail lower-left。

候选修法：

1. 对齐 direct16 的 L0A/L0B 加载顺序，把 full64 helper 中的 source row/col 显式映射到同一 packed block 坐标。
2. 如果 full64 Fixpipe 写回 stride 不适合只保留 lower16，改成 full64 计算但按 direct16 的目标 offset 写回两个 lower16 子块。
3. 若 full64 写回必须保留整块，则在写回前确保 L0C 里非目标区域不会污染后续读取。

每次只改一个点。每个候选都要 build/install 后跑：

```bash
python3 tests/pytest/compare_chunk_gdr_C128_cpu_golden.py \
  --seed 21 --batch-size 1 --seqlen 128 --nk 4 --nv 4 --dk 128 --dv 128 \
  --chunk-size 128 --atol 0.005 --rtol 0.005
```

通过后再跑 pytest 参数集。

O2 修复结果（2026-05-16）：

- 根因：16->32 full64 的两个 lower16 子块装入 L0 时，把 full64 L0 目的地址当成
  compact-NZ/col-block-major offset 使用了；对角块 offset 恰好重合，lower 子块不重合，
  因而 MMAD 看到的 lower 区域为 0。
- 修复：将 lower16 目的 offset 改为 full64 L0 row-block-major 的
  `(rowBlock=1,colBlock=0)` offset：
  `C64_L0_ROW1_COL0_OFFSET = C64_PACKED_BLOCK_COUNT * RAW_HALF_FRACTAL_ELEMS`。
- O2 诊断包：
  `/tmp/cgdr_c128_pkg_o2_l0offset_20260516_102050`，
  package SHA256 `17570011444d0e926b6671f928d83d446b68b2df5f3d243939380c968607a7db`。
- 结果：`full16` 两个 block 与 direct16 oracle 完全一致：
  - block0 `full16-direct16 max_abs=0`。
  - block1 `full16-direct16 max_abs=0`。

### Phase O3: 临时关闭 direct16 overlay

当 O1/O2 证明 full64 16->32 与 direct16 oracle 等价后，临时移除：

```cpp
OverlayOddEvenLower16To32Direct16ToResident(...)
```

保留 direct32 overlay，验证 full chunk 和 tail：

- `S=64`
- `S=96`
- `S=128`
- `S=256`
- pytest 参数集 `S=128/S=16384`

如果通过，说明 16->32 one-shot 已可接管。

O3 结果（2026-05-16）：

- `C128_USE_DIRECT16_OVERLAY=false`，保留 `C128_USE_DIRECT32_OVERLAY=true`。
- `S=128` CPU golden 直比 PASS：
  `out max_abs=2.457350e-04`，`state max_abs=9.371340e-05`。
- 结论：16->32 one-shot 已可接管 direct16 overlay。

### Phase O4: 建立 full64 vs direct32 的最小对照

目标：只观察 32->64 层。

步骤与 O1 相同，但对比 resident 中 `(32,0)` lower32：

1. full64 `ComposeOddEvenLower32To64AccumFull64()` 写回后的 `(32,0)`。
2. direct32 overlay 后的 `(32,0)`。

如果 O3 已通过，O4 只剩外层 32->64 布局差异。

O4 结论（2026-05-16）：

- 未再单独增加 direct32 dump。O2 已确认同类问题来自 full64 L0 目的布局；
  32->64 的 `LoadPacked32BlockToFull64L0()` 使用相同错误模式：
  `CompactMatrixNzOffset(dstRow,dstCol,64)` 加 `dstGap=3`，而 full64 L0 MMAD
  需要 row-block-major 目的布局。
- 因此直接进入 O5，修正 32->64 full64 L0 目的 offset，并用完整 no-overlay gate 验证。

### Phase O5: 修 full64 的 32->64 布局并关闭 direct32 overlay

只允许改 32->64 full64 compose 的布局/写回。

关闭：

```cpp
OverlayOddEvenLower32To64Direct32ToResident(...)
```

完整 gate：

- single shape: `S=64/80/96/112/128/160/256`
- pytest: `S=128/S=16384`
- 如果要与 Neumann 对齐，再追加 `S=1024` 类长序列 smoke。

O5 结果（2026-05-16）：

- 修复：新增 `L0MatrixBlockOffset(row,col,blockSize)`，将
  `LoadPacked32BlockToFull64L0()` 的目的 offset 改为 full64 L0 row-block-major，
  并将 `dstGap` 改为 0。
- 关闭：
  - `C128_USE_DIRECT16_OVERLAY=false`
  - `C128_USE_DIRECT32_OVERLAY=false`
- no-overlay 单 shape gate 全通过：
  - `S=64`: `out max_abs=2.530664e-04`; `state max_abs=1.288950e-04`
  - `S=80`: `out max_abs=3.597140e-04`; `state max_abs=1.390129e-04`
  - `S=96`: `out max_abs=2.775341e-04`; `state max_abs=1.088232e-04`
  - `S=112`: `out max_abs=2.815574e-04`; `state max_abs=1.123250e-04`
  - `S=128`: `out max_abs=2.457350e-04`; `state max_abs=9.371340e-05`
  - `S=160`: `out max_abs=2.637506e-04`; `state max_abs=1.048371e-04`
  - `S=256`: `out max_abs=3.460795e-04`; `state max_abs=9.822100e-05`
- pytest gate 全通过：
  - `S=128`: `out max_abs=2.457350e-04`; `state max_abs=9.371340e-05`
  - `S=16384`: `out max_abs=2.181083e-04`; `state max_abs=8.013844e-05`
  - 结果：`2 passed`

### Phase O6: 清理 overlay

只有 O3 和 O5 都通过后，才清理 overlay/debug helper：

- active one-shot path不调用 overlay。
- clean active 源码不再保留 overlay helper；oracle 证据保存在历史 baseline 和本文档中。
- `todo.md` 记录新 baseline SHA、package SHA、gate 结果。

O6 状态（2026-05-16）：

- 已完成 active path 回收并清理：
  - 删除 `C128_USE_DIRECT16_OVERLAY` / `C128_USE_DIRECT32_OVERLAY`。
  - 删除 direct16/direct32 overlay helper。
  - 删除 O1/O2 dump helper。
  - 删除 `C128_DEBUG_RETURN_POINT` / `CopyRaw` 早退诊断路径。
- clean package 通过 CPU golden pytest gate，结果 `2 passed`：
  - `S=128`: `out max_abs=2.457350e-04`; `state max_abs=9.371340e-05`
  - `S=16384`: `out max_abs=2.181083e-04`; `state max_abs=8.013844e-05`

## 关键约束

- 不允许同时改 16->32 和 32->64。
- 不允许在精度未过时继续做性能优化。
- 不允许把 partial smoke 当 baseline；至少要过 `S=64/96/128/256` 和 pytest 参数集。
- 当前 overlay-pass baseline 是回退点，不要覆盖。
