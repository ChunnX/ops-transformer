# 2026-05-04 C64_3232 标量前代基线与 L1 承接计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

用户追问 `INV += INV @ Lpow` 累加完成后是否必须写回 GM，是否可以搬到 L1/TSCM。

结论是：不必须写 GM。GM 只是当前 `MatmulImpl` 串接和精度二分最稳的验证路径。最终目标应是 `L0C accumulator -> Fixpipe -> L1/TSCM`，再经 `LoadData` 作为后续 MMAD 输入，避免 `GM workspace` 往返。

## 本轮基线

本轮先在稳定 `C64_3232_FP32_CVInverse` 线上新增 `DEBUG_SCALAR_FORWARD_GM_INVERSE`，使用 AIC 标量前代递推直接从 `L_powWsGm_` 构造 `INVWsGm_`：

```cpp
INV[row, col] = 1, row == col
INV[row, col] = -L[row, col] - sum(L[row, k] * INV[k, col]), col < row
INV[row, col] = 0, otherwise
```

该版本的目的不是性能优化，而是建立一个公式正确、易对齐的基线。后续 local/TSCM 或 raw MMAD 累加版本只要与该基线对齐，就可以缩小问题范围到搬运/layout/累加 API。

## 验证结果

- selector 切到 `C64_3232_FP32_CVInverse`。
- 构建安装双算子包通过。
- Neumann-only smoke 三个 case 通过：
  - `S=16, nk=1, nv=1, dk=dv=16`
  - `S=17, nk=1, nv=1, dk=dv=16`
  - `S=64, nk=4, nv=4, dk=dv=128`
- 全量三方精度通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - `20 passed in 6.16s`

## 性能结果

使用 `msprof --application ... --task-time=on` 跑最小性能对比，结果位于：

- `tests/pytest/prof_out/chunk_gdr_c64_3232_scalar_forward_gm_smoke/performance_compare.csv`
- `tests/pytest/prof_out/chunk_gdr_c64_3232_scalar_forward_gm_smoke/performance_raw.csv`

结果：

| case | normal us | scalar neumann us | neumann/normal | 主耗时 |
|---|---:|---:|---:|---|
| `B=8,S=16,D=64,Hq=16,Hkv=4` | 633.08 | 1628.66 | 2.573x | AIC scalar |
| `B=8,S=64,D=64,Hq=16,Hkv=4` | 637.12 | 15199.28 | 23.856x | AIC scalar |

结论：标量版证明了公式正确，但性能不可作为最终方向。`S=64` 的 AIC scalar time 从 normal `113.649 us` 放大到 `11284.828 us`，必须回到底层 MMAD/L0C 累加。

## L1/TSCM 承接方向

累加后搬到 L1/TSCM 是可行方向，不需要强制到 GM。难点在三点：

1. `L0C` 不能直接作为下一轮 A/B 输入，必须经过 `Fixpipe -> L1/TSCM/VECCALC`，再 `LoadData -> L0A/L0B`。
2. Fixpipe 到 L1/TSCM 后的 layout 必须与下一次 `LoadData/Mmad` 的期望一致，尤其是 compact NZ tile 的行列偏移。
3. `INV += INV @ P` 最终不应由标量循环完成，应使用底层 `MmadParams(..., cmatrixSource, cmatrixInitVal)`：
   - `cmatrixInitVal=true`：初始化 L0C。
   - `cmatrixInitVal=false`：在已有 L0C 上继续累加。
   - `cmatrixSource=true`：从额外 C/bias tensor 作为 C 初值来源。

## 下一步

1. 保留本轮标量 GM 前代版本作为公式基线。
2. 在 local/TSCM PoC 中先验证 compact 16x16/32x32 tile 搬运与子块读取。
3. 用标量 local 前代先做不经 GM 的正确性路径。
4. 精度通过后，把标量 `AddLocalMatrix()` 替换为 `cmatrixInitVal=false` 或 `cmatrixSource=true` 的 MMAD C 累加。
5. 全量 compare 通过后再跑 msprof，不再对精度失败版本做性能测量。
