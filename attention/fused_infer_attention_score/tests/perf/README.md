# FIA 性能网格 Sweep Harness

按支持场景做网格遍历，对每个 case 用 `msprof op` 上板采集，自动按 Bound 类型归类汇总。

## 快速使用

```bash
cd tests/perf

# 1) 生成网格（preset 名见下表）
python3 profile_grid.py --preset smoke -o cases.smoke.json

# 2) 一键 sweep（msprof op → perf_summary → msprof_pipeline_analyze → CSV）
bash run_grid_sweep.sh \
    --cases cases.smoke.json \
    --out-dir results/run_$(date +%Y%m%d_%H%M%S)

# 3) 查看结果
#   - results/run_*/grid_summary.md   — 按 Bound + 按 (scenario, bound) 双重分组的 Markdown 报告
#   - results/run_*/grid_results.csv  — 每行一个 case 的扁平结果
#   - results/run_*/docs/perf/case_*  — 单 case 原始 CSV + summary.txt + bound.md
```

## 预设 (preset)

| preset | scenario | 规模 | 用途 |
|---|---|---|---|
| `smoke` | no_quant | ~16 case | harness 自检 |
| `decode` | no_quant (BNSD/BSND) | ≤400 case | Phase 1 decode |
| `prefill` | no_quant (BNSD/BSND) | ≤200 case | Phase 1 prefill |
| `bsh_decode` / `bsh_prefill` | no_quant (BSH) | ~30 / ~30 case | BSH layout |
| `bnsd_bsnd_prefill` | no_quant (BNSD_BSND) | ~30 case | mixed in/out layout |
| `mla_decode` | mla_decode | ~30 case | D=512 decode + rope |
| `mla_prefill_split` | mla_prefill_split | ~30 case | D=128 + 独立 rope |
| `mla_prefill_combine` | mla_prefill_combine | ~30 case | D=192/V=128, rope 已合并 |
| `antiquant_perchannel` | antiquant mode=0 | ~50 case | KV 反量化 per-channel |
| `antiquant_pertoken` | antiquant mode=1 | ~30 case | KV 反量化 per-token |
| `antiquant_pertoken_perhead` | antiquant mode=3 | ~20 case | per-token + per-head |
| `full_quant_per_tensor` | full_quant | ~50 case | INT8 Q/K/V，输出量化 |
| `contiguous_decode` / `contiguous_prefill` | contiguous | ~30 / ~20 case | block_table=None |
| `mask_no` / `mask_full` / `mask_leftup` / `mask_rightdown` / `mask_band` / `mask_alibi_pse` | mask | ~20 case 各 | sparse_mode 0-4 + ALiBi pseShift |
| `phase2_full` | 上述全部 | ≤1500 case | Phase 2 总集 |
| `all` | phase1 ∪ phase2 | ≤2000 case | 全量 |

`profile_grid.py` 会先用 `tests/pytest/check_valid_param.validate_config`（仅对 BSH/BSND/BNSD layout 启用）做一次基础剪枝，再调用 `scenarios/<scenario>.validate(case)` 做场景级剪枝。

## 已覆盖矩阵（Phase 1 + Phase 2）

| 维度 | 取值 |
|---|---|
| layout | BNSD, BSND, **BSH**, **BNSD_BSND** |
| dtype | float16, bfloat16, **int8**（仅 full_quant） |
| 量化 | no-quant, **antiquant (per-channel/per-token/per-token-per-head)**, **full_quant (per-tensor)** |
| KV cache | paged（block_table）, **contiguous（block_table=None）** |
| MLA | **decode (D=512+rope)**, **prefill split (D=128+rope)**, **prefill combine (D=192/V=128)** |
| mask | sparse_mode 0/1/2/3/4 (no/full/leftUp/rightDown/band) + **ALiBi (pseShift)** |

## 未覆盖（明确跳过）

| 场景 | 原因 |
|---|---|
| **tensor_list KV** | `torch_npu.npu_fused_infer_attention_score` Python API 的 key/value 形参为 `Tensor` 而非 `TensorList`；要覆盖需走 aclnn C++ 直调，工程量超出本期范围 |
| **tree mask** | V5 文档 sparseMode 5-8 标注为 unsupported，op 拒收 |
| **prefix attention** | `key_shared_prefix`/`value_shared_prefix` — 不在用户列出的范围内 |
| **antiquant 模式 4-7** | PA、group 等高级模式；本期仅覆盖最常用的 0/1/3 |
| **D ∉ {128, 192, 512}** | 网格保持主流 head_dim |

## 输出目录结构

```
results/run_<TS>/
├── cases.json                  # 本轮使用的网格定义
├── grid_results.csv            # 每行一个 case 的扁平结果
├── grid_summary.md             # Bound + (scenario, bound) 分组报告
├── sweep.log                   # 编排日志
├── raw/case_<id>/OPPROF_*/     # msprof 原始输出
└── docs/perf/case_<id>/
    ├── *.csv                   # 8 个 msprof CSV
    ├── summary.txt             # perf_summary.py 输出
    └── bound.md                # msprof_pipeline_analyze.py 输出
```

## CSV 字段

`case_id, tag, scenario, status, in_layout, dtype, out_dtype, batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, block_size, cache_layout, antiquant_mode, mask_subtype, sparse_mode, has_pse, has_block_table, task_duration_us, block_dim, core_prefix, bound_type, scalar_subtype, scalar_ratio_pct, scalar_pct, scalarldst_pct, mte2_pct, mte3_pct, cube_pct, vec_ratio_pct, error_msg`

`bound_type` 取值：`MTE2 BOUND / CUBE BOUND / VEC BOUND / FIXP BOUND / MTE3 BOUND / SCALAR BOUND / 无 bound / ERROR`

字段口径说明：
- `mte2_pct / mte3_pct / cube_pct / vec_ratio_pct` 分母为 `<prefix>_time(us)`（最忙核计算时间），与 `bound.md` 表内一致；并行流水间存在重叠，相加可能 > 100%（属正常现象）。
- `scalar_ratio_pct` 为 msprof 原生 `<prefix>_scalar_ratio`（分母 = Task Duration），是 SCALAR BOUND 判定输入（≥ 70% 触发 SCALAR BOUND）。
- `scalar_pct / scalarldst_pct` 仍按 `metric-mapping.md §3` 代理公式（分母 = `<prefix>_time(us)`），仅供 SCALAR 子类细分使用。

## 复用的外部脚本

- `ops/skills/ops-profiling/scripts/perf_summary.py` — `OPPROF → round_NNN/`
- `ops/skills/ascendc-performance-analysis/scripts/msprof_pipeline_analyze.py` — bound 分类

可通过 `--perf-summary` / `--pipeline-analyze` 覆盖路径。

## 文件结构

```
tests/perf/
├── README.md
├── profile_grid.py             # cases.json 生成器
├── case_driver.py              # 单 case msprof 入口（dispatcher）
├── collect_row.py              # 每 case CSV 行抽取
├── run_grid_sweep.sh           # 编排器
├── aggregate_results.py        # CSV → grid_summary.md
└── scenarios/
    ├── __init__.py             # REGISTRY 字典
    ├── _common.py              # 共享 helper（block_table、layout-aware tensor 构造、base_validate）
    ├── no_quant.py             # BNSD/BSND/BSH/BNSD_BSND no-quant paged
    ├── mla.py                  # MLA decode + 2 prefill 子模式
    ├── antiquant.py            # KV 反量化 mode 0/1/3
    ├── full_quant.py           # per-tensor full quant，INT8 Q/K/V
    ├── contiguous.py           # block_table=None
    └── mask.py                 # sparse_mode 0-4 + ALiBi pseShift
```

## 常见问题

- **NPU 设备**：默认 device 0，可通过 `--device 1` 或环境变量 `FIA_PROFILE_DEVICE=1` 覆盖。
- **launch-count 与 round 命名**：始终通过 `--round-name=case_<id>` 强制命名以禁用 `perf_summary.py` 的自动递增。
- **单 case 失败**：sweep 不中断；CSV 中该行 `status=ERROR`，详情见 `sweep.log`。常见失败：MLA decode 在不支持 INT8 kernel 的环境上会以 RuntimeError 失败，是预期行为。
