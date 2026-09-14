# quant_flash_mla_with_kvcache NPU 性能测试工具

从 ops-transformer-testkit（非量化 FA 算子测试框架）抽取的最小独立性能测试脚本集，
用于测试**全量化 MLA FlashAttention**（`QuantFlashMlaWithKvcache`）在 NPU 上的性能。

## 目录结构

| 文件 | 作用 |
|---|---|
| `run_perf.py` | CLI 入口：逐 case 独立 msprof 进程，汇总 `perf.csv` |
| `perf_cases.py` | 用例集（由 `../pytest/case.py` 的 TEST_PARAMS 自动转换，共 105 个） |
| `perf_codegen.py` | 生成单 case 自包含性能脚本（数据构造 + 热循环调用） |
| `perf_profiler.py` | `msprof --aic-mode=task-based` 采集，拷出 `op_summary.csv` |
| `perf_parser.py` | 按 `OP Type=QuantFlashMlaWithKvcache` 过滤，热启动统计（丢首条取均值） |

## 快速开始

```bash
cd tests/perf

# 1. 环境准备（每次新终端）
conda activate fzj
source /home/fanzijian/Ascend/ascend-toolkit/set_env.sh                    # 基础 CANN 包
source /home/fanzijian/custom/vendors/custom_transformer/bin/set_env.bash  # custom 算子包

# 2. 列出全部用例（105 个）
python3 run_perf.py --list

# 3. 单用例冒烟
python3 run_perf.py --runs 5 --case-filter _000000

# 4. 全量跑（建议放后台）
nohup python3 run_perf.py --runs 5 > full_run.log 2>&1 &

# 5. 按子串过滤（逗号分隔，任一子串命中即选中）
python3 run_perf.py --runs 5 --case-filter Decode,NZ
python3 run_perf.py --runs 5 --case-filter PA_BnNBsD
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--runs` | 5 | 每 case 迭代次数；第 1 次为预热丢弃，需 ≥ 2 |
| `--case-filter` | 无 | 逗号分隔的用例名子串过滤 |
| `--output` | `results/<时间戳>` | 输出目录 |
| `--list` | - | 只列出用例名，不运行 |

## 输出结构

```
results/<时间戳>/          （或 --output 指定目录）
├── perf.csv               # 汇总: case_name, duration_us, samples, status, message
└── <case_name>/
    ├── <case_name>.py     # 生成的性能脚本（可直接单独调试）
    ├── <case_name>.msprof.log
    └── op_summary.csv     # msprof 采集的原始算子条目
```

终端同时打印摘要：

```
[SUMMARY] 105 cases, 0 failed, results -> .../perf.csv
  [HOT] Decode_..._TND_PA_lSE_000000    117.83 us
```

## 用例字段

`perf_cases.py` 中每个用例是一个 tuple：

```
(B, N_q, N_kv, seqused_q, cache_seqlens, enable_pa, kv_cache_layout,
 sparse_mode, enable_lse, num_blocks)
```

| 字段 | 说明 |
|---|---|
| `B / N_q / N_kv` | batch 数 / Q 头数 / KV 头数 |
| `seqused_q` | 逐 batch 实际 Q 序列长 |
| `cache_seqlens` | 逐 batch KV cache 序列长 |
| `enable_pa` | 仅标注；新接口 `block_table` 必填，False 的用例也按 PA 顺序映射构造，kernel 数据量与 noPA 语义等价 |
| `kv_cache_layout` | `BnNBsD` / `BnBsH` / `NZ` |
| `sparse_mode` | 即 mask_mode：0=无掩码，3=causal |
| `enable_lse` | 是否返回 softmax lse |
| `num_blocks` | 物理 cache 块复用：0=不重用（顺序分配）；>0 时按块数随机映射，模拟生产环境块复用 |

新增用例：在 `_CASES` 里按 tuple 格式加一行即可，无需改其它文件。

## 统计口径

- **过滤**：只统计 `OP Type=QuantFlashMlaWithKvcache` 的 AI Core kernel；
  Metadata（AI_CPU）与 ZerosLike 等干扰算子自动排除。
- **热启动**：每次独立进程中第 1 次调用（含 tiling 缓存预热）丢弃，其余取均值。
- **Metadata 开销**：`quant_flash_mla_with_kvcache_metadata` 在热循环外只调一次，
  不计入 kernel 时长。若要测含 metadata 的端到端开销，把生成脚本里的 metadata 调用
  挪进循环即可（脚本在输出目录 `<case_name>.py` 中，可手动改后单跑）。

## 已知说明

- 数据构造对齐 `tests/pytest/common/qmla_with_kvcache_golden.py` 的 NPU 路径：
  Q per-token-head 量化 → TND `(T, N_q, 576)`；K per-tensor 量化 → PA cache；
  V 由算子内部从 K 前 512 维复用。
- `results/` 与 msprof 输出含数据库和大文件，请勿提交到 git（建议加入 `.gitignore`）。
