# sparse_attention_score_prefill A2 调试工作流

目标：每次修改都完成“修改 -> 编译 -> 安装 -> 功能验证 -> 精度验证 -> 失败分类 -> 继续修改”闭环。

## 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rrh
source /opt/rrh/zhengbao/0708/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/opt/rrh/zhengbao/0708/cann/opp
source /opt/rrh/zhengbao/0708/cann/opp/vendors/custom_transformer/bin/set_env.bash
export NPU_DEVICE=npu:2
cd /opt/rrh/code3/ops-transformer/attention/sparse_attention_score_prefill
```

## Auditable A2 debug workflow

Use the wrapper below for every debug run. It creates a unique directory and
keeps the kernel log, extracted `DumpTensor` values, golden tensors, NPU output,
comparison tensors, comparison JSON, and run metadata together.

```bash
python tools/run_a2_debug_workflow.py \
  --dump-root /tmp/sasa_a2_runs \
  --q-len 4 --kv-len 512 --q-heads 8 --kv-heads 2 \
  --block-size 128 --top-k 3 --seed 42 \
  --compare-stage final
```

The command streams runner/kernel output to the terminal and `kernel.log`, and
prints `run_dir`, `dump_dir`, `kernel_log`, `compare_<stage>.log`,
`compare_<stage>.json`, `workflow_result.json`, the comparison summary, and the
final `verdict`. The `final` stage compares `npu_debug_output.pt` with
`golden_final.pt`. P dump comparison is disabled by default because it has
already been checked; use `--compare-p` only when explicitly needed.

## 闭环步骤

1. 修改代码，记录根因假设、预期结果和回归用例。
2. 本地执行 `git -C D:\code3\ops-transformer diff --check`。
3. 只同步本轮修改的文件：

```powershell
scp <local-file> 121.37.53.41:/opt/rrh/code3/ops-transformer/<same-relative-path>
```

4. 在 A2 编译：

```bash
cd /opt/rrh/code3/ops-transformer
bash build.sh --pkg --soc=ascend910b --ops=sparse_attention_score_prefill >/tmp/a2_build.log 2>&1
rc=$?
tail -80 /tmp/a2_build.log
exit $rc
```

5. 安装并刷新环境：

```bash
cd /opt/rrh/code3/ops-transformer
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --force --install-path=/opt/rrh/zhengbao/0708/cann/opp
source /opt/rrh/zhengbao/0708/cann/opp/vendors/custom_transformer/bin/set_env.bash
```

6. 功能验证：先运行 `test_bf16_minimal_debug`，再运行 GQA、尾块、长序列和生产用例。ACL/AICore、UB/L0C 越界、未对齐、stream synchronize、shape/dtype 错误及 NaN/Inf 均属于功能问题，出现后保存设备日志和 `extra-info/data-dump`，回到步骤 1。

7. 精度验证：

```bash
cd /opt/rrh/code3/ops-transformer/attention/sparse_attention_score_prefill
python -m pytest -q tests/test_sparse_attention_score_prefill_npu.py -s
python tests/test_sparse_attention_score_prefill_vs_decode_perf.py
```

8. 失败分类：全 NaN 查 rowMax/rowSum、Phase2 读取和除零；部分 NaN 查 task/core 映射和输出覆盖；数值偏差查 GM staging 行跨度、ND/NZ 布局和 cast；特定长度查 `RoundUp`、尾块 mask 和 `DataCopyPad`；多核失败查 CrossCore/Event 同步。

9. 只有全部用例通过、无 NaN/Inf、无设备异常且误差满足各测试 `prec` 才算完成。每轮记录修改摘要、编译/安装结果、功能结果、最大/平均误差和下一步假设。

## BxS 泛化验证

当前出版范围固定 `D=128`、`block_size=128`，只泛化 `B x S`，上限为
`131072 (128K)`。prefill 与 decode 必须一起安装，避免参考算子缺失：

```bash
cd /opt/rrh/code3/ops-transformer
bash build.sh --pkg --soc=ascend910b \
  --ops=sparse_attention_score_prefill\;sparse_attention_score
bash build_out/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --force --install-path=/opt/rrh/zhengbao/0708/cann/opp
```

运行单档用例：

```bash
export CANN_OPS_TRANSFORMER_LAZY_IMPORT=1
export NPU_DEVICE=npu:2
export BATCH_SIZE=8
export Q_SEQLEN=16384
export KV_SEQLENS=32768
timeout -k 5s 180s python -u -m pytest -q -s \
  tests/test_sparse_attention_score_prefill_vs_decode_perf.py::\
TestSparseAttentionPrefillVsDecode::test_prefill_vs_decode_sweep
```

2026-08-12 A2 验证结果（`q_heads=64`、`kv_heads=4`、`topK=16`、
`innerPrecise=1`）：

| B | S | BxS | KV/Batch | max diff | mean diff | result |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1024 | 1K | 2K | 0.000977 | 0.000060 | PASS |
| 2 | 4096 | 8K | 8K | 0.000977 | 0.000056 | PASS |
| 4 | 8192 | 32K | 16K | 0.000977 | 0.000056 | PASS |
| 8 | 8192 | 64K | 16K | 0.000977 | 0.000056 | PASS |
| 8 | 16384 | 128K | 32K | 0.000977 | 0.000056 | PASS |
| 1 | 131072 | 128K | 128K | 0.003906 | 0.000056 | PASS |
| 128 | 1024 | 128K | 2K | 0.001221 | 0.000059 | PASS |

每个用例都要求 `nonfinite=0`，且单算子执行超过 180 秒即按卡死处理。
